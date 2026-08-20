"""Validate the MPAS->wrfout nearest-cell regridding against a planted analytic field.

The instrument is tested in both directions: a field the regridder should
reproduce almost exactly (smooth, well resolved by the mesh) and a field it
must NOT smooth (a single-cell spike). Counts, not fits.
"""
import json
import shutil
import subprocess
import sys

import netCDF4
import numpy as np

SP = sys.argv[1]
CONV = sys.argv[2]
SRC = f"{SP}/tierb-hist/cuda-history.2026-08-12_06.00.00.nc"
MESH = f"{SP}/tierb-hist/x1.40962.init.gfs-20260812-06z.nc"
WORK = f"{SP}/validate"
R = np.pi / 180.0
EARTH_KM = 6371.0

results = {}


def plant(path, values):
    shutil.copyfile(SRC, path)
    d = netCDF4.Dataset(path, "a")
    d.variables["t2"][:] = values.astype(np.float32).reshape(d.variables["t2"].shape)
    d.close()


def convert(hist, outdir, tag):
    subprocess.run(
        [CONV, "--history", hist, "--mesh", MESH, "--out-dir", outdir,
         "--window", "focus", "--field-set", "full", "--clobber",
         "--json", f"{outdir}/report-{tag}.json"],
        check=True, capture_output=True,
    )
    return f"{outdir}/wrfout_d01_2026-08-12_06_00_00"


src = netCDF4.Dataset(SRC)
lat = np.asarray(src.variables["latCell"][:], dtype=np.float64)   # radians
lon = np.asarray(src.variables["lonCell"][:], dtype=np.float64)
ncell = lat.size

# ---------------------------------------------------------------------------
# Direction 1: a smooth analytic field the 120 km mesh resolves well.
# f = sin(3*lon) * cos(2*lat), a degree-3/2 pattern with wavelength ~13000 km.
# Nearest-cell error must be bounded by |grad f| * nearest-cell distance.
# ---------------------------------------------------------------------------
def smooth(la, lo):
    return np.sin(3.0 * lo) * np.cos(2.0 * la)


import os
os.makedirs(f"{WORK}/smooth", exist_ok=True)
plant(f"{WORK}/smooth-history.2026-08-12_06.00.00.nc", smooth(lat, lon))
out = convert(f"{WORK}/smooth-history.2026-08-12_06.00.00.nc", f"{WORK}/smooth", "smooth")

o = netCDF4.Dataset(out)
got = np.asarray(o.variables["T2"][:], dtype=np.float64).squeeze()
tlat = np.asarray(o.variables["XLAT"][:], dtype=np.float64).squeeze() * R
tlon = np.asarray(o.variables["XLONG"][:], dtype=np.float64).squeeze() * R
truth = smooth(tlat, tlon)
err = np.abs(got - truth)

# The predicted bound: |grad f| on the sphere times the nearest-cell distance.
# |df/dlat| <= 2, |df/dlon|/cos(lat) <= 3/cos(lat); take the sup over the window.
grad_max = np.sqrt(4.0 + (3.0 / np.cos(tlat)) ** 2).max()  # per radian
report = json.load(open(f"{WORK}/smooth/report-smooth.json"))
results["smooth"] = {
    "analytic_field": "sin(3*lonCell) * cos(2*latCell), dimensionless, range [-1, 1]",
    "target_points": int(got.size),
    "max_abs_error": float(err.max()),
    "mean_abs_error": float(err.mean()),
    "rms_error": float(np.sqrt((err ** 2).mean())),
    "field_range_in_window": [float(truth.min()), float(truth.max())],
    "max_error_as_fraction_of_range": float(err.max() / (truth.max() - truth.min())),
    "predicted_bound": float(grad_max * (69.843 / EARTH_KM)),
    "predicted_bound_note": (
        "|grad f| (per radian, sup over window) x max nearest-cell arc (69.843 km / 6371 km). "
        "Measured max error must sit under this."
    ),
    "points_over_predicted_bound": int((err > grad_max * (69.843 / EARTH_KM)).sum()),
}

# ---------------------------------------------------------------------------
# Direction 2: nearest-cell must NOT smooth. Every emitted value has to be a
# value the model actually carried in some cell -- bit-exact membership.
# ---------------------------------------------------------------------------
src_vals = np.float32(smooth(lat, lon))
member = np.isin(np.float32(got).view(np.int32), src_vals.view(np.int32))
results["no_new_values"] = {
    "claim": "every regridded value is bit-identical to some source cell value",
    "target_points": int(got.size),
    "points_matching_a_source_cell_bitwise": int(member.sum()),
    "points_not_matching": int((~member).sum()),
}

# ---------------------------------------------------------------------------
# Direction 3: a planted single-cell spike. A smoothing interpolant would pull
# its amplitude down; nearest-cell must carry it at full height, and must not
# spread it beyond the cells that own those target points.
# ---------------------------------------------------------------------------
# Pick a cell near the middle of the CONUS window.
d2 = (lat - 39.0 * R) ** 2 + (lon - (-97.0 * R)) ** 2
spike_cell = int(np.argmin(d2))
SPIKE = 70.0
vals = np.zeros(ncell)
vals[spike_cell] = SPIKE
os.makedirs(f"{WORK}/spike", exist_ok=True)
plant(f"{WORK}/spike-history.2026-08-12_06.00.00.nc", vals)
out2 = convert(f"{WORK}/spike-history.2026-08-12_06.00.00.nc", f"{WORK}/spike", "spike")
o2 = netCDF4.Dataset(out2)
got2 = np.asarray(o2.variables["T2"][:], dtype=np.float64).squeeze()
hit = got2 == SPIKE
results["spike"] = {
    "claim": "a single-cell 70.0 spike survives at full amplitude and stays local",
    "planted_cell": spike_cell,
    "planted_cell_lat_deg": float(lat[spike_cell] / R),
    "planted_cell_lon_deg": float(lon[spike_cell] / R),
    "planted_amplitude": SPIKE,
    "max_in_regridded_field": float(got2.max()),
    "amplitude_loss": float(SPIKE - got2.max()),
    "target_points_at_full_amplitude": int(hit.sum()),
    "target_points_strictly_between_0_and_amplitude": int(
        ((got2 > 0.0) & (got2 < SPIKE)).sum()
    ),
    "note": (
        "A smoothing interpolant would put a nonzero count in the "
        "'strictly between' row and lose amplitude. Nearest-cell must show 0 there."
    ),
}

# ---------------------------------------------------------------------------
# Resolution limit, stated in the units that matter.
# ---------------------------------------------------------------------------
results["resolution_limit"] = {
    "mesh": "x1.40962 quasi-uniform global, 40962 cells",
    "mesh_mean_cell_spacing_km": float(np.sqrt(4 * np.pi * EARTH_KM ** 2 / ncell)),
    "target_grid": "Lambert 240x150 at dx = 22 km (the 'focus' window)",
    "mean_nearest_cell_distance_km": 41.875,
    "max_nearest_cell_distance_km": 69.843,
    "statement": (
        "The target grid is finer than the mesh that feeds it. Regridding creates "
        "no information: each MPAS cell paints a contiguous patch of target pixels "
        "roughly 120 km across, so a rendered field is blocky at cell scale and "
        "nothing smaller than about two mesh cells (~240 km) is represented at all."
    ),
}

print(json.dumps(results, indent=2))
json.dump(results, open(f"{SP}/regrid-validation.json", "w"), indent=2)

"""Derives the groundwater-bucket prior rows of params.v1.json from NOAA's NWM v3.1.6
GWBUCKPARM_CONUS_FullRouting.nc (US government work, NCO nwm.v3.1.6/parm/domain).

Table generation only (run once, output committed as data/gw_prior.nwm-v3.1.6.json):
  python3 -I gen_gw_prior.py GWBUCKPARM_CONUS_FullRouting.nc OUT.json
"""
import hashlib, json, sys
import netCDF4
import numpy as np

src = sys.argv[1]
d = netCDF4.Dataset(src)
coeff = np.asarray(d["Coeff"][:], float)
expon = np.asarray(d["Expon"][:], float)
zmax = np.asarray(d["Zmax"][:], float)
sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
url = "https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/nwm.v3.1.6/parm/domain/GWBUCKPARM_CONUS_FullRouting.nc"
base = f"NOAA NWM v3.1.6 GWBUCKPARM_CONUS_FullRouting.nc ({url}, sha256 {sha}), {len(coeff)} CONUS buckets"
out = {
    "gw_coeff_m3_s": {"value": float(np.median(coeff)), "unit": "m3 s-1",
                      "source": base + f": Coeff is {coeff.min():g}..{coeff.max():g} (one value) for every bucket. CFE Cgw [m h-1] = Coeff * 3600 / catchment area [m2] (the NWM bucket outflow Q = Coeff (exp(Expon z/Zmax) - 1) is a volume rate per catchment; CFE takes the same law as a depth rate)"},
    "gw_expon": {"value": float(np.median(expon)), "unit": "-",
                 "source": base + f": CONUS median of the calibrated Expon (p05 {np.percentile(expon,5):.3f}, p95 {np.percentile(expon,95):.3f}); a national prior where no calibration exists"},
    "max_gw_storage_m": {"value": float(np.median(zmax)) / 1000.0, "unit": "m",
                         "source": base + f": CONUS median of the calibrated Zmax {np.median(zmax):.4f} mm (p05 {np.percentile(zmax,5):.2f}, p95 {np.percentile(zmax,95):.2f} mm), converted to m as the NextGen packets carry it"},
}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v["value"] for k, v in out.items()}))

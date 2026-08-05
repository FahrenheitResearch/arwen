#!/usr/bin/env python3
"""Measure how far downstream of a laminar nest inflow boundary a child
domain's resolved turbulence takes to reach its interior value.

THE QUESTION.  A one-way LES child receives smoothed parent fields through
its specified/relaxation boundary.  Nothing perturbs that inflow, so the
air entering the child carries the parent's resolved scales and none of the
child's own.  It has to grow them, and it grows them while being advected,
so the spin-up is a DISTANCE, not only a time.  This module measures that
distance on an already-integrated tree.  It creates no turbulence, changes
no model code, and is deliberately usable on any one-way child whose parent
diagnoses a boundary-layer depth.

THE METRIC, stated before any data was read.

1.  *Where.*  One mass level per frame: the level whose column-mean height
    AGL is nearest ``0.5 * PBLH``, with PBLH taken from the PARENT over the
    child's own footprint.  A PBL-off child writes ``PBLH`` identically
    zero, so reading depth from the child is a silent factor-of-two error
    on exactly this kind of score; the parent-PBLH rule is the correction
    that receipt recorded.

2.  *Which faces are inflow.*  WRF's ``flow_dep_bdy`` convention, read off
    ``gpuwm/ingest/lateral_bc.py`` (``_apply_flow_dependent_boundary_generic``):
    a face is OUTflow where the boundary-normal flux leaves the domain, so
    inflow is ``u > 0`` on west, ``u < 0`` on east, ``v > 0`` on south and
    ``v < 0`` on north.  The wind is the PARENT's, on the child's footprint,
    at the same mid-CBL level, averaged along the face.  Faces are ranked by
    inward normal wind speed; the strongest is the dominant inflow face.

3.  *The fetch coordinate.*  Distance measured perpendicular to an inflow
    face, in child cells, converted to km.  A column is assigned to a face
    only where that face is its NEAREST inflow face, so two inflow faces
    never both claim the same corner.  The profile at distance ``d`` is the
    statistic over every claimed column at that distance.

4.  *The two statistics per strip.*  Both are computed on the child's
    vertical velocity at the mid-CBL level.
      A. ``var(w)`` over the strip's claimed columns.
      B. Child-band spectral energy: the strip's 1-D power spectrum along
         the face-parallel direction, Parseval-normalised so the bins sum
         to the strip variance, integrated over wavelengths between the
         CHILD's 7dx and the PARENT's 7dx.  That band is the one the
         refinement is supposed to add and the parent cannot carry, so it
         isolates newly generated scales from inherited ones.

5.  ``D90``.  The plateau ``P`` is the median of the profile over the
    interior window ``[0.75, 0.95] * D_max``, where ``D_max`` is the
    face-normal extent less the boundary zone at the far side.  The profile
    is smoothed by an 11-strip running median (2.75 km at 250 m, roughly one
    energy-containing eddy) and ``D90`` is the SMALLEST distance at which
    the smoothed profile reaches ``0.9 * P`` and then HOLDS above it for at
    least 5 km.  A first touch is not enough -- a strip variance built from
    a few hundred samples has several per cent of noise and would let one
    blip declare spin-up complete -- and "above for the whole remaining
    domain" is too strict for the same reason in the other direction.  If no
    such distance exists inside the domain the result is recorded as NOT
    REACHED with the domain extent as a lower bound; that outcome is a
    finding, not a failure of the meter.  These three constants were fixed
    against synthetic fields with known ramps (``tests/test_inflow_fetch_meter.py``)
    before the meter was pointed at any model output.

6.  *Normalisation.*  ``D90`` is reported in km, in units of the parent
    ``PBLH`` (z_i) for comparison with the boundary-layer literature, and
    with the along-flow correction ``D90 / cos(theta)`` where ``theta`` is
    the angle between the mean mid-CBL wind and the face normal -- the
    normal distance understates the distance actually travelled whenever
    the flow is oblique to the face.

7.  *The consequence for a whole-domain score.*  A score that averages over
    the entire child averages the fetch zone in with the interior.  So the
    meter also reports, per frame: the fraction of the child's area lying
    inside the fetch zone (nearer than ``D90`` to its nearest inflow face,
    with ``D90`` from the var(w) metric on that column's own face), the
    child's mid-CBL ``var(w)`` over the whole domain and over the interior
    alone, the parent's ``var(w)`` over the matching ground in both cases,
    and the two child/parent ratios.  The difference between those ratios
    is the size of the contamination, measured rather than argued.

USAGE
    python tools/inflow_fetch_meter.py WRFOUT_DIR OUT_JSON \\
        --parent-domain d02 --child-domain d03 \\
        --i-parent-start 318 --j-parent-start 344 --parent-grid-ratio 3 \\
        --child-dx-m 250 --parent-dx-m 750 \\
        --frames 2026-08-01_20_00_00 2026-08-01_21_00_00

No case, campaign or experiment name appears in this file; every geometry
number arrives as an argument.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import netCDF4 as nc
import numpy as np

FACES = ("west", "east", "south", "north")
G = 9.81
BOUNDARY_ZONE_CELLS = 5          # spec_zone 1 + relax_zone 4
SMOOTH_STRIPS = 11               # running median width for the D90 test
PERSIST_KM = 5.0                 # how far a crossing must hold to count
PLATEAU_WINDOW = (0.75, 0.95)    # fraction of the usable range
PLATEAU_FLATNESS_MAX = 1.10      # a plateau still rising this much is none
D90_FRACTION = 0.90


def _unstagger(a: np.ndarray, axis: int) -> np.ndarray:
    lo = [slice(None)] * a.ndim
    hi = [slice(None)] * a.ndim
    lo[axis] = slice(None, -1)
    hi[axis] = slice(1, None)
    return 0.5 * (a[tuple(lo)] + a[tuple(hi)])


def _heights_agl(ds: nc.Dataset) -> np.ndarray:
    ph = np.asarray(ds.variables["PH"][0], dtype=np.float64)
    phb = np.asarray(ds.variables["PHB"][0], dtype=np.float64)
    z = (ph + phb) / G
    zm = 0.5 * (z[:-1] + z[1:])
    hgt = np.asarray(ds.variables["HGT"][0], dtype=np.float64)
    return zm - hgt[None]


def _strip_band_energy(line: np.ndarray, dx_km: float,
                       lam_lo_km: float, lam_hi_km: float) -> float:
    """Variance of one 1-D line that lives between two wavelengths.

    Parseval is enforced exactly rather than approximately: a Hann window
    stops a non-periodic strip leaking across the whole band but removes
    variance, and correcting by the window's mean square only recovers it
    when field and window are uncorrelated -- on an organised field they
    are not.  So the windowed periodogram supplies the SHAPE and the bins
    are rescaled to sum to the strip's own unwindowed variance.  This is
    the same correction the nested score had to make.
    """
    f = np.asarray(line, dtype=np.float64)
    n = f.size
    if n < 8:
        return float("nan")
    f = f - f.mean()
    target = float((f ** 2).mean())
    if target <= 0.0:
        return 0.0
    win = np.hanning(n)
    fw = f * win
    fw = fw - fw.mean()
    coef = np.fft.rfft(fw) / float(n)
    power = np.abs(coef) ** 2
    power[1:] *= 2.0                      # fold the negative frequencies
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    power *= target / total
    k = np.arange(power.size)
    with np.errstate(divide="ignore"):
        lam = n * dx_km / np.maximum(k, 1e-30)
    lam[0] = np.inf
    sel = (lam >= lam_lo_km) & (lam <= lam_hi_km)
    return float(power[sel].sum())


def _running_median(a: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return a.copy()
    half = width // 2
    out = np.empty_like(a)
    for i in range(a.size):
        lo = max(0, i - half)
        hi = min(a.size, i + half + 1)
        window = a[lo:hi]
        window = window[np.isfinite(window)]
        out[i] = np.median(window) if window.size else np.nan
    return out


def _d90(profile: np.ndarray, dist_km: np.ndarray) -> dict:
    """First distance beyond which the profile stays at >= 90 % of plateau."""
    usable = np.isfinite(profile)
    if usable.sum() < 20:
        return {"reached": False, "reason": "too few usable strips"}
    idx = np.flatnonzero(usable)
    lo_i, hi_i = int(idx[0]), int(idx[-1])
    span = dist_km[hi_i] - dist_km[lo_i]
    plateau_lo = dist_km[lo_i] + PLATEAU_WINDOW[0] * span
    plateau_hi = dist_km[lo_i] + PLATEAU_WINDOW[1] * span
    win = usable & (dist_km >= plateau_lo) & (dist_km <= plateau_hi)
    if win.sum() < 3:
        return {"reached": False, "reason": "empty plateau window"}
    plateau = float(np.median(profile[win]))
    if not np.isfinite(plateau) or plateau <= 0.0:
        return {"reached": False, "reason": "non-positive plateau"}
    # Is the "plateau" actually flat?  If the profile is still climbing where
    # the plateau is measured then the fetch is longer than the domain and
    # any D90 read off it would be an artefact of where the domain ends.
    wi = np.flatnonzero(win)
    half = wi[wi.size // 2]
    first = float(np.median(profile[wi[wi < half]])) if (wi < half).sum() else np.nan
    second = float(np.median(profile[wi[wi >= half]]))
    flatness = second / first if first and np.isfinite(first) else np.nan
    if not np.isfinite(flatness) or flatness > PLATEAU_FLATNESS_MAX:
        return {"reached": False, "plateau": plateau,
                "plateau_flatness_ratio": float(flatness),
                "lower_bound_km": float(dist_km[hi_i]),
                "reason": "the plateau window is still rising "
                          f"({flatness:.3f} > {PLATEAU_FLATNESS_MAX}); the "
                          "profile has not converged inside the domain"}
    smooth = _running_median(profile, SMOOTH_STRIPS)
    target = D90_FRACTION * plateau
    above = smooth >= target
    above[~usable] = True                 # do not let masked strips break a run
    # A finite profile is noisy, so a crossing counts only if it HOLDS: the
    # first strip from which the smoothed profile stays above the target for
    # at least PERSIST_KM.  A single blip cannot declare spin-up complete,
    # and a single far-field dip cannot un-declare it.
    step_km = float(dist_km[1] - dist_km[0]) if dist_km.size > 1 else 0.0
    hold = max(1, int(round(PERSIST_KM / step_km))) if step_km > 0 else 1
    ok_from = None
    for i in range(lo_i, hi_i + 1):
        end = min(hi_i + 1, i + hold)
        if end - i < min(hold, hi_i + 1 - i):
            break
        if above[i:end].all():
            ok_from = i
            break
    if ok_from is None:
        return {"reached": False, "plateau": plateau,
                "lower_bound_km": float(dist_km[hi_i]),
                "reason": "profile never holds 90 % of plateau for "
                          f"{PERSIST_KM} km"}
    return {
        "reached": True,
        "plateau": plateau,
        "plateau_flatness_ratio": float(flatness),
        "d90_km": float(dist_km[ok_from]),
        "d90_strip_index": int(ok_from),
        "value_at_d90": float(smooth[ok_from]),
    }


def _face_geometry(face: str, ny: int, nx: int):
    """(distance index array shape (ny,nx), along-face index array, extent)."""
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    if face == "west":
        return ii, jj, nx
    if face == "east":
        return nx - 1 - ii, jj, nx
    if face == "south":
        return jj, ii, ny
    return ny - 1 - jj, ii, ny


def measure_frame(parent_path: Path, child_path: Path, *,
                  i_parent_start: int, j_parent_start: int,
                  parent_grid_ratio: int, child_dx_m: float,
                  parent_dx_m: float, stamp: str) -> dict:
    with nc.Dataset(parent_path) as a, nc.Dataset(child_path) as b:
        a.set_auto_mask(False)
        b.set_auto_mask(False)
        w = _unstagger(np.asarray(b.variables["W"][0], dtype=np.float64), 0)
        z = _heights_agl(b)
        nz, ny, nx = w.shape
        foot = nx // parent_grid_ratio
        j0, i0 = j_parent_start - 1, i_parent_start - 1
        sl = (slice(j0, j0 + foot), slice(i0, i0 + foot))

        pblh = float(np.mean(np.asarray(
            a.variables["PBLH"][0], dtype=np.float64)[sl]))
        zcol = z.mean(axis=(1, 2))
        k = int(np.argmin(np.abs(zcol - 0.5 * max(pblh, 200.0))))
        k = max(1, min(k, nz - 1))

        up = _unstagger(np.asarray(a.variables["U"][0], dtype=np.float64), 2)
        vp = _unstagger(np.asarray(a.variables["V"][0], dtype=np.float64), 1)
        # the parent level nearest the same height, over the same ground
        zp = _heights_agl(a)[:, sl[0], sl[1]]
        kp = int(np.argmin(np.abs(zp.mean(axis=(1, 2)) - zcol[k])))
        u_f = up[kp][sl]
        v_f = vp[kp][sl]
        ubar, vbar = float(u_f.mean()), float(v_f.mean())
        parent_field = _unstagger(
            np.asarray(a.variables["W"][0], dtype=np.float64), 0)[kp][sl]
        parent_foot_w = parent_field.shape[0]

        # face-mean inward normal component of the parent wind
        edge = max(1, foot // 20)
        inward = {
            "west": float(u_f[:, :edge].mean()),
            "east": float(-u_f[:, -edge:].mean()),
            "south": float(v_f[:edge, :].mean()),
            "north": float(-v_f[-edge:, :].mean()),
        }
        inflow_faces = [f for f in FACES if inward[f] > 0.0]

    field = w[k]
    child_dx_km = child_dx_m / 1000.0
    lam_lo = 7.0 * child_dx_m / 1000.0
    lam_hi = 7.0 * parent_dx_m / 1000.0

    # nearest-inflow-face ownership
    dist_maps = {f: _face_geometry(f, ny, nx)[0] for f in FACES}
    if inflow_faces:
        nearest = np.min(np.stack([dist_maps[f] for f in inflow_faces]), axis=0)
    else:
        nearest = np.full((ny, nx), -1)

    faces_out = {}
    for face in inflow_faces:
        dmap, pmap, extent = _face_geometry(face, ny, nx)
        owns = dmap <= nearest
        d_max_idx = extent - 1 - BOUNDARY_ZONE_CELLS
        dist_km = np.arange(extent) * child_dx_km
        var_prof = np.full(extent, np.nan)
        band_prof = np.full(extent, np.nan)
        for d in range(BOUNDARY_ZONE_CELLS, d_max_idx + 1):
            sel = (dmap == d) & owns
            if sel.sum() < 32:
                continue
            # strips are whole rows/columns, so order along the face is the
            # along-face index; take the contiguous owned run
            line = field[sel]
            order = np.argsort(pmap[sel])
            line = line[order]
            var_prof[d] = float(line.var())
            band_prof[d] = _strip_band_energy(line, child_dx_km, lam_lo, lam_hi)

        theta = None
        speed = math.hypot(ubar, vbar)
        if speed > 1e-6:
            normal = {"west": (1.0, 0.0), "east": (-1.0, 0.0),
                      "south": (0.0, 1.0), "north": (0.0, -1.0)}[face]
            cos_t = (ubar * normal[0] + vbar * normal[1]) / speed
            theta = math.degrees(math.acos(max(-1.0, min(1.0, cos_t))))

        # ---- controls ------------------------------------------------
        # C1, the meteorological control: the SAME profile computed on the
        # parent over the same ground.  The parent has no inflow spin-up
        # problem of its own at this scale -- it inherited its turbulence
        # from far upstream -- so any structure the two profiles share is
        # the weather and the terrain, not the child's fetch.
        pdmap, ppmap, pextent = _face_geometry(face, parent_foot_w,
                                               parent_foot_w)
        pnear = (np.min(np.stack([_face_geometry(f, parent_foot_w,
                                                 parent_foot_w)[0]
                                  for f in inflow_faces]), axis=0)
                 if inflow_faces else np.full((parent_foot_w,) * 2, -1))
        powns = pdmap <= pnear
        parent_dx_km = parent_dx_m / 1000.0
        pdist_km = np.arange(pextent) * parent_dx_km
        pvar = np.full(pextent, np.nan)
        for d in range(2, pextent - 2):
            sel = (pdmap == d) & powns
            if sel.sum() < 32:
                continue
            pvar[d] = float(parent_field[sel].var())

        # C2, the self-normalised control: the FRACTION of the strip's own
        # variance that lives in the child band.  Terrain and mesoscale
        # forcing modulate the whole spectrum together, so they largely
        # divide out; growth of newly resolved small scales does not.
        with np.errstate(invalid="ignore", divide="ignore"):
            frac_prof = band_prof / var_prof
        frac_prof[~np.isfinite(frac_prof)] = np.nan

        entry = {
            "inward_normal_wind_ms": inward[face],
            "wind_to_face_normal_deg": theta,
            "profile_distance_km": dist_km.tolist(),
            "var_w_profile": var_prof.tolist(),
            "band_energy_profile": band_prof.tolist(),
            "band_fraction_profile": frac_prof.tolist(),
            "parent_profile_distance_km": pdist_km.tolist(),
            "parent_var_w_profile": pvar.tolist(),
            "d90_var_w": _d90(var_prof, dist_km),
            "d90_band_energy": _d90(band_prof, dist_km),
            "d90_band_fraction": _d90(frac_prof, dist_km),
            "d90_parent_var_w_control": _d90(pvar, pdist_km),
        }
        for key in ("d90_var_w", "d90_band_energy", "d90_band_fraction",
                    "d90_parent_var_w_control"):
            res = entry[key]
            if res.get("reached"):
                res["d90_zi"] = res["d90_km"] * 1000.0 / pblh if pblh > 0 else None
                if theta is not None and abs(math.cos(math.radians(theta))) > 0.2:
                    res["d90_alongflow_km"] = (
                        res["d90_km"] / abs(math.cos(math.radians(theta))))
                    res["d90_alongflow_zi"] = (
                        res["d90_alongflow_km"] * 1000.0 / pblh
                        if pblh > 0 else None)
        faces_out[face] = entry

    # --- what the fetch zone does to a whole-domain score ---------------
    zone = np.zeros((ny, nx), dtype=bool)
    for face in inflow_faces:
        res = faces_out[face]["d90_var_w"]
        dmap = dist_maps[face]
        owns = dmap <= nearest
        reach = (res["d90_km"] / child_dx_km if res.get("reached")
                 else float(_face_geometry(face, ny, nx)[2]))
        zone |= owns & (dmap < reach)
    interior = ~zone
    ratio_block = {
        "fetch_zone_area_fraction": float(zone.mean()),
        "child_var_w_whole_domain": float(field.var()),
        "child_var_w_interior": (float(field[interior].var())
                                 if interior.sum() > 64 else None),
    }
    # the same ground on the parent, by integer decimation of the child mask
    r = parent_grid_ratio
    par_int = interior[::r, ::r][:parent_foot_w, :parent_foot_w]
    ratio_block["parent_var_w_whole_footprint"] = float(parent_field.var())
    ratio_block["parent_var_w_interior"] = (
        float(parent_field[par_int].var()) if par_int.sum() > 16 else None)
    ratio_block["var_w_ratio_whole_domain"] = (
        ratio_block["child_var_w_whole_domain"]
        / max(ratio_block["parent_var_w_whole_footprint"], 1e-12))
    if (ratio_block["child_var_w_interior"] is not None
            and ratio_block["parent_var_w_interior"] is not None):
        ratio_block["var_w_ratio_interior"] = (
            ratio_block["child_var_w_interior"]
            / max(ratio_block["parent_var_w_interior"], 1e-12))
    else:
        ratio_block["var_w_ratio_interior"] = None

    return {
        "time": stamp,
        "pblh_parent_footprint_m": pblh,
        "whole_domain_vs_interior": ratio_block,
        "mid_cbl_level_index": k,
        "mid_cbl_height_m": float(zcol[k]),
        "parent_mean_wind_ms": {"u": ubar, "v": vbar,
                                "speed": math.hypot(ubar, vbar)},
        "inward_normal_wind_ms": inward,
        "inflow_faces": inflow_faces,
        "child_band_km": [lam_lo, lam_hi],
        "faces": faces_out,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wrfout", type=Path)
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--parent-domain", default="d02")
    ap.add_argument("--child-domain", default="d03")
    ap.add_argument("--i-parent-start", type=int, required=True)
    ap.add_argument("--j-parent-start", type=int, required=True)
    ap.add_argument("--parent-grid-ratio", type=int, required=True)
    ap.add_argument("--child-dx-m", type=float, required=True)
    ap.add_argument("--parent-dx-m", type=float, required=True)
    ap.add_argument("--frames", nargs="*", default=None,
                    help="timestamps to score; default = every paired frame")
    args = ap.parse_args(argv)

    stamps = sorted(
        re.sub(rf"^wrfout_{args.child_domain}_", "", p.name)
        for p in args.wrfout.glob(f"wrfout_{args.child_domain}_*"))
    if args.frames:
        stamps = [s for s in stamps if s in set(args.frames)]
    frames = []
    for s in stamps:
        a = args.wrfout / f"wrfout_{args.parent_domain}_{s}"
        b = args.wrfout / f"wrfout_{args.child_domain}_{s}"
        if a.exists() and b.exists():
            frames.append((s, a, b))
    if not frames:
        raise SystemExit("no paired parent/child frames selected")

    rows = []
    for s, a, b in frames:
        print(f"metering {s}", flush=True)
        rows.append(measure_frame(
            a, b, i_parent_start=args.i_parent_start,
            j_parent_start=args.j_parent_start,
            parent_grid_ratio=args.parent_grid_ratio,
            child_dx_m=args.child_dx_m, parent_dx_m=args.parent_dx_m,
            stamp=s))

    summary = {}
    for metric in ("d90_var_w", "d90_band_energy", "d90_band_fraction",
                   "d90_parent_var_w_control"):
        for scope, pick in (("dominant_face", True), ("all_inflow_faces", False)):
            vals_km, vals_zi, misses = [], [], 0
            for r in rows:
                items = r["faces"].items()
                if pick and r["inflow_faces"]:
                    best = max(r["inflow_faces"],
                               key=lambda f: r["inward_normal_wind_ms"][f])
                    items = [(best, r["faces"][best])]
                for _, e in items:
                    res = e[metric]
                    if res.get("reached"):
                        vals_km.append(res["d90_km"])
                        if res.get("d90_zi") is not None:
                            vals_zi.append(res["d90_zi"])
                    else:
                        misses += 1
            summary[f"{metric}__{scope}"] = {
                "n_reached": len(vals_km),
                "n_not_reached": misses,
                "mean_km": float(np.mean(vals_km)) if vals_km else None,
                "sd_km": float(np.std(vals_km, ddof=1)) if len(vals_km) > 1 else None,
                "min_km": float(np.min(vals_km)) if vals_km else None,
                "max_km": float(np.max(vals_km)) if vals_km else None,
                "mean_zi": float(np.mean(vals_zi)) if vals_zi else None,
                "sd_zi": float(np.std(vals_zi, ddof=1)) if len(vals_zi) > 1 else None,
            }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps({"instrument": "inflow-fetch-meter-v1",
                    "frames": rows, "summary": summary}, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

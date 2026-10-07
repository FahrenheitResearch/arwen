"""Measurement tooling: the DA controller's analysis step around the LETKF.

Runs the tree's own assimilate_radar_grid (H(x), batches, dispersion gate
and its withheld solves, positivity, saturation bound, restagger) and the
controller's own pre/post code (spread screen, member staging, increment
statistics, merge, member-0 record) on 32 synthetic Thompson members at
241x241x49, with a fast deterministic stand-in for the LETKF transform so
the data movement around the solve is what is timed.  Host only.

    python -m tools.da_analysis_host_bench setup --root DIR
    python -m tools.da_analysis_host_bench run --root DIR --label L

``--small`` shrinks the case to 8 members at 10x61x61 for a smoke test.
Prints one JSON line per run; ``pending_sha256`` fingerprints the accepted
increments so two trees can be compared for byte identity.
"""
from __future__ import annotations

import argparse
import json
try:
    import resource
except ImportError:
    resource = None
import shutil
import sys
import time
from pathlib import Path

import numpy as np

R, NZ, NY, NX = 32, 49, 241, 241
FIELDS = ("thp", "qv", "u", "v", "qs", "qg", "qc", "nc", "qr", "nr", "qi",
          "ni", "nwfa", "nifa")
def cycle_argv():
    """The deck's analysis flags (receipts/da-arm-argv.json), host solve."""
    return ["--source", "rap-native", "--prepared-root", "/x", "--physics-profile",
        "thompson-mp28-mynn-gsd41-mynn-ruc-rrtmg-legacy-v1",
        "--proof-sha256", "0" * 64, "--source-manifest-sha256", "0" * 64,
        "--prepared-content-sha256", "0" * 64, "--members", str(R),
        "--solve-device", "host", "--memory-budget-mib", "14336",
        "--horizontal-loc-m", "12000", "--vertical-loc-m", "6000",
        "--hydrometeors", "--reflectivity-analysis", "--clear-air-analysis",
        "--positivity-policy", "clip", "--velocity-dispersion-gate", "2",
        "--velocity-dispersion-batch-gate", "3", "--no-hotstart",
        "--run-seconds", "7200", "--history-interval-seconds", "120", "--out", "/x"]


def rss():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024**2
    except OSError:
        return float("nan")


def peak():
    if resource is None:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def grid():
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid
    projection = LambertGrid(ref_lat=31.0, ref_lon=-89.0, truelat1=30.0,
                             truelat2=60.0, stand_lon=-89.0, dx=3000.0,
                             dy=3000.0, e_we=NX + 1, e_sn=NY + 1)
    eta = np.linspace(0.0, 1.0, NZ + 1)
    return TargetGrid.from_projection(projection, z_w=20.0 + 19000.0 * eta**1.5,
                                      name="bench")


def setup(root):
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import GriddedObservations, SuperobParams
    from gpuwm.da import obsop
    root.mkdir(parents=True, exist_ok=True)
    g = grid()
    rng = np.random.default_rng(5)
    shape = (NZ, NY, NX)
    column = np.linspace(1.0e5, 2.0e4, NZ)[:, None, None]
    hydro = np.zeros(shape, np.float32)
    jj, ii = np.mgrid[0:NY, 0:NX]
    storms = np.zeros((NY, NX))
    for cj, ci in rng.integers(NY // 10, NY - NY // 10, size=(12, 2)):
        storms += np.exp(-(((jj - cj) / 9.0) ** 2 + ((ii - ci) / 9.0) ** 2))
    hydro[:] = (storms[None] * np.exp(-((np.arange(NZ) - 15) / 10.0) ** 2)
                [:, None, None]).astype(np.float32)
    for m in range(R):
        f = lambda s=shape: rng.random(s, dtype=np.float32)
        lognormal = lambda: np.exp(0.7 * rng.standard_normal(shape)).astype(np.float32)
        state = {"thp": 300 + f(), "qv": (0.01 * (0.9 + 0.2 * f())),
                 "u": 5 + rng.standard_normal((NZ, NY, NX + 1)).astype(np.float32),
                 "v": rng.standard_normal((NZ, NY + 1, NX)).astype(np.float32),
                 "w": 0.3 * rng.standard_normal((NZ + 1, NY, NX)).astype(np.float32),
                 "php": f((NZ + 1, NY, NX)), "mup": f((NY, NX)),
                 "p": (column * (1 + 0.001 * f())).astype(np.float32),
                 "al": f(), "alt": (0.8 + 0.4 * f()), "h_diabatic": f(),
                 "effc": f(), "effi": f(), "effs": f(),
                 "nwfa2d": f((NY, NX)), "nifa2d": f((NY, NX))}
        for name, scale in (("qs", 1e-3), ("qg", 5e-4), ("qc", 5e-4),
                            ("qr", 1e-3), ("qi", 1e-4)):
            state[name] = (scale * hydro * lognormal()).astype(np.float32)
        for name, scale in (("nc", 1e8), ("nr", 1e4), ("ni", 1e5)):
            state[name] = (scale * hydro * lognormal()).astype(np.float32)
        state["nwfa"] = (1e9 * (0.5 + f())).astype(np.float32)
        state["nifa"] = (1e4 * (0.5 + f())).astype(np.float32)
        np.savez(root / f"gpuwmrst_d01_m{m:03d}.npz",
                 **{f"state/{k}": v for k, v in state.items()},
                 **{f"driver/x{i:02d}": f() for i in range(30)})
        np.save(root / f"hz_{m:03d}.npy",
                (60 * hydro * lognormal() - 30).astype(np.float64))
    # Observations through the radar lane's own writer.
    nrad = 8
    geometry = obsop.GridGeometry.from_target_grid(g)
    east = np.zeros((nrad,) + shape, np.float32)
    north, up = east.copy(), east.copy()
    vr_obs = np.zeros((nrad,) + shape, np.float32)
    vr_mask = np.zeros((nrad,) + shape, np.int8)
    sites = []
    for slot, (cj, ci) in enumerate(rng.integers(30, NY - 30, size=(nrad, 2))):
        site = obsop.RadarSite(latitude_deg=float(g.lat[cj, ci]),
                               longitude_deg=float(g.lon[cj, ci]),
                               altitude_m=100.0, name=f"K{slot:03d}")
        sites.append(site)
        beam = obsop.beam_geometry(geometry, site)
        e, n, u = (np.broadcast_to(np.asarray(c, np.float64), shape)
                   for c in beam.unit_vector_enu())
        east[slot], north[slot], up[slot] = e, n, u
        disc = ((jj - cj) ** 2 + (ii - ci) ** 2) <= (NY // 3) ** 2
        mask = (rng.random(shape) < 0.03) & disc[None]
        vr_mask[slot] = mask
        vr_obs[slot] = np.where(mask, 5 * e + rng.standard_normal(shape), 0)
    z_mask = (rng.random(shape) < 0.03) & (hydro > 0.05)
    z_obs = np.where(z_mask, 40 * hydro + rng.standard_normal(shape) * 3, 0)
    z0 = (rng.random(shape) < 0.04) & (hydro < 0.01)
    obs = GriddedObservations(
        z_obs=z_obs.astype(np.float32), z_mask=z_mask.astype(np.int8),
        z_err=np.where(z_mask, 4.0, 0.0).astype(np.float32),
        z_max=z_obs.astype(np.float32), z_mean=z_obs.astype(np.float32),
        z_count=z_mask.astype(np.int32),
        z0_mask=z0.astype(np.int8), z0_count=(z0 * 5).astype(np.int32),
        z0_err=np.where(z0, 7.5, 0.0).astype(np.float32),
        vr_obs=vr_obs, vr_mask=vr_mask,
        vr_err=np.where(vr_mask.astype(bool), 1.5, 0.0).astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((nrad,) + shape, np.int32),
        vr_beam_east=east, vr_beam_north=north, vr_beam_up=up,
        radars=[{"id": s.name, "lat_deg": s.latitude_deg,
                 "lon_deg": s.longitude_deg, "alt_m": s.altitude_m,
                 "valid_time": "2026-10-03T23:00:00Z"} for s in sites],
        counts=[], provenance=[])
    write_radar_grid(root / "radar.nc", obs, g,
                     valid_time="2026-10-03T23:00:00Z",
                     params=SuperobParams(), overwrite=True)
    print("setup done", flush=True)


def runner(prior, batches, geometry, config, diagnostics, **options):
    """Stand-in transform: deterministic, whole-ensemble float64 output of
    the solver's exact shapes, so everything around it moves real bytes."""
    out = {}
    for name in config.analysis_fields:
        x = prior[name]
        out[name] = 0.05 * (x - x.mean(axis=0, keepdims=True))
    diagnostics.active_points = int(np.prod(prior[config.analysis_fields[0]].shape[1:]))
    diagnostics.total_points = diagnostics.active_points
    return out


def run(root, label):
    from gpuwm.da import moments
    from gpuwm.da import radar_assimilation as ra
    from gpuwm.da.obs_radar import read_document
    from tools import da_cycle_prepared as cp_
    new = hasattr(ra, "CheckpointStateView")
    g = grid()
    args = cp_.build_parser().parse_args(cycle_argv())
    hz = {m: np.load(root / f"hz_{m:03d}.npy") for m in range(R)}
    doc = read_document(root / "radar.nc", expected_grid=g)
    restarts = {m: root / f"gpuwmrst_d01_m{m:03d}.npz" for m in range(R)}
    t = {}
    # What the controller holds when the analysis starts: base = the
    # whole-state mirrors load_result produced; new = restart views.
    if new:
        snapshots = {m: ra.CheckpointStateView(p) for m, p in restarts.items()}
    else:
        snapshots = {}
        for m, p in restarts.items():
            with np.load(p) as d:
                snapshots[m] = {k[6:]: np.array(d[k]) for k in d.files
                                if k.startswith("state/")}
    out = {"label": label, "tree": "new" if new else "base",
           "rss_at_analysis_start_gib": rss()}
    started = time.perf_counter()
    stage = root / f"stage-{label}"
    if new:
        checkpoints = cp_.analysis_backgrounds(snapshots, R, directory=stage,
                                              t_end=7200.0, files=False)
    else:
        checkpoints = {}
        for m in range(R):
            d = stage / f"member_{m:03d}"
            d.mkdir(parents=True)
            path = d / "gpuwmrst_d01_007200.npz"
            np.savez(path, **{f"state/{k}": v for k, v in snapshots[m].items()})
            checkpoints[m] = path
    t["stage_inputs_s"] = time.perf_counter() - started
    started = time.perf_counter()
    carried = set(snapshots[0])
    candidates = tuple(n for n in moments.analysis_fields(28) if n in carried)
    for name in candidates:
        stack = np.stack([snapshots[i][name] for i in range(R)]).astype(np.float64)
        float(np.abs(stack).max()); float(stack.std(axis=0, ddof=1).max())
        del stack
    t["spread_screen_s"] = time.perf_counter() - started
    cfg = cp_.plan_radar_assimilation(args, 28, analysis_fields=candidates,
                                      cwp=False)
    started = time.perf_counter()
    increments, prov = ra.assimilate_radar_grid(
        checkpoints, root / "radar.nc", g, cfg,
        reflectivity_provider=lambda i, s: hz[int(i)], analysis_runner=runner)
    t["assimilate_s"] = time.perf_counter() - started
    out["rss_after_assimilate_gib"] = rss()
    started = time.perf_counter()
    for field in candidates:
        stack = np.stack([increments[i][field] for i in sorted(increments)])
        float(np.abs(stack).max())
        float(np.mean(stack != 0.0))
        del stack
    pending = {}
    for index in range(R):
        merged, _, _ = cp_.merge_hotstart_increments(
            increments[index], {}, prior=snapshots[index],
            positivity_policy="clip", report_overlap=index == 0)
        pending[index] = merged
    if new:
        cp_.write_member_increments(root / f"inc-{label}.npz", pending[0]) \
            if hasattr(cp_, "write_member_increments") else None
    else:
        np.savez_compressed(root / f"inc-{label}.npz",
                            **{k: v.astype(np.float32) for k, v in pending[0].items()})
    shutil.rmtree(stage, ignore_errors=True)
    if hasattr(cp_, "release_backgrounds") and new:
        increments = None
        checkpoints = None
        cp_.release_backgrounds(snapshots)
    t["controller_post_s"] = time.perf_counter() - started
    out.update(t)
    out["analysis_step_s"] = sum(t.values())
    out["rss_after_merge_gib"] = rss()
    out["peak_rss_gib"] = peak()
    out["positivity"] = {k: prov["positivity"].get(k) for k in ("negative_points", "mass_added_by_clip")}
    out["withheld_solves"] = len((prov.get("velocity_dispersion") or {}).get("withheld") or [])
    # Fingerprint of the accepted increments: must match across trees.
    import hashlib
    h = hashlib.sha256()
    for m in range(R):
        for k in sorted(pending[m]):
            h.update(k.encode()); h.update(np.ascontiguousarray(pending[m][k]).tobytes())
    out["pending_sha256"] = h.hexdigest()
    print(json.dumps(out), flush=True)


def main(argv=None):
    global R, NZ, NY, NX
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("setup", "run"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--label", default="x")
    p.add_argument("--small", action="store_true")
    a = p.parse_args(argv)
    if a.small:
        R, NZ, NY, NX = 8, 10, 61, 61
    setup(a.root) if a.mode == "setup" else run(a.root, a.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())

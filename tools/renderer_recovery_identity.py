"""Record emitted configuration bytes and bounded production forecast bytes.

Run the same helper against a staging source and a recovery source, then use
``--compare`` on the recovery run.  The dry and moist forecasts each take
three fixed steps through ``execute_experiment``.  Existing verification
initializers own their numerical construction.  Only byte hashes and small
receipts are saved; no prepared inputs or history arrays are retained.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def emitted_config_bytes():
    """Use the configuration front door's real text emitter."""
    from gpuwm.domain_wizard import render_config
    from gpuwm.physics_compat import MYNN_RUC_PROFILE_ID

    result = {}
    for name, source, profile in (
            ("generic", "gfs", None),
            ("non_feature_gfs", "gfs", MYNN_RUC_PROFILE_ID),
            ("non_feature_era5", "era5", MYNN_RUC_PROFILE_ID),
            ("non_feature_rrfs", "rrfs", MYNN_RUC_PROFILE_ID)):
        text = render_config(
            name="renderer-identity", start_time=datetime(2026, 10, 2, 21),
            hours=1,
            projection={"map_proj": "lambert", "ref_lat": 38.5,
                        "ref_lon": -97.5, "truelat1": 38.5,
                        "truelat2": 38.5, "stand_lon": -97.5},
            dims=[(50, 50)], ratios=(), root_dx_m=3000,
            fetch_hints={"source": source}, case_data=None, profile=profile)
        payload = text.encode("utf-8")
        result[name] = {"bytes": len(payload), "sha256": _digest(payload)}
    return result


def _small_forecast(profile):
    import cupy as cp
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.model import execute_experiment
    from gpuwm.experiment import (
        DomainConfig, ExperimentConfig, ProjectionConfig, VerticalConfig)
    from gpuwm.io.restart import configuration_echo
    from gpuwm.verify.cases import straka, wk82
    from gpuwm.verify.cases.nest_ideal_common import (
        assemble_idealized_tree, consume_history_reflectivity)
    from tilestream.physics_inventory import carrier_manifest, carrier_scalars

    if profile == "generic_dry":
        cfg = replace(straka.default_config(), nx=24, ny=24, nz=32,
                      dx=1000., dy=1000., dt=1., run_seconds=3.,
                      output_interval_s=1.)
        sounding, build = straka.sounding, straka.build
    else:
        cfg = replace(wk82.default_config(), nx=24, ny=24, nz=32,
                      dx=3000., dy=3000., dt=6., run_seconds=18.,
                      output_interval_s=6., mp_physics=6)
        sounding, build = lambda z: wk82.wk82_sounding(z)[0], wk82.build
    domain = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=cfg.output_interval_s, run=cfg,
        time_step=int(cfg.dt))
    experiment = ExperimentConfig(
        name="renderer_identity", start_time=datetime(2000, 1, 1),
        run_seconds=cfg.run_seconds, vertical=VerticalConfig((), 0., 1, .2),
        projection=ProjectionConfig("lambert", 35., -97., 30., 60., -97.),
        restart_interval_s=0., domains=(domain,))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, sounding, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = build(cfg, coord, base)
    model = assemble_idealized_tree(experiment, state)
    history, steps = [], []

    def snapshot(_model, node, ticks):
        consume_history_reflectivity(node, ticks)
        history.append(node.clock.elapsed_seconds)

    def observe(**row):
        steps.append({key: row[key] for key in ("dt",)})

    started = time.perf_counter()
    execute_experiment(model, experiment=experiment, history_handler=snapshot,
                       step_observer=observe, validate_state=True)
    cp.cuda.Stream.null.synchronize()
    assert model.root.clock.elapsed_seconds == cfg.run_seconds
    assert len(steps) == 3, steps
    fields = {}
    for name, value in sorted(carrier_manifest(state).items()):
        host = value.get() if hasattr(value, "get") else value
        payload = canonical_bytes({"dtype": host.dtype.str,
                                   "shape": list(host.shape)})
        payload += host.tobytes(order="C")
        if host.dtype.kind == "f":
            assert bool(cp.isfinite(cp.asarray(host)).all()), name
        fields[name] = {"shape": list(host.shape), "dtype": host.dtype.str,
                        "bytes": len(host.tobytes(order="C")),
                        "sha256": _digest(payload)}
    assert fields
    identity = {"configuration_sha256": _digest(canonical_bytes(asdict(cfg))),
                "checkpoint_config_sha256": _digest(
                    canonical_bytes(configuration_echo(cfg))),
                "history_seconds": history, "steps": steps,
                "elapsed_seconds": state.elapsed_seconds,
                "fields": fields, "scalars": carrier_scalars(state)}
    return identity, time.perf_counter() - started


def compare_identity(reference, candidate):
    """No timing, installation path or source revision enters this comparison."""
    return reference["identity"] == candidate["identity"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    import gpuwm
    result = {"schema": "gpuwm-renderer-recovery-identity-v1",
              "source_revision": args.revision, "status": "PASS",
              "source_package": str(Path(gpuwm.__file__).resolve()),
              "runtime": sys.executable,
              "identity": {"emitted_config_bytes": emitted_config_bytes()}}
    if not args.cpu_only:
        import cupy as cp
        from cupy.cuda import nvrtc
        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = properties["name"]
        result["device"] = name.decode() if isinstance(name, bytes) else name
        result["cuda_runtime_version"] = cp.cuda.runtime.runtimeGetVersion()
        result["nvrtc_version"] = list(nvrtc.getVersion())
        result["forecast_wall_seconds"] = {}
        forecasts = result["identity"]["forecasts"] = {}
        for profile in ("generic_dry", "non_feature_moist"):
            forecasts[profile], result["forecast_wall_seconds"][profile] = (
                _small_forecast(profile))
    if args.compare:
        reference = json.loads(args.compare.read_text(encoding="utf-8"))
        result["reference_revision"] = reference["source_revision"]
        result["byte_identical"] = compare_identity(reference, result)
        if not result["byte_identical"]:
            result["status"] = "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items()
                      if key != "identity"}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

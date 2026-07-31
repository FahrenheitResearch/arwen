#!/usr/bin/env python3
"""Run a small idealized two-domain microphysics-edge integration.

This is evidence tooling, not a new forecast configuration surface.  It uses
the production parent-only initializer, NestCoupler, physics drivers, and
recursive experiment executor, then writes a strict JSON receipt containing
the edge contract, finite-state census, target moment checks, and a final
prognostic-state digest.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time


PORTED = (1, 6, 8, 10, 18)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-mp", type=int, choices=PORTED, required=True)
    parser.add_argument("--child-mp", type=int, choices=PORTED, required=True)
    parser.add_argument("--minutes", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[1],
        help="code checkout to import; supports an immutable baseline checkout")
    return parser.parse_args(argv)


def _host(value):
    import cupy as cp
    import numpy as np

    if isinstance(value, cp.ndarray):
        value = cp.asnumpy(value)
    return np.ascontiguousarray(value)


def _field_statistics(state, names):
    import numpy as np

    result = {}
    for name in names:
        attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(name, name)
        value = getattr(state, attr, None)
        if value is None:
            continue
        host = _host(value)
        finite = np.isfinite(host)
        result[name] = {
            "count": int(host.size),
            "finite_count": int(finite.sum()),
            "nonfinite_count": int(host.size - finite.sum()),
            "minimum": float(np.min(host)) if finite.all() else None,
            "maximum": float(np.max(host)) if finite.all() else None,
            "nonzero_count": int(np.count_nonzero(host)),
        }
    return result


def _moment_consistency(state, mp):
    import numpy as np

    pairs = {
        8: (("qr", "nr"), ("qi", "ni")),
        10: (
            ("qr", "nr"), ("qi", "ni"), ("qs", "ns"), ("qg", "ng"),
        ),
        18: (
            ("qc", "qndrop"), ("qr", "qnr"), ("qi", "qni"),
            ("qs", "qns"), ("qg", "qng"), ("qh", "qnh"),
        ),
    }.get(mp, ())
    checks = []
    for mass_name, moment_name in pairs:
        mass = _host(getattr(state, mass_name))
        moment = _host(getattr(state, moment_name))
        active = mass > np.float32(1.0e-12)
        bad = active & (
            ~np.isfinite(moment) | (moment <= np.float32(0.0)))
        checks.append({
            "mass_field": mass_name,
            "moment_field": moment_name,
            "active_mass_count": int(active.sum()),
            "invalid_active_moment_count": int(bad.sum()),
            "pass": bool(not bad.any()),
        })
    if mp == 18:
        for mass_name, volume_name in (
                ("qg", "qvolg"), ("qh", "qvolh")):
            mass = _host(getattr(state, mass_name))
            volume = _host(getattr(state, volume_name))
            active = mass > np.float32(1.0e-12)
            bad = active & (
                ~np.isfinite(volume) | (volume <= np.float32(0.0)))
            checks.append({
                "mass_field": mass_name,
                "moment_field": volume_name,
                "active_mass_count": int(active.sum()),
                "invalid_active_moment_count": int(bad.sum()),
                "pass": bool(not bad.any()),
            })
    return checks


def _state_digest(state, names):
    digest = hashlib.sha256()
    inventory = []
    for name in names:
        attr = {"t": "thp", "ph": "php", "mu": "mup"}.get(name, name)
        value = getattr(state, attr, None)
        if value is None:
            continue
        host = _host(value)
        digest.update(name.encode("ascii"))
        digest.update(str(host.shape).encode("ascii"))
        digest.update(host.dtype.str.encode("ascii"))
        digest.update(host.tobytes(order="C"))
        inventory.append(name)
    return digest.hexdigest(), inventory


def run(args):
    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root))

    import cupy as cp
    import numpy as np

    from gpuwm.config import validate_run_config
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.preflight import nest_field_kinds
    from gpuwm.experiment import (
        DomainConfig, ExperimentConfig, ProjectionConfig, VerticalConfig,
    )
    from gpuwm.verify.cases import wk82
    from gpuwm.verify.cases.nest_ideal_common import (
        assemble_idealized_tree, consume_history_reflectivity,
    )

    seconds = int(args.minutes) * 60
    if seconds < 1800:
        raise ValueError("integration must cover at least 30 child minutes")
    mixed = args.parent_mp != args.child_mp
    transition = (
        "mp8-to-mp18-mass-diagnosed-v1"
        if (args.parent_mp, args.child_mp) == (8, 18)
        else "mp-edge-mass-diagnosed-v1" if mixed
        else "same-scheme-only"
    )
    parent_run = replace(
        wk82.default_config(),
        nx=36, ny=36, nz=20, run_seconds=float(seconds),
        output_interval_s=float(seconds), mp_physics=args.parent_mp,
        moist=True, moist_cq=True, km_opt=1, khdif=0.0, kvdif=0.0,
        grid_id=1, case="wk82_edge_matrix_parent")
    child_run = replace(
        parent_run,
        nx=30, ny=30, dx=parent_run.dx / 3.0,
        dy=parent_run.dy / 3.0,
        dt=float(np.float32(parent_run.dt) / np.float32(3.0)),
        mp_physics=args.child_mp, grid_id=2, nested=True,
        specified=False, open_x=False, open_y=False,
        nest_microphysics_transition=transition,
        case="wk82_edge_matrix_child")
    validate_run_config(parent_run)
    validate_run_config(child_run)
    root_dc = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=float(seconds), run=parent_run,
        time_step=int(parent_run.dt))
    child_dc = DomainConfig(
        grid_id=2, parent_id=1, i_parent_start=13, j_parent_start=13,
        parent_grid_ratio=3, parent_time_step_ratio=3,
        history_interval_s=float(seconds), run=child_run)
    exp = ExperimentConfig(
        name="wk82_microphysics_edge_matrix",
        start_time=datetime(1982, 5, 20),
        run_seconds=float(seconds),
        vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig(
            "lambert", 35.0, -97.0, 30.0, 60.0, -97.0),
        restart_interval_s=0.0, domains=(root_dc, child_dc))

    coord = make_vertical_coord(parent_run.nz)
    base = make_base_state(
        coord, lambda z: wk82.wk82_sounding(z)[0],
        p_surf=parent_run.p_surf, ztop=parent_run.ztop)
    root_state = wk82.build(parent_run, coord, base)
    model = assemble_idealized_tree(exp, root_state)
    child = model.node(2)
    init_checks = _moment_consistency(child.state, args.child_mp)
    samples = []

    def snapshot(_model, node, ticks):
        consume_history_reflectivity(node, ticks)
        if node is child:
            fields = nest_field_kinds(node.cfg.run)
            census = _field_statistics(node.state, fields)
            samples.append({
                "ticks": int(ticks),
                "elapsed_seconds": float(node.clock.elapsed_seconds),
                "nonfinite_count": sum(
                    item["nonfinite_count"] for item in census.values()),
            })

    started = time.perf_counter()
    execution = __import__(
        "gpuwm.core.model", fromlist=["execute_experiment"]
    ).execute_experiment(model, history_handler=snapshot)
    cp.cuda.Stream.null.synchronize()
    wall_seconds = time.perf_counter() - started

    target_fields = nest_field_kinds(child.cfg.run)
    final_census = _field_statistics(child.state, target_fields)
    final_checks = _moment_consistency(child.state, args.child_mp)
    final_digest, digest_fields = _state_digest(
        child.state, (*target_fields, "h_diabatic"))
    transition_receipt = dict(child.coupler.transition_receipt())
    finite = all(
        item["nonfinite_count"] == 0 for item in final_census.values())
    completed = float(child.clock.elapsed_seconds) >= float(seconds)
    gpu = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    report = {
        "schema": "gpuwm-nest-microphysics-edge-integration-v1",
        "status": "PASS" if finite and completed else "FAIL",
        "source_mp_physics": args.parent_mp,
        "target_mp_physics": args.child_mp,
        "requested_child_minutes": int(args.minutes),
        "integrated_child_seconds": float(child.clock.elapsed_seconds),
        "wall_seconds": wall_seconds,
        "gpu": {
            "name": gpu["name"].decode()
            if isinstance(gpu["name"], bytes) else str(gpu["name"]),
            "compute_capability": [
                int(gpu["major"]), int(gpu["minor"]),
            ],
        },
        "executor": {
            "steps": int(execution.steps),
            "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "transition_receipt": transition_receipt,
        "initial_moment_consistency": init_checks,
        "final_moment_consistency": final_checks,
        "moment_probe_disposition": (
            "diagnostic-only after the destination scheme has advanced; "
            "edge-kernel q>0/N>0 reconstruction is pinned independently, "
            "while this integration gates destination-scheme evaluability "
            "on completion and finite state"
        ),
        "history_samples": samples,
        "final_field_census": final_census,
        "final_state_sha256": final_digest,
        "digest_fields": digest_fields,
        "imported_repo_root": str(repo_root),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main(argv=None):
    args = _parse_args(argv)
    report = run(args)
    print(json.dumps({
        key: report[key]
        for key in (
            "status", "source_mp_physics", "target_mp_physics",
            "integrated_child_seconds", "wall_seconds",
            "final_state_sha256",
        )
    }, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

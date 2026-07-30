"""Executable N2c ratio-3 WK82 interface and high-resolution evidence.

Placement is tied to the frozen WK82 hodograph/metric conventions.  From
``HODOGRAPH_ZUV``, the 10 m wind is (-8.0998, -4.9419) m/s and the 6 km wind
is (21.4585, 3.5) m/s: the 0--6 km shear is (29.5583, 8.4419) m/s and its
right normal is (8.4419, -29.5583), toward the southeast.  The frozen WK82
builder applies its documented constant storm-motion translation to keep the
right mover near the domain centre, while its metric helpers diagnose
``right_of_shear_distance`` and document tracked split separation growing
from 41.445 to 70.329 km.  The 40-by-40 parent-km child occupies the southeast
quadrant: its west and north mass-grid interfaces lie only +0.167 km east and
-0.167 km south of the release point.  Thus the right mover crosses the
corner interfaces as the split forms and remains covered at the 7200 s valid
time (half the documented final separation is about 35 km).

Nested and uniform modes are separate controller runs.  Uniform mode covers
the full 168 km parent footprint at dx/3, then restricts the pinned valid-time
statistics to the child-coincident interior.  This avoids placing the WK82
thermal at a different local-domain centre.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import math
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import tomllib

import numpy as np

from gpuwm.core.grid import make_base_state, make_vertical_coord
from gpuwm.core.model import execute_experiment
from gpuwm.experiment import (DomainConfig, ExperimentConfig,
                              ProjectionConfig, VerticalConfig)
from gpuwm.verify.cases import wk82
from gpuwm.verify.cases.nest_ideal_common import (
    assemble_idealized_tree, consume_history_reflectivity, write_json)
from gpuwm.verify.cases.nest_ideal_r1_moist import DEFAULT_CONFIG
from gpuwm.verify.metrics import boundary_zone_blowup
from gpuwm.verify.nest_gates import gate


BOUNDARY_METRIC = "wk_r3_boundary_zone_blowup"
INTERFACE_METRIC = "wk_r3_interface_crossing"
INTERIOR_METRIC = "wk_r3_interior_vs_uniform_highres"

INTERIOR_FIELDS = ("w", "thp", "qc", "qr", "qi", "qs", "qg")
PINNED_VALID_TIME_S = float(wk82.ANALYSIS_TIMES[-1])
CROSSING_WINDOW_S = (0.0, float(wk82.ANALYSIS_TIMES[-1]))


def _raw_n2c(path=DEFAULT_CONFIG) -> dict[str, object]:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    try:
        table = raw["identity_nest"]["n2c"]
    except (KeyError, TypeError) as exc:
        raise ValueError("scaffold requires [identity_nest.n2c]") from exc
    required = {
        "parent_nx", "parent_ny", "child_nx", "child_ny", "nz", "dx",
        "dt", "ztop", "run_seconds", "history_interval_s",
        "parent_grid_ratio", "parent_time_step_ratio", "i_parent_start",
        "j_parent_start", "moist", "mp_physics", "time_step_sound",
        "w_damping", "damp_opt", "zdamp", "dampcoef", "diff_6th_opt",
        "parent_diff_6th_factor", "fine_diff_6th_factor",
    }
    if set(table) != required:
        raise ValueError(
            f"n2c scaffold keys differ: missing={sorted(required-set(table))}, "
            f"extra={sorted(set(table)-required)}")
    return table


def load_scaffold(path=DEFAULT_CONFIG) -> ExperimentConfig:
    """Load the committed ratio-3 geometry as a programmatic experiment."""
    p = _raw_n2c(path)
    ratio = int(p["parent_grid_ratio"])
    tratio = int(p["parent_time_step_ratio"])
    if (ratio, tratio) != (3, 3):
        raise ValueError("N2c requires exact grid/time ratios 3/3")
    if not bool(p["moist"]) or int(p["mp_physics"]) != 10:
        raise ValueError("N2c requires moist Morrison physics")

    history = float(p["history_interval_s"])
    root_run = replace(
        wk82.default_config(),
        nx=int(p["parent_nx"]), ny=int(p["parent_ny"]), nz=int(p["nz"]),
        dx=float(p["dx"]), dy=float(p["dx"]), dt=float(p["dt"]),
        ztop=float(p["ztop"]), run_seconds=float(p["run_seconds"]),
        output_interval_s=history, mp_physics=10,
        time_step_sound=int(p["time_step_sound"]),
        w_damping=int(p["w_damping"]), damp_opt=int(p["damp_opt"]),
        zdamp=float(p["zdamp"]), dampcoef=float(p["dampcoef"]),
        diff_6th_opt=int(p["diff_6th_opt"]),
        diff_6th_factor=float(p["parent_diff_6th_factor"]),
        case="wk82_n2c_parent", grid_id=1)
    child_dt = float(np.float32(root_run.dt) / np.float32(tratio))
    child_run = replace(
        root_run,
        nx=int(p["child_nx"]), ny=int(p["child_ny"]),
        dx=root_run.dx / ratio, dy=root_run.dy / ratio, dt=child_dt,
        open_x=False, open_y=False, specified=False, nested=True,
        diff_6th_factor=float(p["fine_diff_6th_factor"]),
        case="wk82_n2c_child", grid_id=2)
    root = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=history, run=root_run,
        time_step=int(root_run.dt))
    child = DomainConfig(
        grid_id=2, parent_id=1,
        i_parent_start=int(p["i_parent_start"]),
        j_parent_start=int(p["j_parent_start"]),
        parent_grid_ratio=ratio, parent_time_step_ratio=tratio,
        history_interval_s=history, run=child_run)
    return ExperimentConfig(
        name="wk82_n2c_ratio3", start_time=datetime(1982, 5, 20),
        run_seconds=root_run.run_seconds,
        vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig(
            "lambert", 35.0, -97.0, 30.0, 60.0, -97.0),
        restart_interval_s=0.0, domains=(root, child))


def uniform_scaffold(path=DEFAULT_CONFIG) -> ExperimentConfig:
    """Full-parent-footprint uniform dx/3 companion to the nested case."""
    nested = load_scaffold(path)
    ratio = nested.domains[1].parent_grid_ratio
    parent = nested.root.run
    history = nested.root.history_interval_s
    cfg = replace(
        parent, nx=parent.nx * ratio, ny=parent.ny * ratio,
        dx=parent.dx / ratio, dy=parent.dy / ratio,
        dt=float(np.float32(parent.dt) / np.float32(ratio)),
        diff_6th_factor=nested.domains[1].run.diff_6th_factor,
        output_interval_s=history, case="wk82_n2c_uniform_highres",
        grid_id=1)
    root = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=history, run=cfg,
        time_step=2)
    return ExperimentConfig(
        name="wk82_n2c_uniform_highres", start_time=nested.start_time,
        run_seconds=nested.run_seconds, vertical=nested.vertical,
        projection=nested.projection, restart_interval_s=0.0,
        domains=(root,))


def placement_evidence(exp=None) -> dict[str, object]:
    exp = load_scaffold() if exp is None else exp
    child = exp.domains[1]
    z = wk82.HODOGRAPH_ZUV[:, 0]
    k0 = int(np.argmin(np.abs(z - 10.0)))
    k6 = int(np.argmin(np.abs(z - 6000.0)))
    low = wk82.HODOGRAPH_ZUV[k0, 1:]
    high = wk82.HODOGRAPH_ZUV[k6, 1:]
    shear = high - low
    right_normal = np.array([shear[1], -shear[0]])
    span_x = child.run.nx / child.parent_grid_ratio
    span_y = child.run.ny / child.parent_grid_ratio
    parent = exp.root.run
    centre_i = (parent.nx + 1) / 2.0
    centre_j = (parent.ny + 1) / 2.0
    west_i = ((child.i_parent_start - 0.5)
              + 0.5 / child.parent_grid_ratio)
    north_j = ((child.j_parent_start - 0.5)
               + (child.run.ny - 0.5) / child.parent_grid_ratio)
    return {
        "wind_10m_uv_ms": low.tolist(),
        "wind_6km_uv_ms": high.tolist(),
        "shear_0_6km_uv_ms": shear.tolist(),
        "right_normal_uv_ms": right_normal.tolist(),
        "i_parent_start": child.i_parent_start,
        "j_parent_start": child.j_parent_start,
        "child_parent_span_cells": [span_x, span_y],
        "west_interface_from_release_km": (
            (west_i - centre_i) * parent.dx / 1000.0),
        "north_interface_from_release_km": (
            (north_j - centre_j) * parent.dy / 1000.0),
        "crossing_window_seconds": list(CROSSING_WINDOW_S),
        "documented_tracked_separation_m": [41445.0, 70329.0],
    }


def _build_root(exp):
    cfg = exp.root.run
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(
        coord, lambda z: wk82.wk82_sounding(z)[0],
        p_surf=cfg.p_surf, ztop=cfg.ztop)
    return wk82.build(cfg, coord, base)


def _statistics(array) -> dict[str, object]:
    value = np.asarray(array, dtype=np.float32)
    finite = np.isfinite(value)
    if value.size == 0 or not finite.all():
        return {"count": int(value.size), "finite": bool(finite.all())}
    value64 = value.astype(np.float64)
    return {
        "count": int(value.size),
        "finite": True,
        "mean": float(np.mean(value64)),
        "stddev": float(np.std(value64)),
        "rms": float(np.sqrt(np.mean(value64 * value64))),
        "minimum": float(np.min(value64)),
        "maximum": float(np.max(value64)),
        "p01": float(np.percentile(value64, 1.0)),
        "p50": float(np.percentile(value64, 50.0)),
        "p99": float(np.percentile(value64, 99.0)),
        "nonzero_fraction": float(np.count_nonzero(value) / value.size),
    }


def _host(value) -> np.ndarray:
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value, dtype=np.float32)


def _child_interior(state, field: str, width: int) -> np.ndarray:
    value = _host(getattr(state, field))
    return value[..., width:-width, width:-width]


def _uniform_child_interior(state, field: str, nested_exp,
                            width: int) -> np.ndarray:
    child = nested_exp.domains[1]
    ratio = child.parent_grid_ratio
    j0 = (child.j_parent_start - 1) * ratio
    i0 = (child.i_parent_start - 1) * ratio
    value = _host(getattr(state, field))
    colocated = value[..., j0:j0 + child.run.ny, i0:i0 + child.run.nx]
    expected = (*value.shape[:-2], child.run.ny, child.run.nx)
    if colocated.shape != expected:
        raise RuntimeError(
            f"uniform restriction {colocated.shape} != {expected}")
    return colocated[..., width:-width, width:-width]


def _rough_vram(exp) -> dict[str, object]:
    """Conservative unshared-pool estimate from the audited field manifests."""
    from gpuwm.core.preflight import estimate_domain

    by_id = {dc.grid_id: dc for dc in exp.domains}
    estimates = tuple(estimate_domain(
        dc, parent=(None if dc.parent_id == 0 else by_id[dc.parent_id]))
        for dc in exp.domains)
    resident = sum(item.resident_bytes for item in estimates)
    transient = max((item.transient_bytes for item in estimates), default=0)
    estimate = math.ceil(1.15 * (resident + transient))
    return {
        "resident_bytes": resident,
        "transient_peak_bytes": transient,
        "headroom_factor": 1.15,
        "rough_alloc_estimate_bytes": estimate,
        "rough_alloc_estimate_gib": estimate / 2**30,
        "domain_state_field_counts": [
            sum(1 for item in domain.items if item.category == "state")
            for domain in estimates
        ],
    }


def _effective_dynamics(exp) -> list[dict[str, object]]:
    """Emit the case-only 333 m stability settings with every GPU report."""
    names = (
        "grid_id", "dx", "dt", "time_step_sound", "w_damping",
        "damp_opt", "zdamp", "dampcoef", "diff_6th_opt",
        "diff_6th_factor", "emdiv", "mp_physics",
    )
    return [{name: getattr(dc.run, name) for name in names}
            for dc in exp.domains]


def run_nested(outdir: str | Path) -> dict[str, object]:
    """Run parent+child and emit boundary/interface plus pinned statistics."""
    from gpuwm.core.dycore import stability_report

    outdir = Path(outdir)
    exp = load_scaffold()
    root_state = _build_root(exp)
    from gpuwm.static.lambert import grids_from_projection_config
    grids = grids_from_projection_config(exp)
    model = assemble_idealized_tree(exp, root_state, grids=grids)
    child = model.node(2)
    gate_width = child.cfg.run.spec_bdy_width
    interface_width = gate_width + exp.blend_width
    boundary_gate = gate("N2", BOUNDARY_METRIC)
    if boundary_gate.kind != "max" or boundary_gate.threshold is None:
        raise RuntimeError("N2c boundary gate kind/threshold changed")

    series = []
    interface_series = []
    sections: dict[str, np.ndarray] = {}
    pinned_stats = None

    def snapshot(_model, node, ticks):
        nonlocal pinned_stats
        consume_history_reflectivity(node, ticks)
        if node is not child:
            return
        if child.clock.ticks != model.root.clock.ticks:
            raise RuntimeError("N2c evidence requested outside a sync instant")
        seconds = float(child.clock.elapsed_seconds)
        health = stability_report(
            child.state, child.cfg.run, boundary_width=gate_width)
        fired = boundary_zone_blowup(
            health["boundary_w_max"], health["interior_w_max"])
        metric = float(fired)
        series.append({
            "ticks": int(ticks),
            "elapsed_seconds": seconds,
            "boundary_max_abs_w_ms": float(health["boundary_w_max"]),
            "interior_max_abs_w_ms": float(health["interior_w_max"]),
            "boundary_zone_blowup": metric,
            "pass": bool(np.isfinite(metric)
                         and metric <= boundary_gate.threshold),
        })
        interface_health = stability_report(
            child.state, child.cfg.run, boundary_width=interface_width)
        interface_series.append({
            "ticks": int(ticks),
            "elapsed_seconds": seconds,
            "boundary_max_abs_w_ms": float(
                interface_health["boundary_w_max"]),
            "interior_max_abs_w_ms": float(
                interface_health["interior_w_max"]),
        })
        if CROSSING_WINDOW_S[0] <= seconds <= CROSSING_WINDOW_S[1]:
            stamp = f"t_{int(round(seconds)):08d}_child"
            child_w = _host(child.state.w)
            sections[f"{stamp}_north_interface_w"] = child_w[:, -1, :]
            sections[f"{stamp}_west_interface_w"] = child_w[:, :, 0]
        if seconds == PINNED_VALID_TIME_S:
            pinned_stats = {
                name: _statistics(_child_interior(
                    child.state, name, interface_width))
                for name in INTERIOR_FIELDS
            }

    execution = execute_experiment(model, history_handler=snapshot)
    if pinned_stats is None:
        raise RuntimeError("pinned N2c valid time was not sampled")
    outdir.mkdir(parents=True, exist_ok=True)
    section_path = outdir / "interface_cross_sections.npz"
    np.savez_compressed(section_path, **sections)
    expected_outputs = int(round(
        exp.run_seconds / exp.root.history_interval_s)) + 1
    expected_sections = int(round(
        (CROSSING_WINDOW_S[1] - CROSSING_WINDOW_S[0])
        / exp.root.history_interval_s)) * 2 + 2
    report = {
        "case": exp.name,
        "gates": {
            BOUNDARY_METRIC: dataclasses.asdict(boundary_gate),
            INTERFACE_METRIC: dataclasses.asdict(
                gate("N2", INTERFACE_METRIC)),
            INTERIOR_METRIC: dataclasses.asdict(
                gate("N2", INTERIOR_METRIC)),
        },
        "geometry": placement_evidence(exp),
        "effective_dynamics": _effective_dynamics(exp),
        "boundary_frame_width_cells": gate_width,
        "interface_diagnostic_frame_width_cells": interface_width,
        "expected_output_samples": expected_outputs,
        "boundary_zone_series": series,
        "interface_boundary_vs_interior_series": interface_series,
        "interface_cross_sections": {
            "path": section_path.name,
            "expected_arrays": expected_sections,
            "arrays": {name: list(value.shape)
                       for name, value in sections.items()},
        },
        "pinned_valid_time_seconds": PINNED_VALID_TIME_S,
        "nested_interior_statistics": pinned_stats,
        "executor": {
            "steps": int(execution.steps), "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "initializer": "parent_only_init",
        "coupler": "ConcreteNestCoupler",
        "rough_vram": _rough_vram(exp),
        "boundary_pass": bool(
            len(series) == expected_outputs
            and all(item["pass"] for item in series)),
    }
    write_json(outdir / "nested_report.json", report)
    return report


def run_uniform(outdir: str | Path) -> dict[str, object]:
    """Run the full-footprint uniform dx/3 companion and restrict statistics."""
    outdir = Path(outdir)
    exp = uniform_scaffold()
    nested_exp = load_scaffold()
    root_state = _build_root(exp)
    from gpuwm.static.lambert import grids_from_projection_config
    grids = grids_from_projection_config(exp)
    model = assemble_idealized_tree(exp, root_state, grids=grids)
    width = nested_exp.domains[1].run.spec_bdy_width + nested_exp.blend_width
    pinned_stats = None

    def snapshot(_model, node, _ticks):
        nonlocal pinned_stats
        consume_history_reflectivity(node, _ticks)
        if float(node.clock.elapsed_seconds) == PINNED_VALID_TIME_S:
            pinned_stats = {
                name: _statistics(_uniform_child_interior(
                    node.state, name, nested_exp, width))
                for name in INTERIOR_FIELDS
            }

    execution = execute_experiment(model, history_handler=snapshot)
    if pinned_stats is None:
        raise RuntimeError("uniform pinned valid time was not sampled")
    report = {
        "case": exp.name,
        "effective_dynamics": _effective_dynamics(exp),
        "gate": dataclasses.asdict(gate("N2", INTERIOR_METRIC)),
        "pinned_valid_time_seconds": PINNED_VALID_TIME_S,
        "uniform_restricted_interior_statistics": pinned_stats,
        "executor": {
            "steps": int(execution.steps), "forces": int(execution.forces),
            "feedback_calls": int(execution.feedback_calls),
        },
        "rough_vram": _rough_vram(exp),
    }
    write_json(outdir / "uniform_report.json", report)
    return report


def _verdict(value: str | None) -> bool | None:
    return None if value is None else value == "pass"


def adjudicate(outdir: str | Path, *, interface_verdict: str | None,
               interior_verdict: str | None) -> dict[str, object]:
    """Join existing GPU evidence with explicit structural verdicts."""
    outdir = Path(outdir)
    nested = json.loads((outdir / "nested_report.json").read_text("utf-8"))
    uniform = json.loads((outdir / "uniform_report.json").read_text("utf-8"))
    interface = _verdict(interface_verdict)
    interior = _verdict(interior_verdict)
    section_evidence = nested["interface_cross_sections"]
    sections_complete = (
        len(section_evidence["arrays"])
        == section_evidence["expected_arrays"])
    nested_stats = nested["nested_interior_statistics"]
    uniform_stats = uniform["uniform_restricted_interior_statistics"]
    stats_complete = (set(nested_stats) == set(uniform_stats)
                      == set(INTERIOR_FIELDS)
                      and all(nested_stats[name].get("finite") is True
                              and uniform_stats[name].get("finite") is True
                              for name in INTERIOR_FIELDS))
    statistic_differences = {
        field: {
            name: float(nested_stats[field][name]
                        - uniform_stats[field][name])
            for name in ("mean", "stddev", "rms", "minimum", "maximum",
                         "p01", "p50", "p99", "nonzero_fraction")
            if name in nested_stats[field] and name in uniform_stats[field]
        }
        for field in INTERIOR_FIELDS
    } if stats_complete else {}
    report = {
        "case": "wk82_n2c_ratio3",
        "boundary": {
            "gate": nested["gates"][BOUNDARY_METRIC],
            "series": nested["boundary_zone_series"],
            "pass": bool(nested["boundary_pass"]),
        },
        "interface_crossing": {
            "gate": nested["gates"][INTERFACE_METRIC],
            "boundary_vs_interior_series":
                nested["interface_boundary_vs_interior_series"],
            "cross_sections": section_evidence,
            "evidence_complete": sections_complete,
            "verdict": interface,
            "pass": interface is True and sections_complete,
        },
        "interior_vs_uniform_highres": {
            "gate": nested["gates"][INTERIOR_METRIC],
            "pinned_valid_time_seconds": PINNED_VALID_TIME_S,
            "nested_statistics": nested_stats,
            "uniform_statistics": uniform_stats,
            "nested_minus_uniform_statistics": statistic_differences,
            "evidence_complete": stats_complete,
            "verdict": interior,
            "pass": interior is True and stats_complete,
        },
    }
    report["pass"] = bool(
        report["boundary"]["pass"]
        and report["interface_crossing"]["pass"]
        and report["interior_vs_uniform_highres"]["pass"])
    write_json(outdir / "report.json", report)
    return report


def _release_gpu_memory() -> None:
    gc.collect()
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


def run(outdir: str | Path, *, mode: str = "both",
        interface_verdict: str | None = None,
        interior_verdict: str | None = None) -> dict[str, object]:
    if mode == "nested":
        return run_nested(outdir)
    if mode == "uniform":
        return run_uniform(outdir)
    if mode == "adjudicate":
        return adjudicate(
            outdir, interface_verdict=interface_verdict,
            interior_verdict=interior_verdict)
    if mode != "both":
        raise ValueError(f"unknown N2c mode {mode!r}")
    run_nested(outdir)
    _release_gpu_memory()
    run_uniform(outdir)
    _release_gpu_memory()
    return adjudicate(
        outdir, interface_verdict=interface_verdict,
        interior_verdict=interior_verdict)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gpuwm.verify.cases.nest_ideal_r3")
    parser.add_argument("--outdir", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("nested", "uniform", "both", "adjudicate"),
        default="both")
    parser.add_argument("--interface-verdict", choices=("pass", "fail"))
    parser.add_argument("--interior-verdict", choices=("pass", "fail"))
    args = parser.parse_args(argv)
    report = run(
        args.outdir, mode=args.mode,
        interface_verdict=args.interface_verdict,
        interior_verdict=args.interior_verdict)
    if args.mode in ("nested", "uniform"):
        return 0
    return 0 if report["pass"] else 1


__all__ = [
    "BOUNDARY_METRIC", "CROSSING_WINDOW_S", "INTERFACE_METRIC",
    "INTERIOR_FIELDS", "INTERIOR_METRIC", "PINNED_VALID_TIME_S",
    "adjudicate", "load_scaffold", "main", "placement_evidence", "run",
    "run_nested", "run_uniform", "uniform_scaffold",
]


if __name__ == "__main__":  # pragma: no cover - controller entry point
    raise SystemExit(main())

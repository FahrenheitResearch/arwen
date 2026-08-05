#!/usr/bin/env python3
"""Ask every static gate on the HRRR run route about ONE experiment config.

The observation battery composes case configs and runs them through
``tools/prepared_domain_tree_forecast.py`` over an HRRR-prepared root.
That route is guarded by four independent admission gates that live in
four modules and fire at four different costs:

===========================================  ==============================
gate                                          when it fires today
===========================================  ==============================
``gpuwm.config.validate_run_config``          config load
``gpuwm.hrrr_route_inputs``                   wizard emission
``physics_compat`` single-domain profile      root preparation (after fetch)
``gpuwm.hrrr_hierarchy_direct``               tree assembly (after prepare)
===========================================  ==============================

The expensive ones are the last two: a suite the root preparer refuses is
refused AFTER 10-15 GB of HRRR has been fetched and decoded, and a suite
the hierarchy gate refuses is refused after the root preparation has been
paid for as well.  This tool asks all four in one CPU-side pass, before
anything is fetched, and prints the refusal each one WOULD raise -- the
same code objects the route calls, never a restatement of their rules.

It also prices the run: device-memory envelope from the product's own
allocation estimator, and wall-clock/output projections scaled from the
committed 3 km anchor receipt (``evidence/grounding-3km-conus-run-report.json``),
which is the only in-tree measurement of this engine at this resolution.

Exit status is 0 when every gate admits the config and 1 when any refuses,
so this is usable as the campaign's pre-launch check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import traceback


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

RECEIPT_SCHEMA = "gpuwm-battery-route-preflight-v1"

#: The committed anchor this tool scales its projections from.  It is a
#: real forecast receipt in this tree, not a constant: 796x636x49 at
#: dx 3 km, dt 15 s, Thompson/YSU/MM5/Noah with the 4/4 radiation pair.
ANCHOR_RECEIPT = Path("evidence") / "grounding-3km-conus-run-report.json"

#: The device-memory fit published beside that anchor
#: (``evidence/grounding-3km-conus-sizing.md``): slope 0.75-0.84 KiB/cell,
#: best single value 0.80, intercept 2.9-3.4 GiB on the RTX 5090.  Carried
#: here as the INDEPENDENT cross-check on the allocation estimator, which
#: is the authority; two routes agreeing is the point.
FIT_SLOPE_KIB_PER_CELL = (0.75, 0.80, 0.84)
FIT_INTERCEPT_GIB = (2.9, 3.4)

_GIB = 1024.0 ** 3
_STATUS_ORDER = {"ADMITS": 0, "INFO": 1, "REFUSES": 2}


@dataclass(frozen=True)
class Gate:
    """One admission question and the answer the route's own code gave."""

    gate_id: str
    authority: str
    status: str
    detail: str

    def as_json(self) -> dict[str, str]:
        return {
            "gate": self.gate_id,
            "authority": self.authority,
            "status": self.status,
            "detail": self.detail,
        }


def _ask(gate_id: str, authority: str, call) -> Gate:
    """Run one gate; an exception IS its refusal, quoted verbatim."""

    try:
        detail = call()
    except Exception as error:  # noqa: BLE001 -- the refusal is the answer
        return Gate(gate_id, authority, "REFUSES",
                    f"{type(error).__name__}: {error}")
    if isinstance(detail, str) and detail:
        return Gate(gate_id, authority, "REFUSES", detail)
    return Gate(gate_id, authority, "ADMITS",
                "" if detail is None else str(detail))


def _runner_selection(exp) -> Gate:
    """Name the run-step runner BEFORE a node is booked.

    The rented-node smoke of 2026-08-04 passed every static gate, the
    fetch, the preparation and the tree build, and was then refused at
    the run step: ``gpuwm.prepared_domain_tree_forecast`` carries a
    designed two-domain floor (its own docstring: it "does not flatten
    a nest tree into the single-domain benchmark runner").  The floor
    is division of labor, not a defect, and it is not widened; what was
    missing is this gate -- the preflight naming the refusal, and the
    right runner, before anything is paid for.
    """

    if len(exp.domains) >= 2:
        return Gate(
            "run.runner_selection",
            "gpuwm.prepared_domain_tree_forecast",
            "INFO",
            f"{len(exp.domains)}-domain tree: the run step is "
            "python -m gpuwm.prepared_domain_tree_forecast")
    return Gate(
        "run.runner_selection",
        "gpuwm.prepared_domain_tree_forecast two-domain floor; "
        "tools.hrrr_single_domain_benchmark runner_capabilities",
        "INFO",
        "single-domain config: gpuwm.prepared_domain_tree_forecast REFUSES "
        "it ('prepared domain-tree runner requires at least two domains' -- "
        "a designed division of labor, not widened here).  The run step is "
        "tools/hrrr_single_domain_benchmark.py forecast mode (--io-mode "
        "history --history-interval-seconds N), restoring the root "
        "preparation's own prepared cache; run it twice for the dual-run "
        "byte screen")


def _fd_budget(exp) -> Gate:
    """File-descriptor budget for the preparation and export, with remedy.

    The 2026-08-04 battery case run's stock-WRF export died on EMFILE
    under the default soft limit of 1024 (25 forcing hours, 8 default
    workers).  The estimate below is deliberately conservative -- the
    measured failure shows each (hour x worker) slot costs several
    descriptors, not one -- and the gate is INFO because the limit is a
    property of the LAUNCHING shell, which may not be this one; the
    fail-closed export step is the enforcing guard.
    """

    hours = len(_forcing_hours(exp.run_seconds))
    workers = 8  # the route's default pipeline/export worker count
    need = hours * workers * 8 + 128
    authority = ("resource.getrlimit(RLIMIT_NOFILE); measured EMFILE at "
                 "25 h x 8 workers under soft limit 1024")
    try:
        import resource
    except ImportError:
        return Gate(
            "run.fd_budget", authority, "INFO",
            f"not measurable on this platform; the run shell needs a "
            f"file-descriptor soft limit of at least ~{need} "
            f"({hours} forcing hours x {workers} workers, conservative). "
            f"On the nodes: ulimit -n {max(need, 4096)} before step 1")
    soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < need:
        return Gate(
            "run.fd_budget", authority, "INFO",
            f"soft limit {soft} is BELOW the conservative need ~{need} "
            f"({hours} forcing hours x {workers} workers) -- the export "
            f"step will refuse loudly on EMFILE.  Remedy, in the shell "
            f"that launches the run: ulimit -n {max(need, 4096)}")
    return Gate(
        "run.fd_budget", authority, "INFO",
        f"soft limit {soft} covers the conservative need ~{need} "
        f"({hours} forcing hours x {workers} workers)")


def _forcing_hours(run_seconds: float) -> tuple[int, ...]:
    """The f00..fNN inventory a root prepared for this run would carry."""

    from gpuwm.hrrr_forecast import hrrr_forcing_end_hour

    return tuple(range(hrrr_forcing_end_hour(run_seconds) + 1))


def _profile_admission(exp) -> Gate:
    """Which shipped HRRR profile, if any, this config IS.

    The HRRR root preparation is the single-domain benchmark runner in
    ``--prepare-only`` mode (``tools/prepare_hrrr_wrf.py``), and that
    runner's ``--physics-profile`` is a closed choice list with a WSM6
    default.  Naming a profile asserts the config IS that shipped suite
    and the validator refuses on any switch drift, so "no profile
    matches" means the root cannot be prepared for this suite at all --
    which is a fact worth learning before the fetch, not after it.
    """

    from gpuwm.physics_compat import (
        SINGLE_DOMAIN_PHYSICS_PROFILES,
        validate_single_domain_physics_profile,
    )

    run = exp.domains[0].run
    admitted: list[str] = []
    drift: dict[str, str] = {}
    for profile in SINGLE_DOMAIN_PHYSICS_PROFILES:
        try:
            validate_single_domain_physics_profile(profile, config=run)
        except Exception as error:  # noqa: BLE001
            drift[profile] = str(error)
            continue
        admitted.append(profile)
    authority = "gpuwm.physics_compat.validate_single_domain_physics_profile"
    if admitted:
        return Gate("hrrr.root_preparation.profile", authority, "ADMITS",
                    "prepare with --physics-profile "
                    + ", ".join(admitted))
    closest = min(drift, key=lambda name: len(drift[name])) if drift else ""
    return Gate(
        "hrrr.root_preparation.profile", authority, "REFUSES",
        "no shipped HRRR physics profile matches this suite, so "
        "tools/prepare_hrrr_wrf.py cannot prepare a root for it "
        f"({len(SINGLE_DOMAIN_PHYSICS_PROFILES)} profiles checked; closest "
        f"is {closest}: {drift.get(closest, '')})")


def _memory_estimate(exp) -> dict[str, object]:
    """Device-memory envelope, by the product's own allocation estimator."""

    from gpuwm.core import preflight

    estimate = preflight.estimate_experiment(exp)
    # ``peak_envelope_bytes`` is the number the allocation gate compares
    # against the measured budget; ``alloc_estimate_bytes`` is the pool
    # request under it.  Both are reported because the anchor receipt
    # publishes an OBSERVED device peak, and only the envelope is
    # comparable to that.
    total = int(estimate.peak_envelope_bytes)
    cells = sum(d.run.nx * d.run.ny * d.run.nz for d in exp.domains)
    fit = {
        f"slope_{slope:g}_kib_per_cell_intercept_{intercept:g}_gib": round(
            slope * cells / (1024.0 * 1024.0) + intercept, 3)
        for slope in FIT_SLOPE_KIB_PER_CELL
        for intercept in FIT_INTERCEPT_GIB
    }
    return {
        "peak_envelope_bytes": total,
        "peak_envelope_gib": round(total / _GIB, 3),
        "alloc_estimate_bytes": int(estimate.alloc_estimate_bytes),
        "alloc_estimate_gib": round(
            int(estimate.alloc_estimate_bytes) / _GIB, 3),
        "envelope_family": str(estimate.envelope_family),
        "cells": cells,
        "independent_fit_gib": fit,
        "fit_source": "evidence/grounding-3km-conus-sizing.md",
    }


def _anchor_projection(exp, *, repository_root: Path) -> dict[str, object]:
    """Scale the committed 3 km anchor linearly in cells to this shape.

    Linear-in-cells is stated as an assumption, not a measurement: the
    anchor is one grid size, and a smaller grid keeps more of the fixed
    per-step cost.  The projection is therefore an upper-bound-flavoured
    estimate whose only job is to be superseded by this config's own
    sizing smoke.
    """

    path = repository_root / ANCHOR_RECEIPT
    if not path.is_file():
        return {"available": False, "reason": f"{ANCHOR_RECEIPT} not found"}
    anchor = json.loads(path.read_text(encoding="utf-8"))
    domain = anchor["domain"]
    anchor_cells = int(domain["nx"]) * int(domain["ny"]) * int(domain["nz"])
    anchor_sim_minutes = float(anchor["run_seconds"]) / 60.0
    anchor_rate = float(anchor["wall_seconds"]) / anchor_sim_minutes
    anchor_frames = int(anchor["gridded_output"]["frame_count"])
    anchor_bytes_per_frame = (
        int(anchor["gridded_output"]["total_bytes"]) / anchor_frames)

    cells = sum(d.run.nx * d.run.ny * d.run.nz for d in exp.domains)
    scale = cells / anchor_cells
    rate = anchor_rate * scale
    sim_minutes = float(exp.run_seconds) / 60.0
    cadence = float(exp.domains[0].history_interval_s)
    frames = int(float(exp.run_seconds) // cadence) + 1 if cadence > 0 else 0
    return {
        "available": True,
        "anchor": str(ANCHOR_RECEIPT).replace("\\", "/"),
        "anchor_cells": anchor_cells,
        "anchor_seconds_per_sim_minute": round(anchor_rate, 4),
        "anchor_bytes_per_frame": int(anchor_bytes_per_frame),
        "scaling": "linear in cells (assumption, not a measurement)",
        "cells": cells,
        "projected_seconds_per_sim_minute": round(rate, 4),
        "projected_wall_seconds": round(rate * sim_minutes, 1),
        "projected_wall_hours": round(rate * sim_minutes / 3600.0, 3),
        "projected_frames": frames,
        "projected_output_bytes": int(anchor_bytes_per_frame * scale * frames),
        "projected_output_gib": round(
            anchor_bytes_per_frame * scale * frames / _GIB, 2),
    }


def evaluate(config_path: Path, *,
             repository_root: Path = REPOSITORY_ROOT) -> dict[str, object]:
    """Every static answer the HRRR route has about this config."""

    from gpuwm import hrrr_route_inputs
    from gpuwm.config import validate_run_config
    from gpuwm.experiment import load_experiment
    from gpuwm.hrrr_hierarchy_direct import _supported_hierarchy_slice

    config_path = Path(config_path)
    exp = load_experiment(config_path)
    gates = [
        Gate("experiment.load", "gpuwm.experiment.load_experiment", "ADMITS",
             f"{len(exp.domains)} domain(s), "
             f"{float(exp.run_seconds) / 3600.0:g} h from {exp.start_time}"),
    ]
    for domain in exp.domains:
        gates.append(_ask(
            f"run_config.validate.d{domain.grid_id:02d}",
            "gpuwm.config.validate_run_config",
            lambda run=domain.run: validate_run_config(run) and None))
    gates.append(_ask(
        "hrrr.coverage", "gpuwm.hrrr_route_inputs.coverage_refusal",
        lambda: hrrr_route_inputs.coverage_refusal(exp)))
    gates.append(_ask(
        "hrrr.route_inputs.physics",
        "gpuwm.hrrr_route_inputs.validate_route_physics",
        lambda: hrrr_route_inputs.validate_route_physics(exp)))
    gates.append(_profile_admission(exp))

    hours = _forcing_hours(exp.run_seconds)

    def _slice() -> None:
        target = hrrr_route_inputs.target_domain(exp)
        _supported_hierarchy_slice(exp, target, forcing_hours=hours)

    gates.append(_ask(
        "hrrr.hierarchy.slice",
        "gpuwm.hrrr_hierarchy_direct._supported_hierarchy_slice", _slice))
    gates.append(_runner_selection(exp))
    gates.append(_fd_budget(exp))

    memory = _ask("arwen.device_memory",
                  "gpuwm.core.preflight.estimate_experiment",
                  lambda: None)
    sizing: dict[str, object]
    try:
        sizing = _memory_estimate(exp)
    except Exception as error:  # noqa: BLE001
        sizing = {"error": f"{type(error).__name__}: {error}"}
        memory = Gate(memory.gate_id, memory.authority, "REFUSES",
                      str(sizing["error"]))
    else:
        memory = Gate(
            memory.gate_id, memory.authority, "INFO",
            f"{sizing['peak_envelope_gib']} GiB peak envelope for "
            f"{sizing['cells']:,} cells "
            f"({sizing['alloc_estimate_gib']} GiB pool request)")
    gates.append(memory)

    projection = _anchor_projection(exp, repository_root=repository_root)
    if projection.get("available"):
        gates.append(Gate(
            "arwen.wall_clock_projection", str(ANCHOR_RECEIPT), "INFO",
            f"{projection['projected_seconds_per_sim_minute']} s/sim-min -> "
            f"{projection['projected_wall_hours']} h wall, "
            f"{projection['projected_output_gib']} GiB of wrfout"))
    gates.append(Gate(
        "hrrr.forcing_inventory", "gpuwm.hrrr_forecast.hrrr_forcing_end_hour",
        "INFO",
        f"a root prepared for this run seals f00..f{hours[-1]:02d} "
        f"({len(hours)} HRRR analyses)"))

    refusals = [gate for gate in gates if gate.status == "REFUSES"]
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "REFUSED" if refusals else "ADMITTED",
        "experiment_config": str(config_path).replace("\\", "/"),
        "experiment_name": exp.name,
        "domains": [
            {
                "grid_id": domain.grid_id,
                "nx": domain.run.nx, "ny": domain.run.ny, "nz": domain.run.nz,
                "dx_m": float(domain.run.dx),
                "dt_s": float(domain.run.dt),
                "mp_physics": domain.run.mp_physics,
                "bl_pbl_physics": domain.run.bl_pbl_physics,
                "sf_sfclay_physics": domain.run.sf_sfclay_physics,
                "sf_surface_physics": domain.run.sf_surface_physics,
                "cu_physics": domain.run.cu_physics,
                "ra_lw_physics": domain.run.ra_lw_physics,
                "ra_sw_physics": domain.run.ra_sw_physics,
                "ra_rrtmg_variant": domain.run.ra_rrtmg_variant,
                "history_interval_s": float(domain.history_interval_s),
            }
            for domain in exp.domains
        ],
        "run_seconds": float(exp.run_seconds),
        "gates": [gate.as_json() for gate in gates],
        "refusing_gates": [gate.gate_id for gate in refusals],
        "sizing": sizing,
        "projection": projection,
    }


def render(receipt: dict[str, object]) -> str:
    lines = [
        f"battery route preflight: {receipt['experiment_config']}",
        f"  experiment {receipt['experiment_name']}, "
        f"{receipt['run_seconds'] / 3600.0:g} h",
    ]
    for domain in receipt["domains"]:
        lines.append(
            f"  d{domain['grid_id']:02d} {domain['nx']}x{domain['ny']}"
            f"x{domain['nz']} dx={domain['dx_m']:g} dt={domain['dt_s']:g} "
            f"mp{domain['mp_physics']} pbl{domain['bl_pbl_physics']} "
            f"sfclay{domain['sf_sfclay_physics']} "
            f"lsm{domain['sf_surface_physics']} cu{domain['cu_physics']} "
            f"ra {domain['ra_lw_physics']}/{domain['ra_sw_physics']} "
            f"({domain['ra_rrtmg_variant']})")
    lines.append("")
    width = max(len(gate["gate"]) for gate in receipt["gates"])
    for gate in sorted(receipt["gates"],
                       key=lambda item: _STATUS_ORDER[item["status"]]):
        head = f"  [{gate['status']:8s}] {gate['gate']:<{width}s}"
        lines.append(head if not gate["detail"]
                     else f"{head}  {gate['detail']}")
        if gate["status"] == "REFUSES":
            lines.append(f"  {'':10s} {'':{width}s}  authority: "
                         f"{gate['authority']}")
    lines.append("")
    lines.append(f"  status: {receipt['status']}")
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--receipt", type=Path,
                        help="write the machine-readable receipt here")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = evaluate(args.experiment_config)
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 2
    if not args.quiet:
        print(render(receipt))
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    return 0 if receipt["status"] == "ADMITTED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

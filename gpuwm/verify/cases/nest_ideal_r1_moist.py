"""CPU-buildable scaffold and executable N2b moist identity-nest case.

Production experiment TOMLs correctly reject ratio-1 children.  This module
constructs the deliberately non-production identity oracle directly from the
small committed config, leaving GPU state construction/execution to the
controller while CPU tests pin its grid, clock, schedule, and Morrison field
inventory.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import tomllib

from gpuwm.config import RunConfig
from gpuwm.experiment import (DomainConfig, ExperimentConfig,
                              ProjectionConfig, VerticalConfig)


DEFAULT_CONFIG = (Path(__file__).resolve().parents[3]
                  / "configs" / "scaffolds" / "wk82_identity_r1.toml")

GATE_METRIC = "identity_nest_r1_moist_wk82"

# Full mutable moist prognostic state.  qnr/qni/qns/qng are the WRF
# Registry names; the shared view maps them to gpuwm's nr/ni/ns/ng arrays.
# h_diabatic is retained across steps and feeds the following RK solve, so it
# is state rather than a disposable microphysics diagnostic.
MOIST_PROGNOSTIC_FIELDS = (
    "u", "v", "w", "thp", "php", "mup",
    "qv", "qc", "qr", "qi", "qs", "qg",
    "qnr", "qni", "qns", "qng", "h_diabatic",
)


def load_scaffold(path: str | Path = DEFAULT_CONFIG, *, variant: str = "n2b"
                  ) -> ExperimentConfig:
    with Path(path).open("rb") as stream:
        raw = tomllib.load(stream)
    table = raw.get("identity_nest")
    if set(raw) != {"identity_nest"} or not isinstance(table, dict):
        raise ValueError("identity scaffold needs exactly [identity_nest]")
    if table.get("case") != "wk82":
        raise ValueError("identity scaffold case must be 'wk82'")
    if (table.get("parent_grid_ratio") != 1
            or table.get("parent_time_step_ratio") != 1):
        raise ValueError("N2 identity scaffold requires exact ratio 1/1")
    profiles = {name: table.get(name) for name in ("n2a", "n2b")}
    if any(not isinstance(profile, dict) for profile in profiles.values()):
        raise ValueError("identity scaffold requires [identity_nest.n2a/n2b]")
    if variant not in profiles:
        raise ValueError(
            f"identity scaffold variant must be n2a/n2b, got {variant!r}")
    profile = profiles[variant]

    dt = float(table["dt"])
    run_seconds = float(table["run_seconds"])
    history = float(table["history_interval_s"])
    common = dict(
        nx=int(table["nx"]), ny=int(table["ny"]), nz=int(table["nz"]),
        dx=float(table["dx"]), dy=float(table["dx"]),
        ztop=float(table["ztop"]), dt=dt, run_seconds=run_seconds,
        output_interval_s=history, moist=bool(profile["moist"]),
        mp_physics=int(profile["mp_physics"]),
        moist_adv_opt=(1 if profile["moist"] else 0),
        km_opt=4, c_s=0.25, diff_6th_opt=2, diff_6th_factor=0.12,
        damp_opt=3, zdamp=5000.0, dampcoef=0.2, emdiv=0.01,
        spec_bdy_width=5, spec_zone=1, relax_zone=4,
        case=f"wk82_identity_r1_{variant}")
    root_run = RunConfig(**common, grid_id=1)
    child_run = RunConfig(
        **common, grid_id=2, nested=True, specified=False)
    root = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=history, run=root_run,
        time_step=int(dt))
    child = DomainConfig(
        grid_id=2, parent_id=1, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=history, run=child_run)
    return ExperimentConfig(
        name=f"wk82_identity_r1_{variant}",
        start_time=datetime(1982, 5, 20),
        run_seconds=run_seconds,
        vertical=VerticalConfig((), 0.0, 1, 0.2),
        projection=ProjectionConfig(
            "lambert", 35.0, -97.0, 30.0, 60.0, -97.0),
        restart_interval_s=0.0, domains=(root, child))


def build_case():
    """Build the real Morrison WK82 parent and its executable experiment."""
    from gpuwm.config import validate_run_config
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.verify.cases import wk82
    from gpuwm.verify.cases.nest_ideal_common import (
        identity_halo_run, synchronized_identity_config)

    exp = synchronized_identity_config(load_scaffold(variant="n2b"))
    # Both identity domains inherit the historical WK82 no-PBL km_opt=4
    # combination.  WRF runs vertical_diffusion_2 in that branch; reject
    # before GPU state construction until its vertical stresses and surface
    # flux policy are implemented.  Low-level nest/oracle construction tests
    # may continue to inspect the scaffold without advancing it.
    for domain in exp.domains:
        validate_run_config(domain.run)

    def state(cfg):
        coord = make_vertical_coord(cfg.nz)
        base = make_base_state(
            coord, lambda z: wk82.wk82_sounding(z)[0],
            p_surf=cfg.p_surf, ztop=cfg.ztop)
        return wk82.build(cfg, coord, base)

    return exp, state(exp.root.run), state(identity_halo_run(exp))


def run(outdir: str | Path) -> dict[str, object]:
    from gpuwm.verify.cases.nest_ideal_common import (
        INACTIVE_MOIST_FIELDS, run_identity_case)

    exp, root_state, halo_state = build_case()
    return run_identity_case(
        exp=exp, root_state=root_state, halo_state=halo_state,
        metric=GATE_METRIC,
        field_names=MOIST_PROGNOSTIC_FIELDS,
        inactive_fields=INACTIVE_MOIST_FIELDS, outdir=outdir)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gpuwm.verify.cases.nest_ideal_r1_moist")
    parser.add_argument("--outdir", required=True, type=Path)
    args = parser.parse_args(argv)
    return 0 if run(args.outdir)["pass"] else 1


__all__ = [
    "DEFAULT_CONFIG", "GATE_METRIC", "MOIST_PROGNOSTIC_FIELDS",
    "build_case", "load_scaffold", "main", "run",
]


if __name__ == "__main__":  # pragma: no cover - controller entry point
    raise SystemExit(main())

"""Build a surface-clustered eta ladder and score its boundary-layer levels.

ArWen shares one vertical grid across every domain of a tree
(``gpuwm/experiment.py:193`` rejects per-domain vertical keys), so an LES
child runs its root's levels.  The shipped 49-level ladder puts only 18
half levels below 1.7 km -- an effective dz of 96.7 m in the boundary
layer (``docs/public/LES.md:263-303``).  A resolved-turbulence child needs
more than that, and the only lever is the shared ladder itself.

This tool is the missing generator.  It builds ``nz + 1`` full levels from
a geometrically stretched layer-thickness profile, converts those heights
to WRF's analytic base-state dry pressure, normalises to eta, and then
scores the result THROUGH THE SAME PATH the published receipts use, so the
count it reports is the count the model will have.

The scoring path, and why it is exact at the reference column
-------------------------------------------------------------
Half-level dry pressure in WRF v4 is
``pd[k] = c3h[k]*(ps - p_top) + c4h[k] + p_top``.  With
``c4h = (znu - c3h)*(P0 - p_top)`` (``gpuwm/core/grid.py:131``) this
collapses at ``ps = P0`` to ``pd[k] = znu[k]*(P0 - p_top) + p_top`` --
the hybrid coefficients cancel exactly.  So at the reference column the
half-level heights depend on ``hybrid_opt``/``etac`` not at all, and
``analytic_base_terrain_height`` (``grid.py:185``, WRF's own
``module_initialize_real.F:3787-3803`` inverted) turns them into metres
with no state, no sounding, and no GPU.

Verified against the published number: run with ``--certified-ladder`` and
the tool reports 18 half levels below 1700 m for the shipped 49-level
ladder, matching ``docs/public/LES.md:265-266``.

Generic by construction: no case, campaign, or configuration name appears
here or in anything it emits.

Usage
-----
    python tools/build_stretched_eta_ladder.py --nz 72 --dz0 20 \
        --dz-max 650 --bl-top 1700 --emit-toml
    python tools/build_stretched_eta_ladder.py --certified-ladder
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gpuwm.core import constants as c              # noqa: E402
from gpuwm.core.grid import (                      # noqa: E402
    analytic_base_terrain_height,
    compute_hybrid_coeffs,
)
from gpuwm.native_wrf_contract import CERTIFIED_ETA_LEVELS  # noqa: E402

#: WRF's analytic base-state lapse (``grid.py:148``, Registry ``base_lapse``).
BASE_LAPSE_K = 50.0


def analytic_base_pressure(z: np.ndarray, base_temp: float = 290.0
                           ) -> np.ndarray:
    """Forward of :func:`analytic_base_terrain_height`.

    ``p_s = P0*exp(-t00/a + sqrt((t00/a)^2 - 2*g*z/(a*Rd)))``, WRF
    ``module_initialize_real.F:3787-3803``.  Vectorised; float64.
    """

    z = np.asarray(z, dtype=np.float64)
    ratio = float(base_temp) / BASE_LAPSE_K
    radicand = ratio ** 2 - 2.0 * c.G * z / (BASE_LAPSE_K * c.RD)
    if np.any(radicand < 0.0):
        raise ValueError(
            "requested column depth exceeds the analytic base state's "
            "reach; lower --ztop or raise --p-top")
    return c.P0 * np.exp(-ratio + np.sqrt(radicand))


def stretched_thicknesses(nz: int, dz0: float, dz_max: float,
                          total: float) -> np.ndarray:
    """``nz`` layer thicknesses: geometric from ``dz0``, capped at ``dz_max``.

    The growth ratio is bisected so the thicknesses sum to ``total``
    exactly.  Capping at ``dz_max`` keeps the upper troposphere from
    running away once the near-surface spacing is set aggressively; the
    cap is what makes the profile a ramp-then-uniform rather than a pure
    geometric, which is the shape WRF's own auto-levels produce.
    """

    def build(ratio: float) -> np.ndarray:
        out = np.empty(nz, dtype=np.float64)
        dz = float(dz0)
        for k in range(nz):
            out[k] = min(dz, dz_max)
            dz *= ratio
        return out

    lo, hi = 1.0, 1.5
    if build(hi).sum() < total:
        raise ValueError(
            f"nz={nz} layers from dz0={dz0} m capped at dz_max={dz_max} m "
            f"cannot span {total:.1f} m even at ratio {hi}; raise --dz-max "
            f"or --dz0, or lower the model top")
    if build(lo).sum() > total:
        raise ValueError(
            f"nz={nz} uniform layers of dz0={dz0} m already exceed "
            f"{total:.1f} m; lower --dz0")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if build(mid).sum() < total:
            lo = mid
        else:
            hi = mid
    thick = build(0.5 * (lo + hi))
    return thick * (total / thick.sum())


def eta_from_heights(z_full: np.ndarray, p_top: float,
                     base_temp: float = 290.0) -> np.ndarray:
    """Full-level eta for a full-level height profile, endpoints pinned."""

    p_full = analytic_base_pressure(z_full, base_temp)
    eta = (p_full - p_top) / (p_full[0] - p_top)
    eta[0] = 1.0
    eta[-1] = 0.0
    return eta


def score_ladder(eta: np.ndarray, *, p_top: float, hybrid_opt: int,
                 etac: float, bl_top: float, base_temp: float = 290.0
                 ) -> dict:
    """Half-level heights and boundary-layer level count for one ladder.

    Evaluated at the reference column ``ps = P0`` through
    :func:`compute_hybrid_coeffs`, so the hybrid identity above is
    exercised rather than assumed.
    """

    eta = np.asarray(eta, dtype=np.float64)
    hy = compute_hybrid_coeffs(eta, hybrid_opt, etac, c.P0, p_top)
    pd_half = hy["c3h"] * (c.P0 - p_top) + hy["c4h"] + p_top
    z_half = np.array([analytic_base_terrain_height(float(p), base_temp)
                       for p in pd_half])
    below = int(np.count_nonzero(z_half < bl_top))
    return {
        "nz": int(eta.size - 1),
        "p_top_pa": float(p_top),
        "hybrid_opt": int(hybrid_opt),
        "etac": float(etac),
        "bl_top_m": float(bl_top),
        "levels_below_bl_top": below,
        "levels_below_2000m": int(np.count_nonzero(z_half < 2000.0)),
        "first_half_level_m": float(z_half[0]),
        "top_half_level_m": float(z_half[-1]),
        "mean_dz_below_bl_top_m": (float(bl_top / below) if below else None),
        "max_dz_below_bl_top_m": (
            float(np.diff(np.concatenate(([0.0], z_half[:below]))).max())
            if below else None),
        "max_dz_m": float(np.diff(z_half).max()),
        "half_level_heights_m": [round(float(v), 2) for v in z_half],
    }


def format_toml_array(eta: np.ndarray, per_line: int = 5,
                      indent: str = "    ") -> str:
    """The ``eta_levels = [...]`` body, in the shipped configs' layout."""

    cells = []
    for value in eta:
        if value == 1.0:
            cells.append("1.0")
        elif value == 0.0:
            cells.append("0.0")
        else:
            cells.append(f"{value:.5f}".rstrip("0"))
    lines = [indent + " ".join(f"{v}," for v in cells[i:i + per_line])
             for i in range(0, len(cells), per_line)]
    return "eta_levels = [\n" + "\n".join(lines) + "\n]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nz", type=int, default=72,
                    help="mass levels; the ladder has nz+1 entries")
    ap.add_argument("--dz0", type=float, default=20.0,
                    help="thickness of the lowest layer, metres")
    ap.add_argument("--dz-max", type=float, default=650.0,
                    help="thickness cap, metres")
    ap.add_argument("--p-top", type=float, default=10000.0,
                    help="model top pressure, Pa")
    ap.add_argument("--bl-top", type=float, default=1700.0,
                    help="boundary-layer top to count levels below, metres")
    ap.add_argument("--hybrid-opt", type=int, default=2)
    ap.add_argument("--etac", type=float, default=0.2)
    ap.add_argument("--base-temp", type=float, default=290.0)
    ap.add_argument("--require-below", type=int, default=None,
                    help="exit 1 unless at least this many half levels "
                         "fall below --bl-top")
    ap.add_argument("--certified-ladder", action="store_true",
                    help="score the shipped 49-level ladder instead of "
                         "building one (the tool's own control)")
    ap.add_argument("--emit-toml", action="store_true",
                    help="print the eta_levels TOML block")
    ap.add_argument("--json", type=Path, default=None,
                    help="write the score to this path")
    args = ap.parse_args(argv)

    if args.certified_ladder:
        eta = np.asarray(CERTIFIED_ETA_LEVELS, dtype=np.float64)
        source = "gpuwm/native_wrf_contract.py:CERTIFIED_ETA_LEVELS"
    else:
        depth = analytic_base_terrain_height(args.p_top, args.base_temp)
        thick = stretched_thicknesses(args.nz, args.dz0, args.dz_max, depth)
        z_full = np.concatenate(([0.0], np.cumsum(thick)))
        eta = eta_from_heights(z_full, args.p_top, args.base_temp)
        source = (f"stretched dz0={args.dz0} dz_max={args.dz_max} "
                  f"nz={args.nz} p_top={args.p_top}")

    if eta[0] != 1.0 or eta[-1] != 0.0:
        raise SystemExit("ladder endpoints are not 1.0 / 0.0")
    if np.any(np.diff(eta) >= 0.0):
        raise SystemExit("ladder is not strictly decreasing")

    score = score_ladder(eta, p_top=args.p_top, hybrid_opt=args.hybrid_opt,
                         etac=args.etac, bl_top=args.bl_top,
                         base_temp=args.base_temp)
    score["source"] = source
    score["dz0_m"] = None if args.certified_ladder else args.dz0
    score["dz_max_m"] = None if args.certified_ladder else args.dz_max

    print(f"source                 : {source}")
    print(f"nz                     : {score['nz']} "
          f"({score['nz'] + 1} full levels)")
    print(f"first half level       : {score['first_half_level_m']:.2f} m")
    print(f"levels below {args.bl_top:.0f} m    : "
          f"{score['levels_below_bl_top']}")
    print(f"levels below 2000 m    : {score['levels_below_2000m']}")
    if score["mean_dz_below_bl_top_m"]:
        print(f"effective dz in the BL : "
              f"{score['mean_dz_below_bl_top_m']:.2f} m")
        print(f"largest dz in the BL   : "
              f"{score['max_dz_below_bl_top_m']:.2f} m")
    print(f"largest dz in the column: {score['max_dz_m']:.2f} m")
    print(f"top half level         : {score['top_half_level_m']:.1f} m")

    if args.emit_toml:
        print()
        print(format_toml_array(eta))

    if args.json is not None:
        args.json.write_text(json.dumps(score, indent=2) + "\n",
                             encoding="utf-8")

    if args.require_below is not None \
            and score["levels_below_bl_top"] < args.require_below:
        print(f"REFUSED: {score['levels_below_bl_top']} half levels below "
              f"{args.bl_top:.0f} m, required >= {args.require_below}",
              file=sys.stderr)
        return 1
    return 0


__all__ = [
    "analytic_base_pressure", "stretched_thicknesses", "eta_from_heights",
    "score_ladder", "format_toml_array", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())

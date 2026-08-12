"""Two independent VRAM models, priced side by side, at four domain sizes.

WHY THIS FILE EXISTS
--------------------
ArWen answers the question "does this domain fit on this card?" TWICE, with
two models that were built by different people for different gates and never
compared:

* :func:`gpuwm.core.preflight.estimate_experiment` -- an ITEMIZED estimate.
  It enumerates every persistent device array from the shape formulas in
  ``state.py`` / ``physics.py`` / the scratch registry, adds the step
  transients and a 15% allocator headroom, and is the number ``gpuwm go``'s
  memory gate refuses on and ``gpuwm check --alloc`` enforces.
* :func:`tilestream.autoplan.plan` -- a FITTED model.  Four measured rungs,
  each a ``(process fixed, per-buffer fixed, bytes per cell)`` triple fitted
  by non-negative least squares over 29 measured allocations on two cards,
  inflated 6% past its worst under-prediction.  It is the number
  ``[tiles] mode = "auto"`` decides on.

``mode = "auto"`` therefore asks autoplan whether a domain fits while the
route that will run it asks preflight.  If the two disagree, ``auto`` streams
a domain preflight thinks is resident, or runs resident a domain autoplan
thinks needs streaming -- and in the second case the run dies at exactly the
allocation the mode existed to avoid.

WHAT IS COMPARED, AND WHY IT IS NOT ONE NUMBER
----------------------------------------------
The two models do not price the same thing, so three numbers are printed:

``alloc_estimate``      preflight's ENFORCED pool number: headroom x (resident
                        + shared workspace + step transients).  It is a POOL
                        number -- it does not include the CUDA context or the
                        kernels' local-memory backing store.
``peak_envelope``       preflight's DEVICE number: the alloc estimate plus the
                        non-pool intercept (context + local memory) plus an
                        unmodelled term.  This is what ``go``'s gate refuses
                        on, so it is the one a user actually meets.
``autoplan_resident``   autoplan's ``Footprint.resident_bytes``: CUDA context
                        + process fixed + one buffer of the whole domain,
                        x 1.06.  A DEVICE number by construction.

``peak_envelope`` and ``autoplan_resident`` are the comparable pair; the
alloc estimate is printed because it is the one the drift guard and the N0
gate are written against.

Run::

    python -m tilestream.ledger_ladder            # the table
    python -m tilestream.ledger_ladder --json     # machine-readable
"""

from __future__ import annotations

import dataclasses
import json
import sys
from datetime import datetime
from pathlib import Path

GIB = 1 << 30

#: The ladder the feature brief names: four square domains x 49 levels.
SIZES = (672, 1024, 2048, 4096)
NZ = 49

#: MEASURED free VRAM on the three cards this project rents, rounded down to
#: what a fresh process actually sees (nvidia-smi total minus the driver's
#: own reservation).  The budgets are computed FROM these, by each model's
#: own reserve policy, exactly as production computes them.
CARDS = (
    ("RTX 5070 12 GB", 11.9),
    ("RTX 4090 24 GB", 23.5),
    ("RTX 5090 32 GB", 31.4),
)


def _real74_experiment(n: int, *, p_top: float | None = None):
    """``configs/real74_d01.toml`` at ``n x n x 49``, as an experiment.

    The full(real74) physics rung as a USER configures it -- RRTMGP 4/4,
    Kain-Fritsch, YSU, Noah, Thompson -- not a synthetic approximation of it.
    Only ``nx``/``ny`` are replaced; every physics selector, the vertical
    grid and the specified lateral boundaries are the config's own.

    ``experiment_from_run_config`` wraps a scalar RunConfig with
    ``p_top = 0.0`` because a scalar config does not carry one, and the
    RRTMGP workspace is sized against p_top.  ``p_top=`` overrides it so the
    sensitivity of the comparison to that wrapper artefact can be shown
    rather than assumed away.
    """
    from gpuwm.config import load_config
    from gpuwm.experiment import experiment_from_run_config

    root = Path(__file__).resolve().parents[1]
    cfg = dataclasses.replace(
        load_config(root / "configs" / "real74_d01.toml"), nx=n, ny=n)
    exp = experiment_from_run_config(cfg, datetime(1974, 4, 3, 12))
    if p_top is not None:
        exp = dataclasses.replace(
            exp, vertical=dataclasses.replace(exp.vertical, p_top=p_top))
    return cfg, exp


def _harness_full(n: int):
    """The same rung as autoplan's own ``_config_for_rung(n, n, 49, "full")``.

    Printed alongside the real74 config because autoplan's ``full`` footprint
    was MEASURED on this one.  If the two configs price differently, the
    disagreement is about the configuration and not about the models.
    """
    from datetime import datetime as _dt

    from gpuwm.experiment import experiment_from_run_config
    from tilestream import autoplan as A

    cfg = A._config_for_rung(n, n, NZ, "full")
    return cfg, experiment_from_run_config(cfg, _dt(1974, 4, 3, 12))


def price(n: int, *, real74: bool = True, p_top: float | None = None) -> dict:
    """Both models' numbers for one rung of the ladder."""
    from gpuwm.core import preflight as pf
    from tilestream import autoplan as A

    cfg, exp = (_real74_experiment(n, p_top=p_top) if real74
                else _harness_full(n))
    est = pf.estimate_experiment(exp)
    fp = A.footprint_for(cfg)
    cells = int(cfg.nx) * int(cfg.ny) * int(cfg.nz)
    return {
        "n": n,
        "cells": cells,
        "rung": A.rung_of(cfg),
        "footprint_rung": fp.rung,
        "alloc_estimate": int(est.alloc_estimate_bytes),
        "peak_envelope": int(est.peak_envelope_bytes),
        "non_pool": int(est.envelope_intercept_bytes),
        "autoplan_resident": int(fp.resident_bytes(cells)),
        "autoplan_store": int(fp.store_bytes(cells)),
        "estimate": est,
        "exp": exp,
        "cfg": cfg,
    }


def verdicts(row: dict, free_gib: float) -> dict:
    """What each model would ACTUALLY say on a card with ``free_gib`` free.

    Each model applies its OWN reserve policy to the same measured free
    VRAM, because that is what production does -- ``go``'s gate calls
    ``ReservePolicy.n0_alloc`` and ``Machine.detect`` applies an 8% headroom.
    Both are then also compared against ONE common budget, so that a
    disagreement can be attributed to the models rather than to the two
    reserve policies.
    """
    from gpuwm.core import preflight as pf

    free = int(free_gib * GIB)
    est = row["estimate"]
    reserve = pf.ReservePolicy.n0_alloc(
        row["exp"], estimate_bytes=est.alloc_estimate_bytes)
    pf_budget = reserve.budget_bytes(free)
    ap_budget = int(free * (1.0 - 0.08))          # autoplan VRAM_HEADROOM
    return {
        "free_bytes": free,
        "pf_budget": pf_budget,
        "ap_budget": ap_budget,
        # go_cli.memory_gate: refuse when the envelope exceeds the whole
        # card, warn when it exceeds the reserved budget.
        "pf_refuse": est.peak_envelope_bytes > free,
        "pf_warn": est.peak_envelope_bytes > pf_budget,
        # The N0 leg the ledger is written against.
        "pf_alloc_fits": est.alloc_estimate_bytes <= pf_budget,
        # autoplan: resident or tiled.
        "ap_resident": row["autoplan_resident"] <= ap_budget,
        # Same budget, both models, so the reserve policies cannot be
        # blamed for a disagreement.
        "common_budget": ap_budget,
        "pf_fits_common": est.peak_envelope_bytes <= ap_budget,
        "ap_fits_common": row["autoplan_resident"] <= ap_budget,
    }


def _g(x: int | float) -> str:
    return f"{x / GIB:8.3f}"


def band(step: int = 16) -> None:
    """The domain sizes where ``mode = "auto"`` answers the wrong question.

    ``auto`` streams only when AUTOPLAN says the domain will not fit
    resident.  The route that then runs it prices the same domain with
    PREFLIGHT.  Between the two models there is a band of sizes where
    autoplan says "resident is fine", so streaming never fires, and
    preflight then refuses the resident run -- or warns and lets it proceed
    into an allocation preflight already said would not fit.

    The band is walked at ``step``-cell granularity per card and its two
    edges are printed with the numbers on both sides, because a band nobody
    can name a size in is not a finding.
    """
    print()
    print("=== the band where 'auto' does not stream and preflight refuses "
          "anyway ===")
    print("  walked in steps of "
          f"{step} cells on n, real74 physics, nz={NZ}")
    print(f"{'card':>16} {'n':>6} {'ap.resident':>11} {'ap.budget':>10} "
          f"{'pf.envelope':>11} {'free':>8}  {'auto streams?':>13} "
          f"{'go refuses?':>11}")
    for name, free_gib in CARDS:
        free = int(free_gib * GIB)
        ap_budget = int(free * 0.92)
        first = last = None
        rows = {}
        n = step
        while n <= 4096:
            row = price(n, real74=True)
            streams = row["autoplan_resident"] > ap_budget
            refuses = row["peak_envelope"] > free
            rows[n] = (row, streams, refuses)
            if (not streams) and refuses:
                first = n if first is None else first
                last = n
            elif first is not None and streams:
                break
            n += step
        if first is None:
            print(f"{name:>16}  -- no band: the two models never straddle a "
                  f"size on this card")
            continue
        for n in (first - step, first, last, last + step):
            if n not in rows:
                continue
            row, streams, refuses = rows[n]
            print(f"{name:>16} {n:>6} {_g(row['autoplan_resident'])} "
                  f"{_g(ap_budget)} {_g(row['peak_envelope'])} "
                  f"{_g(free)}  {str(streams):>13} {str(refuses):>11}")
        print(f"{'':>16} band = n in [{first}, {last}] "
              f"({first}^2 .. {last}^2 x {NZ})")


def main(argv: list[str]) -> int:
    as_json = "--json" in argv
    rows = []
    for label, builder in (("real74 (configs/real74_d01.toml, nx=ny=n)", True),
                           ("harness full rung (autoplan's own)", False)):
        print()
        print(f"=== {label} ===")
        print(f"{'n':>6} {'Mcell':>8} {'rung':>18} "
              f"{'pf.alloc':>9} {'pf.envelope':>11} {'autoplan':>9} "
              f"{'ap/pf.env':>10} {'ap/pf.alloc':>12}")
        for n in SIZES:
            row = price(n, real74=builder)
            r_env = row["autoplan_resident"] / row["peak_envelope"]
            r_all = row["autoplan_resident"] / row["alloc_estimate"]
            print(f"{n:>6} {row['cells'] / 1e6:>8.2f} {row['rung']:>18} "
                  f"{_g(row['alloc_estimate'])} {_g(row['peak_envelope']):>11} "
                  f"{_g(row['autoplan_resident'])} "
                  f"{r_env:>10.3f} {r_all:>12.3f}")
            row["ratio_envelope"] = r_env
            row["ratio_alloc"] = r_all
            row["family"] = label
            rows.append(row)

    print()
    print("=== fit/no-fit VERDICT, per card ===")
    print("  pf.refuse  = go_cli.memory_gate hard refusal (envelope > free)")
    print("  pf.warn    = envelope > preflight's reserved budget")
    print("  ap.tiles   = autoplan says the domain does NOT fit resident")
    print("  DISAGREE   = the two models differ on the SAME budget")
    disagreements = 0
    for label in dict.fromkeys(r["family"] for r in rows):
        print(f"\n--- {label} ---")
        print(f"{'card':>16} {'n':>6} {'pf.budget':>10} {'ap.budget':>10} "
              f"{'pf.refuse':>10} {'pf.warn':>8} {'ap.tiles':>9} "
              f"{'same budget: pf-fits ap-fits':>30}")
        for name, free_gib in CARDS:
            for row in [r for r in rows if r["family"] == label]:
                v = verdicts(row, free_gib)
                agree = v["pf_fits_common"] == v["ap_fits_common"]
                if not agree:
                    disagreements += 1
                print(f"{name:>16} {row['n']:>6} {_g(v['pf_budget'])} "
                      f"{_g(v['ap_budget'])} "
                      f"{str(v['pf_refuse']):>10} {str(v['pf_warn']):>8} "
                      f"{str(not v['ap_resident']):>9} "
                      f"{str(v['pf_fits_common']):>16}"
                      f"{str(v['ap_fits_common']):>9}"
                      f"{'  DISAGREE' if not agree else ''}")

    print(f"\nverdict disagreements on a common budget: {disagreements}")
    band()

    if as_json:
        print(json.dumps([
            {k: v for k, v in r.items()
             if k not in ("estimate", "exp", "cfg")} for r in rows], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

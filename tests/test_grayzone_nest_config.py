"""The gray-zone parent chain (P4): config-pair identity pins.

`configs/les_nest_250m_grayzone.toml` is the shipped nested-LES tree with
its 750 m parent moved from YSU to Shin-Hong.  The claim its doc makes --
"any difference between the two runs is the parent closure and nothing
else" -- is only worth making if it is pinned, so this file diffs the two
RESOLVED experiment trees field by field and requires the delta to be
exactly {d02: bl_pbl_physics 1 -> 11}.  All CPU.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from gpuwm.experiment import load_experiment

REPO = Path(__file__).resolve().parents[1]
SHIPPED = REPO / "configs" / "les_nest_250m_km3.toml"
GRAYZONE = REPO / "configs" / "les_nest_250m_grayzone.toml"

#: Domain-level (non-RunConfig) fields that define the tree topology and
#: cadence.  Compared explicitly so a geometry edit cannot hide behind
#: the RunConfig diff (`specified`/`nested` live on the RunConfig and are
#: covered by the field-for-field diff below).
_DOMAIN_FIELDS = (
    "grid_id", "parent_id", "i_parent_start", "j_parent_start",
    "parent_grid_ratio", "parent_time_step_ratio", "history_interval_s",
    "time_step", "start_time",
)


def test_grayzone_tree_differs_from_shipped_by_the_one_parent_row():
    shipped = load_experiment(SHIPPED)
    grayzone = load_experiment(GRAYZONE)
    assert len(shipped.domains) == len(grayzone.domains) == 3
    deltas = {}
    for a, b in zip(shipped.domains, grayzone.domains):
        for field in _DOMAIN_FIELDS:
            assert getattr(a, field) == getattr(b, field), field
        run_a, run_b = dataclasses.asdict(a.run), dataclasses.asdict(b.run)
        assert run_a.keys() == run_b.keys()
        for key, value in run_a.items():
            if value != run_b[key]:
                deltas[(a.grid_id, key)] = (value, run_b[key])
    assert deltas == {(2, "bl_pbl_physics"): (1, 11)}


def test_grayzone_tree_selects_the_ladder_scored_configuration():
    """The gray-zone parent runs the exact reading the ladder scored:
    Shin-Hong vertical transport with horizontal Smagorinsky (km_opt=4);
    the coarse parent stays YSU by default; the child stays PBL-off on
    the LES closure with the shipped block's rows.

    ``mix_isotropic`` reads 1 and read 0 until 2026-08-09.  At 0 the
    250 m child sat at ``mix_upper_bound*(dz_max/dx)^2 = 0.702``, 2.8x
    the explicit horizontal diffusion limit, on the same criterion whose
    4.23 aborted a 100 m tornado child; both this file and the shipped
    tree moved to the isotropic length together, which is what keeps the
    one-row claim above true.  The P4 campaign's numbers belong to the
    archived bytes at ``configs/frozen/les_nest_250m_grayzone.toml``.
    """
    exp = load_experiment(GRAYZONE)
    assert [d.run.bl_pbl_physics for d in exp.domains] == [1, 11, 0]
    assert [d.run.km_opt for d in exp.domains] == [4, 4, 3]
    assert [d.run.dx for d in exp.domains] == [3000.0, 750.0, 250.0]
    child = exp.domains[2].run
    assert (child.c_s, child.mix_isotropic, child.mix_upper_bound,
            child.isfflx) == (0.25, 1, 0.1, 1)

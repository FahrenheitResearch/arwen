"""X12: the WRF-input door streams when only the card refused it resident.

The deterministic leg of a 5 km CONUS DA cycle runs ``gpuwm run --wrfinput``
(1080x636x49, HRRR recipe + GF).  On a 32 GB card it was refused at its
resident envelope with a remedy -- "[tiles] mode = 'auto' (or --tiles auto)"
-- that names settings this door does not have: its experiment comes from
``namelist.input`` and it takes no --tiles.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm.core.streaming import StreamingOptions
from gpuwm.ingest.memory_refusal import ResidentMemoryRefused
from gpuwm.prepared_domain_tree_forecast import _external_door_streams


def _exp():
    from gpuwm.experiment import load_experiment

    return load_experiment(
        Path(__file__).parents[1] / "configs" / "hrrr_prs_3km_demo.toml")


def _refusal():
    # The X12 terms: 5 km CONUS 1080x636x49, HRRR recipe + GF.
    terms = {"model state": int(7.59 * 2**30), "physics": int(7.10 * 2**30),
             "lateral boundary tables": int(0.92 * 2**30),
             "workspaces, step transients, CUDA context and allocator headroom":
                 int(29.61 * 2**30)}
    return ResidentMemoryRefused("refused", need_bytes=sum(terms.values()),
                                 free_bytes=int(30.46 * 2**30), terms=terms)


def test_an_external_door_the_card_cannot_hold_streams(capsys):
    exp = _exp()
    assert not exp.tiles.enabled
    streamed = _external_door_streams(exp, SimpleNamespace(), _refusal())
    assert streamed.tiles.mode == "auto"
    # The tiles get what the restored domain leaves on the card.
    held = int(7.59 * 2**30) + int(7.10 * 2**30) + int(0.92 * 2**30)
    assert streamed.tiles.vram_budget_bytes == int(30.46 * 2**30) - held
    assert streamed.domains == exp.domains
    said = capsys.readouterr().out
    assert "envelope against 30.46 GiB free" in said
    assert "15.61 GiB held" in said
    assert "streams" in said


@pytest.mark.parametrize("change", ["prepared-door", "tree-table", "domain-table",
                                    "unread-card"])
def test_the_refusal_stands_where_a_mode_can_be_named(change):
    exp, initialization = _exp(), SimpleNamespace()
    if change == "prepared-door":
        initialization = None
    elif change == "tree-table":
        exp = replace(exp, tiles=StreamingOptions(mode="auto", store="host"))
    elif change == "unread-card":
        refusal = ResidentMemoryRefused("refused", need_bytes=1)
        assert _external_door_streams(exp, initialization, refusal) is None
        return
    else:
        exp = replace(exp, domains=tuple(
            replace(dc, tiles=StreamingOptions(mode="off"))
            for dc in exp.domains))
    assert _external_door_streams(exp, initialization, _refusal()) is None


def test_the_tiles_are_planned_beside_what_the_analysis_door_holds(tmp_path, capsys):
    """tiled-leg-oom: the analysis door's IAU forcing and dry-mass increment are
    allocated after the plan, so the plan must leave them room.

    camp-us-20230710-r n5q15 10:00 mem003 at 13.10 GiB free planned 147x146 x12
    tiles in 9.30 GiB (13.10 less 3.80 held) and ran out of card memory on its
    first step beside the door's 0.71 GiB (DA-CYCLE-SPEED-2026-10-27 20:16:02Z).
    The split's shapes are that leg's: 588x438x49, at once MU and PH, spread 13
    fields."""
    import numpy as np

    from gpuwm.da import analysis_door as door

    nz, ny, nx = 49, 438, 588
    mass, stag_z = (nz, ny, nx), (nz + 1, ny, nx)
    shapes = {"U": (nz, ny, nx + 1), "V": (nz, ny + 1, nx), "W": stag_z, "PH": stag_z,
              **{name: mass for name in ("T", "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW",
                                         "QGRAUP", "QNICE", "QNRAIN")}}
    split = tmp_path / door.SPLIT
    door._write_split(split, {"MU": np.zeros((ny, nx)), "PH": np.zeros(stag_z)},
                      {name: np.zeros(shape) for name, shape in shapes.items()})
    plan = {"start": "cycling-restart", "bundle": {"grid": {"nx": nx, "ny": ny}},
            "iau": {"on": True, "mode": "3d", "window_seconds": 600.0}, "record": {},
            "work": str(tmp_path), "forcing": {"path": str(split)}}
    initialization = door.AnalysisInitialization(SimpleNamespace(lateral_boundaries=None), plan,
                                                 outdir=tmp_path)
    terms = initialization.device_terms()
    spread = 4 * sum(int(np.prod(shape)) for shape in shapes.values())
    assert terms["IAU forcing (float32)"] == spread
    assert terms["dry-mass increment (float64)"] == 8 * (ny * nx + int(np.prod(stag_z)))
    assert terms["balance probe planes"] == 12 * ny * nx
    assert 0.70 * 2**30 < sum(terms.values()) < 0.72 * 2**30

    # The failing leg's own refusal: 13.10 GiB free, 3.80 GiB held by the
    # restored state, physics and boundary tables.
    refusal_terms = {"model state": int(2.00 * 2**30), "physics": int(1.50 * 2**30),
                     "lateral boundary tables": int(0.30 * 2**30),
                     "workspaces, step transients, CUDA context and allocator headroom":
                         int(9.65 * 2**30)}
    refusal = ResidentMemoryRefused("refused", need_bytes=sum(refusal_terms.values()),
                                    free_bytes=int(13.10 * 2**30), terms=refusal_terms)
    streamed = _external_door_streams(_exp(), initialization, refusal)
    held = sum(refusal_terms[name] for name in ("model state", "physics", "lateral boundary tables"))
    assert streamed.tiles.vram_budget_bytes == int(13.10 * 2**30) - held - sum(terms.values())
    said = capsys.readouterr().out
    assert "IAU forcing (float32) 0.61 GiB" in said
    assert "the door holds beside it" in said

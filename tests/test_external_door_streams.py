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

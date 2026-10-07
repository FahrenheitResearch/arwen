"""AQ is opt-in but reachable: the documented switch reaches history through the doors users run.

The breakage this prevents: the smoke and air-quality program shipped with
no link from the README, the configuration guide or the starter, so the only
way to turn it on was to already know the key names. A doc example that
drifts from what the experiment door accepts, or a fitted starter that drops
the chem block, fails here before a user meets it.
"""
from __future__ import annotations

import re
import tomllib
from argparse import Namespace
from pathlib import Path

import pytest

from gpuwm import starter_template as st
from gpuwm.experiment import load_experiment
from gpuwm.io.history_layout import produced_history_shapes

ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "configs" / "starters" / "regional-gfs-rrtmgp.toml"
CHEM_HISTORY = {"smoke", "PM2_5_DRY", "PM10", "SMOKE_SFC", "SMOKE_COLUMN",
                "DUST_SFC", "AOD5502D"}


def documented_block() -> dict:
    """The worked example in CONFIGURATION.md, parsed exactly as written."""
    text = (ROOT / "docs" / "public" / "CONFIGURATION.md").read_text(encoding="utf-8")
    section = text.split("## Smoke, dust and air quality", 1)[1]
    body = re.search(r"```toml\n(.*?)```", section, re.S).group(1)
    return tomllib.loads(body)["shared"]


def starter_block() -> dict:
    """The commented block at the end of the starter's [shared], uncommented."""
    lines = [line[2:] for line in STARTER.read_text(encoding="utf-8").splitlines()
             if re.match(r"# (chem_|dust_opt|seas_opt|wetscav_onoff)", line)]
    return tomllib.loads("\n".join(lines))


def test_the_docs_and_the_starter_name_the_same_switch():
    assert documented_block() == starter_block()
    assert "SMOKE-AND-AIR-QUALITY.md" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert "SMOKE-AND-AIR-QUALITY.md" in STARTER.read_text(encoding="utf-8")


def _fitted(tmp_path, extra: dict | None) -> Path:
    raw = tomllib.loads(STARTER.read_text(encoding="utf-8"))
    if extra:
        raw["shared"].update(extra)
    path = tmp_path / "starter.toml"
    path.write_text(st.render_tables(raw), encoding="utf-8")
    out = tmp_path / "fit" / "fitted.toml"
    args = Namespace(template=path, out=out, point="40,-100", polygon=None,
                     buffer_km=None, source=None, card=None, vram_gib=16,
                     start_time="2026-09-05T00Z", hours=None, write=True)
    assert st.fit_main(args) == 0
    return out


def test_the_shipped_starter_stays_off(tmp_path):
    run = load_experiment(_fitted(tmp_path, None)).domains[0].run
    assert not run.chem_sets
    assert not CHEM_HISTORY & set(produced_history_shapes(run))


def test_the_documented_switch_reaches_history_through_fit_and_the_experiment_door(tmp_path):
    out = _fitted(tmp_path, documented_block())
    fitted = tomllib.loads(out.read_text(encoding="utf-8"))
    for key, value in documented_block().items():
        assert fitted["shared"][key] == value, key
    run = load_experiment(out).domains[0].run
    assert run.chem_sets == "gocart_primary,smoke"
    assert run.chem_sources == "rave-3km"
    shapes = produced_history_shapes(run)
    missing = CHEM_HISTORY - set(shapes)
    assert not missing, missing
    assert shapes["SMOKE_SFC"] == (run.ny, run.nx)


def test_the_cumulus_requirement_named_in_the_example_is_real(tmp_path):
    block = dict(documented_block())
    block.pop("chem_conv_tr")
    with pytest.raises(Exception, match="chem_conv_tr"):
        load_experiment(_fitted(tmp_path, block))

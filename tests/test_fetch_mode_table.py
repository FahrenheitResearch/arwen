"""The byte transport (``--mode``) lives in the experiment's ``[fetch]`` table.

Reported by the Updraft storm-day factory (2026-10-27): a source folder
fetched with ``gpuwm fetch --mode idx-subset`` for an aerosol-aware mp=28
HRRR recipe was read by ``gpuwm go --data-dir``, whose preparation needs
the analyzed aerosol records.  Every wrfnat subset was moved aside with a
bare "moved rejected file aside" and downloaded again as a WHOLE file
(410 MB subset -> 670-715 MB each, about 26 GB per 24 h window instead of
about 13 GB), because the table refused ``mode`` as an unknown key and the
run's own fetch could only take the default.  The table now takes ``mode``
with the values the flag takes, ``gpuwm go`` carries it to its fetch, and
the quarantine line names what the subset lacked and the flag that fixes it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from gpuwm import fetch, go_cli
from gpuwm.experiment import load_experiment

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEMO = "hrrr_native_quick_demo"
OUT_LINE = f'out = "data/{DEMO}"'


def _hrrr_config(tmp_path: Path, mode: str) -> Path:
    for companion in CONFIGS.glob(f"{DEMO}.*"):
        shutil.copy(companion, tmp_path / companion.name)
    config = tmp_path / f"{DEMO}.toml"
    text = config.read_text(encoding="utf-8")
    assert text.count(OUT_LINE) == 1
    config.write_text(text.replace(OUT_LINE, f'{OUT_LINE}\nmode = "{mode}"'), encoding="utf-8")
    return config


@pytest.mark.parametrize("mode", fetch.FETCH_MODES)
def test_the_table_takes_the_modes_the_flag_takes(mode):
    fetch.validate_fetch_hints(
        {"source": "hrrr", "cycle": "2026-08-20T00", "hours": 1, "mode": mode},
        source="case.toml")


def test_an_unknown_mode_is_refused_by_name():
    with pytest.raises(ValueError, match="mode = 'subset' in \\[fetch\\] of case.toml is not a fetch mode"):
        fetch.validate_fetch_hints(
            {"source": "hrrr", "cycle": "2026-08-20T00", "hours": 1, "mode": "subset"},
            source="case.toml")


def test_an_experiment_naming_a_mode_loads(tmp_path):
    # Refused on the head: unknown key(s) ['mode'] in [fetch].
    load_experiment(_hrrr_config(tmp_path, "idx-subset"))


def test_go_carries_the_mode_to_its_fetch():
    plan = {"source": "hrrr", "cycle": "2026-08-20T00", "hours": 3,
            "area": "34.65,-107.62,47.22,-91.38", "data": "src", "fetch_mode": "idx-subset"}
    command = go_cli.fetch_command(plan)
    assert command[command.index("--mode") + 1] == "idx-subset"
    plan.pop("fetch_mode")
    assert "--mode" not in go_cli.fetch_command(plan)

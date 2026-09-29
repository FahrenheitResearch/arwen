"""The card memory a fit priced travels as a record, on every route New forecast drives.

New forecast showed how close a draft sits to its card ("needs 4.9 of 14.5 GiB") by reading a line the
engine printed.  Only ``gpuwm check`` prints that line, and the wizard runs the check only when the run's
inputs are already on disk.  A GFS draft has none to wait for; an ERA5 draft's forcing is fetched when the
run starts, so its check is deferred, the line never came, and the page showed the card name alone for
every ERA5 draft, at 80 or 100 levels as at 49.

The wizard prices every draft before it writes the configuration, so that is where the figure is taken:
``gpuwm run-plan --resolve`` answers with a ``memory`` record, in bytes, and the page reads the record.

A draft too big for its card was the same defect on the refusal side.  The page read how big it was from
the refusal's words, which say ``at X GiB peak envelope`` for a source whose preprocessing is priced (ERA5,
GFS) and ``needs X GiB peak envelope ... is NOT PRICED`` for one whose preprocessing is not (HRRR, RAP,
ECMWF open data).  Only the first was read, so a too-big HRRR draft said the engine could not fit the box
and to read the server's log.  The refusal now carries the same two figures as a document on the machine
channel, and the page reads the document.

These tests hold the records to the figures the wizard and the check print, on every route, and hold the
page's reader to the records.  No network and no card: a declared card is priced on the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import pytest

from gpuwm.gui.api import Api, memory_fit, region_polygon
from gpuwm.gui.files import write_json
from gpuwm.gui.jobs import Runner
from gpuwm.gui.server import build_server, serve_in_thread
from gpuwm.runplan import PLAN_SCHEMA, generate_intent_config, load_plan, resolve_plan
from test_gui_server import request

GIB = float(1 << 30)

#: The wizard's one-line verdict, printed on every route (gpuwm/domain_wizard.py, sizing_summary).
_SIZING = re.compile(r"peak envelope ([\d.]+) GiB of a ([\d.]+) GiB budget")
#: The check's verdict, printed only where the check runs (gpuwm/core/preflight.py, check_main).
_CHECKED = re.compile(r"BINDING PHASE: \S+ needs ([\d.]+) GiB; whole-process budget ([\d.]+) GiB")

#: The wizard's too-big sentence, in both of its wordings: the peak envelope comes first in each, the budget
#: after EXCEEDS (gpuwm/core/preflight.py, the phase estimate's verdict).
_REFUSED = re.compile(r"([\d.]+) GiB peak envelope.*?EXCEEDS the ([\d.]+) GiB budget", re.S)

#: The ERA5 draft that showed no memory: 600 by 600 km at 3 km with 80 levels, sized for a 16 GB card.
_DRAFT = {"name": "", "lat": 35.3, "lon": -97.6, "width_km": 600.0, "height_km": 600.0, "hours": 6,
          "dx_km": 3.0, "nz": 80, "card": "16gb", "profile": None, "products": None}
#: A draft too big for an 8 GB card: 2000 by 2000 km at 1 km with 100 levels, over 340 GiB on every source.
_TOO_BIG = {"width_km": 2000.0, "height_km": 2000.0, "dx_km": 1.0, "nz": 100, "card": "8gb"}


def _gui_plan(tmp_path: Path, *, source: str, route: str, cycle: str, **changes) -> Path:
    """The plan New forecast writes for a fit: its own plan document and the box it draws."""

    draft = {**_DRAFT, **changes, "source": source, "route": route, "cycle": cycle}
    folder = tmp_path / source
    folder.mkdir()
    write_json(folder / "region.geojson",
               region_polygon(draft["lat"], draft["lon"], draft["width_km"], draft["height_km"]))
    path = folder / "plan.json"
    write_json(path, Api.plan_document(draft, folder))
    return path


def _gib(value: int) -> float:
    return round(value / GIB, 2)


def test_an_era5_draft_is_priced_as_a_record_though_its_check_waits_for_the_download(tmp_path, capsys):
    resolution, _exp, data = resolve_plan(
        load_plan(_gui_plan(tmp_path, source="era5", route="experiment", cycle="1999-05-03T18")),
        require_inputs=False)
    printed = capsys.readouterr().out
    # The route that broke: it reads forcing files the run fetches when it starts, so the wizard defers the
    # check until they are on disk and no check line was printed to read.
    assert data is not None and data.forcing
    sizing = _SIZING.search(printed)
    assert sizing, printed[-2000:]

    record = resolution["memory"]
    assert record is not None, "the ERA5 fit carried no memory record"
    assert (_gib(record["peak_envelope_bytes"]), _gib(record["budget_bytes"])) == (
        float(sizing.group(1)), float(sizing.group(2)))
    assert record["sizing_basis"] == "declared-capacity" and record["vram_gib"] == 16
    assert 0 < record["peak_envelope_bytes"] <= record["budget_bytes"]

    # What Review and the levels step show, read from the record.
    memory = memory_fit(resolution)
    assert memory == {"need_gib": float(sizing.group(1)), "budget_gib": float(sizing.group(2)), "fits": True}
    assert memory["budget_gib"] == 14.54


def test_a_gfs_draft_record_is_the_figure_its_check_prints(tmp_path, capsys):
    resolution, _exp, data = resolve_plan(
        load_plan(_gui_plan(tmp_path, source="gfs", route="prepared", cycle="2024-05-03T12")),
        require_inputs=False)
    printed = capsys.readouterr().out
    assert data is None
    checked = _CHECKED.search(printed)
    assert checked, printed[-2000:]

    # The figure the page read off this line before is the figure the record now carries.
    record = resolution["memory"]
    assert (_gib(record["peak_envelope_bytes"]), _gib(record["budget_bytes"])) == (
        float(checked.group(1)), float(checked.group(2)))
    assert memory_fit(resolution) == {"need_gib": float(checked.group(1)),
                                      "budget_gib": float(checked.group(2)), "fits": True}


def test_resolve_prints_the_record_on_its_machine_channel(tmp_path, capsys):
    from gpuwm.cli import build_parser
    from gpuwm.runplan import run_plan_main

    plan = _gui_plan(tmp_path, source="era5", route="experiment", cycle="1999-05-03T18")
    assert run_plan_main(build_parser().parse_args(["run-plan", str(plan), "--resolve"])) == 0
    document = json.loads(capsys.readouterr().out)
    assert set(document["memory"]) >= {"peak_envelope_bytes", "budget_bytes", "binding_phase", "sizing_basis"}
    assert memory_fit(document)["fits"] is True


def test_a_plan_naming_its_own_config_has_no_fitted_memory(tmp_path):
    era5 = load_plan(_gui_plan(tmp_path, source="era5", route="experiment", cycle="1999-05-03T18"))
    config, _ = generate_intent_config(era5, destination=tmp_path / "written")
    path = tmp_path / "own-config.json"
    path.write_text(json.dumps({"schema": PLAN_SCHEMA, "name": "own", "route": "experiment",
                                "config": {"path": str(config)}, "output_root": str(tmp_path / "run")}),
                    encoding="utf-8")
    resolution, _exp, _data = resolve_plan(load_plan(path), require_inputs=False)
    assert resolution["memory"] is None
    assert memory_fit(resolution) is None


@pytest.mark.parametrize("document", [{}, {"memory": None}, {"memory": {"budget_bytes": 1}},
                                      {"memory": {"peak_envelope_bytes": 1, "budget_bytes": 0}}])
def test_a_document_without_a_usable_record_shows_the_grid_alone(document):
    assert memory_fit(document) is None


def test_the_record_is_read_in_bytes_and_shown_in_gib():
    record = {"peak_envelope_bytes": int(5.92 * GIB), "budget_bytes": int(14.54 * GIB)}
    assert memory_fit({"memory": record}) == {"need_gib": 5.92, "budget_gib": 14.54, "fits": True}
    over = {"peak_envelope_bytes": int(40.97 * GIB), "budget_bytes": int(14.54 * GIB)}
    assert memory_fit({"memory": over})["fits"] is False


@pytest.mark.parametrize("source, route, cycle, words", [
    # Preprocessing not priced for the source: the wording the page could not read.
    ("hrrr", "prepared", "2026-09-24T12", "is NOT PRICED here"),
    ("rap", "prepared", "2026-09-24T12", "is NOT PRICED here"),
    ("ecmwf-open-data", "prepared", "2026-09-24T12", "is NOT PRICED here"),
    # Preprocessing priced: the wording it could.
    ("era5", "experiment", "1999-05-03T18", "is the memory-binding phase at"),
])
def test_a_draft_too_big_for_its_card_is_refused_with_its_figures(tmp_path, capsys, source, route, cycle, words):
    from gpuwm.cli import main

    plan = _gui_plan(tmp_path, source=source, route=route, cycle=cycle, **_TOO_BIG)
    assert main(["run-plan", str(plan), "--resolve"]) == 2
    out, err = capsys.readouterr()
    # The sentence is still the refusal a person reads, on stderr.
    assert "run plan 'config.intent' does not fit" in err and words in err

    document = json.loads(out)
    assert (document["schema"], document["kind"], document["created"]) == (
        "arwen.configuration-error.v1", "memory", False)
    # The whole refusal, both layers, without the marker between them.
    assert words in document["error"] and "[[explain]]" not in document["error"]
    sentence = _REFUSED.search(document["error"])
    record = document["memory"]
    assert record["binding_phase"] == "forecast"
    assert (_gib(record["peak_envelope_bytes"]), _gib(record["budget_bytes"])) == (
        float(sentence.group(1)), float(sentence.group(2)))

    memory = memory_fit(document)
    assert memory == {"need_gib": float(sentence.group(1)), "budget_gib": 6.75, "fits": False}
    assert memory["need_gib"] > 340


def test_estimate_refuses_a_too_big_draft_with_the_same_figures(tmp_path, capsys):
    from gpuwm.cli import main

    # The price of a plan resolves it first, so a draft too big for its card is refused there, and the
    # figures reach the machine channel of --estimate as they reach that of --resolve.
    plan = _gui_plan(tmp_path, source="hrrr", route="prepared", cycle="2026-09-24T12", **_TOO_BIG)
    assert main(["run-plan", str(plan), "--estimate"]) == 2
    out, err = capsys.readouterr()
    assert "run plan 'config.intent' does not fit" in err and "is NOT PRICED here" in err

    document = json.loads(out)
    assert (document["schema"], document["kind"]) == ("arwen.configuration-error.v1", "memory")
    sentence = _REFUSED.search(document["error"])
    memory = memory_fit(document)
    assert memory == {"need_gib": float(sentence.group(1)), "budget_gib": 6.75, "fits": False}
    assert memory["need_gib"] > 340


def test_a_refusal_that_is_not_the_cards_prints_no_document(tmp_path, capsys):
    from gpuwm.cli import main

    # Too few levels is refused before any layout is priced.
    plan = _gui_plan(tmp_path, source="era5", route="experiment", cycle="1999-05-03T18", nz=2)
    assert main(["run-plan", str(plan), "--resolve"]) == 2
    out, err = capsys.readouterr()
    assert out == "" and "--nz must be at least 4" in err


def test_new_forecast_says_how_big_a_too_big_hrrr_draft_is(tmp_path, monkeypatch):
    """The page server and the engine it runs, as New forecast asks them: no fake runner."""

    # The start's availability is asked in the page server's own process; no server is asked in a test.
    monkeypatch.setattr("gpuwm.fetch._head_ok", lambda url: True)
    monkeypatch.setattr("gpuwm.source_availability.quick_head", lambda url: True)
    server = build_server(tmp_path / "runs", port=0, runner=Runner(tmp_path / "jobs"), token="t" * 43)
    serve_in_thread(server)
    try:
        response, body = request(server, "POST", "/api/create/fit", timeout=600, body={
            "source": "hrrr", "cycle": "2026-09-24T12", "lat": 35.3, "lon": -97.6, "hours": 6,
            "width_km": 2000, "height_km": 2000, "dx_km": 1, "nz": 100, "card": "8gb"})
    finally:
        server.shutdown()
        server.server_close()
    assert response.status == 422, body
    memory = body["memory"]
    assert memory["fits"] is False and memory["budget_gib"] == 6.75 and 340 < memory["need_gib"] < 360, memory
    assert body["message"] == (f"Too big for the 8 GB card: this grid needs about {memory['need_gib']:.1f} GiB "
                               "and 6.8 GiB is usable."), body

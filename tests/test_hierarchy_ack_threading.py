"""The hierarchy forwards --ack into its namelist import.

The concrete breakage this gate prevents: every shortwave-only shipped
profile (the ones the wizard offers small cards) requires the
``constant-downward-longwave-v1`` declared-experiment acknowledgement,
and a WRF namelist has no spelling for it -- so before the seam existed
``gpuwm.hrrr_hierarchy_direct`` refused ITS OWN wizard-emitted namelist
pair at import time and no nested tree could ever be prepared on those
profiles.  The importer grew the ``acknowledgements`` kwarg for exactly
this; the hierarchy just never passed it.
"""

from __future__ import annotations

from pathlib import Path

import gpuwm.hrrr_hierarchy_direct as hh


def test_native_experiment_threads_acknowledgements(monkeypatch):
    seen = {}

    def fake_import(wps, inp, name=None, acknowledgements=(), **kwargs):
        seen["acknowledgements"] = acknowledgements
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(hh, "import_namelists", fake_import)
    try:
        hh._native_experiment(
            Path("a.wps"), Path("a.input"),
            acknowledgements=("constant-downward-longwave-v1",))
    except RuntimeError as err:
        assert "stop-after-capture" in str(err)
    assert seen["acknowledgements"] == ("constant-downward-longwave-v1",)


def test_default_import_carries_no_acknowledgements(monkeypatch):
    """Absence stays an empty tuple, so every existing import is
    byte-identical (the importer documents empty as the no-op)."""
    seen = {}

    def fake_import(wps, inp, name=None, acknowledgements=(), **kwargs):
        seen["acknowledgements"] = acknowledgements
        raise RuntimeError("stop-after-capture")

    monkeypatch.setattr(hh, "import_namelists", fake_import)
    try:
        hh._native_experiment(Path("a.wps"), Path("a.input"))
    except RuntimeError as err:
        assert "stop-after-capture" in str(err)
    assert seen["acknowledgements"] == ()

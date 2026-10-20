"""Measured capacity gates apply to new inference, after reusable receipts."""
from types import SimpleNamespace

import pytest

from gpuwm import earth2_run, earth2_job


def test_capacity_is_row_driven_and_names_the_selected_model(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected")
    commands = []

    def query(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="15841, 15000\n")

    monkeypatch.setattr(earth2_run.subprocess, "run", query)
    row = dict(min_vram_gib=30, vram_note="Measured single allocation 20.37 GiB.")
    with pytest.raises(ValueError, match="another-model.*20.37 GiB"):
        earth2_run.require_inference_vram("another-model", row)
    assert commands[0][-1] == "--id=GPU-selected"
    assert earth2_run.require_inference_vram("small-model", {}) is None


def test_nominal_32gb_card_can_meet_measured_floor(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(earth2_run.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="32607, 32000\n"))
    receipt = earth2_run.require_inference_vram("large-model", dict(min_vram_gib=30))
    assert receipt["total_mib"] == 32607
    assert receipt["minimum_gib"] == 30


def test_reused_fetch_does_not_enter_inference_capacity_gate(tmp_path, monkeypatch):
    from gpuwm import earth2_fetch_cache
    path = tmp_path / "spec.json"
    from gpuwm.earth2_door import build_spec, canonical_json
    path.write_text(canonical_json(build_spec(model="fcn3", cycle="2024-09-24T12", hours=1)))
    monkeypatch.setattr(earth2_fetch_cache, "reuse", lambda *a: {"cache_reuse": True})

    def unexpected(**kwargs):
        pytest.fail("a verified reusable receipt must avoid new model inference")

    monkeypatch.setattr(earth2_run, "run_model", unexpected)
    earth2_job._fetch(path, tmp_path / "output")

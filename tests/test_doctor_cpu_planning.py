"""Doctor on a box without CuPy keeps the declared-card sizing doors open.

It said such a box "cannot run gpuwm check, gpuwm domain", while
docs/public/WITHOUT-A-GPU.md walks a GPU-less reader through exactly
those two with a declared card, and both run: only their
measure-this-card forms need the device.
"""

from __future__ import annotations

from gpuwm import doctor
from gpuwm.cli import main


def _missing_cupy(monkeypatch):
    monkeypatch.setattr(doctor, "_installed_cupy_wheels", lambda: [])
    monkeypatch.setattr(doctor, "_import_probe", lambda *_args, **_kw: (False, "not installed"))
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: None)


def test_missing_cupy_names_only_the_measuring_forms_of_domain_and_check(monkeypatch):
    _missing_cupy(monkeypatch)
    result = doctor._cupy_check()
    assert result.status == "missing" and result.blocking
    unavailable, available = result.detail.split(".  It can still run ", 1)
    assert ", gpuwm check," not in unavailable and ", gpuwm domain," not in unavailable
    assert "gpuwm domain and gpuwm check sized by measuring this machine's card" in unavailable
    assert "gpuwm domain with --card or --vram-gib" in available
    assert "gpuwm check with --free-gib or --budget-gib and --vram-gib" in available


def test_the_gpu_extra_lines_say_the_same(monkeypatch):
    for extra in ("gpu-cu12", "gpu-cu13"):
        facts = doctor._EXTRA_FACTS[extra]
        assert "gpuwm domain" not in facts.doors and "gpuwm check" not in facts.doors
        assert "--card" in facts.still_works and "--budget-gib" in facts.still_works


def test_the_declared_forms_doctor_now_names_do_run_without_cupy(tmp_path, capsys, monkeypatch):
    """Held against the doors themselves, not against the sentence.

    Exit 2 is the refusal code (a missing capability included); 0, 1 and
    4 are the check's own verdicts on a priced configuration.
    """
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    out = tmp_path / "cpuwalk.toml"
    assert main(["domain", "--point", "35.3,-97.5", "--card", "12gb", "--source", "gfs",
                 "--cycle", "2026-09-27T18", "--hours", "3", "--out", str(out)]) == 0
    assert main(["check", str(out), "--budget-gib", "8.23", "--vram-gib", "12",
                 "--json"]) in (0, 1, 4)
    assert "Traceback" not in capsys.readouterr().err

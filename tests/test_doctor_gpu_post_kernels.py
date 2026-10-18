"""``gpuwm doctor``'s "GPU post kernels" line, judged by the exporter's own
``--probe-post-device``.

Breakage it names: through 2.9.0 the GRIB2 post PTX (NVRTC 13.4, ISA 9.4)
was refused by drivers older than CUDA 13.4 (595.91 measured), and every card
without a built-in CUBIN computed GRIB2 products on the CPU without a word.
"""
from pathlib import Path
from types import SimpleNamespace

from gpuwm import doctor, grib2_export


def _check(monkeypatch, probe, *, exe=Path("rw_grib2export"), no_gpu=False):
    monkeypatch.setattr(doctor, "no_local_gpu", lambda: no_gpu)
    monkeypatch.setattr(grib2_export, "BINARY", SimpleNamespace(find=lambda: exe))
    monkeypatch.setattr(grib2_export, "probe_post_device", lambda path: probe)
    return doctor._gpu_post_kernels_check()


def test_kernels_that_load_are_verified(monkeypatch):
    check = _check(monkeypatch, {"ok": True, "driver": True,
                                 "device": "gpu NVIDIA GeForce RTX 4090 (sm_89, compute_60 PTX)"})
    assert check.status == "verified" and "compute_60 PTX" in check.detail


def test_a_driver_refusal_is_a_degraded_gap_naming_the_error_and_the_floor(monkeypatch):
    error = ("load post kernels: NVIDIA A100 (sm_80), driver CUDA 12.2: compute_60 PTX: "
             "CUDA_ERROR_UNSUPPORTED_PTX_VERSION")
    check = _check(monkeypatch, {"ok": False, "driver": True, "error": error,
                                 "operation": "load post kernels",
                                 "ptx_driver_floor": "525.60.13 (Linux) / 527.41 (Windows), CUDA 12.0"})
    assert check.status == "missing" and check.severity == doctor.SEVERITY_DEGRADED
    assert not check.blocking
    assert error in check.detail and "on the CPU" in check.detail
    assert "525.60.13" in check.remedy and "525.60.13" in check.action


def test_no_driver_is_information_not_a_gap(monkeypatch):
    check = _check(monkeypatch, {"ok": False, "driver": False, "error": "no NVIDIA driver library"})
    assert check.status == "info"


def test_untested_without_exporter_older_exporter_or_with_the_opt_out(monkeypatch):
    assert _check(monkeypatch, None, exe=None).status == "untested"
    assert _check(monkeypatch, None).status == "untested"
    assert _check(monkeypatch, {"ok": True}, no_gpu=True).status == "untested"


def test_the_line_is_in_the_default_estate():
    import inspect
    assert "_gpu_post_kernels_check()" in inspect.getsource(doctor._collect_checks)


def test_a_card_that_does_not_open_is_not_blamed_on_the_driver_floor(monkeypatch):
    check = _check(monkeypatch, {"ok": False, "driver": True, "operation": "open CUDA device",
                                 "error": "open CUDA device: DriverError(CUDA_ERROR_NO_DEVICE)"})
    assert check.status == "untested" and "CUDA_ERROR_NO_DEVICE" in check.detail

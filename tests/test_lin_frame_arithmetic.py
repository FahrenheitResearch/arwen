from dataclasses import replace
from types import MappingProxyType

import pytest
from gpuwm.core import kernel_frame_recordings as kfr


def test_lin_recording_selects_the_measured_arithmetic_without_changing_default(monkeypatch):
    from gpuwm import wrf_exact
    fp = {"device_compute_capability": "120", "nvrtc_build": "12.9.86"}
    rows = [r for r in kfr.KERNEL_LOCAL_FRAME_RECORDINGS
            if r.platform_key == (fp["device_compute_capability"], fp["nvrtc_build"])
            and "lin" in r.frames and r.strict_recording is not None]
    assert len(rows) == 1
    monkeypatch.setattr(wrf_exact, "ENABLED", False)
    default = rows[0]
    assert dict(default.frames) == {"lin": 37808, "lin_refl": 14080}
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    strict = default.strict_recording
    assert dict(strict.frames) == {"lin": 35744, "lin_refl": 14080}
    assert strict.platform_key == default.platform_key
    assert dict(default.frames) == {"lin": 37808, "lin_refl": 14080}


def test_recorded_strict_frames_cannot_escape_the_safe_ceiling(monkeypatch):
    row = replace(kfr.KERNEL_LOCAL_FRAME_RECORDINGS[0],
                  frames=MappingProxyType({"probe": 16}),
                  strict_recording=replace(kfr.KERNEL_LOCAL_FRAME_RECORDINGS[0],
                      frames=MappingProxyType({"probe": 32}), strict_recording=None))
    monkeypatch.setattr(kfr, "KERNEL_LOCAL_FRAME_RECORDINGS", (row,))
    assert kfr.frame_ceiling() == {"probe": 32}


def test_legacy_recordings_keep_their_identity_in_both_arithmetic_modes(monkeypatch):
    from gpuwm import wrf_exact
    row = next(r for r in kfr.KERNEL_LOCAL_FRAME_RECORDINGS
               if r.compute_capability == "86" and r.nvrtc_build == "12.9.86")
    monkeypatch.setattr(kfr, "KERNEL_LOCAL_FRAME_RECORDINGS", (row,))
    fp = {"device_compute_capability": "86", "nvrtc_build": "12.9.86"}
    for exact in (False, True):
        monkeypatch.setattr(wrf_exact, "ENABLED", exact)
        assert kfr.recording_for(fp) is row


@pytest.mark.gpu
def test_lin_recording_matches_the_actual_default_and_strict_images():
    cp = pytest.importorskip("cupy")
    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("no CUDA GPU / cupy")
    if not count:
        pytest.skip("no CUDA GPU / cupy")
    from gpuwm.certify.compile_platform import compile_platform_fingerprint
    from gpuwm.core.kernels import get_kernel
    fingerprint = compile_platform_fingerprint()
    if (fingerprint["device_compute_capability"], fingerprint["nvrtc_build"]) != ("120", "12.9.86"):
        pytest.skip("Lin arithmetic frame recording was measured at sm120 / NVRTC 12.9.86")
    recording = kfr.recording_for(fingerprint)
    observed = {module: int(get_kernel(module, function).local_size_bytes)
                for module, function in (("lin", "lin_column"),
                                         ("lin_refl", "refl10cm_lin_column"))}
    assert observed == dict(recording.frames)
    for module, actual in observed.items():
        assert kfr.frame_ceiling()[module] >= actual

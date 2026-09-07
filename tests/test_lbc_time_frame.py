"""The boundary-time operation must remain visible to local-memory pricing."""
from datetime import datetime
from pathlib import Path
import re

import pytest

from gpuwm.config import load_config
from gpuwm.core import preflight as pf
from gpuwm.experiment import experiment_from_run_config

ROOT = Path(__file__).resolve().parents[1]


def test_runtime_prices_the_boundary_time_module_and_future_frame_growth(monkeypatch):
    exp = experiment_from_run_config(load_config(ROOT / "configs/real74_d01.toml"),
                                     datetime(1974, 4, 3, 12))
    assert "lbc_time" in pf.physics_kernel_modules(exp)
    frames = pf.kernel_local_frame_bytes(exp)
    assert frames["lbc_time"] == 0  # Both recorded driver readings.
    original = pf.kernel_local_memory_bytes(exp)
    # Discriminating control: a source/compiler growing this frame must alter
    # admission, even with every configured physics selector unchanged.
    monkeypatch.setitem(pf.KERNEL_MAX_LOCAL_SIZE_BYTES, "lbc_time", 65536)
    assert pf.kernel_local_frame_bytes(exp)["lbc_time"] == 65536
    assert pf.kernel_local_memory_bytes(exp) > original


@pytest.mark.gpu
def test_every_boundary_time_entry_has_a_driver_measured_frame():
    from conftest import HAS_GPU
    if not HAS_GPU:
        pytest.skip("no CUDA GPU / cupy")
    from gpuwm.core.kernels import load_module
    from gpuwm.certify.compile_platform import compile_platform_fingerprint

    source = (ROOT / "gpuwm/core/kernels/lbc_time.cu").read_text(encoding="utf-8")
    symbols = set(re.findall(r'extern\s+"C"\s+__global__\s+void\s+([A-Za-z_][A-Za-z0-9_]*)', source))
    assert symbols == {"evaluate_linear_boundary", "evaluate_rational_boundary"}
    module = load_module("lbc_time")
    # Every symbol lookup is required; a compile/lookup error cannot become0.
    actual = {name: int(module.get_function(name).attributes["local_size_bytes"])
              for name in symbols}
    widest = max(actual.values())
    assert widest <= pf.KERNEL_MAX_LOCAL_SIZE_BYTES["lbc_time"], actual
    recording = pf.kernel_frame_recording_for(compile_platform_fingerprint())
    if recording is not None and "lbc_time" in recording.frames:
        assert widest == recording.frames["lbc_time"], actual

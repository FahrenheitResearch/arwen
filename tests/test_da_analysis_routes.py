"""Which storage route solves an analysis, and what a failure falls to.

Breakage prevented: the device LETKF's memory failure fell straight to
the host-staged route (23 minutes per analysis on the 241 x 241 x 49,
32-member case), and a broken import of the device route silently did
the same with nothing in the receipt.  The fallback is now the host
route, whose chunk loop runs on forked workers and needs no card memory.
"""
import numpy as np
import pytest

from gpuwm.da import letkf
from gpuwm.da import radar_assimilation as owner


class _Namespace:
    class cuda:  # noqa: N801
        class runtime:  # noqa: N801
            @staticmethod
            def deviceSynchronize():
                return None

    def get_default_memory_pool(self):
        return self

    def free_all_blocks(self):
        return None


def _prior():
    return {"u": np.zeros((4, 2, 3, 3))}


def test_device_memory_failure_falls_to_the_host_route_on_numpy(monkeypatch):
    calls = []

    def fake_device(prior, batches, geometry, config, diagnostics, progress=None):
        calls.append("device")
        raise MemoryError("device LETKF scratch")

    def fake_analyze(prior, batches, geometry, config, diagnostics, **options):
        calls.append(("host", options.get("solve_namespace")))
        return {name: np.ones_like(value) for name, value in prior.items()}
    fake_analyze.supports_host_staging = True

    import gpuwm.da.letkf_device as device
    monkeypatch.setattr(device, "analyze_device", fake_device)
    monkeypatch.setattr(device, "supported", lambda members, dtype: True)
    monkeypatch.setattr(letkf, "analyze", fake_analyze)
    config = letkf.LetkfConfig(letkf.Localization(9000.0, 2000.0), ("u",), 0.5)
    out = owner._execute_analysis(fake_analyze, _prior(), [], None, config,
                                  namespace=_Namespace(), device="cuda")
    increments, _diag, _s, _u, storage, attempts = out
    assert storage == "host"
    assert calls == ["device", ("host", np)]
    assert [row["status"] for row in attempts] == ["memory-failed", "computed"]
    assert attempts[0]["storage"] == "cuda-obs-sparse"


def test_a_device_route_that_cannot_import_is_recorded(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "gpuwm.da.letkf_device":
            raise ImportError("no cupy here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    config = letkf.LetkfConfig(letkf.Localization(9000.0, 2000.0), ("u",), 0.5)
    takes, why = owner._obs_sparse_takes(letkf.analyze, _prior(), config)
    assert not takes and "no cupy here" in why


def test_other_solvers_keep_their_own_routes():
    takes, why = owner._obs_sparse_takes(lambda *a, **k: None, _prior(), None)
    assert (takes, why) == (False, None)

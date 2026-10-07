"""RUC cold start computed once per process for the same inputs.

THE BREAKAGE THIS PREVENTS: a DA card server wires every member leg from the
same prepared cache, and each wire re-ran the scalar ruclsminit transcription
over the whole slab, several seconds of one host core per member leg with the
card idle.  The result is reused only for byte-identical inputs, the same
land-use dataset and the same parameter bundle.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gpuwm.core import ruc_runtime


def _inputs(shift=0.0):
    tslb = np.full((9, 3, 4), 280.0 + shift, np.float32)
    smois = np.full((9, 3, 4), 0.3, np.float32)
    return (tslb, smois, np.ones((3, 4), np.int32), np.ones((3, 4), np.int32),
            np.zeros((3, 4), np.float32))


def test_same_inputs_compute_once_and_changed_inputs_recompute(monkeypatch):
    calls = []

    def fake(*inputs, mminlu, parameters):
        calls.append(float(inputs[0].flat[0]))
        return SimpleNamespace(sh2o=inputs[1].copy())

    monkeypatch.setattr(ruc_runtime, "ruc_initialize_cold_start", fake)
    monkeypatch.setattr(ruc_runtime, "_COLD_STARTS", {})
    params = SimpleNamespace(bundle=("bundle", 1), dataset_identifier="MODIS")
    first = ruc_runtime._cold_start_once(_inputs(), params)
    again = ruc_runtime._cold_start_once(_inputs(), params)
    assert again is first and calls == [280.0]
    ruc_runtime._cold_start_once(_inputs(1.0), params)
    assert calls == [280.0, 281.0]
    other = SimpleNamespace(bundle=("bundle", 2), dataset_identifier="MODIS")
    ruc_runtime._cold_start_once(_inputs(1.0), other)
    other_land = SimpleNamespace(bundle=("bundle", 2), dataset_identifier="USGS")
    ruc_runtime._cold_start_once(_inputs(1.0), other_land)
    assert calls == [280.0, 281.0, 281.0, 281.0]

"""Runtime CUDA gate for Noah-MP VEGE_FLUX.

The older oracle driver under ``tools/`` proved the CUDA arithmetic but was
not collected by pytest and used a fixture-only packing path.  This gate calls
the runtime batcher with the physical Python argument list, requires max_ulp 0
against every unmodified-WRF VEGE_FLUX row, and demonstrates that substituting
CUDA's device libm makes the same comparison fail.
"""

from __future__ import annotations

import struct

from conftest import requires_gpu

import test_noahmp_vegeflux as oracle

from gpuwm.core.noahmp_vegeflux_gpu import (VegeFluxBatch,
                                            VegeFluxBatchPending,
                                            evaluate_vege_flux_calls)


def _bits(value) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _captured_calls(monkeypatch):
    batch = VegeFluxBatch()
    monkeypatch.setattr(oracle, "vege_flux", batch.capture)
    tags = sorted(oracle.TABLE["vegeflux"])
    for tag in tags:
        try:
            oracle.call_vege_flux(oracle.TABLE["vegeflux"][tag])
        except VegeFluxBatchPending:
            pass
        else:
            raise AssertionError("the VEGE_FLUX capture seam did not pause")
    return tags, batch.calls


@requires_gpu
def test_runtime_device_batch_is_max_ulp_zero(monkeypatch):
    tags, calls = _captured_calls(monkeypatch)
    results = evaluate_vege_flux_calls(calls)
    mismatches = []
    compared = 0
    for tag, state in zip(tags, results):
        for name, want_hex in oracle.TABLE["vegeflux"][tag]["out"].items():
            if not hasattr(state, name):
                continue
            compared += 1
            got = _bits(getattr(state, name))
            want = int(want_hex, 16)
            if got != want:
                mismatches.append(
                    f"{tag}.{name}: got 0x{got:08x}, want 0x{want:08x}")
    assert compared == 300
    assert not mismatches, "max_ulp != 0:\n" + "\n".join(mismatches[:20])


@requires_gpu
def test_cuda_device_libm_negative_control_fails_the_same_gate(monkeypatch):
    """Prove the max_ulp gate observes the libm implementation it claims."""
    tags, calls = _captured_calls(monkeypatch)
    results = evaluate_vege_flux_calls(calls, use_device_libm=True)
    mismatches = 0
    for tag, state in zip(tags, results):
        for name, want_hex in oracle.TABLE["vegeflux"][tag]["out"].items():
            if hasattr(state, name):
                mismatches += (
                    _bits(getattr(state, name)) != int(want_hex, 16))
    assert mismatches > 0, (
        "CUDA device libm unexpectedly passed every WRF output; the positive "
        "max_ulp gate no longer distinguishes the shipped transcription")

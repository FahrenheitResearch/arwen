"""WRF oracle and executed ownership gates for LW upper corrections.

The generated taumol kernel and the public batched entry are compared with
frozen native WRF words using poisoned output buffers, repeated launches,
reused storage, and short chunk tails. Their floating arithmetic is unchanged.

A separately compiled module inserts atomic counters before the thirteen
fixed-index corrections, retaining each assignment and its owning-lane guard.
This checks executed ownership independently of whether a particular GPU
schedule makes the numerical race visible. It is not a timing measurement.

The compact fixture and its provenance are checked into tests/fixtures.
These tests do not regenerate expectations or require an external oracle deck.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pytest


CORRECTED_POINTS = {4: tuple(range(8, 15)), 7: tuple(range(6, 12))}
POISONS = (0x7FC12345, 0x3F314159)
CHUNK_CASES = ((1, 1), (9, 8), (17, 7))
MCICA_INPUTS = {"cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl"}
ORACLE_KEYS = ("uflx", "dflx", "hr", "uflxc", "dflxc", "hrc")


def _body(source, name):
    start = source.index("{", source.index(name + "("))
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


@pytest.fixture(scope="module")
def native_deck():
    if os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"):
        pytest.skip("GPUWM_NO_LOCAL_GPU prevents the oracle GPU tests")
    import cupy as cp
    import gpuwm
    from gpuwm.core import rrtmg_lw as lw
    from gpuwm.core.rrtmg_legacy import _lw_coeffs

    root = Path(gpuwm.__file__).resolve().parent.parent
    folder = root / "tests" / "fixtures"
    provenance = json.loads((folder / "rrtmg_lw_gpoint_ownership.provenance.json").read_text())
    assert provenance["schema"] == "rrtmg-lw-gpoint-ownership-oracle/v1"
    path = folder / provenance["array_file"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["array_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    C = _lw_coeffs()
    runtime = dict(C, **{"wvn/ngs": np.asarray(lw.NGS, dtype=np.int32)})
    coeffs = provenance["packaged_coefficients"]
    assert coeffs["runtime_all_equal"] is True
    for name in coeffs["runtime_keys"]:
        value = np.asarray(runtime[name])
        actual = {"dtype": value.dtype.str, "shape": list(value.shape),
                  "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
        assert actual == coeffs["comparison"][name]["native"], (
            name, "runtime coefficient differs from the native oracle")
    # Ensure the targeted products are present and nonzero, so identical zeros
    # cannot hide a skipped upper-atmosphere correction.
    for band, points in CORRECTED_POINTS.items():
        offset = int(lw.NGS[band - 2])
        values = arrays["upper__taug"][..., offset + np.asarray(points) - 1]
        assert np.isfinite(values).all() and np.all(values > 0), band
    lw.gpu_preflight(force=True)
    print("LW_ORACLE_DECK " + json.dumps({
        "array_sha256": provenance["array_sha256"],
        "native_case_count": provenance["native_deck"]["cases"],
        "runtime_coefficient_keys": len(coeffs["runtime_keys"]),
        "generated_source_sha256": hashlib.sha256(lw._gpu_source().encode()).hexdigest(),
        "device": int(cp.cuda.Device().id), "cupy": cp.__version__, "numpy": np.__version__}), flush=True)
    return {"cp": cp, "lw": lw, "C": C, "arrays": arrays, "provenance": provenance}


def _upper_columns(deck):
    """Select layer-local oracle states without recomputing any input value."""
    lw = deck["lw"]
    output = []
    for index, sample in enumerate(deck["provenance"]["upper_samples"]):
        st = {name: deck["arrays"]["upper__state__" + name][index].copy()
              for name in (*lw.GPU_FSLOTS, *lw.GPU_ISLOTS)}
        st["wx"] = deck["arrays"]["upper__state__wx"][index].copy()
        st["laytrop"] = 0
        output.append({"state": st, "case_index": sample["case_index"],
                       "case": sample["case"], "case_sha256": sample["case_sha256"],
                       "layers": sample["layers_zero_based"],
                       "taug": deck["arrays"]["upper__taug"][index],
                       "fracs": deck["arrays"]["upper__fracs"][index]})
    return output


def _word_check(actual, expected, label):
    actual = np.ascontiguousarray(actual)
    expected = np.ascontiguousarray(expected)
    assert actual.dtype == expected.dtype == np.dtype(np.float32), label
    assert actual.shape == expected.shape, label
    unequal = actual.view(np.uint32) != expected.view(np.uint32)
    indices = np.argwhere(unequal)
    return {"label": label, "words": int(actual.size), "different": int(unequal.sum()),
            "first_indices": indices[:8].tolist(),
            "actual_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "oracle_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}


def _band_checks(actual, expected, label, lw):
    checks = []
    for band, points in CORRECTED_POINTS.items():
        offset = int(lw.NGS[band - 2])
        stop = int(lw.NGS[band - 1])
        checks.append(_word_check(actual[..., offset:stop], expected[..., offset:stop],
                                  f"{label}/band{band}/all"))
        corrected = offset + np.asarray(points) - 1
        checks.append(_word_check(actual[..., corrected], expected[..., corrected],
                                  f"{label}/band{band}/corrected"))
    return checks


def _writer_instrumented_source(source):
    """Add counters before actual correction stores, retaining owner clauses."""
    slot = 0
    for band, points in CORRECTED_POINTS.items():
        original = _body(source, f"rlw_gband{band}")
        instrumented = original
        for point in points:
            pattern = rf"TAUG\(gs \+ {point}\) = [^;\n]+;"
            found = list(re.finditer(pattern, instrumented))
            assert len(found) == 1, (band, point, "missing or ambiguous correction")
            statement = found[0].group(0)
            replacement = (
                "{ atomicAdd(&rlw_proof_writers["
                f"((long long)col * nl + lay - 1) * 13 + {slot}], 1u); "
                + statement + " }")
            instrumented = instrumented[:found[0].start()] + replacement + instrumented[found[0].end():]
            slot += 1
        assert original in source
        source = source.replace(original, instrumented, 1)
    assert slot == 13
    # A one-thread setter avoids relying on a toolchain-specific global-symbol
    # lookup API. Counter allocation and every launch use the same stream.
    return ("__device__ unsigned int* rlw_proof_writers;\n"
            'extern "C" __global__ void rlw_proof_bind_writers(unsigned int* counts) {\n'
            "    if (blockIdx.x == 0 && threadIdx.x == 0) rlw_proof_writers = counts;\n"
            "}\n" + source)


@pytest.mark.gpu
def test_actual_upper_correction_writer_counts(native_deck):
    """Executed ownership: old helpers produce 16 writers; owning helpers 1.

    Counters alter scheduling, so this is an ownership control, not an oracle
    or a timing result. Uninstrumented oracle tests separately check values.
    """
    deck = native_deck
    cp, lw = deck["cp"], deck["lw"]
    from cupy.cuda import compiler

    samples = _upper_columns(deck)
    selected = [samples[index % len(samples)] for index in range(3)]
    nc, nl = len(selected), 3
    fs = np.stack([np.stack([sample["state"][name] for sample in selected])
                   for name in lw.GPU_FSLOTS]).astype(np.float32, copy=False)
    isv = np.stack([np.stack([sample["state"][name] for sample in selected])
                    for name in lw.GPU_ISLOTS]).astype(np.int32, copy=False)
    wx = np.stack([sample["state"]["wx"][:lw.MAXXSEC] for sample in selected])
    source = _writer_instrumented_source(lw._gpu_source())
    # Match the engine's direct NVRTC options, including subnormal handling.
    ptx, _ = compiler.compile_using_nvrtc(source, ("-std=c++17", "--ftz=false"),
                                         None, "rrtmg_lw_writer_proof.cu")
    module = cp.cuda.function.Module()
    module.load(ptx.encode() if isinstance(ptx, str) else ptx)
    counts = cp.zeros((nc, nl, 13), dtype=cp.uint32)
    module.get_function("rlw_proof_bind_writers")((1,), (1,), (counts,))
    inputs = (cp.zeros(nc, dtype=cp.int32), cp.asarray(np.ascontiguousarray(fs)),
              cp.asarray(np.ascontiguousarray(isv)), cp.asarray(np.ascontiguousarray(wx)))
    K = lw._lw_dev_consts(cp, deck["C"])
    taug = cp.full((nc, nl, lw.NGPTLW), np.nan, dtype=cp.float32)
    fracs = cp.full_like(taug, np.nan)
    module.get_function("rlw_taumol_batched")(
        ((nc * nl * 16 + 127) // 128, 16), (128,),
        (np.int32(nc), np.int32(nl), *inputs,
         K["chi"], K["oneminus"], K["bandptrs"], taug, fracs))
    actual = cp.asnumpy(counts)
    values, frequencies = np.unique(actual, return_counts=True)
    print("LW_EXECUTED_WRITERS " + json.dumps({
        "columns": nc, "upper_layers": nl, "corrections_per_layer": 13,
        "instrumented_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "count_distribution": {str(int(value)): int(count)
                               for value, count in zip(values, frequencies)},
        "counts": actual.tolist()}), flush=True)
    np.testing.assert_array_equal(actual, np.ones_like(actual),
        err_msg="each fixed upper-atmosphere correction requires exactly one executed writer")


@pytest.mark.gpu
@pytest.mark.parametrize("columns,chunk", CHUNK_CASES)
@pytest.mark.parametrize("poison", POISONS, ids=("nan-payload", "finite-stale"))
def test_upper_batched_bands_match_wrf_with_poison_and_tail(native_deck, columns, chunk, poison):
    deck = native_deck
    cp, lw, C = deck["cp"], deck["lw"], deck["C"]
    samples = _upper_columns(deck)
    checks = []
    # Retain a standalone oracle anchor within this focused regression.
    for sample in samples:
        for band in CORRECTED_POINTS:
            taug, fracs = lw.gpu_taugb(band, sample["state"], C)
            offset, stop = int(lw.NGS[band - 2]), int(lw.NGS[band - 1])
            for name, value in (("taug", taug), ("fracs", fracs)):
                checks.append(_word_check(cp.asnumpy(value)[0, :, offset:stop],
                    sample[name][:, offset:stop], f"standalone/{sample['case']}/{band}/{name}"))
    assert not any(row["different"] for row in checks), checks

    selected = [samples[index % len(samples)] for index in range(columns)]
    nl, ng = 3, lw.NGPTLW
    fs = np.stack([np.stack([sample["state"][name] for sample in selected])
                   for name in lw.GPU_FSLOTS]).astype(np.float32, copy=False)
    isv = np.stack([np.stack([sample["state"][name] for sample in selected])
                    for name in lw.GPU_ISLOTS]).astype(np.int32, copy=False)
    wx = np.stack([sample["state"]["wx"][:lw.MAXXSEC] for sample in selected])
    expected = {name: np.stack([sample[name] for sample in selected])
                for name in ("taug", "fracs")}
    K = lw._lw_dev_consts(cp, C)
    kernel = lw._gpu_kernel("rlw_taumol_batched")
    guard = 32
    capacity = chunk * nl * ng
    buffers = {name: cp.empty(capacity + 2 * guard, dtype=cp.float32)
               for name in expected}
    launch_rows = []
    for repeat in range(3):
        actual = {name: np.empty_like(value) for name, value in expected.items()}
        for start in range(0, columns, chunk):
            end = min(columns, start + chunk)
            count = end - start
            size = count * nl * ng
            for buffer in buffers.values():
                buffer.view(cp.uint32).fill(np.uint32(poison))
            arrays = {name: buffer[guard:guard + size].reshape(count, nl, ng)
                      for name, buffer in buffers.items()}
            inputs = (cp.zeros(count, dtype=cp.int32),
                      cp.asarray(np.ascontiguousarray(fs[:, start:end])),
                      cp.asarray(np.ascontiguousarray(isv[:, start:end])),
                      cp.asarray(np.ascontiguousarray(wx[start:end])))
            blocks = (count * nl * 16 + 127) // 128
            kernel((blocks, 16), (128,),
                   (np.int32(count), np.int32(nl), *inputs,
                    K["chi"], K["oneminus"], K["bandptrs"], arrays["taug"], arrays["fracs"]))
            for name, array in arrays.items():
                actual[name][start:end] = cp.asnumpy(array)
                words = cp.asnumpy(buffers[name]).view(np.uint32)
                assert np.all(words[:guard] == poison), f"{name} prefix guard overwritten"
                assert np.all(words[guard + size:] == poison), f"{name} tail guard overwritten"
            launch_rows.append({"repeat": repeat, "start": start, "count": count,
                                "blocks": blocks, "active_threads": count * nl * 16})
        for name in expected:
            checks.extend(_band_checks(actual[name], expected[name], f"repeat{repeat}/{name}", lw))
            checks.append(_word_check(actual[name], expected[name], f"repeat{repeat}/{name}/all16bands"))
    print("LW_UPPER_ORACLE " + json.dumps({
        "columns": columns, "chunk": chunk, "poison_uint32": poison,
        "samples": [{k: value for k, value in sample.items() if k not in ("state", "taug", "fracs")}
                    for sample in samples], "launches": launch_rows, "checks": checks}), flush=True)
    assert not any(row["different"] for row in checks), [row for row in checks if row["different"]]


@pytest.mark.gpu
@pytest.mark.parametrize("chunk", (1, 7, 16, 17))
@pytest.mark.parametrize("poison", POISONS, ids=("nan-payload", "finite-stale"))
def test_public_batched_fluxes_match_wrf_after_poison(native_deck, monkeypatch, chunk, poison):
    deck = native_deck
    cp, lw = deck["cp"], deck["lw"]
    idx = deck["arrays"]["full__repeat_indices"]
    ins = {}
    for key, value in deck["arrays"].items():
        if not key.startswith("full__input__"):
            continue
        name = key.removeprefix("full__input__")
        ins[name] = (value.item() if value.ndim == 0 else
                     value[:, idx, :] if name in MCICA_INPUTS else value[idx])
    ins["ncol"] = int(idx.size)
    assert ins["ncol"] == 17
    expected = {name: deck["arrays"]["full__expected__" + name][idx] for name in ORACLE_KEYS}
    original = lw._gpu_kernel
    launches = []

    def kernel(name):
        function = original(name)
        if name != "rlw_taumol_batched":
            return function

        def launch(grid, block, args, **kwargs):
            # Poison only the two write-before-read outputs, on their existing
            # stream. The generated kernel and every floating statement stay
            # unchanged. This hook is not used by the original repeat probe.
            args[-2].view(cp.uint32).fill(np.uint32(poison))
            args[-1].view(cp.uint32).fill(np.uint32(poison))
            launches.append(int(args[0]))
            return function(grid, block, args, **kwargs)
        return launch

    monkeypatch.setattr(lw, "_gpu_kernel", kernel)
    checks = []
    expected_launches = [min(chunk, 17 - start) for start in range(0, 17, chunk)]
    for repeat in range(2):
        actual = lw.gpu_rrtmg_lw_batched(**ins, C=deck["C"], column_chunk=chunk)
        for name in ORACLE_KEYS:
            checks.append(_word_check(actual[name], expected[name], f"repeat{repeat}/{name}"))
    print("LW_BATCHED_ORACLE " + json.dumps({
        "columns": 17, "chunk": chunk, "poison_uint32": poison,
        "case_indices": idx.tolist(), "launch_column_counts": launches, "checks": checks}), flush=True)
    assert launches == expected_launches * 2, "chunk and short-tail launches were not exercised"
    assert not any(row["different"] for row in checks), [row for row in checks if row["different"]]

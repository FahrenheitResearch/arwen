"""CPU workspace checks and opt-in GPU uint32 oracle gates.

Set LW_SPEED_INPUTS and LW_SPEED_ORACLE to directories prepared by
 tools/rrtmg_lw_speed_gate.py. No tolerance comparison is permitted.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import rrtmg_lw as lw


def test_workspace_reuses_written_profiles():
    scratch = lw.LWBatchScratch(np)
    for name in lw.LW_SCRATCH_SLOTS:
        dtype = np.uint8 if name == "iclddn_p" else np.float32
        a = scratch.take(name, (257, 7, 140), dtype)
        a.fill(123)
        b = scratch.take(name, (1, 7, 140), dtype)
        assert np.shares_memory(a, b)
        assert (b == 123).all()
        assert b.flags.c_contiguous
    with pytest.raises(KeyError):
        scratch.take("unknown", (1,), np.float32)


def test_address_twin_preserves_float_statements():
    import re
    kdir = Path(lw.__file__).parent / "kernels"
    chain = (kdir / "rrtmg_lw_chain.cu").read_text(encoding="utf-8")
    twin = lw._lw_coalesced_source(chain)
    # The committed twin is exactly the generator's output after its header,
    # so an edit to the chain that is not regenerated here fails.
    committed = (kdir / "rrtmg_lw_chain_coalesced.cu").read_text(
        encoding="utf-8")
    assert committed.startswith(lw._LW_COALESCED_HEADER)
    assert committed[committed.index("\n\n") + 2:] == twin
    # Intrinsic arithmetic statements retain their exact source order.
    assert re.findall(r"RLW_\w+\([^;]+", chain) == re.findall(r"RLW_\w+\([^;]+", twin)
    assert "#define PRO(a, l) a[((long long)col * nl + ((l) - 1)) * NGPTLW + (igc - 1)]" in twin


@pytest.mark.parametrize("chunk", [1, 256, 1536, 4096])
def test_saved_engine_oracle_uint32_dual_run(chunk):
    source = os.environ.get("LW_SPEED_INPUTS")
    oracle = os.environ.get("LW_SPEED_ORACLE")
    if not source or not oracle:
        pytest.skip("set LW_SPEED_INPUTS and LW_SPEED_ORACLE")
    pytest.importorskip("cupy")
    from gpuwm.core.rrtmg_legacy import _lw_coeffs
    C = _lw_coeffs()
    for path in sorted(Path(source).glob("*.npz")):
        with np.load(path) as deck:
            inputs = {k: (v.item() if v.ndim == 0 else v) for k, v in deck.items()}
        with np.load(Path(oracle) / path.name) as want:
            for run in range(2):
                got = lw.gpu_rrtmg_lw_batched(**inputs, C=C, column_chunk=chunk)
                for key in got:
                    np.testing.assert_array_equal(got[key].view(np.uint32),
                                                  want[key].view(np.uint32),
                                                  err_msg=f"{path.name}/{key}/chunk={chunk}/run={run}")


def test_every_empty_slot_has_a_write_before_read_reason():
    import ast
    tree = ast.parse(inspect_source())
    engine = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "gpu_rrtmg_lw_batched_device")
    found = set()
    for n in ast.walk(engine):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            f = n.value.func
            if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
                if f.value.id == "cp" and f.attr == "empty":
                    found.update(t.id for t in n.targets if isinstance(t, ast.Name))
    assert found == set(lw.LW_EMPTY_SLOTS)
    assert all(lw.LW_EMPTY_SLOTS.values())


def inspect_source():
    return Path(lw.__file__).read_text()


def test_constants_reuse_and_invalidate_by_object_and_device(monkeypatch):
    from collections import defaultdict
    from types import SimpleNamespace
    device = [0]
    uploads = []
    def upload(a):
        uploads.append(a)
        return np.array(a, copy=True)
    stream = SimpleNamespace(ptr=0, wait_event=lambda event: None)
    synchronized = []
    cp = SimpleNamespace(asarray=upload, cuda=SimpleNamespace(
        Device=lambda: SimpleNamespace(
            id=device[0], synchronize=lambda: synchronized.append(device[0])),
        get_current_stream=lambda: stream,
        Event=lambda **kw: SimpleNamespace(record=lambda: None, done=True),
        runtime=SimpleNamespace(getDevice=lambda: device[0])))
    monkeypatch.setattr(lw, "_LW_CONST_CACHE", {})
    monkeypatch.setattr(lw, "gpu_band_tabs", lambda band, C: ([], SimpleNamespace(data=SimpleNamespace(ptr=band))))
    C = defaultdict(lambda: np.ones(1, dtype=np.float32))
    a = lw._lw_dev_consts(cp, C)
    count = len(uploads)
    assert lw._lw_dev_consts(cp, C) is a
    assert len(uploads) == count
    other = defaultdict(lambda: np.ones(1, dtype=np.float32))
    assert lw._lw_dev_consts(cp, other) is not a
    # The replaced copy may still be read on another stream of the card, so
    # the replacement waits for the card first (lane 282-multigpu).
    assert synchronized == [0]
    device[0] = 1
    assert lw._lw_dev_consts(cp, other) is not lw._LW_CONST_CACHE[0][1]


def test_device_inputs_and_workspace_pricing():
    source = os.environ.get("LW_SPEED_INPUTS")
    oracle = os.environ.get("LW_SPEED_ORACLE")
    if not source or not oracle:
        pytest.skip("set LW_SPEED_INPUTS and LW_SPEED_ORACLE")
    cp = pytest.importorskip("cupy")
    from gpuwm.core.rrtmg_legacy import _lw_coeffs
    C = _lw_coeffs()
    path = Path(source) / "synthetic_mixed_5000.npz"
    with np.load(path) as deck:
        ins = {k: (v.item() if v.ndim == 0 else cp.asarray(v))
               for k, v in deck.items()}
    lw._lw_dev_consts(cp, C)
    pool = cp.get_default_memory_pool()
    for chunk in (1, 256, 1536, 4096):
        for run in range(2):
            base = pool.used_bytes()
            peak = [base]
            def probe(stage):
                peak[0] = max(peak[0], pool.used_bytes())
            out = lw.gpu_rrtmg_lw_batched(**ins, C=C, column_chunk=chunk,
                                        _stage_probe=probe)
            with np.load(Path(oracle) / path.name) as want:
                for k in out:
                    np.testing.assert_array_equal(out[k].view(np.uint32), want[k].view(np.uint32))
            estimate = lw.lw_batched_vram_bytes(chunk, ins["nlay"], ins["ncol"])
            assert peak[0] - base <= estimate, (chunk, peak[0] - base, estimate)
            print("device uint32 IDENTICAL", chunk, run, "pool", peak[0] - base,
                  "priced", estimate, flush=True)


def test_deferred_cloud_abort_text():
    source = os.environ.get("LW_SPEED_INPUTS")
    if not source:
        pytest.skip("set LW_SPEED_INPUTS")
    pytest.importorskip("cupy")
    from gpuwm.core.rrtmg_legacy import _lw_coeffs
    with np.load(Path(source) / "synthetic_overcast_257.npz") as deck:
        ins = {k: (v.item() if v.ndim == 0 else v.copy()) for k, v in deck.items()}
    ins["cldfmcl"][:, :-1, :] = 0
    ins["inflglw"] = 1
    for chunk in (1, 256, 1536, 4096):
        with pytest.raises(ValueError) as error:
            lw.gpu_rrtmg_lw_batched(**ins, C=_lw_coeffs(), column_chunk=chunk)
        assert str(error.value) == ("rlw_cldprmc device abort, code 1 "
                                    "(bounds violation, mirrors the Fortran stop)")

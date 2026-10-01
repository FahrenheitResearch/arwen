"""Real prep-deck bitwise gate, with a saved starting-engine result."""
import argparse
import inspect
import json
import time
from pathlib import Path
import numpy as np


def run(decks, receipts, baseline):
    import cupy as cp
    import gpuwm
    from gpuwm.core.rrtmg_legacy import _sw_tables
    from gpuwm.core.rrtmg_legacy_prep import swrad_prep_batch
    from gpuwm.core.rrtmg_mcica import gpu_generate_sw_subcolumns
    from gpuwm.core.rrtmg_sw import CudaSW, sw_batched_vram_bytes
    root = Path(__file__).resolve().parents[1]
    assert Path(gpuwm.__file__).resolve().is_relative_to(root), gpuwm.__file__
    print('gpuwm', gpuwm.__file__, flush=True)
    receipts.mkdir(parents=True, exist_ok=True)
    engine = CudaSW(_sw_tables())
    frames = engine.local_frame_bytes()
    frame_path = receipts / 'frames.json'
    if baseline:
        frame_path.write_text(json.dumps(frames, sort_keys=True))
    else:
        old_frames = json.loads(frame_path.read_text())
        assert all(value <= old_frames.get(name, 0) for name, value in frames.items()), (frames, old_frames)
    print('local frames', json.dumps(frames, sort_keys=True), flush=True)
    keys = set(inspect.signature(engine.rrtmg_sw_batched_device).parameters)
    deck_files = sorted(decks.glob('sw_*.npz'))
    assert deck_files, 'no shortwave prep decks'
    for deck in deck_files:
        prepared = receipts / (deck.stem + '_inputs.npz')
        if baseline and not prepared.exists():
            with np.load(deck) as data:
                kw = {k: cp.asarray(data[k]) if data[k].ndim else data[k] for k in data.files}
            inputs = swrad_prep_batch(**kw, subcolumn_generator=gpu_generate_sw_subcolumns)
            inputs = {k: v for k, v in inputs.items() if k in keys}
            inputs["adjes"] = np.full(int(inputs["ncol"]), inputs["adjes"], np.float32)
            np.savez(prepared, **{k: cp.asnumpy(v) if isinstance(v, cp.ndarray) else v for k, v in inputs.items()})
            del kw
        else:
            with np.load(prepared) as data:
                inputs = {k: cp.asarray(data[k]) if data[k].ndim else data[k] for k in data.files}
        layouts = ("gpoint",) if baseline else ("gpoint", "column")
        slab_names = ("cldfmcl", "taucmcl", "ssacmcl", "asmcmcl", "fsfcmcl",
                      "ciwpmcl", "clwpmcl", "cswpmcl")
        for layout in layouts:
            engine_inputs = dict(inputs)
            if layout == "column":
                for name in slab_names:
                    engine_inputs[name] = cp.ascontiguousarray(inputs[name].transpose(1, 2, 0))
            for chunk in (1, 256, 1536, 4096):
                reference = receipts / (deck.stem + f'_chunk{chunk}.npz')
                done = receipts / (reference.stem + '.done')
                if baseline and done.exists():
                    print(deck.stem, chunk, 'saved uint32 IDENTICAL dual run', flush=True)
                    continue
                previous = None
                for repeat in range(2):
                    engine.release_scratch()
                    cp.get_default_memory_pool().free_all_blocks()
                    required = sw_batched_vram_bytes(min(int(inputs['ncol']), chunk),
                                                    int(inputs['nlay']), int(inputs['ncol']), mcica_layout=layout)
                    for attempt in range(3):
                        while cp.cuda.runtime.memGetInfo()[0] < required + 2 * 2**30:
                            time.sleep(2)
                        try:
                            if layout == "column":
                                for name in slab_names:
                                    engine_inputs[name] = cp.ascontiguousarray(inputs[name].transpose(1, 2, 0))
                            result = engine.rrtmg_sw_batched_device(**engine_inputs, column_chunk=chunk, mcica_layout=layout)
                            break
                        except cp.cuda.memory.OutOfMemoryError:
                            if attempt == 2:
                                raise
                            engine.release_scratch()
                            cp.get_default_memory_pool().free_all_blocks()
                            time.sleep(2)

                    got = {k: cp.asnumpy(v).view(np.uint32) for k, v in result.items()}
                    del result
                    if previous is not None:
                        for k in got:
                            np.testing.assert_array_equal(got[k], previous[k], err_msg=f'dual run {deck.stem} {chunk} {k}')
                    if baseline and repeat == 0:
                        np.savez(reference, **got)
                    else:
                        with np.load(reference) as want:
                            for k in got:
                                np.testing.assert_array_equal(got[k], want[k], err_msg=f'{deck.stem} {chunk} {k}')
                    previous = got
                if baseline:
                    done.write_text('uint32 identical dual run\n')
                print(deck.stem, layout, chunk, 'uint32 IDENTICAL dual run', flush=True)
        del inputs
        engine.release_scratch()
        cp.get_default_memory_pool().free_all_blocks()


def test_layout_has_no_transpose_workspace():
    from gpuwm.core.rrtmg_sw import SW_TAKE_SLOTS
    assert not {'zcldfmc', 'ztaucmc', 'ztaormc', 'zasycmc', 'zomgcmc'} & {k for k, _ in SW_TAKE_SLOTS}


def test_abort_flag_is_zeroed_and_read_once_per_call():
    import ast
    import inspect
    import textwrap
    from gpuwm.core.rrtmg_sw import CudaSW, SW_CALL_ZEROED_SLOTS
    tree = ast.parse(textwrap.dedent(inspect.getsource(CudaSW.rrtmg_sw_batched_device)))
    function = tree.body[0]
    loop = next(node for node in function.body if isinstance(node, ast.For))
    for node in ast.walk(loop):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != 'asnumpy'
    assert SW_CALL_ZEROED_SLOTS == ('err',)
    source = inspect.getsource(CudaSW.rrtmg_sw_batched_device)
    assert source.count('scratch.zeros("err"') == 1
    assert source.count('cp.asnumpy(err_d)') == 1


def test_mcica_copy_keeps_gpoint_fastest():
    from gpuwm.core.rrtmg_sw import NGPTSW, _sw_dev_mcica
    source = np.arange(NGPTSW * 5 * 4, dtype=np.float32).reshape(NGPTSW, 5, 4)
    copied = _sw_dev_mcica(np, source, 2, 5, 3)
    assert copied.shape == (3, 3, NGPTSW)
    assert copied.strides[-1] == np.dtype(np.float32).itemsize
    np.testing.assert_array_equal(copied.view(np.uint32),
                                  source[:, 2:5, :3].transpose(1, 2, 0).view(np.uint32))


def test_fused_arithmetic_statement_correspondence():
    from collections import Counter
    import hashlib
    import re
    from gpuwm.core.rrtmg_sw import SPCVMC_WK_ARRAYS, SPCVMC_WKC_ARRAYS
    root = Path(__file__).resolve().parents[1]
    source = (root / 'gpuwm/core/kernels/rrtmg_sw.cu').read_text()
    expected = json.loads((root / 'tools/rrtmg_sw_witness/fusion_statement_manifest.json').read_text())

    def section(start, end):
        begin = source.index(start)
        return source[begin:source.index(end, begin)]

    def statements(body):
        body = re.sub(r'//[^\n]*', '', body)
        names = ('ztauc zomcc zgcc ztauo zomco zgco zrefc zrefdc ztrac ztradc '
                 'zref zrefd ztra ztrad zrefo zrefdo ztrao ztrado zdbtc ztdbtc '
                 'zdbt ztdbt zdbtc_nodel ztdbtc_nodel zdbt_nodel ztdbt_nodel '
                 'pref prefd ptra ptrad pgg ptau pw').split()
        result = []
        for statement in re.findall(r'[^;{}]+;', body):
            if not re.search(r'\b(AD|SU|MU|DV|__fsqrt_rn|rsw_etbl)\s*\(', statement):
                continue
            statement = re.sub(r'\breal\s+', '', statement)
            for name in names:
                statement = re.sub(r'\b' + name + r'\[(?:J|jk)(?:\s*\+\s*1)?\]', name, statement)
            for name in ('zrefc', 'zrefdc', 'ztrac', 'ztradc'):
                statement = statement.replace(name + '_layer', name)
            result.append(re.sub(r'\s+', '', statement))
        return dict(Counter(result))

    helper = section('__device__ __forceinline__ void rsw_reftra_layer(', '// ztdn is caller-provided')
    fused = section('__device__ void rsw_spcvmc_body(', 'extern "C" __global__\nvoid rsw_spcvmc_gpt(')
    assert statements(helper) == expected['reftra_layer']
    assert statements(fused) == expected['spcvmc_layer']
    vrtqdr = section('__device__ void rsw_vrtqdr(', '// ---------------------------------------------------------------------------\n// spcvmc_sw')
    assert hashlib.sha256(re.sub(r'\s+', '', vrtqdr).encode()).hexdigest() == expected['vrtqdr_sha256']
    assert SPCVMC_WK_ARRAYS == 16 and SPCVMC_WKC_ARRAYS == 0
    assert re.search(r'#define RSW_SPCVMC_WK 16\b', source)
    assert re.search(r'#define RSW_SPCVMC_WKC 0\b', source)
    assert fused.count('for (int J = 0; J < klev; ++J)') == 1
    assert fused.count('rsw_reftra_layer(') == 2
    assert 'const rsw_wk_t fd_col' not in fused
    assert 'const rsw_wkc_t lrtchk' not in fused


def test_sw_mcica_column_real_values():
    import os
    import pytest
    cp = pytest.importorskip("cupy")
    from gpuwm.core import rrtmg_legacy_device as dev, rrtmg_mcica as mc
    folder = os.environ.get("GPUWM_RRTMG_LEGACY_DECKS")
    if not folder:
        pytest.skip("real prep decks not configured")
    paths = sorted(Path(folder).glob("sw_*.npz"))
    assert paths
    slabs = {"cldfmcl", "taucmcl", "ssacmcl", "asmcmcl", "fsfcmcl",
             "ciwpmcl", "clwpmcl", "cswpmcl"}
    for path in paths:
        with np.load(path) as data:
            kw = {k: v.item() if v.ndim == 0 else v for k, v in data.items()}
        def generator(*args, layout):
            assert layout == "column"
            old = mc.gpu_generate_sw_subcolumns(*args)
            previous = None
            for _ in range(2):
                new = mc.gpu_generate_sw_subcolumns(*args, layout=layout)
                for name in old:
                    want = old[name].transpose(1, 2, 0) if name in slabs else old[name]
                    np.testing.assert_array_equal(cp.asnumpy(new[name]).view(np.uint32),
                                                  cp.asnumpy(want).view(np.uint32))
                    if previous is not None:
                        np.testing.assert_array_equal(cp.asnumpy(new[name]).view(np.uint32),
                                                      cp.asnumpy(previous[name]).view(np.uint32))
                previous = new
            return new
        dkw = {k: cp.asarray(v) if isinstance(v, np.ndarray) and v.ndim else v
               for k, v in kw.items()}
        dev.swrad_prep_batch_device(**dkw, subcolumn_generator=generator)


def test_column_layout_vram_accuracy():
    import pytest
    cp = pytest.importorskip("cupy")
    import test_rrtmg_sw_cuda as deck
    from gpuwm.core import rrtmg_sw as sw, rrtmg_lw as lw
    from gpuwm.core.rrtmg_legacy import _lw_coeffs
    pool = cp.get_default_memory_pool()
    for tall in (False, True):
        groups = deck._deck_groups()
        cs = max(groups.values(), key=(lambda cs: deck._flag_key(cs[0])[5]) if tall else len)
        nlay = deck._flag_key(cs[0])[5]
        engine = deck.cuda()
        for chunk in (len(cs), 8):
            for _ in range(2):
                engine.release_scratch()
                pool.free_all_blocks()
                ins = deck._group_inputs(cs)
                for name in deck._IN_MCICA:
                    if ins[name].ndim == 3:
                        ins[name] = cp.asarray(np.ascontiguousarray(ins[name].transpose(1, 2, 0)))
                base, peak = pool.used_bytes(), [0]
                def probe(stage):
                    peak[0] = max(peak[0], pool.used_bytes())
                # The helper's host fetch is outside the measured stages.
                icld, inflg, iceflg, liqflg, dyofyr, _ = deck._flag_key(cs[0])
                import inspect
                accepted = inspect.signature(engine.rrtmg_sw_batched_device).parameters
                kwargs = {k: v for k, v in ins.items() if k in accepted}
                result = engine.rrtmg_sw_batched_device(len(cs), nlay, icld,
                    **kwargs, inflgsw=inflg, iceflgsw=iceflg, liqflgsw=liqflg,
                    dyofyr=dyofyr, column_chunk=chunk, mcica_layout="column", _stage_probe=probe)
                estimate = sw.sw_batched_vram_bytes(min(chunk, len(cs)), nlay,
                    len(cs), mcica_layout="column")
                assert estimate >= peak[0] - base >= 0.5 * estimate
                del result, ins, kwargs
    import os
    folder = os.environ.get("LW_SPEED_INPUTS")
    if not folder:
        pytest.skip("LW saved inputs not configured")
    C = _lw_coeffs()
    constants = lw._lw_dev_consts(cp, C)
    for path in sorted(Path(folder).glob("lw_*.npz")):
        for _ in range(2):
            with np.load(path) as data:
                ins = {k: v.item() if v.ndim == 0 else cp.asarray(v) for k, v in data.items()}
            for name in ("cldfmcl", "taucmcl", "ciwpmcl", "clwpmcl", "cswpmcl"):
                ins[name] = cp.ascontiguousarray(ins[name].transpose(1, 2, 0))
            pool.free_all_blocks()
            base, peak = pool.used_bytes(), [0]
            def probe(stage):
                peak[0] = max(peak[0], pool.used_bytes())
            result = lw.gpu_rrtmg_lw_batched_device(**ins, C=C, column_chunk=int(ins["ncol"]), mcica_layout="column", _stage_probe=probe)
            estimate = lw.lw_batched_vram_bytes(ins["ncol"], ins["nlay"], mcica_layout="column")
            assert estimate >= peak[0] - base >= 0.5 * estimate
            del result, ins


def test_taumol_and_ordered_accumulation_arithmetic():
    import hashlib
    import re
    source = (Path(__file__).resolve().parents[1] / "gpuwm/core/kernels/rrtmg_sw.cu").read_text()
    # Ordered intrinsic statements from the starting engine, before remapping.
    expected = {'rsw_taumol_body': '2c341abc862d4441c6262f07beb39bf72ffba3e6784fc3aeb8e8c8c4c4c3072c', 'rsw_spc_accum_body': 'f02934fc1bc47d86b8f758b5c5dddcd84cf3c3f0539580e62d0c329cdbbc6b5b'}
    for name, digest in expected.items():
        start = source.index(name + "(")
        start = source.index("{", start)
        end, depth = start + 1, 1
        while depth:
            depth += (source[end] == "{") - (source[end] == "}")
            end += 1
        body = re.sub(r"//[^\n]*", "", source[start:end])
        statements = "\n".join(re.sub(r"\s+", "", item)
            for item in re.findall(r"[^;{}]+;", body)
            if re.search(r"\b(AD|SU|MU|DV|__fsqrt_rn|rsw_etbl)\s*\(", item))
        assert hashlib.sha256(statements.encode()).hexdigest() == digest


def test_column_mcica_is_a_view():
    from gpuwm.core.rrtmg_sw import NGPTSW, _sw_dev_mcica
    source = np.arange(5 * 4 * NGPTSW, dtype=np.float32).reshape(5, 4, NGPTSW)
    view = _sw_dev_mcica(np, source, 2, 5, 4, layout="column")
    assert np.shares_memory(source, view)
    np.testing.assert_array_equal(view.view(np.uint32), source[2:5].view(np.uint32))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('decks', type=Path)
    parser.add_argument('receipts', type=Path)
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    run(args.decks, args.receipts, args.baseline)

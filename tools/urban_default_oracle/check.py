"""Reject any default urban boundary mismatch, recording actual compiled options."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))


def tool(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    lane = Path(sys.argv[1])
    os.environ["CUDA_CACHE_PATH"] = str(lane / "cuda-cache")
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.cuda_subnormals import install
    from gpuwm.core import urban_bem as ub
    from gpuwm.core.kernels import load_module, module_source
    import _urban_bem_regime as B

    mode = "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default"
    receipt = lane / "receipts"
    install()
    options = []
    context = []
    original = compiler._NVRTCProgram.compile
    raw = cp.cuda.nvrtc.compileProgram

    def program(self, options=(), log_stream=None):
        context.append(hashlib.sha256(self.src.encode()).hexdigest())
        try:
            return original(self, options, log_stream)
        finally:
            context.pop()

    def compile_program(handle, args):
        options.append({"source_sha256": context[-1], "options": list(args)})
        return raw(handle, args)

    compiler._NVRTCProgram.compile = program
    cp.cuda.nvrtc.compileProgram = compile_program
    for name in ("urban_ucm", "urban_bep", "urban_bep_couple"):
        load_module(name).compile()
    ub._bem_module()
    sources = {name: hashlib.sha256(module_source(name).encode()).hexdigest()
               for name in ("urban_ucm", "urban_bep", "urban_bep_couple")}
    sources['urban_bep_bem'] = hashlib.sha256(ub.module_source().encode()).hexdigest()
    compiled = {name: [row for row in options if row['source_sha256'] == digest]
                for name, digest in sources.items()}
    for name, rows in compiled.items():
        assert rows, (name, "use a fresh cache")
        for row in rows:
            assert [o for o in row['options'] if o.lstrip('-').split('=')[0] == 'ftz'] == ['--ftz=false'], row
    (receipt / f"compile-{mode}.json").write_text(json.dumps({
        "cupy": cp.__version__, "nvrtc": cp.cuda.nvrtc.getVersion(),
        "modules": compiled}, indent=1) + '\n')

    slucm = tool(ROOT / "tools/urban_slucm_regime_oracle/compare.py")
    hashes = []
    measure = slucm._measure
    def hashed_measure(port, ref, scenario):
        for key in ref:
            a = np.ascontiguousarray(port[key], dtype=np.float32).tobytes()
            b = np.ascontiguousarray(ref[key], dtype=np.float32).tobytes()
            hashes.append({'scheme': 'slucm', 'field': key,
                           'port_sha256': hashlib.sha256(a).hexdigest(),
                           'wrf_sha256': hashlib.sha256(b).hexdigest()})
        return measure(port, ref, scenario)
    slucm._measure = hashed_measure
    slucm.main(['compare', str(lane / 'slucm-subnormal'), str(receipt / f'slucm-{mode}.json')])
    result = json.loads((receipt / f'slucm-{mode}.json').read_text())
    assert sum(v['replay']['rows'] for v in result['variants'].values()) == 1584
    assert all(v[arm]['rows_differ'] == 0 for v in result['variants'].values()
               for arm in ('replay', 'carried')), result
    slucm.main(['compare', str(lane / 'slucm-original'), str(receipt / f'slucm-original-{mode}.json')])
    result = json.loads((receipt / f'slucm-original-{mode}.json').read_text())
    assert sum(v['replay']['rows'] for v in result['variants'].values()) == 8856
    assert all(v[arm]['rows_differ'] == 0 for v in result['variants'].values()
               for arm in ('replay', 'carried')), result

    bep = tool(ROOT / "tools/urban_bep_wrf461_oracle/compare.py")
    add = bep.Tally.add
    def hashed_add(self, arm, field, got, want, label='', calm=None):
        g = np.ascontiguousarray(got, dtype=np.float32).reshape(-1)
        w = np.ascontiguousarray(want, dtype=np.float32).reshape(-1)
        keep = np.ones(w.shape, bool) if calm is None else \
            ~(np.ascontiguousarray(calm).reshape(-1) & ~np.isfinite(w))
        hashes.append({'scheme': 'bep', 'arm': arm, 'field': field, 'label': label,
                       'port_sha256': hashlib.sha256(g[keep].tobytes()).hexdigest(),
                       'wrf_sha256': hashlib.sha256(w[keep].tobytes()).hexdigest()})
        return add(self, arm, field, got, want, label, calm=calm)
    bep.Tally.add = hashed_add
    from gpuwm.core import urban_bep
    from _urban_calm_pbl import follow_calm_surface
    after_lsm = urban_bep.after_lsm
    calm_receipts = []
    def followed_after_lsm(state, params, **kwargs):
        result = after_lsm(state, params, **kwargs)
        atm, fields = kwargs['atmosphere'], kwargs['fields']
        mask = cp.asnumpy((state.frc_urb2d == cp.float32(1)) &
                         (atm['u'][0] == cp.float32(0)) &
                         (atm['v'][0] == cp.float32(0)))
        if mask.any():
            us = cp.asnumpy(fields['ust'])[mask]
            assert np.isfinite(us).all() and (us > 0).all(), us
            calm_receipts.append({'lsm': kwargs['lsm'], 'step': kwargs['itimestep'],
                                  'columns': int(mask.sum()), 'ust': us.tolist()})
            if kwargs['itimestep'] == 1:
                hf, qf = cp.asnumpy(fields['hfx'])[mask], cp.asnumpy(fields['qfx'])[mask]
                for u, h, q in zip(us, hf, qf):
                    follow_calm_surface(float(u), float(h), float(q))
        return result
    urban_bep.after_lsm = followed_after_lsm
    result = bep.run(lane / 'bep-extra/fixtures')
    (receipt / f'bep-{mode}.json').write_text(json.dumps(result, indent=1) + '\n')
    assert result['words'] == 13125154, result['words']
    assert result['bits_differ'] == result['max_ulp'] == result['calm_woof_nonfinite_words'] == 0, result['first_differences']
    assert sum(r['columns'] for r in calm_receipts) == 14 * 2 * 3, calm_receipts
    (receipt / f'calm-{mode}.json').write_text(json.dumps({
        'coupling': calm_receipts, 'pbl_steps_per_column': 3,
        'pbl_columns_per_scheme': 28, 'ysu_finite': True, 'mynn_finite': True}, indent=1) + '\n')

    bem = tool(ROOT / "tools/urban_bem_regime_oracle/compare.py")
    bem.HERE = lane / 'oracle-tools/urban_bem_regime_oracle'
    run_port = B.run_port
    def hashed_port(fx):
        steps = run_port(fx)
        nz = int(fx['nz'][0])
        for step, got in steps.items():
            assert isinstance(step, int), steps
            names = (*B.LEVEL_OUTPUTS, 'sf', *B.SURFACE_OUTPUTS,
                     *B.BEM_2D_OUTPUTS, *B.NDM_OUTPUTS)
            for name in names:
                want = B.to_gpuwm(fx[f'step{step}/out/{name}'])
                if name in B.LEVEL_OUTPUTS:
                    want = want[:nz]
                elif name == 'sf':
                    want = want[:nz + 1]
                hashes.append({'scheme': 'bem', 'step': step, 'field': name,
                    'port_sha256': hashlib.sha256(np.ascontiguousarray(got[name]).tobytes()).hexdigest(),
                    'wrf_sha256': hashlib.sha256(np.ascontiguousarray(want).tobytes()).hexdigest()})
            if step == int(fx['nsteps'][0]):
                for name in B.STATE_ARRAYS:
                    hashes.append({'scheme': 'bem', 'field': 'final/' + name,
                        'port_sha256': hashlib.sha256(np.ascontiguousarray(got[name]).tobytes()).hexdigest(),
                        'wrf_sha256': hashlib.sha256(np.ascontiguousarray(B.to_gpuwm(fx['final/' + name])).tobytes()).hexdigest()})
        return steps
    B.run_port = hashed_port
    bem.main(['compare', str(lane / 'bem-npz2'), str(receipt / f'bem-{mode}.json')])
    result = json.loads((receipt / f'bem-{mode}.json').read_text())
    assert result['total']['words'] == 3025318, result['total']
    assert result['total']['words_differ'] == result['total']['max_ulp'] == 0, result['total']
    selection = []
    for name in ('subnormal-lcz', 'subnormal-nlcd'):
        fx = B.load_fixture(name, lane / 'bem-npz2')
        frc = B.to_gpuwm(fx['state0/frc_urb2d']).ravel()
        host = np.flatnonzero(frc > 0).tolist()
        device = cp.asnumpy(ub.urban_columns(cp.asarray(frc))).tolist()
        selection.append({'fixture': name, 'host_selected': host, 'gpu_selected': device})
        assert host == device
    (receipt / f'selection-{mode}.json').write_text(json.dumps(selection, indent=1) + '\n')
    (receipt / f'field-hashes-{mode}.json').write_text(json.dumps(hashes, indent=1) + '\n')
    assert all(r['port_sha256'] == r['wrf_sha256'] for r in hashes)
    print(mode, 'all boundary corpora match bitwise')


if __name__ == '__main__':
    main()

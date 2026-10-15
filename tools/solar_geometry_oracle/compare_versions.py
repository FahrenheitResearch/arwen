"""Grade every solar word, and reject a planted coefficient bit error."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np


def grade(label, got, ref):
    a, b = np.asarray(got, np.float32), np.asarray(ref, np.float32)
    bad = np.flatnonzero(a.view('u4') != b.view('u4'))
    return {'label':label, 'words':a.size, 'differing_words':int(bad.size),
            'first_cells':[{'cell':int(i), 'got':hex(int(a.view('u4')[i])),
                            'ref':hex(int(b.view('u4')[i]))} for i in bad[:16]],
            'got_sha256':hashlib.sha256(a.tobytes()).hexdigest(),
            'ref_sha256':hashlib.sha256(b.tobytes()).hexdigest()}


def main():
    import cupy as cp
    from gpuwm.core.solar import solar_geometry_device
    from gpuwm.core.kernels import module_source, compile_noftz_module
    from gpuwm.core.rrtmg_legacy import _DEGRAD, _DPD
    root, out = map(Path,sys.argv[1:])
    rows, negative = [], []
    for gen, tag in [('wrf_461','v4.6.1'),('wrf_480','v4.8.0')]:
        fixture = np.load(root/('oracle-'+tag)/'solar_refs.npz')
        for k, (j,xt,g) in enumerate(fixture['scalars']):
            ds, cz, ha = solar_geometry_device(j,xt,g,fixture['lat'],fixture['lon'],generation=gen)
            got = np.concatenate([cp.asnumpy(ds),cp.asnumpy(cz).ravel(),cp.asnumpy(ha).ravel()])
            rows.append(grade(f'{gen}/{k}',got,fixture['ref'][k]))
        for k, (j,xt,g) in enumerate(fixture['polar_scalars']):
            ds, cz, ha = solar_geometry_device(j,xt,g,fixture['polar_lat'],fixture['polar_lon'],generation=gen)
            got = np.concatenate([cp.asnumpy(ds),cp.asnumpy(cz).ravel(),cp.asnumpy(ha).ravel()])
            rows.append(grade(f'{gen}/polar-midnight/{k}',got,fixture['polar_ref'][k]))
    # Compile a private poisoned image. Production source on disk stays intact.
    # Select one mantissa bit that survives the single-rounded output clock.
    # The least-significant coefficient bit rounds away in the ten planes.
    bit_mask = 1 << 20
    bits = int(np.asarray(0.040849,np.float32).view('u4')) ^ bit_mask
    src = module_source('rrtmg_legacy_solar').replace('0.040849f',f'__uint_as_float({bits}u)')
    assert src != module_source('rrtmg_legacy_solar')
    # The production no-FTZ route and options, on a private poisoned source.
    mod = compile_noftz_module('rrtmg_legacy_solar',src,'solar-planted-coefficient-bit')
    fixture = np.load(root/'oracle-v4.8.0'/'solar_refs.npz')
    lat, lon = cp.asarray(fixture['lat']),cp.asarray(fixture['lon'])
    n = lat.size
    ds, cz, ha = cp.empty(2,cp.float32),cp.empty_like(lat),cp.empty_like(lat)
    for k, (j,xt,g) in enumerate(fixture['scalars']):
        mod.get_function('rls_radconst')((1,),(1,),(j,_DEGRAD,_DPD,ds))
        mod.get_function('rls_calc_coszen_hrang')(((n+255)//256,),(256,),
            (np.int64(n),lat,lon,j,xt,g,_DEGRAD,np.int32(1),ds,cz,ha))
        got = np.concatenate([cp.asnumpy(ds),cp.asnumpy(cz).ravel(),cp.asnumpy(ha).ravel()])
        negative.append(grade(f'planted/{k}',got,fixture['ref'][k]))
    report = {'strict':os.environ.get('GPUWM_WRF_EXACT','0'),'rows':rows,
              'planted_xor_mask':hex(bit_mask), 'planted_coefficient_word':hex(bits),
              'negative':negative,'cupy':cp.__version__,'device':cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('words',sum(r['words'] for r in rows),'differing',sum(r['differing_words'] for r in rows),
          'planted differing',sum(r['differing_words'] for r in negative))
    assert not any(r['differing_words'] for r in rows), rows
    assert any(r['differing_words'] for r in negative), 'oracle did not reject planted one-bit coefficient error'

if __name__ == '__main__':
    main()

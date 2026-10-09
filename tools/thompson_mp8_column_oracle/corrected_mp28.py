#!/usr/bin/env python3
"""mp28 whole-column oracle with only MIN(idx_bg,dimNRHG) changed in WRF.

This closes the rain-meets-graupel coverage gap in the older stock-WRF
fixture gate. Builds and GPU checks must run on the sprint boxes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/thompson_aerosol_column_oracle'))
import oracle_io as io
import compare as comparison

PINNED = 'fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802'
OLD = 'idx_g1,idx_g,idx_bg(k),idx_r1,idx_r)'
NEW = 'idx_g1,idx_g,MIN(idx_bg(k),dimNRHG),idx_r1,idx_r)'


def corrected_source(raw):
    """Repair only the eight reads beyond the one rain-graupel table slab."""
    if hashlib.sha256(raw).hexdigest() != PINNED:
        raise ValueError('corrected RACG oracle requires the pinned pristine WRF v4.6.1 source')
    text = raw.decode()
    if text.count(OLD) != 8 or NEW in text:
        raise ValueError('corrected RACG oracle requires exactly eight pristine table reads')
    repaired = text.replace(OLD, NEW).encode()
    if repaired.decode().replace(NEW, OLD) != text:
        raise ValueError('corrected RACG oracle changed source outside the eight table reads')
    return repaired


def build(wrf, dest, tables):
    dest, wrf, tables = Path(dest).resolve(), Path(wrf).resolve(), Path(tables).resolve()
    target = dest / 'pristine'
    target.mkdir(parents=True, exist_ok=True)
    raw = (wrf / 'phys/module_mp_thompson.F').read_bytes()
    repaired = corrected_source(raw)
    (target / 'module_mp_thompson.F').write_bytes(repaired)
    flags = ['-O2', '-fno-tree-vectorize', '-ffree-form', '-ffree-line-length-none']
    sources = [ROOT / 'tools/thompson_wrf461_oracle/stub_wrf.F90',
               wrf / 'phys/module_mp_radar.F', target / 'module_mp_thompson.F',
               ROOT / 'tools/thompson_real_column_parity/run_columns_aero.F90']
    for source in sources:
        subprocess.run(['gfortran', '-c', *flags, '-cpp', '-DWRF_CHEM=0', str(source)],
                       cwd=target, check=True)
    objects = ['stub_wrf.o', 'module_mp_radar.o', 'module_mp_thompson.o', 'run_columns_aero.o']
    subprocess.run(['gfortran', *flags, '-o', 'run_columns_aero', *objects], cwd=target, check=True)
    (dest / 'run').mkdir(exist_ok=True)
    for name in ('qr_acr_qg_V4.dat', 'qr_acr_qsV2.dat', 'freezeH2O.dat', 'CCN_ACTIVATE.BIN'):
        link = dest / 'run' / name
        if link.is_symlink():
            assert link.resolve() == (tables / name).resolve(), link
        else:
            link.symlink_to(tables / name)
    receipt = {'stock_module_sha256': PINNED,
               'corrected_module_sha256': hashlib.sha256(repaired).hexdigest(),
               'old': OLD, 'new': NEW, 'replacement_count': 8,
               'compiler_flags': flags,
               'binary_sha256': hashlib.sha256((target / 'run_columns_aero').read_bytes()).hexdigest()}
    (dest / 'BUILD.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


def gpu_answers(out, dt):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gate = io.load(ROOT / 'tests/data/mp28_column_oracle_wrf461.npz')
    names = ('p', 'th', 'geop', 'w', 'nwfa2d', 'nifa2d') + io.SPECIES
    cols = {name: gate[name] for name in names}
    cols['labels'] = gate['labels']
    cols['regime'] = np.asarray(['column-gate'] * len(gate['labels']))
    columns, gpu_file = out / f'columns-{dt}.npz', out / f'gpu-{dt}.npz'
    np.savez(columns, **cols)
    subprocess.run([sys.executable, str(ROOT / 'tools/thompson_aerosol_column_oracle/gpu_run.py'),
                    str(columns), str(gpu_file), '--dt', str(dt)], check=True)
    gpu = io.load(gpu_file)
    deleted = [{'path': str(path), 'bytes': path.stat().st_size}
               for path in (columns, gpu_file)]
    for path in (columns, gpu_file):
        path.unlink()
    return cols, gpu, deleted


def field_hashes(data):
    return {name: hashlib.sha256(np.asarray(data[name], np.float32).tobytes()).hexdigest()
            for name in io.FIELDS}


def reference_fixture(build_dir, path):
    """Keep field hashes, never the WRF arrays, as the default GPU gate."""
    from gpuwm.core import constants as C
    fixture = ROOT / 'tests/data/mp28_column_oracle_wrf461.npz'
    gate = io.load(fixture)
    names = ('p', 'th', 'geop', 'w', 'nwfa2d', 'nifa2d') + io.SPECIES
    cols = {name: gate[name] for name in names}
    z = (np.zeros_like(cols['geop']) + cols['geop']) / np.float32(C.G)
    record = {'source': json.loads((Path(build_dir) / 'BUILD.json').read_text()),
              'input_fixture_sha256': hashlib.sha256(fixture.read_bytes()).hexdigest(),
              'columns': 157, 'levels': 49, 'fields': list(io.FIELDS), 'dts': {}}
    for dt in (5, 20):
        inputs = {'in_pii': gate[f'in_pii_dt{dt}'],
                  'in_dz': (z[:, 1:] - z[:, :-1]).copy(), 'in_hgt': z[:, :-1].copy()}
        wrf = comparison.wrf_answers(cols, inputs, build_dir, dt)
        record['dts'][str(dt)] = {'sha256': field_hashes(wrf),
                                 'words': sum(np.asarray(wrf[name]).size for name in io.FIELDS)}
    Path(path).write_text(json.dumps(record, indent=2) + '\n')
    return record


def check(build_dir, out, dt):
    cols, gpu, deleted = gpu_answers(out, dt)
    wrf = comparison.wrf_answers(cols, gpu, build_dir, dt)
    report = comparison.summarize(cols, gpu, wrf)
    meets = ((cols['qr'] > 1.e-12) & (cols['qg'] > 1.e-12)).any(axis=1)
    report['rain_meets_graupel_columns'] = int(meets.sum())
    report['rain_meets_graupel_differing_words'] = 0
    for name in io.FIELDS:
        a, b = np.asarray(gpu[name], np.float32), np.asarray(wrf[name], np.float32)
        report['rain_meets_graupel_differing_words'] += int(
            (a[meets].view(np.uint32) != b[meets].view(np.uint32)).sum())
    report['full_field_sha256'] = {
        side: field_hashes(data) for side, data in (('wrf', wrf), ('woof', gpu))}
    report['deleted'] = deleted
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build')
    b.add_argument('wrf', type=Path)
    b.add_argument('dest', type=Path)
    b.add_argument('tables', type=Path)
    c = sub.add_parser('check')
    c.add_argument('build', type=Path)
    c.add_argument('out', type=Path)
    c.add_argument('--dt', type=int, required=True)
    c.add_argument('--json', type=Path, required=True)
    f = sub.add_parser('fixture')
    f.add_argument('build', type=Path)
    f.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.action == 'build':
        build(args.wrf, args.dest, args.tables)
        return 0
    if args.action == 'fixture':
        reference_fixture(args.build, args.out)
        return 0
    report = check(args.build, args.out, args.dt)
    args.json.write_text(json.dumps(report, indent=2) + '\n')
    print(f"mp28 dt={args.dt}: {report['cells_differ']} of {report['cells']} words differ; "
          f"rain-graupel {report['rain_meets_graupel_differing_words']}")
    return int(report['cells_differ'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())

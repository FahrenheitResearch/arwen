"""Generate float32 boundary probes: make_subnormal_cases.py OUTDIR.

35 levels, dt 20 s, two steps. The optional .bin.step2 sidecar is read by
run_camuwpbl.F90 before recording step 2, after the step-1 reset. NaNs in
that sidecar mean retain the preceding output; they are never WRF inputs.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    'make_cases', Path(__file__).with_name('make_cases.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('outdir', type=Path)
    dest = parser.parse_args(argv).outdir
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(9301735)
    nk = 35
    values = np.array([1, 71362, 0x007fffff], np.uint32).view(np.float32)
    columns, metadata, overrides = [], [], []
    base = m.build_column(rng, 'convective', nk, 14000.)
    for field in ('qc', 'qi', 'qnc', 'qni', 'cldfra', 'wsedl3d'):
        base[field][:] = 0.

    def add(probe, changes=None, carry=None, family=None):
        c = {n: a.copy() for n, a in base.items()}
        if family:
            c = m.build_column(rng, family, nk, 14000.)
        details = {}
        for field, (levels, vals) in (changes or {}).items():
            c[field][levels] = vals
            details[field] = {'levels_1based': [int(k)+1 for k in levels],
                              'words_hex': [f'{int(w):08x}' for w in
                                            c[field][levels].view(np.uint32)]}
        columns.append(c)
        metadata.append({'column_1based': len(columns), 'probe': probe,
                         'inputs': details, 'step2_carried': carry or {}})
        overrides.append(carry or {})

    for family in ('convective', 'stable', 'stratocu'):
        add('control, no subnormal inputs', family=family)
    for field in ('qc', 'qi', 'qnc', 'qni', 'qv', 'cldfra',
                  'rthratenlw', 'wsedl3d'):
        levels = list(range(6)) if field in ('rthratenlw', 'wsedl3d') else [0, 2, 4]
        vals = np.concatenate([values, -values]) if len(levels) == 6 else values
        changes = {field: (levels, vals)}
        if field == 'qv':
            floor = np.float32(1.e-30)
            changes[field] = ([0, 2, 4, 6, 8, 10], np.concatenate([
                values, [np.nextafter(floor, np.float32(0)), floor,
                         np.nextafter(floor, np.float32(np.inf))]]))
        add('widen '+field, changes)
    for j, value in enumerate(values):
        add('signed surface flux magnitude '+str(j),
            {'hfx': ([0], [value]), 'qfx': ([0], [-value])})
        add('opposite surface flux signs magnitude '+str(j),
            {'hfx': ([0], [-value]), 'qfx': ([0], [value])})
    for field in ('kvm3d', 'kvh3d'):
        add('step 2 widen '+field, carry={field: {
            'levels_1based': [2, 4, 6],
            'words_hex': [f'{int(w):08x}' for w in values.view(np.uint32)]}})
    # Signed carried stresses cover all magnitudes in both directions.
    for j, magnitude in enumerate(values):
        value = magnitude if j == 1 else -magnitude
        add('step 2 signed carried stresses magnitude '+str(j), carry={
            'tauresx2d': {'levels_1based': [1], 'words_hex': [f'{int(value.view(np.uint32)):08x}']},
            'tauresy2d': {'levels_1based': [1], 'words_hex': [f'{int((-value).view(np.uint32)):08x}']}})
    for field in ('qc', 'qi', 'qni'):
        add('normal input, subnormal output '+field,
            {field: ([0, 2, 4, 18, 20, 22],
                     np.array([1.e-37]*3+[1.e-33]*3, np.float32))})
    add('signed wind widen and float32 multiply operands', {
        'u': ([0, 2, 4], values*np.array([-1, 1, -1], np.float32)),
        'v': ([6, 8, 10], values*np.array([1, -1, 1], np.float32))})
    add('float32 stress multiply operands', {'ust': ([0], [values[1]])})
    add('height widen and float32 subtraction operand', {'ht': ([0], [-values[2]])})
    path = dest/'cases-sub35.bin'
    m.write_case_file(path, columns, nk, 2, 20.)
    with open(str(path)+'.step2', 'wb') as stream:
        for field, width in (('kvm3d', nk+1), ('kvh3d', nk+1),
                             ('tauresx2d', 1), ('tauresy2d', 1)):
            a = np.full((len(columns), width), np.nan, dtype='<f4')
            for col, carry in enumerate(overrides):
                if field in carry:
                    entry = carry[field]
                    for level, word in zip(entry['levels_1based'], entry['words_hex']):
                        a.view(np.uint32)[col, level-1] = int(word, 16)
            a.tofile(stream)
    index = {'sub35': {'file': path.name, 'nk': nk, 'dt': 20., 'nsteps': 2,
                      'seed': 9301735, 'columns': metadata,
                      'subnormal_words_hex': [f'{int(w):08x}' for w in values.view(np.uint32)],
                      'sidecar': path.name+'.step2',
                      'note': 'Sidecar injects four carried fields before step 2; NaN retains previous output.'}}
    (dest/'cases-sub35-index.json').write_text(json.dumps(index, indent=2)+'\n')


if __name__ == '__main__':
    main()

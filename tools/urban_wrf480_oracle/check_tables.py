"""Word grades and a source mutation in the changed crown-normalization term."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gpuwm.core import urban_tables


def grade(root: Path, loader=urban_tables.load_urban_params, *, only480=False):
    result = {'cases': [], 'words': 0, 'differences': []}
    versions = [('v4.8.0', 'wrf_480')] if only480 else [
        ('v4.6.1', 'wrf_461'), ('v4.7.0', 'wrf_470'),
        ('v4.7.1', 'wrf_471'), ('v4.8.0', 'wrf_480')]
    for tag, generation in versions:
        for option in (1, 2, 3):
            for lcz in (0, 1):
                params = loader(option, lcz, generation=generation, tbl_dir=root/tag)
                csv_path = root/tag/f'opt{option}_lcz{lcz}.csv'
                actual, expected = [], []
                fields = set()
                with csv_path.open() as handle:
                    for name, index, word in csv.reader(handle):
                        field = np.asarray(params.values[name])
                        got = int(field.reshape(-1, order='F').view(np.uint32)[int(index)-1])
                        want = int(word, 16)
                        actual.append(got); expected.append(want); fields.add(name)
                        if got != want:
                            result['differences'].append({
                                'tag': tag, 'option': option, 'lcz': lcz,
                                'field': name, 'index_fortran': int(index),
                                'got': f'{got:08x}', 'want': word,
                                'xor': f'{got ^ want:08x}'})
                result['words'] += len(actual)
                result['cases'].append({
                    'tag': tag, 'generation': generation, 'option': option,
                    'lcz': lcz, 'words': len(actual), 'fields': len(fields),
                    'table_sha256': params.sha256,
                    'got_sha256': hashlib.sha256(np.asarray(actual, '<u4').tobytes()).hexdigest(),
                    'want_sha256': hashlib.sha256(np.asarray(expected, '<u4').tobytes()).hexdigest(),
                })
    return result


def plant(source: Path, dest: Path):
    text = source.read_text()
    needle = 'v[f"{name}_TBL"][lc] = f32(float(v[f"{name}_TBL"][lc]) / width)'
    assert text.count(needle) == 1, 'changed normalization term moved'
    replacement = needle + '\n' + '\n'.join([
        '                if name == "RTREE" and lc == 0:',
        '                    bits = np.float32(v[f"{name}_TBL"][lc]).view(np.uint32)',
        '                    v[f"{name}_TBL"][lc] = np.uint32(int(bits) ^ 1).view(np.float32)',
    ])
    dest.write_text(text.replace(needle, replacement))
    spec = importlib.util.spec_from_file_location('urban_tables_planted', dest)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.load_urban_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('oracle', type=Path)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--plant', type=Path)
    args = parser.parse_args()
    if args.plant:
        loader = plant(Path(urban_tables.__file__), args.plant)
        result = grade(args.oracle, loader, only480=True)
    else:
        result = grade(args.oracle)
    result['engine_table_source_sha256'] = hashlib.sha256(Path(urban_tables.__file__).read_bytes()).hexdigest()
    args.receipt.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'words': result['words'], 'differences': len(result['differences'])}))
    return bool(result['differences'])


if __name__ == '__main__':
    raise SystemExit(main())

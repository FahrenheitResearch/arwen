"""Summarize and compare complete history fields decoded by the Rust reader."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from gpuwm import netcdf_bridge

REQUIRED = ('U', 'V', 'W', 'T', 'P', 'QVAPOR', 'QCLOUD', 'QRAIN', 'QICE',
            'QSNOW', 'RAINNC', 'T2')

def summary(directory):
    frames = {}
    for path in sorted(directory.glob('wrfout*')):
        if not path.is_file():
            continue
        with netcdf_bridge.open_dataset(path) as dataset:
            fields = {}
            for name, variable in dataset.variables.items():
                if variable.dtype.kind not in 'fiu':
                    continue
                value = np.ascontiguousarray(variable[...], dtype=variable.dtype)
                finite = np.isfinite(value)
                fields[name] = {
                    'shape': list(value.shape), 'dtype': str(value.dtype),
                    'sha256': hashlib.sha256(value.tobytes()).hexdigest(),
                    'words': int(value.size), 'nonfinite': int((~finite).sum()),
                    'min': float(value[finite].min()) if finite.any() else None,
                    'max': float(value[finite].max()) if finite.any() else None,
                }
            assert all(n in fields for n in REQUIRED), sorted(set(REQUIRED) - fields.keys())
            assert all(fields[n]['nonfinite'] == 0 for n in REQUIRED), path
            frames[path.name] = {
                'file_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'file_bytes': path.stat().st_size, 'fields': fields}
    assert len(frames) >= 2, 'the CLI must write the initial and final frames'
    final = frames[sorted(frames)[-1]]['fields']
    # WRF native radar diagnostics are due after a microphysics call; the
    # initial frame may precede that call. The final frame must carry it.
    assert 'REFL_10CM' in final and final['REFL_10CM']['nonfinite'] == 0
    return frames

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = {'schema': 'wsm5-cli-history-receipt/v1', 'history': str(args.history),
              'frames': summary(args.history), 'required_finite_fields': list(REQUIRED)}
    if args.compare:
        other = summary(args.compare)
        mismatches = []
        count = words = 0
        assert report['frames'].keys() == other.keys()
        for frame, row in report['frames'].items():
            assert row['fields'].keys() == other[frame]['fields'].keys()
            for field, value in row['fields'].items():
                count += 1
                words += value['words']
                if value['sha256'] != other[frame]['fields'][field]['sha256']:
                    mismatches.append([frame, field])
        report['compare'] = {'history': str(args.compare), 'field_pairs': count,
                             'words': words, 'mismatches': mismatches}
        assert not mismatches, mismatches
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'frames': len(report['frames']), 'compare': report.get('compare')}))

if __name__ == '__main__':
    main()

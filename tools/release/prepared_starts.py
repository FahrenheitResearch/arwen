"""Refuse a repeatability comparison that starts from different prepared arrays.

Run before comparing cut/R11 forecast fields or pictures::

    python -m tools.release.prepared_starts CUT_PREP R11_PREP --out starts.json

The cut prepares without a card; R11's automatic backend may select CUDA.
The historical 2.9.9 moving-nest mismatch was traced to SINT contraction
and EOS contraction/log1p/pow differences. Preparation now pins those
operations across CPU and CUDA. The 00Z ICON-EU moving-nest regression
matches all 18,170,276 prepared values, including 10,136,778 child values.
This guard still checks the actual arrays for any pair of runs.

This checks every array declared by prepared-cache headers, with no numeric
tolerance. PASS establishes equality of those arrays only. The caller must
also fix the build, configuration, boundary stream, statics and runtime
options. Reusing one complete --prepared-root is preferable to preparing
twice. This tool neither changes preparation nor relaxes output comparisons.
"""
import argparse
import json
from pathlib import Path

from gpuwm.ingest.prepared_cache import read_manifest_array


def inventory(root):
    root = Path(root).resolve()
    result = {}
    for header in sorted(root.rglob('header.json')):
        document = json.loads(header.read_text(encoding='utf-8'))
        if document.get('schema') != 'gpuwm-prepared-real-cache-v1':
            continue
        for name, row in document['arrays'].items():
            key = f'{header.relative_to(root).as_posix()}:{name}'
            payload = (header.parent / row['file']).resolve()
            if not payload.is_relative_to(root):
                raise ValueError(f'prepared payload is outside its root: {key}')
            read_manifest_array(header.parent, key, row)
            result[key] = {field: row[field] for field in ('dtype', 'shape', 'sha256')}
    if not result:
        raise ValueError(f'no prepared arrays in {root}; cannot establish equal starts')
    return result


def compare_starts(left, right):
    a, b = inventory(left), inventory(right)
    different = [name for name in sorted(a.keys() | b.keys()) if a.get(name) != b.get(name)]
    return dict(
        schema='gpuwm-prepared-start-comparison-v1',
        status='REFUSE' if different else 'PASS',
        scope='prepared-cache arrays only',
        left=str(Path(left).resolve()), right=str(Path(right).resolve()),
        arrays=len(a.keys() | b.keys()), different=different,
        reason=('Prepared arrays differ before forecast step 1; output differences '
                'cannot establish forecast nondeterminism.' if different else
                'All declared prepared arrays are byte-identical.'),
        remedy='Run both forecasts from one --prepared-root with the same build and configuration.',
        left_arrays=a, right_arrays=b,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('left', type=Path)
    parser.add_argument('right', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = compare_starts(args.left, args.right)
    except (ValueError, OSError, KeyError) as error:
        result = dict(status='REFUSE', reason=str(error))
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"{result['status']}: {result['reason']}")
    for name in result.get('different', []):
        print(name)
    return 0 if result['status'] == 'PASS' else 2


if __name__ == '__main__':
    raise SystemExit(main())

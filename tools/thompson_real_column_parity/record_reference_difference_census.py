"""Record the difference between two independent WRF classic references."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

def census(root=ROOT):
    from tools.thompson_real_column_parity.mp8_wrf_ulp import LEVEL_FIELDS, SURFACE_FIELDS
    paths = [root / 'tests/data' / ('thompson_real_columns_wrf461_mp8' + suffix + '.npz')
             for suffix in ('', '_repaired')]
    from tools.thompson_real_column_parity.corrected_real_reference import input_words
    rows = []
    with np.load(paths[0], allow_pickle=False) as stock, np.load(paths[1], allow_pickle=False) as repaired:
        assert input_words(stock) == input_words(repaired), 'reference input words differ'
        for name in LEVEL_FIELDS + SURFACE_FIELDS:
            a, b = (np.asarray(z[f'wrf_{name}'], dtype=np.float32) for z in (stock, repaired))
            if name in ('re_cloud', 're_ice', 're_snow'):
                a, b = (x * np.float32(1e6) for x in (a, b))
            assert a.shape == b.shape, name
            count = int(np.count_nonzero(a.view(np.uint32) != b.view(np.uint32)))
            rows.append(dict(field=name, values=int(a.size), values_differing=count))
    return dict(schema='wrf461-independent-reference-difference-v1',
                inputs_byte_identical=True, reader='numpy float32 words; effective radii in microns',
                fixtures=[dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                          for p in paths], fields=rows, values_differing=sum(r['values_differing'] for r in rows))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    doc = census()
    if args.write:
        target = ROOT / 'tests/test_thompson_real_column_host_parity.py'
        text = target.read_text()
        text, n = re.subn(r'assert total == \d+, total', f"assert total == {doc['values_differing']}, total", text)
        assert n == 1, n
        text = re.sub(r'\(\d+ on this fixture:', f"({doc['values_differing']} on this fixture:", text)
        target.write_text(text)
    args.out.write_text(json.dumps(doc, indent=2) + '\n')
    print(json.dumps(dict(values_differing=doc['values_differing'], fields=len(doc['fields']))))

if __name__ == '__main__':
    main()

"""Exact little-endian uint32 comparison, including zeros and NaN payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

def compare(left, right, words):
    if words <= 0:
        raise ValueError('positive expected word count required')
    a, b = Path(left).read_bytes(), Path(right).read_bytes()
    if len(a) != words * 4 or len(b) != words * 4:
        raise ValueError('artifact size differs from expected word count')
    different, first = 0, None
    for n in range(words):
        av, bv = struct.unpack_from('<I', a, n * 4)[0], struct.unpack_from('<I', b, n * 4)[0]
        if av != bv:
            different += 1
            if first is None:
                first = {'word': n, 'reference': f'{av:08x}', 'candidate': f'{bv:08x}'}
    return {'words': words, 'mismatches': different, 'first': first,
            'reference_sha256': hashlib.sha256(a).hexdigest(),
            'candidate_sha256': hashlib.sha256(b).hexdigest()}

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('candidate', type=Path)
    p.add_argument('--words', type=int, required=True)
    a = p.parse_args(argv)
    try:
        result = compare(a.reference, a.candidate, a.words)
    except (OSError, ValueError) as exc:
        print(json.dumps({'error': str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return int(result['mismatches'] > 0)

if __name__ == '__main__':
    raise SystemExit(main())

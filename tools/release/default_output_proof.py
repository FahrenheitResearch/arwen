"""Compare all stored history fields by full hashes and unconditional byte reads."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from netCDF4 import Dataset


def _stored_words(variable):
    variable.set_auto_maskandscale(False)
    variable.set_auto_chartostring(False)
    values = np.ascontiguousarray(variable[:])
    if values.dtype.kind not in 'biufcS':
        raise TypeError(f'unsupported stored field dtype: {values.dtype}')
    # Swap bytes, never perform floating point conversion, on a big endian host.
    if values.dtype.byteorder == '>' or (values.dtype.byteorder == '=' and not np.little_endian):
        values = values.byteswap().view(values.dtype.newbyteorder('<'))
    return values


def _audit(values):
    result = {'values': int(values.size)}
    if values.dtype.kind == 'f':
        result.update(finite=int(np.isfinite(values).sum()), nan=int(np.isnan(values).sum()),
                      infinity=int(np.isinf(values).sum()),
                      negative_zero=int(((values == 0) & np.signbit(values)).sum()))
    return result


def compare_folders(base, tip, hash_bytes=None):
    base, tip = Path(base), Path(tip)
    hash_bytes = hash_bytes or (lambda data: hashlib.sha256(data).hexdigest())
    left = {p.name: p for p in base.glob('wrfout*') if p.is_file()}
    right = {p.name: p for p in tip.glob('wrfout*') if p.is_file()}
    if not left or not right:
        raise ValueError('both arms must contain history frames')
    rows, missing = [], []
    for frame in sorted(set(left) | set(right)):
        if frame not in left or frame not in right:
            missing.append(dict(frame=frame, base=frame in left, tip=frame in right))
            continue
        with Dataset(left[frame]) as x, Dataset(right[frame]) as y:
            for name in sorted(set(x.variables) | set(y.variables)):
                if name not in x.variables or name not in y.variables:
                    v = _stored_words((x if name in x.variables else y).variables[name])
                    payload = v.tobytes(order='C')
                    row = dict(frame=frame, field=name, base=name in x.variables, tip=name in y.variables,
                               dtype=v.dtype.str, shape=list(v.shape), present_sha256=hash_bytes(payload),
                               before='present' if name in x.variables else 'ABSENT',
                               after='present' if name in y.variables else 'ABSENT',
                               first_values=[str(value) for value in v.ravel()[:4]],
                               first_word_hex=[payload[i*v.dtype.itemsize:(i+1)*v.dtype.itemsize].hex()
                                               for i in range(min(4, v.size))],
                               byte_compare_performed=False, reason='field absent in one arm', **_audit(v))
                    if v.dtype.kind in 'fiu':
                        row.update(minimum=str(v.min()), maximum=str(v.max()))
                    missing.append(row)
                    continue
                a, b = _stored_words(x.variables[name]), _stored_words(y.variables[name])
                if a.dtype != b.dtype or a.shape != b.shape:
                    raise ValueError(f'{frame}/{name}: stored dtype or shape differs: '
                                     f'{a.dtype}/{a.shape} versus {b.dtype}/{b.shape}')
                ab, bb = a.tobytes(order='C'), b.tobytes(order='C')
                ha, hb = hash_bytes(ab), hash_bytes(bb)
                # Compare even when the hashes match. Signed zeros and NaN payloads participate.
                equal = ab == bb
                unequal = (np.frombuffer(ab, np.uint8).reshape(-1, a.dtype.itemsize) !=
                           np.frombuffer(bb, np.uint8).reshape(-1, b.dtype.itemsize)).any(axis=1)
                n = int(unequal.sum())
                assert equal == (n == 0)
                row = dict(frame=frame, field=name, shape=list(a.shape), dtype=a.dtype.str,
                           base_sha256=ha, tip_sha256=hb, byte_compare_performed=True,
                           byte_equal=equal, values_differing=n, base_audit=_audit(a),
                           tip_audit=_audit(b), values=int(a.size))
                if n:
                    row['examples'] = [dict(flat_index=int(i), base=str(a.ravel()[i]), tip=str(b.ravel()[i]),
                                            base_word_hex=ab[i*a.dtype.itemsize:(i+1)*a.dtype.itemsize].hex(),
                                            tip_word_hex=bb[i*b.dtype.itemsize:(i+1)*b.dtype.itemsize].hex())
                                       for i in np.flatnonzero(unequal)[:8]]
                rows.append(row)
    differences = sum(r['values_differing'] for r in rows)
    return dict(schema='r290-default-full-field-byte-compare-v2', base_path=str(base), tip_path=str(tip),
                fields=len(rows), values=sum(r['values'] for r in rows), values_differing=differences,
                missing=missing, rows=rows,
                reader='netCDF4.Dataset; stored dtype; mask, scale and character conversion disabled',
                verdict='IDENTICAL' if not differences and not missing else 'DIFFERENT')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('tip', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = compare_folders(args.base, args.tip)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'missing')}))
    print(json.dumps(dict(missing_fields_or_frames=len(result['missing']))))


if __name__ == '__main__':
    main()

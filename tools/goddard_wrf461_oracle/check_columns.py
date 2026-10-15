"""Mechanical full-word check for the WRF Goddard two-band column fixture.

This compares a separately produced candidate, never the oracle to itself.
SW outputs are fractions of the incident solar flux. LW outputs are W/m2.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

FIELDS = ('sw_net', 'sw_down', 'sw_up', 'sw_surface', 'lw_net', 'lw_down',
          'lw_up', 'sw_tau', 'sw_ssa', 'sw_asy', 'lw_tau', 'lw_ssa', 'lw_asy')
SCENARIOS = ('clear', 'overcast_low', 'deep_ice', 'thin_cirrus', 'polar_night',
             'low_sun', 'high_terrain', 'edge')

def read_columns(path):
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise ValueError('truncated column header')
    magic, count, levels, batches, kind = struct.unpack('<5i', raw[:20])
    if (magic, count, levels, batches) != (0x47524435, 64, 32, 4) or kind not in (32,64):
        raise ValueError('unexpected column contract')
    sizes = [16*33]*3 + [16*4] + [16*33]*3 + [16*32*11]*3 + [16*32*10]*3
    words = np.frombuffer(raw, dtype='<u4', offset=20)
    if words.size != batches * sum(sizes):
        raise ValueError(f'truncated or extra output: {words.size} words')
    result = {name: [] for name in FIELDS}
    at = 0
    for batch in range(batches):
        for name,size in zip(FIELDS, sizes):
            # Fortran arrays have the 16-column index contiguous.
            result[name].append(words[at:at+size].reshape((-1,16)).T.copy())
            at += size
    return {name:np.concatenate(values) for name,values in result.items()}, kind

def ordered_bits(bits):
    wide=bits.astype(np.int64)
    return np.where(wide & 0x80000000, 0x80000000-(wide & 0x7fffffff),
                    0x80000000+wide)

def compare_fields(reference, candidate, strict):
    if set(reference) != set(candidate):
        raise ValueError('candidate field set differs')
    receipt = {}
    passed = True
    for name,ref in reference.items():
        got=candidate[name]
        if ref.shape != got.shape:
            raise ValueError(f'{name}: candidate shape differs')
        a=ref.view('<f4').astype(np.float64)
        b=got.view('<f4').astype(np.float64)
        finite=np.isfinite(a)&np.isfinite(b)
        difference=np.abs(a-b)
        ulp=np.abs(ordered_bits(ref)-ordered_bits(got))
        # Default bounds are predeclared, not fitted to results.
        absolute = .1 if name.startswith('lw_') and name in FIELDS[4:7] else 1.e-4
        relative = 1.e-4
        mismatch=(ref != got) if strict else (difference > absolute+relative*np.abs(a))
        mismatch |= ~finite
        ok=not bool(np.any(mismatch))
        passed &= ok
        worst=tuple(int(v) for v in np.unravel_index(np.argmax(ulp),ulp.shape))
        receipt[name]={'pass':ok, 'words':int(ref.size),
          'different_words':int(np.count_nonzero(ref!=got)),
          'failed_words':int(np.count_nonzero(mismatch)),
          'max_ulp':int(np.max(ulp)), 'max_abs':float(np.max(difference)),
          'worst_ulp_column_word':worst, 'default_atol':absolute,'default_rtol':relative}
    return passed,receipt

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--oracle',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--strict',action='store_true')
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    report={'pass':False, 'strict':args.strict,'oracle':str(args.oracle),
            'candidate':str(args.candidate),'scenarios':SCENARIOS}
    try:
        ref,rkind=read_columns(args.oracle)
        got,ckind=read_columns(args.candidate)
        passed,fields=compare_fields(ref,got,args.strict)
        report.update(pass_=passed,fields=fields,oracle_solver_bits=rkind,
                      candidate_solver_bits=ckind,columns=64)
        report['pass']=report.pop('pass_')
        report['oracle_sha256']=hashlib.sha256(args.oracle.read_bytes()).hexdigest()
        report['candidate_sha256']=hashlib.sha256(args.candidate.read_bytes()).hexdigest()
    except (OSError,ValueError) as error:
        report['error']=str(error)
    args.receipt.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pass':report['pass'],'error':report.get('error')}))
    return 0 if report['pass'] else 1
if __name__=='__main__':
    raise SystemExit(main())

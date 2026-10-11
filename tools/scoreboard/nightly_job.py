"""Schedule and time the native job. Observation/weather processing stays in Rust."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--as-of',help='Frozen UTC cutoff; default is the current UTC midnight')
    parser.add_argument('--binary',type=Path,help='Native scorer, otherwise shipped binary resolution')
    args=parser.parse_args(argv)
    if args.binary is None:
        from gpuwm.scoreboard import BINARY
        args.binary=BINARY.find()
    if args.binary is None:
        parser.error('native scorer is unavailable')
    now=datetime.now(timezone.utc)
    cutoff=args.as_of or now.strftime('%Y-%m-%dT00:00:00Z')
    args.out.mkdir(parents=True,exist_ok=True)
    # Requests are outside the published directory. Different invocations never overwrite them.
    import tempfile
    fd,path=tempfile.mkstemp(prefix='nightly-request-',suffix='.json',dir=args.out.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump({'action':'nightly','catalog':str(args.catalog.resolve()),'as_of':cutoff,'output_dir':str(args.out.resolve())},f)
        start=time.perf_counter()
        result=subprocess.run([str(args.binary),'--request',path],capture_output=True)
        elapsed=time.perf_counter()-start
        # Operational telemetry is separate from byte-stable score publications.
        receipt={'schema':'woof.scoreboard.nightly-timing.v1','as_of':cutoff,'wall_seconds':elapsed,'exit_code':result.returncode,'gpu_seconds':0}
        if result.returncode==0:
            receipt['publication_sha256']=hashlib.sha256((args.out/'nightly.json').read_bytes()).hexdigest()
        print(json.dumps(receipt,sort_keys=True))
        if result.stderr:
            print(result.stderr.decode(errors='replace'),end='',file=__import__('sys').stderr)
        return result.returncode
    finally:
        Path(path).unlink(missing_ok=True)


if __name__=='__main__':
    raise SystemExit(main())

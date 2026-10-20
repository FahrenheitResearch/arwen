#!/usr/bin/env python3
"""Acquire verification radar and drive native fixed-Earth radar gridding."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess
import time

from gpuwm import rustwx
from gpuwm.obs.nexrad import find_nexrad_bin, run_fetch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--valid', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    end = datetime.fromisoformat(args.valid.replace('Z', '+00:00'))
    end = (end.replace(tzinfo=timezone.utc) if end.tzinfo is None else end.astimezone(timezone.utc))
    valid = end.strftime('%Y-%m-%dT%H:%M:%SZ')
    start = (end-timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%SZ')
    binary, renderer = find_nexrad_bin(), rustwx.find_renderer()
    if binary is None or renderer is None:
        raise RuntimeError('rw_nexrad and this source tree\'s rw_wrfbatch are required')
    before = time.perf_counter()
    def fetch(site):
        record = run_fetch(binary, site=site, start=start, end=valid,
            out=args.out/site, cache_dir=args.out/'cache')
        (args.out/f'fetch-{site}.json').write_text(json.dumps(record, indent=2)+'\n')
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, ('KTBW', 'KBYX', 'KAMX')))
    fetch_seconds = time.perf_counter()-before
    request = {'out': str(args.out/'observed'), 'volumes': [row['path'] for record in records for row in record['files']],
        'valid_time': valid, 'south': 22, 'north': 31, 'west': -87, 'east': -77,
        'nx': 371, 'ny': 401, 'spacing_m': 3000}
    path = args.out/'request.json'
    path.write_text(json.dumps(request, indent=2)+'\n')
    before = time.perf_counter()
    with (args.out/'grid.log').open('w') as log:
        subprocess.run([str(renderer), '--cyclone-radar', str(path)], stdout=log, stderr=subprocess.STDOUT, check=True)
    receipt = {'fetch_seconds': fetch_seconds, 'native_grid_seconds': time.perf_counter()-before,
        'parallelism': '3 concurrent archive requests; native decode/grid across available cores',
        'frame': str(args.out/'observed/frame.json')}
    (args.out/'timings.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()

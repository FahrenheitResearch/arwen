#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run actual Rust CSV/NetCDF humidity preparation on literal synthetic data.

No host numerical transformation: ncgen/CSV fixtures declare exact r=1,3,7,15
and q=.5,.75,.875,.9375. Rust owns scaling, conversion and area remapping.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', required=True, type=Path)
    parser.add_argument('--scratch', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = args.scratch.resolve()
    root.mkdir(parents=True, exist_ok=False)
    table = Path(__file__).resolve().parents[2] / 'gpuwm/data/hydro/forcing-sources.v1.json'
    source = json.loads(table.read_text())['profiles']['decoded-canonical-grid']
    profiles = {}
    for label, format_, humidity, scale in (
            ('canonical-grid', 'grid_netcdf', 'specific_humidity', 1),
            ('mixing-grid', 'grid_netcdf', 'mixing_ratio', .001),
            ('canonical-csv', 'catchment_csv', 'specific_humidity', 1),
            ('mixing-csv', 'catchment_csv', 'mixing_ratio', .001),
            ('invalid-mode', 'grid_netcdf', 'relative_humidity_typo', 1)):
        row = json.loads(json.dumps(source))
        row.update(format=format_, humidity=humidity)
        row['fields']['Q2D'] = dict(variable='humidity_native', scale=scale)
        profiles[label] = row
    profile_path = root / 'arbitrary-table.json'
    profile_path.write_text(json.dumps(dict(profiles=profiles), indent=2) + '\n')
    weights = root / 'weights.json'
    weights.write_text(json.dumps(dict(catchment_ids=['catchment-1'], grid_shape=[2],
                                      offsets=[0, 2], indices=[0, 1], weights=[.25, .75])) + '\n')
    fields = ('T2D', 'humidity_native', 'U2D', 'V2D', 'PSFC', 'LWDOWN', 'SWDOWN', 'RAINRATE')
    generated = []
    for label, humid in (('canonical', '.5,.75,.875,.9375'),
                         ('mixing', '1000,3000,7000,15000'),
                         ('negative', '1000,-3000,7000,15000')):
        # Time is last. Rust must normalize it, scale g/kg to kg/kg, convert
        # each cell, then average. Converting an area-mean r gives another q.
        cdl = root / (label + '.cdl')
        declarations = ['double time(time); time:units="hours since 2015-12-01 00:00:00";']
        declarations += [f'double {name}(x,time);' for name in fields]
        assignments = ['time=0,1;', f'humidity_native={humid};']
        for field, literal in (('T2D', '300'), ('U2D', '-1'), ('V2D', '2'),
                               ('PSFC', '100000'), ('LWDOWN', '350'),
                               ('SWDOWN', '200'), ('RAINRATE', '.001')):
            assignments.append(f'{field}=' + ','.join([literal] * 4) + ';')
        cdl.write_text('netcdf sample {\ndimensions: time=2; x=2;\nvariables:\n'
                       + '\n'.join(declarations) + '\ndata:\n' + '\n'.join(assignments) + '\n}\n')
        argv = ['ncgen', '-4', '-o', str(root / (label + '.nc')), str(cdl)]
        start = time.perf_counter()
        subprocess.run(argv, capture_output=True, check=True)
        generated.append(dict(argv=argv, seconds=time.perf_counter() - start))
    header = 'time,' + ','.join(fields) + '\n'
    for label, q0, q1 in (('canonical', '.5', '.75'), ('mixing', '1000', '3000'),
                           ('negative', '1000', '-3000')):
        (root / (label + '.csv')).write_text(header
            + f'2015-12-01 00:00:00,300,{q0},-1,2,100000,350,200,.001\n'
            + f'2015-12-01 01:00:00,300,{q1},-1,2,100000,350,200,.001\n')
    results = []
    for label, row, filename, expected_error in (
            ('reference-grid', 'canonical-grid', 'canonical.nc', None),
            ('mixing-grid', 'mixing-grid', 'mixing.nc', None),
            ('negative-grid', 'mixing-grid', 'negative.nc', 'negative humidity mixing ratio'),
            ('invalid-mode', 'invalid-mode', 'canonical.nc', 'unknown variant'),
            ('reference-csv', 'canonical-csv', 'canonical.csv', None),
            ('mixing-csv', 'mixing-csv', 'mixing.csv', None),
            ('negative-csv', 'mixing-csv', 'negative.csv', 'negative humidity mixing ratio')):
        output = root / (label + '-packet')
        request = dict(source_profile=str(profile_path), profile_row=row,
                       inputs=[str(root / filename)], output=str(output),
                       time_origin='2015-12-01T00:00:00Z', interval_seconds=3600,
                       catchment_ids=['catchment-1'])
        if filename.endswith('.nc'):
            request['area_weights'] = str(weights)
        path = root / (label + '-request.json')
        path.write_text(json.dumps(request, indent=2) + '\n')
        argv = [str(args.executable.resolve()), str(path)]
        start = time.perf_counter()
        run = subprocess.run(argv, capture_output=True, text=True)
        result = dict(case=label, argv=argv, seconds=time.perf_counter() - start,
                      exit_code=run.returncode, stderr=run.stderr)
        if run.returncode == 0:
            receipt = json.loads(run.stdout)
            result.update(shape=receipt['shape'], data_sha256=receipt['data_sha256'],
                          preparation_seconds=receipt['preparation_seconds'],
                          parallelism=receipt['parallelism'])
        result['pass'] = (run.returncode == 0 if expected_error is None else
                          run.returncode != 0 and expected_error in run.stderr and not output.exists())
        results.append(result)
    by_name = {r['case']: r for r in results}
    exact = all(by_name['reference-' + format_].get('data_sha256') is not None and
                by_name['reference-' + format_].get('data_sha256') == by_name['mixing-' + format_].get('data_sha256')
                for format_ in ('grid', 'csv'))
    receipt = dict(scope='synthetic Rust input conversion; not native/GPU physics proof',
                   executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                   fixture_generation=generated, results=results,
                   per_cell_before_csr_and_csv_packet_hashes_exact=exact,
                   passed=exact and all(r['pass'] for r in results))
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(passed=receipt['passed'], results=results), indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

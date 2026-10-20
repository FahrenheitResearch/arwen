#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual Rust wind-basis preparation with literal analytic fixtures.

Opposite grid vectors at opposite quarter-turns both point earth north.
Their earth east components cancel exactly before averaging. No Python
numerical processing or new comparator: compare the typed packet hashes.
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
    parser.add_argument('--netcdf-format', choices=('classic', 'netcdf4'), default='netcdf4')
    args = parser.parse_args()
    root = args.scratch.resolve()
    root.mkdir(parents=True, exist_ok=False)
    table = Path(__file__).resolve().parents[2] / 'gpuwm/data/hydro/forcing-sources.v1.json'
    source = json.loads(table.read_text())['profiles']['decoded-canonical-grid']
    profiles = {}
    for name, format_, wind in (
            ('earth', 'grid_netcdf', None),
            ('rotated', 'grid_netcdf', dict(frame='grid_relative', angle=dict(
                variable='basis_degrees', scale=0.017453292519943295))),
            ('wrong-mode', 'grid_netcdf', dict(frame='unknown')),
            ('missing-angle', 'grid_netcdf', dict(frame='grid_relative')),
            ('wrong-temporal', 'grid_netcdf', dict(frame='grid_relative', angle=dict(
                variable='basis_degrees', temporal='interval_total'))),
            ('csv-earth', 'catchment_csv', None),
            ('csv-rotated', 'catchment_csv', dict(frame='grid_relative', angle=dict(
                variable='native_basis', scale=0.5, offset=-1)))):
        row = json.loads(json.dumps(source))
        row['format'] = format_
        if wind is not None:
            row['wind'] = wind
        # Source labels/units need no source-specific code.
        row['fields']['U2D'] = dict(variable='native_u', scale=2)
        row['fields']['V2D'] = dict(variable='native_v', scale=2)
        profiles[name] = row
    profile_path = root / 'arbitrary-table.json'
    profile_path.write_text(json.dumps(dict(profiles=profiles), indent=2) + '\n')
    weights = root / 'weights.json'
    weights.write_text(json.dumps(dict(catchment_ids=['catchment-1'], grid_shape=[2],
        offsets=[0, 2], indices=[0, 1], weights=[.5, .5])) + '\n')
    fields = ('T2D', 'Q2D', 'native_u', 'native_v', 'PSFC', 'LWDOWN', 'SWDOWN', 'RAINRATE')
    fixtures = (
        ('earth-dynamic', 'x,time', '0,0,0,0', '.5,-1,.5,-1', None, None),
        ('earth-static', 'x,time', '0,0,0,0', '.5,1,.5,1', None, None),
        ('rotated-dynamic', 'x,time', '.5,1,-.5,-1', '0,0,0,0', 'x,time', '90,-90,-90,90'),
        ('rotated-first', 'time,x', '.5,-.5,1,-1', '0,0,0,0', 'time,x', '90,-90,-90,90'),
        ('rotated-static', 'x,time', '.5,1,-.5,-1', '0,0,0,0', 'x', '90,-90'),
        ('invalid-shape', 'x,time', '.5,1,-.5,-1', '0,0,0,0', 'wrong', '90'),
        ('invalid-fill', 'x,time', '.5,1,-.5,-1', '0,0,0,0', 'x', '90,_'),
        ('invalid-missing', 'x,time', '.5,1,-.5,-1', '0,0,0,0', None, None),
    )
    generated = []
    for label, axes, u, v, angle_axes, angles in fixtures:
        declarations = ['double time(time); time:units="hours since 2015-12-01 00:00:00";']
        declarations += [f'double {name}({axes});' for name in fields]
        assignments = ['time=0,1;', f'native_u={u};', f'native_v={v};']
        if angle_axes:
            declarations.append(f'double basis_degrees({angle_axes});')
            assignments.append(f'basis_degrees={angles};')
        for name, literal in (('T2D', '300'), ('Q2D', '.01'), ('PSFC', '100000'),
                              ('LWDOWN', '350'), ('SWDOWN', '200'), ('RAINRATE', '.001')):
            assignments.append(f'{name}=' + ','.join([literal] * 4) + ';')
        cdl = root / (label + '.cdl')
        cdl.write_text('netcdf sample {\ndimensions: time=2; x=2; wrong=1;\nvariables:\n'
            + '\n'.join(declarations) + '\ndata:\n' + '\n'.join(assignments) + '\n}\n')
        argv = ['ncgen', '-4' if args.netcdf_format == 'netcdf4' else '-3',
                '-o', str(root / (label + '.nc')), str(cdl)]
        start = time.perf_counter()
        subprocess.run(argv, capture_output=True, check=True)
        generated.append(dict(argv=argv, seconds=time.perf_counter() - start))
    header = 'time,' + ','.join(fields)
    for label, extra in (('earth', ''), ('zero-angle', ',native_basis'), ('missing-angle', '')):
        # 2*.5-1=0 radians: verify CSV scale/offset and column binding without
        # a platform libm-dependent near-zero golden floating-point word.
        suffix = ',2' if extra else ''
        (root / (label + '.csv')).write_text(header + extra + '\n'
            + f'2015-12-01 00:00:00,300,.01,1,-2,100000,350,200,.001{suffix}\n'
            + f'2015-12-01 01:00:00,300,.01,2,-3,100000,350,200,.001{suffix}\n')
    results = []
    for label, row, filename, error in (
            ('earth-dynamic', 'earth', 'earth-dynamic.nc', None),
            ('earth-static', 'earth', 'earth-static.nc', None),
            ('dynamic', 'rotated', 'rotated-dynamic.nc', None),
            ('first-axis', 'rotated', 'rotated-first.nc', None),
            ('static', 'rotated', 'rotated-static.nc', None),
            ('wrong-shape', 'rotated', 'invalid-shape.nc', 'grid shape differs'),
            ('missing-values', 'rotated', 'invalid-fill.nc', 'missing/nonfinite forcing cell'),
            ('missing-variable', 'rotated', 'invalid-missing.nc', 'basis_degrees'),
            ('wrong-mode', 'wrong-mode', 'earth-dynamic.nc', 'unknown variant'),
            ('missing-angle', 'missing-angle', 'earth-dynamic.nc', 'missing field'),
            ('wrong-temporal', 'wrong-temporal', 'earth-dynamic.nc', 'wind angle must be instantaneous'),
            ('csv-earth', 'csv-earth', 'earth.csv', None),
            ('csv-zero', 'csv-rotated', 'zero-angle.csv', None),
            ('csv-missing', 'csv-rotated', 'missing-angle.csv', 'missing CSV wind angle')):
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
                preparation_seconds=receipt['preparation_seconds'], parallelism=receipt['parallelism'])
        result['pass'] = (run.returncode == 0 if error is None else
            run.returncode != 0 and error in run.stderr and not output.exists())
        results.append(result)
    by_name = {r['case']: r for r in results}
    hashes_exact = all(by_name[a].get('data_sha256') is not None and
        by_name[a].get('data_sha256') == by_name[b].get('data_sha256') for a, b in (
            ('earth-dynamic', 'dynamic'), ('earth-dynamic', 'first-axis'),
            ('earth-static', 'static'), ('csv-earth', 'csv-zero')))
    receipt = dict(scope='literal analytic Rust transport regression; not native/GPU physics proof',
        executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        fixture_generation=generated, results=results,
        per_cell_before_csr_and_csv_packet_hashes_exact=hashes_exact,
        passed=hashes_exact and all(r['pass'] for r in results))
    args.out.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

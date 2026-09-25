"""Compose registered observation doors into immutable per-cycle inputs.

No network protocol or meteorological decoder is implemented here. A frozen
receipt is reused on recovery; late fetches cannot silently replace the
observations from a partially completed analysis.
"""
from __future__ import annotations

from datetime import timedelta
import contextlib
import json
from pathlib import Path
import sys

from gpuwm.local_da import PlanError
from gpuwm.obs.goes_cwp_policy import utc

WINDOW_SCHEMA = 'arwen.local-da-observation-window.v1'


def _stamp(value):
    return value.strftime('%Y-%m-%dT%H:%M:%SZ')


def assigned_document(document, when, schedule, max_age):
    """Assign a pregridded observation to exactly one causal cycle.

    Multi-radar products also disclose each contributing scan instant.
    A future scan rejects the whole product, not merely its filename time.
    """
    times = [utc(document['valid_time'])]
    provenance = document.get('provenance') or {}
    if document.get('schema') == 'gpuwm-obs.goes-grid.v1':
        from gpuwm.obs.goes_window import ACQUISITION_SCHEMA
        acquisition = provenance.get('acquisition')
        if acquisition is not None:
            if acquisition.get('schema') != ACQUISITION_SCHEMA:
                raise ValueError('The CWP acquisition record has an unknown schema')
            if utc(acquisition['assigned_analysis_time']) != when:
                return False
            cutoff = utc(acquisition['information_cutoff_utc'])
            publication_cutoff = utc(acquisition['publication_cutoff_utc'])
            if publication_cutoff > cutoff:
                raise ValueError('The CWP publication cutoff is after its information cutoff')
            scans = acquisition['scans']
            if not scans:
                return False
            for scan in scans:
                start, end = utc(scan['scan_start']), utc(scan['scan_end'])
                if start > end:
                    raise ValueError('The CWP source has a reversed scan interval')
                optical_end = utc(scan.get('optical_scan_end', scan['scan_end']))
                if not start <= optical_end <= end:
                    raise ValueError('The CWP optical interval contradicts its complete interval')
                if end > when or not 0 <= (when - optical_end).total_seconds() <= acquisition['max_age_seconds']:
                    return False
                if scan.get('publication_utc') and utc(scan['publication_utc']) > publication_cutoff:
                    return False
                for source in scan['source_files']:
                    if source.get('publication_utc') and utc(source['publication_utc']) > publication_cutoff:
                        return False
                    if utc(source['available_on_disk_utc']) > cutoff:
                        return False
            return times[0] == when
        # Older explicit products keep their fixed schedule assignment, but
        # a nominal valid_time cannot conceal future source measurements.
        pack = provenance.get('pack')
        if pack is not None:
            start, end = utc(pack['scan_start']), utc(pack['scan_end'])
            if start > end:
                raise ValueError('The CWP pack has a reversed scan interval')
            times.append(end)
    for radar in provenance.get('per_radar', ()):
        # A volume's last radial is the instant its last gate was measured;
        # a product whose radar was still scanning at the analysis time
        # holds the future.  The header start stands in only for a product
        # built before the pack carried the end.
        if radar.get('volume_end_time'):
            times.append(utc(radar['volume_end_time']))
        elif radar.get('volume_valid_time'):
            times.append(utc(radar['volume_valid_time']))
    eligible = [t for t in schedule if all(0 <= (t - observed).total_seconds() <= max_age for observed in times)]
    return bool(eligible and eligible[0] == when)


def choose_document(paths, reader, *, grid, when, schedule, max_age):
    choices, reports = [], []
    for path in paths:
        document = reader(path, expected_grid=grid)
        keep = assigned_document(document, when, schedule, max_age)
        reports.append(dict(path=str(path), valid_time=document.get('valid_time'),
                            status='eligible' if keep else 'outside_assigned_window'))
        if keep:
            choices.append((utc(document['valid_time']), str(path), document))
    if not choices:
        return None, reports
    # These are alternative complete gridded products, not pieces of a
    # volume. Joining them would duplicate gates and erase beam identities.
    choices.sort(key=lambda item: (item[0], item[1]))
    selected = choices[-1]
    for report in reports:
        if report['status'] == 'eligible':
            report['status'] = 'selected' if report['path'] == selected[1] else 'superseded_product'
    return Path(selected[1]), reports


def _surface_fetch(route, directory, when, cadence):
    from gpuwm.obs.frontdoor import ASOS
    from gpuwm.ensemble.manifest import write_json_atomically
    records = []
    for index, box in enumerate(route['boxes']):
        out = directory / f'part_{index:02d}'
        out.mkdir(parents=True, exist_ok=True)
        stations, observations, record = out / 'stations.json', out / 'observations.csv', out / 'surface.json'
        frozen = ASOS.run('stations', ['--networks', ','.join(route['networks']),
            '--bbox', ','.join(str(v) for v in box), '--out', str(stations)], schema='gpuwm-obs.asos-stations.v1')
        # Do not fetch after the analysis time. The decoder records each
        # report's own observation_time beside the slot it serves; when the
        # archive received the report is still not known here.
        fetched = ASOS.run('fetch', ['--stations', str(stations), '--start', _stamp(when - timedelta(seconds=cadence)),
            '--end', _stamp(when), '--out', str(observations)], schema='gpuwm-obs.asos-fetch.v1')
        decoded = ASOS.run('decode', ['--stations', str(stations), '--obs', str(observations),
            '--start', _stamp(when), '--end', _stamp(when), '--step-hours', '1',
            '--min-report-rate', '0', '--out', str(record)], schema='gpuwm-obs.asos-surface.v2')
        write_json_atomically(out / 'fetch-receipt.json', dict(stations=frozen, fetched=fetched,
            decoded=decoded, time_policy='past-only fetch; each report carries its observation_time beside the slot it serves; arrival at the archive unverified'))
        if not record.is_file():
            raise RuntimeError('The registered surface decoder produced no record; repair its output before retrying the window.')
        records.append(record)
    return records


def _radar_fetch(backend, route, directory, when):
    from tools.obs_radar_grid_build import main
    from gpuwm.obs.radar_grid import read_radar_grid
    from gpuwm.obs.radar_source import RadarSourceError
    out = directory / 'radar.nc'
    args = ['--valid-time', _stamp(when), '--grid-wrfout', str(backend.reference_path),
            '--out', str(out), '--work-dir', str(directory / 'raw'), '--allow-partial',
            '--max-range-km', str(route['max_range_km'])]
    for site in route['sites']:
        args.extend(['--site', site['id']])
    if not route['sites']:
        return None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            code = main(args)
    except (SystemExit, RadarSourceError) as exc:
        raise RuntimeError('The registered radar window builder could not supply this cycle: ' + str(exc)) from exc
    if code not in (0, None) or not out.is_file():
        raise RuntimeError('The registered radar builder produced no complete grid; retry a window with available scans.')
    document = read_radar_grid(out, expected_grid=backend.grid)
    if not assigned_document(document, when, backend.analysis_times, backend.plan['selected']['cadence_seconds']):
        raise RuntimeError('The selected radar product contains a future or previously assigned scan; this cycle skips it rather than moving its time.')
    return out


def _used_cwp_scans(backend, cycle_index, when, max_age):
    """Prior committed windows own consumption, so there is no second ledger."""
    from gpuwm.local_da_runtime import _sha
    from gpuwm.obs.goes_window import ACQUISITION_SCHEMA
    used = set()
    for index in range(cycle_index - 1, -1, -1):
        path = backend.root / 'observations' / f'cycle_{index:03d}' / 'window.json'
        if not path.is_file():
            continue
        receipt = json.loads(path.read_text())
        if (receipt.get('schema') != WINDOW_SCHEMA
                or receipt.get('review_sha256') != backend.plan['review_sha256']
                or receipt.get('grid_identity') != backend.grid.identity_sha256()):
            raise PlanError('A prior CWP window belongs to another review or grid.', code='OBSERVATION_WINDOW_CHANGED')
        if (when - utc(receipt['analysis_time'])).total_seconds() > max_age:
            break
        for record in receipt['assets']:
            if record['kind'] != 'cwp-acquisition':
                continue
            if _sha(record['path']) != record['sha256']:
                raise PlanError('A prior CWP acquisition record changed; restore the frozen bytes.', code='OBSERVATION_WINDOW_CHANGED')
            acquisition = json.loads(Path(record['path']).read_text())
            if acquisition.get('schema') != ACQUISITION_SCHEMA:
                raise PlanError('A prior CWP acquisition schema changed.', code='OBSERVATION_WINDOW_CHANGED')
            used.update(acquisition['consumed_scan_ids'])
    return used


def _cwp_fetch(backend, route, directory, when, cycle_index):
    from gpuwm.obs.goes_window import AcquisitionPolicy, NativeGoes, acquire_window
    policy = AcquisitionPolicy.from_payload(route['acquisition_policy'])
    used = _used_cwp_scans(backend, cycle_index, when, policy.max_age_seconds)
    native = NativeGoes(route['binary'], expected_sha256=route.get('binary_sha256'))
    return acquire_window(grid=backend.grid, when=when, directory=directory,
        cache=backend.root / 'observations' / '.goes-cache', native=native,
        policy=policy, used_scan_ids=used)


def _load_frozen(backend, receipt, when):
    from gpuwm.local_da_runtime import _sha
    from gpuwm.da.obs_point import read_tables
    from gpuwm.da.obs_surface import read_record
    from gpuwm.obs.radar_grid import read_radar_grid
    from gpuwm.obs.goes_grid import read_goes_grid
    if receipt.get('schema') != WINDOW_SCHEMA or receipt.get('review_sha256') != backend.plan['review_sha256'] or receipt.get('analysis_time') != _stamp(when) or receipt.get('grid_identity') != backend.grid.identity_sha256():
        raise PlanError('The saved observation window belongs to a different review, clock or grid; restore the matching receipt before resuming.', code='OBSERVATION_WINDOW_CHANGED')
    assets = receipt['assets']
    for asset in assets:
        if _sha(asset['path']) != asset['sha256']:
            raise PlanError('Frozen observation bytes changed at ' + asset['path'] + '; restore those bytes before resuming.', code='OBSERVATION_WINDOW_CHANGED')
    rows, table_records = read_tables([a['path'] for a in assets if a['kind'] == 'table']) if any(a['kind'] == 'table' for a in assets) else ([], [])
    surfaces = [read_record(a['path']) for a in assets if a['kind'] == 'surface']
    radar_paths = [a['path'] for a in assets if a['kind'] == 'radar']
    cwp_paths = [a['path'] for a in assets if a['kind'] == 'cwp']
    return dict(rows=rows, surface=surfaces,
                radar=read_radar_grid(radar_paths[0], expected_grid=backend.grid) if radar_paths else None,
                cwp=read_goes_grid(cwp_paths[0], expected_grid=backend.grid) if cwp_paths else None,
                receipts=[*receipt['routes'], dict(frozen_window=receipt, tables=table_records)])


def observation_window(backend, cycle_index, when, member_states):
    """Fetch only reviewed routes, freeze all inputs, then use owner readers."""
    from gpuwm.local_da_runtime import _sha, _atomic
    from gpuwm.obs.radar_grid import read_radar_grid
    from gpuwm.obs.goes_grid import read_goes_grid
    directory = backend.root / 'observations' / f'cycle_{cycle_index:03d}'
    path = directory / 'window.json'
    if path.is_file():
        return _load_frozen(backend, json.loads(path.read_text()), when)
    directory.mkdir(parents=True, exist_ok=True)
    assets, reports = [], []
    def add(kind, filename):
        filename = Path(filename).resolve()
        assets.append(dict(kind=kind, path=str(filename), sha256=_sha(filename)))
    for name in backend.plan['request']['obs_tables']:
        add('table', name)
    for route in backend.plan['observations']:
        report = dict(id=route['id'], status=route['status'], reason=route.get('reason', ''))
        reports.append(report)
        if route['status'] not in ('ready', 'candidate'):
            continue
        if (route['route'] == 'cloud-water-path' and not route.get('files')
                and getattr(backend, 'cwp_unavailable_reason', None)):
            report.update(status='unavailable', reason=backend.cwp_unavailable_reason)
            continue
        try:
            if route['route'] in ('radar-grid', 'cloud-water-path'):
                kind = 'radar' if route['route'] == 'radar-grid' else 'cwp'
                reader = read_radar_grid if kind == 'radar' else read_goes_grid
                if route.get('files'):
                    selected, decisions = choose_document(route['files'], reader, grid=backend.grid,
                        when=when, schedule=backend.analysis_times, max_age=backend.plan['selected']['cadence_seconds'])
                    report['selection'] = decisions
                elif kind == 'radar':
                    selected = _radar_fetch(backend, route, directory / route['id'], when)
                else:
                    acquired = _cwp_fetch(backend, route, directory / route['id'], when, cycle_index)
                    selected = acquired.path
                    report['acquisition'] = str(acquired.manifest_path)
                    report['acquisition_status'] = acquired.receipt['status']
                    report['observed_columns'] = acquired.receipt['observed_columns']
                    report['latency_class'] = acquired.receipt['latency_class']
                    report['wall_seconds'] = acquired.receipt['wall_seconds']
                    report['consumed_scan_ids'] = acquired.receipt['consumed_scan_ids']
                    for record in acquired.assets:
                        if selected is not None and Path(record['path']) == selected:
                            continue
                        kind_aux = 'cwp-acquisition' if Path(record['path']) == acquired.manifest_path else 'cwp-source'
                        add(kind_aux, record['path'])
                if selected is not None:
                    add(kind, selected)
                    report.update(status='fetched', path=str(selected))
                else:
                    status = report.get('acquisition_status', 'empty')
                    report.update(status=status, reason='Automatic CWP sources failed; see the acquisition receipt.'
                        if status == 'unavailable' else 'No registered product belongs to this analysis window.')
            elif route['route'] == 'surface-v1':
                paths = _surface_fetch(route, directory / route['id'], when, backend.plan['selected']['cadence_seconds'])
                for filename in paths:
                    add('surface', filename)
                report.update(status='fetched', files=[str(p) for p in paths], latency='unverified')
            elif route['route'] == 'neutral-stream':
                from arwen_global.obs_streams import fetch_stream
                from gpuwm.local_da_observations import registry_inspection
                with registry_inspection():
                    fetched = fetch_stream(route['id'], _stamp(when - timedelta(seconds=backend.plan['selected']['cadence_seconds'])),
                        _stamp(when), directory, networks=','.join(route.get('networks', ())) or None)
                add('table', fetched.table)
                report.update(status='fetched', manifest=fetched.manifest)
        except PlanError:
            raise
        except (OSError, RuntimeError) as exc:
            # Acquisition failures are visible and do not fabricate an
            # observation. Schema and geometry errors in explicit inputs
            # are ValueErrors and remain fatal.
            report.update(status='unavailable', reason=str(exc))
    receipt = dict(schema=WINDOW_SCHEMA, review_sha256=backend.plan['review_sha256'],
                   analysis_time=_stamp(when), grid_identity=backend.grid.identity_sha256(),
                   assets=assets, routes=reports)
    _atomic(path, receipt)
    return _load_frozen(backend, receipt, when)

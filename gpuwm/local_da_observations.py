"""Read the existing observation registries and compose their public doors."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np


@contextmanager
def registry_inspection():
    """Do not repair bridge caches or retain companion environment bindings."""
    from gpuwm.bridges import inspection_only
    before = dict(os.environ)
    try:
        with inspection_only():
            yield
    finally:
        # Only restore variables changed by these resolvers, not unrelated
        # process environment values.
        for key in set(os.environ) | set(before):
            if key.startswith('GPUWM_') and os.environ.get(key) != before.get(key):
                if key in before:
                    os.environ[key] = before[key]
                else:
                    os.environ.pop(key, None)


def _door_record(door):
    try:
        binary = door.find()
        if binary is None:
            return None, 'The registered binary is missing; stage its published bridge bundle to enable this stream.'
        ok, reason = door.probe(binary)
        return (str(binary), reason) if ok else (None, reason)
    except (OSError, RuntimeError) as exc:
        return None, str(exc)


def _box_pieces(lat, lon, center):
    arc = center + (np.asarray(lon) - center + 180.) % 360. - 180.
    w, e = float(arc.min()), float(arc.max())
    s, n = float(np.min(lat)), float(np.max(lat))
    while w < -180.:
        w, e = w + 360., e + 360.
    while w >= 180.:
        w, e = w - 360., e - 360.
    return [(w, s, min(e, 180.), n)] + ([(-180., s, e - 360., n)] if e > 180. else [])


def inspect_routes(request, rung, experiment):
    """Static capability review, not a claim that an observation arrived.

    Discovery consults installed tables and read-only binary probes only.
    Actual window availability is receipted by the cycle's fetch stage.
    """
    from gpuwm import domain_wizard as dw
    from gpuwm.obs.frontdoor import FRONT_DOORS
    from gpuwm.obs.surface_networks import networks_for_bbox, SurfaceNetworkError
    from gpuwm.obs.coverage import read_site_table, sites_covering
    from gpuwm.obs.nexrad import find_nexrad_bin, probe_nexrad_bin
    from gpuwm.obs.superob import SuperobParams
    projection = dw._root_grid(rung['projection'], rung['nx'], rung['ny'], rung['dx_m'])
    lat, lon = projection.latlon_mass()
    target = SimpleNamespace(lat=lat, lon=lon)
    boxes = _box_pieces(lat, lon, rung['point'][1])
    rows = []
    with registry_inspection():
        try:
            binary = find_nexrad_bin()
            ok, reason = (False, 'The registered radar binary is missing; stage the engine bridge bundle.') if binary is None else probe_nexrad_bin(binary)
            candidates = [] if not ok else sites_covering(target, read_site_table(binary),
                            max_range_km=SuperobParams().max_range_km)
            rows.append(dict(id='radar', route='radar-grid', status='ready' if request.radar_grids else 'candidate' if candidates else 'unavailable',
                reason='Explicit grid inputs are checked again against the prepared columns.' if request.radar_grids else
                    'Range coverage candidates, not verified scan availability.' if candidates else reason if not ok else 'No registered radar has range coverage over this domain.',
                files=list(request.radar_grids), sites=[item.to_payload() for item in candidates],
                binary=str(binary) if ok else None, max_range_km=SuperobParams().max_range_km,
                batch_bound=max(1, len(candidates)) + 2))
        except (OSError, RuntimeError, ValueError) as exc:
            rows.append(dict(id='radar', route='radar-grid', status='ready' if request.radar_grids else 'unavailable',
                             files=list(request.radar_grids), sites=[], reason=str(exc), batch_bound=3))
        networks = set()
        for box in boxes:
            try:
                networks.update(networks_for_bbox(*box))
            except SurfaceNetworkError:
                pass
        binary, reason = _door_record(FRONT_DOORS['asos'])
        rows.append(dict(id='surface', route='surface-v1', status='candidate' if binary and networks else 'unavailable',
                         reason='Registered station-network coverage; reports are checked per analysis window.' if binary and networks else reason if not binary else 'No station network overlaps the domain.',
                         networks=sorted(networks), boxes=boxes, binary=binary, batch_bound=2))
        binary, reason = _door_record(FRONT_DOORS['goes'])
        from gpuwm.obs.goes_window import AcquisitionPolicy, sha256
        manual_cwp = bool(request.satellite_grids)
        cwp_policy = AcquisitionPolicy().to_payload()
        rows.append(dict(id='cloud-water-path', route='cloud-water-path',
                         status='ready' if manual_cwp else 'candidate' if binary else 'unavailable',
                         files=list(request.satellite_grids), binary=None if manual_cwp else binary,
                         binary_sha256=sha256(binary) if binary and not manual_cwp else None,
                         batch_bound=3, acquisition_policy=cwp_policy,
                         reason='Explicit phase-aware cloud-water-path grids; not brightness temperatures.' if manual_cwp else
                           'Automatic native GOES discovery, coverage and QC per window; up to 30-minute unconsumed scans, with explicit provisional errors.' if binary else reason))
        for name, door in sorted(FRONT_DOORS.items()):
            if name in ('asos', 'goes'):
                continue
            rows.append(dict(id=name, route='registered-observation', status='unavailable', batch_bound=0,
                             reason='Registered ingest exists, but this launcher has no corresponding 3D analysis product. Composite and accumulation products remain verification inputs.'))
        if importlib.util.find_spec('arwen_global') is not None:
            from arwen_global.obs_streams import STREAMS
            from arwen_global.obs_doors import front_door
            for name, spec in sorted(STREAMS.items()):
                status, binary = 'unavailable', None
                reason = spec.notes
                if spec.account_gated:
                    reason = 'This route requires an account; it is not fetched by this launcher. ' + reason
                elif not spec.decoder_built or spec.door is None:
                    reason = 'The registered route has no decoded table; raw payloads cannot enter the analysis. ' + reason
                elif spec.subscribes:
                    reason = 'A subscription is not a bounded analysis-window fetch; supply its decoded neutral table.'
                else:
                    binary, reason = _door_record(front_door(spec.door))
                    if binary:
                        status = 'candidate'
                        reason = 'Registered window fetch; local coverage, arrival time and supported operators are checked before use.'
                rows.append(dict(id=name, route='neutral-stream', status=status, reason=reason,
                                 binary=binary, stream=asdict(spec), batch_bound=max(1, len(spec.variables) * len(spec.measurements) * len(spec.sources)),
                                 networks=sorted(networks), latency_class=spec.latency_class))
        else:
            rows.append(dict(id='neutral-streams', route='neutral-stream', status='unavailable', batch_bound=0,
                reason='The shared neutral-table package is not installed; install gpuwm-global to reuse its registered observation streams.'))
        if request.obs_tables:
            from gpuwm.da.obs_point import read_tables
            decoded, receipts = read_tables(request.obs_tables)
            keys = {(r.source, r.variable, r.measurement) for r in decoded}
            rows.append(dict(id='neutral-tables', route='neutral-table', status='ready', files=list(request.obs_tables),
                             reason='Existing neutral-table decoder; row-level QC, vertical coverage and causal time assignment at analysis.',
                             input_receipts=receipts, batch_bound=len(keys), row_count=len(decoded)))
        rows.append(dict(id='radiances', route='column-radiance', status='unavailable', batch_bound=0,
                         reason='The global radiance operators need their own vertical column, coefficient identity, bias and error setup. A regional column adapter is not included; brightness temperature rows are never treated as point temperatures.'))
    # Prefer the shared vector/screen-height stream to the legacy speed
    # record when its actual ABI is available, to avoid using the same
    # surface report through two decoders.
    shared_surface = any(r['route'] == 'neutral-stream' and r['status'] == 'candidate'
                         and 'screen_temperature_2m' in r['stream']['measurements'] for r in rows)
    if shared_surface:
        for r in rows:
            if r['route'] == 'surface-v1':
                r.update(status='unavailable', reason='The shared neutral stream supplies these reports with vector winds; the legacy speed path is not duplicated.')
    return rows

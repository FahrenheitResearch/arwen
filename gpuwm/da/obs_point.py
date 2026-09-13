"""Regional operators for the existing neutral observation table.

Decode stays with the shared table reader. Pressure-level observations are
interpolated in each member's own column, with no vertical extrapolation.
Surface diagnostics must be supplied from the end of the forecast leg.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from gpuwm.da.letkf import GriddedObs, Localization
from gpuwm.da.radar_assimilation import member_earth_winds, grid_rotation


def read_tables(paths):
    """Read local neutral tables through their decoder, never fetch data."""
    try:
        from arwen_global.obs_table import decode_neutral_csv
    except ImportError as exc:
        # The decoder really is missing, so this is a refusal rather than
        # an admission question; what it owed and did not give was the way
        # out.  A bare ImportError string reached the door and was printed
        # as the whole of what the caller was told.
        raise ImportError(
            "A neutral observation table was given, and its decoder is not "
            "installed: arwen_global.obs_table is absent, so these rows "
            "cannot be read and no analysis can use them. Install "
            "gpuwm-global to read neutral tables, or review and run without "
            "--obs-table, which uses the radar and surface routes alone."
        ) from exc
    rows, receipts = [], []
    for path in paths:
        raw = Path(path).read_bytes()
        source, decoded, counts = decode_neutral_csv(raw.decode('utf-8'))
        rows.extend(decoded)
        receipts.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                             sources=source, counts=counts))
    return rows, receipts


def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError('Observation time has no UTC offset; provide an aware timestamp.')
    return value.astimezone(timezone.utc)


def _identity(row):
    return hashlib.sha256(json.dumps(row.identity(), separators=(',', ':')).encode()).hexdigest()


def _interp_logp(p, values, target):
    p, values = np.asarray(p, dtype=float), np.asarray(values, dtype=float)
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(values)) and np.all(p > 0)
            and np.all(np.diff(p) < 0) and p[-1] <= target <= p[0]):
        return np.nan
    return float(np.interp(np.log(target), np.log(p[::-1]), values[::-1]))


def state_columns(states, setup, grid):
    """Use full diagnosed pressure, base theta and native wind rotation."""
    from gpuwm.core import constants as c
    rotation = grid_rotation(grid)
    result = []
    for state in states:
        p = np.asarray(state['p'], dtype=float)
        base = np.asarray(setup['thb'], dtype=float)
        if base.ndim == 1:
            base = base[:, None, None]
        theta = np.asarray(state['thp'], dtype=float) + base
        qv = np.asarray(state['qv'], dtype=float)
        u, v, _ = member_earth_winds(state, rotation, where='neutral observation operator')
        # Regional qv is a mixing ratio, while the shared humidity
        # operators take specific humidity.
        q = qv / (1. + qv)
        temperature = theta * (p / c.P0) ** c.RCP
        result.append(dict(p=p, t=temperature, q=q, u=u, v=v))
    return result


def point_batches(rows, *, states, setup, grid, analysis_time, analysis_times,
                  surface=None, max_age_seconds=900., gross_sigma=4.,
                  localization=None, used_identities=(), columns=None, error_inflation=1.):
    """Return sparse gridded batches plus time, QC and attribution counts.

    Each report is assigned to its first eligible analysis boundary, never
    to a later boundary a second time. Receipt and publication cutoffs are
    applied before revision selection. Unknown arrival times are counted,
    not represented as proof of causal availability.
    """
    from arwen_global.obs_table import VARIABLE_TABLE
    from arwen_global.obs_operators import refractivity_n
    from arwen_global.surface_energy import dewpoint_from_specific_humidity
    if not np.isfinite(max_age_seconds) or max_age_seconds <= 0 or not np.isfinite(gross_sigma) or gross_sigma <= 0:
        raise ValueError('Observation age and QC thresholds must be finite and positive; correct the analysis settings.')
    if not np.isfinite(error_inflation) or error_inflation < 1.:
        raise ValueError('Observation error inflation must be finite and at least one; use the cadence settings.')
    when = _utc(analysis_time)
    schedule = tuple(_utc(t) for t in analysis_times)
    if schedule != tuple(sorted(set(schedule))) or when not in schedule:
        raise ValueError('Analysis boundaries are duplicated, unordered or omit this cycle; use the cycle clock schedule.')
    localization = localization or Localization(horizontal_m=15000., vertical_m=3000.)
    counts, revisions = Counter(), {}
    used = set(used_identities)
    for row in rows:
        counts['input'] += 1
        identity = _identity(row)
        if identity in used:
            counts['already_assimilated'] += 1
            continue
        valid = _utc(row.valid_time)
        received = None if row.received_time is None else _utc(row.received_time)
        published = None if row.published_time is None else _utc(row.published_time)
        if received is not None and received > when:
            counts['received_after_cutoff'] += 1
            continue
        if published is not None and published > when:
            counts['published_after_cutoff'] += 1
            continue
        if valid > when:
            counts['future_observation'] += 1
            continue
        eligible = [t for t in schedule if valid <= t and (t - valid).total_seconds() <= max_age_seconds
                    and (received is None or received <= t) and (published is None or published <= t)]
        if not eligible or eligible[0] != when:
            counts['outside_assigned_window'] += 1
            continue
        if received is None:
            counts['latency_unverified'] += 1
        floor = datetime.min.replace(tzinfo=timezone.utc)
        rank = (received or floor, published or floor, str(row.revision))
        if identity in revisions:
            counts['duplicate_or_superseded'] += 1
            if rank <= revisions[identity][0]:
                continue
        revisions[identity] = (rank, row)
    if columns is None:
        columns = state_columns(states, setup, grid)
    count = len(states)
    if len(columns) != count or count < 1:
        raise ValueError('Member columns do not match the forecast roster; evaluate every member once.')
    shape = (grid.nz, grid.ny, grid.nx)
    groups, accepted = {}, []
    temperatures = {(r.source, r.station_id, r.valid_time, r.level_pa): r.value
                    for _, r in revisions.values() if r.variable == 'temperature_k'}
    for identity, (_, row) in sorted(revisions.items()):
        bounds = VARIABLE_TABLE.get(row.variable)
        # The registry owns variable bounds; reject numerical corruption
        # even when a caller supplies already-decoded row objects.
        if bounds is None:
            counts['unknown_variable'] += 1
            continue
        lo, hi = bounds['gross_bounds']
        if not np.isfinite(row.value) or not np.isfinite(row.error) or row.error <= 0:
            counts['invalid_value_or_error'] += 1
            continue
        if (lo is not None and row.value < lo) or (hi is not None and row.value > hi):
            counts['gross_physical_range'] += 1
            continue
        if row.variable == 'dewpoint_k' and row.value > temperatures.get((row.source, row.station_id, row.valid_time, row.level_pa), np.inf):
            counts['supersaturated_report'] += 1
            continue
        x, y = grid.mass_index(row.latitude_deg, row.longitude_deg)
        if not np.isfinite(x) or not np.isfinite(y) or not (-.5 <= x < grid.nx - .5 and -.5 <= y < grid.ny - .5):
            counts['outside_domain'] += 1
            continue
        i, j = int(np.floor(float(x) + .5)), int(np.floor(float(y) + .5))
        label, pressure = row.measurement, row.level_pa
        h, k = [], 0
        if row.variable == 'brightness_temperature_k':
            counts['requires_column_radiance_operator'] += 1
            continue
        if row.variable == 'refractivity_n' and label == 'ro_refractivity_tangent_point':
            z = .5 * (grid.z_w[:-1, j, i] + grid.z_w[1:, j, i])
            height = row.elevation_m
            if not np.isfinite(height) or height < z[0] or height > z[-1]:
                counts['outside_vertical_column'] += 1
                continue
            k = int(np.argmin(abs(z - height)))
            for col in columns:
                p = np.exp(np.interp(height, z, np.log(col['p'][:, j, i])))
                t = np.interp(height, z, col['t'][:, j, i])
                q = np.interp(height, z, col['q'][:, j, i])
                h.append(float(refractivity_n(p, t, q)))
        elif pressure is not None and label in ('sonde_level', 'amv_assigned_pressure', ''):
            if row.variable not in ('temperature_k', 'wind_u_m_s', 'wind_v_m_s', 'dewpoint_k'):
                counts['no_pressure_level_operator'] += 1
                continue
            mean_p = np.mean([col['p'][:, j, i] for col in columns], axis=0)
            k = int(np.argmin(abs(np.log(mean_p) - np.log(pressure))))
            for col in columns:
                p = col['p'][:, j, i]
                if row.variable == 'dewpoint_k':
                    q = _interp_logp(p, col['q'][:, j, i], pressure)
                    h.append(float(dewpoint_from_specific_humidity(q, pressure)))
                else:
                    key = {'temperature_k': 't', 'wind_u_m_s': 'u', 'wind_v_m_s': 'v'}[row.variable]
                    h.append(_interp_logp(p, col[key][:, j, i], pressure))
        elif pressure is None:
            if not np.isfinite(row.elevation_m) or abs(row.elevation_m - grid.terrain_m[j, i]) > 200.:
                counts['surface_elevation_mismatch'] += 1
                continue
            key = {('temperature_k', 'screen_temperature_2m'): 't2',
                   ('wind_u_m_s', 'anemometer_wind_10m'): 'u10_earth',
                   ('wind_v_m_s', 'anemometer_wind_10m'): 'v10_earth',
                   ('wind_u_m_s', 'anemometer_wind_5m_reduced_to_10m'): 'u10_earth',
                   ('wind_v_m_s', 'anemometer_wind_5m_reduced_to_10m'): 'v10_earth'}.get((row.variable, label))
            if key is None or surface is None or key not in surface:
                counts['missing_surface_operator_or_diagnostic'] += 1
                continue
            data = np.asarray(surface[key])
            if data.shape != (count, grid.ny, grid.nx):
                raise ValueError(f'Surface diagnostic {key} has shape {data.shape}; supply end-of-leg diagnostics for every member.')
            h = data[:, j, i]
        else:
            counts['unknown_measurement_operator'] += 1
            continue
        h = np.asarray(h, dtype=float)
        if h.shape != (count,) or not np.all(np.isfinite(h)):
            counts['outside_vertical_column_or_invalid_operator'] += 1
            continue
        spread = float(np.var(h, ddof=1)) if count > 1 else 0.
        # A static sample stack carries the actual background at index 0.
        # Its caller provides this fact through surface metadata.
        mean = float(h[0]) if setup.get('static_covariance', False) else float(np.mean(h))
        if abs(row.value - mean) > gross_sigma * np.sqrt((row.error * error_inflation) ** 2 + spread):
            counts['background_check'] += 1
            continue
        name = f'point:{row.source}:{row.variable}:{label or "pressure_level"}'
        cell = (k, j, i)
        rank = (row.error, (when - row.valid_time).total_seconds(), identity)
        current = groups.setdefault(name, {})
        if cell in current:
            counts['thinned_colocation'] += 1
            if rank >= current[cell][0]:
                continue
        current[cell] = (rank, row.value, row.error * error_inflation, h, identity)
    batches = []
    for name, cells in sorted(groups.items()):
        values, errors, mask = np.zeros(shape), np.ones(shape), np.zeros(shape, bool)
        simulated = np.zeros((count, *shape))
        for cell, (_, val, err, h, identity) in cells.items():
            values[cell], errors[cell], mask[cell] = val, err, True
            simulated[(slice(None), *cell)] = h
            accepted.append(identity)
        batches.append(GriddedObs(name=name, values=values, errors=errors, simulated=simulated,
                                  mask=mask, localization=localization))
    counts['accepted'] = len(accepted)
    return batches, dict(schema='arwen.local-point-observations.v1', counts=dict(counts),
                         accepted_identities=sorted(accepted), analysis_time=when.isoformat(),
                         operator='member-specific log-pressure interpolation; native earth winds; end-of-leg surface diagnostics',
                         temporal_operator='3D analysis with causal windows, not a 4D trajectory operator',
                         vertical_extrapolation=False, error_inflation=float(error_inflation))

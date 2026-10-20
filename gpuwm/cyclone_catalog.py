"""Observed identity lookup for explicit cyclone names, independent of source.

The catalogue identifies a model track at the cycle. It never supplies forecast
fixes, intensity ranking or follower positions. URLs and matching limits are
owned by the same authority as the field tracker.
"""
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import time
from urllib.request import urlopen
from urllib.error import HTTPError


def _read(url, cache):
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / hashlib.sha256(url.encode()).hexdigest()
    if not path.is_file() or time.time() - path.stat().st_mtime > 3600:
        with urlopen(url, timeout=30) as response:
            raw = response.read(8 << 20)
        if len(raw) >= 8 << 20:
            raise ValueError(f"cyclone catalogue exceeds 8MiB: {url}")
        temporary = path.with_suffix(f'.{os.getpid()}.new')
        temporary.write_bytes(raw)
        temporary.replace(path)
    raw = path.read_bytes()
    return raw, {'url': url, 'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


def match_explicit(storm, start, candidates):
    """Return a unique cycle-position match, or None for a noncatalogue token."""
    if not (re.fullmatch(r'[A-Za-z]{2}\d{6}', storm) or
            re.fullmatch(r'[A-Za-z][A-Za-z -]*', storm)):
        return None
    from gpuwm.tc_tracks import TABLE, best_track, distance
    from gpuwm.cyclone_tracks import fix_at, _time
    policy = json.loads(TABLE.read_text())['identity_catalog']
    cache = Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'gpuwm/cyclone-identities'
    raw, catalog = _read(policy['index_url'], cache)
    columns = policy['columns']
    matches = {}
    for fields in csv.reader(io.StringIO(raw.decode())):
        if len(fields) <= max(columns.values()):
            continue
        row = {key: fields[index].strip() for key, index in columns.items()}
        if row['basin'] not in policy['basins']:
            continue
        if storm.casefold() not in (row['name'].casefold(), row['id'].casefold()):
            continue
        began = datetime.strptime(row['start'], '%Y%m%d%H')
        ended = datetime.strptime(row['end'], '%Y%m%d%H')
        if began <= start <= ended:
            matches[row['id']] = row
    if not matches:
        raise ValueError(f'track-storm-not-active: {storm!r} has no catalogue entry at {start.isoformat()} in {policy["basins"]}')
    if len(matches) != 1:
        raise ValueError(f'track-storm-ambiguous: {storm!r} matches catalogue ids {sorted(matches)}; use an ATCF id')
    row = next(iter(matches.values()))
    url = policy['archive_url'].format(year=row['year'], id=row['id'].lower())
    try:
        raw, reference = _read(url, cache)
    except HTTPError as error:
        if error.code != 404 or int(row['year']) != datetime.now(timezone.utc).year:
            raise
        url = policy['active_url'].format(id=row['id'].lower())
        raw, reference = _read(url, cache)
    decoded = gzip.decompress(raw) if url.endswith('.gz') else raw
    # The existing best-track parser consumes a local file and records its hash.
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (hashlib.sha256(decoded).hexdigest() + '.dat')
    if not path.is_file():
        path.write_bytes(decoded)
    best = best_track(path, row['id'])
    fixes = best['fixes']
    for fix in fixes:
        fix['lead_hours'] = (_time(fix['valid']) - start).total_seconds() / 3600
    relevant = [fix for fix in fixes if abs(fix['lead_hours']) <= policy['maximum_interpolation_hours']]
    observed, interpolation = fix_at(relevant, start)
    if observed is None:
        raise ValueError(f'track-storm-no-cycle-fix: {row["id"]} at {start.isoformat()}')
    nearby = [(track, distance(observed, track['center_fix'])) for track in candidates]
    nearby = [(track, km) for track, km in nearby if km <= policy['maximum_match_km']]
    if len(nearby) != 1:
        raise ValueError(f'track-storm-position-ambiguous: {row["id"]} has {len(nearby)} model tracks within {policy["maximum_match_km"]}km at the cycle; select a track id')
    chosen, km = nearby[0]
    return chosen, dict(storm_id=row['id'], name=row['name'], catalogue=catalog,
                        best_track=reference, decoded_sha256=best['sha256'],
                        cycle_fix=observed, interpolation=interpolation,
                        distance_km=km, maximum_match_km=policy['maximum_match_km'],
                        role='cycle identity only; model fixes drive the forecast')

"""Observed identity lookup for explicit cyclone names, independent of source.

Two roles, both read from the same public-domain catalogue:

* With model tracks (``--tracks``), the catalogue only identifies a model
  track at the cycle (:func:`match_explicit`); the model fixes drive the
  placement.
* Without model tracks, :func:`observed_tracks` writes the catalogue's own
  observed best-track fixes for every system active at the cycle as an
  ordinary tracks file, so ``cyclone-setup --storm auto|ID|NAME`` selects a
  storm with no user-supplied file.  Those fixes are setup guidance only
  (parent placement); the live follower still tracks the model vortex.

URLs and matching limits are owned by the same authority as the field tracker.
"""
from datetime import datetime, timedelta, timezone
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


def _cache():
    return Path(os.environ.get('XDG_CACHE_HOME', str(Path.home() / '.cache'))) / 'gpuwm/cyclone-identities'


def _policy():
    from gpuwm.tc_tracks import TABLE
    return json.loads(TABLE.read_text())['identity_catalog']


def _index(policy, cache):
    """Catalogue rows in the policy's basins, with parsed start/end."""
    raw, catalog = _read(policy['index_url'], cache)
    columns = policy['columns']
    rows = []
    for fields in csv.reader(io.StringIO(raw.decode())):
        if len(fields) <= max(columns.values()):
            continue
        row = {key: fields[index].strip() for key, index in columns.items()}
        if row['basin'] not in policy['basins']:
            continue
        try:
            row['began'] = datetime.strptime(row['start'], '%Y%m%d%H')
            row['ended'] = datetime.strptime(row['end'], '%Y%m%d%H')
        except ValueError:
            continue
        rows.append(row)
    return rows, catalog


def _best_track(policy, row, cache):
    """The storm's decoded best track, archive first, active file for this year."""
    from gpuwm.tc_tracks import best_track
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
    return best_track(path, row['id']), reference


def match_explicit(storm, start, candidates):
    """Return a unique cycle-position match, or None for a noncatalogue token."""
    if not (re.fullmatch(r'[A-Za-z]{2}\d{6}', storm) or
            re.fullmatch(r'[A-Za-z][A-Za-z -]*', storm)):
        return None
    from gpuwm.tc_tracks import distance
    from gpuwm.cyclone_tracks import fix_at, _time
    policy = _policy()
    cache = _cache()
    rows, catalog = _index(policy, cache)
    matches = {}
    for row in rows:
        if storm.casefold() not in (row['name'].casefold(), row['id'].casefold()):
            continue
        if row['began'] <= start <= row['ended']:
            matches[row['id']] = row
    if not matches:
        raise ValueError(f'track-storm-not-active: {storm!r} has no catalogue entry at {start.isoformat()} in {policy["basins"]}')
    if len(matches) != 1:
        raise ValueError(f'track-storm-ambiguous: {storm!r} matches catalogue ids {sorted(matches)}; use an ATCF id')
    row = next(iter(matches.values()))
    best, reference = _best_track(policy, row, cache)
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


OBSERVED_MEMBER = 'nhc-best-track'


def observed_tracks(start, hours):
    """Write a ``gpuwm.tc-tracks.v1`` file of the catalogue's observed systems.

    Every catalogue storm active at ``start`` (naive UTC) becomes one track,
    labelled with its ATCF id and name so ``select_track`` resolves ``auto``
    (lowest cycle pressure), an ATCF id or a name without a model tracks
    file.  Fixes are the observed best track over ``[start - interpolation
    limit, start + hours + interpolation limit]``; a live storm simply has
    no fixes past its latest observation.  Returns the written path.
    """
    from gpuwm.tc_tracks import SCHEMA
    from concurrent.futures import ThreadPoolExecutor
    policy = _policy()
    cache = _cache()
    rows, catalog = _index(policy, cache)
    active = {row['id']: row for row in rows if row['began'] <= start <= row['ended']}
    pad = timedelta(hours=policy['maximum_interpolation_hours'])
    first, last = start - pad, start + timedelta(hours=hours) + pad

    def track(row):
        best, reference = _best_track(policy, row, cache)
        fixes = []
        for fix in best['fixes']:
            valid = datetime.strptime(fix['valid'], '%Y-%m-%dT%H:%M:%SZ')
            if first <= valid <= last:
                fixes.append(dict(fix, lead_hours=(valid - start).total_seconds() / 3600))
        return dict(track_id=row['id'], member_id=OBSERVED_MEMBER, weight=1.0,
                    atcf_id=row['id'], name=row['name'], aliases=[row['name'], row['id']],
                    best_track=reference, fixes=fixes)

    with ThreadPoolExecutor(max_workers=max(1, min(8, len(active)))) as pool:
        tracks = [t for t in pool.map(track, sorted(active.values(), key=lambda r: r['id'])) if t['fixes']]
    stamp = start.strftime('%Y-%m-%dT%H:%M:%SZ')
    document = dict(schema=SCHEMA,
                    tracker=dict(backend='nhc-atcf-best-track', catalogue=catalog,
                                 basins=policy['basins'], source=policy['source'],
                                 role='observed best-track fixes; setup placement guidance only, '
                                      'the live follower tracks the model vortex'),
                    members=[dict(member_id=OBSERVED_MEMBER, family='analysis', source='nhc-atcf',
                                  init=stamp, weight=1.0, ancestry={})],
                    tracks=tracks)
    text = json.dumps(document, indent=2, allow_nan=False) + '\n'
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'observed-tracks-{start:%Y%m%d%H}-{hours}h-{hashlib.sha256(text.encode()).hexdigest()[:16]}.json'
    if not path.is_file():
        temporary = path.with_suffix(f'.{os.getpid()}.new')
        temporary.write_text(text)
        temporary.replace(path)
    return path

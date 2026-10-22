"""Select source trajectories from small track metadata through existing doors."""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from datetime import timedelta
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from gpuwm.tc_tracks import SCHEMA, binary, distance, dump, run, sha, stamp, utc

TABLE = Path(__file__).with_name('authorities') / 'tc-track-strata.v1.json'


def finite_fix(f):
    return bool(f and all(isinstance(f.get(k), (int, float)) and math.isfinite(f[k])
                          for k in ('lat', 'lon')))


def associate(document, target, radius, start, end, storm=None, tracks_path=None):
    """One target identity per parent. Missing target members retain their mass."""
    results = []
    for m in sorted(document['members'], key=lambda m: m['member_id']):
        tracks = [t for t in document['tracks'] if t['member_id'] == m['member_id']]
        choices = []
        if storm == 'auto':
            from gpuwm.cyclone_tracks import select_track
            # Use the same per-member start/window and intensity rule as setup.
            try:
                selected = select_track(tracks_path, member=m['member_id'], storm='auto',
                                        source=m['source'], start=start.replace(tzinfo=None),
                                        hours=(end-start).total_seconds()/3600)
            except ValueError as error:
                if not str(error).startswith('track-no-eligible-system:'):
                    raise
                selected = None
            track_id = selected['lineage']['track_id'] if selected else None
            choices = [(0., t['track_id'], t, None) for t in tracks if t['track_id'] == track_id]
        else:
            for t in tracks:
                fixes = [f for f in t['fixes'] if finite_fix(f) and utc(f['valid']) == utc(target['valid'])]
                for f in fixes:
                    d = distance(f, target)
                    if d <= radius:
                        choices.append((d, t['track_id'], t, f))
        pick = min(choices, key=lambda x: (x[0], x[1])) if choices else None
        results.append(dict(member_id=m['member_id'], track_id=pick[1] if pick else None,
                            distance_km=pick[0] if pick else None, fix=pick[3] if pick else None,
                            outcome='associated' if pick else 'no-storm'))
    return results


def allocate(groups, count):
    """Bounded Hamilton allocation after coverage, with exact integer tie breaks."""
    result = {g['id']: 1 for g in groups}
    remaining = count - len(groups)
    while remaining:
        active = [g for g in groups if result[g['id']] < len(g['members'])]
        total = sum(len(g['members']) for g in active)
        quota = {g['id']: divmod(remaining * len(g['members']), total) for g in active}
        used = 0
        for g in active:
            n = min(quota[g['id']][0], len(g['members']) - result[g['id']])
            result[g['id']] += n
            used += n
        remaining -= used
        for g in sorted(active, key=lambda g: (-quota[g['id']][1], g['id'])):
            if not remaining:
                break
            if result[g['id']] < len(g['members']):
                result[g['id']] += 1
                remaining -= 1
    return result


def select(document, crossings, *, target, start, end, count, seed, rules,
           storm=None, tracks_path=None):
    if document.get('schema') != SCHEMA:
        raise ValueError(f'expected {SCHEMA}')
    members = document.get('members', [])
    ids = [m['member_id'] for m in members]
    if not ids or len(set(ids)) != len(ids) or any(not isinstance(i, str) for i in ids):
        raise ValueError('selection needs unique nonempty string member IDs')
    families = {m['family'] for m in members}
    if len(families) != 1:
        raise ValueError('select one parent family: track family weights are separate distributions')
    if not 1 <= count <= len(members) or seed < 0 or rules['max_strata'] < 1:
        raise ValueError('select must be 1..N and seed nonnegative')
    weights = {m['member_id']: float(m['weight']) for m in members}
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError('parent weights must be finite and positive')
    total = math.fsum(weights.values())
    if not math.isfinite(total):
        raise ValueError('parent weight normalization must be finite')
    weights = {k: v / total for k, v in weights.items()}
    assoc = associate(document, target, rules['association_radius_km'], start, end, storm, tracks_path)
    by_track = {(t['member_id'], t['track_id']): t for t in document['tracks']}
    if len(by_track) != len(document['tracks']) or any(t['member_id'] not in weights for t in document['tracks']):
        raise ValueError('track IDs must be unique within known members')
    groups = {}
    for a in assoc:
        key = 'no-storm'
        if a['track_id'] is not None:
            t = by_track[a['member_id'], a['track_id']]
            fixes = [f for f in t['fixes'] if f and start <= utc(f['valid']) <= end]
            pressures = [f['mslp_pa'] for f in fixes if finite_fix(f) and
                         isinstance(f.get('mslp_pa'), (int, float)) and math.isfinite(f['mslp_pa'])]
            intensity = str(bisect_right(rules['peak_mslp_pa_edges'], min(pressures))) if pressures else 'missing'
            hits = [c for c in crossings['crossings'] if c['member_id'] == a['member_id'] and
                    c['track_id'] == a['track_id'] and start <= utc(c['valid']) <= end]
            hit = min(hits, key=lambda c: (utc(c['valid']), c['lon'], c['lat'])) if hits else None
            a['landfall'] = hit
            a['peak_mslp_pa'] = min(pressures) if pressures else None
            if hit:
                sector = bisect_right(rules['sector']['edges'], hit[rules['sector']['coordinate']])
                hour = (utc(hit['valid']) - start).total_seconds() / 3600
                key = f'landfall:s{sector}:t{bisect_right(rules["landfall_hours_edges"], hour)}:p{intensity}'
            else:
                key = f'no-landfall:p{intensity}'
        groups.setdefault(key, []).append(a['member_id'])
        a['initial_stratum'] = key
    groups = [dict(id=k, components=[k], members=sorted(v)) for k, v in sorted(groups.items())]
    merges = []
    limit = min(rules['max_strata'], count)
    if any(g['id'] == 'no-storm' for g in groups) and len(groups) > 1 and limit < 2:
        raise ValueError('K/max-strata must be at least 2 to retain no-storm separately from storm outcomes')
    while len(groups) > limit:
        pairs = []
        for i, a in enumerate(groups):
            for b in groups[i+1:]:
                if 'no-storm' in a['components'] + b['components']:
                    continue
                outcomes = {c.split(':')[0] for c in a['components'] + b['components']}
                pairs.append((len(outcomes), len(a['members']) + len(b['members']), a['id'], b['id'], a, b))
        _, _, _, _, a, b = min(pairs, key=lambda p: p[:4])
        components = sorted(a['components'] + b['components'])
        g = dict(id='|'.join(components), components=components, members=sorted(a['members'] + b['members']))
        merges.append(dict(inputs=[a['id'], b['id']], output=g['id']))
        groups = sorted([x for x in groups if x is not a and x is not b] + [g], key=lambda x: x['id'])
    allocation = allocate(groups, count)
    selected = []
    for number, g in enumerate(groups):
        g['label'] = f'S{number+1}'
        g['mass'] = math.fsum(weights[i] for i in g['members'])
        g['allocation'] = k = allocation[g['id']]
        g['inclusion_probability'] = k / len(g['members'])
        # Hash ordering is independent of JSON input order and Python RNG versions.
        def rank(i):
            return hashlib.sha256(json.dumps([seed, g['id'], i], separators=(',', ':')).encode()).digest(), i
        for i in sorted(g['members'], key=rank)[:k]:
            selected.append(dict(member_id=i, stratum=g['label'], stratum_id=g['id'],
                                 parent_weight=weights[i], weight=g['mass']/k,
                                 inclusion_probability=g['inclusion_probability']))
    return dict(schema='gpuwm.tc-selection.v1', seed=seed, select=count,
                window=dict(start=stamp(start), end=stamp(end)),
                association=dict(mode=storm or 'target-fix', target=target,
                                 radius_km=rules['association_radius_km'], members=assoc),
                parent_weight_total=total, strata=groups, merges=merges, allocation=allocation,
                inclusion_probabilities={i:g['inclusion_probability'] for g in groups for i in g['members']},
                selected=selected, weight_sum=math.fsum(s['weight'] for s in selected),
                estimator='stratum mass / allocated representatives; uniform sampling, not calibrated probabilities')


def author_member(row, member, *, document, target, start, end, rules, out):
    """Use the ordinary cyclone planner and its existing track-placement code."""
    from gpuwm import cyclone_setup, domain_wizard as dw
    from gpuwm.cyclone_tracks import select_track
    from gpuwm.cyclone_sources import companion_input_files
    from gpuwm.hrrr_prepared_bundle import render_wps_namelist
    from gpuwm.hrrr_route_inputs import candidate_companions
    from gpuwm.starter_template import _publish_new_files
    path = out / ('member-' + hashlib.sha256(member['member_id'].encode()).hexdigest()[:12]) / 'experiment.toml'
    selection = None
    if row['track_id'] is not None:
        selection = select_track(row['tracks'], storm=row['track_id'], member=member['member_id'],
                                 source=row['source'], start=start.replace(tzinfo=None),
                                 hours=(end-start).total_seconds()/3600)
        fix = selection['lineage']['center_fix']
    else:
        if target is None:
            raise ValueError('no-storm auto member has no nest seed; supply a target-fix association for an explicit no-storm placement')
        fix = target
    first = (start-utc(row['cycle'])).total_seconds()/3600
    if first != int(first):
        raise ValueError('nest start must be an integer source lead hour')
    planned = cyclone_setup.plan_cyclone(cycle=utc(row['cycle']).strftime('%Y%m%d%H'),
        point=(fix['lat'], fix['lon']), hours=int((end-start).total_seconds()/3600),
        sizing=dw.resolve_sizing_budget(rules['nest']['card'], None),
        forcing_source=row['source'], member=row['member'], start_hour=int(first),
        nest_km=rules['nest']['nest_km'], tiles=rules['nest']['tiles'],
        source=str(path), track_selection=selection)
    text = planned['config_text']
    exp = dw.experiment_from_text(text, source=str(path))
    wps = render_wps_namelist(exp, interval_seconds=planned['forcing_interval_seconds'])
    inputs = companion_input_files(row['source'], path)
    companions = candidate_companions(path, exp, wps_text=wps, source=row['source'])
    path.parent.mkdir(parents=True, exist_ok=True)
    _publish_new_files((*inputs, *companions, (path, text)))
    dump(path.with_suffix('.cyclone.json'), {k:v for k,v in planned.items() if k != 'config_text'})
    row.update(configuration=str(path), configuration_sha256=sha(path))
    return dict(member_id=member['member_id'], domains=planned['domains'],
                memory=planned['memory'], track_placement=planned.get('track_placement'),
                no_storm_placement='declared target fix' if selection is None else None)


def execute(args):
    begin = time.monotonic()
    source = Path(args.tracks).resolve()
    document = json.loads(source.read_bytes())
    table = Path(args.rules).resolve()
    rules = json.loads(table.read_bytes())
    if rules.get('schema') != 'gpuwm.tc-track-strata.v1' or rules['max_strata'] < 1:
        raise ValueError('invalid track stratum authority')
    if args.max_strata is not None:
        rules['max_strata'] = args.max_strata
    if rules['max_strata'] < 1:
        raise ValueError('max-strata must be positive')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    start = utc(args.start or document['members'][0]['init'])
    end = start + timedelta(hours=args.hours or rules['nest']['hours'])
    target = None
    if args.target:
        lat, lon, valid = args.target.split(',', 2)
        target = dict(lat=float(lat), lon=float(lon), valid=stamp(utc(valid)))
        if not -90 <= target['lat'] <= 90 or not -180 <= target['lon'] <= 180:
            raise ValueError('target coordinates outside latitude/longitude bounds')
    from gpuwm.rustwx import renderer_env
    native_env = renderer_env()
    native_env['RAYON_NUM_THREADS'] = str(args.workers)
    before = time.monotonic()
    run([binary('rw_wrfbatch'), '--tc-track-landfalls', source, out/'landfalls.json'], env=native_env)
    crossing_seconds = time.monotonic() - before
    result = select(document, json.loads((out/'landfalls.json').read_bytes()), target=target,
                    start=start, end=end, count=args.select, seed=args.seed, rules=rules,
                    storm=args.storm, tracks_path=source)
    root = Path(__file__).resolve().parents[1]
    result.update(inputs=dict(tracks=dict(path=str(source), sha256=sha(source)),
                              rule_table=dict(path=str(table), sha256=sha(table)),
                              crossings=dict(path=str(out/'landfalls.json'), sha256=sha(out/'landfalls.json'))),
                  rules=rules, code_revision=run(['git','-C',root,'rev-parse','HEAD'],capture_output=True).stdout.strip(),
                  selector_sha256=sha(Path(__file__)))
    dump(out/'selection.json', result)
    plume = deepcopy(document)
    selected = {s['member_id']: s for s in result['selected']}
    plume['members'] = [dict(m, weight=selected[m['member_id']]['weight'], stratum=selected[m['member_id']]['stratum'])
                        for m in document['members'] if m['member_id'] in selected]
    associated = {a['member_id']: a['track_id'] for a in result['association']['members']}
    plume['tracks'] = [dict(t, weight=selected[t['member_id']]['weight']) for t in document['tracks']
                       if t['member_id'] in selected and t['track_id'] == associated[t['member_id']]]
    if args.best_track:
        from gpuwm.tc_tracks import best_track
        plume['best_track'] = best_track(args.best_track, args.storm_id)
    dump(out/'selected-tracks.json', plume)
    member_map = json.loads(Path(args.member_map).read_bytes()) if args.member_map else {}
    if args.member_map:
        result['inputs']['member_map'] = dict(path=str(Path(args.member_map).resolve()), sha256=sha(args.member_map))
        dump(out/'selection.json', result)
    roster = []
    plans = []
    for m in plume['members']:
        ancestry = m.get('ancestry', {})
        token = member_map.get(m['member_id'], m.get('member', ancestry.get('member')))
        # Fetch identity must be explicit. A tracking label is not a source token.
        from gpuwm.source_adapters import get_source_adapter
        adapter = get_source_adapter(m['source'])
        from importlib import import_module
        acquisition_validator = (getattr(import_module(adapter.acquisition_owner), 'validate_member', None)
                                 if adapter.acquisition_owner else None)
        if token is None and (adapter.selection_owner or adapter.member_set or acquisition_validator):
            raise ValueError(f'{m["member_id"]}: no source member token in member/ancestry.member; selection saved, cannot invent fetch identity')
        s = selected[m['member_id']]
        roster.append(dict(source=m['source'], cycle=m['init'], member=None if token is None else str(token),
                           weight=s['weight'], stratum=s['stratum'], track_member_id=m['member_id'],
                           tracks=str(out/'selected-tracks.json'), track_id=associated[m['member_id']]))
    if args.author_nests:
        for row, m in zip(roster, plume['members']):
            plans.append(author_member(row, m, document=document, target=target, start=start, end=end, rules=rules, out=out))
        dump(out/'nest-plans.json', dict(plans=plans, capacity=rules['nest']['card'],
             conservative_sum_envelope_bytes=sum(p['memory']['peak_envelope_bytes'] for p in plans),
             basis='sum of individual ordinary cyclone tree envelopes; no GPU allocation measured'))
    dump(out/'roster.json', dict(schema='gpuwm.tc-roster.v1', selection_sha256=sha(out/'selection.json'), trajectories=roster))
    stages = [dict(stage='coastline-crossings', seconds=crossing_seconds,
                   parallelism=f'{args.workers} Rust threads'),
              dict(stage='metadata-selection-and-nest-plans', seconds=time.monotonic()-begin-crossing_seconds,
                   parallelism='1 thread; small track metadata and CPU tree estimates')]
    if args.render:
        render_start = time.monotonic()
        png = out/'d01'/'tc-track-plume'/stamp(start)[:10]/'selected-plume.png'
        png.parent.mkdir(parents=True, exist_ok=True)
        run([binary('rw_wrfbatch'), '--tc-track-plume', out/'selected-tracks.json', png], env=native_env)
        stages.append(dict(stage='weighted-plume', seconds=time.monotonic()-render_start,
                           parallelism=f'{args.workers} Rust threads'))
    dump(out/'receipt.json', dict(stages=stages,
        total_seconds=time.monotonic()-begin, command=sys.argv))
    print(out/'selection.json')
    return 0


def register_cli(subparsers):
    p = subparsers.add_parser('tc-strata', help='Select weighted source members across cyclone track outcomes')
    p.add_argument('--tracks', required=True)
    association = p.add_mutually_exclusive_group(required=True)
    association.add_argument('--target', help='LAT,LON,UTC initial fix; exact common valid time, no extrapolation')
    association.add_argument('--storm', choices=('auto',))
    p.add_argument('--select', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--rules', default=str(TABLE))
    p.add_argument('--member-map', help='JSON mapping tracking member IDs to explicit source member tokens')
    p.add_argument('--max-strata', type=int)
    p.add_argument('--start')
    p.add_argument('--hours', type=int)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', required=True)
    p.add_argument('--render', action='store_true')
    p.add_argument('--author-nests', action='store_true', help='Author each selected member through cyclone-setup placement and pricing')
    p.add_argument('--best-track')
    p.add_argument('--storm-id')
    p.set_defaults(func=main)


def main(args):
    try:
        if args.workers < 1:
            raise ValueError('workers must be positive')
        return execute(args)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        print(f'gpuwm tc-strata: {e}', file=sys.stderr)
        return 2

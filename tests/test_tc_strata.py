"""Small handwritten populations exercise selection, never forecast evidence."""
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path

import pytest

from gpuwm.tc_strata import TABLE, allocate, associate, select
from gpuwm.tc_tracks import utc

START = utc('2031-02-03T00:00:00Z')
TARGET = dict(lat=18., lon=-70., valid='2031-02-03T00:00:00Z')


def population():
    doc = dict(schema='gpuwm.tc-tracks.v1', tracker={}, members=[], tracks=[])
    crossings = dict(crossings=[])
    for i in range(12):
        mid = f'm{i:02d}'
        doc['members'].append(dict(member_id=mid, family='ai', source='unregistered-test-source',
                                   init=TARGET['valid'], weight=(i+1)/78, ancestry=dict(member=str(i))))
        if i == 11:
            continue
        fixes = [dict(TARGET, mslp_pa=100000-i*900),
                 dict(lat=24., lon=-69., valid='2031-02-04T00:00:00Z', mslp_pa=99000-i*900)]
        doc['tracks'].append(dict(track_id=mid+'-a', member_id=mid, weight=(i+1)/78, fixes=fixes))
        if i < 8:
            crossings['crossings'].append(dict(member_id=mid, track_id=mid+'-a',
                valid='2031-02-04T00:00:00Z', lon=-125+i*20, lat=23.))
    return doc, crossings


def choose(doc, crossings, **kw):
    return select(doc, crossings, **(dict(target=TARGET, start=START, end=START+timedelta(hours=48),
                   count=6, seed=1729, rules=json.loads(TABLE.read_text())) | kw))


def test_stratification_weights_no_storm_and_input_order():
    doc, crossings = population()
    a = choose(doc, crossings)
    reverse = deepcopy(doc)
    reverse['members'].reverse()
    reverse['tracks'].reverse()
    for t in reverse['tracks']:
        t['fixes'].reverse()
    assert choose(reverse, crossings) == a
    assert len(a['strata']) == len(a['selected']) == 6
    assert sum(s['weight'] for s in a['selected']) == pytest.approx(1.)
    assert {s['member_id'] for s in a['selected']} >= {'m11'}
    for g in a['strata']:
        selected = [s for s in a['selected'] if s['stratum_id'] == g['id']]
        assert len(selected) == g['allocation']
        assert sum(s['weight'] for s in selected) == pytest.approx(g['mass'])
        assert all(s['inclusion_probability'] == len(selected)/len(g['members']) for s in selected)


def test_allocation_coverage_largest_remainder_and_capacity():
    groups = [dict(id='a', members=list(range(1))), dict(id='b', members=list(range(5))),
              dict(id='c', members=list(range(7)))]
    assert allocate(groups, 6) == dict(a=1, b=2, c=3)
    assert allocate(groups, 13) == dict(a=1, b=5, c=7)


def test_no_storm_never_disappears_when_budget_too_small():
    doc, crossings = population()
    with pytest.raises(ValueError, match='retain no-storm'):
        choose(doc, crossings, count=1)
    r = choose(doc, crossings, count=2)
    assert len(r['strata']) == 2
    assert any(g['id'] == 'no-storm' for g in r['strata'])


def test_radius_exact_time_and_competing_storm_tie():
    doc, _ = population()
    competitor = deepcopy(doc['tracks'][0]); competitor['track_id']='lexical-first'
    doc['tracks'].append(competitor)
    a = associate(doc, TARGET, 500, START, START+timedelta(hours=48))
    assert a[0]['track_id'] == 'lexical-first'
    doc['tracks'][0]['fixes'][0]['lon'] = 40
    competitor['fixes'][0]['valid'] = '2031-02-03T06:00:00Z'
    a = associate(doc, TARGET, 500, START, START+timedelta(hours=48))
    assert a[0]['outcome'] == 'no-storm'


def test_equal_parent_weight_formula_and_full_population():
    doc, crossings = population()
    for m in doc['members']: m['weight']=1/12
    result = choose(doc, crossings, count=12)
    assert len({s['member_id'] for s in result['selected']}) == 12
    assert all(s['weight'] == pytest.approx(1/12) and s['inclusion_probability'] == 1
               for s in result['selected'])
    with pytest.raises(ValueError, match='unique'):
        doc['members'].append(doc['members'][0]); choose(doc, crossings)


def test_roster_request_retains_weights_and_rejects_bad_mass():
    from gpuwm.ensemble.request import EnsembleRequest
    entries = [dict(source='gefs', cycle=TARGET['valid'], member=f'p{i+1:02d}', weight=w,
                    tracks='/tmp/explicit-tracks.json', track_id='a', track_member_id=f'm{i}', stratum='S1')
               for i,w in enumerate((.25, .75))]
    request = EnsembleRequest(members=2, recipe='member-roster', trajectories=entries)
    assert request.receipt()['trajectories'][1]['weight'] == .75
    assert len(request.member_variants) == 2
    entries[1]['weight']=.5
    with pytest.raises(ValueError, match='sum to one'):
        EnsembleRequest(members=2, recipe='member-roster', trajectories=entries)


def test_native_selector_and_member_configs(tmp_path, monkeypatch):
    """CPU interface test on handwritten members; no real-data proof claim."""
    import os
    from types import SimpleNamespace
    from gpuwm.tc_strata import execute
    from gpuwm.ensemble.recipe_door import member_configuration, plan_recipe
    from gpuwm.ensemble.request import EnsembleRequest
    from gpuwm.experiment import load_experiment
    import tomllib
    native = os.environ.get('GPUWM_RW_WRFBATCH')
    if not native:
        pytest.skip('built native renderer not named')
    doc = dict(schema='gpuwm.tc-tracks.v1', tracker=dict(backend='handwritten-test-fixture'), members=[], tracks=[])
    for i in range(3):
        mid=f'm{i}'
        doc['members'].append(dict(member_id=mid, family='ai', source='earth2-fcn3',
            init=TARGET['valid'], weight=1/3, ancestry=dict(member=str(i))))
        if i == 2: continue
        fixes = [dict(TARGET, lead_hours=0, mslp_pa=99000., max_wind10_ms=25., flags=[]),
                 dict(valid='2031-02-04T00:00:00Z', lead_hours=24, lat=20.+i, lon=-71.,
                      mslp_pa=97000., max_wind10_ms=35., flags=[]),
                 dict(valid='2031-02-05T00:00:00Z', lead_hours=48, lat=22.+i, lon=-72.,
                      mslp_pa=96000., max_wind10_ms=40., flags=[])]
        doc['tracks'].append(dict(member_id=mid, track_id=mid+'-a', weight=1/3, fixes=fixes))
    doc['best_track'] = dict(fixes=doc['tracks'][0]['fixes'])
    source=tmp_path/'handwritten.json'; source.write_text(json.dumps(doc))
    out=tmp_path/'selection'
    args=SimpleNamespace(tracks=source, rules=TABLE, max_strata=None, out=out,
        start=None, hours=48, target='18,-70,'+TARGET['valid'], storm=None, select=2,
        seed=1729, workers=4, member_map=None, author_nests=True, render=True, best_track=None)
    assert execute(args) == 0
    roster=json.loads((out/'roster.json').read_text())['trajectories']
    assert len(roster)==2 and sum(r['weight'] for r in roster)==pytest.approx(1)
    assert list(out.rglob('selected-plume.png'))
    first=Path(roster[0]['configuration']); raw=tomllib.loads(first.read_text()); exp=load_experiment(first)
    request=EnsembleRequest(members=2, recipe='member-roster', trajectories=roster)
    recipe=plan_recipe(request, raw, exp)
    for member in recipe.members:
        config,wps,experiment=member_configuration(raw,first,first.with_suffix('.namelist.wps'),recipe,member,
                                                  tmp_path/f'review-{member.index}')
        assert wps.is_file()
        assert len(experiment.domains)==2
        actual=tomllib.loads(config.read_text())
        assert actual['fetch']['member']==roster[member.index]['member']
    # Preserve this touched-test's small artifacts under the lane for inspection.
    receipt=os.environ.get('GPUWM_TC_TEST_ARTIFACTS')
    if receipt:
        import shutil
        shutil.copytree(out, receipt)


def test_selected_roster_cli_dry_run(tmp_path):
    """Review a saved handwritten interface-test roster through the actual CLI."""
    import os
    import subprocess
    import sys
    artifacts = os.environ.get('GPUWM_TC_TEST_ARTIFACTS')
    if not artifacts:
        pytest.skip('saved handwritten interface fixture not named')
    roster = Path(artifacts)/'roster.json'
    entries = json.loads(roster.read_text())['trajectories']
    command = [sys.executable, '-m', 'gpuwm', 'ensemble', entries[0]['configuration'],
               '--recipe', 'member-roster', '--members', str(len(entries)),
               '--trajectories', str(roster), '--dry-run']
    result = subprocess.run(command, text=True, capture_output=True)
    (Path(artifacts)/'ensemble-dry-run.log').write_text('command: '+json.dumps(command)+'\n'+result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout+result.stderr
    assert result.stdout.count('dx=3000m') == len(entries)
    assert 'weight=0.666666666667' in result.stdout
    assert 'execution unavailable: weighted trajectory' not in result.stdout
    assert 'dry run: nothing was fetched' in result.stdout

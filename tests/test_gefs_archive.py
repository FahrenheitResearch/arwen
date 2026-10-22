"""Measured legacy GEFS publications resolve before member/lead planning."""
from datetime import datetime, timezone
import json
import pytest
from gpuwm.fetch_routes import resolve_request
from gpuwm.source_adapters import source_for_cycle, get_source_adapter
from gpuwm.source_authorities import packaged_member_grammar, packaged_authorities
from gpuwm.member_grammar import load_member_grammar, MemberIdentityRefusal
from gpuwm.ensemble.recipes import ensemble_population, SourceTrajectory
from gpuwm.go_cli import fetch_request

@pytest.mark.parametrize('stamp,source,key', [
 ('2018-07-26T18','gefs-v11-flat','gefs.20180726/18/gec00.t18z.pgrb2af000'),
 ('2018-07-27T00','gefs-v11','gefs.20180727/00/pgrb2a/gec00.t00z.pgrb2af00'),
 ('2020-09-23T06','gefs-v11','gefs.20200923/06/pgrb2a/gec00.t06z.pgrb2af00'),
 ('2020-09-23T12','gefs','gefs.20200923/12/atmos/pgrb2ap5/gec00.t12z.pgrb2a.0p50.f000'),
])
def test_publication_boundaries(stamp, source, key):
 cycle=datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc)
 plan=resolve_request('gefs',cycle=cycle,hours=6)
 assert plan.source_id==source
 assert plan.objects[0].key==key
 assert SourceTrajectory('gefs',cycle,'p01').source==source
 assert fetch_request({'source':'gefs','cycle':stamp})['source']==source

@pytest.mark.parametrize('stamp,count,last', [('2018-01-01T00',21,'p20'),('2019-02-01T00',21,'p20'),('2021-01-01T00',31,'p30')])
def test_population_uses_publication_before_member_validation(stamp,count,last):
 population=ensemble_population('gefs',datetime.fromisoformat(stamp).replace(tzinfo=timezone.utc))
 assert len(population)==count
 assert population[-1].member==last

@pytest.mark.parametrize('stamp', ['2018-01-01T00','2019-02-01T00'])
def test_unpublished_member_and_lead_refuse(stamp):
 cycle=datetime.fromisoformat(stamp)
 with pytest.raises(ValueError):resolve_request('gefs',cycle=cycle,hours=3,cadence=3)
 with pytest.raises(ValueError):resolve_request('gefs',cycle=cycle,hours=6,member='p21')

@pytest.mark.parametrize('stamp,suffix', [('2018-01-01T00','pgrb2af102'),('2019-02-01T00','pgrb2af102')])
def test_legacy_lead_width_is_minimum_not_truncation(stamp,suffix):
 cycle=datetime.fromisoformat(stamp)
 plan=resolve_request('gefs',cycle=cycle,hours=0,start_hour=102,member='p20')
 assert plan.objects[0].key.endswith(suffix)
 adapter=get_source_adapter(plan.source_id)
 grammar=load_member_grammar(packaged_member_grammar(adapter.member_set))
 path=grammar.relative_path('p20','pgrb2a',cycle,102)
 assert path==plan.objects[0].key
 assert grammar.classify_relative_path(path).member_id=='p20'

@pytest.mark.parametrize('stamp', ['2016-12-31T18','2010-02-27T00'])
def test_cycle_before_public_archive_refuses(stamp):
 with pytest.raises(ValueError,match='2017-01-01'):
  resolve_request('gefs',cycle=datetime.fromisoformat(stamp),hours=6)

def test_legacy_column_is_not_the_v12_ladder_or_population():
 adapter=get_source_adapter('gefs-v11')
 mapping=json.loads(packaged_authorities(adapter.packaged_profile)['mapping'].read_text())
 levels=mapping['coordinates']['vertical']['levels']
 assert len(levels)==26 and min(levels)==1000 and max(levels)==100000
 grammar=load_member_grammar(packaged_member_grammar(adapter.member_set))
 assert grammar.declared_member_count==21
 with pytest.raises(MemberIdentityRefusal):grammar.member('p21')
 assert adapter.forcing_interval_seconds==21600

def test_publication_selection_uses_utc_and_leaves_other_sources():
 assert source_for_cycle('gefs-ensemble','2020-09-23T06')=='gefs-v11'
 assert source_for_cycle('gefs','latest')=='gefs'
 assert source_for_cycle('gfs','2019-02-01T00')=='gfs'

@pytest.mark.parametrize('source', ['gefs-v11-flat','gefs-v11'])
def test_prepared_identity_preserves_the_publication(source, tmp_path):
 from gpuwm.stage_cli import packaged_source_of
 adapter=get_source_adapter(source)
 authorities=packaged_authorities(adapter.packaged_profile)
 evidence=tmp_path/'source-evidence';evidence.mkdir()
 for role in ['mapping','composition']:
  (evidence/(role+'.json')).write_bytes(authorities[role].read_bytes())
 (evidence/'input-manifest.json').write_text(json.dumps({'schema':'gpuwm-mapped-composition-inputs-v1'}))
 assert packaged_source_of(tmp_path)==source

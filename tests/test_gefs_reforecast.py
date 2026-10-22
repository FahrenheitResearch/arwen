"""Retrospective file groups, native indexed leads and physical donors."""
from datetime import datetime
from pathlib import Path
import json
import pytest
from gpuwm.fetch_routes import resolve_request
from gpuwm.source_adapters import source_for_cycle,get_source_adapter
from gpuwm.source_authorities import packaged_member_grammar,packaged_authorities
from gpuwm.member_grammar import load_member_grammar
from gpuwm.mapped_source import load_mapping
from gpuwm.mapped_composition import load_composition

@pytest.mark.parametrize('stamp,count,last',[('2019-02-01T00',5,'p04'),('2019-02-06T00',11,'p10')])
def test_real_publication_member_namespace(stamp,count,last):
 cycle=datetime.fromisoformat(stamp);source=source_for_cycle('gefs-reforecast',cycle)
 adapter=get_source_adapter(source);grammar=load_member_grammar(packaged_member_grammar(adapter.member_set))
 assert grammar.declared_member_count==count
 assert grammar.member(last).verification.ensemble_size==10
 key=grammar.relative_path(last,'pressure',cycle,3)
 assert key.startswith('GEFSv12/reforecast/2019/'+cycle.strftime('%Y%m%d%H'))
 assert grammar.classify_relative_path(key).member_id==last
 with pytest.raises(ValueError):grammar.member('p%02d'%count)

def test_leads_are_slices_of_variable_objects_and_never_f000():
 plan=resolve_request('gefs-reforecast',cycle=datetime(2019,2,1),start_hour=3,hours=6,member='p02')
 assert {o.lead for o in plan.objects}=={3,6,9}
 assert len({o.relpath for o in plan.objects})==len(plan.objects)
 urls={o.url for o in plan.objects};assert len(urls)*3==len(plan.objects)
 assert all(o.member_index=={'format':'wgrib2'} for o in plan.objects)
 with pytest.raises(ValueError):resolve_request('gefs-reforecast',cycle=datetime(2019,2,1),hours=6,member='p05')
 with pytest.raises(ValueError):resolve_request('gefs-reforecast',cycle=datetime(2019,2,1),hours=6)

def test_day_ten_changes_group_and_cadence_without_upper_file_duplication():
 plan=resolve_request('gefs-reforecast',cycle=datetime(2019,2,1),start_hour=240,hours=12,member='c00')
 assert {o.lead for o in plan.objects}=={240,246,252}
 assert all(('Days:1-10/' if o.lead==240 else 'Days:10-16/') in o.url for o in plan.objects)
 assert not any('abv700mb' in o.url for o in plan.objects if o.lead>240)

def test_complete_source_and_declared_surface_contributions():
 a=packaged_authorities(get_source_adapter('gefs-reforecast').packaged_profile)
 mapping=load_mapping(a['mapping']);load_composition(a['composition'],a['mapping'])
 assert mapping['coordinates']['horizontal']['normalization']=='coarsest_exact_subset'
 assert len(mapping['coordinates']['vertical']['levels'])==25
 comp=json.loads(Path(a['composition']).read_text())
 assert comp['field_sources']['seaice']['time_alignment']=='source_cycle_analysis_select_broadcast'
 assert comp['field_sources']['seaice']['grid_alignment']=='nearest_coordinate'
 for stem in ['gefs-retrospective-land','ncep-r1-seaice-netcdf']:
  load_mapping(Path(a['mapping']).parent/('rw-wps-'+stem+'.mapping.json'))

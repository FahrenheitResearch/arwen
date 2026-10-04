"""Actual source owners, with deterministic clocks and no weather acquisition."""
from datetime import datetime, timezone

import pytest


def test_capability_serializer_reads_immutable_dataclass_fields_without_deepcopy():
    from dataclasses import dataclass
    from types import MappingProxyType
    from gpuwm.background_contract import _plain
    @dataclass(frozen=True)
    class FileRule:
        roles: object
    original = FileRule(MappingProxyType({"fields": ("temperature", "pressure"),
                                         "defaults": MappingProxyType({"cycle": 0})}))
    actual = _plain(original)
    assert actual == {"roles": {"fields": ["temperature", "pressure"], "defaults": {"cycle": 0}}}
    actual["roles"]["defaults"]["cycle"] = 12
    assert original.roles["defaults"]["cycle"] == 0

from gpuwm.background_contract import capability, catalog, from_record, plan
from gpuwm.da.background import plan_background_cycle
from gpuwm.fetch import gfs_forecast_hours
from gpuwm.runplan import drivability_for
from gpuwm.source_adapters import source_adapters


def moment(hour):
    return datetime(2026, 9, 10, hour, tzinfo=timezone.utc)


def test_catalog_reuses_every_declared_preparation_chain():
    rows={row['source']:row for row in catalog()['sources']}
    assert set(rows)=={row.source_id for row in source_adapters()}
    for adapter in source_adapters():
        row=rows[adapter.source_id]
        assert 'error' not in row,row
        assert row['local_preparation_operation']==drivability_for(adapter.source_id).get('chain')


def test_nonzero_initial_lead_fetches_a_duration_not_the_end_lead():
    selection=plan('gfs',init=moment(4),now=moment(12),run_seconds=3600)
    hints=selection.fetch_hints()
    assert hints['cycle']=='2026-09-10T00'
    assert hints['forecast_start_hour']==4 and hints['hours']==1
    assert tuple(gfs_forecast_hours(hints['hours'],hints['cadence'],hints['forecast_start_hour']))==(4,5)
    assert from_record(selection.record())==selection


def test_automatic_cycle_uses_the_declared_complete_publication_window():
    selection=plan('gfs',init=moment(6),now=moment(9),run_seconds=900)
    assert selection.cycle=='2026-09-10T00:00:00+00:00'
    assert selection.init=='2026-09-10T06:00:00+00:00'
    assert selection.forecast_leads==(6,7)


def test_native_hourly_search_reaches_the_earlier_extended_cycle():
    selection=plan('hrrr',init=moment(5),now=moment(7),run_seconds=30*3600)
    assert selection.cycle=='2026-09-10T00:00:00+00:00'
    assert selection.forecast_leads==tuple(range(5,36))


def test_coarser_source_ladder_is_retained_in_compatibility_receipt():
    selection=plan('gem-gdps',init=moment(0),now=moment(12),run_seconds=7*3600)
    compatibility=plan_background_cycle('gem-gdps',init=moment(0),now=moment(12),run_seconds=7*3600)
    assert selection.forcing_interval_seconds==3*3600
    assert compatibility.forecast_hours==selection.forecast_leads==(0,3,6,9)


@pytest.mark.parametrize('source',['gdas','era5','era5-l137','20crv3','20crv3-cf','mapped'])
def test_existing_preparation_owners_are_not_relabelled_missing(source):
    row=capability(source)
    verdict=drivability_for(source)
    assert row['local_preparation_operation']==verdict.get('chain')
    assert row['preparation_routes']==list(verdict.get('routes',()))


@pytest.mark.parametrize('value',[True,False,0,1.0,''])
def test_source_identity_is_not_coerced_to_a_default(value):
    with pytest.raises((ValueError,TypeError)):
        plan(value,init=moment(0),now=moment(12),run_seconds=900)

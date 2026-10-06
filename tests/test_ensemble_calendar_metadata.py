"""Real prepared calendar fields retain exact grouping and binding authority."""
from datetime import date, datetime, timedelta, timezone, tzinfo
from dataclasses import replace

import pytest

from gpuwm.config import RunConfig
from gpuwm.experiment import DomainConfig
from gpuwm.ensemble.batch_state import _exact_key, BatchStateUnsupported
from gpuwm.ensemble.packed_schedule import _exact
from gpuwm.ensemble.batch_rrtmgp import _scalar, _config_identity


@pytest.mark.parametrize("value", [date(2026, 10, 4), datetime(2026, 10, 4, 12, 1, 2, 123456),
    datetime(2026, 10, 4, 12, 1, 2, 123456, tzinfo=timezone.utc),
    datetime(2026, 10, 4, 12, tzinfo=timezone(timedelta(hours=-7), "offset-zone"))])
def test_all_component_doors_accept_and_repeat_typed_calendar_authority(value):
    for key in (_exact_key, _exact, _scalar):
        assert key(value) == key(value.replace())
        assert key(value) != key(value.isoformat())


@pytest.mark.parametrize("key", [_exact_key, _exact, _scalar])
def test_calendar_date_time_precision_fold_and_zone_names_do_not_normalize(key):
    base = datetime(2026, 10, 4, 12, 1, 2, 123456)
    assert key(base) != key(base.replace(microsecond=123457))
    assert key(base) != key(base.replace(fold=1))
    assert key(base.date()) != key(datetime(2026, 10, 4))
    aware = base.replace(tzinfo=timezone.utc)
    assert key(base) != key(aware)
    assert key(aware) != key(aware.astimezone(timezone(timedelta(hours=-7))))
    assert key(aware) != key(aware.replace(tzinfo=timezone(timedelta(0), "alternate-zero-name")))


def test_real_domain_start_time_passes_all_prepared_bank_and_physics_identity_doors():
    cfg = RunConfig(nx=12, ny=9, nz=8, dx=2250.0, dy=2250.0, ztop=10000.0, dt=12.0,
                    run_seconds=60.0)
    domain = DomainConfig(1, 0, 1, 1, 1, 1, 12.0, cfg,
                          start_time=datetime(2026, 10, 4, 12, tzinfo=timezone.utc))
    same = replace(domain)
    changed = replace(domain, start_time=domain.start_time + timedelta(microseconds=1))
    for key in (_exact_key, _exact, _config_identity):
        assert key(domain) == key(same)
        assert key(domain) != key(changed)


@pytest.mark.parametrize("key", [_exact_key, _exact, _scalar])
def test_opaque_timezone_cannot_replace_future_transition_authority(key):
    class OpaqueZone(tzinfo):
        def utcoffset(self, value):
            return timedelta(0)
        def dst(self, value):
            return timedelta(0)
    value = datetime(2026, 10, 4, tzinfo=OpaqueZone())
    with pytest.raises(BatchStateUnsupported, match="future calendar transitions"):
        key(value)

"""A doctor crash and a broken native must fail; optional doors do not block."""
import json
import pytest
from tools.publish_packet_smoke import doctor_problems


def finding(name, blocking=True, detail='not installed'):
    return dict(name=name, status='missing', blocking=blocking, detail=detail)


@pytest.mark.parametrize('code,rows,accepted', [
    (0, [dict(name='python', status='verified', blocking=True)], True),
    (1, [finding('cupy (GPU runtime)'), finding('WPS_GEOG', False),
         finding('obs front door rw_asos', False), finding('thompson tables', False)], True),
    (1, 'Traceback: import failed', False),
    (2, [finding('cupy (GPU runtime)')], False),
    (0, [finding('rw_fetch', detail='shared library missing')], False),
    (1, [finding('cupy (GPU runtime)'), finding('rw_fetch', detail='shared library missing')], False),
])
def test_doctor_separates_expected_host_gaps_optional_doors_and_broken_installs(code, rows, accepted):
    assert (not doctor_problems(code, json.dumps(rows))) is accepted

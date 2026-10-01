"""The launch cap must reach the memory price before any allocation."""
from types import SimpleNamespace

import pytest

from gpuwm.core.ntiedtke import nt_tile_columns, nt_workspace_bytes


@pytest.mark.parametrize("sms,ncol,expected", [(128, 50000, 50000),
                                             (128, 100000, 50000),
                                             (82, 50000, 25000),
                                             (82, 125952, 41984),
                                             (82, 125953, 31489),
                                             (70, 16384, 16384)])
def test_tile_budget_reaches_preflight(sms, ncol, expected):
    from gpuwm.core.preflight import ntiedtke_column_workspace_bytes

    columns = nt_tile_columns(ncol, sms)
    assert columns == expected
    exp = SimpleNamespace(domains=[SimpleNamespace(run=SimpleNamespace(
        cu_physics=16, nx=ncol, ny=1, nz=50))])
    profile = SimpleNamespace(multiprocessor_count=sms)
    assert ntiedtke_column_workspace_bytes(exp, profile=profile) == nt_workspace_bytes(50, columns)

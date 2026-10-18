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


@pytest.mark.parametrize("per_sm,ncol,expected", [(128, 40962, 20481),
                                                  (128, 21760, 21760),
                                                  (128, 655362, 21141),
                                                  (512, 40962, 40962)])
def test_a_caller_narrows_the_cap_per_instance(per_sm, ncol, expected):
    """The per-multiprocessor cap is an argument (default unchanged) and the
    adapter reads it from its own instance: gpuwm-hex runs New Tiedtke at 128
    columns per SM on its global meshes, where the workspace held at the
    chunk width was the largest physics block (hex memory census item 3)."""
    from gpuwm.core.ntiedtke import NT_TILE_COLUMNS_PER_SM, NewTiedtke

    assert nt_tile_columns(ncol, 170, per_sm) == expected
    assert nt_tile_columns(ncol, 170) == nt_tile_columns(ncol, 170, NT_TILE_COLUMNS_PER_SM)
    adapter = NewTiedtke()
    assert getattr(adapter, "tile_columns_per_sm", NT_TILE_COLUMNS_PER_SM) == 512
    adapter.tile_columns_per_sm = per_sm
    assert nt_tile_columns(ncol, 170, adapter.tile_columns_per_sm) == expected

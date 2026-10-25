"""GPU checks of the shared member workspace's turns and admission (no forecast)."""
import types

import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu


def _config(**changes):
    from gpuwm.config import RunConfig
    base = RunConfig(nx=12, ny=10, nz=8, dx=3000.0, dy=3000.0, ztop=20000.0, dt=15.0, run_seconds=60.0,
                     diff_6th_opt=2, diff_6th_factor=0.12, km_opt=0)
    return types.SimpleNamespace(run=RunConfig(**{**base.__dict__, **changes}))


def test_executor_turn_leaves_the_rebuilt_lock_to_the_executor():
    from gpuwm.ensemble.sequential_workspace import SequentialMemberWorkspace
    ws = SequentialMemberWorkspace((_config(),))
    with ws.turn(1, executor=True):
        assert ws.member == 1
        with ws.rebuilt.acquire("STEP d01"):
            pass
        with pytest.raises(RuntimeError, match="cannot take the resident workspace while member 1"):
            with ws.turn(2, executor=True):
                pass
    with ws.turn(2):
        with pytest.raises(RuntimeError, match="already owned"):
            with ws.rebuilt.acquire("STEP d01"):
                pass
    assert ws.member is None and ws.turns == 2
    assert ws.turn_members == [1, 2]  # a refused nested acquisition is not a turn


def test_admission_refuses_a_member_whose_aliases_the_arena_would_break():
    from gpuwm.ensemble.sequential_workspace import SequentialMemberWorkspace, WorkspaceRefused
    plain = _config()
    ws = SequentialMemberWorkspace((plain,))
    ws.admits(plain)
    ws.admits(_config(diff_6th_factor=0.2))  # same audit, same shapes
    with pytest.raises(WorkspaceRefused, match="lifetime aliases"):
        ws.admits(_config(km_opt=4))
    with pytest.raises(WorkspaceRefused, match="does not fit"):
        ws.admits(_config(nx=40, ny=30))
    joined = SequentialMemberWorkspace((plain, _config(km_opt=4)))
    joined.admits(_config(km_opt=4))
    joined.admits(plain)

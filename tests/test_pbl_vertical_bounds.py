"""The PBL column kernels' vertical bounds, and where a user meets them.

Two surfaces, one pair of numbers each.  ``gpuwm check`` must refuse an
impossible vertical grid BEFORE a user pays for fetch and preparation, and
the launcher must refuse it by name if anything ever reaches it anyway.  The
defect these pin: a 130-level YSU configuration passed ``gpuwm check`` and
both preparation stages, then died mid-run on a bare
``ValueError: nz=130 exceeds YSU_KMAX=128``.

No device is opened here.  Every launcher rejects the level count before it
touches an array, so a host array of the right shape reaches the refusal.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.physics_vertical_contract import (
    MYJ_VERTICAL_LEVEL_BOUNDS,
    PhysicsVerticalPreflightError,
    SASE_VERTICAL_LEVEL_BOUNDS,
    SHINHONG_VERTICAL_LEVEL_BOUNDS,
    YSU_VERTICAL_LEVEL_BOUNDS,
)


def test_the_contract_bounds_are_the_launchers_own_bounds():
    """The standalone contract restates each launcher's number; pin them.

    ``gpuwm/physics_vertical_contract.py`` deliberately imports no forecast
    executor, so it cannot read these off the runtime modules.  This is what
    stops the restatement drifting from the kernel that owns it.
    """
    from gpuwm.core import myjpbl, sase_limits, shinhong, ysu

    assert YSU_VERTICAL_LEVEL_BOUNDS == ysu.VERTICAL_LEVEL_BOUNDS
    assert SHINHONG_VERTICAL_LEVEL_BOUNDS == shinhong.VERTICAL_LEVEL_BOUNDS
    assert MYJ_VERTICAL_LEVEL_BOUNDS == (myjpbl.MYJ_MIN_COLUMN_LEVELS,
                                         myjpbl.MYJ_MAX_COLUMN_LEVELS)
    assert SASE_VERTICAL_LEVEL_BOUNDS[1] == sase_limits.MAX_COLUMN_LEVELS


@pytest.mark.parametrize("define, module_name, bounds", (
    ("YSU_KMAX", "ysu", YSU_VERTICAL_LEVEL_BOUNDS),
    ("SHINHONG_KMAX", "shinhong", SHINHONG_VERTICAL_LEVEL_BOUNDS),
    ("MYJ_KMAX", "myjpbl", MYJ_VERTICAL_LEVEL_BOUNDS),
    ("SASE_KMAX", "sase", SASE_VERTICAL_LEVEL_BOUNDS),
))
def test_the_declared_ceiling_is_the_compiled_kernel_ceiling(
        define, module_name, bounds):
    """The ceiling is a compile-time array extent, so read it from the .cu."""
    from pathlib import Path

    import gpuwm.core.kernels as kernels

    source = (Path(kernels.__file__).with_name(f"{module_name}.cu")
              .read_text(encoding="utf-8"))
    line = next(ln for ln in source.splitlines()
                if ln.startswith(f"#define {define} "))
    assert int(line.split()[-1]) == bounds[1]


def _column(nz: int) -> np.ndarray:
    return np.zeros((nz, 2, 2), dtype=np.float32)


def test_ysu_refuses_too_many_levels_by_name():
    from gpuwm.core.ysu import launch_ysu

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch_ysu(*(_column(130) for _ in range(6)), None, None, None, None,
                   psfc=None, znt=None, ust=None, hfx=None, qfx=None,
                   wspd=None, br=None, psim=None, psih=None, xland=None,
                   u10=None, v10=None, dt=60.0)
    message = str(caught.value)
    assert "YSU PBL requires 4 <= nz <= 128, got nz=130" in message
    assert "gpuwm check" in message


def test_ysu_refuses_too_few_levels_by_name():
    from gpuwm.core.ysu import launch_ysu

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch_ysu(*(_column(3) for _ in range(6)), None, None, None, None,
                   psfc=None, znt=None, ust=None, hfx=None, qfx=None,
                   wspd=None, br=None, psim=None, psih=None, xland=None,
                   u10=None, v10=None, dt=60.0)
    assert "YSU PBL requires 4 <= nz <= 128, got nz=3" in str(caught.value)


def test_shinhong_refuses_out_of_range_levels_by_name():
    from gpuwm.core.shinhong import launch_shinhong

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch_shinhong(*(_column(130) for _ in range(6)), None, None, None,
                        None, None, psfc=None, znt=None, ust=None, hfx=None,
                        qfx=None, wspd=None, br=None, psim=None, psih=None,
                        xland=None, u10=None, v10=None, corf=None,
                        dt=60.0, dx=1000.0, dy=1000.0)
    assert ("Shin-Hong PBL requires 4 <= nz <= 128, got nz=130"
            in str(caught.value))


def test_myj_refuses_out_of_range_levels_by_name():
    from gpuwm.core.myjpbl import launch_myj_pbl

    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        launch_myj_pbl({"dz": _column(130)}, {"psfc": np.zeros((2, 2))},
                       None, None, None, dtturbl=60.0, flqi=False)
    assert ("MYJ PBL requires 4 <= nz <= 128, got nz=130"
            in str(caught.value))


def test_a_named_refusal_is_still_a_value_error():
    """Existing callers catch ValueError; the named class must not break them."""
    assert issubclass(PhysicsVerticalPreflightError, ValueError)


def test_the_preflight_and_the_launcher_name_the_same_thing():
    """One grammar, so a user meets the same sentence at check and at run."""
    from gpuwm.physics_compat import (
        PhysicsVerticalPreflightError as CompatError,
        validate_resolved_physics_vertical_levels,
    )

    assert CompatError is PhysicsVerticalPreflightError
    with pytest.raises(PhysicsVerticalPreflightError) as caught:
        validate_resolved_physics_vertical_levels({
            "nz": 130, "cu_physics": 0, "sf_surface_physics": 0,
            "mp_physics": 8, "bl_pbl_physics": 1, "ra_lw_physics": 0,
            "ra_sw_physics": 0, "sf_sfclay_physics": 0,
            "ra_rrtmg_variant": "rte-rrtmgp",
        })
    assert "YSU PBL requires 4 <= nz <= 128, got nz=130" in str(caught.value)

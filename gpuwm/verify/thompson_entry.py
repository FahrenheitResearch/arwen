"""Thompson's entry moment-consistency block: re-exported from its home.

The mirror moved to :mod:`gpuwm.core.thompson_entry` so the mp=28
real-data cold start can run it in the preparation-only distribution,
which omits this verification package.  Everything is imported from
there; this module keeps the name the tests and the assimilation repair
read it under.
"""
from __future__ import annotations

from gpuwm.core.thompson_entry import *  # noqa: F401,F403
from gpuwm.core.thompson_entry import (  # noqa: F401
    AM_I, BM_I, ICE_MAX_DIAMETER_M, ICE_MIN_DIAMETER_M, MU_I, MU_R,
    MVD_FACTOR, NU_C_MAX, RAIN_INITIAL_MVD_M, RAIN_MVD_MAX_M,
    RAIN_MVD_MIN_M, __all__, _nu_c,
)

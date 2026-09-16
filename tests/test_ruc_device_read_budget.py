"""How many times one RUC call on the card makes the host wait.

The profile of record (50 warm root steps of the 299x299 + 282x129 pair on a
5090) measured 3,206 blocking device-to-host reads per root step, each one a
stream synchronisation, and attributed 2,839 of them -- 89.8 per cent -- to
the RUC land-surface scheme: about 568 per call, five calls per step, almost
all of them admission checks reading one field at a time.

Two changes removed them, and both funnel through a single function each, so
counting those two functions counts the synchronisations this scheme now
initiates:

* ``ruc_validation._host_list`` -- the ONE read that resolves a whole batch
  of admission tests, however many fields the batch holds;
* ``ruc._selected`` -- the ONE read that turns a dispatch arm's boolean mask
  into column indices, after which every gather and scatter through that arm
  is free.

The CPU-tier model in ``tests/ruc_device_read_counter.py`` puts the total for
a warm production-shaped grid at 22 reads inside ``gpuwm.core.ruc``, of which
5 belong to host leaves a forecast does not run, plus roughly ten batch
flushes in the device leaves: about 27 against the 568 measured before.

**Taken on a card, 2026-09-15, an RTX 5090**, 4,096 columns per call:
**17 reads without snow** (12 batch flushes, 5 arm conversions) and **28 with
snow** (17 and 11).  The model was right to within one read of the snow case,
and the scheme now costs 17 to 28 reads per land-surface call against about
568, so RUC stops being 89.8 per cent of a root step's device-to-host reads.

The bound is set from that measurement rather than from the loose 60 it
carried while no card had printed a figure.  It is not set AT 28: the arm
conversions are one read per dispatch arm that has work, so a column mix that
lights more arms than this fixture's legitimately costs a few more.  40 is
the measured worst case plus that margin, and it is still fourteen times
below what the scheme cost before.  The test PRINTS what it finds, so a
future shape that approaches the bound is visible before it trips it.
"""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

import gpuwm.core.ruc as ruc  # noqa: E402
import gpuwm.core.ruc_validation as ruc_validation  # noqa: E402
from gpuwm.core.ruc_gpu import (RUC_DEVICE_ARRAYS,  # noqa: E402
                                RUC_SFCTMP_DEVICE_LEAVES_RESIDENT,
                                RUC_SFCTMP_DEVICE_STAGES_RESIDENT)
from gpuwm.core.ruc_runtime import (C1SN, C2SN, DEFINED_ILNB,  # noqa: E402
                                    ISNCOVR_OPT, RucRuntimeParameters)
from test_ruc_column_batching import _columns  # noqa: E402

_PARAMS = RucRuntimeParameters()

#: What the CPU-tier model predicts, and what this run should print near.
#: Measured on the card: 17 without snow, 28 with it.
_MODELLED = 27

#: What the same call cost before the admission batch and the dispatch-arm
#: index landed, from the profile of record: about 568 reads per call, five
#: calls per root step, 2,839 of the step's 3,206 device-to-host reads.
_BEFORE = 568

#: The bound this file enforces: the worst case measured on a card
#: (28 reads, the snow arm, 2026-09-15 on an RTX 5090) plus the margin
#: the module docstring prices, against the roughly 568 this call used to
#: cost.  Tightened from 60, which was a placeholder for a figure nobody had.
_BUDGET = 40


def _resident(values, ivgtyp, isltyp, *, ktau=2, dt=12.0):
    return ruc.ruc_land_surface_step(
        {name: cp.asarray(np.ascontiguousarray(array))
         for name, array in values.items()},
        dt=dt, ktau=ktau, zs=_PARAMS.zs,
        ivgtyp=cp.asarray(ivgtyp), isltyp=cp.asarray(isltyp),
        em_core=0, ilnb=DEFINED_ILNB, ilnb_chain=False, c1sn=C1SN, c2sn=C2SN,
        isncovr_opt=ISNCOVR_OPT, mminlu=_PARAMS.dataset_identifier,
        parameters=_PARAMS.bundle,
        leaves=RUC_SFCTMP_DEVICE_LEAVES_RESIDENT,
        stages=RUC_SFCTMP_DEVICE_STAGES_RESIDENT,
        arrays=RUC_DEVICE_ARRAYS)


@pytest.fixture()
def reads(monkeypatch):
    tally = {"batch": 0, "arm": 0}
    host_list = ruc_validation._host_list
    selected = ruc._selected

    def counted_host_list(flags):
        tally["batch"] += 1
        return host_list(flags)

    def counted_selected(mask, arrays, populated=True):
        if populated:
            tally["arm"] += 1
        return selected(mask, arrays, populated)

    monkeypatch.setattr(ruc_validation, "_host_list", counted_host_list)
    monkeypatch.setattr(ruc, "_selected", counted_selected)
    return tally


@pytest.mark.parametrize("snow", (False, True))
def test_one_device_land_surface_call_stays_inside_its_read_budget(
        reads, snow, capsys):
    values, ivgtyp, isltyp = _columns(4096, snow=snow, seed=5)
    _resident(values, ivgtyp, isltyp)
    total = reads["batch"] + reads["arm"]
    with capsys.disabled():
        print(f"\nRUC device reads, snow={snow}: {total} "
              f"({reads['batch']} batch flushes, {reads['arm']} arm "
              f"conversions); modelled {_MODELLED}, was about {_BEFORE}")
    assert total <= _BUDGET, (
        f"one RUC land-surface call made {total} blocking device reads, "
        f"above the {_BUDGET} measured on a card on 2026-09-15 plus its "
        "margin.  Each read is a stream synchronisation on the step path, "
        "and this scheme runs five times per root step: a regression here "
        "is host time on every step of every run, not a test preference")


def test_the_read_counter_is_wired_to_something(reads):
    """The falsification: a fixture that patched nothing would pass above."""
    values, ivgtyp, isltyp = _columns(512, snow=False, seed=6)
    _resident(values, ivgtyp, isltyp)
    assert reads["batch"] > 0 and reads["arm"] > 0

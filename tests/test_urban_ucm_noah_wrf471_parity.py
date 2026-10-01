"""The UCM's Noah coupling against WRF v4.7.1's own ``lsm``, bit for bit.

``tools/urban_wrf471_oracle/ucm_noah_oracle.F90`` calls the byte-unmodified
``lsm`` (``module_sf_noahdrv.F``) with ``sf_urban_physics = 1`` over fifteen
urban columns and one grass column for four carried steps: the three NLCD
urban types, table urban fractions plus FRC 0.05, 0.99 and 1.0, exchange
coefficients under the 1.0E-02 floors, day, a calm night (the 1 m/s wind
clamp) and rain.  WRF runs the UCM inside Noah's column loop; gpuwm runs it
after the LSM on the rural values the LSM hands over.  The fixture carries
those values as WRF held them at the UCM's entry, read through one inserted
statement whose build proves it perturbs nothing (``build_ucm_noah.sh``).

**Measured: every blended grid field (TSK HFX QFX LH GRDFLX ALBEDO QSFC UST
and the floored CHS CHS2 CQS2), every urban state word and every per-step
urban output (TS SH LH G RN PSIM PSIH GZ1OZ0 U10 V10 TH2 Q2 UST AKMS) is
bit-identical to WRF on all 60 urban column steps**, from recorded and from
carried state, and the grass column is left untouched.

What this does NOT prove: that gpuwm's Noah kernel hands over WRF's rural
values.  That is the LSM hook's own gate (the infra lane's
``tests/test_urban_noah_hook_wrf471_parity.py``), and this fixture's
``tap_*`` columns are exactly the words it has to reproduce.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu


@requires_gpu
@pytest.mark.parametrize("carried", [False, True], ids=["recorded", "carried"])
def test_noah_blend_is_bitwise_wrf(carried):
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    from gpuwm.verify.urban_ucm_oracle import (load_variant, noah_replay,
                                               port_params)

    _, table, switches = load_variant("ucm-nlcd-default")
    port, ref, untouched = noah_replay(port_params(table, switches),
                                       carried=carried)
    assert untouched, "the kernel wrote a non-urban column"
    lines = []
    for key, want in ref.items():
        got = np.asarray(port[key], np.float32)
        bad = np.nonzero(got.view(np.uint32) != want.view(np.uint32))[0]
        if bad.size:
            lines.append(f"{key}: {bad.size} rows, max "
                         f"{int(np.max(fp32_ulp_distance(got, want)))} ULP")
    assert not lines, "\n".join(lines)
    assert port["tsk"].size == 60


@requires_gpu
@pytest.mark.parametrize("deferred", [False, True], ids=["immediate", "ledger"])
def test_wrfs_first_level_fatal_stops_the_run(deferred):
    """module_sf_urban.F:825: a first level at or below ZDC+Z0C+2 m is WRF's
    fatal.  The surface step refuses it by name -- at once, or at the health
    ledger's drain when the tiled driver defers the step's status reads."""
    import cupy as cp

    from gpuwm.core import health_ledger
    from gpuwm.core.urban_ucm import UrbanCanopyError
    from gpuwm.verify.urban_ucm_oracle import (load_variant, noah_replay,
                                               port_params)

    _, table, switches = load_variant("ucm-nlcd-default")

    def low_first_level(state, fields):
        state["rural"]["zlvl"][...] = cp.float32(4.0)

    ledger = health_ledger.HealthLedger(label="urban test") if deferred else None
    with pytest.raises(UrbanCanopyError, match=r"ZDC\+Z0C\+2m"):
        with health_ledger.deferring(ledger):
            noah_replay(port_params(table, switches), carried=False,
                        mutate=low_first_level)
            if ledger is not None:
                assert ledger.records > 0
                ledger.drain()


def test_noah_fixture_reaches_what_it_claims():
    from gpuwm.verify.urban_ucm_oracle import load_noah

    rows = load_noah()
    urban = rows["tapped"] == 1
    assert int(urban.sum()) == 60 and int((~urban).sum()) == 4
    assert np.any(rows["tap_chs"][urban] < 0.01)       # the floors fire
    assert np.all(rows["chs"][urban] >= np.float32(0.01))
    assert np.any(rows["frc"][urban] == 1.0) and np.any(rows["frc"][urban] < 0.1)
    assert np.any(rows["tap_rainbl"][urban] > 0)
    ua = np.sqrt(rows["u1"] ** 2 + rows["v1"] ** 2)
    assert np.any(ua[urban] < 1.0)                      # the 1 m/s clamp

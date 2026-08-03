"""The dry-CBL case: registry contract, and both closures configurable.

CPU-only.  Integrating the case is a GPU job covered by the closure's own
suites and by the run receipts; what is worth a fast test is the part
that silently rots -- the case-registry contract, the gate convention,
and whether a second closure can still be selected on it at all.
"""
from __future__ import annotations

import pytest

from gpuwm.config import SASE_PBL_SCHEME, validate_run_config
from gpuwm.verify.cases import cbl_dry


def test_the_case_registry_discovers_it_with_both_capabilities():
    from gpuwm.verify import cases

    manifest = {entry["name"]: entry for entry in cases.manifest()}
    assert "cbl_dry" in manifest, sorted(manifest)
    capabilities = set(manifest["cbl_dry"]["capabilities"])
    assert {"verify", "script"} <= capabilities, capabilities


def test_nan_is_reported_but_never_a_gate_row():
    """The driver checks ``nan`` first and by itself.

    Listing it in GATES too would gate a zero-valued metric with a zero
    bound, and the driver's intervals are OPEN -- so the case would fail
    precisely when nothing was wrong.  That is not hypothetical: it is
    what this case did before the convention was matched.
    """
    assert "nan" not in cbl_dry.GATES
    for name, (lo, hi) in cbl_dry.GATES.items():
        assert lo is None or hi is None or lo < hi, name


@pytest.mark.parametrize("selector", [1, 5, SASE_PBL_SCHEME])
def test_every_pbl_closure_this_engine_has_can_drive_the_case(selector):
    """The case's whole purpose is comparison, so a closure that cannot
    be configured on it is a case defect, not a closure defect."""
    cfg = cbl_dry.default_config(selector)
    assert validate_run_config(cfg) is cfg
    assert cfg.bl_pbl_physics == selector


def test_the_experimental_closure_gets_km_opt_zero_and_the_others_do_not():
    """The one configuration difference between the arms, and its reason.

    SASE supplies the mixing the km_opt operator would otherwise apply.
    If this case handed it km_opt=4 as well, the comparison would be
    between one closure and another closure PLUS a Smagorinsky operator
    -- and the difference would be attributed to the closure.
    """
    assert cbl_dry.default_config(SASE_PBL_SCHEME).km_opt == 0
    assert cbl_dry.default_config(1).km_opt == 4
    assert cbl_dry.default_config(5).km_opt == 4


def test_the_gates_admit_a_range_of_closures_not_one_profile():
    """A validity case must not encode a preferred answer.

    Each bound is checked to be loose enough that two closures differing
    by a plausible margin both sit inside it -- the mixed-layer depth
    band spans better than a factor of three, and the warming band better
    than two orders of magnitude.  A gate tight enough to separate two
    reasonable closures would make this case an unratified referee.
    """
    lo, hi = cbl_dry.GATES["mixed_layer_depth"]
    assert hi / lo >= 3.0
    lo, hi = cbl_dry.GATES["theta_surface_warming"]
    assert hi / lo >= 100.0
    lo, hi = cbl_dry.GATES["surface_heat_consistency"]
    assert lo < 1.0 < hi, "order unity must sit strictly inside the band"

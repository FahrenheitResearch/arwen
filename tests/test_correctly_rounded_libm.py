"""``exp_cr``/``log_cr`` give the correctly rounded double on every host.

The point of the module under test is that its answers do not depend on whose
C library the interpreter is linked against, so the assertions here are
absolute values -- the correctly rounded binary64 result, spelled in hex so a
one-ulp move is visible in the text of the diff -- and not a comparison
against :mod:`math`.  Comparing against :mod:`math` is what the module exists
to stop doing.

Provenance of the expected values: computed at 300 bits with ``mpmath`` and
pasted here.  ``mpmath`` is not a declared dependency of this project, so the
cross-check against it is a separate test that skips when it is absent; the
pinned table is what runs everywhere.
"""

from __future__ import annotations

import math

import pytest

from gpuwm.core.correctly_rounded_libm import exp_cr, log_cr

#: ``exp(0.61 * log(3000))``.  The one argument in the aerosol-aware Thompson
#: droplet-number ladder where glibc and the Microsoft UCRT disagree: the
#: exact value is 132.142940103429296889..., glibc returns the correctly
#: rounded 0x1.08492f71fb081p+7 and the UCRT returns 0x1.08492f71fb080p+7.
#: That single bin moved DERIVED_CONSTANT_SHA256 and failed the whole mp=28
#: tier on Windows.
_THE_DIVERGENCE = 0.61 * math.log(3000.0)

_EXP_CASES = (
    (_THE_DIVERGENCE, "0x1.08492f71fb081p+7"),
    (1.0, "0x1.5bf0a8b145769p+1"),
    (0.5, "0x1.a61298e1e069cp+0"),
    (2.0, "0x1.d8e64b8d4ddaep+2"),
    (10.0, "0x1.5829dcf950560p+14"),
    (-5.0, "0x1.b993fe00d5376p-8"),
    (20.5, "0x1.7d6c4f0bcdd5cp+29"),
)

_LOG_CASES = (
    (3000.0, "0x1.003429c1da031p+3"),
    (1.0, "0x0.0p+0"),
    (2.0, "0x1.62e42fefa39efp-1"),
    (0.5, "-0x1.62e42fefa39efp-1"),
    (1.0e-06, "-0x1.ba18a998fffa0p+3"),
    (121.97554094669343, "0x1.3371cbb56c03bp+2"),
    (8.704432e18, "0x1.5ce20684add5dp+5"),
)


@pytest.mark.parametrize("argument,expected", _EXP_CASES)
def test_exp_cr_is_the_correctly_rounded_double(argument, expected):
    assert exp_cr(argument).hex() == expected


@pytest.mark.parametrize("argument,expected", _LOG_CASES)
def test_log_cr_is_the_correctly_rounded_double(argument, expected):
    assert log_cr(argument).hex() == expected


def test_log_cr_of_one_is_exactly_zero():
    """The ladder adds ``log(edges[0])`` with ``edges[0] == 1.0``.

    A helper that returned -0.0 or a subnormal here would perturb every bin
    through the addition, so the identity is asserted rather than assumed.
    """
    value = log_cr(1.0)
    assert value == 0.0
    assert math.copysign(1.0, value) == 1.0


def test_the_ladder_the_pin_is_built_from_is_reproduced_exactly():
    """All hundred ``t_Nc`` edges, as the contract module builds them.

    This is the loop from ``_build_droplet_bins``.  If it ever disagrees with
    the module under test, ``DERIVED_CONSTANT_SHA256`` moves, so pinning the
    two endpoints and the divergence bin here localizes the cause to this
    module rather than to the hundred-element digest.
    """
    log_of_span = log_cr(3000.0 / 1.0)
    edges = [exp_cr(n / 100.0 * log_of_span + log_cr(1.0))
             for n in range(1, 100)]
    assert len(edges) == 99
    assert edges[0].hex() == "0x1.1556d26ffbd7dp+0"      # n = 1
    assert edges[60].hex() == "0x1.08492f71fb081p+7"     # n = 61, the one
    assert edges[-1].hex() == "0x1.5a2586cbcc658p+11"    # n = 99


def test_the_helpers_are_not_merely_the_hosts_libm():
    """At least one of the pinned points must be reachable only by this path.

    On a host whose libm is correctly rounded everywhere in the table this is
    vacuous -- but it is exactly the hosts where it is NOT vacuous that the
    module exists for, and there the assertion is the whole story.  Stated as
    a report rather than a gate so it cannot fail on a good libm.
    """
    disagreements = [x for x, _ in _EXP_CASES if math.exp(x) != exp_cr(x)]
    disagreements += [x for x, _ in _LOG_CASES if math.log(x) != log_cr(x)]
    print(f"\nhost libm disagrees with the correctly rounded result at "
          f"{len(disagreements)} of {len(_EXP_CASES) + len(_LOG_CASES)} "
          f"pinned points: {disagreements}")


def test_against_an_independent_arbitrary_precision_reference():
    """The pinned table is not self-certifying; mpmath is the second opinion."""
    mp = pytest.importorskip("mpmath", reason="mpmath is not a dependency")
    mp.mp.prec = 300
    for argument, _ in _EXP_CASES:
        assert exp_cr(argument) == float(mp.exp(mp.mpf(argument))), argument
    for argument, _ in _LOG_CASES:
        assert log_cr(argument) == float(mp.log(mp.mpf(argument))), argument

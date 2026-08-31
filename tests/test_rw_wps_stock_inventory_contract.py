"""The two halves of the stock-WRF inventory contract, held equal.

``gpuwm/wrf_physics_inventory.py`` is the PRODUCER: it derives one
``wrfinput`` package inventory per ``mp_physics`` from WRF v4.6.1
``Registry/Registry.EM_COMMON`` package declarations and field I/O flags,
and ``gpuwm/namelist_compat.py`` publishes those rows inside the
``rw-wps.namelist-support.v1`` report.
``tools/rw_wps/crates/rw-wps/src/namelist.rs`` is the CONSUMER: it
re-validates that report and refuses any ``mp_physics`` it has no evidenced
package contract for.

The concrete breakage this gate prevents: the consumer's id set was written
as a bare ``matches!(domain.mp_physics, 6 | 8 | 10)`` and then stayed there
while the producer grew rows for 18, 28 and 50.  A scheme added to the
producer therefore became UNREACHABLE through the frontend -- the documented
step-one preflight before ``wrfinput_dNN`` is built and an unchanged WRF is
launched -- and said so with a message that named no breakage.  mp=50 (P3
one-category) is the row that made it visible: the producer inventories
``package p3_1category`` in full and the frontend refused the report.

The two directions are bound separately, because they fail differently:

* the consumer must never admit an id the producer cannot inventory, which
  would be a widened gate certifying an unchecked package member list;
* an id the producer inventories and the consumer still refuses must carry a
  RECORDED reason beside the consumer's set, in the pattern
  ``DELIBERATE_STALE_SITES`` established for mp=28.  A silent omission is
  not a refusal.

This is a text property of the Rust source, so it holds with no Rust
toolchain installed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from gpuwm.wrf_physics_inventory import supported_stock_wrf_mp_physics

_ROOT = Path(__file__).resolve().parents[1]
_NAMELIST_RS = (
    _ROOT / "tools" / "rw_wps" / "crates" / "rw-wps" / "src" / "namelist.rs"
)

#: The consumer's declaration, together with the doc-comment block that
#: records why the set stops where it does.
_DECLARATION = re.compile(
    r"(?P<reasons>(?:^/{3}.*\n)+)"
    r"^const STOCK_WRF_INVENTORIED_MP_PHYSICS: &\[u16\] = &\[(?P<ids>[^\]]*)\];",
    re.M,
)


def _consumer() -> tuple[frozenset[int], str]:
    match = _DECLARATION.search(_NAMELIST_RS.read_text(encoding="utf-8"))
    if match is None:
        pytest.fail(
            "tools/rw_wps/crates/rw-wps/src/namelist.rs no longer declares "
            "STOCK_WRF_INVENTORIED_MP_PHYSICS with a recorded-reason block. "
            "The stock-WRF inventory contract has a consuming half; it must "
            "stay a named set with its exclusions written down, not an "
            "inline literal."
        )
    ids = frozenset(
        int(value) for value in match.group("ids").replace(",", " ").split()
    )
    return ids, match.group("reasons")


def test_frontend_never_admits_an_uninventoried_scheme() -> None:
    """A widened gate would certify a package member list nobody checked."""

    consumer, _ = _consumer()
    producer = frozenset(supported_stock_wrf_mp_physics())
    assert consumer <= producer, sorted(consumer - producer)


def test_p3_mp50_is_reachable_through_the_rust_frontend() -> None:
    """P3's inventory is evidenced on both halves, so it must forward."""

    consumer, _ = _consumer()
    assert 50 in supported_stock_wrf_mp_physics()
    assert 50 in consumer


def test_every_refused_inventory_row_records_its_reason() -> None:
    """An omission is not a refusal; each one names what it prevents."""

    consumer, reasons = _consumer()
    for mp_physics in sorted(set(supported_stock_wrf_mp_physics()) - consumer):
        assert f"mp={mp_physics}" in reasons, (
            f"gpuwm inventories mp_physics={mp_physics} and the rw-wps "
            "frontend refuses it with no recorded reason. Add the breakage "
            "the refusal prevents beside STOCK_WRF_INVENTORIED_MP_PHYSICS, "
            "or admit the id."
        )

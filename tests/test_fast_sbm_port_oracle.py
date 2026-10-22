"""WOOF's mp_physics = 30 port (gpuwm/core/kernels/fast_sbm_*.cuh) graded
against WRF's own Fortran.

The device headers compile on the host (tools/fast_sbm_host) in WRF's two
precisions: float64 (WRF's source promoted, the O64 oracle build) and
float32 (WRF as it runs, O32).  Under the ``wrf_v480`` behaviour every
full FAST_SBM call the oracle recorded is reproduced bit for bit: the warm
column KC-1 (16 calls), the cold column KC-2 in the hail build (7) and the
graupel build (3), and the low-edge tile (3), in both precisions, along
with every derived table of FAST_HUCMINIT.  The corrected behaviour (the
default) differs from WRF only through the catalogued defects.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.fast_sbm_host import host as H

try:
    H.library()
    _HOST = None
except H.HostBuildUnavailable as exc:  # pragma: no cover - environment
    _HOST = str(exc)

pytestmark = pytest.mark.skipif(_HOST is not None, reason=f"host build unavailable: {_HOST}")

from tools.fast_sbm_host import replay as RP  # noqa: E402
from tools.fast_sbm_wrf480_oracle import fixtures as fx  # noqa: E402

BUILDS = ("o64", "o32")
#: The two tests that compare with the oracle's bits.  The others compare
#: with no glibc-computed bits and run wherever the host build does.
glibc_bits = pytest.mark.skipif(not H.GLIBC_LIBM, reason=H.NOT_GLIBC_REASON)
FAMILIES = (("kc1", 16), ("kc2", 7), ("kc2g", 3), ("kc2edge", 3))


@glibc_bits
@pytest.mark.parametrize("build", BUILDS)
def test_derived_tables_are_wrfs_bit_for_bit(build):
    t = fx.load_fixture(f"tables_{build}")
    der = {k.split(".", 1)[1]: v[0] for k, v in t.items() if k.startswith("derived.")}
    d = RP.host_for(build, H.flags_wrf()).derived()
    pairs = (("xl_mg", "xl_mg_6078"), ("ima", "ima"), ("chucm", "chucm"),
             ("ecoal_wrf", "ecoalmassm"), ("prob", "prob"),
             ("gain_var_new", "gain_var_new"), ("dropradii", "dropradii"),
             ("fccnr_con", "fccnr_con"), ("fccnr_mar", "fccnr_mar"),
             ("xccn", "xccn"), ("rccn", "rccn"))
    for mine, theirs in pairs:
        want = np.asarray(der[theirs], dtype=np.float64).ravel(order="F")
        assert np.array_equal(d[mine], want), mine
    assert int(d["ikr_spon_break"][0]) == int(der["ikr_spon_break"])


def test_own_bin_efficiencies_match_the_packed_tables():
    """D10 corrected: each bin uses its own fall speed; compare the pinned pack."""
    mine = RP.host_for("o64", H.flags_corrected()).derived()["ecoal_own"]
    with np.load(H.REPO_ROOT / "gpuwm/data/fast_sbm/tables-float64.npz") as packed:
        assert np.array_equal(mine, packed["ecoal_own"])
    wrf = RP.host_for("o64", H.flags_wrf()).derived()["ecoal_wrf"]
    assert int(np.sum(mine != wrf)) > 200


@glibc_bits
@pytest.mark.parametrize("build", BUILDS)
@pytest.mark.parametrize("family,count", FAMILIES)
def test_every_recorded_fast_sbm_call_is_reproduced_bit_for_bit(family, count, build):
    n = 0
    for step, status, _tally, got, want in RP.run_replays(family, build):
        assert status == 0, (family, build, step, status)
        for field in RP.POST_FIELDS:
            g, w = np.asarray(got[field]), np.asarray(want[field])
            assert g.dtype == w.dtype, field
            diff = int(np.sum(g != w))
            assert diff == 0, (family, build, step, field, diff)
        n += 1
    assert n == count


def test_the_replay_gate_can_fail():
    """Negative control: one changed constant (the fusion heat of FREEZ,
    J_W_MELT and riming moved from WRF's 333/320 to 334, finding D13) breaks
    the bitwise gate on the cold column."""
    flags = H.flags_wrf()
    flags["fusion_heat_334"] = 1
    host = RP.host_for("o64", flags)
    changed = 0
    for _step, status, _t, got, want in RP.run_replays("kc2", "o64", flags=flags, host=host):
        assert status == 0
        changed += int(np.sum(got["th"] != want["th"]))
    assert changed > 0


def test_calls_are_not_trivial():
    """Each replayed call moves the state: the gate compares real work."""
    case = RP.Case("kc2", "o64")
    for _step, pre, post in fx.iter_replay("kc2_replay_o64"):
        s = case.state(pre.fields)
        before = np.asarray(case.tile_view(s, "th"), dtype=np.float64)
        after = np.asarray(post.fields["th"], dtype=np.float64)
        assert np.max(np.abs(after - before)) > 1e-3


@pytest.mark.parametrize("family", ("kc1", "kc2"))
def test_corrected_preset_runs_every_recorded_call(family):
    """The default behaviour completes every recorded call (no refusal) and
    differs from WRF where a catalogued defect acts."""
    flags = H.flags_corrected(hail=1)
    host = RP.host_for("o64", flags)
    moved = 0
    for _step, status, _t, got, want in RP.run_replays(family, "o64", flags=flags, host=host):
        assert status == 0
        moved += int(np.sum(got["qc"] != want["qc"]))
    assert moved > 0

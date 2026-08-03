"""The mp=28 column oracle must stay re-derivable from its stated inputs.

The aerosol-aware sibling of ``tests/test_thompson_oracle_provenance.py``,
and it exists because of what that file found.  The classic mp=8 oracle was
self-contradictory for four waves: ``p_pa`` and ``pii`` are built in
``run_column.F90`` from ``z`` alone, inside a loop that never reads the
scenario id, so every fixture sharing a ``dz`` must agree on them exactly --
and three did not.  The cause was not a different machine, which is what the
mp=8 freeze receipt had recorded.  From GCC 12 on, ``-O2`` implies
``-ftree-vectorize``; a vectorised ``exp``/``pow``/``log`` loop links glibc's
libmvec SIMD entry points (``_ZGVbN4v_expf``, ``_ZGVbN4vv_powf``,
``_ZGVbN2v_exp``, ``_ZGVbN2vv_pow``, ``_ZGVbN2v_log10``) in place of the
scalar routines, libmvec is not bit-identical to scalar libm, and whether a
given loop vectorises is a cost-model decision that depends on how much
UNRELATED source surrounds it.  ``run_column.F90`` was 227 lines when three
fixtures were generated and roughly a thousand when the other 43 were.  The
harness merely growing changed the oracle's numbers.

``run_column_aero.F90`` builds its base state the same way (``p0*exp(-z/8000)``
and ``(p/p0)**rd_over_cp`` at :231-232, ``max(1e-5, 0.014*exp(-z/2500))`` at
:273) and ``build_aero.sh`` used the same bare ``-O2``.  Same exposure.

WHAT WAS MEASURED HERE, and it is the reassuring half.  The aerosol deck was
rebuilt from an unmodified WRF v4.6.1 checkout with ``-fno-tree-vectorize``,
on the toolchain ``README-AEROSOL.md`` records -- GNU Fortran 13.3.0
(Ubuntu 13.3.0-6ubuntu2~24.04.1), Ubuntu 24.04, glibc 2.39, one RTX 5090 box
-- and ALL 49 committed CSVs plus ``tnccn_act_native.bin`` came back BYTE FOR
BYTE identical.  A control build at plain ``-O2`` on the same machine linked
``_ZGVbN2v_exp``, ``_ZGVbN2vv_pow`` and ``_ZGVbN4vv_powf`` and was refused by
the new guard, so the vectoriser really does reach this harness; it simply
never reached the arithmetic these fixtures record.  The deck was correct by
where the cost model happened to land, exactly as the mp=8 lane predicted, and
``-fno-tree-vectorize`` is what converts that from luck into a property.

Four promises, the first three the same three the classic file holds.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import pathlib

import pytest


_REPO = pathlib.Path(__file__).parents[1]
_ORACLE = _REPO / "gpuwm" / "data" / "thompson" / "oracle-aero"
_HARNESS = _REPO / "tools" / "thompson_wrf461_oracle"
_RECEIPT = _ORACLE / "PROVENANCE.txt"

_HARNESS_FILES = ("build_aero.sh", "stub_wrf.F90", "generate_tables.F90",
                  "dump_ccn_table.F90", "probe_aero_functions.F90",
                  "run_column_aero.F90")

#: The one committed CSV ``build_aero.sh`` does not produce, so the receipt
#: cannot list it.  ``aero-exit-temperature.csv`` is written by
#: ``build_exit_temperature_aero.sh``, which builds the harness TWICE -- once
#: pristine, once instrumented -- and refuses to emit unless all 44 pristine
#: outputs are byte-identical between the two.  Named here rather than
#: tolerated, so a SECOND unreceipted file cannot appear quietly.
_RECEIPT_EXEMPT = frozenset({"aero-exit-temperature.csv"})

#: Every build script that compiles the aerosol harness.  All of them must
#: carry the flag; one that does not is a path back to a libmvec build.
_BUILD_SCRIPTS = ("build_aero.sh", "build_aero_instrumented.sh",
                  "build_aero_probes.sh", "build_probe_warm_frozen.sh")


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt() -> dict[str, dict[str, str]]:
    """Parse the INI-ish receipt ``build_aero.sh`` writes."""
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in _RECEIPT.read_text(encoding="ascii").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
            continue
        assert current is not None, f"key outside a section: {line}"
        key, _, value = line.partition(" = ")
        current[key] = value
    return sections


def test_receipt_exists_and_names_every_input():
    assert _RECEIPT.is_file(), (
        "the aerosol oracle has no provenance receipt; regenerate it with "
        "tools/thompson_wrf461_oracle/build_aero.sh")
    receipt = _receipt()
    for section in ("wrf_source", "aerosol_asset", "harness_source",
                    "toolchain", "binaries", "table_cache", "fixtures",
                    "fixture_sha256"):
        assert section in receipt, f"receipt is missing [{section}]"

    wrf = receipt["wrf_source"]
    assert wrf["commit"] == "d66e442fccc04111067e29274c9f9eaccc3cef28"
    assert wrf["tag"] == "v4.6.1"
    # The commit alone does not prove the checkout, so the two files that
    # actually get compiled are hashed too.  These are the same two hashes
    # the classic receipt records, which is itself a cross-check: both decks
    # were built from the same bytes of WRF.
    assert wrf["phys/module_mp_thompson.F"] == (
        "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802")
    assert wrf["phys/module_mp_radar.F"] == (
        "aa99da858be41efa579966680708d230123a7417560af0eb2e24f4c94e253688")

    # The one input no recompilation can regenerate.  It is not committed
    # (see gpuwm/data/thompson/PROVENANCE.md), so the receipt is the only
    # place the bytes that produced these fixtures are named at all.
    asset = receipt["aerosol_asset"]
    assert asset["CCN_ACTIVATE.BIN"] == (
        "f2b8d3916560f9046f89f8ac5f32c5292a1800498fd75301e422f147c82a3dbd")
    assert asset["CCN_ACTIVATE.BIN.bytes"] == "35288"

    tool = receipt["toolchain"]
    for key in ("fortran", "libc", "uname", "opt_flags", "compile",
                "compile_thompson", "link", "libmvec_symbols"):
        assert tool.get(key), f"receipt does not record toolchain.{key}"


def test_optimisation_flags_keep_the_oracle_compiler_invariant():
    """``-fno-tree-vectorize`` is the property, not a preference.

    Asserted three ways, because the receipt alone would only prove what one
    build did: the flag is in the receipt, the receipt records zero libmvec
    symbols, and EVERY build script that compiles this harness carries the
    flag in the tree as it stands.
    """
    tool = _receipt()["toolchain"]
    assert "-fno-tree-vectorize" in tool["opt_flags"], (
        "the aerosol oracle was built without -fno-tree-vectorize, so its "
        "numbers depend on GCC's vectoriser cost model rather than on WRF")
    assert tool["libmvec_symbols"] == "0", (
        "the aerosol oracle binary links glibc libmvec SIMD math, whose "
        "results are not bit-identical to scalar libm")
    for name in _BUILD_SCRIPTS:
        script = (_HARNESS / name).read_text(encoding="utf-8")
        assert "-fno-tree-vectorize" in script, (
            f"{name} compiles the aerosol harness without "
            "-fno-tree-vectorize; every build path has to carry it or the "
            "deck can be regenerated by a libmvec build")
        assert "_ZGV" in script, (
            f"{name} has no libmvec link guard; the flag alone is a "
            "convention, the `nm -D | grep _ZGV` refusal is the enforcement")


@pytest.mark.parametrize("name", _HARNESS_FILES)
def test_harness_source_still_matches_the_receipt(name):
    """The fixtures and the harness that made them must not drift apart.

    This is the check the classic oracle did not have, and its absence is
    exactly how ninety of its ninety-two fixtures came to predate the harness
    source sitting beside them.
    """
    recorded = _receipt()["harness_source"]
    assert name in recorded, f"receipt does not record {name}"
    actual = _sha256(_HARNESS / name)
    assert actual == recorded[name], (
        f"{name} has changed since the aerosol fixtures were generated "
        f"(receipt {recorded[name][:16]}..., tree {actual[:16]}...).  "
        "Regenerate with tools/thompson_wrf461_oracle/build_aero.sh and "
        "commit the new fixtures together with the new receipt; do not edit "
        "the receipt by hand.")


def test_every_fixture_matches_the_receipt():
    recorded = _receipt()["fixture_sha256"]
    present = {p.name for p in _ORACLE.glob("*.csv")}
    unreceipted = present - set(recorded) - _RECEIPT_EXEMPT
    assert not unreceipted, (
        f"committed but absent from the receipt: {sorted(unreceipted)}")
    missing = set(recorded) - present
    assert not missing, f"receipted but not committed: {sorted(missing)}"
    for name in sorted(recorded):
        assert _sha256(_ORACLE / name) == recorded[name], \
            f"{name} does not hash to what the receipt recorded"


def test_fixture_set_rolls_up_to_the_recorded_digest():
    fixtures = _receipt()["fixtures"]
    names = sorted(_receipt()["fixture_sha256"])
    assert int(fixtures["count"]) == len(names) == 49
    rollup = hashlib.sha256()
    for name in names:
        rollup.update((_ORACLE / name).read_bytes())
    assert rollup.hexdigest() == fixtures["rollup"]


def test_base_state_is_consistent_across_every_scenario():
    """``p_pa`` and ``pii`` come from ``z`` alone, so they cannot vary.

    ``run_column_aero.F90`` builds them at :231-232 inside a loop that never
    looks at the scenario id, so two fixtures that share a ``dz`` and
    disagree here did not come from the same binary.  This is promise 3 from
    the classic file, and it is the promise that was broken there.

    The aerosol deck groups into exactly three base states, separated only by
    ``dz`` -- 500 m for the ordinary columns, 1e8 for the single-layer
    activation sweeps, and 20 m for the wp08 family.  Reading only committed
    repository bytes, this needs no gfortran, no WRF tree and no rebuild.
    """
    groups: dict[tuple, list[str]] = collections.defaultdict(list)
    for path in sorted(_ORACLE.glob("*-column.csv")):
        with path.open(newline="", encoding="ascii") as stream:
            rows = list(csv.DictReader(stream))
        before = [r for r in rows if r["phase"] == "before"]
        if not before:
            continue
        key = (tuple(sorted({r["dz_m"] for r in rows})),
               tuple((r["z_m"], r["p_pa"], r["pii"]) for r in before))
        groups[key].append(path.name)

    by_dz: dict[tuple, list[tuple]] = collections.defaultdict(list)
    for key in groups:
        by_dz[key[0]].append(key)
    for dz, keys in by_dz.items():
        assert len(keys) == 1, (
            f"fixtures at dz={dz} disagree about the scenario-independent "
            "base state (p_pa/pii), so they were not produced by one build: "
            + " | ".join(
                "{" + ", ".join(sorted(groups[k])) + "}" for k in keys))
    assert len(by_dz) == 3, (
        "the aerosol deck should carry exactly three base states, one per "
        f"dz; it carries {len(by_dz)}: {sorted(by_dz)}")

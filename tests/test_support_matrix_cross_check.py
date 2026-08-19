"""The public support matrix and the adapter ledger must agree, checked.

``gpuwm/native_wrf_support_v1.json`` (what ``rw-wps --show-support-matrix``
prints) and ``gpuwm.source_adapters.source_capability_manifest()`` (the
evidence ledger the code maintains) describe the same certifications, are
edited by different commits, and until this file existed were tested only
against themselves.  That is a drift class, and it produced a real instance
on the 2.5.0 line: the ledger invalidated the ERA5 NetCDF mapped-composition
gate (the shipped mapping gained the time/valid_time accepted-spelling list,
``replacement_status: stock_wrf_regate_required``) while the JSON kept
v2.4.1's ``genuine_era5_single_domain_stock_wrf_proof_passed`` -- a public
certification claim the code itself said no longer held.  The sibling GRIB2
entry was updated when its gate was re-earned, which is exactly how the two
files drift: each edit updates the file it lives in.

Every test here reads BOTH sides and fails on contradiction, so the next
invalidation (or re-certification) that touches only one file fails CI
instead of shipping a false public claim.
"""

from __future__ import annotations

import json
from pathlib import Path

from gpuwm.source_adapters import (
    AdapterStatus,
    source_adapters,
    source_capability_manifest,
)

ROOT = Path(__file__).parents[1]

#: The formats the declarative mapped-composition rows may name, and the
#: only values ``source_format`` may take in the evidence ledger.
_MAPPED_FORMATS = {"grib1", "grib2", "netcdf"}

#: JSON meteorological_sources row -> the adapter whose certification the
#: row's stock-WRF claims rest on.
_NAMED_SOURCE_ROWS = {
    "hrrr_conus_sfc_plus_nat": "hrrr",
    "gfs_pgrb2_0p25": "gfs",
    "era5_combined_grib1": "era5",
    "20crv3_member_grib2": "20crv3",
}


def _support_matrix() -> dict:
    path = ROOT / "gpuwm" / "native_wrf_support_v1.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _normalized(claim: str) -> str:
    return claim.lower().replace("-", "_").replace(" ", "_")


def _claims_stock_wrf_proof(claim: str) -> bool:
    """True when a matrix value asserts a stock-WRF proof PASSED.

    Both spellings the matrix uses ("stock_wrf_proof_passed",
    "stock-WRF proof passed through d04") normalize to the same token, and
    negative statements ("not_yet_stock_wrf_gated", "proof invalidated",
    "regate required") do not contain it.
    """

    return "proof_passed" in _normalized(claim)


def test_every_evidence_entry_names_its_format():
    """Without a machine-readable join key nothing can cross-check.

    ``source_format`` is what joins an evidence entry to its
    ``declarative_mapped_composition`` row; an entry without one is
    invisible to every test below, which is how drift hides.
    """

    manifest = source_capability_manifest()
    for entry in (
        *manifest["mapped_stock_wrf_evidence"],
        *manifest["invalidated_mapped_stock_wrf_evidence"],
    ):
        assert entry.get("source_format") in _MAPPED_FORMATS, (
            f"evidence for gate {entry.get('gate')!r} declares no "
            f"source_format, so the support-matrix cross-check cannot see it"
        )


def test_declarative_mapped_composition_rows_match_the_evidence_ledger():
    """A proof claim in the JSON must be a retained gate in the ledger.

    Both directions: a row claiming a passed proof with no retained
    evidence is a false public claim (the ERA5 NetCDF instance), and a
    row NOT claiming one while retained evidence exists understates what
    was earned (the drift that would hide a re-certification).
    """

    manifest = source_capability_manifest()
    retained = manifest["mapped_stock_wrf_evidence"]
    invalidated = manifest["invalidated_mapped_stock_wrf_evidence"]
    rows = {
        fmt: claim
        for fmt, claim in _support_matrix()["meteorological_sources"][
            "declarative_mapped_composition"
        ].items()
        if fmt != "target_ceiling"
    }
    assert set(rows) == _MAPPED_FORMATS, (
        "declarative_mapped_composition names formats the ledger's "
        "source_format vocabulary does not cover (or dropped one)"
    )

    for fmt, claim in rows.items():
        fmt_retained = [
            entry for entry in retained if entry["source_format"] == fmt
        ]
        fmt_invalidated = [
            entry for entry in invalidated if entry["source_format"] == fmt
        ]
        if fmt_retained:
            assert _claims_stock_wrf_proof(claim), (
                f"native_wrf_support_v1.json {fmt!r} row does not claim the "
                f"stock-WRF proof the ledger retains "
                f"(gate {fmt_retained[0]['gate']!r}): {claim!r}"
            )
        else:
            assert not _claims_stock_wrf_proof(claim), (
                f"native_wrf_support_v1.json {fmt!r} row claims a stock-WRF "
                f"proof but source_capability_manifest() retains no evidence "
                f"for that format: {claim!r}"
            )
            if any(
                entry["replacement_gate"] is None
                for entry in fmt_invalidated
            ):
                normalized = _normalized(claim)
                assert (
                    "invalidated" in normalized
                    and "regate_required" in normalized
                ), (
                    f"the ledger invalidated the {fmt!r} gate with no "
                    f"replacement, so the matrix row must say the proof is "
                    f"invalidated and a regate is required: {claim!r}"
                )


def test_invalidated_replacement_gates_are_retained_gates():
    """An invalidated entry naming a replacement gate must point at one
    that was actually re-earned, or the invalidated list quietly
    re-certifies; one naming none must say a regate is required."""

    manifest = source_capability_manifest()
    retained_gates = {
        (entry["source_format"], entry["gate"])
        for entry in manifest["mapped_stock_wrf_evidence"]
    }
    for entry in manifest["invalidated_mapped_stock_wrf_evidence"]:
        replacement = entry["replacement_gate"]
        if replacement is None:
            assert "regate_required" in entry["replacement_status"], (
                f"invalidated gate {entry['gate']!r} names no replacement "
                f"gate but its status does not demand a regate: "
                f"{entry['replacement_status']!r}"
            )
        else:
            assert (entry["source_format"], replacement) in retained_gates, (
                f"invalidated gate {entry['gate']!r} names replacement "
                f"{replacement!r} which is not a retained "
                f"{entry['source_format']} gate in the same ledger"
            )


def test_named_source_rows_match_adapter_certification():
    """meteorological_sources rows vs the adapters' own status."""

    adapters = {adapter.source_id: adapter for adapter in source_adapters()}
    sources = _support_matrix()["meteorological_sources"]
    for row_name, source_id in _NAMED_SOURCE_ROWS.items():
        adapter = adapters[source_id]
        row = sources[row_name]
        single = row["single_domain"]
        if adapter.status is AdapterStatus.CERTIFIED:
            assert _claims_stock_wrf_proof(single), (
                f"{row_name}.single_domain does not claim the stock-WRF "
                f"proof adapter {source_id!r} is certified by: {single!r}"
            )
            assert adapter.stock_wrf_gate.startswith("wrf-"), (
                f"adapter {source_id!r} is certified without naming a "
                f"stock-WRF gate: {adapter.stock_wrf_gate!r}"
            )
        else:
            assert not _claims_stock_wrf_proof(single), (
                f"{row_name}.single_domain claims a stock-WRF proof but "
                f"adapter {source_id!r} is {adapter.status.value!r}: "
                f"{single!r}"
            )
        nested = row["nested_hierarchy"]
        if _claims_stock_wrf_proof(nested):
            assert adapter.status is AdapterStatus.CERTIFIED, (
                f"{row_name}.nested_hierarchy claims a stock-WRF proof but "
                f"adapter {source_id!r} is {adapter.status.value!r}: "
                f"{nested!r}"
            )


def test_exact_gfs_acceptance_pins_match_the_ledger():
    """The wrf.exe pins duplicated across the two files must be equal.

    The JSON's exact_f9760c9 acceptance block and the ledger's retained
    GRIB2 evidence both pin the stock WRF commit and binary; a re-earned
    gate that bumps one file and not the other is the same drift class.
    """

    accept = _support_matrix()["physics_initialization"][
        "exact_f9760c9_gfs_d01_d04_stock_acceptance"
    ]
    grib2 = [
        entry
        for entry in source_capability_manifest()["mapped_stock_wrf_evidence"]
        if entry["source_format"] == "grib2"
    ]
    assert len(grib2) == 1
    assert accept["wrf_commit"] == grib2[0]["wrf_commit"]
    assert accept["wrf_exe_sha256"] == grib2[0]["wrf_exe_sha256"]

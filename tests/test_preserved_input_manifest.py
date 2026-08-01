"""A preservation manifest is only worth writing if it can catch drift.

Every check here is paired with a mutation that must break it.  A verifier
that passes its own evidence proves nothing on its own -- it has to be shown
failing on evidence that was tampered with, or it might be answering
"fine" unconditionally.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

import preserved_input_manifest as pim  # noqa: E402

COMMITTED_MANIFEST = REPO_ROOT / "evidence/wp7b/preserved-met-em-manifest.json"
COMMITTED_CAPTURE = REPO_ROOT / "evidence/wp7b/preserved-met-em-capture-20260731.txt"

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64

M1 = "frame.d01.t0_00_00.nc"
M2 = "frame.d02.t0_00_00.nc"

SYNTHETIC_CAPTURE = f"""\
### capture_host test-node
### capture_utc 2026-01-01T00:00:00Z
### filesystem
/dev/sda1 11111111-2222-3333-4444-555555555555 ext4
### SHA256 SOURCE-LOCATIONS
{SHA_A}  /srv/set-origin/{M1}
{SHA_B}  /srv/set-origin/{M2}
{SHA_A}  /srv/set-replica/{M1}
{SHA_B}  /srv/set-replica/{M2}
### SHA256 STAGED-COPY
{SHA_A}  /srv/preserved/{M1}
{SHA_B}  /srv/preserved/{M2}
### STAT bytes device inode links mtime path
100 2049 11 1 1700000000 /srv/set-origin/{M1}
200 2049 12 1 1700000000 /srv/set-origin/{M2}
100 2049 21 1 1700000000 /srv/preserved/{M1}
200 2049 22 1 1700000000 /srv/preserved/{M2}
"""


def _build(capture_text: str = SYNTHETIC_CAPTURE) -> dict:
    return pim.build_manifest(
        pim.parse_capture(capture_text), source_of_copy="/srv/set-origin"
    )


# --------------------------------------------------------------------------
# transcription
# --------------------------------------------------------------------------


def test_capture_is_transcribed_not_invented():
    manifest = _build()
    assert manifest["schema"] == pim.SCHEMA_ID
    assert manifest["host"] == "test-node"
    assert manifest["captured_utc"] == "2026-01-01T00:00:00Z"
    assert manifest["member_count"] == 2
    assert manifest["total_bytes"] == 300
    shas = {m["member"]: m["sha256"] for m in manifest["members"]}
    assert shas == {M1: SHA_A, M2: SHA_B}
    roles = {loc["root"]: loc["role"] for loc in manifest["locations"]}
    assert roles["/srv/preserved"] == "preserved-copy"
    assert roles["/srv/set-origin"] == "source-of-copy"
    assert roles["/srv/set-replica"] == "existing-replica"


def test_member_name_folds_the_timestamp_spelling():
    """The same file written with ':' and with '_' is one member, not two."""

    colon_m1 = "frame.d01.t0:00:00.nc"
    colon_m2 = "frame.d02.t0:00:00.nc"
    capture = SYNTHETIC_CAPTURE.replace(
        "/srv/set-replica/" + M1, "/srv/set-replica/" + colon_m1
    ).replace("/srv/set-replica/" + M2, "/srv/set-replica/" + colon_m2)
    manifest = pim.build_manifest(pim.parse_capture(capture))
    assert manifest["member_count"] == 2
    replica = next(
        loc for loc in manifest["locations"] if loc["root"] == "/srv/set-replica"
    )
    assert replica["members"][M1] == colon_m1


def test_bytes_absent_from_stat_are_marked_unavailable_not_guessed():
    """A size nothing measured is declared unavailable, never inferred from a peer."""

    capture = "\n".join(
        line
        for line in SYNTHETIC_CAPTURE.splitlines()
        if not (line.startswith("200 2049") and line.endswith(M2))
    )
    manifest = pim.build_manifest(pim.parse_capture(capture))
    by_member = {m["member"]: m for m in manifest["members"]}
    assert by_member[M2]["bytes_status"] == "unavailable"
    assert by_member[M2]["bytes"] is None
    assert by_member[M1]["bytes_status"] == "measured"


# --------------------------------------------------------------------------
# CONTROL 1: divergent replica -- the build must refuse rather than choose
# --------------------------------------------------------------------------


def test_control_divergent_replica_refuses_to_pick_a_winner():
    clean = _build()
    assert pim.verify_manifest(clean) == []

    divergent = SYNTHETIC_CAPTURE.replace(
        f"{SHA_A}  /srv/set-replica/" + M1,
        f"{SHA_C}  /srv/set-replica/" + M1,
    )
    with pytest.raises(pim.PreservationConflict) as excinfo:
        pim.build_manifest(pim.parse_capture(divergent))
    message = str(excinfo.value)
    assert M1 in message
    assert "/srv/set-replica" in message
    assert SHA_C in message


# --------------------------------------------------------------------------
# CONTROL 2: capture drift -- flipped hash, vanished member, vanished location
# --------------------------------------------------------------------------


def test_control_flipped_hash_in_a_later_capture_is_caught():
    manifest = _build()
    assert verify_issues(manifest, SYNTHETIC_CAPTURE) == []

    drifted = SYNTHETIC_CAPTURE.replace(
        f"{SHA_A}  /srv/preserved/" + M1,
        f"{SHA_C}  /srv/preserved/" + M1,
    )
    issues = verify_issues(manifest, drifted)
    assert issues, "a flipped hash must be reported"
    assert any("/srv/preserved" in issue and M1 in issue for issue in issues)


def test_control_removed_member_in_a_later_capture_is_caught():
    manifest = _build()
    drifted = "\n".join(
        line
        for line in SYNTHETIC_CAPTURE.splitlines()
        if line != f"{SHA_B}  /srv/preserved/" + M2
    )
    issues = verify_issues(manifest, drifted)
    assert any(M2 in issue and "gone" in issue for issue in issues)


def test_control_removed_location_in_a_later_capture_is_caught():
    manifest = _build()
    drifted = "\n".join(
        line
        for line in SYNTHETIC_CAPTURE.splitlines()
        if "/srv/preserved/" not in line
    )
    issues = verify_issues(manifest, drifted)
    assert any("/srv/preserved" in issue and "absent" in issue for issue in issues)


def verify_issues(manifest: dict, capture_text: str) -> list[str]:
    return pim.verify_against_capture(manifest, pim.parse_capture(capture_text))


# --------------------------------------------------------------------------
# CONTROL 3: the manifest binds itself
# --------------------------------------------------------------------------


def test_control_edited_manifest_breaks_its_own_binding():
    manifest = _build()
    assert pim.verify_manifest(manifest) == []

    tampered = json.loads(json.dumps(manifest))
    tampered["members"][0]["sha256"] = SHA_C
    issues = pim.verify_manifest(tampered)
    assert any("manifest_binding_sha256" in issue for issue in issues)


def test_control_manifest_without_a_preserved_copy_is_refused():
    capture = SYNTHETIC_CAPTURE.replace("### SHA256 STAGED-COPY", "### SHA256 OTHER-REPLICA")
    manifest = pim.build_manifest(pim.parse_capture(capture))
    issues = pim.verify_manifest(manifest)
    assert any("preserved-copy" in issue for issue in issues)


def test_control_missing_member_at_one_location_is_refused():
    manifest = _build()
    tampered = json.loads(json.dumps(manifest))
    for location in tampered["locations"]:
        if location["root"] == "/srv/preserved":
            location["members"].pop(M2)
    tampered[pim.DIGEST_KEY] = pim.canonical_sha256(
        {k: v for k, v in tampered.items() if k != pim.DIGEST_KEY}
    )
    issues = pim.verify_manifest(tampered)
    assert any("missing members" in issue for issue in issues)


# --------------------------------------------------------------------------
# CONTROL 4: an admission set says which preserved bytes a consumer reads
# --------------------------------------------------------------------------

ADMITTED_ROOT = "/srv/admitted-elsewhere"


def _rebind(document: dict) -> dict:
    """Re-seal a document after editing it, so a check other than the binding fires."""

    document[pim.DIGEST_KEY] = pim.canonical_sha256(
        {k: v for k, v in document.items() if k != pim.DIGEST_KEY}
    )
    return document


def _admitting() -> dict:
    """The synthetic manifest plus one admission set, sealed.

    Deliberately the shape the hard case has: one member admitted as exactly
    the preserved bytes, one admitted as *different* bytes under the same
    member name and from its own root, and the preserved member that displaces
    recorded as not admitted with a reason.  A set where every row is the
    preserved row would exercise none of the ambiguity the section exists for.
    """

    document = json.loads(json.dumps(_build()))
    document[pim.ADMISSION_KEY] = [
        {
            "id": "consumer",
            "note": "What the consumer is initialized from.",
            "member_count": 2,
            "total_bytes": 300,
            "members": [
                {
                    "bytes": 100,
                    "bytes_status": "measured",
                    "member": M1,
                    "preserved_member_relation": pim.RELATION_SAME_BYTES,
                    "sha256": SHA_A,
                },
                {
                    "bytes": 200,
                    "bytes_status": "measured",
                    "member": M2,
                    "preserved_member_relation": pim.RELATION_DIFFERENT_BYTES,
                    "root": ADMITTED_ROOT,
                    "sha256": SHA_C,
                },
            ],
            "not_admitted": [
                {
                    "member": M2,
                    "reason": "Preserved as history; this consumer cannot read it.",
                    "sha256": SHA_B,
                }
            ],
        }
    ]
    return _rebind(document)


def admission_issues(document: dict) -> list[str]:
    return pim.verify_admission_sets(document)


def test_an_admission_set_resolves_to_the_bytes_it_names():
    document = _admitting()
    assert pim.verify_manifest(document) == []

    admitted = {
        (row["member"], row["sha256"])
        for row in document[pim.ADMISSION_KEY][0]["members"]
    }
    assert admitted == {(M1, SHA_A), (M2, SHA_C)}
    # The member name alone does not decide the bytes: the preserved set spells
    # M2 as SHA_B and the admission set spells it SHA_C.  That is the whole
    # reason an admission row is keyed on the digest as well as the name.
    preserved = {m["member"]: m["sha256"] for m in document["members"]}
    assert preserved[M2] == SHA_B


def test_a_manifest_without_an_admission_set_is_unaffected():
    """The section is optional; a manifest that carries none says nothing new."""

    document = _build()
    assert pim.ADMISSION_KEY not in document
    assert admission_issues(document) == []
    assert pim.verify_manifest(document) == []


def test_an_earlier_schema_document_still_verifies():
    """The addition is additive: a document from before it keeps verifying."""

    document = json.loads(json.dumps(_build()))
    document["schema"] = pim.SUPPORTED_SCHEMA_IDS[0]
    assert document["schema"] != pim.SCHEMA_ID
    assert pim.verify_manifest(_rebind(document)) == []


def test_control_an_admitted_member_that_misstates_the_preserved_bytes_is_refused():
    """A row claiming the preserved bytes must hash what the preserved member does."""

    document = _admitting()
    document[pim.ADMISSION_KEY][0]["members"][0]["sha256"] = SHA_C
    issues = pim.verify_manifest(_rebind(document))
    assert any(
        M1 in issue and pim.RELATION_SAME_BYTES in issue for issue in issues
    ), issues


def test_control_a_short_admission_set_is_refused():
    """Dropping a member from admission without saying so is not a decision."""

    document = _admitting()
    entry = document[pim.ADMISSION_KEY][0]
    entry["members"] = [row for row in entry["members"] if row["member"] != M1]
    entry["member_count"] = 1
    entry["total_bytes"] = 200
    issues = pim.verify_manifest(_rebind(document))
    assert any(
        M1 in issue and "neither admitted nor recorded" in issue for issue in issues
    ), issues


def test_control_an_extra_admitted_member_is_refused():
    """Admitting bytes nothing in the manifest preserves is the prevented case."""

    document = _admitting()
    entry = document[pim.ADMISSION_KEY][0]
    entry["members"].append(
        {
            "bytes": 300,
            "bytes_status": "measured",
            "member": "frame.d03.t0_00_00.nc",
            "preserved_member_relation": pim.RELATION_SAME_BYTES,
            "sha256": SHA_C,
        }
    )
    entry["member_count"] = 3
    entry["total_bytes"] = 600
    issues = pim.verify_manifest(_rebind(document))
    assert any("frame.d03" in issue and "no preserved member" in issue for issue in issues), issues


def test_control_a_preserved_member_left_out_without_a_reason_is_refused():
    """The fail-closed hinge: not-admitted is a written decision, not a silence."""

    document = _admitting()
    document[pim.ADMISSION_KEY][0]["not_admitted"] = []
    issues = pim.verify_manifest(_rebind(document))
    assert any(
        M2 in issue and "neither admitted nor recorded" in issue for issue in issues
    ), issues

    # A blank reason is the same silence with a key in front of it.
    document = _admitting()
    document[pim.ADMISSION_KEY][0]["not_admitted"][0]["reason"] = "   "
    issues = pim.verify_manifest(_rebind(document))
    assert any("with no written reason" in issue for issue in issues), issues


def test_control_admitted_bytes_no_preserved_location_holds_must_say_where_they_are():
    """Bytes outside the preserved locations are unfindable unless a root is given."""

    document = _admitting()
    document[pim.ADMISSION_KEY][0]["members"][1].pop("root")
    issues = pim.verify_manifest(_rebind(document))
    assert any(M2 in issue and "records no root" in issue for issue in issues), issues


def test_control_an_admission_set_on_an_earlier_schema_document_is_refused():
    """A version bump that a document can ignore records nothing."""

    document = _admitting()
    document["schema"] = pim.SUPPORTED_SCHEMA_IDS[0]
    issues = pim.verify_manifest(_rebind(document))
    assert any(pim.ADMISSION_KEY in issue and "declares" in issue for issue in issues), issues


def test_control_an_admission_set_that_documents_nothing_is_refused():
    """A second keyed set nobody explained re-opens the ambiguity it removes."""

    document = _admitting()
    document[pim.ADMISSION_KEY][0]["note"] = ""
    issues = pim.verify_manifest(_rebind(document))
    assert any("what it admits" in issue for issue in issues), issues


# --------------------------------------------------------------------------
# the committed manifest itself
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not COMMITTED_MANIFEST.exists(), reason="preservation evidence is not in this tree"
)
def test_committed_manifest_verifies_against_its_committed_capture():
    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    assert pim.verify_manifest(manifest) == []
    issues = verify_issues(manifest, COMMITTED_CAPTURE.read_text(encoding="utf-8"))
    assert issues == []


@pytest.mark.skipif(
    not COMMITTED_MANIFEST.exists(), reason="preservation evidence is not in this tree"
)
def test_committed_manifest_carries_a_preserved_copy_distinct_from_its_source():
    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    roles = {loc["role"] for loc in manifest["locations"]}
    assert "preserved-copy" in roles
    assert "source-of-copy" in roles
    preserved = [loc for loc in manifest["locations"] if loc["role"] == "preserved-copy"]
    source = [loc for loc in manifest["locations"] if loc["role"] == "source-of-copy"]
    assert len(preserved) == 1
    assert len(source) == 1
    assert preserved[0]["root"] != source[0]["root"]


#: The exact bytes the committed admission set resolves to.  Written out here
#: rather than read back out of the manifest: a test that asks the document
#: what it says and then agrees with it checks nothing.
COMMITTED_ADMITTED: set[tuple[str, str]] = {
    (
        "met_em.d01.1974-04-03_12_00_00.nc",
        "baeae9cdf9737eb2c947500fdab8be4e03bc22d05ada45450f984fad75b507e9",
    ),
    (
        "met_em.d01.1974-04-03_18_00_00.nc",
        "d95c422caa78af8bcf76ecccc22376c70e958bf54101c71dae018b20f3313d5d",
    ),
    (
        "met_em.d02.1974-04-03_12_00_00.nc",
        "d90deb78d0e1fa35d618b49e4875f124dfd349d4c23fe3e23c552f5b5ffec03c",
    ),
    (
        "met_em.d03.1974-04-03_12_00_00.nc",
        "18c9909283364e03b311a42433abf628eeacb53353b2dd29f202c98e1debdec5",
    ),
    (
        "met_em.d04.1974-04-03_12_00_00.nc",
        "d9c18f6ac449e2662b0c788f28bef25650ed34e45542161fec9b6441fc2e3f2d",
    ),
}

#: Preserved, and explicitly not admitted.  Shares a member name with one of
#: the five above and is a different file.
COMMITTED_NOT_ADMITTED: tuple[str, str] = (
    "met_em.d04.1974-04-03_12_00_00.nc",
    "daac4909e856df0f9529ff94ecab2228a9ca7402fc4028a7294287da7079bd82",
)


@pytest.mark.skipif(
    not COMMITTED_MANIFEST.exists(), reason="preservation evidence is not in this tree"
)
def test_committed_admission_set_resolves_to_exactly_the_files_it_names():
    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    entries = manifest[pim.ADMISSION_KEY]
    assert len(entries) == 1
    entry = entries[0]

    admitted = {(row["member"], row["sha256"]) for row in entry["members"]}
    assert admitted == COMMITTED_ADMITTED
    assert entry["member_count"] == len(COMMITTED_ADMITTED)

    excluded = {(row["member"], row["sha256"]) for row in entry["not_admitted"]}
    assert excluded == {COMMITTED_NOT_ADMITTED}
    assert all(row["reason"].strip() for row in entry["not_admitted"])


@pytest.mark.skipif(
    not COMMITTED_MANIFEST.exists(), reason="preservation evidence is not in this tree"
)
def test_committed_manifest_still_preserves_what_it_declines_to_admit():
    """Not admitted is a statement about use, never about whether it survives."""

    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    preserved = {m["member"]: m["sha256"] for m in manifest["members"]}
    name, sha = COMMITTED_NOT_ADMITTED
    assert preserved[name] == sha
    # Every location that held it before still holds it.
    for location in manifest["locations"]:
        assert name in location["members"]

    # ...and the admitted file of that same name is a different file, which is
    # why the admission section keys on the digest and not the name.
    admitted = {member: digest for member, digest in COMMITTED_ADMITTED}
    assert admitted[name] != sha


@pytest.mark.skipif(
    not COMMITTED_MANIFEST.exists(), reason="preservation evidence is not in this tree"
)
def test_committed_admission_set_states_which_bytes_it_shares_with_the_preserved_set():
    """The overlap is recorded per member and measured, not left to the reader."""

    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    preserved = {m["member"]: m["sha256"] for m in manifest["members"]}
    entry = manifest[pim.ADMISSION_KEY][0]

    relations = {
        row["member"]: row["preserved_member_relation"] for row in entry["members"]
    }
    shared = {
        member for member, rel in relations.items()
        if rel == pim.RELATION_SAME_BYTES
    }
    assert len(shared) == 4
    for row in entry["members"]:
        if row["preserved_member_relation"] == pim.RELATION_SAME_BYTES:
            assert row["sha256"] == preserved[row["member"]]
        else:
            assert row["sha256"] != preserved[row["member"]]
            assert row["root"], "bytes outside the preserved locations need a root"


@pytest.mark.skipif(
    not COMMITTED_MANIFEST.exists(), reason="preservation evidence is not in this tree"
)
def test_committed_manifest_records_whether_the_copies_share_a_device():
    """The single-device fact is recorded, whatever it is -- it is the risk D-35 answers."""

    manifest = json.loads(COMMITTED_MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(manifest["distinct_devices"], list)
    assert manifest["distinct_devices"], "device identity must be measured, not omitted"

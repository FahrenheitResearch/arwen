"""``tools/check_case_token_leakage.py`` runs, and it bites.

The standing rule is that no case or data-source token may appear in a
generic position.  The gate that enforces it existed but nothing invoked it,
and its suppression key was ``<path>::<text>`` -- position-blind, so one
baseline entry for ``hrrr`` in the physics registry absorbed ``hrrr``
*anywhere* in that document, including as a physics-template id.  That is the
one thing the gate is for.

These pins hold both halves:

* the gate passes on the real tree (nothing new has leaked in), and
* it fails on a scratch tree where a case token is injected into a position
  the real baseline already names *somewhere else* in the same file.

The second half is the load-bearing one.  A gate that only ever passes is not
evidence of anything.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.check_case_token_leakage import (
    BASELINE,
    BASELINE_ERROR,
    BASELINE_PATH,
    PROTECTED_ZONES,
    Violation,
    load_baseline,
    partition,
    reason_is_a_decision,
    reason_is_debt,
    resolve_target,
    scan_file,
    scan_root,
    scan_zones,
    zones_under,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_REL = "gpuwm/physics_registry_v2.json"
REGISTRY = REPO_ROOT / REGISTRY_REL

#: Occurrences the baseline carries as REAL violations rather than as data.
#: Pinned so the debt can shrink without ceremony and cannot grow quietly.
CARRIED_DEBT = 3


def _raw_baseline() -> dict:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _scratch_registry(tmp_path: Path, mutate) -> Path:
    """A scratch root holding a mutated copy of the real physics registry.

    The real ``gpuwm/physics_registry_v2.json`` is never touched: editing it
    changes ``registry_sha256`` and invalidates every bound plan.
    """
    document = copy.deepcopy(json.loads(REGISTRY.read_text(encoding="utf-8")))
    mutate(document)
    root = tmp_path / "root"
    target = root / REGISTRY_REL
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return root


def _new_against_real_baseline(root: Path) -> list[Violation]:
    new, _suppressed = partition(scan_root(root))
    return new


# --------------------------------------------------------------------------
# the gate itself
# --------------------------------------------------------------------------


def test_the_baseline_loads():
    """A baseline nobody can parse suppresses nothing; say so out loud."""
    assert BASELINE_ERROR is None, BASELINE_ERROR
    assert BASELINE, "the baseline parsed to nothing"


def test_no_new_case_token_in_a_generic_position():
    """THE gate.  Anything not named in the baseline fails the build."""
    new, suppressed = partition(scan_root(REPO_ROOT))
    assert new == [], "\n".join(f"{v}\n  key: {v.key}" for v in new)
    assert suppressed, "the scan found nothing at all -- did the zones move?"


def test_every_baseline_entry_carries_a_written_reason():
    """Baselining is a decision somebody writes down, not a paste of a key."""
    declared = _raw_baseline()["known_violations"]
    unreasoned = sorted(k for k, r in declared.items() if not reason_is_a_decision(r))
    assert unreasoned == [], unreasoned
    assert set(declared) == set(BASELINE)


def test_carried_debt_does_not_grow():
    """Entries the baseline admits are real violations, not data."""
    debt = sorted(k for k, reason in BASELINE.items() if reason_is_debt(reason))
    assert len(debt) <= CARRIED_DEBT, debt
    # The source-named physics template is the headline one; it must stay
    # visible as debt and never be re-labelled as legitimate.
    assert any("/templates/" in key for key in debt), debt


def test_the_baseline_names_no_occurrence_that_has_gone():
    """A stale entry is dead suppression: retire it when the code is fixed."""
    live = {v.key for v in scan_root(REPO_ROOT)}
    # Entries for paths that do not exist in this repository belong to another
    # root the tool can also be pointed at; only judge what is here.
    stale = sorted(
        key
        for key in BASELINE
        if (REPO_ROOT / key.split("::", 1)[0]).exists() and key not in live
    )
    assert stale == [], stale


# --------------------------------------------------------------------------
# negative controls: the suppression key is position-aware
# --------------------------------------------------------------------------


def test_a_physics_template_named_after_a_source_is_reported(tmp_path: Path):
    """The audit's exact injection.

    The real baseline names ``hrrr`` at three positions in this document.
    Under the old ``<path>::<text>`` key that absorbed a *template id* named
    ``hrrr`` -- the position the gate's own rules call illegitimate.
    """

    def mutate(document: dict) -> None:
        template = next(iter(document["templates"].values()))
        document["templates"]["hrrr"] = copy.deepcopy(template)

    new = _new_against_real_baseline(_scratch_registry(tmp_path, mutate))
    keys = [v.key for v in new]
    assert any("/templates/hrrr::registry key::hrrr" in k for k in keys), keys


def test_a_component_option_named_after_a_source_is_reported(tmp_path: Path):
    """The audit's second injection: an option key named for a source."""

    def mutate(document: dict) -> None:
        options = document["components"]["microphysics"]["options"]
        options["hrrr"] = copy.deepcopy(next(iter(options.values())))

    new = _new_against_real_baseline(_scratch_registry(tmp_path, mutate))
    keys = [v.key for v in new]
    assert any(
        "/components/microphysics/options/hrrr::registry key::hrrr" in k for k in keys
    ), keys


def test_a_source_id_appended_to_an_already_baselined_list_is_reported(
    tmp_path: Path,
):
    """The ``af04478`` shape: append a source to a route that already has some.

    The old gate printed a clean bill of health for exactly this edit.
    """

    appended_at: list[int] = []

    def mutate(document: dict) -> None:
        route = document["runner_routes"]["tools.prepared_single_domain_forecast"]
        # The ordinal is READ, not written down.  It used to be the literal
        # 3, which made this gate's own proof a hostage to the length of a
        # list the product is expected to grow: when the route legitimately
        # gained its fourth source, the append landed at 4, index 3 was a
        # baselined entry, and this test failed while the property it
        # states was still perfectly true.
        appended_at.append(len(route["source_ids"]))
        route["source_ids"].append("hrrr")

    new = _new_against_real_baseline(_scratch_registry(tmp_path, mutate))
    keys = [v.key for v in new]
    index = appended_at[0]
    assert any(
        "tools.prepared_single_domain_forecast/source_ids/"
        f"{index}::registry value::hrrr" in k
        for k in keys
    ), (index, keys)


def test_an_unmutated_copy_of_the_registry_reports_nothing(tmp_path: Path):
    """The control root must be clean before a pin dirties it."""
    root = _scratch_registry(tmp_path, lambda document: None)
    assert _new_against_real_baseline(root) == []


def test_the_key_distinguishes_two_occurrences_of_the_same_text():
    """The defect, pinned directly: position is part of the key."""
    here = Violation(REGISTRY_REL, 0, "registry key", "hrrr", "/templates/hrrr")
    there = Violation(
        REGISTRY_REL, 0, "registry key", "hrrr", "/runner_routes/r/source_ids/0"
    )
    assert here.key != there.key
    assert here.position in here.key and there.position in there.key


def test_a_second_python_occurrence_in_the_same_scope_is_reported(tmp_path: Path):
    """Ordinals make the key sensitive to addition without pinning line numbers."""
    root = tmp_path / "root"
    module = root / "gpuwm" / "core" / "leaky.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("def f():\n    hrrr_nodes = 1\n    return hrrr_nodes\n", "utf-8")

    found = scan_file(module, "gpuwm/core/leaky.py")
    ordinals = [v.position for v in found]
    assert ordinals == ["<module>.f#1", "<module>.f#2"], ordinals

    baseline = {found[0].key: "LEGITIMATE: pretend the first one was reviewed."}
    new, suppressed = partition(found, baseline)
    assert [v.position for v in suppressed] == ["<module>.f#1"]
    assert [v.position for v in new] == ["<module>.f#2"]


def test_python_positions_survive_an_edit_above_them(tmp_path: Path):
    """A line-number key would fire on every unrelated edit; this one does not."""
    root = tmp_path / "root"
    module = root / "gpuwm" / "core" / "leaky.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    body = "def f():\n    hrrr_nodes = 1\n    return hrrr_nodes\n"
    module.write_text(body, "utf-8")
    before = [v.key for v in scan_file(module, "gpuwm/core/leaky.py")]

    module.write_text("import os\n\n\n" + body, "utf-8")
    after = [v.key for v in scan_file(module, "gpuwm/core/leaky.py")]
    assert before == after


# --------------------------------------------------------------------------
# the certification measurement path is inside the zones
# --------------------------------------------------------------------------

#: Modules the t=0 digest work package added to PROTECTED_ZONES.  A digest
#: that named one campaign's frames would score that campaign's staging
#: layout rather than a staged pair, and would read clean to every other
#: check in the tree.
_CERTIFICATION_MODULES = (
    "gpuwm/verify/state_equiv.py",
    "gpuwm/verify/t0_state_digest.py",
    "tools/matched_wrfout_t0_state_digest.py",
)


def _scratch_copy(tmp_path: Path, rel: str, suffix: str = "") -> Path:
    root = tmp_path / "root"
    target = root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        (REPO_ROOT / rel).read_text(encoding="utf-8") + suffix,
        encoding="utf-8")
    return root


@pytest.mark.parametrize("rel", _CERTIFICATION_MODULES)
def test_a_case_token_injected_into_the_digest_path_is_reported(
    tmp_path: Path, rel: str
):
    """The zone entries are live: injecting one token makes the gate fail."""
    injected = "\n\nSTAGED_FRAMES = ('wrfout_d01_1974-04-03_12_00_00',)\n"
    root = _scratch_copy(tmp_path, rel, injected)
    new = _new_against_real_baseline(root)
    assert [(v.path, v.kind) for v in new] == [(rel, "string literal")], new


@pytest.mark.parametrize("rel", _CERTIFICATION_MODULES)
def test_the_unmutated_digest_path_reports_nothing(tmp_path: Path, rel: str):
    """The control root must be clean before an injection dirties it."""
    assert _new_against_real_baseline(_scratch_copy(tmp_path, rel)) == []


#: The published receipt.  The measurement path is only half the zone: a
#: receipt is where a case name has an honest reason to appear (its input
#: inventory records what was staged), which is exactly why the file needs
#: watching -- an inventory entry is data, and a case name anywhere else in
#: the document would mean the scoring contract itself was written for one
#: campaign.
PUBLISHED_RECEIPT = "gpuwm/data/certification/t0_state_parity_digest.json"


def test_the_published_receipt_is_in_a_protected_zone():
    from tools.check_case_token_leakage import PROTECTED_ZONES

    assert (REPO_ROOT / PUBLISHED_RECEIPT).is_file()
    assert any(PUBLISHED_RECEIPT.startswith(zone) for zone in PROTECTED_ZONES)


def test_the_published_receipt_reports_nothing_new(tmp_path: Path):
    """Its inventory names are baselined; nothing else in it may offend."""
    root = tmp_path / "root"
    target = root / PUBLISHED_RECEIPT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((REPO_ROOT / PUBLISHED_RECEIPT).read_bytes())
    assert _new_against_real_baseline(root) == []


def test_a_case_token_in_a_receipt_scoring_position_is_reported(
    tmp_path: Path,
):
    """The injection control for the receipt half of the zone.

    The token goes into a *scoring* position -- a carrier-group name -- not
    into the input inventory.  The baseline covers the inventory entries by
    position, so if it were absorbing this one too the key would be too
    wide, which is the failure mode this gate was rewritten to close.
    """
    document = json.loads(
        (REPO_ROOT / PUBLISHED_RECEIPT).read_text(encoding="utf-8"))
    groups = document["registration"]["carrier_groups"]
    groups["real74_dynamics"] = groups.pop("dry_dynamics")

    root = tmp_path / "root"
    target = root / PUBLISHED_RECEIPT
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2), encoding="utf-8")

    new = _new_against_real_baseline(root)
    assert [(v.kind, v.text) for v in new] == [
        ("registry key", "real74_dynamics")], new


# --------------------------------------------------------------------------
# specialization that arrives by import
# --------------------------------------------------------------------------

#: A generic name bound to one case's data -- the shape the per-file scanner
#: structurally cannot see, taken from the real
#: gpuwm/verify/nest_gates.py CHILD_REFERENCE_FRAMES table.
_CASE_DATA_MODULE = '''\
"""A legitimate case-data module: it is not in a protected zone."""

CHILD_REFERENCE_FRAMES = {
    "d02": "wrfout_d02_1974-04-03_13_15_00",
    "d03": "wrfout_d03_1974-04-03_13_15_00",
}
REEXPORTED_FRAMES = CHILD_REFERENCE_FRAMES
FP64_MIRROR_MAX_ULPS = 2
'''


def _import_tree(tmp_path: Path, importer_rel: str, body: str) -> Path:
    """Scratch root: a case-data module plus one importing module."""
    root = tmp_path / "root"
    data = root / "gpuwm" / "verify" / "frames.py"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(_CASE_DATA_MODULE, encoding="utf-8")
    importer = root / importer_rel
    importer.parent.mkdir(parents=True, exist_ok=True)
    importer.write_text(body, encoding="utf-8")
    return root


def test_the_case_data_module_itself_is_never_flagged(tmp_path: Path):
    """The control that keeps the gate honest: the module that legitimately
    HOLDS the case data is not in a protected zone, so scanning the tree
    reports nothing about it.  If this failed, every finding below would
    just be the gate objecting to case data living in case data."""
    root = _import_tree(tmp_path, "gpuwm/core/quiet.py", "X = 1\n")
    assert _new_against_real_baseline(root) == []
    # ...and it is genuinely full of case tokens when scanned directly.
    direct = scan_file(root / "gpuwm" / "verify" / "frames.py",
                       "gpuwm/core/frames.py", root)
    assert direct, "the fixture carries no case token at all"


def test_a_protected_module_importing_case_data_is_flagged(tmp_path: Path):
    """The blind spot, closed.

    ``CHILD_REFERENCE_FRAMES`` carries no case token in its spelling; the
    importing file contains no case token anywhere.  Only opening the
    module it imports FROM shows the specialization."""
    root = _import_tree(
        tmp_path, "gpuwm/core/leaky.py",
        "from gpuwm.verify.frames import CHILD_REFERENCE_FRAMES\n"
        "\n"
        "\n"
        "def frame(domain):\n"
        "    return CHILD_REFERENCE_FRAMES[domain]\n")
    new = _new_against_real_baseline(root)
    assert [(v.path, v.kind, v.text) for v in new] == [
        ("gpuwm/core/leaky.py", "imported value",
         "gpuwm.verify.frames.CHILD_REFERENCE_FRAMES")]
    assert "1974" in new[0].detail
    assert new[0].position == "<module>#1"


def test_the_import_check_follows_a_re_export(tmp_path: Path):
    root = _import_tree(
        tmp_path, "gpuwm/core/leaky.py",
        "from gpuwm.verify.frames import REEXPORTED_FRAMES\n")
    new = _new_against_real_baseline(root)
    assert [v.text for v in new] == [
        "gpuwm.verify.frames.REEXPORTED_FRAMES"]


def test_importing_a_generic_value_from_the_same_module_is_quiet(tmp_path):
    """Negative control on the same import, same module, same scanner: a
    name bound to a value with no case token is not reported.  Without
    this, the check above would be indistinguishable from 'any import
    from a module that mentions a case anywhere'."""
    root = _import_tree(
        tmp_path, "gpuwm/core/quiet.py",
        "from gpuwm.verify.frames import FP64_MIRROR_MAX_ULPS\n"
        "\n"
        "\n"
        "def tolerance():\n"
        "    return FP64_MIRROR_MAX_ULPS\n")
    assert _new_against_real_baseline(root) == []


def test_an_unprotected_module_may_import_case_data(tmp_path: Path):
    """Zones still decide.  The same import outside a protected zone is
    legitimate -- that is how a case module reaches its own data."""
    root = _import_tree(
        tmp_path, "gpuwm/verify/cases/some_case.py",
        "from gpuwm.verify.frames import CHILD_REFERENCE_FRAMES\n")
    assert _new_against_real_baseline(root) == []


def test_a_source_named_import_is_flagged_by_its_spelling(tmp_path: Path):
    """The other half: the module path or bound name carries the token
    itself.  ``ast.Import``/``ast.ImportFrom`` hold neither a Name node nor
    a string constant, so nothing recorded these before."""
    root = tmp_path / "root"
    module = root / "gpuwm" / "core" / "leaky.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(
        "import gpuwm.ingest.hrrr\n"
        "from gpuwm.ingest.soil import HRRR_SOIL_NODE_DEPTHS_M\n",
        encoding="utf-8")
    found = scan_file(module, "gpuwm/core/leaky.py", root)
    assert [(v.kind, v.text) for v in found] == [
        ("import", "gpuwm.ingest.hrrr"),
        ("import", "gpuwm.ingest.soil.HRRR_SOIL_NODE_DEPTHS_M"),
    ]


def test_import_findings_do_not_disturb_existing_keys(tmp_path: Path):
    """Adding a kind must not renumber the ordinals of the others, or every
    baseline entry in the file would go stale at once."""
    root = tmp_path / "root"
    module = root / "gpuwm" / "core" / "leaky.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    body = "def f():\n    hrrr_nodes = 1\n    return hrrr_nodes\n"
    module.write_text(body, encoding="utf-8")
    before = [v.key for v in scan_file(module, "gpuwm/core/leaky.py", root)]

    module.write_text("import gpuwm.ingest.hrrr\n\n\n" + body, encoding="utf-8")
    after = [v.key for v in scan_file(module, "gpuwm/core/leaky.py", root)]
    assert set(before) <= set(after)
    assert len(after) == len(before) + 1


# --------------------------------------------------------------------------
# the baseline fails closed
# --------------------------------------------------------------------------


@pytest.mark.parametrize("reason", ["", "   ", "TODO: decide", "UNREVIEWED: x", None])
def test_an_undecided_reason_suppresses_nothing(tmp_path: Path, reason):
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({"known_violations": {"a::b::c::d": reason}}), encoding="utf-8"
    )
    table, error = load_baseline(path)
    assert error is None
    assert table == {}


def test_the_old_list_shaped_baseline_is_refused(tmp_path: Path):
    """The legacy form has no positions in it; accepting it would re-open the hole."""
    path = tmp_path / "baseline.json"
    path.write_text(
        json.dumps({"known_violations": ["gpuwm/x.json::hrrr"]}), encoding="utf-8"
    )
    table, error = load_baseline(path)
    assert table == {}
    assert error is not None and "position-blind" in error


def test_a_missing_baseline_suppresses_nothing(tmp_path: Path):
    table, error = load_baseline(tmp_path / "absent.json")
    assert table == {}
    assert error is not None


def test_a_case_token_in_the_new_verification_instruments_is_reported(
        tmp_path: Path):
    """CONTROL for the zone extension.

    The metric and envelope modules score whatever they are pointed at.  A
    case token in one of them specializes the yardstick rather than the run,
    which is invisible until somebody points it at a second case -- so each
    new module is checked here, one scratch injection at a time.
    """
    injected = {
        "gpuwm/verify/field_metrics.py": "REAL74_DX = 500.0\n",
        "gpuwm/verify/chaos_envelope.py": "def ohio_rows():\n    return ()\n",
        "gpuwm/verify/spectral.py": "REAL74_BAND = (1, 2)\n",
        "tools/matched_wrfout_envelope.py": "HRRR_LEADS = (3600,)\n",
        "tools/certify_band_from_ensemble.py": "def band_1974():\n    return {}\n",
    }
    for relative, source in injected.items():
        root = tmp_path / relative.replace("/", "-")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")
        new = _new_against_real_baseline(root)
        assert [v.path for v in new] == [relative], (relative, new)


def test_the_new_instruments_are_in_a_protected_zone():
    from tools.check_case_token_leakage import PROTECTED_ZONES

    for relative in ("gpuwm/verify/field_metrics.py",
                     "gpuwm/verify/chaos_envelope.py",
                     "gpuwm/verify/spectral.py",
                     "tools/matched_wrfout_envelope.py",
                     "tools/certify_band_from_ensemble.py"):
        assert relative in PROTECTED_ZONES
        assert (REPO_ROOT / relative).is_file()


# --------------------------------------------------------------------------
# the CLI scans what it claims to scan
# --------------------------------------------------------------------------
#
# MEASURED at b858ad824: ``python tools/check_case_token_leakage.py gpuwm
# tools tests docs`` -- the invocation every lane and every close/verify
# wave ran by hand -- scanned ZERO files and printed green every time.
# ``main`` handed each argument to ``scan_root`` as a repository root, and
# ``scan_root`` spells its zones relative to a REPOSITORY root, so with
# ``root=gpuwm`` the zone ``gpuwm/core/`` resolved to ``gpuwm/gpuwm/core``
# and matched nothing.  A gate that scans nothing prints exactly what a
# clean tree prints, which is the silent-green family this repository has
# been bitten by before (the ``grep -c`` fallback, the buffered log).
#
# These pins hold the CLI to its own claim: it scans the zones, it says
# how many files that was, and it refuses rather than reporting on a
# target that selects none of them.

CLI = REPO_ROOT / "tools" / "check_case_token_leakage.py"


def _cli(*argv: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLI), *argv],
        cwd=str(cwd), capture_output=True, text=True)


def _leaky_tree(tmp_path: Path) -> Path:
    """A scratch repository root with one planted token in the core."""
    root = tmp_path / "root"
    module = root / "gpuwm" / "core" / "leaky.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("def f():\n    hrrr_nodes = 1\n    return hrrr_nodes\n",
                      encoding="utf-8")
    return root


def test_the_cli_scans_a_subtree_argument_rather_than_discarding_it(
        tmp_path: Path):
    """THE defect.  ``... gpuwm`` must scan the zones under ``gpuwm/``.

    The planted token sits in ``gpuwm/core/``, a protected zone, and the
    argument names the subtree that contains it.  Before this fix the run
    exited 0 with "0 known occurrence(s) suppressed".
    """

    root = _leaky_tree(tmp_path)
    completed = _cli("gpuwm", cwd=root)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "hrrr_nodes" in completed.stdout, completed.stdout


def test_the_cli_scans_the_same_file_whether_given_the_root_or_the_subtree(
        tmp_path: Path):
    """A subtree argument narrows the scan; it never silently empties it."""

    root = _leaky_tree(tmp_path)
    whole = _cli(".", cwd=root)
    part = _cli("gpuwm", cwd=root)
    assert whole.returncode == 1, whole.stdout + whole.stderr
    assert part.returncode == 1, part.stdout + part.stderr
    assert "gpuwm/core/leaky.py" in whole.stdout
    assert "gpuwm/core/leaky.py" in part.stdout


def test_the_cli_says_how_many_files_it_scanned():
    """The number that makes a zero-file scan visible instead of green.

    The tool printed "0 known occurrence(s) suppressed" against a
    29-entry baseline for months and nobody read it as a scan of
    nothing.  A file count is not a diagnosis anybody has to infer.
    """

    completed = _cli(".", cwd=REPO_ROOT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "scanned" in completed.stdout, completed.stdout
    assert " 0 file(s)" not in completed.stdout, completed.stdout


def test_the_documented_multi_argument_invocation_scans_the_real_zones():
    """The exact hand-run line, on this repository, over real files."""

    completed = _cli("gpuwm", "tools", "crates", cwd=REPO_ROOT)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "scanned" in completed.stdout
    assert " 0 file(s)" not in completed.stdout, completed.stdout


def test_a_target_holding_no_protected_zone_is_refused(tmp_path: Path):
    """A scan of nothing must not be reportable as a clean tree."""

    empty = tmp_path / "not-a-checkout"
    empty.mkdir()
    completed = _cli(str(empty), cwd=tmp_path)
    assert completed.returncode == 2, completed.stdout + completed.stderr
    message = completed.stderr
    assert "protected zone" in message, message
    assert "check_case_token_leakage.py ." in message, message


def test_resolve_target_finds_the_repository_root_above_a_subtree():
    """The baseline is keyed on REPOSITORY-relative paths, so the root the
    scan reports against has to be the repository, not the argument."""

    assert resolve_target(REPO_ROOT / "gpuwm") == (REPO_ROOT, "gpuwm")
    assert resolve_target(REPO_ROOT) == (REPO_ROOT, None)


def test_zones_under_a_subtree_are_the_zones_inside_it():
    assert zones_under(None) == PROTECTED_ZONES
    inside = zones_under("gpuwm")
    assert "gpuwm/core/" in inside
    assert "tools/matched_wrfout_envelope.py" not in inside
    # A target INSIDE a directory zone narrows that zone to the target.
    assert zones_under("gpuwm/core/dyn") == ("gpuwm/core/dyn/",)


def test_scan_zones_counts_the_files_it_read(tmp_path: Path):
    root = _leaky_tree(tmp_path)
    violations, files = scan_zones(root, zones_under("gpuwm"))
    assert files == 1
    assert [v.path for v in violations] == ["gpuwm/core/leaky.py"] * 2
    assert scan_zones(root, zones_under("tools"))[1] == 0

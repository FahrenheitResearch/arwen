"""The published FTZ statement is generated from the receipt, or it fails.

Every assertion here is about RENDERED TEXT rather than about the tool
that produced it: the criteria this file implements are all of the form
"what got published cannot say X", and a test that only inspected the
renderer would pass while the docs said anything at all.

Each guard ships with the control that makes it non-vacuous -- a
synthetic block carrying exactly the defect, which the guard must reject.
The universal-quantifier control is the retired sentence itself.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.ftz_receipt import probe  # noqa: E402
from tools.ftz_receipt import render_statement as rs  # noqa: E402
from tools.ftz_receipt import sass  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RECEIPT_DIR = ROOT / "tools" / "ftz_receipt" / "receipt"
FIXTURES = ROOT / "tools" / "ftz_receipt" / "fixtures"

#: The sentence this package retired.  It is quoted here as a CONTROL
#: INPUT, never as a claim: the guards below must reject it.
RETIRED_UNIVERSAL = "No loader or kernel supplies an FTZ option."


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads((RECEIPT_DIR / "receipt.json").read_text(
        encoding="utf-8"))


@pytest.fixture(scope="module")
def rendered() -> dict:
    return rs.render_all(ROOT)


def _published(name: str, relpath: str) -> str:
    text = (ROOT / relpath).read_text(encoding="utf-8")
    begin, end = rs.BEGIN.format(name=name), rs.END.format(name=name)
    start, stop = text.find(begin), text.find(end)
    assert start != -1 and stop != -1, f"{relpath}: no {name} region"
    return text[start:stop + len(end)] + "\n"


# ---- the statement is generated, not written ------------------------------

def test_every_block_regenerates_byte_identically(rendered):
    for name, relpath in rs.TARGETS:
        assert _published(name, relpath) == rendered[name], relpath


def test_the_template_states_no_verdict():
    """The outcome words can only have come from the receipt."""
    text = (ROOT / "tools" / "ftz_receipt" / rs.TEMPLATE_NAME).read_text(
        encoding="utf-8")
    for verdict in probe.VERDICTS:
        assert verdict not in text, verdict


def test_the_renderer_states_no_verdict_either():
    text = (ROOT / "tools" / "ftz_receipt"
            / "render_statement.py").read_text(encoding="utf-8")
    for verdict in probe.VERDICTS:
        assert verdict not in text, verdict


def test_the_published_verdict_words_are_the_receipt_s(receipt, rendered):
    """Each verdict the receipt measured appears; none that it did not."""
    published = "".join(rendered.values())
    measured = {cell["verdict"] for cell in receipt["cells"]}
    for verdict in measured:
        assert f"`{verdict}`" in published, verdict
    for verdict in set(probe.VERDICTS) - measured:
        assert f"`{verdict}`" not in published, verdict


def test_a_stale_block_is_caught(tmp_path, rendered):
    """The byte-identity check must fail on an edited block."""
    staged = tmp_path / "PROVENANCE.md"
    name = rs.TARGETS[0][0]
    published = _published(name, rs.TARGETS[0][1])
    staged.write_text(published.replace("6 of 6", "5 of 6"),
                      encoding="utf-8", newline="\n")
    assert staged.read_text(encoding="utf-8") != rendered[name]


# ---- no universal quantifier over loaders ---------------------------------

def test_no_rendered_block_quantifies_over_the_compile_surface(rendered):
    for name, block in rendered.items():
        assert rs.universal_quantifier_problems(block) == [], name


def test_the_retired_universal_is_still_rejected():
    """The named control: the sentence this package replaced."""
    problems = rs.universal_quantifier_problems(RETIRED_UNIVERSAL)
    assert problems, "the guard would not have caught the sentence it exists " \
                     "to catch"


@pytest.mark.parametrize("sentence", [
    "Every loader supplies the same options.",
    "All kernels preserve subnormals.",
    "None of the routes enables FTZ.",
    "No compile site asks for it.",
])
def test_other_universals_are_rejected_too(sentence):
    assert rs.universal_quantifier_problems(sentence)


@pytest.mark.parametrize("sentence", [
    "The loader at `gpuwm/core/kernels/__init__.py:20` supplies `-std=c++17`.",
    "Two routes measured alike and one did not.",
])
def test_enumerated_sentences_are_accepted(sentence):
    assert rs.universal_quantifier_problems(sentence) == []


# ---- no claim rests on a mirror -------------------------------------------

def test_no_published_line_rests_on_a_mirror_alone(receipt, rendered):
    for name, block in rendered.items():
        assert rs.mirror_only_problems(block, receipt) == [], name


def test_a_mirror_only_citation_is_rejected(receipt):
    """The named control: a line whose whole evidence set is a mirror."""
    mirrors = [name for name, record in receipt["artifacts"].items()
               if record.get("provenance") == "mirror"]
    assert mirrors, "the receipt carries no mirror to control against"
    line = f"- The instruction carries no modifier " \
           f"(`tools/ftz_receipt/receipt/{mirrors[0]}`)."
    assert rs.mirror_only_problems(line, receipt)


def test_a_mixed_citation_survives(receipt):
    mirrors = [name for name, record in receipt["artifacts"].items()
               if record.get("provenance") == "mirror"]
    line = (f"- Measured (`tools/ftz_receipt/receipt/bitpatterns.csv`) and "
            f"read (`tools/ftz_receipt/receipt/{mirrors[0]}`).")
    assert rs.mirror_only_problems(line, receipt) == []


# ---- every citation resolves and survives the release filter --------------

def test_published_citations_resolve_and_ship(rendered):
    for name, block in rendered.items():
        assert rs.citation_problems(block, ROOT) == [], name


def test_an_unresolvable_citation_is_rejected():
    line = "- See `tools/ftz_receipt/receipt/no_such_artifact.json`."
    problems = rs.citation_problems(line, ROOT)
    assert any("does not resolve" in problem for problem in problems)


def test_a_release_excluded_citation_is_rejected(tmp_path):
    """The named control: the same citation under an exclusion that hits it."""
    staged = tmp_path / "tree"
    (staged / "tools" / "ftz_receipt" / "receipt").mkdir(parents=True)
    (staged / "tools" / "ftz_receipt" / "receipt" / "receipt.json").write_text(
        "{}", encoding="utf-8")
    line = "- See `tools/ftz_receipt/receipt/receipt.json`."
    (staged / "RELEASE-EXCLUDE.txt").write_text("# none\n", encoding="utf-8")
    assert rs.citation_problems(line, staged) == []
    (staged / "RELEASE-EXCLUDE.txt").write_text(
        "tools/ftz_receipt/**\n", encoding="utf-8")
    problems = rs.citation_problems(line, staged)
    assert any("release-excluded" in problem for problem in problems)


# ---- docs follow the measurement in time ----------------------------------

def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True,
                          text=True, check=True).stdout.strip()


def test_the_receipt_commit_precedes_every_generated_block(receipt):
    """No reconciled wording exists in the tree before its measurement."""
    receipt_commit = _git(
        "log", "--reverse", "--format=%H", "--",
        "tools/ftz_receipt/receipt/receipt.json").splitlines()
    assert receipt_commit, "the receipt is not committed"
    first_receipt = receipt_commit[0]
    for name, relpath in rs.TARGETS:
        marker = rs.BEGIN.format(name=name)
        introduced = _git("log", "--reverse", "--format=%H",
                          f"-S{marker}", "--", relpath).splitlines()
        assert introduced, f"{relpath}: no commit introduces the {name} block"
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", first_receipt,
             introduced[0]],
            cwd=str(ROOT), capture_output=True, text=True)
        assert result.returncode == 0, (
            f"{relpath}: the {name} block was written at {introduced[0][:8]}, "
            f"which does not descend from the receipt commit "
            f"{first_receipt[:8]}")


# ---- the SASS tier --------------------------------------------------------

def test_sass_tier_is_labelled_and_matches_disk(receipt):
    block = receipt["artifacts"]["sass"]
    assert block["provenance"] in {"shipped-path", "unavailable",
                                   "not-collected"}
    if block["provenance"] != "shipped-path":
        assert block["reason"], "an absent SASS tier must say why"
        return
    assert block["disassembler"]
    produced = 0
    for name, record in receipt["artifacts"].items():
        if not name.startswith("sass/"):
            continue
        path = RECEIPT_DIR / name
        assert path.exists(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == \
            record["sha256"], name
        assert record["provenance"] == "shipped-path", name
        assert record["derived_from"] in receipt["artifacts"], name
        assert receipt["artifacts"][record["derived_from"]]["sha256"] == \
            record["derived_from_sha256"], name
        produced += 1
    assert produced == block["produced"]


def test_sass_counts_reproduce_from_the_committed_text(receipt):
    for name, record in receipt["artifacts"].items():
        if not name.startswith("sass/"):
            continue
        text = (RECEIPT_DIR / name).read_text(encoding="utf-8")
        assert sass.sass_ftz_modifiers(text) == record["ftz_modifiers"], name


def test_sass_parser_control_separates_the_fixture_pair():
    present = sass.sass_ftz_modifiers(
        (FIXTURES / "ftz_sass_present.sass").read_text(encoding="utf-8"))
    absent = sass.sass_ftz_modifiers(
        (FIXTURES / "ftz_sass_absent.sass").read_text(encoding="utf-8"))
    assert present["float_instruction_count"] == \
        absent["float_instruction_count"]
    assert present["with_ftz"] and not present["without_ftz"]
    assert absent["without_ftz"] and not absent["with_ftz"]


def test_a_constant_answer_sass_parser_fails_the_pair():
    """A parser that ignored its input would pass the test above."""
    def constant(_text: str) -> dict:
        return {"with_ftz": {"FMUL": 1}, "without_ftz": {},
                "float_instruction_count": 4}

    present = constant((FIXTURES / "ftz_sass_present.sass").read_text(
        encoding="utf-8"))
    absent = constant((FIXTURES / "ftz_sass_absent.sass").read_text(
        encoding="utf-8"))
    assert present == absent
    with pytest.raises(AssertionError):
        assert absent["without_ftz"] and not absent["with_ftz"]


def test_the_disassembly_describes_the_committed_bytes(receipt):
    """A SASS artifact may only be derived from a digest the receipt pins."""
    for name, record in receipt["artifacts"].items():
        if not name.startswith("sass/"):
            continue
        source = RECEIPT_DIR / record["derived_from"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == \
            record["derived_from_sha256"], name


def test_elf_arch_is_read_from_the_object_not_assumed(receipt):
    cubins = [name for name in receipt["artifacts"]
              if name.startswith("cubin/") and (RECEIPT_DIR / name).exists()]
    assert cubins
    for name in cubins:
        arch = sass.elf_sm_arch((RECEIPT_DIR / name).read_bytes())
        assert arch.startswith("SM") and arch[2:].isdigit(), (name, arch)

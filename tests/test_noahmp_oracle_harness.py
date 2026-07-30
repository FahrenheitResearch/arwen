"""Negative controls for the Noah-MP leaf-visibility exposure gate.

`tools/noahmp_wrf461_oracle/build_leaves.sh` compiles a *visibility-patched*
copy of `phys/module_sf_noahmplsm.F` so the 50 `private ::` leaf routines can
be called from a separate program unit.  `check_visibility_patch.py` is the
entire safety argument for that edit, so it needs its own negative controls:
`tests/test_noahmp_oracle.py` checks that the committed patch *body* is
visibility-only, but nothing there checks that the auditor would **reject** a
patched source that is not.

That distinction matters more than it looks, because of the order the auditor
runs its checks in.  `audit()` verifies three pinned SHA-256 digests *first*
and only then compares the two sources line by line.  So on today's committed
bytes the structural checks are unreachable -- any perturbation is caught by
the hash, and checks 4-7 never execute.

They stop being unreachable the moment somebody adds a leaf.  Lifting a 51st
symbol changes the patch, which changes `PATCH_SHA256` and `PATCHED_SHA256`,
and a fresh agent updating those pins is *supposed* to update them to whatever
their new file hashes to.  At that instant the hash gate is satisfied by
construction and the line-by-line audit is the only thing standing between the
project and an arbitrary edit to a pinned physics source.

These tests therefore drive the structural checks with the hash gate satisfied,
which is the state a re-pinning agent will actually be in.  They run on a
synthetic module of the same shape so they need neither WSL nor the pinned WRF
tree; the build-level control against the real 9,300-line source is
`tools/noahmp_wrf461_oracle/build_visibility_crosscheck.sh`.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
ORACLE_TOOLS = REPO_ROOT / "tools" / "noahmp_wrf461_oracle"
CHECKER_PATH = ORACLE_TOOLS / "check_visibility_patch.py"
CROSSCHECK_PATH = ORACLE_TOOLS / "build_visibility_crosscheck.sh"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "noahmp_visibility_check_nc", CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A module of the same shape as the real one: `private ::` accessibility
# statements, a `PRIVATE` entity attribute that must survive untouched, and a
# numeric literal standing in for physics.
PRISTINE = """\
module toy_lsm
  implicit none

  private :: ALPHA
  private :: BETA
  private :: GAMMA

  INTEGER, PRIVATE, PARAMETER :: NSOIL = 4

contains

  subroutine ALPHA(x)
    real :: x
    x = 3.2217E-6 * x * x
  end subroutine ALPHA

  subroutine BETA(y)
    real :: y
    y = y + 1.0
  end subroutine BETA

  subroutine GAMMA(z)
    real :: z
    z = z * 2.0
  end subroutine GAMMA

end module toy_lsm
"""

PATCHED = (PRISTINE
           .replace("  private :: ALPHA", "  public :: ALPHA")
           .replace("  private :: BETA", "  public :: BETA")
           .replace("  private :: GAMMA", "  public :: GAMMA"))

LIFTED = ("ALPHA", "BETA", "GAMMA")


@pytest.fixture
def gate(tmp_path, monkeypatch):
    """The auditor, aimed at a synthetic pair, with the pinned set replaced.

    Returns a callable ``run(patched_text, *, repin=True)``.  With
    ``repin=True`` the patched digest is re-pinned to whatever the caller's
    bytes hash to, which puts the auditor in exactly the state a fresh agent
    is in after adding a leaf: hash gate satisfied, structure unverified.
    """
    checker = _load_checker()
    monkeypatch.setattr(checker, "LIFTED_SYMBOLS", LIFTED)

    pristine_path = tmp_path / "pristine.F"
    patch_path = tmp_path / "visibility.patch"
    # write_bytes, never write_text: on Windows the default newline
    # translation would turn these LF sources into CRLF, changing every digest
    # and leaving a stray \r on each line for the auditor's regexes.
    pristine_path.write_bytes(PRISTINE.encode("ascii"))
    patch_path.write_bytes(b"(placeholder; audit only hashes it)\n")

    def sha(text: str) -> str:
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    monkeypatch.setattr(checker, "PRISTINE_SHA256", sha(PRISTINE))
    monkeypatch.setattr(
        checker, "PATCH_SHA256",
        hashlib.sha256(patch_path.read_bytes()).hexdigest())

    def run(patched_text: str, *, repin: bool = True):
        patched_path = tmp_path / "patched.F"
        patched_path.write_bytes(patched_text.encode("ascii"))
        if repin:
            monkeypatch.setattr(checker, "PATCHED_SHA256", sha(patched_text))
        else:
            monkeypatch.setattr(checker, "PATCHED_SHA256", sha(PATCHED))
        return checker.audit(pristine_path, patched_path, patch_path)

    run.checker = checker
    return run


def test_the_honest_visibility_patch_is_accepted(gate):
    """Control: the gate must not reject a genuine visibility-only rewrite."""
    assert gate(PATCHED) == LIFTED


def test_hash_gate_bites_when_the_patched_digest_is_not_repinned(gate):
    """A perturbed source with a stale pin is caught by the digest alone."""
    mutant = PATCHED.replace("3.2217E-6", "3.2218E-6")
    with pytest.raises(gate.checker.VisibilityPatchError, match="sha256"):
        gate(mutant, repin=False)


# --------------------------------------------------------------------------
# With the hash gate satisfied -- the post-re-pin state -- the structural
# checks must still reject every one of these.
# --------------------------------------------------------------------------

def test_one_non_visibility_character_is_rejected(gate):
    """The brief's control: perturb ONE character of physics, gate must bite.

    `3.2217E-6` -> `3.2218E-6` is a single-character edit to a live numeric
    literal, with both hashes consistent.  Only the line-by-line audit can
    catch it.
    """
    mutant = PATCHED.replace("3.2217E-6", "3.2218E-6")
    assert sum(a != b for a, b in zip(PATCHED, mutant)) == 1
    with pytest.raises(gate.checker.VisibilityPatchError,
                       match="not a visibility substitution"):
        gate(mutant)


def test_an_added_line_is_rejected(gate):
    mutant = PATCHED.replace("    y = y + 1.0\n",
                             "    y = y + 1.0\n    y = y * 2.0\n")
    with pytest.raises(gate.checker.VisibilityPatchError,
                       match="line count changed"):
        gate(mutant)


def test_a_deleted_line_is_rejected(gate):
    mutant = PATCHED.replace("    y = y + 1.0\n", "")
    with pytest.raises(gate.checker.VisibilityPatchError,
                       match="line count changed"):
        gate(mutant)


def test_lifting_a_symbol_outside_the_pinned_set_is_rejected(gate, monkeypatch):
    """Exposing a routine nobody adjudicated must fail.

    This is the realistic failure mode when a fresh agent adds a leaf: the
    substitution is perfectly well formed -- correct indentation, same symbol
    either side of the arrow -- and both hashes are re-pinned. Only the pinned
    symbol list notices that GAMMA was never approved for exposure.
    """
    monkeypatch.setattr(gate.checker, "LIFTED_SYMBOLS", ("ALPHA", "BETA"))
    with pytest.raises(gate.checker.VisibilityPatchError,
                       match="lifted symbol list does not match"):
        gate(PATCHED)


def test_a_surviving_private_statement_is_rejected(gate):
    """Half-applied patches must fail, not silently expose fewer symbols."""
    mutant = PATCHED.replace("  public :: BETA", "  private :: BETA")
    with pytest.raises(gate.checker.VisibilityPatchError,
                       match="private statement survived"):
        gate(mutant)


def test_changed_indentation_is_rejected(gate):
    mutant = PATCHED.replace("  public :: ALPHA", "    public :: ALPHA")
    with pytest.raises(gate.checker.VisibilityPatchError,
                       match="indentation changed"):
        gate(mutant)


def test_a_symbol_renamed_across_the_substitution_is_rejected(gate):
    """`private :: ALPHA` -> `public :: ALPHA_2` is visibility-shaped but wrong.

    Caught by the same-text-after-the-keyword check, before the symbol list is
    even consulted.
    """
    mutant = PATCHED.replace("  public :: ALPHA\n", "  public :: ALPHA_2\n")
    with pytest.raises(gate.checker.VisibilityPatchError,
                       match="text after the keyword changed"):
        gate(mutant)


def test_flipping_a_private_entity_attribute_is_rejected(gate):
    """Module constants must stay private: that is an entity attribute, not a
    visibility statement, and the real module has three of them (MBAND, NSOIL,
    NSTAGE)."""
    mutant = PATCHED.replace("INTEGER, PRIVATE, PARAMETER",
                             "INTEGER, PUBLIC, PARAMETER")
    with pytest.raises(gate.checker.VisibilityPatchError,
                       match="not a visibility substitution"):
        gate(mutant)


def _audit_pair(checker, pristine_text: str, patched_text: str, tmp_path: Path):
    """Run `audit` on an arbitrary pair with all three digests re-pinned."""
    pristine = tmp_path / "pair_pristine.F"
    patched = tmp_path / "pair_patched.F"
    patch = tmp_path / "pair.patch"
    pristine.write_bytes(pristine_text.encode("ascii"))
    patched.write_bytes(patched_text.encode("ascii"))
    patch.write_bytes(b"x\n")
    checker.PRISTINE_SHA256 = hashlib.sha256(pristine.read_bytes()).hexdigest()
    checker.PATCHED_SHA256 = hashlib.sha256(patched.read_bytes()).hexdigest()
    checker.PATCH_SHA256 = hashlib.sha256(patch.read_bytes()).hexdigest()
    return checker.audit(pristine, patched, patch)


def test_the_auditor_requires_at_least_one_private_entity_attribute(
        gate, tmp_path):
    """A source with no `, PRIVATE,` attribute at all must not pass silently.

    Guards the `not attr_before` clause: if a future refactor stripped the
    attributes out of the pristine source, the "attributes were not altered"
    equality would otherwise be vacuously true and the module's private
    constants could be exposed without the gate noticing.
    """
    attribute = "  INTEGER, PRIVATE, PARAMETER :: NSOIL = 4\n"
    replacement = "  INTEGER, PARAMETER :: NSOIL = 4\n"
    stripped = PRISTINE.replace(attribute, replacement)
    stripped_patched = PATCHED.replace(attribute, replacement)
    assert stripped != PRISTINE and stripped_patched != PATCHED
    with pytest.raises(gate.checker.VisibilityPatchError,
                       match="PRIVATE entity attributes"):
        _audit_pair(gate.checker, stripped, stripped_patched, tmp_path)


def test_crosscheck_script_is_present_and_lf_only():
    """The whole-column control ships with the harness and stays runnable.

    `.gitattributes` pins `*.sh text eol=lf`; a CRLF working copy makes the
    script unrunnable under WSL bash (`set: pipefail: invalid option name`),
    which has bitten this project before.
    """
    payload = CROSSCHECK_PATH.read_bytes()
    assert b"\r" not in payload, "build_visibility_crosscheck.sh is not LF-only"
    text = payload.decode("ascii")
    # The stages that make the cross-check meaningful.
    assert "compare_object_code.py" in text
    assert "compile_module pristine" in text
    assert "compile_module patched" in text
    assert "mutant-literal" in text and "mutant-operator" in text
    assert "CROSSCHECK IS VACUOUS" in text
    # The driver-linkage constraint that confines the patch to the harness.
    assert "ambiguous reference to .albedo." in text

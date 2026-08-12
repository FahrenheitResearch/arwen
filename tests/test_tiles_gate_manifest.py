"""The streamed-run lane's gate list, kept honest in the repository.

``tools/battery/tiles_gates.txt`` is how the battery finds the ``[tiles]``
gates.  They are not pytest -- each is a module whose ``main()`` prints a
matrix and one verdict line -- so nothing collects them by accident and
nothing notices when one stops existing.  That is the same hole
``tools/battery/stage1_files.txt`` was written to close for the stage-1
list (task #122), and this suite is the same kind of gate on the same
kind of file: every module resolves, nothing is listed twice, every shard
is one the battery knows how to run, and every entry says what it proves.

It is deliberately cheap and hermetic -- it reads two text files and
touches no card -- because a manifest gate that needs a GPU is a manifest
gate that runs at a release cut instead of on every commit.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPOSITORY_ROOT / "tools" / "battery" / "tiles_gates.txt"

#: The shards the battery knows how to run.  ``cpu`` entries are runnable
#: in the stage-1 environment (``GPUWM_NO_LOCAL_GPU=1``); ``gpu`` entries
#: need a card and go on the GPU leg.
SHARDS = ("cpu", "gpu")


def _lines() -> list[str]:
    return MANIFEST.read_text(encoding="utf-8").splitlines()


def _entries() -> list[tuple[str, str, tuple[str, ...]]]:
    """``[(shard, dotted module, env assignments)]``, as the battery reads it.

    The env column is the third field onward.  It exists because the slice
    gate has a rung selector whose default is the weaker rung, so the
    module name alone is not a complete invocation.
    """

    entries = []
    for raw in _lines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        entries.append((parts[0],
                        parts[1] if len(parts) > 1 else "",
                        tuple(parts[2:])))
    return entries


def _reasons() -> dict[str, str]:
    """Each entry line mapped to the comment block above its group.

    A block covers the whole CONTIGUOUS run of entries beneath it, so two
    invocations of one gate that differ only in environment share the
    rationale that explains the gate.  A blank line ends the group.
    """

    reasons: dict[str, str] = {}
    block: list[str] = []
    for raw in _lines():
        line = raw.strip()
        if not line:
            block = []
        elif line.startswith("#"):
            block.append(line.lstrip("#").strip())
        else:
            reasons[" ".join(line.split())] = " ".join(p for p in block if p)
    return reasons


def test_the_manifest_exists_and_lists_something() -> None:
    assert MANIFEST.is_file(), f"{MANIFEST} is missing"
    assert _entries(), (
        "tools/battery/tiles_gates.txt parsed to zero entries; the battery "
        "would run no gate for [tiles] and still report a clean leg")


@pytest.mark.parametrize("entry", _entries(),
                         ids=lambda e: " ".join((e[1],) + e[2]))
def test_every_listed_gate_is_a_module_in_this_tree(entry) -> None:
    shard, module, env = entry
    assert shard in SHARDS, (
        f"{module} is on shard {shard!r}; the battery knows {SHARDS}")
    path = REPOSITORY_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    assert path.is_file(), (
        f"{module} is listed in tools/battery/tiles_gates.txt but "
        f"{path.relative_to(REPOSITORY_ROOT).as_posix()} does not exist; "
        "`python -m` would fail the whole leg on the missing module, so fix "
        "the path or drop the entry")


def test_no_gate_is_listed_twice() -> None:
    """Identity is module PLUS environment.

    One module listed twice under different environment is two
    invocations of one gate, which is the point of the env column; the
    same module under the same environment is a doubled leg whose
    counts would overstate the coverage.
    """

    invocations = [(module, env) for _, module, env in _entries()]
    duplicates = sorted({i for i in invocations
                         if invocations.count(i) > 1})
    assert not duplicates, (
        f"tools/battery/tiles_gates.txt lists {duplicates} more than once")


@pytest.mark.parametrize("entry", _entries(),
                         ids=lambda e: " ".join((e[1],) + e[2]))
def test_every_gate_carries_its_reason_inline(entry) -> None:
    shard, module, env = entry
    line = " ".join((shard, module) + env)
    reason = _reasons().get(line, "")
    assert reason, (
        f"{module} has no comment block above it in "
        "tools/battery/tiles_gates.txt; the amendment discipline the file's "
        "own header states is that an addition names what the gate proves "
        "that nothing else does")


def test_the_consolidated_gate_and_the_hook_contract_are_both_listed() -> None:
    """Both, or the list proves less than it looks like it proves.

    ``tilestream.test_gate`` reported 233 PASS / 0 FAIL for the whole
    period in which every real-case streamed run died at its first tile
    change: the gate lane is periodic by design, so no configuration in
    it ever calls ``tile_hook``, and the lateral-boundary seam has no
    coverage there in principle rather than by omission.  A list that
    carries the consolidated gate alone would read as complete and would
    be blind in exactly the place the lane has already been burned.
    """

    modules = {module for _, module, _env in _entries()}
    for required in ("tilestream.test_gate",
                     "tilestream.test_tile_hook_contract"):
        assert required in modules, (
            f"{required} was dropped from tools/battery/tiles_gates.txt; the "
            "consolidated gate and the tile_hook contract cover different "
            "seams and neither substitutes for the other")


#: The invocations whose ENVIRONMENT is the acceptance, module name alone
#: being the weaker half of each gate.  ``PHYSICS=1`` takes the slice gate
#: from 22 reassembled arrays to 246; ``GRID=2x2`` takes the seam gate from
#: one split axis to two, which is the only way it gates the corner
#: exchange.  Both were listed late, and in both cases the real acceptance
#: had been living in somebody's shell history rather than in the file --
#: the exact failure the environment column exists to close.
REAL_ACCEPTANCE_INVOCATIONS = (
    ("tilestream.test_decomp_gate", ("PHYSICS=1",)),
    ("tilestream.test_seam_gate", ("GRID=2x2",)),
)


@pytest.mark.parametrize(("module", "env"), REAL_ACCEPTANCE_INVOCATIONS,
                         ids=lambda v: v if isinstance(v, str) else "+".join(v))
def test_the_stronger_invocation_of_each_two_rung_gate_is_listed(
        module: str, env: tuple[str, ...]) -> None:
    """Entry identity is (module, environment), so this is not redundant.

    ``test_every_listed_gate_is_a_module_in_this_tree`` walks whatever the
    file happens to say and stays green when the stronger row is deleted:
    the module is still listed, once, under its weaker default.  Nothing
    else in this suite can tell the difference, and the deletion would read
    as tidying a duplicate line.  These two rows are pinned by name because
    losing either one silently narrows what the battery accepts, which is
    what happened before they were added.
    """

    invocations = {(m, e) for _shard, m, e in _entries()}
    assert (module, env) in invocations, (
        f"`{' '.join(env)} python -m {module}` is not in "
        "tools/battery/tiles_gates.txt.  That invocation IS the acceptance: "
        "the module's own default runs the weaker rung, so the battery would "
        "report a green leg for a gate that never ran the geometry or the "
        "carriers it exists to cover.  Listed invocations: "
        + ", ".join(sorted(" ".join((m,) + e) for m, e in invocations)))


def test_at_least_one_gate_runs_without_a_card() -> None:
    """A GPU-only list is a list that runs at a cut, not on a commit."""

    assert any(shard == "cpu" for shard, _m, _e in _entries()), (
        "every entry in tools/battery/tiles_gates.txt needs a card, so "
        "nothing in the lane is checked between release cuts")


def test_the_file_is_lf_only() -> None:
    """``.gitattributes`` says ``* -text``: disk bytes are committed bytes."""

    assert b"\r" not in MANIFEST.read_bytes(), (
        "tools/battery/tiles_gates.txt contains CR bytes; the repository "
        "commits LF and .gitattributes does no conversion")

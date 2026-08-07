"""Negative control for ``tools/check_parameter_claims.py``.

``tests/test_physics_registry_declarations.py`` asserts the gate PASSES on
the real tree, which is only evidence if the gate can also fail.  These pins
hold the other half: each builds a scratch root where the citation is
deliberately hollow and asserts the gate rejects it.

The hollow forms are the ones the gate promises to catch -- a knob named in
a module docstring, in a class or function docstring, in either quote style,
or in a comment.  The last two pins guard the opposite regression: a knob
read through a non-docstring string literal is real code and must still
pass, and a citation that does not parse must fail closed rather than
silently prove nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm.physics_registry import physics_registry

from tools.check_parameter_claims import check


def _citations() -> dict[str, str]:
    """Every knob that claims a consuming read, mapped to the cited path."""
    cited: dict[str, str] = {}
    for name, spec in physics_registry()["parameters"].items():
        citation = spec.get("consuming_read")
        if spec.get("implemented") is not False and isinstance(citation, str):
            cited[name] = citation
    return cited


def _knobs_citing(path: str) -> list[str]:
    return sorted(name for name, cited in _citations().items() if cited == path)


def _shared_citation() -> str:
    """A citation several knobs claim, chosen from the registry each run.

    These gates need a path with MORE THAN ONE knob behind it, so that a
    hollowed-out file fails for every knob that cited it rather than for
    one.  The knob used to be named here literally (``km_opt``), which
    made the gate a hostage to where a selector lives: when km_opt moved
    out of ``parameters`` and became components/turbulence's selector
    key, three of these tests died with ``KeyError: 'km_opt'`` while the
    behaviour they gate had not changed at all.  Choosing the citation
    from the registry keeps the gate about citation parsing, which is
    what it is for.
    """

    by_path: dict[str, list[str]] = {}
    for name, citation in _citations().items():
        by_path.setdefault(citation, []).append(name)
    shared = sorted((path for path, knobs in by_path.items()
                     if len(knobs) > 1))
    assert shared, "no citation path is claimed by more than one knob"
    return shared[0]


def _honest_root(tmp_path: Path) -> Path:
    """A scratch root where every citation resolves to a genuine code read."""
    root = tmp_path / "root"
    for name, citation in sorted(_citations().items()):
        target = root / citation
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(f"{name} = cfg.{name}\n")
    return root


def _write(root: Path, citation: str, text: str) -> None:
    (root / citation).write_text(text, encoding="utf-8")


def _failures_for(root: Path, knobs: list[str]) -> list[str]:
    return [f for f in check(root) if f.split(":", 1)[0] in knobs]


@pytest.fixture()
def honest_root(tmp_path: Path) -> Path:
    root = _honest_root(tmp_path)
    assert check(root) == [], (
        "the scratch control root must pass before a pin hollows it out")
    return root


def test_a_module_docstring_naming_the_knob_is_not_a_read(honest_root: Path):
    """The audit's exact proof: a file that is nothing but prose must fail."""
    citation = _shared_citation()
    knobs = _knobs_citing(citation)
    _write(honest_root, citation, '"""Prose naming ' + " ".join(knobs) + '."""\n')

    failures = _failures_for(honest_root, knobs)
    assert len(failures) == len(knobs), failures
    for failure in failures:
        assert "only in a comment or docstring" in failure, failure


def test_a_function_docstring_naming_the_knob_is_not_a_read(honest_root: Path):
    citation = _citations()["moist"]
    knobs = _knobs_citing(citation)
    _write(
        honest_root,
        citation,
        "def f():\n    '''Prose naming " + " ".join(knobs) + ".'''\n    return 0\n",
    )

    failures = _failures_for(honest_root, knobs)
    assert len(failures) == len(knobs), failures
    for failure in failures:
        assert "only in a comment or docstring" in failure, failure


def test_a_class_docstring_naming_the_knob_is_not_a_read(honest_root: Path):
    citation = _citations()["usemonalb"]
    knobs = _knobs_citing(citation)
    _write(
        honest_root,
        citation,
        "class C:\n    \"\"\"Prose naming " + " ".join(knobs) + ".\"\"\"\n",
    )

    failures = _failures_for(honest_root, knobs)
    assert len(failures) == len(knobs), failures
    for failure in failures:
        assert "only in a comment or docstring" in failure, failure


def test_every_prose_only_citation_is_rejected_one_file_at_a_time(
        tmp_path: Path):
    """Sweep: hollowing out any one cited file must fail exactly its knobs."""
    citations = _citations()
    for citation in sorted(set(citations.values())):
        root = _honest_root(tmp_path / citation.replace("/", "_"))
        assert check(root) == []
        knobs = _knobs_citing(citation)
        _write(root, citation, '"""Prose naming ' + " ".join(knobs) + '."""\n')
        failures = _failures_for(root, knobs)
        assert len(failures) == len(knobs), (citation, failures)


def test_a_comment_naming_the_knob_is_not_a_read(honest_root: Path):
    citation = _citations()["ra_physics"]
    knobs = _knobs_citing(citation)
    _write(honest_root, citation, "# comment naming " + " ".join(knobs) + "\n")

    failures = _failures_for(honest_root, knobs)
    assert len(failures) == len(knobs), failures
    for failure in failures:
        assert "only in a comment or docstring" in failure, failure


def test_a_read_through_a_string_literal_still_counts_as_code(
        honest_root: Path):
    """Do not regress real code that happens to contain a string.

    ``gpuwm/core/acoustic.py`` proves ``moist_cq`` with
    ``getattr(cfg, "moist_cq", True)``.  Blanking every string literal --
    rather than only docstrings -- would fail that genuine read.
    """
    citation = _citations()["moist_cq"]
    knobs = _knobs_citing(citation)
    body = "".join(f'v = getattr(cfg, "{knob}", None)\n' for knob in knobs)
    _write(honest_root, citation, '"""Docstring naming nothing."""\n' + body)

    assert _failures_for(honest_root, knobs) == []


def test_a_hash_inside_a_string_does_not_truncate_the_read(honest_root: Path):
    """Splitting on ``#`` would cut the line before the read; tokenizing does not."""
    citation = _shared_citation()
    knobs = _knobs_citing(citation)
    body = "".join(f'label = "#{knob}"; v = cfg.{knob}\n' for knob in knobs)
    _write(honest_root, citation, body)

    assert _failures_for(honest_root, knobs) == []


def test_an_unparseable_citation_fails_closed(honest_root: Path):
    citation = _shared_citation()
    knobs = _knobs_citing(citation)
    _write(honest_root, citation, "def broken(:\n    " + knobs[0] + " = cfg\n")

    failures = _failures_for(honest_root, knobs)
    assert len(failures) == len(knobs), failures
    for failure in failures:
        assert "does not parse" in failure, failure

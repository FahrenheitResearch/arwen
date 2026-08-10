"""A shipped config may not carry an acknowledgement token silently.

WHY THIS FILE EXISTS.

An acknowledgement lifts a guard.  ``[experiment].acknowledgements`` is
the config-side declaration channel
(:mod:`gpuwm.physics_compat`), and by design a token there turns a
refusal into a run.  That is correct for a declared experiment and
catastrophic as a default, and the difference between the two is
entirely a matter of whether anybody wrote down WHY.

They did not.  ``asymmetric-radiation-nocturnal-window-v1`` shipped in
1.7.1 and by 1.8.7 ten shipped configs carried it, including
``configs/era5_wrf_direct_proof.toml`` and
``configs/gfs_wrf_direct_proof.toml`` -- the two files a new ERA5 or GFS
user copies first.  Both ran ``ra_lw_physics = 0`` with
``ra_sw_physics = 1``.  Copying one inherited the token, and the token
was checked BEFORE any physics was inspected, so the copy walked past
the guard into a forecast whose downward longwave was a frozen
300 W m-2.  Nothing in any of those files said what the declaration was
for; the comment above the token was the same four boilerplate lines in
every one of them, pasted along with the token.

So: a token is admissible, and an UNJUSTIFIED token is not.  Every
acknowledgement in every shipped config must be answered, in the same
file, by a line of the form

    # JUSTIFY <token>: <reason>

with a reason of real length.  That is a deliberately annoying thing to
type, which is the point -- it is the smallest edit that cannot be made
by copy-paste without noticing, and a reader who opens the file learns
why the guard is off before they learn that it is off.

WHAT THIS BAR IS, stated plainly so nobody mistakes it for more.  The
bar is PROCEDURAL: presence and length, not content.  200 characters of
lorem ipsum under a ``# JUSTIFY`` header pass it, and the block may sit
anywhere in the file (the domain wizard's own emissions put it in the
header comment, above the ``[experiment]`` table).  A test cannot judge
whether a reason is a reason; a REVIEWER can, and the convention's job
is to guarantee the reviewer a written claim to judge, attributable to
whoever typed it.  Nor does the convention stop propagation by itself:
copying a config carries the token AND its justification prose forward
verbatim.  What protects a copy is the pair of load guards -- a copy
that keeps the token keeps a false-but-visible claim a reviewer can
catch; a copy that drops it refuses to load.  The defect the 1.8.7
configs shipped -- proof descriptors users copy FIRST carrying a
standing bypass -- is closed by those files now computing real
radiation, not by this convention.

ONE EXEMPTION: BYTE-FROZEN RECORDS (1.8.8 lead ruling).  A config under
``configs/frozen/`` that ``FROZEN_RECORDS`` pins by sha256 is a record of
a committed run, not a menu item -- a published receipt names it by that
digest, so editing it to satisfy a rule written after it was frozen
would break the pin and leave the receipt naming bytes that no longer
exist.  Those files are skipped, their live siblings carry the rules in
their place, and the exemption is on BYTES rather than on the directory
name: an unpinned file under ``configs/frozen/``, or a pinned one whose
bytes moved, is governed like any other shipped config.  See
:func:`_is_frozen_record` and the two control tests below.

The three records under ``configs/frozen/`` have since been re-pinned,
for the one reason a record may move: at their old bytes they no longer
LOADED, the constant-downward-longwave guard refusing the pairing they
run.  Each gained that declaration in the JUSTIFY convention and its pin
followed, so today they satisfy this convention rather than needing the
exemption from it.  The exemption stands unchanged all the same: what
exempts a record is its pin, not its compliance, and the next rule
written after a record is frozen will need it exactly as this one did.

The test is a governance test, not a physics test: it reads files and
never runs anything.
"""
from __future__ import annotations

from pathlib import Path
import re
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]

#: Every directory a shipped config can live in.  ``tools`` is included
#: because ``tools/ens_sweep/kdmx_case.toml`` carried the nocturnal token
#: too -- a config that ships is a config that gets copied, wherever the
#: tree happens to keep it.
_SEARCH_ROOTS = ("configs", "tools")

#: The shortest reason this test will accept, counted over the whole
#: justification block.  Long enough that "legacy", "see the docs" and
#: "declared validation configuration" all fail -- which is the bar that
#: matters, because the boilerplate this replaces was 168 characters of
#: exactly that and it was pasted into ten files.  The shortest real
#: justification in the tree today is 356.
_MINIMUM_REASON_CHARS = 200

_JUSTIFY = re.compile(r"^\s*#\s*JUSTIFY\s+(?P<token>[A-Za-z0-9._-]+)\s*:"
                      r"(?P<reason>.*)$")


def _frozen_record_pins() -> dict[str, tuple[str, str]]:
    """The byte pins that make ``configs/frozen/`` mean something.

    Read from the suite that owns them rather than copied, because two
    copies of a digest is one copy that goes stale.  Both files live in
    ``tests/``, which pytest puts on ``sys.path``.
    """

    from test_shipped_configs_mixing_stability import FROZEN_RECORDS

    return FROZEN_RECORDS


def _is_frozen_record(path: Path) -> bool:
    """Is this an archived record, pinned at the bytes it is a record of?

    THE EXEMPTION, AND WHY IT IS NARROW (1.8.8 lead ruling).
    ``configs/frozen/**`` holds byte-frozen historical records: a
    committed receipt names each one by sha256, and the digest IS the
    artifact.  Editing one to satisfy a convention invented after it was
    frozen would break the pin, orphan the receipt, and make the file a
    record of nothing.  So the token conventions below -- the JUSTIFY
    line, and the nocturnal/constant-longwave union -- do not apply to
    them, and the LIVE config beside each record carries both.

    The exemption is on BYTES, not on a directory name, exactly as the
    mixing criterion's is: a file under ``configs/frozen/`` that no pin
    names, or one whose bytes no longer match the pin that names it, is
    not a record and is governed like any other shipped config.  That
    keeps ``configs/frozen/`` from becoming a place to park a config
    that would otherwise fail a rule.
    """

    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return False
    if not rel.startswith("configs/frozen/"):
        return False
    pin = _frozen_record_pins().get(rel)
    if pin is None:
        return False
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest() == pin[0]


def _shipped_configs() -> list[Path]:
    paths: list[Path] = []
    for name in _SEARCH_ROOTS:
        for path in sorted((ROOT / name).rglob("*.toml")):
            # Vendored trees are other people's manifests, not shipped
            # gpuwm configs: tools/rustwx/vendor alone holds ~340 Rust
            # Cargo.toml files, and scanning them means this test starts
            # reporting on any vendored TOML that ever grows an
            # [experiment] table it has no business governing.
            if "vendor" in path.relative_to(ROOT).parts:
                continue
            if _is_frozen_record(path):
                continue
            paths.append(path)
    return paths


def _declared_acknowledgements(text: str) -> tuple[str, ...]:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return ()
    experiment = raw.get("experiment")
    if not isinstance(experiment, dict):
        return ()
    declared = experiment.get("acknowledgements")
    if not isinstance(declared, (list, tuple)):
        return ()
    return tuple(str(value) for value in declared)


_COMMENT = re.compile(r"^\s*#\s?(?P<body>.*)$")


def _justifications(text: str) -> dict[str, str]:
    """Map token -> its justification prose.

    A justification runs from its ``# JUSTIFY <token>:`` line through the
    contiguous comment lines that follow it, stopping at the first
    non-comment line or the next ``# JUSTIFY``.  Continuation matters:
    these are 79-column TOML comment blocks, so a real reason is
    physically unable to fit on the opening line, and measuring only that
    line would reward a one-word reason that happens to be long.
    """

    found: dict[str, str] = {}
    token: str | None = None
    parts: list[str] = []

    def flush() -> None:
        if token is not None:
            found.setdefault(token, " ".join(parts).strip())

    for line in text.splitlines():
        match = _JUSTIFY.match(line)
        if match is not None:
            flush()
            token = match.group("token")
            parts = [match.group("reason").strip()]
            continue
        if token is None:
            continue
        comment = _COMMENT.match(line)
        if comment is None:
            flush()
            token, parts = None, []
            continue
        parts.append(comment.group("body").strip())
    flush()
    return found


def _configs_with_acknowledgements() -> list[tuple[Path, tuple[str, ...], str]]:
    rows = []
    for path in _shipped_configs():
        text = path.read_text(encoding="utf-8")
        declared = _declared_acknowledgements(text)
        if declared:
            rows.append((path, declared, text))
    return rows


def test_the_scan_finds_the_configs_that_carry_tokens():
    """The instrument, before it is trusted: it can see a token at all.

    A justification test that silently matched zero files would pass
    forever.  The tree HAS configs with acknowledgements -- the LES
    tornado ladder's six records among them -- so an empty scan means
    the reader is broken, not that the tree is clean.
    """
    rows = _configs_with_acknowledgements()
    assert rows, (
        "no shipped config declares an acknowledgement; either the tree "
        "changed or _declared_acknowledgements stopped reading them")
    names = {path.name for path, _, _ in rows}
    assert "les_tornado_100m_mayfield_20211210_attempt3.toml" in names


@pytest.mark.parametrize(
    "path,declared,text",
    [pytest.param(path, declared, text, id=path.relative_to(ROOT).as_posix())
     for path, declared, text in _configs_with_acknowledgements()])
def test_every_shipped_acknowledgement_is_justified_in_its_own_file(
        path, declared, text):
    """Each declared token has a ``# JUSTIFY <token>:`` line with a reason."""
    justified = _justifications(text)
    for token in declared:
        assert token in justified, (
            f"{path.relative_to(ROOT).as_posix()} declares the "
            f"acknowledgement {token!r} with no justification.  An "
            "acknowledgement lifts a guard, and a guard lifted without a "
            "written reason is how ten shipped configs came to run a "
            "frozen 300 W m-2 downward longwave.  Add a line reading\n"
            f"    # JUSTIFY {token}: <why THIS config needs it>\n"
            "to the file (anywhere in it; next to the token is best).")
        reason = justified[token]
        assert len(reason) >= _MINIMUM_REASON_CHARS, (
            f"{path.relative_to(ROOT).as_posix()} justifies {token!r} with "
            f"{len(reason)} characters ({reason!r}); at least "
            f"{_MINIMUM_REASON_CHARS} are required.  State what this "
            "config is and why the guard does not apply to it, not that a "
            "declaration exists.")


def test_an_unjustified_token_fails_the_check():
    """The detector fires -- proven on a config, not asserted.

    Both directions on one file: strip the justification line out of a
    real shipped config and the check must fail; leave it in and it must
    pass.  A guard test that has never been seen to fail is not a guard.
    """
    path = ROOT / "configs" / "les_tornado_100m_mayfield_20211210.toml"
    text = path.read_text(encoding="utf-8")
    declared = _declared_acknowledgements(text)
    assert declared, "fixture config no longer declares any acknowledgement"
    assert set(declared) <= set(_justifications(text))

    stripped = "\n".join(
        line for line in text.splitlines() if not _JUSTIFY.match(line))
    assert not set(declared) & set(_justifications(stripped))


def test_the_frozen_record_exemption_is_real_and_is_narrow():
    """The one exemption, both directions, on the file that needs it.

    ``configs/frozen/les_tornado_100m_mayfield_20211210.toml`` is the
    attempt #1 record: docs/les/ATTEMPT1-EXPECTATIONS.md registered it
    by sha256 before the run, so its bytes are the artifact and 1.8.8's
    two new rules -- a JUSTIFY line per token, and the nocturnal token
    never standing alone -- are rules it was written before.  It is
    skipped, and the live config it was frozen from satisfies both rules
    in its place.

    SATISFIED RATHER THAN NEEDED, since the 1.8.8 re-pin.  The record
    moved for the one reason a record may -- its old bytes no longer
    LOADED, the constant-downward-longwave guard refusing them -- and
    the declaration it gained is written in the JUSTIFY convention, so
    the file now passes both rules on its own merits.  The exemption is
    unchanged and stays: what makes a record exempt is its pin, not
    whether it happens to comply, and the next rule written after a
    record is frozen will find this arrangement where it left it.  The
    assertions below say the record COMPLIES, where they used to say it
    would fail; that is the fact that moved, not the exemption.

    The narrow half matters more than the exemption: a file under
    ``configs/frozen/`` that no pin names is NOT skipped.  Both halves
    are asserted here so the directory cannot quietly become a bypass.
    """

    from gpuwm.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                      CONSTANT_DOWNWARD_LONGWAVE_ACK)

    record = ROOT / "configs" / "frozen" / \
        "les_tornado_100m_mayfield_20211210.toml"
    assert record.is_file()
    assert _is_frozen_record(record)
    assert record not in _shipped_configs()
    # Exempt by its pin, and -- since the 1.8.8 re-pin -- compliant too.
    frozen_text = record.read_text(encoding="utf-8")
    frozen_tokens = _declared_acknowledgements(frozen_text)
    assert ASYMMETRIC_RADIATION_NOCTURNAL_ACK in frozen_tokens
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in frozen_tokens
    assert set(frozen_tokens) <= set(_justifications(frozen_text))
    # The live config carries what the rules ask, so the coverage the
    # exemption gives up is not coverage of this configuration.
    live = ROOT / "configs" / "les_tornado_100m_mayfield_20211210.toml"
    live_text = live.read_text(encoding="utf-8")
    live_tokens = _declared_acknowledgements(live_text)
    assert set(live_tokens) >= {ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                CONSTANT_DOWNWARD_LONGWAVE_ACK}
    assert set(live_tokens) <= set(_justifications(live_text))
    assert live in _shipped_configs()


def test_an_unpinned_file_under_configs_frozen_is_still_governed(
        monkeypatch):
    """A directory name exempts nothing; a matching byte pin does.

    Written against the real predicate with a real file, because the
    failure this guards against is somebody dropping a config into
    ``configs/frozen/`` to get past a rule -- which reads as compliant
    in a diff and is the whole hazard of an allowlist.
    """

    pinned = ROOT / "configs" / "frozen" / \
        "les_tornado_100m_mayfield_20211210.toml"
    intruder = pinned.parent / "not_a_record_probe.toml"
    assert not intruder.exists(), "leftover probe file from a failed run"
    intruder.write_bytes(pinned.read_bytes())
    try:
        # Same bytes as a pinned record, different name: no pin names it.
        assert not _is_frozen_record(intruder)
        assert intruder in _shipped_configs()
    finally:
        intruder.unlink()

    # And a pin whose digest no longer matches the file exempts nothing
    # either -- the case where somebody edits a record in place.  The
    # pin table is faked rather than the record edited, because editing
    # the record is precisely the act this arrangement forbids.
    rel = pinned.relative_to(ROOT).as_posix()
    monkeypatch.setattr(
        __name__ + "._frozen_record_pins",
        lambda: {rel: ("0" * 64, "a pin that does not match these bytes")})
    assert not _is_frozen_record(pinned)
    assert pinned in _shipped_configs()


def test_the_nocturnal_token_no_longer_covers_the_constant_longwave_one():
    """The two claims are separate tokens, and separately declared.

    The 1.7.1 nocturnal token was checked before any physics was
    inspected, so a config carrying it never had its GLW source looked
    at.  Any config that still runs a longwave-free land surface has to
    say BOTH things now.  Asserted on the shipped tree so a future edit
    that re-collapses them into one token is caught here.
    """
    from gpuwm.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                      CONSTANT_DOWNWARD_LONGWAVE_ACK)

    assert (ASYMMETRIC_RADIATION_NOCTURNAL_ACK
            != CONSTANT_DOWNWARD_LONGWAVE_ACK)
    for path, declared, _text in _configs_with_acknowledgements():
        if ASYMMETRIC_RADIATION_NOCTURNAL_ACK in declared:
            assert CONSTANT_DOWNWARD_LONGWAVE_ACK in declared, (
                f"{path.relative_to(ROOT).as_posix()} declares the "
                "nocturnal token alone.  Every shipped config that still "
                "needs it also runs a land-surface scheme with no "
                "longwave scheme, which is a second, separate claim.")


def test_the_two_proof_configs_ship_with_real_radiation():
    """The files a new user copies must not need a declaration at all.

    ``era5_wrf_direct_proof`` and ``gfs_wrf_direct_proof`` are the
    starting points for the two most-travelled ingest routes, and
    ``gfs_wrf_hierarchy_proof`` is the starting point for a nested tree.
    They shipped through 1.8.7 with a standing nocturnal bypass; the fix
    is that they compute their longwave, not that they declare better.
    """
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.experiment import load_experiment

    for name in ("era5_wrf_direct_proof", "gfs_wrf_direct_proof",
                 "gfs_wrf_hierarchy_proof"):
        experiment = load_experiment(ROOT / "configs" / f"{name}.toml")
        assert experiment.acknowledgements == (), (
            f"{name} still ships an acknowledgement; a config users copy "
            "must not carry a standing bypass")
        for domain in experiment.domains:
            lw, sw = radiation_scheme_ids(domain.run)
            assert lw > 0 and sw > 0, (
                f"{name} d{domain.grid_id:02d} resolves lw={lw} sw={sw}; "
                "both radiation streams must be on")

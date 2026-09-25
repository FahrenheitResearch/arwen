"""The render door asks for what the catalog says it can draw, and no more.

WHAT BREAKAGE THESE PIN (gate law).  ``rw_wrfbatch`` exits nonzero when
``summary.failed > 0``, so ONE product it cannot draw for these frames
discards the whole ``--series`` invocation.  Measured on the shipped
2.7.5 wheel, a sub-hourly child series with ``qpf_1h`` forwarded onto
it::

    FAILED qpf_1h F000: invalid store metadata: run wrf/local_... uses an
    exact-time ordinal axis; production batch/windowed rendering is
    disabled until render requests carry exact lead and valid times
    batch render incomplete: rendered=13 skipped=0 failed=13

exit 1, while the identical call without that one slug exits 0 and draws
the same 13 pictures.  On the downscale route that render is the
finalize stage of a finished child, so it turned six and a half hours of
integration into "Forecast failed" over a partial picture tree.

The door's guard used to match two of the catalog's FIVE windowed
outcomes, by their English sentences.  These cases pin the verdict form:
every NAMED slug whose row is not ``renderable`` is dropped with the
engine's own reason, whatever status or wording the row carries, and a
group keyword is never second-guessed.

Nothing here runs a renderer; the rows are the rows the Rust emitter
prints, and the marker parity is pinned by
``tests/test_render_rust.py::test_the_abi_marker_matches_the_rust_source``.
"""
from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from gpuwm import render, rustwx


def _rows(*records: tuple[str, str, str, str, str]) -> str:
    lines = ["PRODUCT\t" + "\t".join(record) for record in records]
    lines.append(f"CATALOG total={len(records)}")
    return "\n".join(lines)


#: One row per windowed outcome the Rust catalog can emit, beside the two
#: non-windowed statuses a named slug can come back with.
_EVERY_OUTCOME = _rows(
    ("2m_temperature", "direct", "renderable",
     "2m Temperature [fill: temperature_2m_agl]", "renderable"),
    ("10m_wind_gusts", "direct", "missing-fields",
     "not stored: wind_gust_10m_agl", "missing-fields"),
    ("smoke_column", "derived", "blocked",
     "no smoke tracer is carried", "recipe-blocked"),
    ("qpf_1h", "windowed", "excluded",
     "exact-time ordinal axis; fixed-hour windows are undefined on it",
     "windowed-ordinal-axis"),
    ("qpf_total", "windowed", "excluded",
     "windowed accumulations need more than one stored whole-hour frame",
     "windowed-needs-whole-hour-frames"),
    ("qpf_6h", "windowed", "blocked",
     "6-h QPF requires forecast hour >= 6", "windowed-blocked"),
    ("uh_2to5km_1h_max", "windowed", "excluded",
     "windowed compute unavailable: run wrf/local_x uses an exact-time "
     "ordinal axis; production batch/windowed rendering is disabled until "
     "render requests carry exact lead and valid times",
     "windowed-compute-unavailable"),
)


def _stub(monkeypatch, stdout=_EVERY_OUTCOME, returncode=0, stderr=""):
    class Result:
        def __init__(self):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())


def _ask(tmp_path, products, *, paths=None, section=None):
    return render._available_window_request(
        Path("rw_wrfbatch"), tmp_path / "wrfout_d04_1974-04-04_01_00_00",
        products, tmp_path, heavy=False, paths=paths, section=section)


def test_every_windowed_outcome_is_dropped_not_forwarded(tmp_path, monkeypatch):
    """All five, including the two the prose match never caught."""

    _stub(monkeypatch)
    requested = ("2m_temperature,qpf_1h,qpf_total,qpf_6h,uh_2to5km_1h_max")
    available, skipped = _ask(tmp_path, requested)
    assert available == "2m_temperature"
    assert {slug for slug, _reason in skipped} == {
        "qpf_1h", "qpf_total", "qpf_6h", "uh_2to5km_1h_max"}


def test_a_blocked_row_is_dropped(tmp_path, monkeypatch):
    """``blocked`` is not ``excluded``; the old match required ``excluded``."""

    _stub(monkeypatch)
    available, skipped = _ask(tmp_path, "2m_temperature,qpf_6h")
    assert available == "2m_temperature"
    assert [slug for slug, _reason in skipped] == ["qpf_6h"]


def test_a_run_time_composed_reason_is_dropped(tmp_path, monkeypatch):
    """Its sentence names a store and can never be in a frozen set."""

    _stub(monkeypatch)
    available, skipped = _ask(tmp_path, "2m_temperature,uh_2to5km_1h_max")
    assert available == "2m_temperature"
    assert "exact-time ordinal axis" in skipped[0][1]


def test_a_non_windowed_refusal_is_dropped_too(tmp_path, monkeypatch):
    """A missing-fields row failed nothing here, but was never the door's
    to forward either: the catalog has already said no."""

    _stub(monkeypatch)
    available, skipped = _ask(tmp_path, "2m_temperature,10m_wind_gusts,smoke_column")
    assert available == "2m_temperature"
    assert {slug for slug, _reason in skipped} == {"10m_wind_gusts", "smoke_column"}


def test_the_reason_is_the_engines_own_words(tmp_path, monkeypatch):
    """Verbatim, because a paraphrase leaves the reader guessing which
    window or which field was missing."""

    _stub(monkeypatch)
    _available, skipped = _ask(tmp_path, "qpf_6h")
    assert skipped[0][1].endswith("6-h QPF requires forecast hour >= 6")


def test_a_renderable_request_is_passed_through_unchanged(tmp_path, monkeypatch):
    """No skip list, and the spec is the caller's own string."""

    _stub(monkeypatch)
    available, skipped = _ask(tmp_path, "2m_temperature")
    assert (available, skipped) == ("2m_temperature", [])


@pytest.mark.parametrize("spec", ["all", "heavy", "direct,derived"])
def test_a_group_request_is_never_second_guessed(tmp_path, monkeypatch, spec):
    """The engine expands a group itself and leaves out what it cannot
    draw, so a group token carries no promise to check -- and asking
    would cost an availability import for nothing."""

    def refuse(*_args, **_kwargs):
        raise AssertionError("a group request must not ask for a catalog")

    monkeypatch.setattr(subprocess, "run", refuse)
    assert _ask(tmp_path, spec) == (spec, [])


def test_a_named_slug_beside_a_group_is_still_checked(tmp_path, monkeypatch):
    """``all`` expands natively; ``qpf_1h`` spelled out is a promise, and
    forwarding it is what fails the invocation."""

    _stub(monkeypatch)
    available, skipped = _ask(tmp_path, "all,qpf_1h")
    assert available == "all"
    assert [slug for slug, _reason in skipped] == ["qpf_1h"]


def test_a_section_term_passes_through_when_a_line_was_composed(
        tmp_path, monkeypatch):
    """``xsec:`` names no catalog row; the section lane owns it.

    RE-PINNED with the line the section lane needs.  The store listing
    still says nothing about this term and still must not try; what
    decides it is whether the invocation composes a section line.  With
    one, it reaches the renderer untouched, as it always did.  Without
    one, the renderer refuses the whole invocation before it draws
    anything, so the door drops the term instead
    (``tests/test_render_storeless_families.py``).
    """

    _stub(monkeypatch)
    available, _skipped = _ask(tmp_path, "2m_temperature,xsec:QICE",
                               section="40,-100,41,-99")
    assert available == "2m_temperature,xsec:QICE"


def test_a_section_term_with_no_line_is_dropped_not_forwarded(
        tmp_path, monkeypatch):
    """The reading that retired the guard above.

    ``rw-wrfbatch/src/main.rs`` answers an ``xsec:`` term with no
    ``--section`` by returning an error before the store render starts,
    so forwarding it costs ``2m_temperature`` its pictures too.
    """

    _stub(monkeypatch)
    available, skipped = _ask(tmp_path, "2m_temperature,xsec:QICE")
    assert available == "2m_temperature"
    assert [slug for slug, _reason in skipped] == ["xsec:QICE"]


def test_an_unreadable_catalog_keeps_the_request_and_says_so(
        tmp_path, monkeypatch, capsys):
    """The one arm on which a refused slug still reaches the renderer, so
    it is the one arm that has to be audible."""

    _stub(monkeypatch, stdout="", returncode=2, stderr="store is unreadable")
    available, skipped = _ask(tmp_path, "2m_temperature,qpf_1h")
    assert (available, skipped) == ("2m_temperature,qpf_1h", [])
    assert "product availability could not be read" in capsys.readouterr().err


def test_the_series_route_asks_about_the_whole_series(tmp_path, monkeypatch):
    """A window is a property of the STORE, so the verdict has to be the
    store's -- asking about one frame would answer a different question."""

    seen: list[int] = []

    class Result:
        returncode = 0
        stdout = _EVERY_OUTCOME
        stderr = ""

    def record(command, *_args, **_kwargs):
        seen.append(sum(1 for token in command if "wrfout_d04" in str(token)))
        return Result()

    monkeypatch.setattr(subprocess, "run", record)
    series = [tmp_path / f"wrfout_d04_1974-04-04_0{index}_00_00"
              for index in range(1, 4)]
    _ask(tmp_path, "2m_temperature,qpf_1h", paths=series)
    assert seen == [3]


def test_the_group_keywords_are_the_engines_own_set():
    """Spelled once, beside the grammar it belongs to."""

    assert "all" in rustwx.GROUP_KEYWORDS
    assert rustwx._GROUP_KEYWORDS is rustwx.GROUP_KEYWORDS


# -- the generic family: the one place "no row" is proof, not silence --
#
# Measured on the shipped 2.7.5 wheel: the shipped "snow" preset names
# ``var:SNOW`` and ``var:SNOWH`` in WRFOUT spelling while the generic
# catalog is spelled in the store's selector vocabulary, so neither has
# a row.  Forwarded, the renderer answers ``var:SNOWH stored 2-D
# variable "SNOWH" does not exist`` and the whole 13-frame series exits
# 1 with 143 of its pictures thrown away.

_WITH_GENERICS = _rows(
    ("2m_temperature", "direct", "renderable",
     "2m Temperature [fill: temperature_2m_agl]", "renderable"),
    ("var:snow_depth", "generic", "renderable",
     "stored 2-D variable 'snow_depth' [m]", "renderable"))

#: A DEDUPED stored variable, on stderr because that is the stream the
#: renderer reports it on and therefore the one this is read from.
_DEDUPED = "GENERIC_EXCLUDED	orography	already rendered by a named product"


def test_an_absent_generic_variable_is_dropped(tmp_path, monkeypatch):
    _stub(monkeypatch, stdout=_WITH_GENERICS, stderr=_DEDUPED)
    available, skipped = _ask(tmp_path, "2m_temperature,var:SNOWH,var:SNOW")
    assert available == "2m_temperature"
    assert {slug for slug, _reason in skipped} == {"var:SNOWH", "var:SNOW"}


def test_the_drop_names_the_variable_and_the_way_out(tmp_path, monkeypatch):
    _stub(monkeypatch, stdout=_WITH_GENERICS, stderr=_DEDUPED)
    _available, skipped = _ask(tmp_path, "var:SNOWH")
    reason = skipped[0][1]
    assert "'SNOWH'" in reason
    assert "--list-products" in reason


def test_a_present_generic_variable_is_kept(tmp_path, monkeypatch):
    _stub(monkeypatch, stdout=_WITH_GENERICS, stderr=_DEDUPED)
    available, skipped = _ask(tmp_path, "var:snow_depth")
    assert (available, skipped) == ("var:snow_depth", [])


def test_a_deduped_generic_variable_is_kept(tmp_path, monkeypatch):
    """The store HAS it; the catalog dropped the duplicate row because a
    named product already draws that grid.  Silence there is not
    absence, and dropping on it would refuse a spelling this build
    accepts."""

    _stub(monkeypatch, stdout=_WITH_GENERICS, stderr=_DEDUPED)
    available, skipped = _ask(tmp_path, "var:orography")
    assert (available, skipped) == ("var:orography", [])


def test_the_deduped_line_becomes_a_renderable_row(tmp_path, monkeypatch):
    _stub(monkeypatch, stdout=_WITH_GENERICS, stderr=_DEDUPED)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d02"], store_root=tmp_path)
    folded = {row[0]: (row[2], rustwx.catalog_code(row)) for row in rows}
    assert folded["var:orography"] == ("renderable", "generic-deduped")


def test_an_empty_request_is_not_taken_for_a_group_keyword(tmp_path, monkeypatch):
    """``all()`` of nothing is True.

    The group-keyword short circuit returns the request untouched, so
    an empty or comma-only ``--products`` would have come back
    unchecked: an empty string draws nothing and says nothing about
    why, and a comma-only spelling would be handed to the renderer.
    Both come back as an empty spec instead, which is the caller's
    signal to refuse before launching.
    """

    _stub(monkeypatch, stdout=_WITH_GENERICS, stderr=_DEDUPED)
    for spelling in ("", ",", ",  ,"):
        assert _ask(tmp_path, spelling) == ("", []), spelling


_NAME_REFUSED = (
    "GENERIC_EXCLUDED	bad name	name is not request-safe: "
    "a product name cannot contain whitespace")


def test_the_not_request_safe_generic_row_is_not_folded_in(tmp_path, monkeypatch):
    """The emitter prints GENERIC_EXCLUDED for two OPPOSITE outcomes.

    One is the deduped variable above, which the store carries and a
    named product already draws.  The other is a stored variable whose
    NAME no request can carry -- commas, control characters, edge
    whitespace -- printed escape_debug'd.  Folding that one in would
    have ``gpuwm render --list-products`` print it as renderable,
    which is the opposite of what the row says.  No real request can
    match such a name, so the skip refuses nothing.
    """

    _stub(monkeypatch, stdout=_WITH_GENERICS,
          stderr="\n".join([_DEDUPED, _NAME_REFUSED]))
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d02"], store_root=tmp_path)
    folded = {row[0] for row in rows}
    assert "var:orography" in folded
    assert "var:bad name" not in folded


@pytest.mark.parametrize("token", ["mesh:cell_area", "meshdiff:theta"])
def test_the_other_generic_families_are_not_decided_by_the_listing(
        tmp_path, monkeypatch, token):
    """A mesh product is not a store product; a store listing cannot
    decide it and must not try.

    RE-PINNED.  The listing is still not the authority here, and this
    still holds that no catalog row is consulted for these tokens.  What
    changed is the answer for a door that passes no ``--mesh-grid``:
    measured on ``rw-wrfbatch/src/main.rs``, such a term is a usage
    error at argument validation, or an error at the entry to the batch
    render when store products stand beside it, and both come back
    before a picture is drawn.  Forwarding it took ``2m_temperature``
    down with it, so the term is dropped with the engine's own reason
    and the rest of the request is drawn.
    """

    _stub(monkeypatch, stdout=_WITH_GENERICS, stderr=_DEDUPED)
    available, skipped = _ask(tmp_path, f"2m_temperature,{token}")
    assert available == "2m_temperature"
    assert [slug for slug, _reason in skipped] == [token]
    assert "--mesh-grid" in skipped[0][1]

"""The catalog row contract: a machine code and a verdict.

Every case here stubs the renderer's output rather than running one, which
is what lets them run on a box with no build (``find_renderer()`` is None
here and every ``@needs_renderer`` case in ``tests/test_render_rust.py``
skips).  The rows they feed are the rows the Rust emitter prints; the
parity between the two is pinned separately by
``tests/test_render_rust.py::test_the_abi_marker_matches_the_rust_source``.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest

from gpuwm import rustwx


_LISTING = "\n".join([
    "PRODUCT\t2m_temperature\tdirect\trenderable\t2m Temperature [fill: temperature_2m_agl]\trenderable",
    "PRODUCT\tcomposite_reflectivity\tdirect\tmissing-fields\tnot stored: REFL_10CM\tmissing-fields",
    "PRODUCT\tsmoke_column\tderived\tblocked\tno smoke tracer is carried\trecipe-blocked",
    "PRODUCT\tqpf_1h\twindowed\texcluded\texact-time ordinal axis; fixed-hour windows are undefined on it\twindowed-ordinal-axis",
    "CATALOG total=4 blocked=1 excluded=1 missing-fields=1 renderable=1",
])

#: The same listing from a build whose generic catalog enumerated the
#: store.  One ``var:`` row is all it takes to make "no row" a reading
#: of the store rather than a silence.
_LISTING_WITH_GENERICS = "\n".join([
    _LISTING.rsplit("\n", 1)[0],
    "PRODUCT\tvar:sst\tgeneric\trenderable\tstored 2-D variable 'SST' [K]\trenderable",
    "CATALOG total=5 blocked=1 excluded=1 missing-fields=1 renderable=2",
])


def _stub_listing(monkeypatch, stdout=_LISTING, returncode=0):
    class Result:
        def __init__(self):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())


def test_a_catalog_row_carries_a_machine_code(tmp_path, monkeypatch):
    _stub_listing(monkeypatch)
    rows, summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    codes = {row[0]: rustwx.catalog_code(row) for row in rows}
    assert codes["qpf_1h"] == "windowed-ordinal-axis"
    assert codes["qpf_1h"] in rustwx.WINDOW_AXIS_CODES
    assert summary.startswith("total=4")


def test_the_window_skip_survives_a_reworded_reason(tmp_path, monkeypatch):
    """The prose is what a reader sees; the STATUS is what a door decides on.

    The accessor that matched these rows by their English text is gone
    with the defect it could not catch: the catalog has five windowed
    outcomes and two of them were spelled here.  The door reads the
    row's status through :func:`catalog_verdict`, which is why a
    reworded reason changes nothing but the sentence the reader gets.
    """
    reworded = _LISTING.replace(
        "exact-time ordinal axis; fixed-hour windows are undefined on it",
        "this run's frames sit on an exact-time axis, so a fixed-hour window "
        "has no definition")
    _stub_listing(monkeypatch, reworded)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    spec, excluded = rustwx.catalog_verdict(rows, "qpf_1h,2m_temperature")
    assert spec == "2m_temperature"
    assert [slug for slug, _reason in excluded] == ["qpf_1h"]
    assert "fixed-hour window" in dict(excluded)["qpf_1h"]


def test_a_build_with_no_code_column_is_still_read(tmp_path, monkeypatch):
    """A renderer built before the code column still gets its skip."""
    old = "\n".join(line.rsplit("\t", 1)[0] if line.startswith("PRODUCT") else line
                    for line in _LISTING.splitlines())
    _stub_listing(monkeypatch, old)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    assert rustwx.catalog_code(rows[0]) == ""
    spec, excluded = rustwx.catalog_verdict(rows, "qpf_1h")
    assert spec == "" and [slug for slug, _ in excluded] == ["qpf_1h"]


def test_the_four_tuple_listing_is_unchanged(tmp_path, monkeypatch):
    """Its two consumers unpack four fields and keep working."""
    _stub_listing(monkeypatch)
    rows, _summary = rustwx.list_products(
        Path("rw_wrfbatch"), tmp_path / "wrfout_d01", store_root=tmp_path)
    assert all(len(row) == 4 for row in rows)
    slug, kind, status, detail = rows[0]
    assert (slug, kind, status) == ("2m_temperature", "direct", "renderable")
    assert "temperature_2m_agl" in detail


def test_the_verdict_keeps_the_drawable_and_names_every_other_reason(tmp_path, monkeypatch):
    _stub_listing(monkeypatch)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    spec, excluded = rustwx.catalog_verdict(
        rows, "composite_reflectivity,2m_temperature,smoke_column,qpf_1h")
    assert spec == "2m_temperature"
    assert [slug for slug, _reason in excluded] == [
        "composite_reflectivity", "smoke_column", "qpf_1h"]
    assert dict(excluded)["composite_reflectivity"] == "not stored: REFL_10CM"
    assert dict(excluded)["smoke_column"] == "no smoke tracer is carried"


def test_a_request_nothing_can_draw_comes_back_empty(tmp_path, monkeypatch):
    """So the caller refuses at the door instead of launching an empty render."""
    _stub_listing(monkeypatch)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    spec, excluded = rustwx.catalog_verdict(rows, "composite_reflectivity,qpf_1h")
    assert spec == ""
    assert len(excluded) == 2


def test_a_group_keyword_or_undecidable_family_is_never_eaten(tmp_path, monkeypatch):
    """A group keyword, a section and a mesh term all pass through: the
    engine expands the first and the other two are not store products, so
    a store listing cannot decide them and must not try."""

    _stub_listing(monkeypatch)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    spec, excluded = rustwx.catalog_verdict(rows, "all,xsec:QICE,mesh:cell_area")
    assert spec == "all,xsec:QICE,mesh:cell_area" and excluded == []


def test_an_absent_var_family_term_IS_eaten(tmp_path, monkeypatch):
    """The ``var:`` family is the one exception, and it is a measurement
    rather than a preference.

    This case asserted that ``var:wrf_olr`` passed through untouched,
    on the rule that the engine is the authority on an unknown slug.
    Measured on the shipped 2.7.5 wheel, forwarding an absent one is
    never right: the generic catalog enumerates the store's 2-D
    variables, so a missing row is proof, and the renderer answers
    ``var:SNOWH stored 2-D variable "SNOWH" does not exist`` and exits
    nonzero -- taking a 13-frame series' other 143 pictures with it.
    A present-but-DEDUPED variable still passes through
    (``tests/test_render_series_catalog_verdict.py``), which is what
    keeps this from refusing a spelling the build accepts.
    """

    _stub_listing(monkeypatch, _LISTING_WITH_GENERICS)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    spec, excluded = rustwx.catalog_verdict(rows, "2m_temperature,var:wrf_olr")
    assert spec == "2m_temperature"
    assert dict(excluded)["var:wrf_olr"].startswith(
        "no stored 2-D variable 'wrf_olr'")


def test_a_listing_with_no_generic_rows_proves_nothing_absent(
        tmp_path, monkeypatch):
    """The measurement above rests on the enumeration having RUN.

    "there is no row for this variable" is proof of absence only because
    the generic catalog lists every stored 2-D variable.  A listing that
    carries no generic row at all has not made that statement -- a build
    without the generic enumeration is one way to get one -- and reading
    its silence as proof would drop every ``var:`` request a user made,
    by guess, with the engine never asked.  So the term is forwarded and
    the renderer decides, which is what :func:`catalog_rows` already
    does for a build whose rows carry no machine code.
    """

    _stub_listing(monkeypatch)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    assert not any(row[1] == "generic" for row in rows)
    spec, excluded = rustwx.catalog_verdict(rows, "2m_temperature,var:wrf_olr")
    assert spec == "2m_temperature,var:wrf_olr"
    assert excluded == []


def test_a_deduped_generic_row_is_enough_to_prove_the_enumeration_ran(
        tmp_path, monkeypatch):
    """A deduped variable is reported on stderr rather than as a row.

    It is still the generic catalog speaking, so it still proves the
    store was enumerated, and an absent ``var:`` term is still dropped.
    """

    class Result:
        returncode = 0
        stdout = _LISTING
        stderr = ("GENERIC_EXCLUDED\torography\talready drawn by a "
                  "named product")

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    spec, excluded = rustwx.catalog_verdict(rows, "2m_temperature,var:wrf_olr")
    assert spec == "2m_temperature"
    assert [slug for slug, _reason in excluded] == ["var:wrf_olr"]


def test_an_opt_in_family_row_parses_and_keeps_the_status_vocabulary(tmp_path, monkeypatch):
    """An ensemble family is listed with a field reason, not a policy word."""
    listing = "\n".join([
        "PRODUCT\tens_spread_demo\tdirect\tmissing-fields\tnot stored: "
        "height_500hpa; ensemble/probabilistic family: never included by "
        "'all', name the slug explicitly\topt-in-ensemble-family",
        "CATALOG total=1 missing-fields=1",
    ])
    _stub_listing(monkeypatch, listing)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    assert len(rows) == 1
    assert rows[0][2] in {"renderable", "missing-fields", "blocked", "excluded"}
    assert rustwx.catalog_code(rows[0]) == "opt-in-ensemble-family"


# --------------------------------------------------- the section grammar

def test_the_generic_families_are_read_out_of_the_pinned_marker():
    assert rustwx.GENERIC_FAMILIES == ("var:", "xsec:", "mesh:", "meshdiff:")
    assert rustwx.SECTION_PREFIX in rustwx.GENERIC_FAMILIES
    for family in rustwx.GENERIC_FAMILIES:
        assert f"\t{family}\t" in rustwx.RENDERER_ABI_MARKER


@pytest.mark.parametrize("spec,needed", [
    ("2m_temperature,xsec:wa=1,2,5@5", True),
    ("xsec:QICE", True),
    ("all,2m_temperature", False),
    ("all", False),
    ("2m_temperature,composite_reflectivity", False),
])
def test_the_level_list_rule_decides_which_terms_need_a_line(spec, needed):
    """The engine's own level-list rule, read on the live path.

    It used to be pinned through a predicate no door called any more.
    Asked of :func:`gpuwm.rustwx.drop_storeless_terms` instead, the same
    rule is measured where it is used: a term that needs a line is
    dropped by a door that composes none, and kept by one that does.
    """

    _spec, dropped = rustwx.drop_storeless_terms(spec)
    assert bool(dropped) is needed
    kept, none_dropped = rustwx.drop_storeless_terms(
        spec, section="39,-95,40,-94")
    assert (kept, none_dropped) == (spec, [])


def test_the_level_list_continuation_is_not_read_as_a_product():
    store, sections = rustwx.split_section_spec("2m_temperature,xsec:wa=1,2,5@5")
    assert store == "2m_temperature"
    assert sections == ["xsec:wa=1,2,5@5"]


def test_the_storeless_grammar_is_the_only_refusal_builder_left():
    """The two whole-request refusals are retired, not orphaned.

    They said the renderer "would refuse this render and every other
    product in it", and both doors now drop the term per product and
    draw the rest, so the breakage they named cannot be reached from
    either.  Their grammar lives on in
    :func:`gpuwm.rustwx.drop_storeless_terms`, which is where both doors
    ask; `tests/test_render_storeless_families.py` measures both.
    """

    for gone in ("mesh_spec_problem", "section_spec_problem",
                 "section_required"):
        assert not hasattr(rustwx, gone), gone
    assert "mesh:cell_area" in rustwx.drop_storeless_terms(
        "composite_reflectivity,mesh:cell_area")[1][0][0]


def test_the_downscale_door_does_not_price_products_against_a_plan(monkeypatch):
    """The plan is not the run.

    Measured on a real child with the shipped snow preset: the
    renderer build's FILELESS requirement pair called sixteen of its
    twenty-one products undrawable, 2m_temperature and
    500mb_height_winds among them, and that same run drew 143
    pictures of exactly those products.  The pair is gone from this
    module along with its last caller.

    The door still asks the renderer ONE question, its product
    vocabulary, so an unknown slug is refused by name; that question
    is stubbed here.  With it answered, nothing else is launched: no
    second probe for what the build's import plan writes.
    """

    from gpuwm import downscale, go_cli

    monkeypatch.setattr(go_cli, "unknown_render_products", lambda spec: [])
    monkeypatch.setattr(
        subprocess, "run",
        lambda *a, **k: pytest.fail("the door probed the renderer build"))
    downscale._admit_render_products(
        "2m_temperature,composite_reflectivity", dry_run=True)
    for gone in ("catalog_requirements", "undrawable",
                 "parse_catalog_requirements", "window_axis_unavailable"):
        assert not hasattr(rustwx, gone), gone


# ------------------------------------------- the engine's row grammar

def test_the_abi_marker_pins_the_row_grammar_the_engine_publishes():
    """The marker is the engine's own statement of its row shapes."""
    assert "\tdetail\tcode\t" in rustwx.RENDERER_ABI_MARKER
    assert "requirements-v1\tNEEDS\t" in rustwx.RENDERER_ABI_MARKER
    assert "\tPLANNED\tstore_field\t" in rustwx.RENDERER_ABI_MARKER
    source = (Path(rustwx.__file__).resolve().parents[1] / "tools" / "rustwx"
              / "crates" / "rw-wrfbatch" / "src" / "main.rs")
    match = re.search(r'const ABI_MARKER:\s*&str\s*=\s*"(.*?)";',
                      source.read_text(encoding="utf-8"), re.S)
    literal = re.sub(r"\\\n\s*", "", match.group(1)).replace("\\t", "\t")
    assert literal == rustwx.RENDERER_ABI_MARKER

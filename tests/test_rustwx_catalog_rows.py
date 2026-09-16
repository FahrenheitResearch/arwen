"""The catalog row contract: a machine code, a verdict, a fileless answer.

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
    """The prose is what a reader sees; the code is what a door decides on."""
    reworded = _LISTING.replace(
        "exact-time ordinal axis; fixed-hour windows are undefined on it",
        "this run's frames sit on an exact-time axis, so a fixed-hour window "
        "has no definition")
    _stub_listing(monkeypatch, reworded)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    unavailable = rustwx.window_axis_unavailable(rows, "qpf_1h,2m_temperature")
    assert set(unavailable) == {"qpf_1h"}
    assert "fixed-hour window" in unavailable["qpf_1h"]


def test_a_build_with_no_code_column_is_still_read(tmp_path, monkeypatch):
    """The prose match stays, for a renderer built before the code."""
    old = "\n".join(line.rsplit("\t", 1)[0] if line.startswith("PRODUCT") else line
                    for line in _LISTING.splitlines())
    _stub_listing(monkeypatch, old)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    assert rustwx.catalog_code(rows[0]) == ""
    assert set(rustwx.window_axis_unavailable(rows, "qpf_1h")) == {"qpf_1h"}


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


def test_a_group_keyword_or_generic_family_is_never_eaten(tmp_path, monkeypatch):
    _stub_listing(monkeypatch)
    rows, _summary = rustwx.catalog_rows(
        Path("rw_wrfbatch"), [tmp_path / "wrfout_d01"], store_root=tmp_path)
    spec, excluded = rustwx.catalog_verdict(rows, "all,var:wrf_olr,xsec:QICE")
    assert spec == "all,var:wrf_olr,xsec:QICE" and excluded == []


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
def test_section_required_reads_the_engines_own_level_list_rule(spec, needed):
    assert rustwx.section_required(spec) is needed


def test_the_level_list_continuation_is_not_read_as_a_product():
    store, sections = rustwx.split_section_spec("2m_temperature,xsec:wa=1,2,5@5")
    assert store == "2m_temperature"
    assert sections == ["xsec:wa=1,2,5@5"]


def test_a_section_request_with_a_line_is_admitted():
    assert rustwx.section_spec_problem("xsec:wa", section="39,-95,40,-94") is None
    assert rustwx.section_spec_problem("2m_temperature") is None


def test_a_section_request_without_a_line_names_the_breakage_and_the_way_out():
    problem = rustwx.section_spec_problem("xsec:wa")
    assert problem is not None
    assert "--section" in problem and "gpuwm render" in problem


def test_the_downscale_door_refuses_a_section_request_it_cannot_compose():
    from gpuwm.downscale import OfflineChildContractError, _admit_render_products

    with pytest.raises(OfflineChildContractError) as excinfo:
        _admit_render_products("xsec:wa", dry_run=True)
    assert "--section" in str(excinfo.value)
    assert "after the forecast" in str(excinfo.value)


# ------------------------------------------- the fileless requirement pair

_FILELESS = "\n".join([
    "group keywords: all, direct, derived, heavy, windowed",
    "  2m_temperature",
    "  10m_wind_gusts",
    "NEEDS\t2m_temperature\ttemperature_2m_agl",
    "NEEDS\t10m_wind_gusts\twind_gust_10m_agl",
    "PLANNED\ttemperature_2m_agl",
    "PLANNED\tmslp",
    "selectable_slugs=2",
])


def test_the_fileless_pair_answers_availability_before_a_wrfout_exists():
    answer = rustwx.parse_catalog_requirements(_FILELESS)
    assert answer.needs["10m_wind_gusts"] == ("wind_gust_10m_agl",)
    assert "temperature_2m_agl" in answer.planned
    missing = rustwx.undrawable(
        ["2m_temperature", "10m_wind_gusts"], requirements=answer)
    assert set(missing) == {"10m_wind_gusts"}
    assert "wind_gust_10m_agl" in missing["10m_wind_gusts"]
    assert answer.basis in missing["10m_wind_gusts"]


def test_a_slug_the_build_records_no_requirement_for_is_not_claimed():
    answer = rustwx.parse_catalog_requirements(_FILELESS)
    assert rustwx.undrawable(["a_slug_nothing_records"], requirements=answer) == {}


def test_no_resolver_reports_nothing_rather_than_refusing_everything():
    """Unmeasured is not impossible (lane rule 1)."""
    assert rustwx.undrawable(["10m_wind_gusts"], requirements=None) == {}


def test_a_build_too_old_to_answer_the_pair_answers_none(monkeypatch, tmp_path):
    binary = tmp_path / "rw_wrfbatch"
    binary.write_bytes(b"")

    class Result:
        returncode = 0
        stdout = "group keywords: all\n  2m_temperature\nselectable_slugs=1\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Result())
    assert rustwx.catalog_requirements(binary) is None


def test_the_abi_marker_pins_the_row_grammar_the_parser_reads():
    assert "\tdetail\tcode\t" in rustwx.RENDERER_ABI_MARKER
    assert "requirements-v1\tNEEDS\t" in rustwx.RENDERER_ABI_MARKER
    assert "\tPLANNED\tstore_field\t" in rustwx.RENDERER_ABI_MARKER
    source = (Path(rustwx.__file__).resolve().parents[1] / "tools" / "rustwx"
              / "crates" / "rw-wrfbatch" / "src" / "main.rs")
    match = re.search(r'const ABI_MARKER:\s*&str\s*=\s*"(.*?)";',
                      source.read_text(encoding="utf-8"), re.S)
    literal = re.sub(r"\\\n\s*", "", match.group(1)).replace("\\t", "\t")
    assert literal == rustwx.RENDERER_ABI_MARKER

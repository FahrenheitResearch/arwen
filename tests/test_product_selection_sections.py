"""A product list is read term by term, with each section's level list whole.

Breakage these prevent, each measured on a real 3 km frame with the Rust
renderer: ``all,xsec:cloud:QCLOUD/wa`` drew 67 maps and no section, without
a word; two sections with the same level list lost the second list's last
level, so its colour bar spanned 0 to 1 g/kg instead of the requested 0.01
to 0.1; and a level list closed by an overlay (``0.1/wa``) became a store
product the renderer refused, so the invocation drew nothing at all.
"""
from __future__ import annotations

import pytest

from gpuwm import render_receipts, rustwx
from gpuwm.render import parse_products_rust


@pytest.mark.parametrize("spec", [
    "all,xsec:QCLOUD/wa",
    "xsec:QCLOUD/wa,all",
    "all,xsec:cloud:QCLOUD=0.01,0.1/wa",
])
def test_all_keeps_an_explicit_section(spec):
    assert parse_products_rust(spec) == spec


def test_a_section_level_list_is_never_deduplicated_against_another():
    spec = "xsec:cloud:QCLOUD=0.01,0.1,xsec:rain:QRAIN=0.01,0.1"
    assert parse_products_rust(spec) == spec
    assert rustwx.product_spec_terms(spec) == [
        "xsec:cloud:QCLOUD=0.01,0.1", "xsec:rain:QRAIN=0.01,0.1"]


@pytest.mark.parametrize("section", [
    "xsec:QCLOUD=0.01,0.1/wa",
    "xsec:QCLOUD=1e-2,1e-1/wa=1,2,5@5",
    "xsec:wa=-2,-1,1,2@2/tk=270,280",
])
def test_a_level_list_can_close_with_an_overlay(section):
    assert rustwx.split_section_spec("2m_temperature," + section) == (
        "2m_temperature", [section])
    assert parse_products_rust("t2," + section) == "2m_temperature," + section
    kept, dropped = rustwx.drop_storeless_terms(
        "2m_temperature," + section + ",mesh:T", section="42,-71.2,42,-70.8")
    assert kept == "2m_temperature," + section
    assert [term for term, _reason in dropped] == ["mesh:T"]


def test_a_slug_after_a_level_list_is_still_a_product():
    assert rustwx.product_spec_terms("xsec:wa=1,2,t2,5x") == [
        "xsec:wa=1,2", "t2", "5x"]
    # Digit separators are Python's, not the engine's: never a level.
    assert rustwx.product_spec_terms("xsec:wa=1,1_000") == ["xsec:wa=1", "1_000"]


def test_the_receipt_counts_whole_section_products(tmp_path):
    spec = "2m_temperature,xsec:QCLOUD=0.01,0.1/wa"
    summary = render_receipts.publish_invocation(
        root=tmp_path, engine="rust", requested_spec=spec, written=[],
        failures=[], skipped=[], layout="nested")
    assert summary["requested_family_count"] == 2
    assert summary["requested_families"] == [
        "2m_temperature", "xsec:QCLOUD=0.01,0.1/wa"]


@pytest.mark.parametrize("spec,expected", [
    ("all", "all"),
    ("ALL", "all"),
    # The engine refuses a keyword beside a named product in its store
    # list, and `all` already draws every named product.
    ("all,t2", "all"),
    ("t2,all,refl", "all"),
    ("all,all", "all"),
    # What `all` does not draw keeps its place beside it.
    ("all,variables", "all,variables"),
    ("all,mesh:cell_area", "all,mesh:cell_area"),
])
def test_all_folds_only_what_it_already_draws(spec, expected):
    assert parse_products_rust(spec) == expected


def test_aliases_and_duplicate_products_keep_their_behaviour():
    assert parse_products_rust("refl,refl,t2") == (
        "composite_reflectivity,2m_temperature")
    with pytest.raises(ValueError, match="no products"):
        parse_products_rust(",")

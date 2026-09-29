"""The viewer's product bound is the node's own, and its default is the node's.

CPU-only: the renderer is a fixture and no meteorological file is decoded.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gpuwm import remote_processed_v2 as viewer, remote_worker as rw
from test_remote_artifacts import case, encoded
from test_remote_processed_v2 import NODE_DEFAULT_PRODUCTS, native, query


def test_a_selection_up_to_the_nodes_own_bound_is_accepted():
    slugs = [f"p{index}" for index in range(120)]
    first = viewer._selection(viewer.PROFILE, slugs)
    assert first["products"] == sorted(slugs)
    # The identity is stable, so a published entry keeps its name.
    assert first["selection_id"] == viewer._selection(viewer.PROFILE, list(reversed(slugs)))["selection_id"]
    assert viewer._products([f"p{index}" for index in range(viewer.NODE_PRODUCT_LIMIT)])


def test_a_selection_past_the_nodes_bound_is_refused_naming_that_bound():
    with pytest.raises(ValueError) as failure:
        viewer._products([f"p{index}" for index in range(viewer.NODE_PRODUCT_LIMIT + 1)])
    message = str(failure.value)
    assert str(viewer.NODE_PRODUCT_LIMIT + 1) in message
    assert str(viewer.NODE_PRODUCT_LIMIT) in message
    assert "refuse the whole frame" in message
    assert "empty selection" in message


def test_a_selection_that_cannot_fit_the_request_envelope_names_the_bound_it_measured():
    slugs = ["p" + "q" * 900 for _ in range(viewer.NODE_PRODUCT_LIMIT)]
    with pytest.raises(ValueError) as failure:
        viewer._products(slugs)
    message = str(failure.value)
    # The sentence names the bound this refusal actually measured against, and
    # the envelope that bound is carved out of, so both numbers reconcile.
    assert str(viewer.MAX_SELECTION_BYTES) in message
    assert str(rw.MAX_BYTES) in message
    assert str(len(viewer.ra._encoded(slugs))) in message


def test_the_renderers_own_colon_selectors_reach_the_node():
    named = viewer._products(["var:T2", "mesh:qv:colmax", "2m_temperature"])
    assert named == sorted(["var:T2", "mesh:qv:colmax", "2m_temperature"])
    # The selection identity covers them like any other named product.
    assert viewer._selection(viewer.PROFILE, ["var:T2"])["selection_id"] != viewer._selection(viewer.PROFILE, ["var:t2"])["selection_id"]


@pytest.mark.parametrize("profile", [viewer.PROFILE, viewer.SCIENCE_PROFILE])
def test_viewer_without_a_line_refuses_explicit_sections(profile):
    with pytest.raises(ValueError, match="cannot locate the slice"):
        viewer._selection(profile, ["2m_temperature", "xsec:wa"])


def test_viewer_catalog_excludes_sections_and_run_catalog_keeps_them(monkeypatch):
    viewer._CATALOG.clear()
    monkeypatch.setattr("gpuwm.runplan.render_catalog", lambda: {
        "products": [{"name": "2m_temperature"}, {"name": "xsec:wa"}]})
    maps = viewer.node_catalog()
    assert maps["products"] == ["2m_temperature"] and maps["count"] == 1
    assert not any(name.startswith("xsec:") for name in maps["selector_families"])
    forecast = viewer.node_catalog(include_sections=True)
    assert forecast["products"] == ["2m_temperature", "xsec:wa"] and forecast["count"] == 2
    assert any(name.startswith("xsec:") for name in forecast["selector_families"])
    assert viewer.node_catalog()["products"] == ["2m_temperature"]
    viewer._CATALOG.clear()


def test_the_product_bound_and_the_selector_families_come_from_the_nodes_catalog(monkeypatch):
    viewer._CATALOG.clear()
    monkeypatch.setattr("gpuwm.runplan.render_catalog", lambda: {
        "products": [{"name": f"slug_{index}"} for index in range(311)],
        "group_keywords": ["all", "direct"], "source": "the rust renderer's own --list-products"})
    catalog = viewer.node_catalog()
    assert catalog["count"] == 311 and catalog["products"][0] == "slug_0"
    assert catalog["source"] == "the rust renderer's own --list-products"
    with pytest.raises(ValueError) as failure:
        viewer._products([f"p{index}" for index in range(viewer.NODE_PRODUCT_LIMIT + 1)])
    # The reader is told how large this node's own catalog is, and which door
    # prints it, rather than only that their count was too high.
    assert "311 selectable" in str(failure.value)
    assert "list-products" in str(failure.value)
    viewer._CATALOG.clear()


def test_a_node_that_cannot_answer_its_catalog_states_that_and_refuses_nothing(monkeypatch):
    viewer._CATALOG.clear()
    def missing():
        raise RuntimeError("no usable renderer is staged on this node")
    monkeypatch.setattr("gpuwm.runplan.render_catalog", missing)
    catalog = viewer.node_catalog()
    assert catalog["products"] is None and "no usable renderer" in catalog["error"]
    assert viewer._products(["2m_temperature"]) == ["2m_temperature"]
    viewer._CATALOG.clear()


def test_an_invalid_slug_is_still_refused_naming_the_families_that_are_valid():
    with pytest.raises(ValueError) as failure:
        viewer._products(["2m_temperature", "Not A Slug"])
    assert "invalid product slug" in str(failure.value) and "var:" in str(failure.value)
    with pytest.raises(ValueError, match="at least one canonical product slug"):
        viewer._products([])
    with pytest.raises(ValueError, match="invalid product slug"):
        viewer._products(["v" * (viewer.MAX_SELECTOR_CHARS + 1)])


def test_an_empty_selection_is_a_stable_token_for_the_nodes_own_default():
    node_default = viewer._selection(viewer.PROFILE, [])
    assert node_default["products"] == []
    assert node_default["selection_id"] == viewer._selection(viewer.PROFILE, [])["selection_id"]
    # Naming no products at all is the same selection, not a second one that
    # transcribes a product list on this side of the connection.
    assert node_default["selection_id"] == viewer._selection(viewer.PROFILE, None)["selection_id"]
    assert node_default["selection_id"] != viewer._selection(viewer.SCIENCE_PROFILE, [])["selection_id"]


def test_both_doors_of_one_request_take_one_selection():
    request = {"schema": "gpuwm.remote.request.v1", "action": "processed-frame-v2",
               "workspace": "/w", "job": "job-1", "domain": 1, "sequence": 1,
               "profile": viewer.PROFILE, "products": []}
    # The catalog door and the member stream door read the same function, so
    # an entry path and the lease taken over it cannot come from two ids.
    assert viewer.selection_for(request) == viewer._selection(viewer.PROFILE, [])
    assert viewer.selection_for({**request, "products": ["2m_temperature"]}) == viewer._selection(
        viewer.PROFILE, ["2m_temperature"])


def test_the_empty_selection_reaches_the_node_through_the_desktop_door(native, monkeypatch):
    from gpuwm import remote_processed_cache_v2 as cache
    c = native.case
    sent = []
    monkeypatch.setattr("gpuwm.remote_cli._transport",
                        lambda _command, request, **_kwargs: sent.append(request) or {"ok": False, "error": {"message": "stop here"}})
    options = SimpleNamespace(workspace=str(c.tmp_path), job=c.record["id"], domain=1, sequence=1,
                              cache_root=str(c.tmp_path / "cache"), profile=viewer.PROFILE, products="")
    with pytest.raises(ValueError, match="stop here"):
        cache.sync(options, [], [])
    assert sent[0]["products"] == []


def test_a_jobs_own_render_selection_is_what_its_watchers_prepare():
    assert viewer.job_selection({"products": "2m_temperature,sbcape"})["products"] == ["2m_temperature", "sbcape"]
    assert viewer.job_selection({"products": None})["products"] == []
    # The renderer's group vocabulary is not a named product set; the node's
    # own default set is prepared and the basis says so.
    assert viewer.job_selection({"products": "all"})["products"] == []
    assert viewer.selection_basis(viewer.PROFILE, viewer.selectors("all")) == (
        "the node's own viewer profile default set, resolved on the node")


def test_a_recorded_selection_keeps_each_section_term_whole():
    """A section's level list is comma-separated too, and its pieces are no
    selectors: ``0.1/wa`` and ``5@5`` fail ``SELECTOR``, so splitting the
    recorded string on every comma refused a spelling the renderer draws."""
    spec = "composite_reflectivity,xsec:QCLOUD=0.01,0.1/wa,xsec:wa=1,2,5@5"
    assert viewer.selectors(spec) == [
        "composite_reflectivity", "xsec:QCLOUD=0.01,0.1/wa", "xsec:wa=1,2,5@5"]
    assert viewer.job_selection({"products": spec})["products"] == ["composite_reflectivity"]
    assert viewer.job_selection({"products": "xsec:wa"})["products"] == []


def test_an_empty_selection_sends_no_products_and_publishes_the_nodes_answer(native):
    c = native.case
    value = viewer.catalog(query(c, products=[]), c.tmp_path)
    viewer._work_job(c.tmp_path, c.record["id"])
    assert native.calls[0]["products"] == []
    published = viewer.catalog(query(c, products=[]), c.tmp_path)
    assert [row["slug"] for row in published["products"]] == list(NODE_DEFAULT_PRODUCTS)
    assert published["selection_products"] == []
    assert "node's own viewer profile default set" in published["selection_basis"]


def test_a_named_selection_is_published_from_the_nodes_returned_rows(native):
    c = native.case
    viewer.catalog(query(c, products=["2m_temperature"]), c.tmp_path)
    viewer._work_job(c.tmp_path, c.record["id"])
    assert native.calls[0]["products"] == ["2m_temperature"]
    published = viewer.catalog(query(c, products=["2m_temperature"]), c.tmp_path)
    assert [row["slug"] for row in published["products"]] == ["2m_temperature"]
    assert published["selection_basis"] == "the products this request named"


def test_the_first_request_states_that_it_has_no_recorded_basis_and_proceeds(native):
    c = native.case
    value = viewer.catalog(query(c, products=["2m_temperature"]), c.tmp_path)
    estimate = value["selection_estimate"]
    assert estimate["estimated_bytes"] is None and estimate["warn"] is False
    assert "no frame has been published for this job yet" in estimate["basis"]
    assert estimate["products"] == 1


def test_a_later_request_is_priced_from_this_jobs_own_published_frames(native):
    c = native.case
    viewer.catalog(query(c, products=["2m_temperature"]), c.tmp_path)
    viewer._work_job(c.tmp_path, c.record["id"])
    value = viewer.catalog(query(c, products=["2m_temperature"]), c.tmp_path)
    estimate = value["selection_estimate"]
    assert type(estimate["estimated_bytes"]) is int and estimate["estimated_bytes"] > 0
    assert estimate["basis"] == "the largest bytes per product any frame of this job has published"
    assert estimate["cache_budget_bytes"] == viewer.DEFAULT_CACHE_BYTES
    # Pricing states, it never refuses.
    assert estimate["warn"] is False


def test_the_measured_byte_budget_is_still_the_one_that_refuses(native):
    c = native.case
    viewer.catalog(query(c, products=["2m_temperature"]), c.tmp_path)
    with pytest.raises(viewer.Backpressure) as failure:
        viewer._prune(viewer._root(c.tmp_path), c.record["id"], 1024, incoming=4096)
    message = str(failure.value)
    assert "4096" in message and "1024" in message


def test_a_sections_only_run_gets_no_map_selection_and_a_note_not_the_nodes_default_set():
    """A run that asked only for sections read as an empty list, the node's default maps."""
    sections = viewer.job_selection({"products": "xsec:wa,xsec:QCLOUD=0.01,0.1/wa"})
    default = viewer.job_selection({"products": None})
    assert sections["selection_id"] != default["selection_id"]
    assert not viewer.has_map_products(sections) and viewer.has_map_products(default)
    assert sections["products"] == [] and sections["note"] == viewer.NO_MAP_PRODUCTS_NOTE
    assert "cross-section" in sections["note"] and "no map products" in sections["note"]


def test_a_run_that_also_names_sections_keeps_the_identity_of_its_map_terms():
    maps = viewer.job_selection({"products": "composite_reflectivity"})
    both = viewer.job_selection({"products": "composite_reflectivity,xsec:wa=1,2,5@5"})
    assert both == maps
    # The map-only identity is the one the viewer has always published under.
    assert maps["selection_id"] == viewer.ra._sha(viewer.ra._encoded(
        {"profile": viewer.PROFILE, "products": ["composite_reflectivity"]}))


def test_a_sections_only_run_prepares_no_default_maps_and_says_why(native):
    from gpuwm import remote_preparation_v2 as preparation
    c = native.case
    c.record["products"] = "xsec:wa"
    state = preparation.prepare(c.tmp_path, c.record["id"], start=False)
    assert state["state"] == "no_map_products" and state["added"] == 0 and state["queued"] == 0
    assert state["note"] == viewer.NO_MAP_PRODUCTS_NOTE and not state["done"]
    assert not viewer._queue(viewer._root(c.tmp_path), c.record["id"])
    c.status["state"] = "completed"
    assert preparation.prepare(c.tmp_path, c.record["id"], start=False)["done"]
    assert not native.calls

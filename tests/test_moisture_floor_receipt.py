"""The initialization moisture floors, carried into the prepared proof.

The floors were receipted in memory and announced on stderr, and stopped
there: ``proof.json`` said nothing, so a forecast whose initial vapour
field had been modified on the way in was indistinguishable afterwards
from one that had not.  These cells pin the two halves that matter -- a
floor that FIRED appears with its magnitude, and a floor that did NOT is
stated rather than left out -- for the parent and for a spawned child.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, fields as dataclass_fields
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from gpuwm.moisture_floor_receipt import (
    MOISTURE_FLOOR_BY_DOMAIN_KEY,
    MOISTURE_FLOOR_KEY,
    MOISTURE_FLOOR_SCHEMA,
    moisture_floor_block,
    moisture_floor_field_names,
    moisture_floor_proof_entries,
    moisture_floor_proof_entry,
)


#: What a test says when a block has to name why nothing was recorded.
_UNRECORDED = "the test stand-in holds no ingest receipt"


def _result(**floors):
    """A ``RealInitResult`` whose floor receipts are what the test says."""
    from gpuwm.ingest.real import RealInitResult

    array = np.zeros((2, 2), dtype=np.float64)
    state = SimpleNamespace(
        p=np.zeros((2, 2, 2)), phb=np.zeros((3, 2, 2)),
        php=np.zeros((3, 2, 2)), alt=np.zeros((2, 2, 2)),
        c3h=np.array([0.5, 1.0]), c4h=np.array([10.0, 0.0]),
        p_top=5000.0, total_mu=lambda: np.full((2, 2), 90000.0))
    return RealInitResult(
        state=state, coord=object(), base=object(),
        surface_pressure=array, surface_qv=array, dry_mass=array,
        dry_pressure=array, total_pressure=array, total_geopotential=array,
        total_specific_volume=array, integrated_moisture_pressure=array,
        hypsometric_opt=2, **floors)


def test_a_floor_that_did_not_fire_is_stated_rather_than_left_out():
    """THE ABSENCE AND THE ZERO ARE DIFFERENT FACTS.

    An absent key would read as "this bundle was prepared before the
    receipt existed", which is a claim about the release, not about the
    forecast, and one no reader of the bundle could check.
    """

    block = moisture_floor_block(_result(), when_unrecorded=_UNRECORDED)
    assert block["schema"] == MOISTURE_FLOOR_SCHEMA
    assert block["recorded"] is True
    assert block["fired"] is False
    assert block["floors"], "a result with floor fields reports them"
    for name, entry in block["floors"].items():
        assert entry == {"fired": False}, name


def test_a_fired_floor_carries_its_magnitude_into_the_block():
    receipt = {"cells": 7, "min_pre_floor_kg_kg": -3.1e-07,
               "floor_kg_kg": 1e-06}
    block = moisture_floor_block(_result(surface_moisture_floor=receipt),
                                 when_unrecorded=_UNRECORDED)
    assert block["fired"] is True
    assert block["floors"]["surface_moisture_floor"] == {
        "fired": True, "receipt": receipt}
    # And every other floor the result declares still says it did not.
    for name, entry in block["floors"].items():
        if name != "surface_moisture_floor":
            assert entry == {"fired": False}, name


def test_the_reported_floors_are_read_off_the_dataclass_not_a_list_here():
    """The next floor added to the ingest is carried with no edit here.

    The tree's second floor -- the prognostic-column one the use_sh_qv
    lane needs -- lands on ``RealInitResult`` as another
    ``*_moisture_floor`` field.  This cell builds a result type with
    exactly that shape and asserts the block grows to match, so the
    module cannot quietly report the one floor it was written against.
    """

    @dataclass
    class _TwoFloorResult:
        surface_moisture_floor: dict = field(default_factory=dict)
        prognostic_moisture_floor: dict = field(default_factory=dict)
        hydrometeor_initialization: dict = field(default_factory=dict)

    grown = _TwoFloorResult(prognostic_moisture_floor={"columns": 4})
    assert moisture_floor_field_names(grown) == (
        "prognostic_moisture_floor", "surface_moisture_floor")
    block = moisture_floor_block(grown, when_unrecorded=_UNRECORDED)
    assert block["fired"] is True
    assert block["floors"]["prognostic_moisture_floor"] == {
        "fired": True, "receipt": {"columns": 4}}
    assert block["floors"]["surface_moisture_floor"] == {"fired": False}
    # A receipt-shaped field that is not a floor stays out of the block.
    assert "hydrometeor_initialization" not in block["floors"]


def test_a_read_only_receipt_from_a_restored_cache_stays_serializable():
    """A ``MappingProxyType`` anywhere in the tree turns "write the proof"
    into a ``TypeError`` naming nothing."""
    import json

    proxied = MappingProxyType({"cells": 2, "by_level": MappingProxyType(
        {"1": 2})})
    block = moisture_floor_block(
        _result(surface_moisture_floor=proxied), when_unrecorded=_UNRECORDED)
    assert json.loads(json.dumps(block))["floors"][
        "surface_moisture_floor"]["receipt"] == {
            "cells": 2, "by_level": {"1": 2}}


def test_the_parent_and_a_spawned_child_each_get_their_own_answer():
    """One verdict for a tree would lose both facts.

    The child is built the way ``nest_init`` builds it -- through
    ``_updated_real_result`` after the terrain blend and base-state
    re-derivation -- so this also pins that the floors survive the nest
    boundary rather than coming back as the dataclass default.
    """

    import gpuwm.ingest.nest_init as ni

    parent = _result()
    child_source = _result(surface_moisture_floor={"cells": 3})
    child = ni._updated_real_result(child_source, base=object())

    entries = moisture_floor_proof_entries(
        (("d01", parent), ("d02", child)), when_unrecorded=_UNRECORDED)
    blocks = entries[MOISTURE_FLOOR_BY_DOMAIN_KEY]
    assert set(blocks) == {"d01", "d02"}
    assert blocks["d01"]["fired"] is False
    assert blocks["d01"]["floors"]["surface_moisture_floor"] == {
        "fired": False}
    assert blocks["d02"]["fired"] is True
    assert blocks["d02"]["floors"]["surface_moisture_floor"] == {
        "fired": True, "receipt": {"cells": 3}}


def test_the_single_domain_entry_uses_its_own_key_not_the_tree_one():
    """Same key, two shapes is how a consumer starts throwing on half a
    bundle library.  The names differ; both contain "moisture_floors"."""

    entry = moisture_floor_proof_entry(_result(), when_unrecorded=_UNRECORDED)
    assert set(entry) == {MOISTURE_FLOOR_KEY}
    assert MOISTURE_FLOOR_KEY != MOISTURE_FLOOR_BY_DOMAIN_KEY
    assert "moisture_floors" in MOISTURE_FLOOR_BY_DOMAIN_KEY


def test_a_result_with_no_floor_field_says_so_instead_of_saying_zero():
    """ABSENCE (3).  A result that records no floor is not a result whose
    floors did not fire, and writing ``fired: False`` for it would state
    something this writer does not know."""

    @dataclass
    class _NoFloors:
        hydrometeor_initialization: dict = field(default_factory=dict)

    block = moisture_floor_block(_NoFloors(), when_unrecorded="no ingest here")
    assert block == {"schema": MOISTURE_FLOOR_SCHEMA, "recorded": False,
                     "not_recorded_because": "no ingest here"}
    assert "fired" not in block
    # And a bare stand-in with no attributes at all takes the same branch
    # rather than raising inside a proof writer.
    assert moisture_floor_block(object(), when_unrecorded="stand-in")[
        "recorded"] is False


def test_the_unrecorded_sentence_has_no_default():
    """A caller that cannot state one has found a gap, not a formatting
    problem."""

    with pytest.raises(TypeError):
        moisture_floor_block(_result())


#: Every module that writes a ``proof.json`` from a live
#: ``RealInitResult``.  HRRR's portable bundle is deliberately not here:
#: ``publish_hrrr_prepared_bundle`` is fed from on-disk artifacts and
#: holds no initialization result, so carrying the floors there is a
#: separate change through ``tools/prepare_hrrr_wrf.py``.
_PROOF_WRITERS = (
    "gpuwm/era5_direct.py",
    "gpuwm/gfs_direct.py",
    "gpuwm/mapped_direct.py",
)


def _proof_literals(path: Path):
    """Every prepared-proof dict literal in one writer module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        status = [
            value for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "status"]
        if not status or not isinstance(status[0], ast.Constant):
            continue
        if status[0].value != "READY_NOT_YET_STOCK_WRF_GATED":
            continue
        schema = [
            value for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == "schema"]
        if schema:
            found.append(node)
    return found


def test_every_proof_writer_that_holds_a_result_carries_the_receipt():
    """THE DRIFT THIS GATE EXISTS TO PREVENT.

    The receipt is only auditable if it is actually in the document.  A
    proof literal that grows a sibling -- another source, another
    hierarchy variant -- and forgets this key is a bundle whose floors
    are once again a fact that lived in one process's stderr.  Read out
    of the writers' own syntax, so adding a proof document without the
    receipt fails here rather than in a user's audit.
    """

    root = Path(__file__).resolve().parent.parent
    for relative in _PROOF_WRITERS:
        literals = _proof_literals(root / relative)
        assert len(literals) == 2, (
            f"{relative} no longer publishes exactly two prepared proofs; "
            f"it publishes {len(literals)}")
        for literal in literals:
            text = ast.unparse(literal)
            assert "moisture_floor" in text, (
                f"a prepared proof in {relative} carries no moisture-floor "
                "receipt; a floored initialization written by it is not "
                "auditable after the run")


def test_the_mapped_reader_and_writer_agree_about_the_new_keys():
    """The mapped route's forecast runner refuses any proof whose exact
    top-level inventory it does not recognise, and accepts bundles
    prepared before this key existed."""

    from gpuwm import prepared_single_domain_forecast as runner

    assert MOISTURE_FLOOR_KEY in runner.MAPPED_DIRECT_PROOF_KEYS
    assert MOISTURE_FLOOR_BY_DOMAIN_KEY in runner.MAPPED_HIERARCHY_PROOF_KEYS
    assert runner.MAPPED_MOISTURE_FLOOR_KEYS == {
        MOISTURE_FLOOR_KEY, MOISTURE_FLOOR_BY_DOMAIN_KEY}


def test_the_hierarchy_export_result_carries_a_block_per_domain():
    """The child results live only inside the hierarchy export, so this is
    the only place a proof writer can reach them from."""

    from gpuwm.native_hierarchy import NativeHierarchyExportResult

    names = {f.name for f in dataclass_fields(NativeHierarchyExportResult)}
    assert "moisture_floor_receipts" in names
    block = moisture_floor_proof_entries(
        [("d01", _result())], when_unrecorded=_UNRECORDED)
    carried = NativeHierarchyExportResult(
        artifacts=object(), wrf_manifest={}, timings_seconds={},
        moisture_floor_receipts=block)
    assert dict(carried.moisture_floor_receipts) == block


def test_the_hierarchy_result_cannot_be_built_without_its_floor_block():
    """The field has no default, and that is the point.

    Every hierarchy proof writer spreads this mapping into its document
    unconditionally, so a result constructed without it writes a proof
    carrying no `moisture_floors_by_domain` key -- and an ABSENT key is
    exactly the state this receipt exists to prevent, because it reads as
    "prepared before the receipt existed" rather than as a fact about the
    forecast.  With an empty default that was a construction detail no
    refusal would ever catch.
    """

    from gpuwm.native_hierarchy import NativeHierarchyExportResult

    with pytest.raises(TypeError) as refusal:
        NativeHierarchyExportResult(
            artifacts=object(), wrf_manifest={}, timings_seconds={})
    assert "moisture_floor_receipts" in str(refusal.value)


# ---------------------------------------------------------------------------
# The front door.  `gpuwm go` and `gpuwm run` on a [case_data] config take
# the experiment route, which writes a certification capsule and no
# proof.json at all -- so the receipt above was reachable from `gpuwm prep`
# and the direct adapters and NOT from the door most runs go through.  A
# run whose initial vapour was modified on the way in looked exactly like
# one whose was not, in the only document that run produced.
# ---------------------------------------------------------------------------

def test_the_run_route_states_a_block_for_every_domain_it_ran():
    from gpuwm.runtime import _run_moisture_floor_receipts

    receipts = _run_moisture_floor_receipts({
        1: SimpleNamespace(initial_result=_result()),
        2: SimpleNamespace(initial_result=_result(
            surface_moisture_floor={"cells": 4, "min_qv": -1.1e-5})),
    })
    blocks = receipts[MOISTURE_FLOOR_BY_DOMAIN_KEY]
    assert sorted(blocks) == ["d01", "d02"]
    assert blocks["d01"]["recorded"] is True
    assert blocks["d01"]["fired"] is False
    assert blocks["d02"]["fired"] is True
    assert blocks["d02"]["floors"]["surface_moisture_floor"]["receipt"] == {
        "cells": 4, "min_qv": -1.1e-5}


def test_a_run_domain_with_no_floor_field_says_not_recorded_not_not_fired():
    """The store-backed and idealized routes hand a stand-in.  "I did not
    observe this" and "nothing happened" are different sentences."""

    from gpuwm.runtime import _run_moisture_floor_receipts

    blocks = _run_moisture_floor_receipts(
        {1: SimpleNamespace(initial_result=SimpleNamespace(state=object()))}
    )[MOISTURE_FLOOR_BY_DOMAIN_KEY]
    assert blocks["d01"]["recorded"] is False
    assert "fired" not in blocks["d01"]
    assert blocks["d01"]["not_recorded_because"]


def test_a_domain_whose_receipt_cannot_be_read_still_gets_a_block():
    """NON-FATAL, and still not absent.  This is assembled after the last
    model step; it must never turn a finished forecast into a crash, and
    a domain that silently vanished from the mapping would be the exact
    absence the receipt exists to prevent."""

    from gpuwm.runtime import _run_moisture_floor_receipts

    class _Hostile:
        @property
        def initial_result(self):
            raise RuntimeError("the prepared case was released")

    blocks = _run_moisture_floor_receipts(
        {1: _Hostile()})[MOISTURE_FLOOR_BY_DOMAIN_KEY]
    assert blocks["d01"]["recorded"] is False
    assert "the prepared case was released" in blocks["d01"][
        "not_recorded_because"]


def test_the_store_backed_route_carries_its_floor_names_across_the_frame():
    """The case-store route keeps only a metadata stand-in of the init
    result.  Three receipt names were spelled there by hand; a floor that
    fired was computed and then dropped at that boundary, so a
    store-backed run reported "not recorded" for a floor that had fired.
    """

    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "gpuwm" / "runtime.py").read_text(
        encoding="utf-8"))
    carried = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "SimpleNamespace"
        and "aerosol_initialization" in ast.unparse(node)]
    assert len(carried) == 1
    assert "moisture_floor_field_names" in ast.unparse(carried[0])


def test_both_front_door_capsule_sites_hand_over_their_prepared_cases():
    """One emitter builds the block, but each exit must pass what it
    holds.  A site that forgets writes a capsule with no floors at all,
    which is what shipped."""

    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "gpuwm" / "runtime.py").read_text(
        encoding="utf-8"))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_emit_front_door_capsule"]
    assert len(calls) == 2
    for call in calls:
        assert any(keyword.arg == "prepared_cases" for keyword in call.keywords), (
            "a front-door capsule exit does not hand over its prepared "
            "cases, so its capsule states nothing about the initialization's "
            "moisture floors")


def test_a_slots_result_is_read_from_what_it_exposes_not_from_a_dict():
    """A type using __slots__ has no instance __dict__.  Reading that
    attribute reported `recorded: false` for an object that was holding
    floor receipts the whole time -- which is the "not recorded" claim
    this module tells its callers never to guess at."""

    class _Slotted:
        __slots__ = ("surface_moisture_floor",)

        def __init__(self, receipt):
            self.surface_moisture_floor = receipt

    assert not hasattr(_Slotted({}), "__dict__")
    assert moisture_floor_field_names(_Slotted({})) == (
        "surface_moisture_floor",)
    block = moisture_floor_block(_Slotted({"cells": 2}),
                                 when_unrecorded=_UNRECORDED)
    assert block["recorded"] is True
    assert block["fired"] is True
    assert block["floors"]["surface_moisture_floor"]["receipt"] == {"cells": 2}


# ---------------------------------------------------------------------------
# THE DEFAULT RUN.  `gpuwm run` supervises unless --no-supervise is passed,
# and the supervised worker calls runtime.run_experiment -- which writes the
# capsule above -- and then emits its OWN capsule into the same directory
# under the same fixed name.  Measured on a completed 17-frame two-domain
# forecast: `gpuwm go` left a capsule carrying both domains' floors and
# `gpuwm run` on the identical config left one whose receipts were
# ["run_progress"] alone.  The receipt was recorded and then written over,
# so a bare default run still could not say whether its vapour was floored.
# ---------------------------------------------------------------------------

#: Emission sites whose capsule is the ONLY document that route writes
#: about its own initialization, so the floors have to be in it.
_CAPSULE_ONLY_SITES = frozenset({
    "runtime.run_experiment:single-domain",
    "runtime.run_experiment:domain-tree",
    "supervisor:success",
})
#: Emission sites that run from a prepared bundle, whose `proof.json`
#: carries `moisture_floors` / `moisture_floors_by_domain` already and
#: whose digest each of those routes pins before it runs a step.  Their
#: capsule is not the last word on the initialization; the three above
#: have no other word at all.
_PROOF_BACKED_SITES = frozenset({
    "prepared_single_domain_forecast",
    "prepared_domain_tree_forecast",
})


def _capsule_calls(module_name):
    """Every capsule emission in one emitter module, with the sites it
    names and the text a reader would have to look at to see what the
    call states.  One level of indirection is followed, because a
    receipts block assembled by a helper beside the call is still that
    call's own answer."""

    root = Path(__file__).resolve().parent.parent
    path = root / (module_name.replace(".", "/") + ".py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    helpers = {node.name: node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef)}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in {"emit_run_capsule",
                                     "_emit_front_door_capsule"}):
            continue
        sites = set()
        for keyword in node.keywords:
            if keyword.arg != "emission_site":
                continue
            for part in ast.walk(keyword.value):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    sites.add(part.value)
        if not sites:
            continue  # the common front-door wrapper forwards the site
        text = [ast.unparse(node)]
        for part in ast.walk(node):
            if (isinstance(part, ast.Call) and isinstance(part.func, ast.Name)
                    and part.func.id in helpers):
                text.append(ast.unparse(helpers[part.func.id]))
        yield sites, " ".join(text)


def test_every_capsule_only_emission_site_states_its_moisture_floors():
    """The guard the last one was too narrow to be.

    Asserting two `_emit_front_door_capsule` calls inside gpuwm/runtime.py
    is blind to the site that actually ships: the supervisor's success
    capsule, written last, into the same folder, under the same name.
    This asks the question of every declared emission site instead, so a
    sixth site has to be classified before it can pass.
    """

    from gpuwm.certify.capsule import EMISSION_SITES

    assert _CAPSULE_ONLY_SITES | _PROOF_BACKED_SITES == set(EMISSION_SITES)
    seen = set()
    for module_name in ("gpuwm.runtime", "gpuwm.supervisor",
                        "gpuwm.prepared_single_domain_forecast",
                        "gpuwm.prepared_domain_tree_forecast"):
        for sites, text in _capsule_calls(module_name):
            seen |= sites
            if not sites & _CAPSULE_ONLY_SITES:
                continue
            assert ("moisture_floor" in text or "prepared_cases" in text), (
                f"{sorted(sites)} writes the only document its route "
                "produces about the initialization and states nothing "
                "about the moisture floors")
    assert seen == set(EMISSION_SITES), seen


def test_the_supervised_capsule_carries_what_the_run_route_recorded():
    """The replacing capsule may not say less than the one it replaced."""

    from gpuwm import runtime, supervisor

    fragment = runtime._run_moisture_floor_receipts(
        {1: SimpleNamespace(initial_result=_result(
            surface_moisture_floor={"cells": 4, "min_qv": -1.1e-5}))})
    receipts = supervisor._success_receipts(
        Path("outdir"), SimpleNamespace(moisture_floor_receipts=fragment))
    assert "run_progress" in receipts
    assert receipts[MOISTURE_FLOOR_BY_DOMAIN_KEY]["d01"]["fired"] is True


def test_a_summary_with_no_fragment_still_leaves_the_supervisor_capsule_whole():
    """A route that emitted no front-door capsule hands back nothing, and
    that is an absent key rather than a crash or an empty claim."""

    from gpuwm import supervisor

    receipts = supervisor._success_receipts(
        Path("outdir"), SimpleNamespace(moisture_floor_receipts=None))
    assert list(receipts) == ["run_progress"]


def test_each_run_route_hands_its_capsule_fragment_back_on_the_summary():
    """The carry itself.  The supervisor can only restate what it is
    given, so a run route that writes the fragment into its capsule and
    not onto its summary loses it at the process boundary."""

    from gpuwm.runtime import ExperimentRunSummary

    assert "moisture_floor_receipts" in {
        item.name for item in dataclass_fields(ExperimentRunSummary)}
    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / "gpuwm" / "runtime.py").read_text(
        encoding="utf-8"))
    owners = [node for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)
              and any(isinstance(part, ast.Call)
                      and isinstance(part.func, ast.Name)
                      and part.func.id == "_emit_front_door_capsule"
                      for part in ast.walk(node))]
    assert {owner.name for owner in owners} == {
        "run_experiment", "_run_built_experiment"}
    for owner in owners:
        assert "moisture_floor_receipts=" in ast.unparse(owner), (
            f"{owner.name} writes a front-door capsule and returns a "
            "summary that does not carry the floors it stated")

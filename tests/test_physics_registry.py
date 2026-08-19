from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pytest

from gpuwm.physics_registry import (
    MORRISON_TEMPLATE_ID,
    PLAN_SCHEMA,
    REGISTRY_SCHEMA,
    THOMPSON_TEMPLATE_ID,
    VALIDATION_SCHEMA,
    WSM6_TEMPLATE_ID,
    canonical_json,
    canonical_sha256,
    physics_registry,
    registry_sha256,
    validate_physics_plan,
)
from gpuwm.physics_compat import (
    KESSLER_PROFILE_ID,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    identify_single_domain_profile,
    single_domain_runtime_switches,
)
from gpuwm.source_cli import EXIT_CONFIG, main
from tools.hrrr_single_domain_benchmark import (
    runner_capabilities as hrrr_runner_capabilities,
)
from tools.prepared_domain_tree_forecast import (
    runner_capabilities as tree_runner_capabilities,
)
from tools.prepared_single_domain_forecast import (
    runner_capabilities as prepared_single_runner_capabilities,
)


ROOT = Path(__file__).parents[1]
REGISTRY_PATH = ROOT / "gpuwm" / "physics_registry_v2.json"


def _mixed_plan() -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": "mixed-thompson-nssl-proof-v1",
        "registry_sha256": registry_sha256(),
        "context": {
            "source_id": "hrrr",
            "runner_id": "tools.prepared_domain_tree_forecast",
            "topology_id": "one-way-nested-v1",
        },
        "domains": [
            {"domain_id": "d01", "template_id": THOMPSON_TEMPLATE_ID},
            {
                "domain_id": "d02",
                "template_id": THOMPSON_TEMPLATE_ID,
                "components": {"microphysics": "nssl2-mp18"},
                "parameters": {
                    "epssm": 0.1,
                    "nest_microphysics_transition": (
                        "mp8-to-mp18-mass-diagnosed-v1"
                    ),
                },
            },
        ],
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }


def _source_offering(template_id: str) -> str:
    """A source whose declared template list reaches ``template_id``.

    Sources and templates are not interchangeable: v1.1.1 withdrew RUC
    from the GFS route (a GFS-initialised RUC forecast cannot complete
    its first step) while leaving it on ERA5, which was never exercised.
    A helper that hardcoded one source would report that withdrawal as a
    failure of every claim it happens to be checking.
    """

    route = physics_registry()["runner_routes"][
        "tools.prepared_single_domain_forecast"]
    declared = route["source_template_ids"]
    for source_id in ("gfs", *sorted(declared)):
        if template_id in declared.get(source_id, ()):
            return source_id
    for source_id, ids in route.get("expert_template_ids", {}).items():
        if template_id in ids:
            return source_id
    raise AssertionError(
        f"no source offers {template_id}; it is unreachable on this route")


def _single_plan(template_id: str = WSM6_TEMPLATE_ID) -> dict[str, object]:
    plan = {
        "schema": PLAN_SCHEMA,
        "plan_id": "single-domain-proof-v1",
        "registry_sha256": registry_sha256(),
        "context": {
            "source_id": _source_offering(template_id),
            "runner_id": "tools.prepared_single_domain_forecast",
            "topology_id": "single-domain-v1",
        },
        "domains": [{"domain_id": "d01", "template_id": template_id}],
        "edges": [],
    }
    if template_id in (NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID,
                       MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID):
        plan["acknowledgements"] = [
            "noahmp-host-column-throughput-v1"]
    return plan


def _uniform_tree(template_id: str = WSM6_TEMPLATE_ID) -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "plan_id": "uniform-tree-proof-v1",
        "registry_sha256": registry_sha256(),
        "context": {
            "source_id": "hrrr",
            "runner_id": "tools.prepared_domain_tree_forecast",
            "topology_id": "one-way-nested-v1",
        },
        "domains": [
            {"domain_id": "d01", "template_id": template_id},
            {"domain_id": "d02", "template_id": template_id},
        ],
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }


def test_tracked_registry_is_the_exact_canonical_gpuwm_authority():
    registry = physics_registry()
    raw = REGISTRY_PATH.read_bytes()
    assert registry["schema"] == REGISTRY_SCHEMA
    assert registry["plan_schema"] == PLAN_SCHEMA
    assert registry["validation_schema"] == VALIDATION_SCHEMA
    assert raw == canonical_json(registry).encode("utf-8") + b"\n"
    assert registry_sha256() == hashlib.sha256(raw[:-1]).hexdigest()


def test_every_repo_local_registry_citation_still_says_what_the_claim_says():
    """The registry's evidence is line numbers, and line numbers rot.

    ``tools/check_registry_citations.py`` re-resolves every ``file:line`` the
    registry publishes: WRF paths must be DECLARED external (so a typo in a
    repo path cannot be silently downgraded to "must be somebody else's
    file"), and every citation into this worktree must carry an ANCHOR -- a
    substring the CLAIM is about -- that still appears in the cited lines.
    Existence is not enough; a drifted citation resolves to a line that
    exists and says something else.

    WHY THIS TEST EXISTS.  The checker was written, never wired to anything,
    and rotted: run for the first time on 2026-08-01 it reported 90 failures,
    ~30 of them ``RESOLVED`` rows for citations the registry had not carried
    for several releases, and its ``EXTERNAL`` table was missing
    ``phys/module_mp_thompson.F``, ``module_mp_thompson.F``,
    ``dyn_em/module_initialize_real.F`` and ``phys/module_physics_init.F``,
    so every one of the ten mp=28 citations failed. A checker nobody runs
    catches nothing, and its own tables become a second thing to be wrong.

    IT FOUND REAL DEFECTS, AND THEY ARE PUBLISHED RATHER THAN SUPPRESSED.
    Two YSU warnings cite ``kernels/ysu.cu`` lines that do not say what the
    warning says -- one of them (``:1315``) cannot exist, because the file is
    607 lines long. Their text is generated by
    ``tools/ysu_wrf461_oracle/patch_registry_maturity.py``, which this
    package does not own, so they are enumerated in the checker's ``DRIFTED``
    table with the measured evidence and the correct target and filed as an
    integration request. That table is self-retiring: each row carries the
    anchor the claim is about, and ``check`` FAILS the moment the cited range
    contains it, so the exception cannot outlive the defect.
    """
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import check_registry_citations as checker
    finally:
        sys.path.pop(0)

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    found = checker.citations(registry)
    assert len(found) >= 60, (
        "the registry publishes almost no file:line evidence any more; that "
        f"is either a regression or a scanner defect ({len(found)} found)")

    failures = checker.check(registry)
    assert failures == [], "\n  ".join([""] + failures)

    # The open defects are REPORTED, not hidden: every row names its owner so
    # a reader of this test knows where the fix lands.
    for line in checker.drifted_report():
        assert "OPEN (owner " in line, line
    assert set(checker.DRIFTED) == {
        "kernels/ysu.cu:1315", "kernels/ysu.cu:252"}, (
        "the enumerated citation-defect list changed; a new entry needs the "
        "measured evidence and an owner, and a removed one needs its claim "
        f"re-verified: {sorted(checker.DRIFTED)}")

    # And the mp=28 row's own citations are all covered, which is the thing
    # this wave rewrote.
    mp28 = [citation for citation, where in found.items()
            if any(MP28_OPTION_ID in path for path in where)]
    assert len(mp28) >= 10, sorted(mp28)
    for citation in mp28:
        path = citation.rpartition(":")[0]
        assert path in checker.EXTERNAL or citation in checker.RESOLVED, (
            f"the mp=28 option cites {citation}, which nothing resolves")


def test_mixed_thompson_outer_and_nssl_inner_plan_is_launchable():
    report = validate_physics_plan(_mixed_plan())
    assert report["schema"] == VALIDATION_SCHEMA
    assert report["launchable"] is True
    assert report["errors"] == []
    assert report["plan_id"] == "mixed-thompson-nssl-proof-v1"
    assert report["context"] == {
        "source_id": "hrrr",
        "runner_id": "tools.prepared_domain_tree_forecast",
        "topology_id": "one-way-nested-v1",
        "edges": [{"parent_domain_id": "d01", "child_domain_id": "d02"}],
    }
    assert [
        domain["settings"]["mp_physics"] for domain in report["resolved_domains"]
    ] == [8, 18]
    assert report["resolved_domains"][1]["settings"]["epssm"] == 0.1
    assert {item["requirement"]["id"] for item in report["asset_requirements"]} >= {
        "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1",
    }
    warning_codes = {warning["code"] for warning in report["warnings"]}
    assert "maturity" in warning_codes
    assert "component-warning" in warning_codes


def test_same_microphysics_nested_edge_is_launchable():
    report = validate_physics_plan(_uniform_tree())
    assert report["launchable"] is True
    assert report["errors"] == []


@pytest.mark.parametrize(
    ("mutator", "expected_code"),
    [
        (lambda plan: plan.update(edges=[]), "tree-edge-count"),
        (
            lambda plan: plan.update(
                edges=[
                    {"parent_domain_id": "d01", "child_domain_id": "d02"},
                    {"parent_domain_id": "d01", "child_domain_id": "d02"},
                ]
            ),
            "duplicate-edge",
        ),
    ],
)
def test_nested_topology_rejects_missing_and_duplicate_edges(mutator, expected_code):
    plan = _uniform_tree()
    mutator(plan)
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert expected_code in {error["code"] for error in report["errors"]}


def test_nested_topology_rejects_cycles_and_multiple_parents():
    plan = _uniform_tree()
    plan["domains"].append({"domain_id": "d03", "template_id": WSM6_TEMPLATE_ID})
    plan["edges"] = [
        {"parent_domain_id": "d01", "child_domain_id": "d02"},
        {"parent_domain_id": "d02", "child_domain_id": "d03"},
        {"parent_domain_id": "d03", "child_domain_id": "d01"},
    ]
    cyclic = validate_physics_plan(plan)
    assert cyclic["launchable"] is False
    assert "tree-cycle" in {error["code"] for error in cyclic["errors"]}

    plan["edges"] = [
        {"parent_domain_id": "d01", "child_domain_id": "d03"},
        {"parent_domain_id": "d02", "child_domain_id": "d03"},
    ]
    multi_parent = validate_physics_plan(plan)
    assert multi_parent["launchable"] is False
    assert "multiple-parents" in {
        error["code"] for error in multi_parent["errors"]
    }


def test_single_domain_topology_rejects_any_edge():
    plan = _single_plan()
    plan["edges"] = [{"parent_domain_id": "d01", "child_domain_id": "d01"}]
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "single-domain-edges" in {error["code"] for error in report["errors"]}


@pytest.mark.parametrize("domain_id", ["", " ", "\t"])
def test_domain_ids_must_be_nonempty_after_trimming(domain_id):
    plan = _single_plan()
    plan["domains"][0]["domain_id"] = domain_id
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "domain-id" in {error["code"] for error in report["errors"]}


def test_runner_routes_match_live_source_and_topology_contracts():
    # Nesting is a property of the topology, not of the source: the tree runner
    # reads every source's hierarchy document, so every source it declares is
    # routable here. Routable is not validated -- GFS in particular has an
    # open root/child humidity-convention question -- but that is a physics
    # judgement, not a routing one, and does not belong in this gate.
    tree = _uniform_tree()
    for source_id in ("hrrr", "era5", "gfs", "20crv3"):
        tree["context"]["source_id"] = source_id
        assert validate_physics_plan(tree)["launchable"] is True, source_id

    tree = _uniform_tree()
    tree["context"]["source_id"] = "not-a-registered-source"
    wrong_tree_source = validate_physics_plan(tree)
    assert wrong_tree_source["launchable"] is False
    assert "unsupported-source-route" in {
        error["code"] for error in wrong_tree_source["errors"]
    }

    tree = _uniform_tree()
    tree["context"]["topology_id"] = "offline-child-v1"
    offline = validate_physics_plan(tree)
    assert offline["launchable"] is False
    assert "unsupported-topology-route" in {
        error["code"] for error in offline["errors"]
    }

    hrrr_single = _single_plan()
    hrrr_single["context"].update(
        source_id="hrrr", runner_id="tools.hrrr_single_domain_benchmark"
    )
    assert validate_physics_plan(hrrr_single)["launchable"] is True
    hrrr_single["context"]["source_id"] = "era5"
    wrong_single_source = validate_physics_plan(hrrr_single)
    assert wrong_single_source["launchable"] is False
    assert "unsupported-source-route" in {
        error["code"] for error in wrong_single_source["errors"]
    }


@pytest.mark.parametrize("source_id", ("hrrr", "era5", "gfs", "20crv3"))
def test_real_source_mp_off_requires_explicit_moist_carrier(source_id):
    plan = _uniform_tree()
    plan["context"]["source_id"] = source_id
    for domain in plan["domains"]:
        domain["components"] = {"microphysics": "off"}

    refused = validate_physics_plan(plan)
    assert refused["launchable"] is False
    errors = [
        error for error in refused["errors"]
        if error["code"] == (
            "real-source-mp-off-requires-explicit-moist")
    ]
    assert [error["path"] for error in errors] == [
        "domains[0].parameters.moist",
        "domains[1].parameters.moist",
    ]
    assert all("allocates qv/qc/qr carrier fields" in error["message"]
               for error in errors)
    assert all("does not synthesize analyzed clouds" in error["message"]
               for error in errors)
    assert all("source-absent cloud mass stays exact zero"
               in error["message"] for error in errors)

    for domain in plan["domains"]:
        domain["parameters"] = {"moist": True}
    admitted = validate_physics_plan(plan)
    assert admitted["launchable"] is True, admitted["errors"]
    assert all(domain["settings"]["moist"] is True
               for domain in admitted["resolved_domains"])


def test_unnamed_tree_tuple_governance_uses_registry_reachability_only(
        capsys):
    from gpuwm.physics_compat import (
        PhysicsCapabilityError,
        multi_domain_physics_selection,
    )

    normal = single_domain_runtime_switches(WSM6_TEMPLATE_ID)
    receipt = multi_domain_physics_selection({1: normal, 2: normal})
    assert {
        domain["governance"]["state"]
        for domain in receipt["domains"].values()
    } == {"registry-reachable"}
    assert receipt["acknowledgements"] == []
    assert receipt["acknowledgement_provenance"] == {}

    pbl_off = {**normal, "bl_pbl_physics": 0}
    pbl_off_receipt = multi_domain_physics_selection(
        {1: pbl_off, 2: pbl_off})
    assert {
        domain["governance"]["state"]
        for domain in pbl_off_receipt["domains"].values()
    } == {"registry-reachable"}
    assert pbl_off_receipt["acknowledgements"] == []

    outside = {
        **normal,
        "ra_lw_physics": 90,
        "ra_sw_physics": 90,
    }
    # Warn-not-block owns the SEVERITY of this site: a tuple outside the
    # registry's declared reachability is still individually implemented,
    # so it runs and says so -- one line per domain, naming the tuple, the
    # state and both published ways to acknowledge it.  What this test
    # guards is unchanged: that the governance is computed from registry
    # reachability alone, that it is REPORTED, and that the
    # acknowledgement is what flips it.
    unacked = multi_domain_physics_selection({1: outside, 2: outside})
    lines = [line for line in capsys.readouterr().err.splitlines()
             if line.startswith("warning:")]
    assert len(lines) == 2, lines
    message = "\n".join(lines)
    assert "d01 physics tuple" in message
    assert "d02 physics tuple" in message
    assert "ra_lw_physics=90" in message
    assert "ra_sw_physics=90" in message
    assert "outside-registry-declared-reachability" in message
    assert (
        '--ack expert-tuple-v1 or acknowledgements = ["expert-tuple-v1"]'
        in message
    )
    # Warned, and recorded as unacknowledged -- not silently blessed.
    assert {
        domain["governance"]["state"]
        for domain in unacked["domains"].values()
    } == {"outside-registry-declared-reachability"}
    assert not any(domain["governance"]["acknowledged"]
                   for domain in unacked["domains"].values())

    acknowledged = multi_domain_physics_selection(
        {1: outside, 2: outside},
        expert_acknowledgements=("expert-tuple-v1",),
    )
    assert {
        domain["governance"]["state"]
        for domain in acknowledged["domains"].values()
    } == {"outside-registry-declared-reachability"}
    assert all(domain["governance"]["acknowledged"]
               for domain in acknowledged["domains"].values())


def test_unnamed_tree_expert_template_retains_its_specific_acknowledgement(
        capsys):
    from gpuwm.physics_compat import multi_domain_physics_selection

    expert = single_domain_runtime_switches(NOAHMP_PROFILE_ID)
    # The point of this test is that an expert TEMPLATE keeps its own
    # specific acknowledgement id rather than collapsing into the generic
    # outside-reachability one -- and that survives warn-not-block's
    # ruling on severity: the tuple runs, and the line it prints names
    # THIS template's acknowledgement, not expert-tuple-v1.
    unacked = multi_domain_physics_selection({1: expert, 2: expert})
    lines = [line for line in capsys.readouterr().err.splitlines()
             if line.startswith("warning:")]
    assert len(lines) == 2, lines
    message = "\n".join(lines)
    assert "noahmp-host-column-throughput-v1" in message
    assert "expert-tuple-v1" not in message
    assert {
        domain["governance"]["required_acknowledgement"]
        for domain in unacked["domains"].values()
    } == {"noahmp-host-column-throughput-v1"}
    assert not any(domain["governance"]["acknowledged"]
                   for domain in unacked["domains"].values())

    receipt = multi_domain_physics_selection(
        {1: expert, 2: expert},
        expert_acknowledgements=("noahmp-host-column-throughput-v1",),
    )
    assert {
        domain["governance"]["state"]
        for domain in receipt["domains"].values()
    } == {"registry-expert-template"}
    assert all(domain["governance"]["acknowledged"]
               for domain in receipt["domains"].values())
    assert "noahmp-host-column-throughput-v1" not in capsys.readouterr().err


def test_registry_routes_drift_check_against_live_runner_capabilities():
    registry = physics_registry()
    routes = registry["runner_routes"]

    hrrr_capabilities = hrrr_runner_capabilities()
    hrrr_route = routes[hrrr_capabilities["runner"]]
    assert hrrr_route["source_ids"] == hrrr_capabilities["supported_sources"]
    assert (
        hrrr_route["source_template_ids"]["hrrr"]
        + hrrr_route["expert_template_ids"]["hrrr"]
    ) == hrrr_capabilities["physics_profile_ids"]

    single_capabilities = prepared_single_runner_capabilities()
    single_route = routes[single_capabilities["runner"]]
    assert single_route["source_ids"] == single_capabilities["supported_sources"]
    for source_id, source in single_capabilities["source_profiles"].items():
        routed = list(single_route["source_template_ids"][source_id])
        routed.extend(
            single_route.get("expert_template_ids", {}).get(source_id, ()))
        assert routed == source["physics_profile_ids"]

    tree_capabilities = tree_runner_capabilities()
    tree_route = routes[tree_capabilities["runner"]]
    assert tree_route["source_ids"] == tree_capabilities["supported_sources"]
    assert tree_route["topology_ids"] == ["one-way-nested-v1"]


def test_fixed_template_runners_reject_all_overrides():
    plan = _single_plan()
    plan["domains"][0]["components"] = {"microphysics": "wsm6-mp6"}
    components = validate_physics_plan(plan)
    assert components["launchable"] is False
    assert "fixed-template-components" in {
        error["code"] for error in components["errors"]
    }

    plan = _single_plan()
    plan["domains"][0]["parameters"] = {"epssm": 0.1}
    parameters = validate_physics_plan(plan)
    assert parameters["launchable"] is False
    assert "parameter-route" in {error["code"] for error in parameters["errors"]}

    plan = _single_plan()
    plan["domains"][0]["expert_overrides"] = {"settings": {"epssm": 0.1}}
    expert = validate_physics_plan(plan)
    assert expert["launchable"] is False
    assert "expert-setting-route" in {error["code"] for error in expert["errors"]}


@pytest.mark.parametrize("component", ["land_surface", "radiation"])
def test_tree_route_rejects_currently_unsupported_component_variation(component):
    plan = _uniform_tree()
    replacements = {
        "pbl": "off",
        "surface_layer": "revised-mm5",
        "land_surface": "off",
        "radiation": "analytic-clear-sky",
    }
    plan["domains"][1]["components"] = {component: replacements[component]}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "component-override-route" in {
        error["code"] for error in report["errors"]
    }


@pytest.mark.parametrize(
    ("component", "option"),
    [("pbl", "off"), ("surface_layer", "revised-mm5"),
     ("radiation", "off")],
)
def test_tree_route_admits_wrf_legal_harmless_component_variation(
        component, option):
    plan = _uniform_tree()
    plan["domains"][1]["components"] = {component: option}
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]


def test_tree_route_admits_morrison_to_nssl_only_with_matrix_policy():
    plan = _uniform_tree(MORRISON_TEMPLATE_ID)
    plan["domains"][1]["components"] = {"microphysics": "nssl2-mp18"}
    plan["domains"][1]["parameters"] = {
        "nest_microphysics_transition": "mp8-to-mp18-mass-diagnosed-v1"
    }
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "transition-required-setting" in {
        error["code"] for error in report["errors"]
    }

    plan["domains"][1]["parameters"] = {
        "nest_microphysics_transition": "mp-edge-mass-diagnosed-v1"
    }
    admitted = validate_physics_plan(plan)
    assert admitted["launchable"] is True
    assert "transition-maturity" in {
        warning["code"] for warning in admitted["warnings"]
    }


def test_registry_advertises_every_mixed_edge_honestly():
    """Every ordered pair of transported-moment schemes, one row each.

    The count is n*(n-1) over the microphysics options the transition
    enumerates -- 6 schemes (mp 1/6/8/9/10/18) give 30 -- and exactly one
    of them is ratified.  It is written out rather than derived from the
    rules under test, which would make the assertion vacuous.
    """
    rules = physics_registry()["transitions"][
        "microphysics-one-way-v1"]["cross_options"]
    pairs = {
        (rule["parent_option_id"], rule["child_option_id"])
        for rule in rules
    }
    assert len(rules) == len(pairs) == 30
    ratified = [rule for rule in rules if rule["status"] == "ratified"]
    assert [(rule["parent_option_id"], rule["child_option_id"])
            for rule in ratified] == [("thompson-mp8", "nssl2-mp18")]
    experimental = [
        rule for rule in rules if rule["status"] == "experimental"
    ]
    assert len(experimental) == 29
    assert {
        rule["maturity"] for rule in experimental
    } == {"experimental-runtime"}


def test_data_driven_component_constraints_reject_engine_impossible_settings():
    plan = _uniform_tree()
    plan["domains"][1]["parameters"] = {"moist": False}
    dry_mp6 = validate_physics_plan(plan)
    assert dry_mp6["launchable"] is False
    assert "component-required-setting" in {
        error["code"] for error in dry_mp6["errors"]
    }

    plan = _uniform_tree(MORRISON_TEMPLATE_ID)
    plan["domains"][1]["parameters"] = {"num_soil_layers": 9}
    wrong_soil = validate_physics_plan(plan)
    assert wrong_soil["launchable"] is False
    assert {error["code"] for error in wrong_soil["errors"]} >= {
        "parameter-route",
        "component-required-setting",
    }


def test_graph_policy_rejects_nonzero_spec_exp_on_nested_child():
    plan = _uniform_tree()
    plan["domains"][1]["parameters"] = {"spec_exp": 0.33}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "graph-setting-constraint" in {
        error["code"] for error in report["errors"]
    }


def test_huge_numeric_parameter_returns_validation_error_instead_of_crashing(
    tmp_path: Path, capsys
):
    plan = _uniform_tree()
    plan["domains"][1]["parameters"] = {"spec_exp": 10**10_000}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "parameter-value" in {error["code"] for error in report["errors"]}

    plan_path = tmp_path / "huge-number-plan.json"
    cli_plan = _uniform_tree()
    cli_plan["domains"][1]["parameters"] = {"spec_exp": "__HUGE_NUMBER__"}
    rendered = json.dumps(cli_plan).replace('"__HUGE_NUMBER__"', "1e9999")
    plan_path.write_text(rendered, encoding="utf-8")
    assert main(["--validate-physics-plan", str(plan_path)]) == EXIT_CONFIG
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["launchable"] is False
    assert cli_report["errors"]


#: Profiles registered on the HRRR routes only (the Kessler rule: no
#: other source inherits evidence from an HRRR-bound run), so they are
#: unreachable on the prepared-single-domain route the parametrized test
#: below walks and carry their own hrrr-context test instead.
_HRRR_ONLY_PROFILE_IDS = (
    KESSLER_PROFILE_ID,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
)


@pytest.mark.parametrize(
    "template_id",
    tuple(
        profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
        if profile not in _HRRR_ONLY_PROFILE_IDS),
)
def test_v2_templates_preserve_every_existing_v1_runtime_switch(template_id):
    report = validate_physics_plan(_single_plan(template_id))
    assert report["launchable"] is True
    resolved = report["resolved_domains"][0]["settings"]
    for key, value in single_domain_runtime_switches(template_id).items():
        assert resolved[key] == value


@pytest.mark.parametrize("template_id", _HRRR_ONLY_PROFILE_IDS)
def test_hrrr_only_templates_preserve_their_runtime_switches(template_id):
    plan = _single_plan(WSM6_TEMPLATE_ID)
    plan["context"].update(
        source_id="hrrr",
        runner_id="tools.hrrr_single_domain_benchmark",
    )
    plan["domains"][0]["template_id"] = template_id
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]
    resolved = report["resolved_domains"][0]["settings"]
    for key, value in single_domain_runtime_switches(template_id).items():
        assert resolved[key] == value


@pytest.mark.parametrize(
    "template_id",
    (NSSL2_PROFILE_ID, NSSL2_LEGACY_RRTMG_PROFILE_ID),
)
def test_nssl2_radiation_profiles_have_distinct_exact_identities(template_id):
    from types import SimpleNamespace

    switches = single_domain_runtime_switches(template_id)
    assert identify_single_domain_profile(SimpleNamespace(**switches)) == (
        template_id)
    assert switches["ra_rrtmg_variant"] in {
        "rte-rrtmgp", "rrtmg_legacy"}


def test_implemented_unverified_maturity_warns_but_never_blocks():
    registry = physics_registry()
    option = registry["components"]["microphysics"]["options"]["wsm6-mp6"]
    option["maturity"] = "implemented-unverified"
    plan = _single_plan()
    plan["registry_sha256"] = registry_sha256(registry)
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is True
    assert report["errors"] == []
    assert any(
        warning["code"] == "maturity"
        and "implemented-unverified" in warning["message"]
        for warning in report["warnings"]
    )


def test_opaque_future_maturity_warns_conservatively_without_blocking():
    registry = physics_registry()
    option = registry["components"]["microphysics"]["options"]["wsm6-mp6"]
    option["maturity"] = "candidate-from-upstream"
    plan = _single_plan()
    plan["registry_sha256"] = registry_sha256(registry)
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is True
    assert any(
        warning["code"] == "maturity"
        and "candidate-from-upstream" in warning["message"]
        for warning in report["warnings"]
    )


def test_template_maturity_and_warning_are_preserved_for_20cr_profile():
    template_id = (
        "20crv3-wsm6-ysu-mm5-noah-kf-rte-rrtmgp-implemented-unverified-v1"
    )
    plan = _single_plan(template_id)
    plan["context"]["source_id"] = "20crv3"
    report = validate_physics_plan(plan)
    assert report["launchable"] is True
    assert {warning["code"] for warning in report["warnings"]} >= {
        "template-maturity",
        "template-warning",
    }


def test_mynn_component_dependencies_are_the_wrf_v461_cells():
    components = physics_registry()["components"]
    assert components["surface_layer"]["options"]["mynn"]["constraints"][
        "requires_components"
    ] == {"pbl": ["off", "mynn"]}
    assert components["pbl"]["options"]["mynn"]["constraints"][
        "requires_components"
    ] == {
        "surface_layer": ["revised-mm5", "classic-mm5", "mynn"]}


def test_mynn_is_implemented_and_warns_rather_than_blocking():
    """Both halves run, and both say what has not been verified.

    ``implemented: true`` is what puts a scheme in a user's picker, so the
    warnings carry the three things a user has to know before choosing it:
    that no gpuwm/WRF trajectory comparison exists, that four of the CUDA
    leaves are not bitwise twins of their CPU references away from the oracle
    fixtures, and that phim/phih still run on the host at 125 microseconds
    per column.
    """
    components = physics_registry()["components"]
    for component in ("pbl", "surface_layer"):
        option = components[component]["options"]["mynn"]
        assert option["implemented"] is True, component
        assert option["maturity"] == "implemented-unverified", component
        assert option["warnings"], component
        assert any("UNVERIFIED against a WRF forecast" in warning
                   for warning in option["warnings"]), component
    pbl_warnings = " ".join(components["pbl"]["options"]["mynn"]["warnings"])
    assert "125 microseconds per column" in pbl_warnings
    assert "NOT bitwise twins" in pbl_warnings


def test_noahmp_is_implemented_and_warns_rather_than_blocking():
    """It runs, and the warnings say exactly what has not been earned.

    ``implemented: true`` is what puts a scheme in a user's picker, so the
    warnings carry the measurements a user needs before choosing it: that no
    gpuwm/WRF trajectory comparison exists, that the whole column runs on
    the DEVICE through the slab orchestration at a measured 0.202-0.227 s
    per 360,000-column call (2026-07-27, twice, one RTX 5090 -- the figure
    that retired the host-era "6.4 ms per land column" scaling blocker),
    that glacier columns run the dedicated NOAHMP_GLACIER port, that sea
    ice has no energy balance under the configurable xice_threshold, and
    that the WRF six-rate and carried-COSZEN seams are active.

    The pinned figures are the ones
    ``gpuwm/core/noahmp_runtime.py`` ``NOAHMP_RUNTIME_RESTRICTIONS``
    ["column_solver_location"] records and
    ``docs/noahmp_device_column_report.md`` publishes; the previous
    revision of this docstring quoted the host-era figure long after the
    measurement moved, which is exactly how a quoted figure becomes false.
    ``tests/test_noahmp_runtime.py::test_the_column_cost_is_what_the_
    registry_says`` stays the always-on order-of-magnitude gate on the
    small-grid per-column cost.
    """
    option = physics_registry()["components"]["land_surface"]["options"][
        "noah-mp"]
    assert option["implemented"] is True
    assert option["maturity"] == "implemented-unverified"
    text = " ".join(option["warnings"])
    assert "UNVERIFIED against a WRF forecast" in text
    assert "runs on the DEVICE" in text
    # Pin the figures themselves, so the warning cannot drift away from
    # what the slab timing runs measured.
    assert "0.202-0.227 s" in text
    assert "7.3-8.2 wall seconds per simulated minute" in text
    assert "GLACIER columns run the dedicated NOAHMP_GLACIER port" in text
    assert "never to NOAHMP_SFLX" in text
    assert "no sea-ice surface energy balance" in text
    assert "xice_threshold" in text
    assert "SIX-RATE precipitation seam is active" in text
    assert "COSZEN is a radiation-driver carrier" in text
    assert "MEASURED INERT" in text
    # Every knob the option pins must be an implemented knob: an implemented
    # option may not pin one nothing reads.
    from gpuwm.physics_registry import parameter_is_implemented

    declared = physics_registry()["parameters"]
    for name in option["parameters"]:
        assert parameter_is_implemented(declared[name]), name


def test_noahmp_is_admitted_with_mm5_or_the_coupled_mynn_suite():
    """The registry and runtime authorities expose the same pairings.

    "eta-similarity" joined this list with the MYJ port.  The list is a
    STRUCTURAL statement -- which surface layers write the exchange fields
    this LSM seam reads -- and the Eta layer writes all of them
    (UST/CHS/CHS2/CQS2/FLHC/FLQC plus the driver's BR); no Noah-MP runtime
    read is MM5-specific.  Its evidence tier lives in the surface-layer
    option's own maturity, which is implemented-unverified, not here.
    """
    option = physics_registry()["components"]["land_surface"]["options"][
        "noah-mp"]
    assert option["constraints"]["requires_components"] == {
        "surface_layer": ["revised-mm5", "classic-mm5", "mynn",
                          "eta-similarity"]}


def test_registry_refuses_the_same_mynn_surface_ysu_cell_as_wrf():
    """MYNN surface with YSU is the fatal half of WRF's mixed-pair law."""

    plan = _single_plan()
    plan["domains"][0]["components"] = {"surface_layer": "mynn"}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert "component-dependency" in {error["code"]
                                      for error in report["errors"]}


@pytest.mark.parametrize(
    ("field", "value", "error_code"),
    [
        ("plan_id", None, "plan-id"),
        ("plan_id", "", "plan-id"),
        ("registry_sha256", None, "registry-binding"),
        ("registry_sha256", "0" * 64, "stale-registry-binding"),
    ],
)
def test_missing_or_stale_plan_identity_binding_blocks(field, value, error_code):
    plan = _single_plan()
    if value is None:
        plan.pop(field)
    else:
        plan[field] = value
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert error_code in {error["code"] for error in report["errors"]}
    assert all(set(issue) == {"code", "path", "message"} for issue in report["errors"])


@pytest.mark.parametrize(
    ("component", "option"),
    [
        ("microphysics", "not-a-scheme"),
        # ("land_surface", "ruc-lsm") stood here until RUC was admitted, and
        # ("radiation", "wrf-rrtm-dudhia") until the WRF RRTM longwave port
        # landed.  "sase" is now the only registered-but-unimplemented
        # option left in the whole registry, so this path has exactly one
        # case to make it with; the assertion below is guarded so the case
        # cannot go on passing vacuously if that option is admitted too.
        ("microphysics", "sase"),
    ],
)
def test_unknown_and_registered_but_unimplemented_options_block(component, option):
    if option != "not-a-scheme":
        registry = physics_registry()
        assert registry["components"][component]["options"][option][
            "implemented"] is False, (
            f"{component}.{option} is implemented now; point this case at "
            "another registered-but-unimplemented option or the "
            "unimplemented-option error path loses its only witness")
    plan = _single_plan()
    plan["domains"][0]["components"] = {component: option}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    expected = "unknown-option" if option == "not-a-scheme" else "unimplemented-option"
    assert expected in {error["code"] for error in report["errors"]}


def test_expert_settings_are_generic_but_selectors_cannot_claim_implementation():
    registry = physics_registry()
    route = registry["runner_routes"]["tools.prepared_single_domain_forecast"]
    route["mode"] = "experiment-per-domain"
    route["allowed_expert_setting_keys"] = ["future_tuning_coefficient"]
    plan = _single_plan()
    plan["domains"][0]["expert_overrides"] = {
        "settings": {"future_tuning_coefficient": 0.25}
    }
    plan["registry_sha256"] = registry_sha256(registry)
    report = validate_physics_plan(plan, registry=registry)
    assert report["launchable"] is True
    assert report["resolved_domains"][0]["settings"][
        "future_tuning_coefficient"
    ] == 0.25
    assert any(
        warning["code"] == "untyped-expert-setting"
        for warning in report["warnings"]
    )

    plan["domains"][0]["expert_overrides"] = {
        "selectors": {"future_physics_selector": 77}
    }
    blocked = validate_physics_plan(plan, registry=registry)
    assert blocked["launchable"] is False
    assert "unknown-expert-selector" in {
        error["code"] for error in blocked["errors"]
    }

    plan["domains"][0]["expert_overrides"] = {"selectors": {"mp_physics": 3}}
    blocked_value = validate_physics_plan(plan, registry=registry)
    assert blocked_value["launchable"] is False
    assert "unknown-selector-combination" in {
        error["code"] for error in blocked_value["errors"]
    }


def test_registry_and_plan_hashes_are_deterministic_over_mapping_order():
    original = _mixed_plan()
    reordered = {
        "edges": original["edges"],
        "domains": original["domains"],
        "registry_sha256": original["registry_sha256"],
        "context": {
            "topology_id": "one-way-nested-v1",
            "runner_id": "tools.prepared_domain_tree_forecast",
            "source_id": "hrrr",
        },
        "plan_id": original["plan_id"],
        "schema": PLAN_SCHEMA,
    }
    first = validate_physics_plan(original)
    second = validate_physics_plan(reordered)
    assert first["plan_sha256"] == second["plan_sha256"]
    assert first["plan_sha256"] == canonical_sha256(original)
    assert first["registry_sha256"] == second["registry_sha256"]
    assert first["registry_sha256"] == registry_sha256()


def test_source_cli_registry_and_validator_are_compact_and_mutually_exclusive(
    tmp_path: Path, capsys
):
    assert main(["--show-physics-registry"]) == 0
    rendered = capsys.readouterr().out
    assert rendered == canonical_json(physics_registry()) + "\n"

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_mixed_plan()), encoding="utf-8")
    assert main(["--validate-physics-plan", str(plan_path)]) == 0
    report_text = capsys.readouterr().out
    assert "\n" not in report_text.rstrip("\n")
    assert json.loads(report_text)["launchable"] is True

    blocked = _single_plan()
    blocked["domains"][0]["components"] = {"microphysics": "sase"}
    plan_path.write_text(json.dumps(blocked), encoding="utf-8")
    assert main(["--validate-physics-plan", str(plan_path)]) == EXIT_CONFIG
    assert json.loads(capsys.readouterr().out)["launchable"] is False

    plan_path.write_text("{", encoding="utf-8")
    assert main(["--validate-physics-plan", str(plan_path)]) == EXIT_CONFIG
    unreadable = json.loads(capsys.readouterr().out)
    assert unreadable["plan_sha256"] is None
    assert unreadable["plan_id"] is None
    assert unreadable["context"] is None
    assert set(unreadable["errors"][0]) == {"code", "path", "message"}

    with pytest.raises(SystemExit) as raised:
        main(["--show-physics-registry", "--list-sources"])
    assert raised.value.code == 2


def test_source_cli_creates_exact_gpuwm_canonical_plan_without_replacement(
    tmp_path: Path, capsys
):
    plan = _mixed_plan()
    plan["domains"][1]["parameters"]["epssm"] = 1e-7
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    canonical_path = tmp_path / "canonical.json"

    assert main([
        "--validate-physics-plan",
        str(plan_path),
        "--canonical-physics-plan-output",
        str(canonical_path),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["launchable"] is True
    canonical_bytes = canonical_path.read_bytes()
    assert canonical_bytes == canonical_json(plan).encode("utf-8")
    assert b"1e-07" in canonical_bytes
    assert not canonical_bytes.endswith((b"\n", b"\r"))

    canonical_path.write_bytes(b"sentinel")
    assert main([
        "--validate-physics-plan",
        str(plan_path),
        "--canonical-physics-plan-output",
        str(canonical_path),
    ]) == EXIT_CONFIG
    refused = json.loads(capsys.readouterr().out)
    assert refused["launchable"] is False
    assert "canonical-plan-write" in {
        error["code"] for error in refused["errors"]
    }
    assert canonical_path.read_bytes() == b"sentinel"


def test_source_cli_canonical_output_follows_validity_and_flag_contract(
    tmp_path: Path, capsys
):
    plan_path = tmp_path / "plan.json"
    canonical_path = tmp_path / "canonical.json"
    blocked = _single_plan()
    blocked["domains"][0]["components"] = {"microphysics": "sase"}
    plan_path.write_text(json.dumps(blocked), encoding="utf-8")

    assert main([
        "--validate-physics-plan",
        str(plan_path),
        "--canonical-physics-plan-output",
        str(canonical_path),
    ]) == EXIT_CONFIG
    assert json.loads(capsys.readouterr().out)["launchable"] is False
    assert canonical_path.read_bytes() == canonical_json(blocked).encode("utf-8")

    canonical_path.unlink()
    plan_path.write_text("{", encoding="utf-8")
    assert main([
        "--validate-physics-plan",
        str(plan_path),
        "--canonical-physics-plan-output",
        str(canonical_path),
    ]) == EXIT_CONFIG
    assert json.loads(capsys.readouterr().out)["launchable"] is False
    assert not canonical_path.exists()

    with pytest.raises(SystemExit) as raised:
        main(["--canonical-physics-plan-output", str(canonical_path)])
    assert raised.value.code == 2


def test_physics_registry_module_is_stdlib_only_and_cli_does_not_import_cupy():
    script = r'''
from importlib.abc import MetaPathFinder
import sys

class RejectNumerics(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"numpy", "cupy"} or fullname.startswith(("numpy.", "cupy.")):
            raise AssertionError(f"unexpected numerical runtime import: {fullname}")
        return None

sys.meta_path.insert(0, RejectNumerics())
import gpuwm.physics_registry
assert "numpy" not in sys.modules
assert "cupy" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    cli_script = r'''
from importlib.abc import MetaPathFinder
import runpy
import sys

class RejectCupy(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cupy" or fullname.startswith("cupy."):
            raise AssertionError(f"unexpected GPU runtime import: {fullname}")
        return None

sys.meta_path.insert(0, RejectCupy())
sys.argv = ["gpuwm-wrf-init", "--show-physics-registry"]
runpy.run_module("gpuwm.source_cli", run_name="__main__")
'''
    completed = subprocess.run(
        [sys.executable, "-c", cli_script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["schema"] == REGISTRY_SCHEMA


def test_real_windows_subprocess_emits_exact_lf_registry_and_validation_bytes(
    tmp_path: Path,
):
    registry_query = subprocess.run(
        [sys.executable, "-m", "gpuwm.source_cli", "--show-physics-registry"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert registry_query.returncode == 0, registry_query.stderr.decode()
    assert registry_query.stdout == REGISTRY_PATH.read_bytes()
    assert registry_query.stdout.endswith(b"\n")
    assert not registry_query.stdout.endswith(b"\r\n")

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_single_plan()), encoding="utf-8")
    validation = subprocess.run(
        [
            sys.executable,
            "-m",
            "gpuwm.source_cli",
            "--validate-physics-plan",
            str(plan_path),
            "--canonical-physics-plan-output",
            str(tmp_path / "canonical-plan.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    assert validation.returncode == 0, validation.stderr.decode()
    assert validation.stdout.endswith(b"\n")
    assert not validation.stdout.endswith(b"\r\n")
    assert json.loads(validation.stdout)["launchable"] is True
    assert (tmp_path / "canonical-plan.json").read_bytes() == canonical_json(
        _single_plan()
    ).encode("utf-8")


def test_registry_is_pinned_to_lf_for_source_archives():
    attributes = subprocess.run(
        ["git", "check-attr", "eol", "--", "gpuwm/physics_registry_v2.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert attributes.stdout.strip().endswith(": eol: lf")


# ==========================================================================
# Aerosol-aware Thompson (mp_physics=28)
#
# The registry entry for mp=28 is a claim about evidence, and the four things
# a reader most needs from it are the four things prose is worst at holding:
# what the label means, what the measurements actually were, what a user can
# and cannot select, and which WRF behaviours gpuwm deliberately does not
# reproduce.  Each is pinned below against the shipped registry, and where a
# claim has a runtime counterpart the runtime is asked directly rather than
# quoted.
# ==========================================================================

MP28_OPTION_ID = "thompson-aerosol-mp28"

#: Every committed WRF v4.6.1 aerosol column fixture, derived from the files
#: on disk rather than typed out, so the registry's published evidence cannot
#: claim a fixture that does not exist or omit one that does.
#:
#: THE GLOB WIDENED ON 2026-08-01, and that is the point.  It used to be
#: ``aero-*-column.csv`` -- the nineteen scenarios MP28_PORT_SPEC.md names --
#: while ``tests/test_thompson_aerosol_adapter.py::_FIXTURES`` globbed
#: ``*-column.csv`` and drove TWENTY-TWO.  The three extra columns
#: (``wp08-freeze``, ``wp08-melt``, ``wp08-nusweep``, oracle ids 120-122 from
#: the same ``build_aero.sh`` run) were therefore outside the registry's
#: published partition entirely, and two of them MISS the gate.  Matching the
#: gate's own glob is what stops a fixture from failing in a place no
#: published number covers.
def _committed_aerosol_fixture_ids() -> set[str]:
    root = ROOT / "gpuwm" / "data" / "thompson" / "oracle-aero"
    return {path.name[: -len("-column.csv")]
            for path in root.glob("*-column.csv")}


def _spec_aerosol_fixture_ids() -> set[str]:
    """The nineteen ``aero-*`` scenarios MP28_PORT_SPEC.md specifies."""
    return {name for name in _committed_aerosol_fixture_ids()
            if name.startswith("aero-")}


def _mp28_option() -> dict:
    return physics_registry()["components"]["microphysics"]["options"][
        MP28_OPTION_ID]


def _mp28_tree_plan() -> dict:
    """Both domains on mp=28, which is the only nesting the port admits."""
    plan = _uniform_tree()
    for domain in plan["domains"]:
        domain["components"] = {"microphysics": MP28_OPTION_ID}
    return plan


def test_mp28_is_registered_at_the_maturity_its_evidence_earns():
    """implemented, warned, and no higher than implemented-unverified.

    ``implemented: true`` is what puts a scheme in a user's picker, so the
    label beside it has to be the one the evidence supports and not one step
    more.  mp=28 has 19 committed WRF column fixtures, per-kernel Fortran
    oracles and bit-exact device-helper probes -- and no forecast has ever
    been run with it, so ``validation-candidate`` (which requires a ratified
    reference comparison) and ``model-validated`` (which requires a matched
    multi-hour run with published decay tables) are both unavailable.  The
    vocabulary is this tree's own, published in
    ``docs/public/PHYSICS.md``; a rename lane rebasing it must move this pin
    with the rest.
    """
    option = _mp28_option()
    assert option["implemented"] is True
    assert option["maturity"] == "implemented-unverified"
    assert option["selectors"] == {"mp_physics": 28}
    assert option["label"] == "Thompson aerosol-aware / MP28"

    registry = physics_registry()
    policy = registry["warning_policy"]
    assert option["maturity"] in policy["warn_maturities"], (
        "the label must be one the warning policy warns on; a scheme with no "
        "forecast evidence must not resolve silently")
    assert option["maturity"] not in policy["nonwarning_maturities"]

    # Same shape as the model-validated sibling it is a port of: moist state
    # required, cq loading on, and nothing else pinned.
    assert option["parameters"] == {"moist": True, "moist_cq": True}
    assert option["constraints"]["required_settings"] == {"moist": True}

    # An implemented option may not pin a knob nothing honours.
    from gpuwm.physics_registry import parameter_is_implemented

    declared = registry["parameters"]
    for name in option["parameters"]:
        assert parameter_is_implemented(declared[name]), name


def test_mp28_publishes_its_measured_column_residuals_not_a_clean_claim():
    """The gate is not green, and the registry has to say so in numbers.

    ``implemented-unverified`` says "column-oracle-measured".  It does not say
    "column-oracle-CLEAN", and the difference is the whole honesty of the
    label, so the measurement lives on the option where a user reads it
    rather than only in a test file.  The published partition must cover every
    committed fixture exactly once: a fixture that quietly leaves the residual
    list without joining the clean list would otherwise vanish.
    """
    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    committed = _committed_aerosol_fixture_ids()
    spec = _spec_aerosol_fixture_ids()
    assert len(committed) == 22, sorted(committed)
    assert len(spec) == 19, sorted(spec)
    assert sorted(committed - spec) == [
        "wp08-freeze", "wp08-melt", "wp08-nusweep"], sorted(committed - spec)
    # BOTH counts, and the registry must publish both: the number the spec
    # names and the number the gate actually drives.  Publishing only the
    # first is how wp08-freeze and wp08-nusweep sat above the gate in no
    # published class for four waves.
    assert evidence["fixtures"] == len(committed)
    assert evidence["spec_fixtures"] == len(spec)
    assert evidence["gate_relative"] == 2.0e-6
    # A forecast comparison now exists -- docs/public/validation/
    # mp28-matched-trajectory.md, an idealized doubly-periodic single-domain
    # run against unmodified WRF v4.6.1.  The anti-overclaim rule this
    # assertion has always enforced is unchanged, only sharpened: the entry
    # may exist, but it must carry its own FAILED gate and its own list of
    # what it does not establish.  A comparison published without those two
    # is exactly the overclaim the original None guarded against.
    forecast = evidence["forecast_trajectory_comparison"]
    assert isinstance(forecast, dict) and forecast, (
        "forecast_trajectory_comparison must either stay None or be a "
        "populated record; an empty one is a claim with no evidence")
    assert forecast["document"] == (
        "docs/public/validation/mp28-matched-trajectory.md")
    assert forecast["declared_verdict"] == "HOLD", (
        "the pre-declared gate FAILED (V3) and the registry must say so; "
        "changing this to a pass without re-running the comparison is the "
        "overclaim this row exists to prevent")
    for key in ("not_nested_not_real_data", "control_result",
                "what_it_does_not_establish"):
        assert forecast[key].strip(), key
    # The comparison is idealized.  It must never be read as a real-data or
    # nested validation, and the record has to say that in its own words.
    assert "nested" in forecast["what_it_does_not_establish"]
    assert forecast["kind"].startswith("idealized")

    clean = set(evidence["clean_fixtures"])
    residual = set(evidence["residual_fixtures"])
    carved = set(evidence["carved_out_bound"])
    # The near-cancellation bound is the THIRD publication class and has to
    # be counted as one.  It was redundant while aero-reduces-to-classic was
    # also in carved_out_bound; the 1.4.1 merge retired that entry, and
    # without this term the union drops the one fixture the port documents
    # most heavily and reports it as unpublished.
    near = set(evidence["near_cancellation_bound"]["fixtures"])
    assert clean | residual | carved | near == committed, (
        "the published evidence and the committed fixture deck disagree: "
        f"{sorted(committed ^ (clean | residual | carved | near))}")
    assert not (clean & residual) and not (clean & carved) \
        and not (residual & carved)

    # The point of the row: some fixtures do NOT clear the gate, and every
    # residual is published as a number rather than as an adjective.
    assert residual, "an empty residual list would be a clean claim"
    for name, fields in evidence["residual_fixtures"].items():
        assert fields, name
        for field, value in fields.items():
            assert isinstance(value, float) and value > evidence[
                "gate_relative"], (name, field, value)

    # ...and the allowanced fixture is disclosed as such rather than folded
    # in with the clean set, because it is a departure somebody chose.  It is
    # now disclosed under near_cancellation_bound alone: the 1.4.1 merge
    # retired the relative carve-out it also used to sit under, so carved is
    # EMPTY and the assertion moves to `near` rather than being deleted.
    assert carved == set()
    assert near == {"aero-reduces-to-classic"}
    # ONE FIELD, not two.  This literal used to read ``{"qr", "nr_per_kg"}``
    # and stayed green for a wave after WP-13a's level-wise sedimentation
    # density took that fixture's ``qr`` to 1.788e-07 -- inside the FLAT
    # 2.0e-06 gate -- and deleted it from ``_END_TO_END_BOUNDS``.  A registry
    # that publishes a carve-out on a field the gate no longer carves out
    # overstates the port's own relaxation by a whole quantity, which is the
    # opposite of the failure this row exists to prevent, so the set is
    # asserted here as a second opinion and read back from the gate itself by
    # ``test_mp28_evidence_matches_the_bound_the_adapter_gate_actually_
    # applies``.
    # ONE LEVEL, not two.  Same second-opinion role the field-set assertion
    # above it used to play for the retired relative bound.
    assert evidence["near_cancellation_bound"]["fixtures"] == {
        "aero-reduces-to-classic": [6]}

    # THE TWO COUNTS, PUBLISHED SEPARATELY.  ``clean_fixtures`` is the
    # UNEXCEPTIONED list -- nothing held out, no bounds dict -- and the gated
    # count is that plus exactly the carved-out fixtures.  Conflating them is
    # how a port claims a clean number it did not earn, so both are numbers
    # on the option and both are derived from the same partition here.
    assert evidence["clean_unexceptioned"] == len(clean)
    assert evidence["clean_as_gated"] == len(clean) + len(carved) + len(near)
    assert evidence["clean_unexceptioned"] < evidence["fixtures"], (
        "a clean count equal to the deck size is a clean claim, and the "
        "residual table below contradicts it")

    # EVERY departure from the flat gate is named, and each one says which
    # direction it moved.  An unpublished allowance is indistinguishable
    # from a hidden one.
    allowances = evidence["allowances"]
    assert allowances, "the gate has allowances and publishes none"
    for allowance in allowances:
        # ``carved | near`` rather than ``carved`` alone: an allowance may be
        # a METRIC change published under near_cancellation_bound as well as
        # a relative bound published under carved_out_bound, and after the
        # 1.4.1 merge the port's only surviving allowance is the former.
        assert set(allowance["fixtures"]) <= (carved | near), (
            "an allowance that applies to a fixture the registry does not "
            f"publish as carved out: {allowance}")
        assert allowance["direction"], allowance
    # TWO, not three.  ``_REFL_DB_BOUNDS`` was RETIRED by WP-13a, not
    # widened: the residual it covered went 5.283e-04 dB -> 3.242e-05 dB,
    # inside the flat 2.0e-4 dB gate, and the gate's dict is now empty.  A
    # retired allowance may not keep being published as a live one -- that
    # reads as a relaxation the port no longer takes -- so it moves to
    # ``retired_allowances``, which must still name its gate constant and
    # say what happened to it.
    # The literals below are a SECOND OPINION on the gate's own
    # _G3_ALLOWANCES, which test_mp28_evidence_publishes_the_allowances_the_
    # gate_actually_has reads back directly.  Both moved at the 1.4.1 merge,
    # in the retiring direction only: _END_TO_END_BOUNDS went live -> retired
    # when the inherited mp=8 sedimentation reconciliations took the residual
    # it covered from 5.700e-06 to 4.146e-07.
    assert {a["gate_constant"] for a in allowances} == {
        "_NEAR_CANCELLATION_LEVELS"}, allowances
    retired = evidence["retired_allowances"]
    assert {a["gate_constant"] for a in retired} == {
        "_END_TO_END_BOUNDS", "_REFL_DB_BOUNDS"}, retired
    for allowance in retired:
        assert allowance["is"] is None and allowance["direction"], allowance
    assert not ({a["gate_constant"] for a in retired}
                & {a["gate_constant"] for a in allowances}), (
        "an allowance cannot be both live and retired")

    # The gate compares 23 quantities, not the 16 the residual table is
    # published in, and the registry must say so or a reader will read the
    # smaller number as the whole comparison.
    assert evidence["compared_quantities"] == 23
    assert sum(evidence["compared_quantities_breakdown"].values()) == 23
    assert evidence["compared_quantities"] > evidence["compared_fields"]
    assert evidence["gate_reflectivity_db"] == 2.0e-4


def test_mp28_evidence_matches_the_bound_the_adapter_gate_actually_applies():
    """Bind the published numbers to the gate that produced them.

    The registry quotes a 2e-6 relative gate and one carved-out fixture.  Both
    are decisions made in ``tests/test_thompson_aerosol_adapter.py``, so they
    are read back from it: if the port ever widens that default bound or adds
    a second carve-out, the registry's published evidence stops describing the
    measurement and this fails instead of drifting.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_gate",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_gate", module)
    spec.loader.exec_module(module)

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    assert evidence["gate_relative"] == module._END_TO_END_DEFAULT_BOUND
    assert evidence["compared_fields"] == len(module._END_TO_END_FIELDS) + 1, (
        "the registry publishes a field count the gate does not compare; the "
        "+1 is rainnc_mm, which the gate carries beside the column fields")
    assert set(evidence["carved_out_bound"]) == set(
        module._END_TO_END_BOUNDS), (
        "the registry publishes a different set of carved-out fixtures than "
        "the gate applies")
    for name, fields in module._END_TO_END_BOUNDS.items():
        published = evidence["carved_out_bound"][name]
        assert set(published) == set(fields), name
        for field, bound in fields.items():
            assert published[field] <= bound, (
                f"{name}.{field}: the registry publishes {published[field]!r} "
                f"as the measured residual, which cannot exceed the gate's "
                f"own carved-out bound {bound!r}")


def test_mp28_evidence_publishes_the_allowances_the_gate_actually_has():
    """The published allowance LIST is the gate's own, name for name.

    ``test_mp28_publishes_its_measured_column_residuals_not_a_clean_claim``
    asserts the set of gate constants as a literal, which is a second opinion
    and is meant to be one.  This is the first opinion: the list is read back
    out of ``tests/test_thompson_aerosol_adapter.py::_G3_ALLOWANCES``, the
    object that enumerates every departure from the flat gate, so adding an
    allowance to the gate without publishing it -- or leaving a RETIRED one
    published as live -- fails here rather than drifting for a wave.

    BEFORE THIS TEST: ``_REFL_DB_BOUNDS`` was retired by WP-13a (the residual
    it covered went 5.283e-04 dB -> 3.242e-05 dB, inside the flat 2.0e-4 dB
    gate, and the dict was emptied) and the registry kept publishing it as
    one of "the port's three named allowances" -- a relaxation the port does
    not take, advertised as though it did.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_allowances",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_allowances", module)
    spec.loader.exec_module(module)

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    live = {a["gate_constant"] for a in evidence["allowances"]}
    retired = {a["gate_constant"] for a in evidence["retired_allowances"]}

    assert live == {constant for _name, constant, _fixtures, _why
                    in module._G3_ALLOWANCES}, (
        "the registry publishes a different set of live allowances than the "
        f"gate's _G3_ALLOWANCES applies: registry {sorted(live)}, gate "
        f"{sorted({c for _n, c, _f, _w in module._G3_ALLOWANCES})}")

    # A constant published as RETIRED must actually be inert in the gate.
    # ``_REFL_DB_BOUNDS`` is kept as an empty dict rather than deleted so
    # ``_g3_bound`` keeps one code path; empty is the proof it buys nothing.
    for constant in retired:
        assert getattr(module, constant) == {}, (
            f"{constant} is published as retired but the gate still applies "
            f"it: {getattr(module, constant)!r}")

    # ...and the fixtures each live allowance names are the gate's own.
    by_constant = {constant: set(fixtures)
                   for _name, constant, fixtures, _why
                   in module._G3_ALLOWANCES}
    for allowance in evidence["allowances"]:
        assert set(allowance["fixtures"]) == by_constant[
            allowance["gate_constant"]], allowance


def test_mp28_published_clean_set_is_the_gates_unexceptioned_clean_set():
    """``clean_fixtures`` is the gate's flat-gate clean list, not a summary.

    The registry's clean list is the strongest claim on the row -- it says
    these columns agree with unmodified WRF on all 23 compared quantities
    with NOTHING held out -- so it is bound to the gate's own pinned
    ``_G3_UNEXCEPTIONED_CLEAN``, which
    ``tests/test_thompson_aerosol_adapter.py`` re-measures on the device.
    The two counts the option publishes are recomputed from it here as well,
    so a stale count and a stale list cannot cover for each other.

    BEFORE THIS TEST: the registry published 15 clean fixtures and a "15 of
    22 / 16 of 22" pair of counts while the gate measured 17 and 18, i.e. the
    row UNDERSTATED the port by two whole columns; the same stale set listed
    ``aero-drop-evap`` and ``aero-ice-demott-idxin`` as residual fixtures
    carrying rainnc 5.165e-04 and 1.279e-04 that now measure exactly 0.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_clean",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_clean", module)
    spec.loader.exec_module(module)

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    published = set(evidence["clean_fixtures"])
    measured = set(module._G3_UNEXCEPTIONED_CLEAN)
    assert published == measured, (
        "the registry's clean_fixtures is not the gate's "
        f"_G3_UNEXCEPTIONED_CLEAN: only in registry "
        f"{sorted(published - measured)}, only in gate "
        f"{sorted(measured - published)}")

    assert set(evidence["residual_fixtures"]) == set(module._G3_RESIDUALS), (
        "the registry's residual_fixtures is not the gate's _G3_RESIDUALS: "
        f"registry {sorted(evidence['residual_fixtures'])}, gate "
        f"{sorted(module._G3_RESIDUALS)}")
    for name, fields in module._G3_RESIDUALS.items():
        assert set(evidence["residual_fixtures"][name]) == set(fields), name
        for field, value in fields.items():
            assert (f"{evidence['residual_fixtures'][name][field]:.3e}"
                    == f"{value:.3e}"), (name, field)

    assert evidence["clean_unexceptioned"] == len(measured)
    assert evidence["clean_as_gated"] == len(module._G3_GATED_CLEAN)
    # The counts are quoted in prose on the same row; the prose must not be
    # able to say something the numbers do not.
    note = evidence["clean_counts_note"]
    assert f"{len(measured)} of {len(module._FIXTURES)}" in note, note
    assert (f"{len(module._G3_GATED_CLEAN)} of {len(module._FIXTURES)}"
            in note), note


def test_every_gate_the_mp28_row_cites_exists_and_is_a_test():
    """A cited gate that does not exist is a fabricated receipt.

    The option's evidence is now a set of pointers -- the G3 column gate, the
    G4 self-forecast gate, the aerosol-initialisation cost measurement and
    its pinned gap -- and a reader is expected to be able to run each of
    them.  Every ``path::name`` the row publishes is resolved here against
    the file on disk, so a rename that leaves the registry behind fails
    instead of turning the evidence into a claim about a test nobody can
    find.
    """
    option = _mp28_option()
    evidence = option["extensions"]["column_oracle_evidence"]
    initialisation = option["extensions"]["aerosol_initialisation"]

    cited = [
        evidence["test"],
        evidence["self_forecast_gate"]["test"],
        evidence["self_forecast_gate"]["longest_integration"]["test"],
        initialisation["measured_forecast_sensitivity"]["test"],
        initialisation["call_site_pin"],
        initialisation["installed_state_pin"],
        initialisation["gpuwm_implementation_evidence"].split(" --")[0],
    ]
    missing = []
    for citation in cited:
        path, _, name = citation.partition("::")
        source = ROOT / path
        if not source.is_file():
            missing.append(f"{path} does not exist")
            continue
        if f"def {name}(" not in source.read_text(encoding="utf-8"):
            missing.append(f"{path} has no test named {name}")
    assert missing == [], missing


def test_mp28_publishes_the_near_cancellation_relaxation_too():
    """The gate has TWO relaxations; the registry must publish both.

    ``carved_out_bound`` was the only one on the option, but
    ``tests/test_thompson_aerosol_adapter.py`` also replaces the relative
    metric with an absolute one -- 32 ulps of the ENTRY value -- at one level
    of one fixture, where a 10 s step evaporates 99.958 % of the rain and the
    survivor is the difference of two nearly equal float32 numbers.  An
    unpublished relaxation is indistinguishable from a hidden one, so it is
    read back from the gate's own constants here: widen the ulp allowance or
    hold out another level and the published evidence stops describing the
    measurement.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_gate_ulp",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_gate_ulp", module)
    spec.loader.exec_module(module)

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    published = evidence["near_cancellation_bound"]

    assert published["ulps_of_entry_value"] == module._NEAR_CANCELLATION_ULPS
    assert {name: tuple(levels)
            for name, levels in published["fixtures"].items()} == {
        name: tuple(levels)
        for name, levels in module._NEAR_CANCELLATION_LEVELS.items()}, (
        "the registry publishes a different set of near-cancellation levels "
        "than the gate holds out")
    # Every held-out fixture must be one the registry already classifies, and
    # the measured ulp figures must sit under the allowance they are quoted
    # against.
    classified = (set(evidence["clean_fixtures"])
                  | set(evidence["residual_fixtures"])
                  | set(evidence["carved_out_bound"])
                  | set(published["fixtures"]))
    for name, measured in published["measured"].items():
        assert name in classified, name
        for field, value in measured.items():
            assert value < published["ulps_of_entry_value"], (name, field)


def test_mp28_is_reachable_only_as_a_per_domain_override_and_is_no_default():
    """Selectable, and selectable exactly one way.

    Reachability is recomputed from the routes by
    ``tests/test_registry_reachability.py``; what this adds is the negative
    half that a recomputation cannot express as an intention -- that no
    template selects mp=28, that no source route offers one, and that the
    shipped default template is untouched.  Those three together are what
    keep an unverified scheme out of every default suite while still letting
    a user opt into it per domain.
    """
    from gpuwm.physics_registry import (DEFAULT_TEMPLATE_ID,
                                        THOMPSON_KF_TEMPLATE_ID)

    registry = physics_registry()
    assert _mp28_option()["reachability"] == {"state": "component-override"}
    assert "blocker" not in _mp28_option()["reachability"], (
        "a reachable option carrying a blocker is a self-contradiction")

    for template_id, template in registry["templates"].items():
        assert template["components"]["microphysics"] != MP28_OPTION_ID, (
            f"template {template_id!r} selects mp=28; it must be reachable "
            "only as an explicit per-domain override")
    assert DEFAULT_TEMPLATE_ID == THOMPSON_KF_TEMPLATE_ID
    assert registry["templates"][DEFAULT_TEMPLATE_ID]["components"][
        "microphysics"] == "thompson-mp8"

    # It really is selectable, on the one route that permits the override.
    report = validate_physics_plan(_mp28_tree_plan())
    assert report["launchable"] is True, report["errors"]
    assert [domain["settings"]["mp_physics"]
            for domain in report["resolved_domains"]] == [28, 28]

    # ...and refused on the fixed-template routes, like every other override.
    single = _single_plan()
    single["domains"][0]["components"] = {"microphysics": MP28_OPTION_ID}
    refused = validate_physics_plan(single)
    assert refused["launchable"] is False
    assert "fixed-template-components" in {
        error["code"] for error in refused["errors"]}


def test_mp28_plan_warns_at_every_deviation_rather_than_blocking():
    """A user who selects mp=28 is told what has not been earned.

    ``maturity_never_blocks`` is registry policy, so the only protection a
    user has is that the warnings actually say the things.  Each phrase below
    is a distinct deviation the port committed to publishing, and a warning
    list that loses one silently would otherwise still pass every structural
    gate.
    """
    report = validate_physics_plan(_mp28_tree_plan())
    assert report["launchable"] is True, report["errors"]
    codes = {warning["code"] for warning in report["warnings"]}
    assert {"maturity", "component-warning"} <= codes

    text = " ".join(warning["message"] for warning in report["warnings"])
    for phrase in (
        "UNVERIFIED against a WRF forecast",
        "THE COLUMN EVIDENCE IS NOT CLEAN",
        "NO AEROSOL INGEST",
        # The aerosol INITIALISATION, which flipped on 2026-08-01.  Until
        # then the registry warned that the synthetic CCN/IN profile was
        # implemented and never installed; the call is now wired, so what
        # must survive the trip through the planner is the CALLER, the fact
        # that ingest is still missing, and the measured sensitivity -- the
        # same number, which was the cost of the gap and is now the value of
        # the profile.  Asserting the old phrases here would preserve a
        # false statement in the one channel a front end renders.
        "gpuwm/core/physics.py::initialize_physics",
        "the aerosol-free run rains 74.2% MORE",
        "5.6x fewer droplets",
        "wif_input_opt=0 but mp_physics=28",
        "dyn_em/module_initialize_real.F:2734-2736",
        "carries NO aerosol inflow",
        # ...with the number that says how fast, which is the only thing
        # that turns "documented, not fixed" into a decision a user can make.
        "19.8638 m/s",
        "flag_qnc/flag_qnwfa/flag_qnifa to MYNN as literal False",
        "MIXED NESTING IS REFUSED BY NAME",
        "DELIBERATE THERMODYNAMIC DIVERGENCE FROM mp_physics=8",
        "CCN_ACTIVATE.BIN",
    ):
        assert phrase in text, f"the mp=28 warnings no longer say: {phrase}"


def test_mp28_mixed_nest_edge_is_refused_by_registry_and_runtime_alike():
    """One refusal, asserted on both authorities.

    The registry refuses the edge by having registered no transition rule for
    it; ``gpuwm/core/microphysics_transition.py`` refuses it by name.  Two
    independent mechanisms for one decision is exactly the shape that drifts,
    so both are exercised here against the same pair, and the runtime's
    message is required to explain itself rather than fall through to a
    generic "not a ported selector".
    """
    from types import SimpleNamespace

    from gpuwm.core.microphysics_transition import (
        SAME_SCHEME_POLICY,
        resolve_microphysics_transition,
    )

    def cfg(mp_physics, policy=None):
        return SimpleNamespace(
            mp_physics=mp_physics,
            nest_microphysics_transition=(
                SAME_SCHEME_POLICY if policy is None else policy))

    rules = physics_registry()["transitions"]["microphysics-one-way-v1"]
    assert not any(
        MP28_OPTION_ID in (rule["parent_option_id"], rule["child_option_id"])
        for rule in rules["cross_options"]), (
        "a cross-scheme rule for mp=28 would advertise a nest edge the "
        "runtime refuses")

    mixed = _uniform_tree()
    mixed["domains"][0]["components"] = {"microphysics": "thompson-mp8"}
    mixed["domains"][1]["components"] = {"microphysics": MP28_OPTION_ID}
    report = validate_physics_plan(mixed)
    assert report["launchable"] is False
    assert "unsupported-component-transition" in {
        error["code"] for error in report["errors"]}

    for parent_mp, child_mp in ((8, 28), (28, 8)):
        with pytest.raises(ValueError) as caught:
            resolve_microphysics_transition(
                cfg(parent_mp), cfg(child_mp, "mp-edge-mass-diagnosed-v1"))
        message = str(caught.value)
        assert "28" in message, (parent_mp, child_mp, message)
        assert "aerosol" in message.lower(), (parent_mp, child_mp, message)
        assert "not a ported" not in message.lower(), (
            "mp=28 IS ported; a generic 'not a ported selector' message would "
            f"tell an operator the scheme is unavailable: {message}")

    # The same-scheme edge, which IS admitted, on both authorities.
    uniform = validate_physics_plan(_mp28_tree_plan())
    assert uniform["launchable"] is True, uniform["errors"]
    contract = resolve_microphysics_transition(cfg(28), cfg(28))
    assert contract.mixed is False
    assert (contract.source_mp_physics, contract.target_mp_physics) == (28, 28)


def test_mp28_activation_table_is_declared_as_a_shipped_but_separate_asset():
    """The asset row must say the wheel satisfies it, without joining mp=8.

    ``CCN_ACTIVATE.BIN`` is third-party parcel-model output that WRF
    redistributes; since 2026-08-01 this repository redistributes the same
    bytes, so ``redistributed_by_gpuwm`` must say so -- a row still claiming
    the operator has to supply it would send every user to a WRF ``run/``
    directory for a file already installed beside the code.

    ``kind`` deliberately stays out of ``packaged-table-set``: that value
    names the CLASSIC set an mp=8 launch resolves through ``TABLE_SET_ID``,
    and this asset must never join it.  The pins are read back from
    ``gpuwm/core/thompson_aerosol_contract.py`` so the registry and the
    loader cannot disagree about which bytes are meant, and the classic mp=8
    contract is asserted UNCHANGED so no existing launch inherits the
    dependency.
    """
    from gpuwm.core.thompson_aerosol_contract import (
        AEROSOL_ASSET_REDISTRIBUTED,
        AEROSOL_TABLE_ASSETS,
        AEROSOL_TABLE_SET_ID,
    )
    from gpuwm.core.thompson_contract import CLASSIC_TABLE_ASSETS

    requirements = _mp28_option()["asset_requirements"]
    assert len(requirements) == 1
    requirement = requirements[0]
    assert requirement["id"] == AEROSOL_TABLE_SET_ID
    assert requirement["kind"] != "packaged-table-set"
    assert requirement["redistributed_by_gpuwm"] is AEROSOL_ASSET_REDISTRIBUTED
    assert AEROSOL_ASSET_REDISTRIBUTED is True
    assert requirement["regenerable"] is False
    # Shipped is not regenerable: nothing in gpuwm or WRF recomputes these
    # numbers, so a lost copy is fetched from a WRF release, not rebuilt.
    # Inside the gpuwm-data companion distribution since 2.5.0, at the same
    # relative path it always had.
    assert requirement["search_root"] == "gpuwm_data/data/thompson/tables"

    pinned = AEROSOL_TABLE_ASSETS[0]
    assert requirement["assets"] == [{
        "filename": pinned.filename,
        "bytes": pinned.bytes,
        "sha256": pinned.sha256,
    }]
    # An operator with a WRF tree must be able to act on the row alone.
    assert requirement["path_environment_override"] == (
        "GPUWM_THOMPSON_CCN_ACTIVATE")
    assert requirement["root_environment_override"] == (
        "GPUWM_THOMPSON_TABLE_ROOT")

    # The classic set is untouched: no mp=8 launch acquires this file.
    classic = {asset.filename for asset in CLASSIC_TABLE_ASSETS}
    assert pinned.filename not in classic
    mp8 = physics_registry()["components"]["microphysics"]["options"][
        "thompson-mp8"]["asset_requirements"]
    assert len(mp8) == 1
    assert {asset["filename"] for asset in mp8[0]["assets"]} == classic

    # A resolved mp=28 plan really does carry the requirement to the launcher.
    report = validate_physics_plan(_mp28_tree_plan())
    assert AEROSOL_TABLE_SET_ID in {
        item["requirement"]["id"] for item in report["asset_requirements"]}


def test_the_aerosol_roadmap_knobs_are_published_and_stay_unsettable():
    """WRF's aerosol-ingest knobs become discoverable, not honoured.

    mp=28 exists now, so the honest place for ``wif_input_opt`` and its family
    is the published roadmap: typed, reasoned, and refused.  The two knobs
    that must NOT appear are the interesting half -- ``aer_init_opt`` and
    ``aer_fire_emit_opt`` are declared ``derived`` in WRF's own Registry
    (Registry.EM_COMMON:2656 and :2658), so they are not legal user settings
    and publishing them as gpuwm knobs would invent a control WRF does not
    have.
    """
    registry = physics_registry()
    parameters = registry["parameters"]

    for name in (
        "wif_input_opt", "num_wif_levels", "use_aero_icbc",
        "use_rap_aero_icbc", "qna_update", "scalar_pblmix",
        "grav_settling", "dust_emis", "wif_fire_emit", "wif_fire_inj",
        "progn", "naer",
    ):
        spec = parameters[name]
        assert spec["implemented"] is False, name
        assert spec["unimplemented_reason"].strip(), name
        assert "default" not in spec, name

    for name in ("aer_init_opt", "aer_fire_emit_opt"):
        assert name not in parameters, (
            f"{name} is DERIVED in WRF, not a namelist knob; declaring it "
            "would publish a control WRF does not offer")

    # The three rows that pre-date the port now point at the option that
    # exists, instead of describing the scheme as unported.
    for name in ("progn", "naer", "use_aero_icbc"):
        assert MP28_OPTION_ID in parameters[name]["unimplemented_reason"], name

    # Published is not settable.
    plan = _mp28_tree_plan()
    plan["domains"][1]["parameters"] = {"wif_input_opt": 0}
    report = validate_physics_plan(plan)
    assert report["launchable"] is False
    assert {"parameter-value", "parameter-route"} & {
        error["code"] for error in report["errors"]}


def test_mp28_does_not_disturb_the_frozen_mp8_registry_entry():
    """The whole port's premise, asserted at the registry layer too.

    mp=8 is the model-validated scheme every shipped template selects.  A
    sibling entry that quietly changed its maturity, its pins or its asset set
    would move a validated trajectory through configuration rather than
    through code, which no numerical gate in this tree is watching for.
    """
    mp8 = physics_registry()["components"]["microphysics"]["options"][
        "thompson-mp8"]
    # RE-PINNED at the 1.4.1 merge, and this is the one pin in the file that
    # is allowed to move for a reason outside mp=28: the mp=8 lane renamed
    # its own maturity tier on the release line.  The assertion still says
    # exactly what it said -- mp=8's registry entry is whatever the release
    # line makes it and mp=28 does not touch it -- so a change originating
    # in THIS branch still fails here.
    assert mp8["maturity"] == "wrf-matched-run"
    assert mp8["implemented"] is True
    assert mp8["reachability"] == {"state": "template"}
    assert mp8["selectors"] == {"mp_physics": 8}
    assert mp8["parameters"] == {"moist": True, "moist_cq": True}
    assert mp8["asset_requirements"][0]["id"] == (
        "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1")
    assert len(mp8["asset_requirements"][0]["assets"]) == 4


def test_the_published_mp28_evidence_agrees_between_registry_and_docs():
    """Nothing mechanically checks prose, so this does, for the numbers.

    ``docs/public/PHYSICS.md`` republishes the mp=28 column residuals for a
    reader who will never open the registry.  Two copies of a measurement is
    how a measurement goes stale, so every number the registry publishes must
    appear in the page, spelled the same way, and the page's maturity label
    for the row must be the registry's own.  This is deliberately one-
    directional: the page may say MORE than the registry (it carries the
    mechanism for each residual), but it may not disagree.
    """
    page = (ROOT / "docs" / "public" / "PHYSICS.md").read_text(
        encoding="utf-8")
    option = _mp28_option()
    evidence = option["extensions"]["column_oracle_evidence"]

    assert "| Thompson aerosol-aware | 28 |" in page, (
        "the microphysics table has no mp=28 row")
    assert f"| {option['maturity']} |" in page, (
        "the page does not use the maturity vocabulary the registry uses; a "
        "vocabulary rebase must move both")

    missing = []
    for name, fields in evidence["residual_fixtures"].items():
        if f"`{name}`" not in page:
            missing.append(name)
        for field, value in fields.items():
            if f"{value:.3e}" not in page:
                missing.append(f"{name}.{field}={value:.3e}")
    for name, fields in evidence["carved_out_bound"].items():
        if f"`{name}`" not in page:
            missing.append(name)
        for field, value in fields.items():
            if f"{value:.3e}" not in page:
                missing.append(f"{name}.{field}={value:.3e}")
    for name in evidence["clean_fixtures"]:
        if f"`{name}`" not in page:
            missing.append(name)
    assert missing == [], (
        "docs/public/PHYSICS.md no longer publishes what the registry "
        f"measured: {missing}")

    # What the page must and must not claim about forecast evidence.
    #
    # REWRITTEN ON 2026-08-01, in the commit that landed the matched
    # trajectory.  Two things were wrong with what stood here.
    #
    # 1.  It was VACUOUS.  It required "no matched" and "trajectory"
    #     anywhere in the whole page, and what satisfied it was three map
    #     PROJECTION rows -- "no matched WRF run" at PHYSICS.md:628-630 --
    #     which have nothing to do with microphysics.  Deleting every word
    #     about mp=28's forecast evidence would not have failed it.  It is
    #     now sliced to the mp=28 section, the same slice
    #     tests/test_physics_md_aerosol_claims.py uses.
    #
    # 2.  It required a claim that had become FALSE.  "No forecast has ever
    #     been validated against WRF" was true until a single-domain doubly
    #     periodic idealized forecast was run against WRF v4.6.1 and
    #     published, with its own failed declared condition, in
    #     docs/public/validation/mp28-matched-trajectory.md.  A gate that
    #     forces the page to keep publishing a superseded sentence is a gate
    #     that manufactures a false statement.
    #
    # What is still true, and is what this now requires, is the NARROWER
    # claim: no REAL-DATA and no NESTED forecast has been compared, and
    # neither can be -- WRF's own real.exe is a fatal error on the
    # configuration and ArWen has no aerosol lateral boundary condition.
    # Tokens, not sentences, so a rewrite of the prose does not break it
    # and cannot quietly drop the qualification either.
    section = page[page.index("### Thompson aerosol-aware (`mp_physics = 28`)")
                   :page.index("## Planetary boundary layer")].lower()

    assert "real-data" in section and "nested" in section, (
        "the mp=28 section must qualify its forecast evidence: the matched "
        "comparison that exists is idealized, and no real-data or nested "
        "forecast has been validated against WRF")
    assert "validated against wrf" in section, (
        "the mp=28 section must still make an explicit statement about what "
        "has and has not been validated against WRF")
    assert "validation/mp28-matched-trajectory.md" in section, (
        "the mp=28 section must point at the one matched forecast "
        "comparison, so a reader reaches its limits and its failed "
        "declared condition without being told they exist")
    assert "implemented-unverified" in section, (
        "the matched idealized comparison did not raise the maturity label, "
        "and the section is where a reader learns that")
    for overclaim in ("model-validated | 28", "validation-candidate | 28",
                      "wrf-matched-run | 28"):
        assert overclaim not in page, overclaim

    # PROVENANCE.md is the repo-level deviation register and republishes the
    # same measurement in its D9 entry; it drifts the same way.
    register = (ROOT / "PROVENANCE.md").read_text(encoding="utf-8")
    assert "### D9 " in register, (
        "the deviation register has no mp=28 entry")
    drifted = []
    for group in ("residual_fixtures", "carved_out_bound"):
        for name, fields in evidence[group].items():
            for field, value in fields.items():
                if f"{value:.3e}" not in register:
                    drifted.append(f"{name}.{field}={value:.3e}")
    assert drifted == [], (
        f"PROVENANCE.md D9k no longer matches the registry: {drifted}")

    # CONFIGURATION.md is where a user looks for a knob, so every aerosol
    # knob the registry refuses has to be findable there under its own name.
    knobs = (ROOT / "docs" / "public" / "CONFIGURATION.md").read_text(
        encoding="utf-8")
    for name in (
        "use_aero_icbc", "use_rap_aero_icbc", "wif_input_opt",
        "num_wif_levels", "qna_update", "scalar_pblmix", "grav_settling",
        "dust_emis", "wif_fire_emit", "wif_fire_inj",
    ):
        assert f"`{name}`" in knobs, (
            f"CONFIGURATION.md does not document the refused knob {name}")
    assert "aer_init_opt" in knobs and "derived" in knobs, (
        "the page must say why aer_init_opt is not an ArWen knob")


def test_mp28_published_residuals_still_equal_a_live_adapter_measurement():
    """The published evidence is RE-MEASURED here, not just cross-checked.

    Every other gate on this row compares one written-down number against
    another written-down number: registry against docs, registry against the
    gate's declared bound, registry against the fixture deck on disk.  All of
    those stay green while the whole set drifts away from what the port
    actually computes, because a transcription is only as current as the last
    person who retyped it -- and this row's entire honesty rests on the
    numbers being the real ones.  So this runs the nineteen fixtures through
    the shipped adapter on the device and rebuilds the published partition
    from the result.

    Three things are asserted, and the first two are the ones a stale
    transcription breaks:

    * a fixture the registry calls CLEAN must clear the gate on every field
      today -- otherwise the option is publishing a clean claim it no longer
      earns, which is the exact overclaim ``implemented-unverified`` exists to
      prevent;
    * a fixture the registry lists as a residual must still miss, on exactly
      the fields published, at exactly the published values;
    * the values are compared at the precision they are published to
      (``%.3e``), because that is the precision at which they are quoted in
      ``docs/public/PHYSICS.md``, ``PROVENANCE.md`` and the registry alike.

    Exact comparison is legitimate rather than flaky: the measurement was
    repeated three times end to end on this device and all 19x16 values were
    BIT-identical across repeats, so there is no run-to-run jitter for a
    tolerance to absorb.  If that ever stops being true the right response is
    to record the spread, not to widen this.
    """
    import importlib.util

    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:      # pragma: no cover
            pytest.skip("no CUDA device")
    except Exception:                                 # pragma: no cover
        pytest.skip("no CUDA device")

    spec = importlib.util.spec_from_file_location(
        "_mp28_adapter_gate_live",
        ROOT / "tests" / "test_thompson_aerosol_adapter.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_adapter_gate_live", module)
    spec.loader.exec_module(module)
    # The adapter gate skips itself when the four classic tables or
    # CCN_ACTIVATE.BIN are not staged; this must skip for the same reason
    # rather than fail, because an unstaged table is an environment fact.
    module._tables_or_skip()

    evidence = _mp28_option()["extensions"]["column_oracle_evidence"]
    gate = evidence["gate_relative"]
    published_residual = evidence["residual_fixtures"]
    published_carved = evidence["carved_out_bound"]
    clean = set(evidence["clean_fixtures"])

    drift: list[str] = []
    for scenario in module._FIXTURES:
        measured, _ = module._run_g3(cp, scenario)
        carved = module._END_TO_END_BOUNDS.get(scenario, {})
        missing = {field: value for field, value in measured.items()
                   if not value <= carved.get(field, gate)}

        if scenario in clean:
            if missing:
                drift.append(
                    f"{scenario} is published CLEAN but misses "
                    + ", ".join(f"{field}={value:.3e}"
                                for field, value in sorted(missing.items())))
            continue

        expected = published_residual.get(scenario) or published_carved.get(
            scenario)
        if expected is None:
            # THE THIRD CLASS, and it became load-bearing at the 1.4.1 merge.
            # A fixture can also be published as clean-only-under-the-
            # near-cancellation-bound, which is a METRIC change at one level
            # rather than a relative carve-out and so has never lived in
            # published_carved.  aero-reduces-to-classic was in BOTH classes
            # until the merge retired its relative bound; it is now in this
            # one alone, and reading only the first two would have reported
            # the port's best-documented fixture as unpublished.
            if scenario in evidence["near_cancellation_bound"]["fixtures"]:
                if missing:
                    drift.append(
                        f"{scenario} is published clean under the "
                        "near-cancellation bound but misses "
                        + ", ".join(f"{field}={value:.3e}"
                                    for field, value in sorted(
                                        missing.items())))
                continue
            drift.append(f"{scenario} is in no published class")
            continue

        if scenario in published_carved:
            # A carved-out fixture clears the gate only because of the bound,
            # so what is published is the measured value under it.
            for field, value in expected.items():
                got = measured[field]
                if f"{got:.3e}" != f"{value:.3e}":
                    drift.append(
                        f"{scenario}.{field}: published {value:.3e}, "
                        f"measured {got:.3e}")
            continue

        if set(missing) != set(expected):
            drift.append(
                f"{scenario}: published misses {sorted(expected)}, "
                f"measured misses {sorted(missing)}")
        for field, value in expected.items():
            got = measured.get(field)
            if got is None or f"{got:.3e}" != f"{value:.3e}":
                drift.append(
                    f"{scenario}.{field}: published {value:.3e}, measured "
                    + ("absent" if got is None else f"{got:.3e}"))

    assert drift == [], (
        "the registry publishes mp=28 column evidence that the adapter no "
        "longer produces. Re-measure with "
        "tests/test_thompson_aerosol_adapter.py::"
        "test_g3_end_to_end_against_all_nineteen_oracle_fixtures and move the "
        "numbers in tools/build_registry.py (MP28_G3_CLEAN / "
        "MP28_G3_RESIDUALS / MP28_G3_CARVED_OUT), then regenerate the "
        "registry and update docs/public/PHYSICS.md and PROVENANCE.md D9k. "
        "Never round a residual toward the gate.\n  " + "\n  ".join(drift))


# ==========================================================================
# What the registry says an mp=28 run DOES, versus what the shipped tree
# does.  Both gates below exist because a machine-readable claim that is
# false about the model that ships is worse than no claim at all: a front end
# renders it, a user plans around it, and nothing in the tree contradicts it.
# ==========================================================================

def _microphysics_init_production_callers() -> list[str]:
    """Every non-test call site of ``microphysics.microphysics_init``.

    Deliberately the SAME scan as ``tests/test_mp28_forecast_smoke.py::
    test_gap_microphysics_init_has_no_production_call_site``, so the pinned
    gap and the published claim cannot disagree about whether WRF's synthetic
    CCN/IN profile is installed on a real run.  ``microphysics.py`` itself is
    inspected above its own definition only, so the definition is not counted
    as a call.
    """
    root = ROOT / "gpuwm"
    callers: list[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name == "microphysics.py" and "def microphysics_init" in text:
            head = text.split("def microphysics_init", 1)[0]
            if "microphysics_init(" in head:
                callers.append(path.relative_to(ROOT).as_posix())
            continue
        if re.search(r"\bmicrophysics_init\s*\(", text):
            callers.append(path.relative_to(ROOT).as_posix())
    return callers


def test_mp28_publishes_the_aerosol_initialisation_the_tree_actually_performs():
    """The registry may not promise a CCN profile the tree never installs.

    WRF's ``thompson_init`` fills a synthetic CCN/IN profile at domain
    construction whenever the aerosol fields arrive unset
    (phys/module_mp_thompson.F:493-558) and derives ``nwfa2d`` from it at
    :509-510.  gpuwm implements that fill in
    ``gpuwm/core/microphysics.py::microphysics_init`` and proves it against
    WRF -- but until something in ``gpuwm/`` CALLS it, every mp=28 forecast
    starts from ``nwfa = nifa = 0`` and the terminal apply's clamps
    (:3972-4021) hold the aerosol at its floors for the whole run.  The
    registry used to publish the opposite ("Every mp_physics=28 run takes
    thompson_init's SYNTHETIC CCN/IN profile"), which a front end renders as
    a capability.

    So the published claim is derived from the tree here, not transcribed:
    the call sites are scanned, and the registry must name exactly what the
    scan finds.  When the hook is finally wired this test does not go stale --
    it demands that the registry stop warning, and names the caller.
    """
    option = _mp28_option()
    assert "aerosol_initialisation" in option["extensions"], (
        "the registry does not publish which aerosol initialisation an "
        "mp_physics=28 run actually performs; a reader cannot tell whether "
        "WRF's synthetic CCN/IN profile is installed")
    block = option["extensions"]["aerosol_initialisation"]

    from gpuwm.core import microphysics

    assert callable(microphysics.microphysics_init)
    assert block["gpuwm_implementation"] == (
        "gpuwm/core/microphysics.py::microphysics_init")
    assert block["wrf_source"].startswith("phys/module_mp_thompson.F:493-558")

    callers = _microphysics_init_production_callers()
    published = block["production_call_site"]

    warnings = " ".join(option["warnings"])
    # The exact sentence the port shipped as false.  Named as a literal so a
    # rewrite that reintroduces it fails here rather than being rediscovered.
    false_claim = "takes thompson_init's SYNTHETIC CCN/IN profile"

    if not callers:                                   # pragma: no cover
        # The state this row was written for, kept live so that a refactor
        # which drops the call reopens the warning instead of leaving the
        # registry promising a profile nothing installs.
        assert published is None, (
            f"the registry publishes production_call_site={published!r} "
            "while nothing in gpuwm/ calls microphysics_init")
        assert false_claim not in warnings, (
            "the registry still claims mp=28 runs take WRF's synthetic CCN/IN "
            "profile, and nothing in gpuwm/ calls microphysics_init")
        assert "NO AEROSOL INITIALISATION" in warnings, (
            "the warning list does not name the gap at all; a user selecting "
            "the scheme is told nothing about running at the CCN floor")
        assert block["shipped_profile"].startswith("none:"), block
        assert "11.1e6" in block["shipped_consequence"], (
            "the consequence must name the floor the run is pinned at")
        assert block["operator_workaround"], (
            "a documented gap must tell the operator what to do instead")
        return

    # THE CALL IS WIRED (2026-08-01).  What the registry must now do is name
    # the caller exactly, in a form that is repo-relative rather than a path
    # on whoever's machine generated it, and stop warning about a gap that
    # is closed -- a stale warning is as false as a stale capability claim.
    assert isinstance(published, str) and "::" in published, (
        "production_call_site must name module::function, not just a module; "
        f"got {published!r}")
    module_path, _, function = published.partition("::")
    assert module_path == callers[0], (
        f"the registry publishes production_call_site={published!r} while "
        f"the tree has {callers!r}")
    assert not module_path.startswith("/") and ".." not in module_path, (
        f"the published call site is not repo-relative: {module_path!r}")
    source = (ROOT / module_path).read_text(encoding="utf-8")
    assert f"def {function}(" in source, (
        f"{module_path} has no function named {function}")
    assert len(callers) == 1, (
        "WRF calls thompson_init from mp_init and nowhere else; a second "
        f"gpuwm caller is how once-per-domain becomes twice: {callers}")

    assert false_claim not in warnings, false_claim
    assert "NO AEROSOL INITIALISATION" not in warnings, (
        f"{published} now installs the profile; that warning is stale")
    assert "NOTHING in gpuwm/ CALLS it" not in warnings, (
        "the registry still warns that nothing calls the hook, and "
        f"{published} does")
    assert "NO AEROSOL INGEST" in warnings, (
        "the INGEST gap is a separate and still-open one -- no WIF metgrid "
        "stream, no GOCART reader, no nbca -- and closing the "
        "initialisation must not take that warning with it")
    assert published in warnings, (
        "the human-readable warning must name the caller too; a front end "
        "renders warnings and not extensions")

    assert block["shipped_profile"].startswith("WRF's synthetic"), block
    # The installed profile must be described by its OWN values, and they
    # must be the fixture's, not a round number: WRF's :508 exponent carries
    # the first layer's thickness, so the lowest level is nowhere near the
    # naCCN1 + naCCN0 = 350e6 ceiling.
    for value in ("1.478987e+08", "5.000000e+07"):
        assert value in block["shipped_profile"], value
    assert "350e6" not in block["shipped_profile"]
    assert "strictly ABOVE" in block["shipped_consequence"], (
        "the consequence of installing the profile is that a domain starts "
        "above WRF's clamps rather than pinned at them; that is what makes "
        "it a physics difference rather than a cosmetic one")
    assert block["operator_workaround"].startswith("none needed"), (
        "the workaround must stop telling operators to call the hook "
        "themselves now that initialize_physics does")

    cost = block["measured_forecast_sensitivity"]
    for key in ("test", "steps", "timestep_s", "domain", "reading",
                "initial_mean_nwfa_per_kg_with_profile",
                "initial_mean_nwfa_per_kg_without_profile",
                "peak_nc_per_kg_with_profile",
                "peak_nc_per_kg_without_profile",
                "domain_total_rainnc_mm_with_profile",
                "domain_total_rainnc_mm_without_profile",
                "domain_total_rainnc_relative_excess"):
        assert key in cost, key
    # Internally consistent: the published excess must be the ratio of the
    # two published accumulations, not an independently typed number.
    excess = (cost["domain_total_rainnc_mm_without_profile"]
              / cost["domain_total_rainnc_mm_with_profile"] - 1.0)
    assert abs(excess - cost["domain_total_rainnc_relative_excess"]) < 1e-3
    assert cost["domain_total_rainnc_relative_excess"] > 0.10, (
        "the number that made this the port's largest measured error is the "
        "number that now says how much the profile is worth; it does not "
        "disappear when the call lands")
    assert (cost["peak_nc_per_kg_without_profile"]
            < cost["peak_nc_per_kg_with_profile"]), (
        "the published mechanism is fewer droplets without CCN")
    assert cost["initial_mean_nwfa_per_kg_without_profile"] == 0.0
    ratio = (cost["peak_nc_per_kg_with_profile"]
             / cost["peak_nc_per_kg_without_profile"])
    assert abs(ratio - cost["droplet_ratio_with_over_without"]) < 0.05, (
        "the published droplet ratio is not the ratio of the two published "
        f"peaks ({ratio:.3f})")
    assert cost["test"].startswith("tests/test_mp28_forecast_smoke.py::")
    # The numbers must also be in the human-readable warning, because a
    # front end renders warnings and not extensions.
    for value in ("1.5980e+08", "2.8439e+07", "1.781185", "3.102043"):
        assert value in warnings, value


def test_the_published_aerosol_initialisation_cost_is_still_the_measured_one():
    """RE-MEASURE the published defect size rather than trusting the digits.

    Runs the same two forecasts the evidence comes from -- identical in every
    respect except that one calls ``microphysics_init`` -- and checks the
    claim the registry actually makes.

    WHAT IS GATED, AND WHY NOT THE DIGITS.  ``domain_total_rainnc_*`` is
    published as a SNAPSHOT of how large this gap is, not as a physics pin:
    any legitimate mp=28 numerics change (a closed column residual, a
    corrected rate) moves the sixth decimal of both accumulations, and a
    digit-exact gate here would go red on an improvement.  What must not
    change silently is the claim: the aerosol-free run rains MORE, by a lot,
    and it produces FEWER droplets.  So the sign and the >10 % magnitude are
    gated exactly as published, and the published excess is required to stay
    within a factor of two of the live one -- tight enough to catch the gap
    being closed, fixed, or growing an order of magnitude, and loose enough
    not to fail on a legitimate physics correction elsewhere in the port.
    """
    import importlib.util

    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:      # pragma: no cover
            pytest.skip("no CUDA device")
    except Exception:                                 # pragma: no cover
        pytest.skip("no CUDA device")

    spec = importlib.util.spec_from_file_location(
        "_mp28_forecast_smoke_live",
        ROOT / "tests" / "test_mp28_forecast_smoke.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mp28_forecast_smoke_live", module)
    spec.loader.exec_module(module)
    module._tables_or_skip()

    cost = _mp28_option()["extensions"]["aerosol_initialisation"][
        "measured_forecast_sensitivity"]
    assert cost["steps"] == module.FORECAST_STEPS
    assert cost["timestep_s"] == module.FORECAST_DT

    _cfg, filled = module._forecast(cp, bubble=True, initialise=True)
    _cfg2, unfilled = module._forecast(cp, bubble=True, initialise=False)
    assert filled["init_receipt"], "the control run installed no profile"
    assert unfilled["init_receipt"] == {}
    assert float(unfilled["nwfa_initial"].max()) == 0.0

    rain_filled = float(filled["rain_sum"][-1])
    rain_unfilled = float(unfilled["rain_sum"][-1])
    assert rain_filled > 0.0 and rain_unfilled > 0.0
    live = rain_unfilled / rain_filled - 1.0
    published = cost["domain_total_rainnc_relative_excess"]

    assert live > 0.10, (
        "the published defect size no longer holds: removing WRF's CCN "
        f"profile changed domain-total rain by {live:+.2%}")
    assert max(unfilled["nc_max"]) < max(filled["nc_max"]), (
        "the published mechanism (fewer, larger droplets without CCN) is not "
        "the one acting")
    assert 0.5 <= (1.0 + live) / (1.0 + published) <= 2.0, (
        f"the registry publishes a {published:+.1%} rain excess and the tree "
        f"now measures {live:+.1%}; re-measure with {cost['test']} and move "
        "the numbers in tools/build_registry.py")


def test_the_registry_flag_qs_contract_is_the_one_the_shipped_runtime_applies():
    """The published MYNN snow classification must be the runtime's own.

    WRF's Registry.EM_COMMON:3036 gives package ``thompsonaero`` (mp_physics
    == 28) ``moist:qv,qc,qr,qi,qs,qg``, so ``F_QS`` is TRUE for it;
    phys/module_pbl_driver.F:877 forwards that flag and
    phys/module_bl_mynn.F:734/:876 substitute ``sqs = 0`` when it is false.
    The registry has published 28 as a flag_qs-true selector since the scheme
    was registered -- while ``gpuwm/core/mynn_pbl_runtime.py`` passed
    ``flag_qs=False`` for it, so MYNN never saw snow under mp=28 and the
    published claim was false about the model that ships.

    The two are bound together here rather than compared by eye: the runtime
    set is the authority for what ships, WRF's Registry is the authority for
    what is right, and the registry may not disagree with either.
    """
    from gpuwm.core.mynn_pbl_runtime import (
        MYNN_SNOW_MICROPHYSICS, mynn_flag_qs)

    registry = physics_registry()
    species = registry["components"]["pbl"]["options"]["mynn"][
        "extensions"]["supplied_moisture_species"]
    published_true = species["flag_qs_true_microphysics_selectors"]
    published_false = species["flag_qs_false_microphysics_selectors"]

    assert 28 in published_true, (
        "WRF Registry.EM_COMMON:3036 declares qs for the thompsonaero "
        "package, so F_QS is true for mp_physics=28")
    assert sorted(MYNN_SNOW_MICROPHYSICS) == published_true, (
        "the registry publishes a MYNN snow classification the shipped "
        f"runtime does not apply: registry {published_true}, runtime "
        f"{sorted(MYNN_SNOW_MICROPHYSICS)}")
    for selector in published_true:
        assert mynn_flag_qs(selector) is True, selector
    for selector in published_false:
        assert mynn_flag_qs(selector) is False, selector
    assert species["gpuwm_runtime_source"].startswith(
        "gpuwm/core/mynn_pbl_runtime.py::MYNN_SNOW_MICROPHYSICS"), (
        "the published classification must name the shipped set it is "
        "checked against")

    # Every implemented microphysics option has to be on exactly one side.
    live = {int(option["selectors"]["mp_physics"])
            for option in registry["components"]["microphysics"][
                "options"].values()
            if option.get("implemented") is True}
    assert live == set(published_true) | set(published_false)
    assert not set(published_true) & set(published_false)


def test_mp28_mynn_is_handed_snow_rather_than_a_published_promise():
    """The classification is only worth what the driver does with it.

    A selector list is a claim about behaviour, so the behaviour is measured:
    the committed WRF v4.6.1 MYNN driver oracle's snow-bearing column is run
    through gpuwm's own driver with the flag the mp=28 runtime supplies, once
    with the oracle's snow and once with it zeroed.  If mp=28 really is a
    flag_qs-true selector the two must differ; when the runtime passed
    ``flag_qs=False`` they were bitwise identical, which is exactly what
    "MYNN never sees snow under mp=28" means.

    MEASURED on the ``snow_anvil`` column (max sqs 4.08e-05): with the flag
    on, ``qi_bl`` peaks at 5.4863e-07 and with it off it is exactly 0, and
    every other MYNN output moves with it.
    """
    import importlib.util

    import numpy as np

    from gpuwm.core.mynn_pbl import mynn_bl_driver
    from gpuwm.core.mynn_pbl_runtime import mynn_flag_qs

    spec = importlib.util.spec_from_file_location(
        "_mynn_driver_oracle", ROOT / "tests" / "test_mynn_pbl.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("_mynn_driver_oracle", module)
    spec.loader.exec_module(module)

    _blocks, values, initflag, delt = module._driver_step(2)
    index = module.DRIVER_CASES.index("snow_anvil")
    assert float(values["sqs"][index].max()) > 1.0e-5, (
        "the oracle column carries no snow, so this measures nothing")

    supplied = mynn_bl_driver(
        values, initflag=initflag, delt=delt, flag_qs=mynn_flag_qs(28))
    zeroed = {name: array.copy() for name, array in values.items()}
    zeroed["sqs"][...] = 0.0
    withheld = mynn_bl_driver(
        zeroed, initflag=initflag, delt=delt, flag_qs=mynn_flag_qs(28))

    moved = [name for name in ("qi_bl", "qc_bl", "cldfra_bl", "rqvblten",
                               "rthblten", "exch_h")
             if not np.array_equal(np.asarray(supplied[name])[index],
                                   np.asarray(withheld[name])[index])]
    assert "qi_bl" in moved, (
        "MYNN produced the same cloud ice with and without the column's "
        "snow, so mp_physics=28 is not being handed snow at all")
    assert float(np.max(np.asarray(withheld["qi_bl"])[index])) == 0.0
    assert float(np.max(np.asarray(supplied["qi_bl"])[index])) > 0.0
    assert len(moved) >= 5, moved


def test_an_unnamed_mp28_tree_is_registry_reachable_without_an_acknowledgement():
    """The front door a user actually reaches, asked directly.

    ``validate_physics_plan`` is the declarative authority; the tuple-capability
    check in :func:`gpuwm.physics_compat.multi_domain_physics_selection` is the
    one a launch goes through, and it consults the SAME registry reachability.
    A component-override option that resolved as
    ``outside-registry-declared-reachability`` would demand
    ``--ack expert-tuple-v1`` for a scheme the registry says is normally
    selectable, which is the shape of a front door that disagrees with its own
    catalogue.
    """
    from gpuwm.physics_compat import (
        multi_domain_physics_selection,
        single_domain_runtime_switches,
    )

    aerosol = {**single_domain_runtime_switches(WSM6_TEMPLATE_ID),
               "mp_physics": 28}
    receipt = multi_domain_physics_selection({1: aerosol, 2: aerosol})
    assert {domain["governance"]["state"]
            for domain in receipt["domains"].values()} == {
        "registry-reachable"}
    assert receipt["acknowledgements"] == []

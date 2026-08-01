from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    SINGLE_DOMAIN_PHYSICS_PROFILES,
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
    if template_id in (NOAHMP_PROFILE_ID, MYNN_NOAHMP_PROFILE_ID):
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


def test_registry_advertises_all_twenty_mixed_edges_honestly():
    rules = physics_registry()["transitions"][
        "microphysics-one-way-v1"]["cross_options"]
    pairs = {
        (rule["parent_option_id"], rule["child_option_id"])
        for rule in rules
    }
    assert len(rules) == len(pairs) == 20
    ratified = [rule for rule in rules if rule["status"] == "ratified"]
    assert [(rule["parent_option_id"], rule["child_option_id"])
            for rule in ratified] == [("thompson-mp8", "nssl2-mp18")]
    experimental = [
        rule for rule in rules if rule["status"] == "experimental"
    ]
    assert len(experimental) == 19
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


@pytest.mark.parametrize(
    "template_id",
    tuple(
        profile for profile in SINGLE_DOMAIN_PHYSICS_PROFILES
        if profile != KESSLER_PROFILE_ID),
)
def test_v2_templates_preserve_every_existing_v1_runtime_switch(template_id):
    report = validate_physics_plan(_single_plan(template_id))
    assert report["launchable"] is True
    resolved = report["resolved_domains"][0]["settings"]
    for key, value in single_domain_runtime_switches(template_id).items():
        assert resolved[key] == value


def test_hrrr_kessler_template_preserves_its_runtime_switches():
    plan = _single_plan(WSM6_TEMPLATE_ID)
    plan["context"].update(
        source_id="hrrr",
        runner_id="tools.hrrr_single_domain_benchmark",
    )
    plan["domains"][0]["template_id"] = KESSLER_PROFILE_ID
    report = validate_physics_plan(plan)
    assert report["launchable"] is True, report["errors"]
    resolved = report["resolved_domains"][0]["settings"]
    for key, value in single_domain_runtime_switches(
            KESSLER_PROFILE_ID).items():
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
    that glacier columns are refused, that sea ice has no energy balance,
    and that the WRF six-rate and carried-COSZEN seams are active.

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
    assert "GLACIER columns are REFUSED" in text
    assert "no sea-ice surface energy balance" in text
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
    """The registry and runtime authorities expose the same pairings."""
    option = physics_registry()["components"]["land_surface"]["options"][
        "noah-mp"]
    assert option["constraints"]["requires_components"] == {
        "surface_layer": ["revised-mm5", "classic-mm5", "mynn"]}


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
        ("microphysics", "sase"),
        # ("land_surface", "ruc-lsm") stood here until RUC was admitted.  The
        # unimplemented-option path keeps its coverage from the two options
        # that are still registered-but-unimplemented, in two different
        # components, rather than losing a case with the scheme it named.
        ("radiation", "wrf-rrtm-dudhia"),
    ],
)
def test_unknown_and_registered_but_unimplemented_options_block(component, option):
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

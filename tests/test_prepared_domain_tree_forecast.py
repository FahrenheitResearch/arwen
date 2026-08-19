"""CPU contracts for the prepared arbitrary/mixed domain-tree launcher."""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from gpuwm.core.microphysics_transition import MP8_TO_MP18_POLICY
from gpuwm.experiment import load_experiment
from tools import prepared_domain_tree_forecast as runner


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_mixed_config(tmp_path: Path) -> Path:
    path = tmp_path / "mixed.toml"
    path.write_text(
        textwrap.dedent(f"""
        [experiment]
        name = "mixed-four-domain"
        start_time = 2026-07-23T00:00:00
        run_seconds = 3600.0
        restart_interval_s = 0.0

        [shared]
        nz = 8
        ztop = 12000.0
        moist = true
        moist_cq = true
        mp_physics = 8

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = 200
        ny = 200
        dx = 12000.0
        time_step = 60
        history_interval_s = 3600.0

        [[domain]]
        grid_id = 2
        parent_id = 1
        i_parent_start = 51
        j_parent_start = 51
        parent_grid_ratio = 4
        parent_time_step_ratio = 4
        nx = 200
        ny = 200
        history_interval_s = 3600.0

        [[domain]]
        grid_id = 3
        parent_id = 2
        i_parent_start = 51
        j_parent_start = 51
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        nx = 180
        ny = 180
        history_interval_s = 1800.0
        mp_physics = 18
        nest_microphysics_transition = "{MP8_TO_MP18_POLICY}"

        [[domain]]
        grid_id = 4
        parent_id = 3
        i_parent_start = 51
        j_parent_start = 51
        parent_grid_ratio = 2
        parent_time_step_ratio = 2
        nx = 120
        ny = 120
        history_interval_s = 900.0
        mp_physics = 18
    """),
        encoding="utf-8",
    )
    return path


def _write_two_domain_config(tmp_path: Path) -> Path:
    path = tmp_path / "tree.toml"
    path.write_text(
        textwrap.dedent("""
        [experiment]
        name = "prepared-tree"
        start_time = 2026-07-23T00:00:00
        run_seconds = 3600.0
        restart_interval_s = 0.0

        [projection]
        map_proj = "lambert"
        ref_lat = 40.0
        ref_lon = -83.0
        truelat1 = 30.0
        truelat2 = 60.0
        stand_lon = -84.0

        [shared]
        nz = 8
        ztop = 12000.0
        p_top = 10000.0
        eta_levels = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0]
        hybrid_opt = 2
        etac = 0.2
        map_proj = 1
        moist = true
        mp_physics = 6

        [[domain]]
        grid_id = 1
        parent_id = 0
        i_parent_start = 1
        j_parent_start = 1
        parent_grid_ratio = 1
        parent_time_step_ratio = 1
        nx = 100
        ny = 100
        dx = 12000.0
        time_step = 60
        history_interval_s = 3600.0

        [[domain]]
        grid_id = 2
        parent_id = 1
        i_parent_start = 31
        j_parent_start = 31
        parent_grid_ratio = 3
        parent_time_step_ratio = 3
        nx = 90
        ny = 90
        history_interval_s = 1800.0
    """),
        encoding="utf-8",
    )
    return path


def test_sealed_extension_identity_omits_only_enumerated_forecast_stops(
    tmp_path,
):
    first = load_experiment(_write_two_domain_config(tmp_path))
    later_domains = tuple(
        replace(domain, run=replace(domain.run, run_seconds=7200.0))
        for domain in first.domains
    )
    later = replace(first, run_seconds=7200.0, domains=later_domains)
    runtime = {"source": "prepared-tree", "identity": "stable"}

    assert runner.sealed_extension_fingerprint(first, runtime) == \
        runner.sealed_extension_fingerprint(later, runtime)

    changed_history = replace(
        later,
        domains=(replace(
            later.domains[0], history_interval_s=1800.0),
            *later.domains[1:]),
    )
    assert runner.sealed_extension_fingerprint(later, runtime) != \
        runner.sealed_extension_fingerprint(changed_history, runtime)

    divergent_copy = replace(
        later,
        domains=(replace(
            later.domains[0],
            run=replace(later.domains[0].run, run_seconds=7100.0)),
            *later.domains[1:]),
    )
    with pytest.raises(ValueError, match="diverges"):
        runner.sealed_extension_fingerprint(divergent_copy, runtime)


def test_capability_query_is_side_effect_free_and_warning_only(capsys):
    assert runner.main(["--show-capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload == runner.runner_capabilities()
    # Nested execution is a property of the topology, not of the source: every
    # source whose hierarchy document the runner can read is advertised here.
    assert payload["supported_sources"] == list(runner.SUPPORTED_SOURCES)
    assert set(payload["supported_sources"]) == {"hrrr", "era5", "gfs", "20crv3"}
    assert payload["warning_policy"]["implemented_unverified_is_launchable"]
    assert payload["warning_policy"]["consent_gate"] is False
    assert (
        payload["simulation_plans"][runner.THOMPSON_NSSL_PLAN_ID][
            "explicit_expert_consent_required"
        ]
        is False
    )
    arbitrary = payload["simulation_plans"][runner.ARBITRARY_PLAN_ID]
    assert arbitrary["validity_authority"] == "gpuwm.experiment.load_experiment"
    assert arbitrary["geometry_whitelist"] is False


def test_capability_executable_runs_outside_checkout_without_cupy(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.mkdir()
    (blocker / "sitecustomize.py").write_text(
        textwrap.dedent("""
        from importlib.abc import MetaPathFinder
        import sys

        class BlockCupy(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cupy" or fullname.startswith("cupy."):
                    raise ModuleNotFoundError("capability query imported CuPy")
                return None

        sys.meta_path.insert(0, BlockCupy())
    """),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(blocker)
    completed = subprocess.run(
        [sys.executable, str(Path(runner.__file__).resolve()), "--show-capabilities"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["runner"] == runner.RUNNER
    assert payload["simulation_plan_ids"] == [
        runner.ARBITRARY_PLAN_ID,
        runner.THOMPSON_NSSL_PLAN_ID,
    ]
    assert list(tmp_path.iterdir()) == [blocker]


def test_exact_four_domain_thompson_nssl_plan_is_advertised_not_whitelisted(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", "1")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(tmp_path))
    exp = load_experiment(_write_mixed_config(tmp_path))

    plan = runner.resolve_execution_plan(exp)

    assert plan["plan_id"] == runner.THOMPSON_NSSL_PLAN_ID
    assert plan["launch_allowed"] is True
    assert plan["explicit_expert_consent_required"] is False
    assert [row["mp_physics"] for row in plan["domains"]] == [8, 8, 18, 18]
    assert plan["mixed_transition_count"] == 1
    assert plan["transitions"][1]["source_domain"] == 2
    assert plan["transitions"][1]["target_domain"] == 3
    assert plan["transitions"][1]["policy_id"] == MP8_TO_MP18_POLICY
    assert plan["transitions"][2]["mixed"] is False
    assert plan["microphysics_edges_stock_wrf_equivalent"] is False
    assert plan["whole_simulation_stock_wrf_certified"] is False

    arbitrary = load_experiment(_write_two_domain_config(tmp_path))
    arbitrary_plan = runner.resolve_execution_plan(arbitrary)
    assert arbitrary_plan["plan_id"] == runner.ARBITRARY_PLAN_ID
    assert arbitrary_plan["launch_allowed"] is True
    with pytest.raises(
            ValueError, match="static one-way execution plan.*feedback=1"):
        runner.resolve_execution_plan(replace(arbitrary, feedback=1))


def test_mixed_plan_still_rejects_a_missing_real_translation_policy(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GPUWM_EXPERIMENTAL_THOMPSON_MP8", "1")
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(tmp_path))
    path = _write_mixed_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            f'nest_microphysics_transition = "{MP8_TO_MP18_POLICY}"',
            'nest_microphysics_transition = "same-scheme-only"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires explicit"):
        load_experiment(path)


def test_output_claim_is_create_only_and_cannot_overlap_inputs(tmp_path):
    protected = tmp_path / "prepared"
    protected.mkdir()
    output = tmp_path / "runs" / "unique"

    assert (
        runner.claim_output_directory(output, protected_roots=(protected,))
        == output.resolve()
    )
    # An EMPTY directory that already exists is accepted, and has to be:
    # `gpuwm sim` allocates this run's stamped folder create-exclusively
    # before it dispatches, so the folder handed here exists and is empty
    # on every single tree run.  Refusing it prevented nothing and killed
    # the prepare-then-simulate route.
    assert (
        runner.claim_output_directory(output, protected_roots=(protected,))
        == output.resolve()
    )
    # Holding an earlier run is the breakage, and it still refuses.
    (output / "receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already holds a run"):
        runner.claim_output_directory(output, protected_roots=(protected,))
    with pytest.raises(ValueError, match="overlaps protected input"):
        runner.claim_output_directory(protected / "run", protected_roots=(protected,))


class _FakePreparedCacheReader:
    def __init__(self, path, *, expected_identity):
        self.path = Path(path)
        self.header = json.loads(
            (self.path / "header.json").read_text(encoding="utf-8")
        )
        assert self.header["identity"] == expected_identity
        self.content_sha256 = "d" * 64
        self.payload_bytes = 123
        self.arrays = {"state/u": {}}

    def verify_all(self):
        return {"status": "PASS", "content_sha256": self.content_sha256}


def _synthetic_prepared_tree(tmp_path, monkeypatch):
    config = _write_two_domain_config(tmp_path)
    exp = load_experiment(config)
    prepared = tmp_path / "prepared"
    hierarchy = prepared / "hierarchy-artifacts"
    domains_root = hierarchy / "domains"
    domains_root.mkdir(parents=True)
    bridge = "a" * 64
    source = "b" * 64
    namelist = "c" * 64
    domain_receipts = []
    for domain in exp.domains:
        label = f"d{domain.grid_id:02d}"
        bundle = domains_root / label
        cache = bundle / "prepared-cache"
        cache.mkdir(parents=True)
        static_path = bundle / "native-static.npz"
        np.savez(static_path, DUMMY=np.ones((1,), dtype=np.float32))
        geometry = {"geometry": {"grid_id": domain.grid_id}}
        geometry_path = bundle / "geometry-receipt.json"
        geometry_path.write_text(json.dumps(geometry), encoding="utf-8")
        identity = {
            "bridge_manifest_sha256": bridge,
            "source_manifest_sha256": source,
            "static_cache_sha256": _sha(static_path),
            "namelist_sha256": namelist,
            "domain_config": runner._strict_json(asdict(domain)),
            "forcing_hours": [0, 1],
            "source_identity": {"adapter": "fixture", "grid_id": domain.grid_id},
        }
        header = {
            "identity": identity,
            "content_sha256": "d" * 64,
            "metadata": {
                "lbc": {} if domain.parent_id == 0 else None,
                "base_scalars": {"p_top": 10000.0},
            },
        }
        (cache / "header.json").write_text(json.dumps(header), encoding="utf-8")
        artifact = {
            "prepared_cache": {
                "path": "prepared-cache",
                "content_sha256": "d" * 64,
                "payload_bytes": 123,
                "array_count": 1,
            },
            "static_cache": {
                "path": "native-static.npz",
                "bytes": static_path.stat().st_size,
                "sha256": _sha(static_path),
                "fields": ["DUMMY"],
            },
            "geometry_receipt": {
                "path": "geometry-receipt.json",
                "sha256": _sha(geometry_path),
                "geometry": geometry["geometry"],
            },
        }
        domain_receipt = {
            "schema": "gpuwm-native-domain-artifact-build-v1",
            "status": "READY",
            "grid_id": domain.grid_id,
            "parent_id": domain.parent_id,
            "boundary_mode": (
                "external-specified"
                if domain.parent_id == 0
                else "nested-parent-forced"
            ),
            "artifacts": artifact,
            "verification": {
                "schema": "gpuwm-prepared-real-cache-v1",
                "status": "PASS",
                "path": "prepared-cache",
                "content_sha256": "d" * 64,
                "array_count": 1,
                "payload_bytes": 123,
            },
        }
        (bundle / "receipt.json").write_text(
            json.dumps(domain_receipt), encoding="utf-8"
        )
        domain_receipts.append(domain_receipt)

    ids = [domain.grid_id for domain in exp.domains]
    manifest = {
        "schema": runner.ARTIFACT_MANIFEST_SCHEMA,
        "domains": [
            {
                "grid_id": grid_id,
                "prepared_cache": f"domains/d{grid_id:02d}/prepared-cache",
                "static_cache": f"domains/d{grid_id:02d}/native-static.npz",
                "geometry_receipt": f"domains/d{grid_id:02d}/geometry-receipt.json",
            }
            for grid_id in ids
        ],
    }
    manifest_path = hierarchy / "domain-artifacts.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    artifact_receipt = {
        "schema": runner.ARTIFACT_RECEIPT_SCHEMA,
        "status": "READY",
        "domain_count": 2,
        "grid_ids": ids,
        "manifest": {"path": manifest_path.name, "sha256": _sha(manifest_path)},
        "boundary_inventory": {"external": [1], "nested_parent_forced": [2]},
        "domains": domain_receipts,
    }
    (hierarchy / "receipt.json").write_text(
        json.dumps(artifact_receipt), encoding="utf-8"
    )
    preparation = {
        "schema": runner.HIERARCHY_SCHEMA,
        "status": "PASS",
        "valid_time": exp.start_time.isoformat(),
        "domain_count": 2,
        "forcing_hours": [0, 1],
        "provenance": {
            "bridge_manifest_sha256": bridge,
            "source_manifest_sha256": source,
            "native_namelist_input_sha256": namelist,
        },
        "artifact_receipt": artifact_receipt,
    }
    receipt = prepared / "receipt.json"
    receipt.write_text(json.dumps(preparation), encoding="utf-8")

    monkeypatch.setattr(runner, "PreparedCacheReader", _FakePreparedCacheReader)
    monkeypatch.setattr(
        runner, "grids_from_projection_config", lambda _exp: (object(), object())
    )
    monkeypatch.setattr(
        runner, "verify_native_static_receipt", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner, "load_native_static_cache", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(runner, "_validate_vertical", lambda *_args, **_kwargs: None)
    return prepared, receipt, config


def test_preflight_reads_a_namelist_driven_source_hierarchy(tmp_path, monkeypatch):
    """A prepared tree is resolvable whatever source wrote its top document.

    HRRR prepares its tree in a separate pass and writes receipt.json with a
    nested ``provenance`` block.  The namelist-driven sources build theirs
    inside RW-WPS preparation and write proof.json, which carries the
    initialization time as ``forcing_times`` and no provenance at all -- the
    authority triple lives in each domain's own cache identity, which this
    runner already requires every domain to agree on.  Both must resolve, or
    nesting is a property of the source rather than of the topology.
    """

    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    preparation = json.loads(receipt.read_text(encoding="utf-8"))
    proof = {
        "schema": "gpuwm-era5-native-hierarchy-proof-v1",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "forcing_times": [preparation["valid_time"]],
        "domain_count": preparation["domain_count"],
        "forcing_hours": preparation["forcing_hours"],
        "artifact_receipt": preparation["artifact_receipt"],
    }
    receipt.unlink()
    proof_path = prepared / "proof.json"
    proof_path.write_text(json.dumps(proof), encoding="utf-8")

    inputs = runner.preflight_prepared_tree(
        prepared_root=prepared,
        preparation_receipt_sha256=_sha(proof_path),
        experiment_config=config,
        experiment_config_sha256=_sha(config),
    )

    assert inputs.source == "era5"
    assert [item.grid_id for item in inputs.domains] == [1, 2]
    # The authority triple came from the domain caches, and every domain was
    # still required to agree on it.
    assert set(inputs.authority_sha256) >= {"preparation_receipt", "experiment_config"}


def test_single_domain_config_hits_the_designed_two_domain_floor(
        tmp_path, monkeypatch):
    """The two-domain floor is a division of labor, pinned as designed.

    This runner's docstring is explicit -- it "does not flatten a nest
    tree into the single-domain benchmark runner" -- and a rented-node
    smoke (2026-08-04) proved the floor fires for real after a
    successful fetch, preparation and tree build.  The floor stays
    exactly as it is; the route preflight names it up front
    (tools/battery_route_preflight.py run.runner_selection), and a
    single-domain forecast runs through its source's single-domain
    runner instead.
    """

    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    two_domain_text = config.read_text(encoding="utf-8")
    single = tmp_path / "single.toml"
    head, _, _child = two_domain_text.rpartition("[[domain]]")
    single.write_text(head.rstrip() + "\n", encoding="utf-8")
    assert len(load_experiment(single).domains) == 1

    with pytest.raises(ValueError, match="requires at least two domains"):
        runner.preflight_prepared_tree(
            prepared_root=prepared,
            preparation_receipt_sha256=_sha(receipt),
            experiment_config=single,
            experiment_config_sha256=_sha(single),
        )


def test_preflight_refuses_an_unrecognized_hierarchy_document(tmp_path, monkeypatch):
    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    preparation = json.loads(receipt.read_text(encoding="utf-8"))
    preparation["schema"] = "gpuwm-some-other-hierarchy-v9"
    receipt.write_text(json.dumps(preparation), encoding="utf-8")

    # The refusal names the file it read and the schema it found -- the
    # old wording repeated "proof.json digest differs" once per accepted
    # (file, schema) pair and named neither.
    with pytest.raises(
            ValueError,
            match=r"no hierarchy document matching .*receipt\.json carries "
                  r"schema 'gpuwm-some-other-hierarchy-v9'"):
        runner.preflight_prepared_tree(
            prepared_root=prepared,
            preparation_receipt_sha256=_sha(receipt),
            experiment_config=config,
            experiment_config_sha256=_sha(config),
        )


def test_preflight_binds_every_domain_and_detects_identity_drift(tmp_path, monkeypatch):
    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)

    inputs = runner.preflight_prepared_tree(
        prepared_root=prepared,
        preparation_receipt_sha256=_sha(receipt),
        experiment_config=config,
        experiment_config_sha256=_sha(config),
    )

    assert [item.grid_id for item in inputs.domains] == [1, 2]
    assert inputs.forcing_hours == (0, 1)
    assert inputs.source_identity == {"adapter": "fixture"}
    assert inputs.boundary_interval_seconds == 3600
    assert inputs.execution_plan["launch_allowed"] is True
    assert all(
        item.cache_reader.verify_all()["status"] == "PASS" for item in inputs.domains
    )

    header_path = (
        prepared
        / "hierarchy-artifacts"
        / "domains"
        / "d02"
        / "prepared-cache"
        / "header.json"
    )
    header = json.loads(header_path.read_text(encoding="utf-8"))
    header["identity"]["domain_config"]["parent_id"] = 99
    header_path.write_text(json.dumps(header), encoding="utf-8")
    with pytest.raises(ValueError, match="d02 prepared cache was prepared by .*fields differ: parent_id"):
        runner.preflight_prepared_tree(
            prepared_root=prepared,
            preparation_receipt_sha256=_sha(receipt),
            experiment_config=config,
            experiment_config_sha256=_sha(config),
        )



def test_preflight_normalizes_dead_and_write_only_switches_like_the_gate(
        tmp_path, monkeypatch):
    """The last refusal in the v1.3.1 field report, and its control.

    A hierarchy sealed ``cudt_minutes = 5.0`` (RunConfig's live default,
    inherited because a WRF importer omits the key when cumulus is off)
    and ``nwp_diagnostics = 0``; the wizard-emitted experiment wrote the
    profile's ``0.0`` and, because its audience reads UH products, ``1``.
    Preflight compared by exact equality and refused -- after preparation,
    on a cumulus interval with no cumulus scheme and a diagnostic toggle
    that cannot move one grid point.  The hierarchy's own root binding
    already normalized the first of those; both sides go through the same
    normalization now.
    """

    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    header_path = (
        prepared / "hierarchy-artifacts" / "domains" / "d02"
        / "prepared-cache" / "header.json"
    )
    header = json.loads(header_path.read_text(encoding="utf-8"))
    run = header["identity"]["domain_config"]["run"]
    assert run["cu_physics"] == 0
    run["cudt_minutes"] = 5.0
    run["nwp_diagnostics"] = 1
    run["restart_interval_s"] = 3600.0
    run["output_interval_s"] = 900.0
    header_path.write_text(json.dumps(header), encoding="utf-8")

    inputs = runner.preflight_prepared_tree(
        prepared_root=prepared,
        preparation_receipt_sha256=_sha(receipt),
        experiment_config=config,
        experiment_config_sha256=_sha(config),
    )

    assert [item.grid_id for item in inputs.domains] == [1, 2]
    assert inputs.execution_plan["launch_allowed"] is True

    # Control: an ACTIVE cumulus interval is a real difference and still
    # refuses, so this normalization is about dead state only.
    header["identity"]["domain_config"]["run"]["cu_physics"] = 1
    header_path.write_text(json.dumps(header), encoding="utf-8")
    with pytest.raises(ValueError, match="cudt_minutes"):
        runner.preflight_prepared_tree(
            prepared_root=prepared,
            preparation_receipt_sha256=_sha(receipt),
            experiment_config=config,
            experiment_config_sha256=_sha(config),
        )


def test_the_tree_runner_uses_the_shared_radiation_workspace_predicate():
    """A local restatement of the predicate killed legacy-RRTMG trees.

    `any(radiation_scheme_ids == (4, 4))` is also true for the LEGACY
    RRTMG variant, which runs one domain at a time and holds no
    persistent workspace -- so the tree runner allocated one the
    preflight had not priced and died on the memory-ledger drift guard,
    "shared radiation allocation differs from preflight".  That is the
    exact failure `uses_modern_rrtmgp_workspace`'s docstring predicts,
    which is the argument for calling it rather than restating it.
    """
    import inspect
    from pathlib import Path
    from types import SimpleNamespace

    from gpuwm.core.model import uses_modern_rrtmgp_workspace
    from gpuwm.config import radiation_scheme_ids
    from gpuwm.physics_compat import RRTMG_VARIANT_LEGACY

    source = Path(
        inspect.getfile(runner)).read_text(encoding="utf-8")
    assert "if uses_modern_rrtmgp_workspace(exp)" in source, (
        "the tree runner must ask the shared helper")
    assert "if any(radiation_scheme_ids(domain.run) == (4, 4)" not in source

    # And the two predicates genuinely disagree on a legacy-RRTMG tree,
    # which is why the restatement was a bug rather than a duplicate.
    def _exp(variant):
        domains = [SimpleNamespace(run=SimpleNamespace(
            ra_physics=0, ra_lw_physics=4, ra_sw_physics=4,
            ra_rrtmg_variant=variant))]
        return SimpleNamespace(domains=domains)

    legacy = _exp(RRTMG_VARIANT_LEGACY)
    assert all(radiation_scheme_ids(d.run) == (4, 4) for d in legacy.domains)
    assert uses_modern_rrtmgp_workspace(legacy) is False


# ---------------------------------------------------------------------------
# The mp8 table gap: one sentence at preflight, never a traceback mid-run
# ---------------------------------------------------------------------------

def test_a_missing_thompson_table_is_a_refusal_not_a_traceback(
        tmp_path, monkeypatch, capsys):
    """`FileNotFoundError: missing Thompson table asset .../qr_acr_qg_V4.dat`

    A wheel user's nested GFS run died exactly here, five frames deep,
    after paying for a fetch and three minutes of preprocessing -- and
    the path it named was inside site-packages, where `gpuwm
    fetch-tables` had staged nothing this install could keep.  The gap
    is now answered by the preflight, in one sentence naming the table
    and the command that stages it.
    """

    from gpuwm.table_assets import MissingTableAssets

    empty = tmp_path / "no-tables"
    empty.mkdir()
    monkeypatch.setenv("GPUWM_THOMPSON_TABLE_ROOT", str(empty))

    def _preflight(**_kwargs):
        from gpuwm.table_assets import require_thompson_tables

        require_thompson_tables()
        raise AssertionError("preflight should have refused")

    monkeypatch.setattr(runner, "preflight_prepared_tree", _preflight)
    prepared = tmp_path / "prepared-root"
    prepared.mkdir()
    outdir = tmp_path / "run"
    code = runner.main([
        "--prepared-root", str(prepared),
        "--preparation-receipt-sha256", "a" * 64,
        "--experiment-config", str(_write_mixed_config(tmp_path)),
        "--experiment-config-sha256", "b" * 64,
        "--outdir", str(outdir),
    ])
    assert code == 2
    printed = capsys.readouterr().err
    assert "Traceback" not in printed
    assert "prepared_domain_tree_forecast: refused:" in printed
    assert "qr_acr_qg_V4.dat" in printed
    assert "gpuwm fetch-tables" in printed
    # A refusal is not a failed run: nothing ran, so no failed-run
    # receipt claims otherwise.
    assert not (outdir / "evidence" / "failed-run-receipt.json").exists()

    # Negative control: the same runner path with the tables present
    # gets past this gate and fails on its own (fake) preflight instead.
    monkeypatch.delenv("GPUWM_THOMPSON_TABLE_ROOT")
    with pytest.raises(AssertionError, match="should have refused"):
        runner.main([
            "--prepared-root", str(prepared),
            "--preparation-receipt-sha256", "a" * 64,
            "--experiment-config", str(_write_mixed_config(tmp_path)),
            "--experiment-config-sha256", "b" * 64,
            "--outdir", str(tmp_path / "run2"),
        ])
    assert MissingTableAssets is not None


def test_the_tree_preflight_checks_tables_before_the_hierarchy(tmp_path,
                                                               monkeypatch):
    """Placement, not just presence: the gap is named BEFORE the GPU.

    ``_verify_thompson_assets`` used to run at the top of
    ``run_prepared_tree``; it now runs inside
    ``preflight_prepared_tree``, which is the function whose whole job
    is to answer "can this run" without touching a card.
    """

    import inspect

    source = inspect.getsource(runner.preflight_prepared_tree)
    assert "_verify_thompson_assets(exp)" in source


# ---------------------------------------------------------------------------
# [relocation] on the prepared route: the corridor lift and its refusals
# ---------------------------------------------------------------------------

_RELOCATION_FOLLOW_TOML = """
[relocation]
enabled = true
grid_id = 2

[[relocation.move]]
at_seconds = 1800.0
di_parent_cells = 1
dj_parent_cells = 0
"""

_RELOCATION_BOUNDS_TOML = """
[relocation]
enabled = true
grid_id = 2
"""


def _with_relocation(config: Path, block: str) -> None:
    config.write_text(config.read_text(encoding="utf-8") + block,
                      encoding="utf-8")


def _preflight(prepared, receipt, config):
    return runner.preflight_prepared_tree(
        prepared_root=prepared,
        preparation_receipt_sha256=_sha(receipt),
        experiment_config=config,
        experiment_config_sha256=_sha(config),
    )


#: The corridor refusal matrix runs against BOTH hierarchy documents.
#:
#: ``_synthetic_prepared_tree`` writes the HRRR shape -- ``receipt.json``
#: carrying ``gpuwm-native-hrrr-hierarchy-direct-v1`` -- and until the
#: HRRR chain could seal a corridor these cases only ever proved the
#: runner's behaviour against a document no corridor-bearing bundle
#: actually used.  Now both chains emit, so both shapes are exercised:
#: the corridor is read out of whichever document matched the pinned
#: digest, and no branch of this preflight may key on which one it was.
_DOCUMENTS = ("hrrr", "gfs")


def _as_document(prepared: Path, receipt: Path, source: str) -> Path:
    """Re-shape the prepared bundle's top document for ``source``.

    Only the envelope changes -- filename, schema, status and how the
    initialization time is spelled.  ``hierarchy-artifacts/`` is written
    by one artifact writer for every source and is untouched, which is
    exactly why the corridor lives under it.
    """
    if source == "hrrr":
        return receipt
    preparation = json.loads(receipt.read_text(encoding="utf-8"))
    proof = {
        "schema": "gpuwm-gfs-native-hierarchy-proof-v2",
        "status": "READY_NOT_YET_STOCK_WRF_GATED",
        "forcing_times": [preparation["valid_time"]],
        "domain_count": preparation["domain_count"],
        "forcing_hours": preparation["forcing_hours"],
        "artifact_receipt": preparation["artifact_receipt"],
    }
    if "statics_corridor" in preparation:
        proof["statics_corridor"] = preparation["statics_corridor"]
    receipt.unlink()
    path = prepared / "proof.json"
    path.write_text(json.dumps(proof), encoding="utf-8")
    return path


def _bind_corridor(receipt: Path, corridor_set) -> None:
    preparation = json.loads(receipt.read_text(encoding="utf-8"))
    preparation["statics_corridor"] = corridor_set
    receipt.write_text(json.dumps(preparation), encoding="utf-8")


_D02_CORRIDOR_SET = {
    "schema": "gpuwm-statics-corridor-set-v1",
    "status": "READY",
    "domains": {"d02": {"cache": {"path": "d02.npz", "sha256": "e" * 64}}},
}


@pytest.mark.parametrize("source", _DOCUMENTS)
def test_follow_source_without_a_corridor_refuses_with_the_remedy(
        tmp_path, monkeypatch, source):
    """The corridor-less refusal survives, and names its remedy.

    Same sentence whichever chain prepared the bundle: the remedy is a
    flag both preparation doors now spell identically.
    """

    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    _with_relocation(config, _RELOCATION_FOLLOW_TOML)
    document = _as_document(prepared, receipt, source)
    with pytest.raises(ValueError) as excinfo:
        _preflight(prepared, document, config)
    message = str(excinfo.value)
    assert "cannot rebuild a relocated child's statics" in message
    assert "--statics-corridor" in message
    assert "gpuwm run" in message


@pytest.mark.parametrize("source", _DOCUMENTS)
def test_follow_source_with_an_uncovered_child_names_the_coverage(
        tmp_path, monkeypatch, source):
    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    _with_relocation(config, _RELOCATION_FOLLOW_TOML)
    _bind_corridor(receipt, {
        "schema": "gpuwm-statics-corridor-set-v1",
        "status": "READY",
        "domains": {"d09": {"cache": {"path": "d09.npz"}}},
    })
    document = _as_document(prepared, receipt, source)
    with pytest.raises(ValueError, match=r"covers only \['d09'\]"):
        _preflight(prepared, document, config)


@pytest.mark.parametrize("source", _DOCUMENTS)
def test_follow_source_with_a_verified_corridor_is_accepted(
        tmp_path, monkeypatch, source):
    """Corridor present and verified: the preflight resolves a runnable
    plan carrying the loaded corridor, through the real loader seam."""

    import gpuwm.static.corridor as corridor_module

    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    _with_relocation(config, _RELOCATION_FOLLOW_TOML)
    _bind_corridor(receipt, _D02_CORRIDOR_SET)
    document = _as_document(prepared, receipt, source)

    seen = {}
    stub = object()

    def fake_load(directory, *, expected_set_receipt, grid_id, child_dc,
                  parent_run, reference_grid):
        seen.update(directory=Path(directory), grid_id=grid_id,
                    expected=expected_set_receipt,
                    child=int(child_dc.grid_id),
                    parent_nx=int(parent_run.nx))
        return stub

    monkeypatch.setattr(corridor_module, "load_child_statics_corridor",
                        fake_load)
    inputs = _preflight(prepared, document, config)
    assert inputs.source == source
    assert inputs.statics_corridor is stub
    assert seen["grid_id"] == 2 and seen["child"] == 2
    assert seen["parent_nx"] == 100
    assert seen["expected"] == _D02_CORRIDOR_SET
    # The corridor sits at the same place in both bundles, because the
    # artifact writer that makes hierarchy-artifacts/ is one writer.
    assert seen["directory"] == (
        prepared / "hierarchy-artifacts" / "statics-corridor")
    assert inputs.statics_corridor_cache_path == (
        prepared / "hierarchy-artifacts" / "statics-corridor" / "d02.npz")


@pytest.mark.parametrize("source", _DOCUMENTS)
def test_a_failed_corridor_verification_refuses_never_runs_static(
        tmp_path, monkeypatch, source):
    """A corridor that fails digest verification is a LOUD refusal on
    the preflight path -- the run never degrades to a static nest."""

    import gpuwm.static.corridor as corridor_module

    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    _with_relocation(config, _RELOCATION_FOLLOW_TOML)
    _bind_corridor(receipt, _D02_CORRIDOR_SET)
    document = _as_document(prepared, receipt, source)

    def refusing_load(directory, **_kwargs):
        raise corridor_module.CorridorRefusal(
            "d02 statics corridor cache digest mismatch: planted")

    monkeypatch.setattr(corridor_module, "load_child_statics_corridor",
                        refusing_load)
    with pytest.raises(ValueError, match="digest mismatch"):
        _preflight(prepared, document, config)


def test_no_corridor_branch_of_the_preflight_reads_the_source():
    """Source-agnostic by construction, not by coincidence.

    The parametrized cases above show the two documents behaving alike
    today.  This is the reason they must: the resolved source is carried
    on the inputs for reporting, and nothing between the relocation gate
    and the corridor load consults it.
    """
    import inspect

    source = inspect.getsource(runner.preflight_prepared_tree)
    gate = source.index("relocation_follow")
    load = source.index("load_child_statics_corridor")
    assert "prepared_source" not in source[gate:load]


def test_bounds_only_relocation_still_passes_without_a_corridor(
        tmp_path, monkeypatch):
    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    _with_relocation(config, _RELOCATION_BOUNDS_TOML)
    inputs = _preflight(prepared, receipt, config)
    assert inputs.statics_corridor is None
    assert inputs.statics_corridor_cache_path is None

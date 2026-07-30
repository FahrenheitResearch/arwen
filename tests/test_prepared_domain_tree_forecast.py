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
    with pytest.raises(FileExistsError, match="refusing existing"):
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


def test_preflight_refuses_an_unrecognized_hierarchy_document(tmp_path, monkeypatch):
    prepared, receipt, config = _synthetic_prepared_tree(tmp_path, monkeypatch)
    preparation = json.loads(receipt.read_text(encoding="utf-8"))
    preparation["schema"] = "gpuwm-some-other-hierarchy-v9"
    receipt.write_text(json.dumps(preparation), encoding="utf-8")

    with pytest.raises(ValueError, match="no recognized hierarchy document"):
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
    with pytest.raises(ValueError, match="d02 cache domain config differs"):
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

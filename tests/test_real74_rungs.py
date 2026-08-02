"""CPU fixtures for the executable real74 N3/N4/N5 rung cases."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest

from gpuwm.io.wrfout import WrfoutWriter, wrfout_filename
from gpuwm.verify import nest_gates
from gpuwm.verify.cases import real74_chain, real74_d02, real74_n5b


def _tiny_frame(path: Path, *, t_value=1.0, refl_value=20.0) -> Path:
    fields = {
        "T": np.full((2, 3, 4), t_value, dtype=np.float32),
        "REFL_10CM": np.full((2, 3, 4), refl_value, dtype=np.float32),
    }
    with WrfoutWriter(path, nx=4, ny=3, nz=2, dx=1.0, dy=1.0) as writer:
        writer.write_frame("1974-04-03_13:00:00", fields)
    return path


def _mock_inventory(*names):
    members = [
        {"name": name, "dtype": "float32", "shape": [2, 3]}
        for name in sorted(names)]
    return {
        "schema": real74_d02.CANONICAL_INVENTORY_SCHEMA,
        "sha256": real74_d02._inventory_sha256(members),
        "array_count": len(members), "members": members,
    }


def _mock_state_sample(
        exp, domain, offset, inventory, *, label="shared", tick_den=1,
        dtbc_fp32_bits=0):
    grid_id = int(domain.removeprefix("d"))
    inventory = real74_d02._normalize_inventory(inventory)
    scalars = {
        "elapsed_seconds": float(offset),
        "dtbc_fp32_bits": dtbc_fp32_bits,
        "driver": None,
    }
    return {
        "schema": real74_d02.CANONICAL_STATE_SCHEMA,
        "domain": domain,
        "frame": wrfout_filename(
            exp.start_time + timedelta(seconds=offset), grid_id),
        "ticks": int(offset * tick_den), "tick_den": tick_den,
        "valid_seconds": float(offset),
        "sha256": real74_d02.stable_hash(
            [label, domain, offset, inventory["sha256"]]),
        "inventory_sha256": inventory["sha256"],
        "scalar_sha256": real74_d02.stable_hash(scalars),
        "array_count": inventory["array_count"],
        "field_order": [item["name"] for item in inventory["members"]],
        "inventory": inventory,
        "scalars": scalars,
    }


def _production_run(output_dir: Path, exp, payload: bytes,
                    checkpoint: Path | None = None,
                    offsets_by_domain=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for dc in exp.domains:
        offsets = ((real74_d02.RUN_SECONDS,) if offsets_by_domain is None
                   else offsets_by_domain[dc.grid_id])
        for offset in offsets:
            valid = exp.start_time + timedelta(seconds=offset)
            path = output_dir / wrfout_filename(valid, dc.grid_id)
            path.write_bytes(
                payload + bytes([dc.grid_id]) + int(offset).to_bytes(4, "little"))
            paths.append(path)
    inventory = _mock_inventory("state/u")
    hashes = []
    for dc in exp.domains:
        offsets = ((real74_d02.RUN_SECONDS,) if offsets_by_domain is None
                   else offsets_by_domain[dc.grid_id])
        domain = f"d{dc.grid_id:02d}"
        hashes.extend(
            _mock_state_sample(exp, domain, offset, inventory)
            for offset in offsets)
    checkpoints = []
    for dc in exp.domains:
        domain = f"d{dc.grid_id:02d}"
        sample = _mock_state_sample(
            exp, domain, real74_d02.RESTART_SPLIT_SECONDS, inventory)
        sample["frame"] = f"checkpoint@{real74_d02.RESTART_SPLIT_SECONDS}"
        checkpoints.append(sample)
    resume_inventories = {
        f"d{dc.grid_id:02d}": {
            "domain": f"d{dc.grid_id:02d}",
            "ticks": real74_d02.RESTART_SPLIT_SECONDS,
            "tick_den": 1,
            "valid_seconds": float(real74_d02.RESTART_SPLIT_SECONDS),
            "inventory": inventory,
        }
        for dc in exp.domains}
    return real74_d02.ProductionRun(
        output_dir=output_dir, wrfout_paths=tuple(paths),
        checkpoint=checkpoint,
        completed_seconds=(real74_d02.RESTART_SPLIT_SECONDS
                           if checkpoint is not None else real74_d02.RUN_SECONDS),
        execution={
            "canonical_state_hashes": hashes,
            "canonical_checkpoint_state_hashes": checkpoints,
            "canonical_resume_inventories": resume_inventories,
        }, timing={}, memory={})


def _mock_state_schedule(
        domain, *, tick_stop, history_interval_ticks, tick_den=1,
        tick_start=0):
    return {
        "domain": domain,
        "tick_start": tick_start,
        "tick_stop": tick_stop,
        "tick_den": tick_den,
        "history_interval_ticks": history_interval_ticks,
        "expected_samples": len(range(
            tick_start, tick_stop + 1, history_interval_ticks)),
    }


def _ratchet_provenance(domain_count=2, *, tick_stop=None, tick_den=1):
    if tick_stop is None:
        tick_stop = real74_d02.RUN_SECONDS * tick_den
    return {
        "evaluator_commit": real74_d02._git_commit(),
        "experiment_fingerprint": "f" * 64,
        "domain_ids": list(range(1, domain_count + 1)),
        "tick_start": 0, "tick_stop": tick_stop,
        "tick_den": tick_den,
    }


def _publish_mock_ratchet(tmp_path):
    exp, _data = real74_d02.construct_rung_case(2)
    inventory = _mock_inventory("state/u")
    d02_hashes = [
        _mock_state_sample(exp, "d02", offset, inventory)
        for offset in (0, 900)]
    d01_hashes = [
        _mock_state_sample(exp, "d01", offset, inventory)
        for offset in (0, 900)]
    source = tmp_path / "source"
    source.mkdir()
    frames = []
    for sample in d02_hashes:
        frame = source / sample["frame"]
        frame.write_bytes(b"frame")
        frames.append(frame)
    root = tmp_path / "rungs"
    d02_schedule = _mock_state_schedule(
        "d02", tick_stop=900, history_interval_ticks=900)
    state_hashes = [*d01_hashes, *d02_hashes]
    real74_d02.publish_ratchet(
        root, "N3", "d02", frames,
        provenance=_ratchet_provenance(tick_stop=900),
        expected_state_domains=("d01", "d02"),
        state_hashes=state_hashes,
        state_hash_schedules=(
            _mock_state_schedule(
                "d01", tick_stop=900, history_interval_ticks=900),
            d02_schedule,
        ))
    return exp, root, frames, state_hashes, d02_schedule


def _no_blowup_summary(substeps=300):
    return {
        "boundary_w_max_ms": 3.0,
        "interior_w_max_ms": 2.0,
        "boundary_zone_blowup": False,
        "dynamics_substeps": substeps,
    }


def test_gate_consumption_inventory_is_exact_and_uses_gate_lookup(monkeypatch):
    expected = {
        "N3": real74_d02.N3_METRICS,
        "N4": real74_chain.N4_METRICS,
        "N5": real74_chain.N5_METRICS,
    }
    calls = []
    original = nest_gates.gate

    def tracking_gate(milestone, metric):
        calls.append((milestone, metric))
        return original(milestone, metric)

    monkeypatch.setattr(nest_gates, "gate", tracking_gate)
    for milestone, metrics in expected.items():
        assert tuple(g.metric for g in real74_d02.gate_records(
            milestone, metrics)) == metrics
        assert metrics == tuple(g.metric for g in nest_gates.gates_for(milestone))
    assert calls == [(milestone, metric)
                     for milestone, metrics in expected.items()
                     for metric in metrics]


def test_numeric_comparator_reads_the_gate_record_threshold():
    original = nest_gates.gate("N3", "d02_t500_rmse_k")
    tightened = replace(original, threshold=0.125)
    assert real74_d02.gate_result(tightened, value=0.125)["passed"] is True
    assert real74_d02.gate_result(tightened, value=0.126)["passed"] is False


def test_d01_variable_bytes_allow_only_three_ratified_exceptions(tmp_path):
    # F20 amendment (2026-07-17): TITLE joined REFL_10CM and
    # GPUWM_WRITE_COMPLETE as a ratified exception -- value excluded,
    # presence required, both values recorded in the evidence.
    baseline = _tiny_frame(tmp_path / "baseline", t_value=1.0, refl_value=20.0)
    candidate = _tiny_frame(tmp_path / "candidate", t_value=1.0, refl_value=60.0)
    with netCDF4.Dataset(baseline, "r+") as dataset:
        dataset.delncattr("GPUWM_WRITE_COMPLETE")
        dataset.setncattr("TITLE", "frozen Phase-3 baseline title")

    result = real74_d02.compare_d01_phase4_frame(candidate, baseline)
    assert result["passed"] is True
    assert real74_d02.compare_files_exact(candidate, candidate)["passed"] is True
    assert real74_d02.compare_files_exact(candidate, baseline)["passed"] is False
    assert tuple(result["ratified_exceptions"]) == (
        "REFL_10CM", "GPUWM_WRITE_COMPLETE", "TITLE")
    assert result["publication_attribute"] == {
        "name": "GPUWM_WRITE_COMPLETE", "candidate": 1,
        "candidate_present": True, "candidate_required_value": 1,
        "baseline": None, "baseline_value_excluded": True}
    assert result["title_exception"]["value_excluded"] is True
    assert result["title_exception"]["baseline"] == (
        "frozen Phase-3 baseline title")
    assert isinstance(result["title_exception"]["candidate"], str)

    with netCDF4.Dataset(candidate, "r+") as dataset:
        dataset.variables["T"][0, 0, 0, 0] = np.float32(2.0)
    failed = real74_d02.compare_d01_phase4_frame(candidate, baseline)
    assert failed["passed"] is False
    assert any(item.startswith("T:") for item in failed["mismatches"])


@pytest.mark.parametrize(
    ("attack", "mismatch"),
    (("new_global", "global attribute inventory differs"),
     ("missing_title", "TITLE global attribute is missing or empty"),
     ("empty_title", "TITLE global attribute is missing or empty"),
     ("scaled_t", "T variable attribute inventory differs"),
     ("missing_write_complete", "GPUWM_WRITE_COMPLETE global attribute")),
)
def test_d01_attribute_closure_rejects_review_attacks(
        tmp_path, attack, mismatch):
    # F20: a DIFFERING TITLE value is a ratified exception (covered by the
    # three-exceptions test above); a MISSING or EMPTY candidate TITLE
    # still fails -- the exception excludes the value, not the presence.
    baseline = _tiny_frame(
        tmp_path / f"baseline-{attack}", t_value=1.0, refl_value=20.0)
    candidate = _tiny_frame(
        tmp_path / f"candidate-{attack}", t_value=1.0, refl_value=60.0)
    with netCDF4.Dataset(baseline, "r+") as dataset:
        dataset.delncattr("GPUWM_WRITE_COMPLETE")
    with netCDF4.Dataset(candidate, "r+") as dataset:
        if attack == "new_global":
            dataset.setncattr("SNEAKY_NEW_GLOBAL_ATTR", "attack")
        elif attack == "missing_title":
            dataset.delncattr("TITLE")
        elif attack == "empty_title":
            dataset.setncattr("TITLE", "   ")
        elif attack == "scaled_t":
            dataset.variables["T"].setncattr("scale_factor", 2.0)
        elif attack == "missing_write_complete":
            dataset.delncattr("GPUWM_WRITE_COMPLETE")
    result = real74_d02.compare_d01_phase4_frame(candidate, baseline)
    assert result["passed"] is False
    assert any(mismatch in item for item in result["mismatches"])


def test_production_binds_root_boundary_clock_at_sanctioned_sites_only():
    """Davies clock bind (2026-07-28, retires the F20 adjudication): the
    production tree build MUST bind the root's external Davies clock so
    every root boundary launch consumes WRF's post-increment
    dtbc_launch_fp32 (solve_em.F:371-372), and the N5S manual builder
    must bind its own root.  The F20-era review hardening survives as a
    positive inventory: sweeping the whole gpuwm package for the
    identifier -- names, attributes, string literals (getattr evasion),
    and import aliases -- the ONLY modules that may mention it are the
    definition and the adjudicated callers.  The pre-bind Phase-4
    anchor bytes encode the retired elapsed-based semantics; the
    N-series ratchets regenerate against the seam-closure anchor epoch
    (PROVENANCE.md 'Root external-boundary dtbc').

    Adjudicated 2026-07-30: offline_child_run.py.  The standalone
    offline child is a boundary-consumer root in its own process -- it
    binds a DomainClock to its external LBC mirror so child boundary
    launches take the same post-increment dtbc recurrence as the
    production tree root.  Same semantics, third sanctioned site; the
    v1.1 offline-nest lane added the caller without updating this
    inventory.

    Adjudicated 2026-08-01: prepared_domain_tree_forecast.py.  The prepared
    hierarchy runner also constructs DomainNodes without build_experiment,
    so it binds its root's already-attached external mirror before restart
    validation or the first solve.  Same semantics, fourth sanctioned
    site; the sealed-extension route makes this path production-visible."""
    import ast

    package_root = Path(real74_d02.REPOSITORY_ROOT) / "gpuwm"
    sanctioned = {
        package_root / "ingest" / "lateral_bc.py",
        package_root / "core" / "model.py",
        package_root / "verify" / "cases" / "real74_n5s.py",
        package_root / "offline_child_run.py",
        package_root / "prepared_domain_tree_forecast.py",
    }
    referencing = set()
    for path in sorted(package_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            hit = (
                (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and node.name == "bind_lateral_boundary_clock")
                or (isinstance(node, ast.Name)
                    and node.id == "bind_lateral_boundary_clock")
                or (isinstance(node, ast.Attribute)
                    and node.attr == "bind_lateral_boundary_clock")
                or (isinstance(node, ast.Constant)
                    and node.value == "bind_lateral_boundary_clock")
                or (isinstance(node, ast.ImportFrom)
                    and any(alias.name == "bind_lateral_boundary_clock"
                            for alias in node.names)))
            if hit:
                referencing.add(path)
                break
    missing = sorted(str(p.relative_to(package_root))
                     for p in sanctioned - referencing)
    extra = sorted(str(p.relative_to(package_root))
                   for p in referencing - sanctioned)
    assert not missing and not extra, (
        "bind_lateral_boundary_clock production caller inventory drifted "
        f"(missing {missing}, unadjudicated {extra}); the root bind and "
        "the adjudicated direct builders are mandatory, anything else needs "
        "adjudication")


def test_output_frame_static_and_surface_metric_plumbing(tmp_path):
    shape = (12, 12)

    def write(path, offset):
        with netCDF4.Dataset(path, "w") as dataset:
            dataset.createDimension("Time", 1)
            dataset.createDimension("south_north", shape[0])
            dataset.createDimension("west_east", shape[1])
            for name, base in (("HGT", 100.0), ("MUB", 5000.0),
                               ("T2", 290.0), ("TSK", 292.0)):
                var = dataset.createVariable(
                    name, "f4", ("Time", "south_north", "west_east"))
                var[0] = np.full(shape, base + offset[name], np.float32)

    reference = tmp_path / "reference.nc"
    candidate = tmp_path / "candidate.nc"
    write(reference, {name: 0.0 for name in ("HGT", "MUB", "T2", "TSK")})
    write(candidate, {"HGT": 0.25, "MUB": -2.0, "T2": 1.5, "TSK": -0.5})
    static = real74_d02.output_static_recheck(
        candidate, reference, spec_bdy_width=1, blend_width=1)
    assert static == {"hgt_m": 0.25, "mub_pa": 2.0}
    diagnostic = real74_d02.blend_zone_surface_bias(
        candidate, reference, spec_bdy_width=1, blend_width=1)
    assert diagnostic["T2"]["mean_bias_k"] == 1.5
    assert diagnostic["TSK"]["mean_bias_k"] == -0.5


def test_statistical_metric_plumbing_uses_known_arrays(monkeypatch, tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")
    base = np.arange(144, dtype=np.float64).reshape(12, 12)
    actual = {
        "mslp": base * 2.0 + 3.0,
        "levels": {
            500: {"temperature": base + 1.0},
            850: {"temperature": base - 2.0},
        },
    }
    oracle = {
        "mslp": base,
        "levels": {
            500: {"temperature": base},
            850: {"temperature": base},
        },
    }
    monkeypatch.setattr(
        real74_d02.weather_metrics, "wrf_diagnostics",
        lambda path: actual if Path(path) == candidate else oracle)
    monkeypatch.setattr(
        real74_d02, "_composite_reflectivity",
        lambda path: np.full((12, 12), 40.0 if Path(path) == candidate else 40.0))
    verdicts = {"d02_refl_10cm_structure": {"passed": True,
                                             "reviewer": "fixture"}}
    results = real74_d02.score_statistical_frame(
        "N3", "d02", candidate, reference, dx_m=3000.0,
        run_summary=_no_blowup_summary(),
        verdicts=verdicts)
    assert results["d02_mslp_pattern_correlation"]["value"] == pytest.approx(1.0)
    assert results["d02_t500_rmse_k"]["value"] == pytest.approx(1.0)
    assert results["d02_t850_rmse_k"]["value"] == pytest.approx(2.0)
    assert results["d02_refl_10cm_fss"]["passed"] is True
    scores = results["d02_refl_10cm_fss"]["evidence"]["scores"]
    assert all(set(row) == {
        "event_dbz", "radius_km", "minimum", "value", "passed"}
        for row in scores)


def test_composite_reflectivity_rejects_partially_nonfinite_column(monkeypatch):
    reflectivity = np.full((2, 12, 12), 40.0, dtype=np.float64)
    reflectivity[0, 6, 6] = np.nan
    monkeypatch.setattr(
        real74_d02, "_read_field", lambda _path, _name: reflectivity)

    with pytest.raises(ValueError, match="REFL_10CM.*non-finite"):
        real74_d02._composite_reflectivity(Path("candidate"))


def test_w_cfl_evidence_rejects_partially_nonfinite_output(monkeypatch):
    fields = {
        "W": np.ones((2, 2, 2), dtype=np.float64),
        "U": np.ones((2, 2, 3), dtype=np.float64),
        "V": np.ones((2, 3, 2), dtype=np.float64),
        "PH": (np.arange(3, dtype=np.float64)[:, None, None]
               * 100.0 * 9.80665 + np.zeros((3, 2, 2))),
        "PHB": np.zeros((3, 2, 2), dtype=np.float64),
    }
    fields["W"][0, 0, 0] = np.nan
    monkeypatch.setattr(
        real74_chain.shared, "_read_field",
        lambda _path, name: fields[name])

    evidence = real74_chain.w_cfl_evidence(
        Path("candidate"), dx_m=1000.0, dt_s=1.0)

    assert evidence["finite"] is False
    assert np.isnan(evidence["w_max_ms"])


def test_updraft_percentiles_reject_partially_nonfinite_candidate(monkeypatch):
    candidate = Path("candidate")
    reference = Path("reference")
    actual = np.full((2, 12, 12), 20.0, dtype=np.float64)
    actual[0, 6, 6] = np.nan
    oracle = np.full_like(actual, 20.0)
    monkeypatch.setattr(
        real74_chain.shared, "_read_field",
        lambda path, _name: actual if Path(path) == candidate else oracle)

    passed, evidence = real74_chain.updraft_percentile_evidence(
        candidate, reference)

    assert passed is False
    assert "non-finite" in evidence["error"]


def _score_f27_fss20(monkeypatch, tmp_path, *, domain, value):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")
    base = np.arange(144, dtype=np.float64).reshape(12, 12)
    diagnostics = {
        "mslp": base,
        "levels": {
            500: {"temperature": base},
            850: {"temperature": base},
        },
    }
    monkeypatch.setattr(
        real74_d02.weather_metrics, "wrf_diagnostics", lambda _path: diagnostics)
    monkeypatch.setattr(
        real74_d02, "_composite_reflectivity",
        lambda _path: np.full((12, 12), 50.0))
    monkeypatch.setattr(
        real74_d02, "fractions_skill_score",
        lambda *_args, event_threshold, **_kwargs:
            value if event_threshold == 20.0 else 1.0)
    milestone = {"d02": "N3", "d03": "N4", "d04": "N5"}[domain]
    return real74_d02.score_statistical_frame(
        milestone, domain, candidate, reference, dx_m=1000.0,
        run_summary=_no_blowup_summary(),
        verdicts={f"{domain}_refl_10cm_structure": {"passed": True}})


def test_f27_d03_below_envelope_is_documented_deficiency(monkeypatch, tmp_path):
    results = _score_f27_fss20(
        monkeypatch, tmp_path, domain="d03", value=0.70)

    fss = results["d03_refl_10cm_fss"]
    assert fss["passed"] is True
    row20, row30, row40 = fss["evidence"]["scores"]
    assert row20 == {
        "event_dbz": 20.0,
        "radius_km": 5.0,
        "minimum": 0.8558,
        "value": 0.70,
        "passed": True,
        "documented_deficiency": True,
        "envelope_minimum": 0.8558,
        "adjudication": "f25-envelope-standing-deficiency-f27",
    }
    assert set(row30) == {
        "event_dbz", "radius_km", "minimum", "value", "passed"}
    assert set(row40) == {
        "event_dbz", "radius_km", "minimum", "value", "passed"}
    assert (row30["minimum"], row40["minimum"]) == (0.80, 0.70)


@pytest.mark.parametrize("value", [0.8558, 0.86, 0.90])
def test_f27_d03_at_or_above_envelope_reverts_to_normal_row(
        monkeypatch, tmp_path, value):
    results = _score_f27_fss20(
        monkeypatch, tmp_path, domain="d03", value=value)

    fss = results["d03_refl_10cm_fss"]
    assert fss["passed"] is True
    row20 = fss["evidence"]["scores"][0]
    assert row20 == {
        "event_dbz": 20.0,
        "radius_km": 5.0,
        "minimum": 0.8558,
        "value": value,
        "passed": True,
    }


def test_f27_d02_fss20_remains_blocking_at_original_minimum(
        monkeypatch, tmp_path):
    results = _score_f27_fss20(
        monkeypatch, tmp_path, domain="d02", value=0.85)

    fss = results["d02_refl_10cm_fss"]
    assert fss["passed"] is False
    row20 = fss["evidence"]["scores"][0]
    assert row20 == {
        "event_dbz": 20.0,
        "radius_km": 5.0,
        "minimum": 0.90,
        "value": 0.85,
        "passed": False,
    }


def test_f24_degenerate_fss_rows_are_provisional_passes(monkeypatch, tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")
    base = np.arange(144, dtype=np.float64).reshape(12, 12)
    diagnostics = {
        "mslp": base,
        "levels": {
            500: {"temperature": base},
            850: {"temperature": base},
        },
    }
    monkeypatch.setattr(
        real74_d02.weather_metrics, "wrf_diagnostics", lambda _path: diagnostics)
    monkeypatch.setattr(
        real74_d02, "_composite_reflectivity",
        lambda path: np.full(
            (12, 12), 50.0 if Path(path) == candidate else 0.0))

    results = real74_d02.score_statistical_frame(
        "N3", "d02", candidate, reference, dx_m=3000.0,
        run_summary=_no_blowup_summary(),
        verdicts={"d02_refl_10cm_structure": {"passed": True}})
    fss = results["d02_refl_10cm_fss"]
    assert fss["passed"] is True
    for row in fss["evidence"]["scores"]:
        assert set(row) == {
            "event_dbz", "radius_km", "minimum", "value", "passed",
            "degenerate", "candidate_coverage", "reference_coverage",
            "adjudication"}
        assert row["passed"] is True
        assert row["value"] < row["minimum"]
        assert row["degenerate"] is True
        assert row["candidate_coverage"] == 1.0
        assert row["reference_coverage"] == 0.0
        assert row["adjudication"] == "held-for-ensemble-envelope-f24"


def test_f24_identically_empty_fss_keeps_registered_value(monkeypatch, tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")
    base = np.arange(144, dtype=np.float64).reshape(12, 12)
    diagnostics = {
        "mslp": base,
        "levels": {
            500: {"temperature": base},
            850: {"temperature": base},
        },
    }
    monkeypatch.setattr(
        real74_d02.weather_metrics, "wrf_diagnostics", lambda _path: diagnostics)
    monkeypatch.setattr(
        real74_d02, "_composite_reflectivity",
        lambda _path: np.zeros((12, 12), dtype=np.float64))

    results = real74_d02.score_statistical_frame(
        "N3", "d02", candidate, reference, dx_m=3000.0,
        run_summary=_no_blowup_summary(),
        verdicts={"d02_refl_10cm_structure": {"passed": True}})
    scores = results["d02_refl_10cm_fss"]["evidence"]["scores"]
    assert all(row["degenerate"] is True and row["value"] == 1.0
               for row in scores)


def test_f24_degenerate_formula_failure_records_null_fss(monkeypatch, tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")
    base = np.arange(144, dtype=np.float64).reshape(12, 12)
    diagnostics = {
        "mslp": base,
        "levels": {
            500: {"temperature": base},
            850: {"temperature": base},
        },
    }
    monkeypatch.setattr(
        real74_d02.weather_metrics, "wrf_diagnostics", lambda _path: diagnostics)
    monkeypatch.setattr(
        real74_d02, "_composite_reflectivity",
        lambda _path: np.zeros((12, 12), dtype=np.float64))
    monkeypatch.setattr(
        real74_d02, "fractions_skill_score", lambda *_args, **_kwargs: float("nan"))

    results = real74_d02.score_statistical_frame(
        "N3", "d02", candidate, reference, dx_m=3000.0,
        run_summary=_no_blowup_summary(),
        verdicts={"d02_refl_10cm_structure": {"passed": True}})
    rows = results["d02_refl_10cm_fss"]["evidence"]["scores"]
    assert all(row["passed"] is True and row["fss"] is None
               and "value" not in row for row in rows)


def test_f24_empty_fss_interior_fails_loudly(monkeypatch, tmp_path):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.write_bytes(b"candidate")
    reference.write_bytes(b"reference")
    base = np.arange(144, dtype=np.float64).reshape(12, 12)
    diagnostics = {
        "mslp": base,
        "levels": {
            500: {"temperature": base},
            850: {"temperature": base},
        },
    }
    monkeypatch.setattr(
        real74_d02.weather_metrics, "wrf_diagnostics", lambda _path: diagnostics)
    monkeypatch.setattr(
        real74_d02, "_composite_reflectivity",
        lambda _path: np.empty((0, 0), dtype=np.float64))

    results = real74_d02.score_statistical_frame(
        "N3", "d02", candidate, reference, dx_m=333.0,
        run_summary=_no_blowup_summary(),
        verdicts={"d02_refl_10cm_structure": {"passed": True}})

    fss = results["d02_refl_10cm_fss"]
    assert fss["passed"] is False
    assert fss["evidence"]["scores"] == [
        {"error": "FSS reflectivity has an empty registered interior"}]
    with pytest.raises(ValueError, match="empty registered interior"):
        real74_d02._event_coverage(np.empty((0, 0)), 20.0)


def _f24_ensemble_frames(tmp_path, values, *, domain="d04"):
    frames = []
    name = f"wrfout_{domain}_1974-04-03_13_15_00"
    for index, value in enumerate(values):
        member = tmp_path / f"member-{index}"
        member.mkdir()
        frame = member / name
        fields = {
            "T": np.ones((1, 32, 32), dtype=np.float32),
            "REFL_10CM": np.full((1, 32, 32), value, dtype=np.float32),
        }
        with WrfoutWriter(
                frame, nx=32, ny=32, nz=1, dx=333.0, dy=333.0) as writer:
            writer.write_frame("1974-04-03_13:15:00", fields)
        frames.append(frame)
    return frames


def _f24_held_row(event_dbz=40.0):
    return {
        "event_dbz": event_dbz,
        "radius_km": 5.0,
        "minimum": 0.70,
        "passed": True,
        "degenerate": True,
        "candidate_coverage": 0.0,
        "reference_coverage": 0.0,
        "fss": None,
        "adjudication": "held-for-ensemble-envelope-f24",
    }


def test_f24_ensemble_envelope_confirms_majority_degenerate(tmp_path):
    frames = _f24_ensemble_frames(tmp_path, (20.0, 20.0, 50.0))
    result = real74_d02.ensemble_envelope_adjudication(
        [_f24_held_row()], frames, domain="d04", dx_m=333.0)

    row = result["rows"][0]
    assert row["verdict"] == "confirmed-degenerate"
    assert row["confirmed_degenerate"] is True
    assert row["revoked"] is False
    assert row["degenerate_member_count"] == 2
    assert row["majority_required"] == 2
    assert [item["coverage"] for item in row["member_coverages"]] == [
        0.0, 0.0, 1.0]
    envelope = row["pairwise_fss_envelope"]
    assert envelope["min"] == 0.0
    assert envelope["max"] == 1.0
    assert len(envelope["values"]) == len(envelope["pairs"]) == 3
    json.dumps(result, allow_nan=False)


def test_f24_ensemble_envelope_revokes_without_degenerate_majority(tmp_path):
    frames = _f24_ensemble_frames(tmp_path, (20.0, 50.0, 50.0))
    result = real74_d02.ensemble_envelope_adjudication(
        [_f24_held_row()], frames, domain="d04", dx_m=333.0)

    row = result["rows"][0]
    assert row["verdict"] == "revoked"
    assert row["confirmed_degenerate"] is False
    assert row["revoked"] is True
    assert row["degenerate_member_count"] == 1
    assert len(row["pairwise_fss_envelope"]["values"]) == 3
    json.dumps(result, allow_nan=False)


def test_f24_ensemble_envelope_requires_three_members(tmp_path):
    frames = _f24_ensemble_frames(tmp_path, (20.0, 50.0))
    with pytest.raises(ValueError, match="at least 3"):
        real74_d02.ensemble_envelope_adjudication(
            [_f24_held_row()], frames, domain="d04", dx_m=333.0)


@pytest.mark.parametrize(
    ("held", "message"),
    (
        ({
            "event_dbz": 40.0,
            "candidate_coverage": 1.0,
            "reference_coverage": 1.0,
        }, "degenerate is True"),
        ({**_f24_held_row(), "degenerate": False}, "degenerate is True"),
        ({**_f24_held_row(), "adjudication": "unregistered"},
         "registered adjudication marker"),
        ({**_f24_held_row(), "candidate_coverage": 1.0,
          "reference_coverage": 1.0}, "below the registered event floor"),
    ),
)
def test_f24_ensemble_envelope_rejects_non_held_rows(
        tmp_path, held, message):
    frames = _f24_ensemble_frames(tmp_path, (20.0, 20.0, 50.0))
    with pytest.raises(ValueError, match=message):
        real74_d02.ensemble_envelope_adjudication(
            [held], frames, domain="d04", dx_m=333.0)


def test_f24_ensemble_validates_all_held_rows_before_member_voting(
        monkeypatch, tmp_path):
    frames = _f24_ensemble_frames(tmp_path, (20.0, 20.0, 50.0))
    valid = {**_f24_held_row(20.0), "radius_km": 5.0}
    invalid = {**_f24_held_row(40.0), "adjudication": "unregistered"}

    def forbidden_vote(*_args, **_kwargs):
        raise AssertionError("member voting started before row validation")

    monkeypatch.setattr(real74_d02, "_event_coverage", forbidden_vote)
    with pytest.raises(ValueError, match="registered adjudication marker"):
        real74_d02.ensemble_envelope_adjudication(
            [valid, invalid], frames, domain="d04", dx_m=333.0)


def test_f24_ensemble_rejects_nan_inherited_extra(tmp_path):
    frames = _f24_ensemble_frames(tmp_path, (20.0, 20.0, 50.0))
    held = {**_f24_held_row(), "inherited_extra": float("nan")}

    with pytest.raises(ValueError, match="finite numeric evidence"):
        real74_d02.ensemble_envelope_adjudication(
            [held], frames, domain="d04", dx_m=333.0)


def test_restart_split_calls_30_min_prefix_then_resume_and_compares_domains(
        tmp_path):
    exp, data = real74_d02.construct_rung_case(2)
    full_offsets = {
        dc.grid_id: tuple(range(
            0, real74_d02.RUN_SECONDS + 1, int(dc.history_interval_s)))
        for dc in exp.domains}
    prefix_offsets = {
        grid_id: tuple(offset for offset in offsets
                       if offset <= real74_d02.RESTART_SPLIT_SECONDS)
        for grid_id, offsets in full_offsets.items()}
    resume_offsets = {
        grid_id: tuple(offset for offset in offsets
                       if offset > real74_d02.RESTART_SPLIT_SECONDS)
        for grid_id, offsets in full_offsets.items()}
    straight = _production_run(
        tmp_path / "straight", exp, b"same", offsets_by_domain=full_offsets)
    calls = []

    def executor(_exp, _data, output_dir, *, restart=None,
                 stop_at_checkpoint_seconds=None):
        calls.append((Path(output_dir).name, restart,
                      stop_at_checkpoint_seconds))
        if stop_at_checkpoint_seconds is not None:
            checkpoint = Path(output_dir) / "checkpoint.npz"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(b"checkpoint")
            return _production_run(
                Path(output_dir), exp, b"same", checkpoint,
                offsets_by_domain=prefix_offsets)
        resumed = _production_run(
            Path(output_dir), exp, b"same",
            offsets_by_domain=resume_offsets)
        return resumed

    evidence = real74_d02.restart_split_stage(
        exp, data, straight=straight, work_dir=tmp_path / "split",
        executor=executor)
    assert evidence["passed"] is True
    assert calls[0][0] == "prefix"
    assert calls[0][2] == real74_d02.RESTART_SPLIT_SECONDS
    assert calls[1][0] == "resumed"
    assert Path(calls[1][1]).name == "checkpoint.npz"
    assert set(evidence["comparisons"]) == {"d01", "d02"}
    assert len(evidence["comparisons"]["d01"]["frames"]) == 2
    assert len(evidence["comparisons"]["d02"]["frames"]) == 6
    assert evidence["state_inventory_comparisons"]["passed"] is True


def test_per_frame_inventory_growth_is_monotonic_and_allowlisted():
    small = _mock_inventory("state/u")
    grown = _mock_inventory(
        "state/u", "scratch/nest_u_bxs", "scratch/refl_10cm",
        "scratch/lbc_weights_0",
        "scratch/mp_rainnc", "cumulus/w0avg")
    assert real74_d02.validate_inventory_growth(
        None, small, domain="d02", ticks=0) == small
    assert real74_d02.validate_inventory_growth(
        small, grown, domain="d02", ticks=1) == grown


def test_per_frame_inventory_growth_outside_allowlist_raises():
    small = _mock_inventory("state/u")
    drifted = _mock_inventory("state/u", "fields/new_lazy_state")
    with pytest.raises(RuntimeError, match="outside the lazy-member allowlist"):
        real74_d02.validate_inventory_growth(
            small, drifted, domain="d02", ticks=1)


@pytest.mark.parametrize("member", (
    "driver/cumulus_tendencies/rqs",
    "driver/pbl_tendencies/rqs",
))
def test_held_phase_tendencies_are_canonical_not_lazy(member):
    small = _mock_inventory("state/u")
    drifted = _mock_inventory("state/u", member)
    with pytest.raises(RuntimeError, match="outside the lazy-member allowlist"):
        real74_d02.validate_inventory_growth(
            small, drifted, domain="d01", ticks=900)


@pytest.mark.parametrize("counterfeit", (
    "scratch/nest_unrelated_debug_counter",
    "scratch/lbc_weights_unrelated_debug_counter",
    "scratch/refl_unrelated_debug_counter",
    "scratch/refl_t",
))
def test_lazy_inventory_prefix_counterfeits_raise(counterfeit):
    small = _mock_inventory("state/u")
    drifted = _mock_inventory("state/u", counterfeit)
    with pytest.raises(RuntimeError, match="outside the lazy-member allowlist"):
        real74_d02.validate_inventory_growth(
            small, drifted, domain="d02", ticks=1)


def test_cross_run_equal_tick_inventory_mismatch_fails_ratchet_before_hash(
        tmp_path):
    exp, _data = real74_d02.construct_rung_case(2)
    baseline = [
        _mock_state_sample(exp, "d02", 0, _mock_inventory("state/u")),
        _mock_state_sample(exp, "d02", 900, _mock_inventory("state/u")),
    ]
    candidate = [dict(item) for item in baseline]
    candidate[1] = _mock_state_sample(
        exp, "d02", 900,
        _mock_inventory("state/u", "scratch/refl_10cm"))
    # Make the state digest look equal: the inventory gate must still fail
    # first and report that the hash was not compared.
    candidate[1]["sha256"] = baseline[1]["sha256"]
    sources = []
    currents = []
    for sample in baseline:
        source = tmp_path / "source" / sample["frame"]
        current = tmp_path / "candidate" / sample["frame"]
        source.parent.mkdir(exist_ok=True)
        current.parent.mkdir(exist_ok=True)
        source.write_bytes(b"same frame bytes")
        current.write_bytes(source.read_bytes())
        sources.append(source)
        currents.append(current)
    root = tmp_path / "rungs"
    d01_controls = [
        _mock_state_sample(exp, "d01", offset, _mock_inventory("state/u"))
        for offset in (0, 900)]
    real74_d02.publish_ratchet(
        root, "N3", "d02", sources,
        provenance=_ratchet_provenance(tick_stop=900),
        expected_state_domains=("d01", "d02"),
        state_hashes=(*d01_controls, *baseline),
        state_hash_schedules=(
            _mock_state_schedule(
                "d01", tick_stop=900, history_interval_ticks=900),
            _mock_state_schedule(
                "d02", tick_stop=900, history_interval_ticks=900),))
    manifest = real74_d02.load_ratchet(root, "N3", domain="d02")
    ratchet = real74_d02.compare_ratchet_frames(
        currents, root, manifest, "d02",
        candidate_state_hashes=candidate)
    comparison = ratchet["state_hashes"]
    assert ratchet["passed"] is False
    assert comparison["passed"] is False
    assert comparison["samples"][1]["inventory_equal"] is False
    assert comparison["samples"][1]["hash_compared"] is False
    assert comparison["samples"][1]["hash_equal"] is None


def test_cross_run_equal_inventory_hash_mismatch_fails():
    exp, _data = real74_d02.construct_rung_case(2)
    inventory = _mock_inventory("state/u")
    baseline = _mock_state_sample(exp, "d02", 900, inventory)
    candidate = _mock_state_sample(
        exp, "d02", 900, inventory, label="drifted")
    comparison = real74_d02.compare_state_hash_samples(
        [candidate], [baseline], "d02")
    row = comparison["samples"][0]
    assert comparison["passed"] is False
    assert row["inventory_equal"] is True
    assert row["hash_compared"] is True
    assert row["hash_equal"] is False


def test_cross_run_different_tick_den_identical_trajectory_passes_exactly():
    exp, _data = real74_d02.construct_rung_case(2)
    inventory = _mock_inventory("state/u")
    baselines = [
        _mock_state_sample(exp, "d03", offset, inventory, tick_den=1)
        for offset in (0, 900, 1800)]
    candidates = [
        _mock_state_sample(exp, "d03", offset, inventory, tick_den=3)
        for offset in (0, 900, 1800)]
    comparison = real74_d02.compare_state_hash_samples(
        candidates, baselines, "d03")
    assert comparison["passed"] is True
    assert comparison["candidate_only_ticks"] == []
    assert comparison["baseline_only_ticks"] == []
    assert all(row["hash_compared"] for row in comparison["samples"])
    assert comparison["samples"][1]["ticks"] == 2700
    assert comparison["samples"][1]["baseline_ticks"] == 900


def test_dtbc_fp32_bits_are_canonical_and_compared(monkeypatch):
    from gpuwm.io import restart as restart_io

    monkeypatch.setattr(restart_io, "state_manifest", lambda _state: {})
    monkeypatch.setattr(restart_io, "_scratch_manifest", lambda _state: {})
    state = SimpleNamespace(elapsed_seconds=900.0, physics=None, _scratch={})
    clock = SimpleNamespace(dtbc_fp32=np.float32(5.0))
    first = real74_d02.canonical_state_digest(state, clock)
    clock.dtbc_fp32 = np.float32(10.0 / 3.0)
    second = real74_d02.canonical_state_digest(state, clock)
    assert first["scalars"]["dtbc_fp32_bits"] == int(
        np.float32(5.0).view(np.uint32))
    assert first["scalar_sha256"] != second["scalar_sha256"]
    assert first["sha256"] != second["sha256"]

    exp, _data = real74_d02.construct_rung_case(2)
    inventory = _mock_inventory("state/u")
    baseline = _mock_state_sample(
        exp, "d02", 900, inventory, dtbc_fp32_bits=first["scalars"][
            "dtbc_fp32_bits"])
    candidate = _mock_state_sample(
        exp, "d02", 900, inventory, dtbc_fp32_bits=second["scalars"][
            "dtbc_fp32_bits"])
    candidate["sha256"] = baseline["sha256"]
    comparison = real74_d02.compare_state_hash_samples(
        [candidate], [baseline], "d02")
    assert comparison["passed"] is False
    assert comparison["samples"][0]["scalars_equal"] is False
    assert comparison["samples"][0]["hash_compared"] is False


@pytest.mark.parametrize("resume_relation", ("subset", "superset"))
def test_restart_resume_inventory_converges_in_both_directions(
        resume_relation):
    exp, _data = real74_d02.construct_rung_case(2)
    small = _mock_inventory("state/u")
    grown = _mock_inventory("state/u", "scratch/nest_u_bxs")
    split_inventory = grown if resume_relation == "subset" else small
    resume_inventory = small if resume_relation == "subset" else grown

    def sample(offset, inventory):
        return _mock_state_sample(exp, "d02", offset, inventory)

    straight = [sample(0, small), sample(10, split_inventory)]
    resumed = []
    if resume_relation == "subset":
        straight.extend((sample(15, grown), sample(20, grown), sample(30, grown)))
        resumed.extend((sample(15, small), sample(20, grown), sample(30, grown)))
    else:
        straight.extend((sample(15, small), sample(20, grown), sample(30, grown)))
        resumed.extend((sample(15, grown), sample(20, grown), sample(30, grown)))
    prefix = straight[:2]
    straight_checkpoint = sample(10, split_inventory)
    prefix_checkpoint = dict(straight_checkpoint)
    straight_checkpoint["frame"] = "checkpoint@10"
    prefix_checkpoint["frame"] = "checkpoint@10"
    evidence = real74_d02.restart_inventory_convergence(
        straight, prefix, resumed, domain="d02",
        resume_inventory=resume_inventory,
        split_ticks=10, next_synchronized_ticks=20,
        straight_split_sample=straight_checkpoint,
        prefix_split_sample=prefix_checkpoint)
    assert evidence["passed"] is True
    assert evidence["resume_inventory_relation"] == resume_relation
    assert evidence["first_inventory_equal_ticks"] == 20
    assert evidence["post_resume_samples"][0]["hash_compared"] is False
    assert evidence["post_resume_samples"][1]["hash_compared"] is True


def test_restart_resume_dropped_serialized_member_fails():
    exp, _data = real74_d02.construct_rung_case(2)
    stored = _mock_inventory("state/u", "scratch/mp_rainnc")
    dropped = _mock_inventory("state/u")

    def sample(offset, inventory):
        return _mock_state_sample(exp, "d02", offset, inventory)

    straight = [sample(0, stored), sample(10, stored), sample(20, stored)]
    checkpoint = sample(10, stored)
    checkpoint["frame"] = "checkpoint@10"
    evidence = real74_d02.restart_inventory_convergence(
        straight, straight[:2], [sample(20, stored)], domain="d02",
        resume_inventory=dropped, split_ticks=10,
        next_synchronized_ticks=20,
        straight_split_sample=checkpoint,
        prefix_split_sample=dict(checkpoint))
    assert evidence["passed"] is False
    assert evidence["resume_inventory_relation"] == "subset"
    assert evidence["resume_inventory_difference_allowlisted"] is False
    assert evidence["resume_inventory_differing_members"] == [
        "scratch/mp_rainnc"]


@pytest.mark.parametrize("failure", ("missed-deadline", "rediverged"))
def test_restart_inventory_convergence_failure_modes(failure):
    exp, _data = real74_d02.construct_rung_case(2)
    small = _mock_inventory("state/u")
    grown = _mock_inventory("state/u", "scratch/nest_u_bxs")

    def sample(offset, inventory):
        return _mock_state_sample(exp, "d02", offset, inventory)

    straight = [
        sample(0, grown), sample(10, grown), sample(15, grown),
        sample(20, grown), sample(30, grown)]
    if failure == "missed-deadline":
        resumed = [sample(15, small), sample(20, small), sample(30, grown)]
    else:
        resumed = [sample(15, small), sample(20, grown), sample(30, small)]
    checkpoint = sample(10, grown)
    checkpoint["frame"] = "checkpoint@10"
    evidence = real74_d02.restart_inventory_convergence(
        straight, straight[:2], resumed, domain="d02",
        resume_inventory=small, split_ticks=10,
        next_synchronized_ticks=20,
        straight_split_sample=checkpoint,
        prefix_split_sample=dict(checkpoint))
    assert evidence["passed"] is False
    if failure == "missed-deadline":
        assert evidence["first_inventory_equal_ticks"] == 30
        assert evidence["converged_by_next_synchronized_frame"] is False
    else:
        assert evidence["first_inventory_equal_ticks"] == 20
        assert evidence["post_resume_samples"][-1]["passed"] is False


def test_restart_both_missing_checkpoint_samples_fail_loudly():
    exp, _data = real74_d02.construct_rung_case(2)
    inventory = _mock_inventory("state/u")

    def sample(offset):
        return _mock_state_sample(exp, "d02", offset, inventory)

    evidence = real74_d02.restart_inventory_convergence(
        [sample(0), sample(10), sample(20)], [sample(0), sample(10)],
        [sample(20)], domain="d02", resume_inventory=inventory,
        split_ticks=10, next_synchronized_ticks=20)
    assert evidence["passed"] is False
    assert evidence["split_checkpoint"]["passed"] is False
    assert "both restart arms lack" in evidence["split_checkpoint"]["reason"]


def test_ratchet_manifest_round_trip_and_immutability(tmp_path):
    exp, _data = real74_d02.construct_rung_case(2)
    inventory = _mock_inventory("state/u")
    d02_hashes = [
        _mock_state_sample(exp, "d02", offset, inventory)
        for offset in (0, 900)]
    d01_hashes = [
        _mock_state_sample(exp, "d01", offset, inventory)
        for offset in (0, 900)]
    frames = []
    for index, sample in enumerate(d02_hashes):
        path = tmp_path / sample["frame"]
        path.write_bytes(f"frame-{index}".encode("ascii"))
        frames.append(path)
    root = tmp_path / "rungs"
    state_hashes = [*d01_hashes, *d02_hashes]
    schedules = (
        _mock_state_schedule(
            "d01", tick_stop=900, history_interval_ticks=900),
        _mock_state_schedule(
            "d02", tick_stop=900, history_interval_ticks=900),
    )
    manifest = real74_d02.publish_ratchet(
        root, "N3", "d02", frames,
        provenance=_ratchet_provenance(tick_stop=900),
        expected_state_domains=("d01", "d02"),
        state_hashes=state_hashes, state_hash_schedules=schedules)
    assert manifest == root / "N3" / "manifest.json"
    loaded = real74_d02.load_ratchet(root, "N3", domain="d02")
    assert [item.frame for item in loaded.artifacts] == [path.name for path in frames]
    assert len(loaded.state_hashes) == 2
    full = real74_d02.load_ratchet(root, "N3")
    assert len(full.state_hashes) == 4
    assert full.expected_state_domains == ("d01", "d02")
    assert real74_d02.ratchet_frame(
        root, "N3", "d02", frames[1].name).read_bytes() == b"frame-1"
    assert real74_d02.publish_ratchet(
        root, "N3", "d02", frames,
        provenance=_ratchet_provenance(tick_stop=900),
        expected_state_domains=("d01", "d02"),
        state_hashes=state_hashes,
        state_hash_schedules=schedules) == manifest
    frames[0].write_bytes(b"different")
    with pytest.raises(FileExistsError, match="differs"):
        real74_d02.publish_ratchet(
            root, "N3", "d02", frames,
            provenance=_ratchet_provenance(tick_stop=900),
            expected_state_domains=("d01", "d02"),
            state_hashes=state_hashes,
            state_hash_schedules=schedules)


def test_ratchet_publication_requires_complete_state_evidence(tmp_path):
    exp, _data = real74_d02.construct_rung_case(2)
    sample = _mock_state_sample(
        exp, "d02", 0, _mock_inventory("state/u"))
    frame = tmp_path / sample["frame"]
    frame.write_bytes(b"frame")
    with pytest.raises(ValueError, match="requires canonical state evidence"):
        real74_d02.publish_ratchet(
            tmp_path / "rungs", "N3", "d02", (frame,),
            provenance=_ratchet_provenance(tick_stop=900),
            expected_state_domains=("d01", "d02"),
            state_hashes=(),
            state_hash_schedules=(
                _mock_state_schedule(
                    "d02", tick_stop=900,
                    history_interval_ticks=1800),))


def test_n3_publication_rejects_complete_d02_without_d01_control(tmp_path):
    exp, _data = real74_d02.construct_rung_case(2)
    inventory = _mock_inventory("state/u")
    d02_hashes = [
        _mock_state_sample(exp, "d02", offset, inventory)
        for offset in (0, 900)]
    frames = []
    for sample in d02_hashes:
        frame = tmp_path / sample["frame"]
        frame.write_bytes(b"complete-d02")
        frames.append(frame)
    with pytest.raises(ValueError, match="domain inventory"):
        real74_d02.publish_ratchet(
            tmp_path / "rungs", "N3", "d02", frames,
            provenance=_ratchet_provenance(tick_stop=900),
            expected_state_domains=("d01", "d02"),
            state_hashes=d02_hashes,
            state_hash_schedules=(
                _mock_state_schedule(
                    "d02", tick_stop=900, history_interval_ticks=900),))


def test_n3_full_load_rejects_injected_manifest_without_d01_control(tmp_path):
    _exp, root, _frames, _state_hashes, _schedule = _publish_mock_ratchet(
        tmp_path)
    manifest_path = root / "N3" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["state_hashes"] = [
        item for item in payload["state_hashes"] if item["domain"] == "d02"]
    payload["state_hash_schedules"] = [
        item for item in payload["state_hash_schedules"]
        if item["domain"] == "d02"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="domain inventory"):
        real74_d02.load_ratchet(root, "N3")


def test_n4_publication_declares_exact_n5_control_domain_inventory(tmp_path):
    exp, _data = real74_d02.construct_rung_case(3)
    inventory = _mock_inventory("state/u")
    sample = _mock_state_sample(exp, "d03", 0, inventory)
    frame = tmp_path / sample["frame"]
    frame.write_bytes(b"n4-d03-control")
    root = tmp_path / "rungs"
    real74_d02.publish_ratchet(
        root, "N4", "d03", (frame,),
        provenance=_ratchet_provenance(domain_count=3, tick_stop=900),
        expected_state_domains=("d03",), state_hashes=(sample,),
        state_hash_schedules=(
            _mock_state_schedule(
                "d03", tick_stop=900, history_interval_ticks=1800),))
    loaded = real74_d02.load_ratchet(root, "N4")
    assert loaded.expected_state_domains == ("d03",)
    assert (
        *real74_d02.RATCHET_STATE_EVIDENCE_DOMAINS["N3"],
        *loaded.expected_state_domains,
    ) == real74_chain.N5_ANCESTOR_CONTROL_DOMAINS


@pytest.mark.parametrize("corruption", (
    "absent", "empty", "wrong-domain-only", "scheduled-gap"))
def test_ratchet_load_rejects_incomplete_per_domain_state_evidence(
        tmp_path, corruption):
    exp, root, _frames, state_hashes, _schedule = _publish_mock_ratchet(
        tmp_path)
    manifest_path = root / "N3" / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if corruption == "absent":
        del payload["state_hashes"]
    elif corruption == "empty":
        payload["state_hashes"] = []
    elif corruption == "wrong-domain-only":
        payload["state_hashes"] = [
            _mock_state_sample(
                exp, "d01", offset, _mock_inventory("state/u"))
            for offset in (0, 900)]
        payload["state_hash_schedules"] = [
            _mock_state_schedule(
                "d01", tick_stop=900, history_interval_ticks=900)]
    else:
        payload["state_hashes"] = state_hashes[:-1]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
            ValueError, match="canonical state|state evidence|state-hash"):
        real74_d02.load_ratchet(root, "N3", domain="d02")


def test_boundary_blowup_consumes_accumulated_run_summary():
    value, evidence = real74_d02.boundary_zone_blowup_value({
        "boundary_w_max_ms": 9.0,
        "interior_w_max_ms": 1.0,
        "boundary_zone_blowup": True,
        "dynamics_substeps": 300,
    })
    assert value == 1.0
    assert evidence["boundary_w_max_ms"] == 9.0
    assert "every dynamics substep" in evidence["source"]
    with pytest.raises(ValueError, match="RunSummary"):
        real74_d02.boundary_zone_blowup_value({})


def test_complete_ratchet_inventory_catches_intermediate_and_missing_frames(
        tmp_path):
    exp, _data = real74_d02.construct_rung_case(2)
    source = tmp_path / "source"
    candidate = tmp_path / "candidate"
    source.mkdir()
    candidate.mkdir()
    frames = []
    candidates = []
    state_hashes = [
        _mock_state_sample(
            exp, "d02", index * 900, _mock_inventory("state/u"))
        for index in range(6)]
    for index, sample in enumerate(state_hashes):
        name = sample["frame"]
        baseline = source / name
        current = candidate / name
        baseline.write_bytes(f"frame-{index}".encode("ascii"))
        current.write_bytes(baseline.read_bytes())
        frames.append(baseline)
        candidates.append(current)
    root = tmp_path / "rungs"
    real74_d02.publish_ratchet(
        root, "N3", "d02", frames, provenance=_ratchet_provenance(),
        expected_state_domains=("d01", "d02"),
        state_hashes=(
            *(
                _mock_state_sample(
                    exp, "d01", index * 900, _mock_inventory("state/u"))
                for index in range(6)),
            *state_hashes,
        ),
        state_hash_schedules=(
            _mock_state_schedule(
                "d01", tick_stop=real74_d02.RUN_SECONDS,
                history_interval_ticks=900),
            _mock_state_schedule(
                "d02", tick_stop=real74_d02.RUN_SECONDS,
                history_interval_ticks=900),))
    manifest = real74_d02.load_ratchet(root, "N3", domain="d02")
    assert real74_d02.compare_ratchet_frames(
        candidates, root, manifest, "d02",
        candidate_state_hashes=state_hashes)["passed"] is True

    candidates[2].write_bytes(b"intermediate mutation")
    mutated = real74_d02.compare_ratchet_frames(
        candidates, root, manifest, "d02",
        candidate_state_hashes=state_hashes)
    assert mutated["passed"] is False
    assert len(mutated["frames"]) == 6
    candidates[2].write_bytes(frames[2].read_bytes())
    missing = real74_d02.compare_ratchet_frames(
        candidates[:-1], root, manifest, "d02",
        candidate_state_hashes=state_hashes)
    assert missing["passed"] is False
    assert missing["inventory_equal"] is False


def test_ratchet_provenance_and_artifact_corruption_refuse(tmp_path):
    exp, _data = real74_d02.construct_rung_case(2)
    sample = _mock_state_sample(
        exp, "d02", 0, _mock_inventory("state/u"))
    frame = tmp_path / sample["frame"]
    frame.write_bytes(b"binding")
    root = tmp_path / "rungs"
    provenance = _ratchet_provenance(tick_stop=900)
    real74_d02.publish_ratchet(
        root, "N3", "d02", (frame,), provenance=provenance,
        expected_state_domains=("d01", "d02"),
        state_hashes=(
            _mock_state_sample(
                exp, "d01", 0, _mock_inventory("state/u")),
            sample,
        ),
        state_hash_schedules=(
            _mock_state_schedule(
                "d01", tick_stop=900, history_interval_ticks=1800),
            _mock_state_schedule(
                "d02", tick_stop=900, history_interval_ticks=1800),))
    manifest = real74_d02.load_ratchet(root, "N3")
    real74_d02.validate_ratchet_provenance(manifest, provenance)
    assert manifest.experiment_fingerprint == provenance[
        "experiment_fingerprint"]
    assert (manifest.tick_start, manifest.tick_stop, manifest.tick_den) == (
        provenance["tick_start"], provenance["tick_stop"],
        provenance["tick_den"])

    wrong_fingerprint = dict(provenance, experiment_fingerprint="e" * 64)
    with pytest.raises(ValueError, match="provenance mismatch"):
        real74_d02.validate_ratchet_provenance(manifest, wrong_fingerprint)
    wrong_commit = dict(provenance, evaluator_commit="other-commit")
    with pytest.raises(ValueError, match="current run evaluator"):
        real74_d02.validate_ratchet_provenance(manifest, wrong_commit)
    wrong_ticks = dict(provenance, tick_stop=provenance["tick_stop"] - 1)
    with pytest.raises(ValueError, match="provenance mismatch"):
        real74_d02.validate_ratchet_provenance(manifest, wrong_ticks)

    stored = root / "N3" / manifest.artifacts[0].relative_path
    stored.write_bytes(b"corrupted after publication")
    with pytest.raises(ValueError, match="failed hash"):
        real74_d02.load_ratchet(root, "N3")


@pytest.mark.parametrize("runner", (real74_chain.run_n4, real74_chain.run_n5))
def test_rung_refuses_missing_predecessor_before_gpu(
        monkeypatch, tmp_path, runner):
    touched_gpu = False

    def forbidden_execute(*_args, **_kwargs):
        nonlocal touched_gpu
        touched_gpu = True
        raise AssertionError("GPU path must not be reached")

    monkeypatch.setattr(real74_d02, "execute_production_run", forbidden_execute)
    args = SimpleNamespace(
        config=real74_d02.PRODUCTION_CONFIG,
        ratchet_root=tmp_path / "missing-ratchets")
    with pytest.raises(FileNotFoundError):
        runner(args)
    assert touched_gpu is False


def test_production_subsets_and_full_sync_ledger_are_exact():
    for count in (2, 3, 4):
        exp, _data = real74_d02.construct_rung_case(count)
        assert tuple(dc.grid_id for dc in exp.domains) == tuple(range(1, count + 1))
        assert exp.run_seconds == real74_d02.RUN_SECONDS
        assert exp.restart_interval_s == real74_d02.RESTART_SPLIT_SECONDS
    exp, _data = real74_d02.construct_rung_case(4)
    from gpuwm.core.clock import build_schedule, resolve_clock
    schedule = build_schedule(exp, resolve_clock(exp))
    execution = {"clocks": {
        f"d{spec.grid_id:02d}": {
            "ticks": schedule.clock.run_ticks,
            "step_count": schedule.clock.run_ticks // spec.step_ticks,
            "tick_den": schedule.clock.tick_den,
        }
        for spec in schedule.clock.domains}}
    ledger = real74_chain.full_schedule_sync_ledger(
        real74_d02.PRODUCTION_CONFIG,
        executed_exp=exp, execution=execution)
    assert ledger["passed"] is True
    assert ledger["d04_equivalent_steps"] == 25920
    assert ledger["registered_d04_equivalent_steps"] == 25920
    assert ledger["executed_run"]["clocks"]["d04"]["expected"][
        "step_count"] == 2700

    del execution["clocks"]["d04"]
    missing = real74_chain.full_schedule_sync_ledger(
        real74_d02.PRODUCTION_CONFIG,
        executed_exp=exp, execution=execution)
    assert missing["passed"] is False
    assert missing["executed_run"]["inventory_equal"] is False


def test_ancestor_candidate_duplicates_are_rejected_before_dedup(
        monkeypatch, tmp_path):
    exp, _data = real74_d02.construct_rung_case(4)
    inventory = _mock_inventory("state/u")
    controls = {
        "N3": SimpleNamespace(expected_state_domains=("d01", "d02"),
            state_hashes=(
            _mock_state_sample(exp, "d01", 0, inventory),
            _mock_state_sample(exp, "d02", 0, inventory))),
        "N4": SimpleNamespace(expected_state_domains=("d03",), state_hashes=(
            _mock_state_sample(exp, "d03", 0, inventory),)),
    }
    monkeypatch.setattr(
        real74_d02, "load_ratchet",
        lambda _root, rung: controls[rung])
    monkeypatch.setattr(
        real74_d02, "validate_ratchet_provenance",
        lambda _manifest, _expected: None)
    monkeypatch.setattr(
        real74_d02, "run_prefix_provenance",
        lambda _run, domain_count: {"domain_count": domain_count})
    duplicate = _mock_state_sample(exp, "d01", 0, inventory)
    run = real74_d02.ProductionRun(
        output_dir=tmp_path, wrfout_paths=(), checkpoint=None,
        completed_seconds=real74_d02.RUN_SECONDS,
        execution={"canonical_state_hashes": [duplicate, dict(duplicate)]},
        timing={}, memory={})
    with pytest.raises(ValueError, match="duplicate canonical state sample"):
        real74_chain.ancestor_inertness_evidence(
            exp, run, phase4_root=tmp_path, ratchet_root=tmp_path)


def _n5s_files(tmp_path, candidate_distance, *, cpu_pair_values=(1.0, 2.0, 3.0)):
    wrf = tmp_path / "wrf"
    wrf.mkdir()
    members = []
    for i in range(3):
        artifact = wrf / f"member-{i}.json"
        artifact.write_text(json.dumps({"member": i}), encoding="utf-8")
        members.append({
            "id": f"m{i}", "relative_path": artifact.name,
            "sha256": real74_d02.sha256_file(artifact),
            "one_ulp_perturbation": f"field-{i}"})
    unperturbed = wrf / "unperturbed.json"
    unperturbed.write_text(json.dumps({"member": "base"}), encoding="utf-8")
    pairs = {}
    gpu = {}
    for category in real74_chain.N5S_CATEGORIES:
        domains = (("d04",) if category == "d04_reflectivity_fss_distance"
                   else real74_chain.N5S_DOMAINS)
        for domain in domains:
            key = f"{category}:{domain}:fixture"
            pairs[key] = list(cpu_pair_values)
            gpu[key] = candidate_distance
    durations = {domain: 1800 for domain in real74_chain.N5S_DOMAINS}
    (wrf / "n5s-ensemble.json").write_text(json.dumps({
        "wrf_build": {
            "wrf_version": "v4.6.1", "microphysics": "Morrison",
            "pbl": "YSU", "instrumented_build": "T8b"},
        "unperturbed": {
            "relative_path": unperturbed.name,
            "sha256": real74_d02.sha256_file(unperturbed)},
        "restored_input_sha256": "a" * 64,
        "domain_durations_seconds": durations,
        "members": members, "cpu_pair_distances": pairs,
        "cadence": "one fixture frame"}), encoding="utf-8")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "restored_input_sha256": "a" * 64,
        "domain_durations_seconds": durations,
        "cadence": "one fixture frame",
        "gpu_vs_unperturbed_distances": gpu}), encoding="utf-8")
    return candidate, wrf


def _stub_n5s_artifact_reconstruction(monkeypatch):
    """Keep envelope-policy unit tests independent of NetCDF artifact setup.

    End-to-end artifact and registration enforcement is covered by
    ``test_n5s_toolchain``; these tests isolate F28 row disposition.
    """
    def reconstructed(candidate_evidence, wrf_run_directory):
        candidate_path = Path(candidate_evidence).resolve()
        ensemble_path = Path(wrf_run_directory).resolve() / "n5s-ensemble.json"
        ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["gpu_candidate"] = {
            "relative_path": candidate_path.name,
            "sha256": real74_d02.sha256_file(candidate_path),
        }
        registration_sha256 = "c" * 64
        provenance = {
            "candidate_evidence": str(candidate_path),
            "candidate_evidence_sha256": real74_d02.sha256_file(candidate_path),
            "ensemble_manifest": str(ensemble_path),
            "ensemble_manifest_sha256": real74_d02.sha256_file(ensemble_path),
            "registration": {"fixture": "envelope-policy-only"},
            "registration_sha256": registration_sha256,
            "input_artifact_sha256": {
                "candidate_evidence": real74_d02.sha256_file(candidate_path),
                "ensemble_manifest": real74_d02.sha256_file(ensemble_path),
                "registration": registration_sha256,
            },
        }
        return ensemble, candidate, provenance

    monkeypatch.setattr(
        real74_chain, "_verified_n5s_documents", reconstructed)


def test_n5s_external_manifest_envelope_pass_and_single_miss(
        tmp_path, monkeypatch):
    _stub_n5s_artifact_reconstruction(monkeypatch)
    candidate, wrf = _n5s_files(tmp_path, 2.5)
    report = real74_chain.evaluate_n5s_shadow(candidate, wrf)
    assert report["passed"] is True
    expected = sum(
        1 if category == "d04_reflectivity_fss_distance"
        else len(real74_chain.N5S_DOMAINS)
        for category in real74_chain.N5S_CATEGORIES)
    assert len(report["comparisons"]) == expected
    assert report["degenerate_rows"] == 0
    assert report["documented_evidence"] is False
    assert report["all_envelopes_degenerate"] is False
    assert all(set(row) == {
        "metric", "gpu_distance", "cpu_e95", "passed"}
        for row in report["comparisons"])

    payload = json.loads(candidate.read_text(encoding="utf-8"))
    first = next(iter(payload["gpu_vs_unperturbed_distances"]))
    payload["gpu_vs_unperturbed_distances"][first] = 3.5
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    failed = real74_chain.evaluate_n5s_shadow(candidate, wrf)
    assert failed["passed"] is False
    assert sum(not row["passed"] for row in failed["comparisons"]) == 1


def test_f28_all_degenerate_n5s_envelopes_are_documented_evidence(
        tmp_path, monkeypatch):
    _stub_n5s_artifact_reconstruction(monkeypatch)
    candidate, wrf = _n5s_files(
        tmp_path, 2.5, cpu_pair_values=(0.0, 0.0, 0.0))

    report = real74_chain.evaluate_n5s_shadow(candidate, wrf)
    expected = sum(
        1 if category == "d04_reflectivity_fss_distance"
        else len(real74_chain.N5S_DOMAINS)
        for category in real74_chain.N5S_CATEGORIES)
    assert report["passed"] is True
    assert report["degenerate_rows"] == expected
    assert report["documented_evidence"] is True
    assert report["all_envelopes_degenerate"] is True
    assert len(report["comparisons"]) == expected
    for row in report["comparisons"]:
        assert set(row) == {
            "metric", "gpu_distance", "cpu_e95", "passed",
            "documented_evidence", "envelope_degenerate", "adjudication"}
        assert row["gpu_distance"] == 2.5
        assert row["cpu_e95"] == 0.0
        assert row["passed"] is True
        assert row["documented_evidence"] is True
        assert row["envelope_degenerate"] is True
        assert row["adjudication"] == "f28-degenerate-envelope"

    n5_row = real74_chain._consume_controller_gate_report(
        "N5S_matched_physics_wrf_shadow", report)
    assert n5_row["passed"] is True
    assert n5_row["evidence"]["degenerate_rows"] == expected
    assert n5_row["evidence"]["documented_evidence"] is True
    assert n5_row["evidence"]["all_envelopes_degenerate"] is True
    assert n5_row["evidence"]["comparisons"] == report["comparisons"]


@pytest.mark.parametrize("bad_distance", [
    float("nan"), float("inf"), float("-inf"), -1.0,
], ids=["nan", "positive-infinity", "negative-infinity", "negative"])
def test_f28_degenerate_envelope_rejects_invalid_gpu_distance(
        tmp_path, monkeypatch, bad_distance):
    """F28 changes the disposition of a valid distance, never its domain.

    A zero CPU-WRF envelope must not bypass the same finite/non-negative
    distance contract enforced for every non-degenerate row.
    """
    _stub_n5s_artifact_reconstruction(monkeypatch)
    candidate, wrf = _n5s_files(
        tmp_path, bad_distance, cpu_pair_values=(0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="negative/non-finite distance"):
        real74_chain.evaluate_n5s_shadow(candidate, wrf)


def test_f28_non_degenerate_n5s_row_rebinds_and_fails(tmp_path, monkeypatch):
    _stub_n5s_artifact_reconstruction(monkeypatch)
    candidate, wrf = _n5s_files(
        tmp_path, 2.5, cpu_pair_values=(0.0, 0.0, 0.0))
    ensemble_path = wrf / "n5s-ensemble.json"
    ensemble = json.loads(ensemble_path.read_text(encoding="utf-8"))
    binding_metric = next(iter(ensemble["cpu_pair_distances"]))
    ensemble["cpu_pair_distances"][binding_metric] = [1.0, 2.0, 3.0]
    ensemble_path.write_text(json.dumps(ensemble), encoding="utf-8")
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["gpu_vs_unperturbed_distances"][binding_metric] = 3.5
    candidate.write_text(json.dumps(payload), encoding="utf-8")

    report = real74_chain.evaluate_n5s_shadow(candidate, wrf)
    binding_row = next(
        row for row in report["comparisons"]
        if row["metric"] == binding_metric)
    assert binding_row == {
        "metric": binding_metric,
        "gpu_distance": 3.5,
        "cpu_e95": 3.0,
        "passed": False,
    }
    assert report["passed"] is False
    assert report["degenerate_rows"] == len(report["comparisons"]) - 1
    assert report["documented_evidence"] is True
    assert report["all_envelopes_degenerate"] is False


def test_external_gate_consumer_derives_verdict_from_bound_rows():
    metric = "N5S_matched_physics_wrf_shadow"
    report = {
        "schema": 2,
        "metric": metric,
        "evaluator_commit": real74_d02._git_commit(),
        "passed": True,
        "comparisons": [
            {"metric": "low_pass_state_rmse:d01:T", "passed": False},
        ],
        "expected_samples": 1,
        "cadence": "one fixture frame",
        "mask_hash": "a" * 64,
        "baseline_hash": "b" * 64,
        "score_inputs": {
            "metric_inventory": ["low_pass_state_rmse:d01:T"],
            "input_artifact_sha256": {"candidate": "c" * 64},
        },
    }
    report["score_binding_sha256"] = (
        real74_chain._controller_score_binding_sha256(report))

    row = real74_chain._consume_controller_gate_report(metric, report)

    assert row["passed"] is False
    assert row["evidence"]["declared_passed"] is True
    assert row["evidence"]["passed"] is False


def test_external_gate_loader_rejects_hash_shaped_report_without_lineage(
        tmp_path):
    metric = "N5S_matched_physics_wrf_shadow"
    path = tmp_path / "unverifiable-controller-report.json"
    path.write_text(json.dumps({
        "schema": 2,
        "metric": metric,
        "evaluator_commit": real74_d02._git_commit(),
        "passed": True,
        "comparisons": [{"metric": "fake", "passed": False}],
        "expected_samples": 1,
        "cadence": "one fixture frame",
        "mask_hash": "a" * 64,
        "baseline_hash": "b" * 64,
        "score_binding_sha256": "c" * 64,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="re-score lineage"):
        real74_chain._load_gate_report(path, metric)


def test_n5b_geometry_freeze_and_hash_pinned_score(tmp_path, monkeypatch):
    exp, _data, geometry = real74_chain.construct_n5b_shrink_case()
    assert (exp.domain(4).run.ny, exp.domain(4).run.nx) == (498, 498)
    assert geometry.production_shape == (600, 600)
    assert geometry.shrink_shape == (498, 498)
    assert geometry.core_shape == (400, 400)
    assert geometry.production_core == ((100, 500), (100, 500))
    assert geometry.shrink_core == ((49, 449), (49, 449))
    assert (exp.domain(4).i_parent_start,
            exp.domain(4).j_parent_start) == (168, 168)
    assert exp.domain(4).run.nx % exp.domain(4).parent_grid_ratio == 0
    assert exp.domain(4).run.ny % exp.domain(4).parent_grid_ratio == 0

    def member_record(member_id, *, perturbed):
        outdir = tmp_path / member_id / "frames"
        outdir.mkdir(parents=True)
        frame = outdir / "fixture.bin"
        frame.write_bytes(member_id.encode("ascii"))
        inventory = [{
            "domain": "d04", "valid_time": "1974-04-03T12:00:00",
            "offset_seconds": 0, "relative_path": frame.name,
            "bytes": frame.stat().st_size,
            "sha256": real74_d02.sha256_file(frame),
        }]
        perturbation = None
        if perturbed:
            spec = real74_n5b.DEFAULT_PERTURBATION_BY_ID[member_id]
            before = np.float32(len(member_id))
            after = np.float32(np.nextafter(before, np.float32(np.inf)))
            perturbation = {
                "operation": real74_n5b.PERTURBATION_OPERATION,
                "application_surface":
                    real74_n5b.PERTURBATION_APPLICATION_SURFACE,
                "domain": f"d{spec.domain_id:02d}", "field": spec.field,
                "field_definition":
                    real74_n5b.PERTURBATION_FIELD_DEFINITION,
                "index_order": "k,j,i",
                "k": spec.k, "j": spec.j, "i": spec.i,
                "before": float(before),
                "after": float(after),
                "before_hex_bits": real74_n5b._float32_bits(before),
                "after_hex_bits": real74_n5b._float32_bits(after),
            }
        record = {
            "schema": 1, "metric": real74_n5b.METRIC, "id": member_id,
            "variant": "production", "outdir": str(outdir),
            "duration_seconds": 4500, "cadence_seconds": 900,
            "restored_input_sha256": "b" * 64,
            "core_coordinate_sha256": "c" * 64,
            "one_ulp_perturbation": perturbation,
            "frame_inventory": inventory,
            "frame_inventory_sha256": real74_d02.stable_hash(inventory),
            "evaluator_commit": real74_d02._git_commit(),
        }
        path = tmp_path / f"{member_id}.json"
        real74_d02.write_json(path, record)
        return path

    members = [
        member_record(f"m{i:02d}", perturbed=True) for i in range(1, 4)]
    baseline = member_record("unperturbed", perturbed=False)
    evaluator_path = tmp_path / "evaluator.json"
    evaluator = real74_n5b.evaluator_manifest()
    evaluator["core_coordinate_sha256"] = "c" * 64
    real74_d02.write_json(evaluator_path, evaluator)
    monkeypatch.setattr(
        real74_n5b, "_evaluate_same_geometry_snapshot_pair",
        lambda _left, _right, *, evaluator: {
            "refl_fss": 1.0, "cold_pool_edge_km": 0.0,
            "gust_front_arrival_mae_min": 0.0,
            "unmatched_boundary_ci_count": 0.0, "tke_ratio": 1.0,
        })
    ensemble = tmp_path / "ensemble.json"
    real74_n5b.emit_ensemble_manifest(
        members, baseline, ensemble,
        evaluator_manifest_path=evaluator_path)
    frozen_path = tmp_path / "frozen.json"
    frozen = real74_chain.freeze_n5b_envelope(ensemble, frozen_path)
    anchor_path = real74_chain._n5b_freeze_anchor_path(frozen_path)
    anchor_sha256 = real74_d02.sha256_file(anchor_path)
    assert frozen["geometry"]["shrink_shape"] == (498, 498)

    observation = tmp_path / "observed.json"
    production_artifact = tmp_path / "production-evidence.json"
    shrink_artifact = tmp_path / "shrink-evidence.json"
    production_artifact.write_text("production", encoding="utf-8")
    shrink_artifact.write_text("shrink", encoding="utf-8")
    observation.write_text(json.dumps({
        "schema": 1, "metric": real74_n5b.METRIC,
        "frozen_manifest_sha256": real74_d02.sha256_file(frozen_path),
        "freeze_anchor_sha256": anchor_sha256,
        "evaluator_manifest_sha256": real74_d02.sha256_file(evaluator_path),
        "evaluator_commit": real74_d02._git_commit(),
        "evaluator_manifest": {
            "relative_path": evaluator_path.name,
            "sha256": real74_d02.sha256_file(evaluator_path),
        },
        "cadence": "15 min",
        "artifacts": {
            "production": {
                "relative_path": production_artifact.name,
                "sha256": real74_d02.sha256_file(production_artifact),
                "restored_input_sha256": "b" * 64,
                "core_coordinate_sha256": "c" * 64,
                "duration_seconds": 4500},
            "shrink": {
                "relative_path": shrink_artifact.name,
                "sha256": real74_d02.sha256_file(shrink_artifact),
                "restored_input_sha256": "b" * 64,
                "core_coordinate_sha256": "c" * 64,
                "duration_seconds": 4500},
        },
        "metrics": {
            "refl_fss": 0.95, "cold_pool_edge_km": 2.0,
            "gust_front_arrival_mae_min": 4.0,
            "unmatched_boundary_ci_count": 0,
            "tke_ratio": 1.0,
        }}), encoding="utf-8")
    with pytest.raises(TypeError, match="freeze_anchor_sha256"):
        real74_chain.score_n5b_observation(frozen_path, observation)
    report = real74_chain.score_n5b_observation(
        frozen_path, observation, freeze_anchor_sha256=anchor_sha256)
    assert report["passed"] is True
    assert len(report["comparisons"]) == 6
    assert report["freeze_anchor_sha256"] == real74_d02.sha256_file(
        real74_chain._n5b_freeze_anchor_path(frozen_path))
    assert str(ensemble.resolve()) in report["freeze_dependency_sha256"]
    report_path = tmp_path / "N5B-controller-report.json"
    declared = dict(report)
    declared["passed"] = False
    real74_d02.write_json(report_path, declared)
    loaded = real74_chain._load_gate_report(
        report_path, "N5B_d04_boundary_location_invariance")
    assert loaded["passed"] is True
    assert loaded["score_binding_sha256"] == report["score_binding_sha256"]

    original_verify = real74_chain.verify_n5b_freeze
    ensemble_bytes = ensemble.read_bytes()

    def mutate_dependency_after_verify(path, *, expected_anchor_sha256):
        verified = original_verify(
            path, expected_anchor_sha256=expected_anchor_sha256)
        ensemble.write_bytes(ensemble_bytes + b" ")
        return verified

    monkeypatch.setattr(
        real74_chain, "verify_n5b_freeze", mutate_dependency_after_verify)
    with pytest.raises(ValueError, match="dependency changed before scoring"):
        real74_chain.score_n5b_observation(
            frozen_path, observation, freeze_anchor_sha256=anchor_sha256)
    ensemble.write_bytes(ensemble_bytes)
    monkeypatch.setattr(real74_chain, "verify_n5b_freeze", original_verify)

    payload = json.loads(observation.read_text(encoding="utf-8"))
    payload["evaluator_manifest_sha256"] = "0" * 64
    observation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen evaluator"):
        real74_chain.score_n5b_observation(
            frozen_path, observation, freeze_anchor_sha256=anchor_sha256)

    payload["evaluator_manifest_sha256"] = real74_d02.sha256_file(evaluator_path)
    payload["evaluator_commit"] = "stale"
    observation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity/evaluator commit"):
        real74_chain.score_n5b_observation(
            frozen_path, observation, freeze_anchor_sha256=anchor_sha256)

    payload["evaluator_commit"] = real74_d02._git_commit()
    payload["frozen_manifest_sha256"] = "post-look-change"
    observation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pre-look freeze"):
        real74_chain.score_n5b_observation(
            frozen_path, observation, freeze_anchor_sha256=anchor_sha256)

    tampered = json.loads(frozen_path.read_text(encoding="utf-8"))
    tampered["allowed_discrepancies"]["cold_pool_edge_km"] = 999.0
    frozen_path.write_text(json.dumps(tampered), encoding="utf-8")
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["frozen_manifest"]["sha256"] = real74_d02.sha256_file(frozen_path)
    real74_d02.write_json(anchor_path, anchor)
    anchor_sha256 = real74_d02.sha256_file(anchor_path)
    payload["frozen_manifest_sha256"] = real74_d02.sha256_file(frozen_path)
    payload["freeze_anchor_sha256"] = anchor_sha256
    observation.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from reconstruction"):
        real74_chain.score_n5b_observation(
            frozen_path, observation, freeze_anchor_sha256=anchor_sha256)


def test_n5b_shrink_constructor_rejects_nonintegral_parent_shift(monkeypatch):
    monkeypatch.setattr(real74_chain, "n5b_geometry", lambda: real74_chain.N5BGeometry(
        production_shape=(600, 600), shrink_shape=(500, 500),
        production_core=((100, 500), (100, 500)),
        shrink_core=((50, 450), (50, 450))))
    with pytest.raises(AssertionError, match="not integral on the parent lattice"):
        real74_chain.construct_n5b_shrink_case()


def test_n5b_shrink_constructor_rejects_nondivisible_extent(monkeypatch):
    monkeypatch.setattr(real74_chain, "n5b_geometry", lambda: real74_chain.N5BGeometry(
        production_shape=(600, 600), shrink_shape=(499, 499),
        production_core=((100, 500), (100, 500)),
        shrink_core=((49, 449), (49, 449))))
    with pytest.raises(AssertionError, match="not divisible by the parent ratio"):
        real74_chain.construct_n5b_shrink_case()


def test_d04_budget_clipping_readiness_consumes_n6_record(tmp_path):
    path = tmp_path / "budget.json"
    window = {
        "label": "15min-1",
        "dry_mass_delta_storage": 1.0,
        "dry_mass_net_boundary_outflow": 1.0,
        "dry_mass_net_sources": 1.0,
        "dry_mass_residual": 1.0,
        "dry_mass_oracle_residual": 1.0, "dry_mass_throughput": 3.0,
        "total_water_delta_storage": 1.0,
        "total_water_net_boundary_outflow": 1.0,
        "total_water_net_sources": 1.0,
        "total_water_residual": 1.0,
        "total_water_oracle_residual": 1.0,
        "total_water_throughput": 3.0,
        "cumulative_clipped_condensate_fraction": 1.0e-9,
        "oracle_clipped_condensate_fraction": 0.0,
        "per_species_breakdown": {"qv": 0.0},
        "boundary_interior_breakdown": {"boundary": 0.0, "interior": 0.0},
    }
    path.write_text(json.dumps({"windows": [window]}), encoding="utf-8")
    report = real74_chain.evaluate_d04_budget_records(path)
    assert report["registered_milestone"] == "N6"
    assert report["passed"] is True
    assert "dry_mass_residual_limit" in report["windows"][0]


def test_summary_step_patch_target_resolves_and_is_executor_bound():
    """Fix-verification pin: the rung monkeypatches gpuwm.core.dycore.step.

    The executor imports step function-locally from gpuwm.core.dycore at
    call time (so that module attribute is the interceptable name), and
    gpuwm.core.model carries no `step` binding at all -- patching it
    would raise AttributeError and could never intercept.
    """
    import inspect

    import gpuwm.core.dycore as dycore
    import gpuwm.core.model as model

    assert callable(getattr(dycore, "step"))
    assert not hasattr(model, "step")
    source = inspect.getsource(model.execute_experiment)
    assert "from gpuwm.core.dycore import step" in source


def test_f24_production_matched_reference_registry_is_literal():
    assert nest_gates.MATCHED_REFERENCE_FRAMES == {
        "d02": (
            "gpuwm-wrf-matched-mp10-ysu-19740403-v1",
            "wrfout_d02_1974-04-03_13_15_00",
            "65fc6f52205c14cdf618ddf19520e8ae541ec74a9188de45524f9c12e2fb04aa",
            333839612,
        ),
        "d03": (
            "gpuwm-wrf-matched-mp10-ysu-19740403-v2",
            "wrfout_d03_1974-04-03_13_15_00",
            "4382b72ed3fa76662030f92e91ac97b3229e771b8dfce1b2351b58bd4ad25754",
            383352328,
        ),
        "d04": (
            "gpuwm-wrf-matched-mp10-ysu-19740403-v2",
            "wrfout_d04_1974-04-03_13_15_00",
            "be32aa0b1b3b8b79e2d9f5748c8becc6fdd57e579185cdd819131388306622d6",
            488452790,
        ),
    }


@pytest.mark.parametrize("domain,bundle", [
    ("d02", "gpuwm-wrf-matched-mp10-ysu-19740403-v1"),
    ("d03", "gpuwm-wrf-matched-mp10-ysu-19740403-v2"),
    ("d04", "gpuwm-wrf-matched-mp10-ysu-19740403-v2"),
])
def test_f24_matched_reference_pin_is_enforced(
        tmp_path, monkeypatch, domain, bundle):
    """F24: every matched-physics frame enforces its byte/SHA pin."""
    bundle_dir = tmp_path / "mp55-bundle"
    (bundle_dir / "namelists").mkdir(parents=True)
    wps = bundle_dir / "namelists" / "namelist.wps"
    wps.write_text("&share\n/\n", encoding="ascii")
    case_data = SimpleNamespace(wps_namelist=str(wps))

    matched_root = tmp_path / bundle / "wrfout"
    matched_root.mkdir(parents=True)
    frame = matched_root / f"wrfout_{domain}_1974-04-03_13_15_00"
    payload = b"matched-physics reference bytes"
    frame.write_bytes(payload)

    good_sha = real74_d02.sha256_file(frame)
    monkeypatch.setattr(
        nest_gates, "MATCHED_REFERENCE_FRAMES",
        {domain: (bundle, frame.name, good_sha, len(payload))})
    resolved = real74_d02.matched_reference_path(case_data, domain)
    assert resolved == frame

    monkeypatch.setattr(
        nest_gates, "MATCHED_REFERENCE_FRAMES",
        {domain: (bundle, frame.name, "0" * 64, len(payload))})
    with pytest.raises(ValueError, match="SHA-256"):
        real74_d02.matched_reference_path(case_data, domain)

    monkeypatch.setattr(
        nest_gates, "MATCHED_REFERENCE_FRAMES",
        {domain: (bundle, frame.name, good_sha, len(payload) + 1)})
    with pytest.raises(ValueError, match="byte count"):
        real74_d02.matched_reference_path(case_data, domain)

    # Missing file: correct registered name (the wrong-domain guard fires
    # first otherwise), but the frame is deleted from disk.
    monkeypatch.setattr(
        nest_gates, "MATCHED_REFERENCE_FRAMES",
        {domain: (bundle, frame.name, good_sha, len(payload))})
    frame.unlink()
    with pytest.raises(FileNotFoundError):
        real74_d02.matched_reference_path(case_data, domain)
    frame.write_bytes(payload)

    with pytest.raises(KeyError):
        real74_d02.matched_reference_path(case_data, "d01")

    # A wrong-domain frame registered under the domain key must never
    # resolve, even with its own correct byte count and SHA-256
    # (F21 review defect 1).
    wrong_domain = "d04" if domain != "d04" else "d03"
    wrong = matched_root / f"wrfout_{wrong_domain}_1974-04-03_13_15_00"
    wrong.write_bytes(payload)
    monkeypatch.setattr(
        nest_gates, "MATCHED_REFERENCE_FRAMES",
        {domain: (bundle, wrong.name, good_sha, len(payload))})
    with pytest.raises(ValueError, match="wrong-domain"):
        real74_d02.matched_reference_path(case_data, domain)


def test_f24_gate_text_names_per_domain_matched_reference():
    """Every registered FSS record names its per-domain matched bundle."""
    expected = {
        ("N3", "d02_refl_10cm_fss"):
            "gpuwm-wrf-matched-mp10-ysu-19740403-v1",
        ("N4", "d03_refl_10cm_fss"):
            "gpuwm-wrf-matched-mp10-ysu-19740403-v2",
        ("N5", "d04_refl_10cm_fss"):
            "gpuwm-wrf-matched-mp10-ysu-19740403-v2",
    }
    for (milestone, metric), bundle in expected.items():
        row = next(g for g in nest_gates.NEST_GATES
                   if g.milestone == milestone and g.metric == metric)
        assert "F21 MATCHED-PHYSICS" in row.convention
        assert bundle in row.convention
        assert "BLOCKS until" not in row.convention


def test_f24_unregistered_domain_gate_text_still_blocks(monkeypatch):
    monkeypatch.setitem(
        nest_gates.CHILD_REFERENCE_FRAMES, "d01",
        "wrfout_d01_1974-04-03_13_15_00")
    family = nest_gates._statistical_family(
        "synthetic", "d01", "synthetic note", "synthetic anchor")
    fss = next(row for row in family if row.metric == "d01_refl_10cm_fss")
    assert "no matched-physics reference is registered for d01" in fss.convention
    assert "BLOCKS until" in fss.convention

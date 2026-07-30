"""Admission gates for the standalone NSSL QVEXCESS production seam."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

_ROOT = Path(__file__).parents[1]
_ORACLE = _ROOT / "gpuwm" / "data" / "nssl2" / "oracle"
_FIXTURE_SHA256 = (
    "fbbcb323d356f10fe47d147c22c2a901f738fe22cf48427c2ad077abcef17661")


def _rows() -> list[dict[str, str]]:
    with (_ORACLE / "qvexcess.csv").open(
            newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _host(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float32)


def _device(rows, name):
    import cupy as cp

    return cp.asarray(_host(rows, name).reshape(-1, 1, 1))


def _workspace(shape):
    import cupy as cp

    from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace

    return NSSL2DriverWorkspace(
        state=cp.zeros((16, *shape), dtype=cp.float32),
        category_surface_export=cp.zeros((5, *shape[1:]), dtype=cp.float32),
        shape=shape,
    )


def _groups(rows, name):
    values = sorted({float(row[name]) for row in rows})
    for value in values:
        yield value, [row for row in rows if float(row[name]) == value]


def _assert_float32_close(
        actual, expected, *, rtol=1.0e-5, atol=2.0e-8, label=""):
    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=atol, equal_nan=False,
        err_msg=label)


def _git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor",
         ancestor, descendant],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode not in (0, 1):
        completed.check_returncode()
    return completed.returncode == 0


def _git_blob_digest(root: Path, revision: str, relative: str) -> str:
    return hashlib.sha256(
        _git_output(root, "show", f"{revision}:{relative}")).hexdigest()


def test_qvexcess_oracle_is_content_addressed_complete_and_branch_rich():
    payload = (_ORACLE / "qvexcess.csv").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == _FIXTURE_SHA256
    rows = _rows()
    assert len(rows) == 128
    values = np.asarray([
        float(value) for row in rows for key, value in row.items()
        if key != "case"
    ], dtype=np.float64)
    assert np.isfinite(values).all()
    for name in ("iteration1_branch", "iteration2_branch"):
        assert {int(float(row[name])) for row in rows} == {-2, -1, 0, 1}
    assert any(float(row["qvex"]) == 0.0 for row in rows)
    assert any(float(row["qvex"]) > 0.0 for row in rows)
    np.testing.assert_array_equal(
        np.asarray(sorted({
            np.float32(row["target_supersaturation_percent"])
            for row in rows}), dtype=np.float32),
        np.asarray([0.0, 0.4, 90.0, 250.0], dtype=np.float32))
    temperatures = {
        np.float32(row["temperature_initial_k"]) for row in rows}
    assert {
        np.float32(163.149), np.float32(163.15),
        np.float32(2163.149), np.float32(2163.15), np.float32(2163.151),
    }.issubset(temperatures)


def test_qvexcess_evidence_manifest_pins_all_inputs_and_frozen_seams():
    root = Path(__file__).parents[1]
    evidence = json.loads(
        (_ORACLE / "qvexcess-evidence-v1.json").read_text(encoding="ascii"))
    assert evidence["schema"] == "gpuwm.nssl2.qvexcess-evidence.v1"

    def digest(relative):
        # Evidence hashes identify Git-canonical text, independent of the
        # checkout's Windows/POSIX line-ending policy.
        payload = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(payload).hexdigest()

    build = evidence["oracle_build"]
    tracked = {
        "tools/nssl2_wrf461_qvexcess_oracle/build.sh":
            build["build_sh_sha256"],
        "tools/nssl2_wrf461_qvexcess_oracle/visibility.patch":
            build["visibility_patch_sha256"],
        "tools/nssl2_wrf461_qvexcess_oracle/qvexcess.F90":
            build["harness_sha256"],
        "tools/nssl2_wrf461_oracle/stub_wrf.F90":
            build["stub_wrf_sha256"],
        **evidence["implementation"],
    }
    for relative, expected in tracked.items():
        assert digest(relative) == expected

    fixture = evidence["fixture"]
    assert digest("gpuwm/data/nssl2/oracle/qvexcess.csv") == fixture["sha256"]
    assert len(_rows()) == fixture["rows"]

    base_commit = evidence["base_commit"]
    frozen = evidence["frozen_integration_seams"]
    assert frozen["hash_scope"] == "base_commit_git_blob"
    assert frozen["base_commit_must_be_ancestor"] is True
    assert _git_is_ancestor(root, base_commit, "HEAD")
    for relative, expected in frozen.items():
        if relative.startswith("gpuwm/"):
            assert _git_blob_digest(root, base_commit, relative) == expected


def test_split_qvexcess_matches_official_wrf_and_is_pure_return():
    import cupy as cp

    from gpuwm.core.nssl2_qvexcess import launch_qvexcess_split

    rows = _rows()
    names = (
        "theta_base_k", "theta_perturbation_k", "pressure_pa", "exner",
        "qv_base", "qv_perturbation", "qc_initial",
        "condensation_factor_k", "latent_over_cp_k",
    )
    for target, subset in _groups(rows, "target_supersaturation_percent"):
        arrays = {name: _device(subset, name) for name in names}
        before = {name: value.copy() for name, value in arrays.items()}
        output = cp.empty_like(arrays["qv_base"])
        launch_qvexcess_split(
            arrays["theta_base_k"], arrays["theta_perturbation_k"],
            arrays["pressure_pa"], arrays["exner"],
            arrays["qv_base"], arrays["qv_perturbation"],
            arrays["qc_initial"], arrays["condensation_factor_k"],
            arrays["latent_over_cp_k"], target, output)
        _assert_float32_close(
            cp.asnumpy(output).ravel(), _host(subset, "qvex"))
        for name, value in arrays.items():
            assert bool(cp.array_equal(value, before[name]))
        assert bool(cp.all(output >= 0.0))


def test_two_iteration_trace_matches_compiled_fortran_branches():
    import cupy as cp

    from gpuwm.core.kernels import get_kernel

    rows = _rows()
    input_names = (
        "theta_base_k", "theta_perturbation_k", "pressure_pa", "exner",
        "qv_base", "qv_perturbation", "qc_initial",
        "condensation_factor_k", "latent_over_cp_k",
    )
    output_names = (
        "qvex",
        "iteration1_branch", "iteration1_target_qv", "iteration1_qv",
        "iteration1_qc", "iteration1_theta_perturbation_k",
        "iteration2_branch", "iteration2_target_qv", "iteration2_qv",
        "iteration2_qc", "iteration2_theta_perturbation_k",
    )
    kernel = get_kernel("nssl2_qvexcess", "nssl2_qvexcess_trace_split")
    for target, subset in _groups(rows, "target_supersaturation_percent"):
        arrays = {name: _device(subset, name) for name in input_names}
        outputs = [cp.empty_like(arrays["qv_base"]) for _ in output_names]
        size = len(subset)
        blocks = (size + 255) // 256
        kernel(
            (blocks,), (256,),
            (*arrays.values(), np.float32(target), *outputs, np.int32(size)))
        comparisons = list(zip(output_names, outputs, strict=True))
        for name, output in comparisons[1:] + comparisons[:1]:
            actual = cp.asnumpy(output).ravel()
            expected = _host(subset, name)
            if "branch" in name:
                np.testing.assert_array_equal(actual, expected)
            else:
                _assert_float32_close(actual, expected, label=name)


def test_workspace_pure_api_reads_direct_mass_views_without_mutation():
    import cupy as cp

    from gpuwm.core.nssl2_qvexcess import launch_qvexcess_workspace

    rows = _rows()
    for target, subset in _groups(rows, "target_supersaturation_percent"):
        shape = (len(subset), 1, 1)
        workspace = _workspace(shape)
        full_theta = (
            _device(subset, "theta_base_k")
            + _device(subset, "theta_perturbation_k"))
        workspace.field("qv")[...] = (
            _device(subset, "qv_base")
            + _device(subset, "qv_perturbation"))
        workspace.field("qc")[...] = _device(subset, "qc_initial")
        inputs = {
            "full_theta": full_theta,
            "pressure": _device(subset, "pressure_pa"),
            "exner": _device(subset, "exner"),
            "qv": workspace.field("qv"),
            "qc": workspace.field("qc"),
            "factor": _device(subset, "condensation_factor_k"),
            "latent": _device(subset, "latent_over_cp_k"),
        }
        before_state = workspace.state.copy()
        before = {name: value.copy() for name, value in inputs.items()}
        output = cp.empty(shape, dtype=cp.float32)
        launch_qvexcess_workspace(
            workspace, inputs["full_theta"], inputs["pressure"],
            inputs["exner"], inputs["factor"], inputs["latent"],
            target, output)
        _assert_float32_close(
            cp.asnumpy(output).ravel(), _host(subset, "workspace_qvex"))
        assert bool(cp.array_equal(workspace.state, before_state))
        for name, value in inputs.items():
            assert bool(cp.array_equal(value, before[name]))


@pytest.mark.parametrize("couple_number", [False, True])
def test_default_caller_adapter_matches_official_wrf_and_conserves(
        couple_number):
    import cupy as cp

    from gpuwm.core.nssl2_qvexcess import (
        apply_qvexcess_maxsup_to_workspace,
    )

    rows = [
        row for row in _rows()
        if bool(int(float(row["couple_number"]))) is couple_number
    ]
    shape = (len(rows), 1, 1)
    workspace = _workspace(shape)
    full_theta = (
        _device(rows, "theta_base_k")
        + _device(rows, "theta_perturbation_k"))
    workspace.field("qv")[...] = (
        _device(rows, "qv_base") + _device(rows, "qv_perturbation"))
    workspace.field("qc")[...] = _device(rows, "qc_initial")
    workspace.field("qndrop")[...] = _device(
        rows, "cloud_number_initial_m3")
    workspace.field("qnn")[...] = _device(rows, "ccn_initial_m3")
    water_before = (
        workspace.field("qv") + workspace.field("qc")).copy()
    theta_before = full_theta.copy()
    qc_before = workspace.field("qc").copy()
    new_number = cp.empty(shape, dtype=cp.float32)

    apply_qvexcess_maxsup_to_workspace(
        workspace, full_theta, _device(rows, "rho_kg_m3"),
        _device(rows, "exner"), _device(rows, "background_ccn_m3"),
        _device(rows, "cloud_mean_mass_kg"),
        _device(rows, "latent_over_cp_k"), _device(rows, "qvex"),
        new_number, couple_number=couple_number)

    expected = {
        "theta": "theta_direct_after_k",
        "qv": "qv_direct_after",
        "qc": "qc_after",
        "qndrop": "cloud_number_after_m3",
        "qnn": "ccn_after_m3",
        "new": "new_cloud_number_m3",
    }
    actual = {
        "theta": full_theta,
        "qv": workspace.field("qv"),
        "qc": workspace.field("qc"),
        "qndrop": workspace.field("qndrop"),
        "qnn": workspace.field("qnn"),
        "new": new_number,
    }
    for name, fixture_name in expected.items():
        _assert_float32_close(
            cp.asnumpy(actual[name]).ravel(), _host(rows, fixture_name),
            rtol=3.0e-5, atol=3.0e-9)

    _assert_float32_close(
        cp.asnumpy(workspace.field("qv") + workspace.field("qc")).ravel(),
        cp.asnumpy(water_before).ravel(), rtol=3.0e-6, atol=3.0e-9)
    theta_increment = full_theta-theta_before
    expected_increment = (
        _device(rows, "latent_over_cp_k")
        * (workspace.field("qc")-qc_before) / _device(rows, "exner"))
    _assert_float32_close(
        cp.asnumpy(theta_increment).ravel(),
        cp.asnumpy(expected_increment).ravel(),
        rtol=5.0e-5, atol=2.0e-4)
    for name in ("qv", "qc", "qndrop", "qnn"):
        assert bool(cp.all(workspace.field(name) >= 0.0))


def test_zero_qvex_caller_update_is_bitwise_noop():
    import cupy as cp

    from gpuwm.core.nssl2_qvexcess import (
        apply_qvexcess_maxsup_to_workspace,
    )

    shape = (2, 1, 1)
    workspace = _workspace(shape)
    workspace.field("qv").fill(np.float32(0.01))
    workspace.field("qc").fill(np.float32(0.001))
    workspace.field("qndrop").fill(np.float32(2.0e8))
    workspace.field("qnn").fill(np.float32(3.0e8))
    theta = cp.full(shape, 300.0, dtype=cp.float32)
    before_state = workspace.state.copy()
    before_theta = theta.copy()
    new_number = cp.full(shape, -1.0, dtype=cp.float32)
    apply_qvexcess_maxsup_to_workspace(
        workspace, theta,
        cp.full(shape, 1.0, dtype=cp.float32),
        cp.full(shape, 0.95, dtype=cp.float32),
        cp.full(shape, 4.0e8, dtype=cp.float32),
        cp.full(shape, 1.0e-10, dtype=cp.float32),
        cp.full(shape, 2500.0, dtype=cp.float32),
        cp.zeros(shape, dtype=cp.float32), new_number)
    assert bool(cp.array_equal(workspace.state, before_state))
    assert bool(cp.array_equal(theta, before_theta))
    assert bool(cp.all(new_number == 0.0))


def test_validation_alias_and_fail_loud_gates():
    import cupy as cp

    from gpuwm.core.nssl2_qvexcess import (
        apply_qvexcess_maxsup_to_workspace,
        launch_qvexcess_split,
        launch_qvexcess_workspace,
    )

    shape = (2, 1, 1)
    workspace = _workspace(shape)
    positive = cp.ones(shape, dtype=cp.float32)
    zero = cp.zeros(shape, dtype=cp.float32)
    output = cp.empty(shape, dtype=cp.float32)

    with pytest.raises(ValueError, match="must not alias"):
        launch_qvexcess_workspace(
            workspace, positive, positive, positive, positive, positive,
            90.0, workspace.field("qv"))
    with pytest.raises(ValueError, match="greater than -100"):
        launch_qvexcess_workspace(
            workspace, positive, positive, positive, positive, positive,
            -100.0, output)
    bad = positive.copy()
    bad[0, 0, 0] = cp.nan
    with pytest.raises(ValueError, match="finite"):
        launch_qvexcess_workspace(
            workspace, bad, positive, positive, positive, positive,
            90.0, output)
    with pytest.raises(TypeError, match="float32"):
        launch_qvexcess_split(
            positive.astype(cp.float64), zero, positive, positive,
            positive, zero, zero, positive, positive, 90.0, output)
    with pytest.raises(ValueError, match="must not exceed"):
        apply_qvexcess_maxsup_to_workspace(
            workspace, positive, positive, positive, positive, positive,
            positive, cp.full(shape, 2.0, dtype=cp.float32), output)
    with pytest.raises(TypeError, match="couple_number"):
        apply_qvexcess_maxsup_to_workspace(
            workspace, positive, positive, positive, positive, positive,
            positive, zero, output, couple_number=1)


def test_concentration_workspace_path_has_no_registry_round_trip():
    python_source = (
        _ROOT / "gpuwm/core/nssl2_qvexcess.py").read_text(encoding="ascii")
    cuda_source = (
        _ROOT / "gpuwm/core/kernels/nssl2_qvexcess.cu").read_text(
            encoding="ascii")

    assert 'qv = workspace.field("qv")' in python_source
    assert 'qc = workspace.field("qc")' in python_source
    assert 'qndrop = workspace.field("qndrop")' in python_source
    assert 'qnn = workspace.field("qnn")' in python_source
    assert "concentration_space" not in cuda_source
    assert "input_number_scale" not in cuda_source
    assert "output_number_scale" not in cuda_source
    assert "Registry gather/scatter or density unit conversion" in cuda_source

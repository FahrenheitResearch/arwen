"""GPU admission gates for NSSL NUCOND and production diagnostics."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


pytestmark = pytest.mark.gpu

_ORACLE = (
    Path(__file__).parents[1] / "gpuwm" / "data" / "nssl2" / "oracle")
_NUCOND_SHA256 = (
    "1b1b472a832a6c9b9968e218d43a61b06c924880254ae0ea049b67e74eff4911")
_RADAR_SHA256 = (
    "37b4173833d9b1213da323ccdff1edf7cf5f79ba8b8f1eff62fc8ac91d9e65b7")


def _rows(name: str) -> list[dict[str, str]]:
    with (_ORACLE / name).open(newline="", encoding="ascii") as stream:
        return list(csv.DictReader(stream))


def _host(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float32)


def _git_output(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout


def _git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor",
         ancestor, descendant],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode not in (0, 1):
        completed.check_returncode()
    return completed.returncode == 0


def _git_blob_digest(root: Path, revision: str, relative: str) -> str:
    return hashlib.sha256(
        _git_output(root, "show", f"{revision}:{relative}")).hexdigest()


def test_production_oracles_are_content_addressed_and_complete():
    expected = {
        "nucond-production.csv": _NUCOND_SHA256,
        "radardd02.csv": _RADAR_SHA256,
    }
    for name, digest in expected.items():
        payload = (_ORACLE / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        rows = _rows(name)
        assert len(rows) == 64
        assert np.isfinite(np.asarray([
            float(value) for row in rows for key, value in row.items()
            if key not in ("case", "i", "k")], dtype=np.float64)).all()


def test_production_evidence_manifest_pins_every_tracked_input():
    root = Path(__file__).parents[1]
    evidence = json.loads(
        (_ORACLE / "production-evidence-v1.json").read_text(encoding="ascii"))

    def digest(relative):
        # Evidence hashes identify Git-canonical text, independent of the
        # checkout's Windows/POSIX line-ending policy.
        payload = (root / relative).read_bytes().replace(b"\r\n", b"\n")
        return hashlib.sha256(payload).hexdigest()

    build = evidence["oracle_build"]
    oracle_inputs = {
        "tools/nssl2_wrf461_production_oracle/build.sh":
            build["build_sh_sha256"],
        "tools/nssl2_wrf461_production_oracle/visibility.patch":
            build["visibility_patch_sha256"],
        "tools/nssl2_wrf461_production_oracle/nucond_production.F90":
            build["nucond_harness_sha256"],
        "tools/nssl2_wrf461_production_oracle/radardd02.F90":
            build["radardd02_harness_sha256"],
    }
    for relative, expected in {
            **oracle_inputs, **evidence["implementation"]}.items():
        assert digest(relative) == expected

    base_commit = evidence["base_commit"]
    frozen = evidence["frozen_low_level_kernel"]
    assert frozen["hash_scope"] == "base_commit_git_blob"
    assert frozen["base_commit_must_be_ancestor"] is True
    assert _git_is_ancestor(root, base_commit, "HEAD")
    for relative, expected in frozen.items():
        if relative.startswith("gpuwm/"):
            assert _git_blob_digest(root, base_commit, relative) == expected

    for name, item in evidence["fixtures"].items():
        assert digest(f"gpuwm/data/nssl2/oracle/{name}") == item["sha256"]
        assert len(_rows(name)) == item["rows"]


def test_frozen_blob_pins_survive_legitimate_descendant_core_commit(tmp_path):
    repo = tmp_path / "portable-evidence"
    repo.mkdir()
    _git_output(repo, "init", "-q")
    relative = "gpuwm/core/nssl2.py"
    frozen = repo / relative
    frozen.parent.mkdir(parents=True)
    frozen.write_bytes(b"frozen-v1\n")
    expected = hashlib.sha256(frozen.read_bytes()).hexdigest()
    _git_output(repo, "add", relative)
    _git_output(
        repo, "-c", "user.name=GPUWM test",
        "-c", "user.email=gpuwm-test@example.invalid",
        "commit", "-q", "-m", "frozen base")
    base_commit = _git_output(repo, "rev-parse", "HEAD").decode().strip()

    frozen.write_bytes(b"legitimate-later-core-v2\n")
    _git_output(repo, "add", relative)
    _git_output(
        repo, "-c", "user.name=GPUWM test",
        "-c", "user.email=gpuwm-test@example.invalid",
        "commit", "-q", "-m", "later core")

    assert _git_is_ancestor(repo, base_commit, "HEAD")
    assert _git_blob_digest(repo, base_commit, relative) == expected
    assert hashlib.sha256(frozen.read_bytes()).hexdigest() != expected


def test_concentration_space_kernels_have_identity_moment_scales():
    """Pin the production path's no-scatter/no-gather unit contract."""
    root = Path(__file__).parents[1]
    nucond = (
        root / "gpuwm/core/kernels/nssl2_nucond.cu").read_text(
            encoding="ascii")
    diagnostics = (
        root / "gpuwm/core/kernels/nssl2_diagnostics.cu").read_text(
            encoding="ascii")

    assert (
        "input_number_scale = concentration_space ? 1.0f : rho" in nucond)
    assert (
        "output_number_scale = concentration_space ? 1.0f : 1.0f / rho"
        in nucond)
    assert "number_scale = concentration_space ? 1.0f : rho" in diagnostics
    assert "volume_scale = concentration_space ? 1.0f : rho" in diagnostics
    assert "const float nc = fmaxf(cloud_number[idx], 0.0f)" in diagnostics
    assert "cloud_number[idx] * rho" not in diagnostics


def test_nucond_registry_and_concentration_modes_are_equivalent():
    import cupy as cp

    from gpuwm.core.nssl2_nucond import launch_nucond

    row = next(row for row in _rows("nucond-production.csv")
               if int(row["case"]) == 2)
    shape = (3, 1, 1)

    def cell(name):
        return cp.full(shape, np.float32(float(row[name])), dtype=cp.float32)

    density = cell("rho_kg_m3")
    common = {
        "theta": cell("theta_before_k"),
        "density": density,
        "pressure": cell("pressure_pa"),
        "exner": cell("exner"),
        "velocity": cp.full(
            (4, 1, 1), np.float32(float(row["w_m_s"])), dtype=cp.float32),
        "qv": cell("qv_before"),
        "qc": cell("qc_before"),
        "qr": cell("qr_before"),
        "qi": cp.zeros(shape, dtype=cp.float32),
        "qs": cp.zeros(shape, dtype=cp.float32),
        "nc": cell("nc_before_per_kg"),
        "nr": cell("nr_before_per_kg"),
        "ni": cp.zeros(shape, dtype=cp.float32),
        "ns": cp.zeros(shape, dtype=cp.float32),
        "ccn": cell("qnn_before_per_kg"),
    }

    registry = {name: value.copy() for name, value in common.items()}
    concentration = {name: value.copy() for name, value in common.items()}
    for name in ("nc", "nr", "ccn"):
        concentration[name] *= density

    def run(fields, concentration_space):
        launch_nucond(
            fields["theta"], fields["density"], fields["pressure"],
            fields["exner"], fields["velocity"],
            fields["qv"], fields["qc"], fields["qr"],
            fields["qi"], fields["qs"],
            fields["nc"], fields["nr"], fields["ni"], fields["ns"],
            fields["ccn"],
            float(row["dt_s"]), concentration_space=concentration_space)

    run(registry, False)
    run(concentration, True)
    cp.cuda.Stream.null.synchronize()

    for name in ("theta", "qv", "qc", "qr"):
        cp.testing.assert_allclose(
            concentration[name], registry[name], rtol=2.0e-6, atol=2.0e-8)
    for name in ("nc", "nr", "ccn"):
        cp.testing.assert_allclose(
            concentration[name], registry[name] * density,
            rtol=3.0e-6, atol=16.0)

    water_before = common["qv"] + common["qc"] + common["qr"]
    for fields in (registry, concentration):
        cp.testing.assert_allclose(
            fields["qv"] + fields["qc"] + fields["qr"], water_before,
            rtol=0.0, atol=3.0e-9)


@pytest.mark.parametrize("concentration_space", [False, True])
def test_nucond_matches_compiled_official_wrf_and_conserves_water_energy(
        concentration_space):
    import cupy as cp

    from gpuwm.core.nssl2_nucond import launch_nucond

    rows = _rows("nucond-production.csv")
    output_names = (
        "qv_after", "theta_after_k", "qc_after", "qr_after",
        "nc_after_per_kg", "nr_after_per_kg", "qnn_after_per_kg")
    actual_by_name = {name: np.empty(len(rows), dtype=np.float32)
                      for name in output_names}

    for step in sorted(set(_host(rows, "dt_s").tolist())):
        selected = [index for index, row in enumerate(rows)
                    if np.float32(float(row["dt_s"])) == np.float32(step)]
        subset = [rows[index] for index in selected]
        count = len(subset)

        def cell(name):
            values = _host(subset, name)
            host = np.broadcast_to(values, (3, 1, count)).copy()
            return cp.asarray(host)

        theta = cell("theta_before_k")
        density = cell("rho_kg_m3")
        pressure = cell("pressure_pa")
        exner = cell("exner")
        velocity = cp.asarray(np.broadcast_to(
            _host(subset, "w_m_s"), (4, 1, count)).copy())
        qv = cell("qv_before")
        qc = cell("qc_before")
        qr = cell("qr_before")
        qi = cp.zeros_like(qr)
        qs = cp.zeros_like(qr)
        qndrop = cell("nc_before_per_kg")
        qnr = cell("nr_before_per_kg")
        qni = cp.zeros_like(qnr)
        qns = cp.zeros_like(qnr)
        qnn = cell("qnn_before_per_kg")
        if concentration_space:
            qndrop *= density
            qnr *= density
            qnn *= density

        launch_nucond(
            theta, density, pressure, exner, velocity,
            qv, qc, qr, qi, qs, qndrop, qnr, qni, qns, qnn, step,
            concentration_space=concentration_space)
        cp.cuda.Stream.null.synchronize()

        outputs = (qv, theta, qc, qr, qndrop, qnr, qnn)
        for output, name in zip(outputs, output_names):
            values = output[1, 0]
            if concentration_space and name in (
                    "nc_after_per_kg", "nr_after_per_kg",
                    "qnn_after_per_kg"):
                values = values / density[1, 0]
            actual_by_name[name][selected] = cp.asnumpy(values)

    tolerances = {
        "qv_after": (6.0e-5, 2.0e-8),
        "theta_after_k": (8.0e-6, 8.0e-5),
        "qc_after": (2.0e-4, 2.0e-8),
        "qr_after": (2.0e-4, 2.0e-8),
        "nc_after_per_kg": (3.0e-5, 8.0),
        "nr_after_per_kg": (3.0e-5, 2.0e-3),
        "qnn_after_per_kg": (3.0e-5, 8.0),
    }
    for name in output_names:
        rtol, atol = tolerances[name]
        np.testing.assert_allclose(
            actual_by_name[name], _host(rows, name), rtol=rtol, atol=atol)

    water_before = (
        _host(rows, "qv_before") + _host(rows, "qc_before")
        + _host(rows, "qr_before"))
    water_after = (
        actual_by_name["qv_after"] + actual_by_name["qc_after"]
        + actual_by_name["qr_after"])
    np.testing.assert_allclose(
        water_after, water_before, rtol=0.0, atol=3.0e-9)

    initial_temperature = _host(rows, "temperature_k")
    bounded_temperature = np.clip(initial_temperature, 233.15, 313.15)
    latent_over_cp = (
        2500837.367 * (273.15 / bounded_temperature)
        ** (0.167 + 3.67e-4 * bounded_temperature) / 1004.0)
    condensed = (
        actual_by_name["qc_after"] + actual_by_name["qr_after"]
        - _host(rows, "qc_before") - _host(rows, "qr_before"))
    theta_increment = (
        actual_by_name["theta_after_k"] - _host(rows, "theta_before_k"))
    np.testing.assert_allclose(
        theta_increment,
        latent_over_cp * condensed / _host(rows, "exner"),
        rtol=3.0e-3, atol=3.5e-5)

    # Existing-cloud activation moves predicted CCN into droplets without
    # creating number.  Exclude the native full-evaporation restoration rows.
    cloud_mean_mass = _host(rows, "qc_before") / np.maximum(
        _host(rows, "nc_before_per_kg"), np.float32(1.0e-30))
    cloud_min_mass = np.float32(1000.0 * 0.523599 * (4.0e-6 ** 3))
    cloud_max_mass = np.float32(1000.0 * 0.523599 * (120.0e-6 ** 3))
    persistent = (
        (_host(rows, "qc_before") > 1.0e-13)
        & (actual_by_name["qc_after"] > 1.0e-13)
        & (cloud_mean_mass >= cloud_min_mass)
        & (cloud_mean_mass <= cloud_max_mass))
    number_before = (
        _host(rows, "nc_before_per_kg") + _host(rows, "qnn_before_per_kg"))
    number_after = (
        actual_by_name["nc_after_per_kg"]
        + actual_by_name["qnn_after_per_kg"])
    np.testing.assert_allclose(
        number_after[persistent], number_before[persistent],
        rtol=4.0e-6, atol=16.0)


def test_nucond_validation_and_inactive_dry_cell_noop():
    import cupy as cp

    from gpuwm.core.nssl2_nucond import launch_nucond

    shape = (3, 1, 2)
    theta = cp.full(shape, 280.0, dtype=cp.float32)
    density = cp.full(shape, 1.0, dtype=cp.float32)
    pressure = cp.full(shape, 100000.0, dtype=cp.float32)
    exner = cp.ones(shape, dtype=cp.float32)
    velocity = cp.zeros((4, 1, 2), dtype=cp.float32)
    qv = cp.full(shape, 1.0e-5, dtype=cp.float32)
    qc = cp.zeros(shape, dtype=cp.float32)
    qr = cp.zeros(shape, dtype=cp.float32)
    qi = cp.zeros(shape, dtype=cp.float32)
    qs = cp.zeros(shape, dtype=cp.float32)
    qndrop = cp.zeros(shape, dtype=cp.float32)
    qnr = cp.zeros(shape, dtype=cp.float32)
    qni = cp.zeros(shape, dtype=cp.float32)
    qns = cp.zeros(shape, dtype=cp.float32)
    qnn = cp.full(shape, 408163264.0, dtype=cp.float32)
    fields = [theta, density, pressure, exner, velocity,
              qv, qc, qr, qi, qs, qndrop, qnr, qni, qns, qnn]
    before = [field.copy() for field in fields]
    launch_nucond(*fields, 5.0)
    cp.cuda.Stream.null.synchronize()
    for actual, expected in zip(fields, before):
        cp.testing.assert_array_equal(actual, expected)

    bad_shape = list(fields)
    bad_shape[4] = cp.zeros(shape, dtype=cp.float32)
    with pytest.raises(ValueError, match="w_interface"):
        launch_nucond(*bad_shape, 5.0)
    bad_dtype = list(fields)
    bad_dtype[6] = cp.zeros(shape, dtype=cp.float64)
    with pytest.raises(TypeError, match="float32"):
        launch_nucond(*bad_dtype, 5.0)
    bad_value = list(fields)
    bad_value[9] = bad_value[9].copy()
    bad_value[9][0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        launch_nucond(*bad_value, 5.0)
    with pytest.raises(ValueError, match="positive finite"):
        launch_nucond(*fields, 0.0)
    with pytest.raises(ValueError, match="must not alias"):
        launch_nucond(
            *fields, 5.0, supersaturation_scratch=qv)


@pytest.mark.parametrize("frozen_category", ["ice", "snow"])
def test_nucond_frozen_water_blocks_clear_cloud_ccn_restoration(
        frozen_category):
    """Pin WRF's post-cleanup ice+snow gate on predicted-CCN restore."""
    import cupy as cp

    from gpuwm.core.nssl2_nucond import launch_nucond

    shape = (3, 1, 2)
    theta = cp.full(shape, 280.0, dtype=cp.float32)
    density = cp.ones(shape, dtype=cp.float32)
    pressure = cp.full(shape, 100000.0, dtype=cp.float32)
    exner = cp.ones(shape, dtype=cp.float32)
    velocity = cp.zeros((4, 1, 2), dtype=cp.float32)
    qv = cp.full(shape, 1.0e-5, dtype=cp.float32)
    qc = cp.zeros(shape, dtype=cp.float32)
    qr = cp.zeros(shape, dtype=cp.float32)
    qi = cp.zeros(shape, dtype=cp.float32)
    qs = cp.zeros(shape, dtype=cp.float32)
    qndrop = cp.zeros(shape, dtype=cp.float32)
    qnr = cp.zeros(shape, dtype=cp.float32)
    qni = cp.zeros(shape, dtype=cp.float32)
    qns = cp.zeros(shape, dtype=cp.float32)
    qnn = cp.full(shape, 2.0e8, dtype=cp.float32)
    if frozen_category == "ice":
        qi[:, 0, 0] = cp.float32(1.0e-6)
        qni[:, 0, 0] = cp.float32(1.0e5)
    else:
        qs[:, 0, 0] = cp.float32(1.0e-6)
        qns[:, 0, 0] = cp.float32(1.0e5)

    launch_nucond(
        theta, density, pressure, exner, velocity,
        qv, qc, qr, qi, qs, qndrop, qnr, qni, qns, qnn, 2.5,
        concentration_space=True)
    cp.cuda.Stream.null.synchronize()

    cp.testing.assert_array_equal(qnn[:, 0, 0], cp.float32(2.0e8))
    assert bool(cp.all(qnn[:, 0, 1] > cp.float32(2.0e8)))


@pytest.mark.parametrize("concentration_space", [False, True])
def test_radardd02_matches_compiled_official_wrf_all_categories(
        concentration_space):
    import cupy as cp

    from gpuwm.core.nssl2_diagnostics import launch_radardd02

    rows = _rows("radardd02.csv")
    shape = (4, 4, 4)

    def field(name):
        return cp.asarray(_host(rows, name).reshape(shape))

    density = field("rho_kg_m3")

    def moment(name):
        value = field(name)
        return value * density if concentration_space else value

    output = cp.empty(shape, dtype=cp.float32)
    assert launch_radardd02(
        density, field("temperature_k"),
        field("qr"), field("qi"), field("qs"),
        field("qg"), field("qh"),
        moment("nr_per_kg"), moment("ni_per_kg"), moment("ns_per_kg"),
        moment("ng_per_kg"), moment("nh_per_kg"),
        moment("qvolg_m3_per_kg"), moment("qvolh_m3_per_kg"), output,
        output_due=True, concentration_space=concentration_space)
    cp.cuda.Stream.null.synchronize()

    actual = cp.asnumpy(output).reshape(-1)
    expected = _host(rows, "refl_10cm_dbz")
    np.testing.assert_allclose(actual, expected, rtol=2.0e-6, atol=4.0e-5)
    # The ice-only fixture is physically active but remains below WRF's
    # native zero-dBZ reporting floor; the other single-category regimes and
    # the simultaneous all-category regime are positive.
    for case_id in (1, 3, 4, 5, 6, 7, 8):
        assert np.all(actual[_host(rows, "case") == case_id] > 0.0)


def test_radardd02_output_gate_and_validation(monkeypatch):
    import cupy as cp

    import gpuwm.core.nssl2_diagnostics as diagnostics

    sentinel = cp.full((1,), -123.0, dtype=cp.float32)
    monkeypatch.setattr(
        diagnostics, "get_kernel",
        lambda *_args: pytest.fail("non-output step loaded CUDA kernel"))
    assert not diagnostics.launch_radardd02(
        None, None, None, None, None, None, None,
        None, None, None, None, None, None, None, sentinel,
        output_due=False)
    cp.testing.assert_array_equal(sentinel, cp.float32(-123.0))
    with pytest.raises(TypeError, match="output_due"):
        diagnostics.launch_radardd02(
            None, None, None, None, None, None, None,
            None, None, None, None, None, None, None, sentinel,
            output_due=np.bool_(False))

    shape = (1, 1, 1)
    density = cp.ones(shape, dtype=cp.float32)
    temperature = cp.full(shape, 273.15, dtype=cp.float32)
    zero = cp.zeros(shape, dtype=cp.float32)
    with pytest.raises(ValueError, match="must not alias"):
        diagnostics.launch_radardd02(
            density, temperature,
            zero, zero, zero, zero, zero,
            zero, zero, zero, zero, zero, zero, zero, zero,
            output_due=True)


def test_effective_radius_adapter_converts_metres_to_rrtmgp_microns():
    import cupy as cp

    from gpuwm.core.nssl2_radiation import (
        launch_effective_radius_concentration,
        update_effective_radii,
    )

    rows = _rows("effective-radius.csv")
    shape = (3, 4, 4)

    class State:
        def __init__(self):
            self._scratch = {}

        def scratch(self, requested_shape, slot):
            if slot not in self._scratch:
                self._scratch[slot] = cp.zeros(
                    requested_shape, dtype=cp.float32)
            return self._scratch[slot]

    state = State()
    mapping = {
        "qc": "qc", "qndrop": "nc_per_kg",
        "qi": "qi", "qni": "ni_per_kg",
        "qs": "qs", "qns": "ns_per_kg",
    }
    for attr, column in mapping.items():
        setattr(state, attr, cp.asarray(_host(rows, column).reshape(shape)))
    for attr in ("effc", "effi", "effs"):
        setattr(state, attr, cp.full(shape, -1.0, dtype=cp.float32))
    density = cp.asarray(_host(rows, "rho_kg_m3").reshape(shape))

    assert update_effective_radii(state, air_density=density)
    cp.cuda.Stream.null.synchronize()
    for attr, column in (
            ("effc", "re_cloud_m"),
            ("effi", "re_ice_m"),
            ("effs", "re_snow_m")):
        expected = _host(rows, column).reshape(shape) * np.float32(1.0e6)
        np.testing.assert_allclose(
            cp.asnumpy(getattr(state, attr)), expected,
            rtol=2.5e-6, atol=2.0e-5)

    metre_outputs = [cp.empty(shape, dtype=cp.float32) for _ in range(3)]
    launch_effective_radius_concentration(
        density,
        state.qc, state.qndrop * density,
        state.qi, state.qni * density,
        state.qs, state.qns * density,
        *metre_outputs)
    cp.cuda.Stream.null.synchronize()
    for actual, column in zip(
            metre_outputs, ("re_cloud_m", "re_ice_m", "re_snow_m")):
        np.testing.assert_allclose(
            cp.asnumpy(actual), _host(rows, column).reshape(shape),
            rtol=2.0e-6, atol=2.0e-12)
    for actual, attr in zip(metre_outputs, ("effc", "effi", "effs")):
        cp.testing.assert_allclose(
            actual * cp.float32(1.0e6), getattr(state, attr),
            rtol=2.5e-6, atol=2.0e-5)

    with pytest.raises(ValueError, match="air_density"):
        update_effective_radii(state)

    before = state.effc.copy()
    assert not update_effective_radii(object(), radiation_due=False)
    cp.testing.assert_array_equal(state.effc, before)

"""CUDA checks against the unmodified WRF v4.6.1 RUC setup oracle."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance


ORACLE = (
    Path(__file__).parents[1]
    / "gpuwm" / "data" / "ruc" / "oracle" / "soilvegin.csv"
)


def _oracle_fields():
    with ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    return rows, {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        for name in rows[0]
        if name != "case"
    }


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("rdlai2d", [False, True])
def test_ruc_surface_cuda_matches_unmodified_wrf_bit_for_bit(rdlai2d):
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_surface_parameters_cuda

    rows, fields = _oracle_fields()
    selected = fields["rdlai2d"].astype(bool) == rdlai2d
    actual = ruc_surface_parameters_cuda(
        cp.asarray(fields["isltyp"][selected].astype(np.int32)),
        cp.asarray(fields["ivgtyp"][selected].astype(np.int32)),
        cp.asarray(fields["shdmin"][selected]),
        cp.asarray(fields["shdmax"][selected]),
        cp.asarray(fields["vegfrac"][selected]),
        cp.asarray(fields["znt_before"][selected]),
        cp.asarray(fields["lai_before"][selected]),
        rdlai2d=rdlai2d,
    )
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 6
    np.testing.assert_array_equal(
        cp.asnumpy(actual.iforest),
        fields["iforest"][selected].astype(np.int32),
    )
    for name in (
        "emiss", "pc", "znt", "lai", "qwrtz", "rhocs", "bclh",
        "dqm", "ksat", "psis", "qmin", "ref", "wilt",
    ):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            fields[name][selected].view(np.uint32),
            err_msg=name,
        )


@pytest.mark.gpu
@requires_gpu
def test_ruc_surface_cuda_rejects_shape_category_and_option_drift():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_surface_parameters_cuda

    _, fields = _oracle_fields()
    inputs = (
        cp.asarray(fields["isltyp"].astype(np.int32)),
        cp.asarray(fields["ivgtyp"].astype(np.int32)),
        *(cp.asarray(fields[name]) for name in (
            "shdmin", "shdmax", "vegfrac", "znt_before", "lai_before"
        )),
    )
    with pytest.raises(ValueError, match="mosaic_lu=0"):
        ruc_surface_parameters_cuda(*inputs, mosaic_lu=1)
    with pytest.raises(ValueError, match="expected"):
        ruc_surface_parameters_cuda(inputs[0], inputs[1][:-1], *inputs[2:])
    bad = inputs[1].copy()
    bad[0] = 22
    with pytest.raises(ValueError, match="ivgtyp 22"):
        ruc_surface_parameters_cuda(inputs[0], bad, *inputs[2:])


def _soilprop_oracle():
    path = ORACLE.with_name("soilprop.csv")
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profile_names = (
        "fwsat", "lwsat", "tav", "keepfr", "soilmois", "soiliqw",
        "soilice", "soilmoism", "soiliqwm", "soilicem", "thdif",
        "diffu", "hydro", "cap",
    )
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in profile_names
    }
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in ("qwrtz", "rhocs", "dqm", "qmin", "psis", "bclh", "ksat")
    }
    return rows, profiles, columns


@pytest.mark.gpu
@requires_gpu
def test_ruc_soil_properties_cuda_matches_unmodified_wrf_oracle():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_soil_properties_cuda

    rows, profiles, columns = _soilprop_oracle()
    inputs = {
        name: cp.asarray(profiles[name])
        for name in (
            "fwsat", "lwsat", "tav", "keepfr", "soilmois", "soiliqw",
            "soilice", "soilmoism", "soiliqwm", "soilicem",
        )
    }
    inputs.update({name: cp.asarray(value) for name, value in columns.items()})
    actual = ruc_soil_properties_cuda(inputs)
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 36
    for name in ("thdif", "diffu", "hydro", "cap"):
        result = cp.asnumpy(getattr(actual, name))
        np.testing.assert_allclose(
            result, profiles[name],
            rtol=8.0e-6, atol=2.0e-8, err_msg=name,
        )
        ulp = fp32_ulp_distance(result, profiles[name])
        assert int(np.max(ulp)) <= 1, (name, int(np.max(ulp)))
    np.testing.assert_array_equal(cp.asnumpy(actual.diffu[-1]), 0.0)
    np.testing.assert_array_equal(cp.asnumpy(actual.hydro[:, 3]), 0.0)


@pytest.mark.gpu
@requires_gpu
def test_ruc_soil_properties_cuda_rejects_contract_drift():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_soil_properties_cuda

    _, profiles, columns = _soilprop_oracle()
    inputs = {
        name: cp.asarray(profiles[name])
        for name in (
            "fwsat", "lwsat", "tav", "keepfr", "soilmois", "soiliqw",
            "soilice", "soilmoism", "soiliqwm", "soilicem",
        )
    }
    inputs.update({name: cp.asarray(value) for name, value in columns.items()})
    bad = dict(inputs)
    bad["tav"] = bad["tav"][:-1]
    with pytest.raises(ValueError, match="expected"):
        ruc_soil_properties_cuda(bad)
    bad = dict(inputs)
    bad["bclh"] = cp.zeros_like(bad["bclh"])
    with pytest.raises(ValueError, match="bclh must be positive"):
        ruc_soil_properties_cuda(bad)


def _transf_oracle():
    path = ORACLE.with_name("transf.csv")
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in ("soiliqw", "tranf")
    }
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in (
            "iland", "nroot", "tabs", "lai", "gswin", "dqm", "qmin", "ref",
            "wilt", "pc", "transum",
        )
    }
    return rows, profiles, columns


@pytest.mark.gpu
@requires_gpu
def test_ruc_transpiration_cuda_matches_unmodified_wrf_oracle():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_transpiration_cuda

    rows, profiles, columns = _transf_oracle()
    actual = ruc_transpiration_cuda(
        cp.asarray(profiles["soiliqw"]),
        *(cp.asarray(columns[name]) for name in (
            "tabs", "lai", "gswin", "dqm", "qmin", "ref", "wilt", "pc"
        )),
        cp.asarray(columns["iland"].astype(np.int32)),
        nroot=cp.asarray(columns["nroot"].astype(np.int32)),
    )
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 36
    for name, result, expected in (
        ("tranf", cp.asnumpy(actual.tranf), profiles["tranf"]),
        ("transum", cp.asnumpy(actual.transum), columns["transum"]),
    ):
        np.testing.assert_array_equal(
            result.view(np.uint32), expected.view(np.uint32), err_msg=name
        )


@pytest.mark.gpu
@requires_gpu
def test_ruc_transpiration_cuda_rejects_contract_drift():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_transpiration_cuda

    _, profiles, columns = _transf_oracle()
    args = (
        cp.asarray(profiles["soiliqw"]),
        *(cp.asarray(columns[name]) for name in (
            "tabs", "lai", "gswin", "dqm", "qmin", "ref", "wilt", "pc"
        )),
        cp.asarray(columns["iland"].astype(np.int32)),
    )
    with pytest.raises(ValueError, match="outside 1..8"):
        ruc_transpiration_cuda(*args, nroot=9)
    bad = args[-1].copy()
    bad[0] = 22
    with pytest.raises(ValueError, match="iland 22"):
        ruc_transpiration_cuda(*args[:-1], bad)


def _soilmoist_oracle():
    path = ORACLE.with_name("soilmoist.csv")
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in (
            "diffu", "hydro", "transp", "soilice", "soilmois_before",
            "soiliqw_before", "soilmois_after", "soiliqw_after",
        )
    }
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in (
            "delt", "qsg", "qvg", "qcg", "qcatm", "qvatm", "prcp",
            "qkms", "drip", "dew", "smelt", "vegfrac", "snowfrac",
            "soilres", "dqm", "qmin", "ref", "ksat", "ras", "mavail",
            "runoff", "runoff2", "infiltrp", "infmax",
        )
    }
    return rows, profiles, columns


@pytest.mark.gpu
@requires_gpu
def test_ruc_soil_moisture_cuda_matches_unmodified_wrf_oracle():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_soil_moisture_step_cuda

    rows, profiles, columns = _soilmoist_oracle()
    values = {
        name: cp.asarray(profiles[source])
        for name, source in (
            ("diffu", "diffu"), ("hydro", "hydro"),
            ("transp", "transp"), ("soilice", "soilice"),
            ("soilmois", "soilmois_before"),
            ("soiliqw", "soiliqw_before"),
        )
    }
    values.update({
        name: cp.asarray(columns[name])
        for name in (
            "qsg", "qvg", "qcg", "qcatm", "qvatm", "prcp", "qkms",
            "drip", "dew", "smelt", "vegfrac", "snowfrac", "soilres",
            "dqm", "qmin", "ref", "ksat", "ras",
        )
    })
    actual = ruc_soil_moisture_step_cuda(values, delt=60.0)
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 36
    for name, expected in (
        ("soilmois", profiles["soilmois_after"]),
        ("soiliqw", profiles["soiliqw_after"]),
        *((name, columns[name]) for name in (
            "mavail", "runoff", "runoff2", "infiltrp", "infmax"
        )),
    ):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            expected.view(np.uint32),
            err_msg=name,
        )


@pytest.mark.gpu
@requires_gpu
def test_ruc_soil_moisture_cuda_rejects_contract_drift():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_soil_moisture_step_cuda

    _, profiles, columns = _soilmoist_oracle()
    values = {
        "diffu": cp.asarray(profiles["diffu"]),
        "hydro": cp.asarray(profiles["hydro"]),
        "transp": cp.asarray(profiles["transp"]),
        "soilice": cp.asarray(profiles["soilice"]),
        "soilmois": cp.asarray(profiles["soilmois_before"]),
        "soiliqw": cp.asarray(profiles["soiliqw_before"]),
    }
    values.update({
        name: cp.asarray(columns[name])
        for name in (
            "qsg", "qvg", "qcg", "qcatm", "qvatm", "prcp", "qkms",
            "drip", "dew", "smelt", "vegfrac", "snowfrac", "soilres",
            "dqm", "qmin", "ref", "ksat", "ras",
        )
    })
    bad = dict(values)
    bad["diffu"] = bad["diffu"][:-1]
    with pytest.raises(ValueError, match="must have shape"):
        ruc_soil_moisture_step_cuda(bad, delt=60.0)
    with pytest.raises(ValueError, match="positive"):
        ruc_soil_moisture_step_cuda(values, delt=0.0)


def _frozen_infiltration_columns(ncolumn: int, *, seed: int):
    """Frozen soil AND liquid water arriving at the surface, both at once.

    That pair is what ``soilmoist``'s frozen-ground reduction needs to reach
    an output, and it is the one state the pinned fixture cannot supply; see
    :func:`test_ruc_soil_moisture_cuda_agrees_off_fixture`.
    """

    rng = np.random.default_rng(seed)

    def spread(base, width, low=None, high=None):
        out = (base + width * rng.standard_normal(ncolumn)).astype(np.float32)
        if low is not None or high is not None:
            out = np.clip(out, low, high).astype(np.float32)
        return out

    def profile(base, width, low=None, high=None):
        return np.stack([spread(base, width, low, high) for _ in range(9)])

    dqm = spread(0.417, 0.02, 0.30, 0.48)
    values = {
        "diffu": profile(2.0e-6, 8.0e-7, 0.0, None),
        "hydro": profile(4.0e-7, 2.0e-7, 0.0, None),
        "transp": profile(-4.0e-9, 2.0e-9),
        # riw = 0.9, so the frozen fraction is soilice*riw of the pore water.
        "soilice": profile(0.20, 0.05, 0.02, 0.34),
        "soilmois": profile(0.30, 0.03, 0.10, 0.44),
        "soiliqw": profile(0.09, 0.02, 0.01, 0.20),
        "qsg": spread(0.0042, 4.0e-4, 1.0e-4, None),
        "qvg": spread(0.0031, 3.0e-4, 1.0e-4, None),
        "qcg": spread(0.0, 0.0),
        "qcatm": spread(0.0, 0.0),
        "qvatm": spread(0.0033, 4.0e-4, 1.0e-4, None),
        # prcp is -infwater: negative is water going IN, which is what makes
        # px = max(0, -total*delt) positive and lets fcr reach infmax.
        "prcp": spread(-2.0e-5, 8.0e-6, None, -1.0e-6),
        "qkms": spread(0.006, 0.002, 1.0e-4, None),
        "drip": spread(0.0, 0.0),
        "dew": spread(0.0, 0.0),
        "smelt": spread(-6.0e-6, 3.0e-6, None, 0.0),
        "vegfrac": spread(0.35, 0.10, 0.0, 0.95),
        "snowfrac": spread(0.6, 0.2, 0.0, 1.0),
        "soilres": spread(1.0, 0.0),
        "dqm": dqm,
        "qmin": spread(0.06, 0.005, 0.01, 0.12),
        "ref": spread(0.357, 0.02, 0.20, 0.44),
        "ksat": spread(1.7e-6, 5.0e-7, 1.0e-8, None),
        "ras": spread(0.0012, 1.0e-4, 1.0e-5, None),
    }
    values["soiliqw"] = np.minimum(
        values["soiliqw"], values["soilmois"]).astype(np.float32)
    return values


@pytest.mark.gpu
@requires_gpu
def test_ruc_soil_moisture_cuda_agrees_off_fixture():
    """The frozen-ground infiltration arm, which no fixture case can reach.

    ``soilmoist.csv`` has four cases.  Three carry ``soilice = 0`` throughout,
    so ``fdmax > 1.e-2`` (``module_sf_ruclsm.F:5972``) is false and the ACRT
    reduction at ``:5975-5985`` never runs.  The fourth does run it -- and
    then multiplies its answer by an ``infmax1`` of exactly zero, because its
    only surface water is a melt term of the wrong sign, so
    ``px = max(0, -total*delt)`` is zero.  The arm is evaluated exactly once
    in the entire fixture and its result is annihilated before it reaches an
    output.

    That is how every RUC CUDA leaf could be ``max_ulp 0`` against its own
    fixture while the composed device column moved 10 ULP of ``infiltr`` on a
    snow grid: **the oracle was blind to the branch that mattered.**  The
    cause was ``ruc.cu``'s ``powf(acrt, 2.0f)`` -- the one ``**`` site left on
    the CUDA device libm while the host used the float64-evaluated,
    rounded-once form the rest of the file agreed on -- amplified by the
    ``1 - exp(-acrt)*sum`` cancellation at ``:5985``.

    This gate samples the pair the fixture cannot: frozen soil AND liquid
    water arriving at the surface.  It also asserts that the state generator
    actually reaches the arm, because a generator that quietly stopped
    reaching it would make this test pass for nothing.
    """

    import cupy as cp

    from gpuwm.core.ruc import ruc_soil_moisture_step
    from gpuwm.core.ruc_gpu import ruc_soil_moisture_step_cuda

    outputs = ("soilmois", "soiliqw", "mavail", "runoff", "runoff2",
               "infiltrp", "infmax")
    reached = 0
    for seed in range(6):
        values = _frozen_infiltration_columns(256, seed=20260726 + seed)

        # the arm is reached only where BOTH conditions hold; assert it does
        zshalf = np.zeros(9, np.float32)
        depths = np.asarray(
            [0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00], np.float32)
        for level in range(1, 9):
            zshalf[level] = np.float32(
                np.float32(depths[level - 1] + depths[level]) * np.float32(0.5))
        frozen = (values["soilice"][0] * zshalf[1]).astype(np.float32)
        for level in range(1, 8):
            frozen = (frozen + values["soilice"][level]
                      * np.float32(zshalf[level + 1] - zshalf[level])
                      ).astype(np.float32)
        umveg = ((np.float32(1.0) - values["vegfrac"])
                 * values["soilres"]).astype(np.float32)
        total = (values["prcp"] - values["drip"] / np.float32(60.0)
                 - umveg * values["dew"] * values["ras"]
                 - values["smelt"]).astype(np.float32)
        live = (frozen > np.float32(1.0e-2)) & (total < np.float32(0.0))
        reached += int(live.sum())

        host = ruc_soil_moisture_step(values, delt=60.0)
        device = ruc_soil_moisture_step_cuda(
            {name: cp.asarray(array) for name, array in values.items()},
            delt=60.0)
        cp.cuda.get_current_stream().synchronize()
        worst = {}
        for name in outputs:
            gap = int(fp32_ulp_distance(
                np.asarray(getattr(host, name)),
                cp.asnumpy(getattr(device, name))).max())
            if gap:
                worst[name] = gap
        assert worst == {}, (
            f"seed {seed}: the CUDA soilmoist left the host transcription on "
            f"the frozen infiltration arm: {worst}.  Acceptance is max_ulp 0; "
            "report the number, never widen the gate")

    assert reached > 1000, (
        f"the state generator reached the frozen infiltration arm on only "
        f"{reached} columns; this gate is passing for nothing")


def _soiltemp_oracle():
    path = ORACLE.with_name("soiltemp.csv")
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in ("thdif", "cap", "tso_before", "tso_after")
    }
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in (
            "delt", "conflx", "nroot", "prcpms", "rainf", "patm", "tabs",
            "qvatm", "qcatm", "emiss", "rnet", "qkms", "tkms", "pc",
            "rho", "vegfrac", "lai", "drycan", "wetcan", "transum",
            "dew", "mavail", "soilres", "alfa", "dqm", "qmin", "bclh",
            "soilt_before", "soilt_after", "qvg_before", "qvg_after",
            "qsg_after", "qcg_after", "storage",
        )
    }
    return rows, profiles, columns


def _soiltemp_inputs(cp, profiles, columns):
    values = {
        "thdif": cp.asarray(profiles["thdif"]),
        "cap": cp.asarray(profiles["cap"]),
        "tso": cp.asarray(profiles["tso_before"]),
    }
    values.update({
        name: cp.asarray(columns[source])
        for name, source in (
            ("prcpms", "prcpms"), ("rainf", "rainf"), ("patm", "patm"),
            ("tabs", "tabs"), ("qvatm", "qvatm"), ("qcatm", "qcatm"),
            ("emiss", "emiss"), ("rnet", "rnet"), ("qkms", "qkms"),
            ("tkms", "tkms"), ("pc", "pc"), ("rho", "rho"),
            ("vegfrac", "vegfrac"), ("lai", "lai"),
            ("drycan", "drycan"), ("wetcan", "wetcan"),
            ("transum", "transum"), ("dew", "dew"),
            ("mavail", "mavail"), ("soilres", "soilres"),
            ("alfa", "alfa"), ("dqm", "dqm"), ("qmin", "qmin"),
            ("bclh", "bclh"), ("soilt", "soilt_before"),
            ("qvg", "qvg_before"), ("qsg", "qsg_after"),
            ("qcg", "qcg_after"),
        )
    })
    return values


@pytest.mark.gpu
@requires_gpu
def test_ruc_soil_temperature_cuda_matches_unmodified_wrf_oracle():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_soil_temperature_step_cuda

    rows, profiles, columns = _soiltemp_oracle()
    actual = ruc_soil_temperature_step_cuda(
        _soiltemp_inputs(cp, profiles, columns),
        delt=60.0,
        conflx=0.5,
        nroot=cp.asarray(columns["nroot"].astype(np.int32)),
    )
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 36
    for name, expected in (
        ("tso", profiles["tso_after"]),
        *((name, columns[f"{name}_after"]) for name in (
            "soilt", "qvg", "qsg", "qcg"
        )),
        ("storage", columns["storage"]),
    ):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            expected.view(np.uint32),
            err_msg=name,
        )


@pytest.mark.gpu
@requires_gpu
def test_ruc_soil_temperature_cuda_rejects_contract_drift():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_soil_temperature_step_cuda

    _, profiles, columns = _soiltemp_oracle()
    values = _soiltemp_inputs(cp, profiles, columns)
    bad = dict(values)
    bad["cap"] = bad["cap"][:-1]
    with pytest.raises(ValueError, match="expected"):
        ruc_soil_temperature_step_cuda(bad, delt=60.0)
    with pytest.raises(ValueError, match="positive"):
        ruc_soil_temperature_step_cuda(values, delt=0.0)


def _soil_oracle():
    path = ORACLE.with_name("soil.csv")
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    profile_names = (
        "soilmois_before", "soilmois_after", "tso_before", "tso_after",
        "smfrkeep_before", "smfrkeep_after", "keepfr_before",
        "keepfr_after", "soilice", "soiliqw",
    )
    profiles = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(4, 9).T
        for name in profile_names
    }
    columns = {
        name: np.asarray(
            [float(rows[index * 9][name]) for index in range(4)],
            dtype=np.float32,
        )
        for name in rows[0]
        if name not in ("case", "k", *profile_names)
    }
    return rows, profiles, columns


def _soil_inputs(cp, profiles, columns):
    from gpuwm.core.ruc import RUC_SOIL_STEP_COLUMN_INPUTS

    values = {
        "soilmois": cp.asarray(profiles["soilmois_before"]),
        "tso": cp.asarray(profiles["tso_before"]),
        "smfrkeep": cp.asarray(profiles["smfrkeep_before"]),
        "keepfr": cp.asarray(profiles["keepfr_before"]),
    }
    before = {
        "cst": "cst_before", "soilt": "soilt_before",
        "qvg": "qvg_before", "qsg": "qsg_before", "qcg": "qcg_before",
        "mavail": "mavail_before",
    }
    values.update({
        name: cp.asarray(columns[before.get(name, name)])
        for name in RUC_SOIL_STEP_COLUMN_INPUTS
    })
    return values


@pytest.mark.gpu
@requires_gpu
def test_complete_snow_free_soil_step_cuda_matches_unmodified_wrf_oracle():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_soil_step_cuda

    rows, profiles, columns = _soil_oracle()
    actual = ruc_soil_step_cuda(
        _soil_inputs(cp, profiles, columns),
        cp.asarray(columns["iland"].astype(np.int32)),
        nroot=cp.asarray(columns["nroot"].astype(np.int32)),
        delt=float(columns["delt"][0]),
        conflx=float(columns["conflx"][0]),
    )
    cp.cuda.get_current_stream().synchronize()
    assert len(rows) == 36
    comparisons = (
        ("soilmois", profiles["soilmois_after"]),
        ("tso", profiles["tso_after"]),
        ("smfrkeep", profiles["smfrkeep_after"]),
        ("keepfr", profiles["keepfr_after"]),
        ("soilice", profiles["soilice"]),
        ("soiliqw", profiles["soiliqw"]),
        *(({
            "cst_after": "cst", "soilt_after": "soilt",
            "qvg_after": "qvg", "qsg_after": "qsg", "qcg_after": "qcg",
            "mavail_after": "mavail",
        }.get(name, name), columns[name]) for name in (
            "cst_after", "dew", "soilt_after", "qvg_after", "qsg_after",
            "qcg_after", "edir1", "ec1", "ett1", "eeta", "qfx", "hfx",
            "s", "evapl", "prcpl", "fltot", "runoff1", "runoff2",
            "mavail_after", "infiltrp", "smf",
        )),
    )
    max_ulp = {}
    for name, expected in comparisons:
        result = cp.asnumpy(getattr(actual, name))
        max_ulp[name] = int(np.max(fp32_ulp_distance(result, expected)))
        if name == "fltot":
            # This residual subtracts four O(100 W m-2) terms.  A one-ULP
            # CUDA/Fortran temperature difference is amplified by
            # cancellation, while the absolute energy error stays tiny.
            np.testing.assert_allclose(result, expected, rtol=0.0, atol=0.04)
    ulp_limits = {
        "soilmois": 0, "tso": 1, "smfrkeep": 0, "keepfr": 0,
        "soilice": 0, "soiliqw": 0, "cst": 32, "dew": 32,
        "soilt": 1, "qvg": 32, "qsg": 32, "qcg": 32,
        "edir1": 64, "ec1": 64, "ett1": 64, "eeta": 64,
        "qfx": 64, "hfx": 256, "s": 1024, "evapl": 64,
        "prcpl": 0, "runoff1": 128, "runoff2": 0, "mavail": 0,
        "infiltrp": 32, "smf": 0,
    }
    for name, limit in ulp_limits.items():
        assert max_ulp[name] <= limit, (name, max_ulp[name], limit)


@pytest.mark.gpu
@requires_gpu
def test_complete_snow_free_soil_step_cuda_preserves_inputs_and_fails_closed():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_soil_step_cuda

    _, profiles, columns = _soil_oracle()
    values = _soil_inputs(cp, profiles, columns)
    originals = {
        name: cp.asnumpy(values[name]).copy()
        for name in ("soilmois", "tso", "smfrkeep", "keepfr")
    }
    land = cp.asarray(columns["iland"].astype(np.int32))
    roots = cp.asarray(columns["nroot"].astype(np.int32))
    ruc_soil_step_cuda(
        values, land, nroot=roots, delt=60.0, conflx=0.5
    )
    cp.cuda.get_current_stream().synchronize()
    for name, expected in originals.items():
        np.testing.assert_array_equal(cp.asnumpy(values[name]), expected)

    bad = dict(values)
    bad["tso"] = bad["tso"][:-1]
    with pytest.raises(ValueError, match="expected"):
        ruc_soil_step_cuda(bad, land, nroot=roots, delt=60.0, conflx=0.5)
    with pytest.raises(ValueError, match="outside 1..8"):
        ruc_soil_step_cuda(values, land, nroot=9, delt=60.0, conflx=0.5)
    with pytest.raises(ValueError, match="myj=False"):
        ruc_soil_step_cuda(
            values, land, nroot=roots, delt=60.0, conflx=0.5, myj=True
        )


def _qsn_oracle():
    path = ORACLE.with_name("qsn.csv")
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        for name in ("tn", "r_raw", "qsn")
    }
    labels = np.asarray([row["case"] for row in rows])
    return rows, labels, fields


@pytest.mark.gpu
@requires_gpu
def test_ruc_qsn_cuda_matches_unmodified_wrf_bit_for_bit():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_qsn_cuda

    rows, labels, fields = _qsn_oracle()
    assert len(rows) == 81
    assert set(labels) == {
        "clamp_low", "node_low", "interior", "node_high", "clamp_high",
    }
    actual = cp.asnumpy(ruc_qsn_cuda(cp.asarray(fields["tn"])))
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_array_equal(
        actual.view(np.uint32), fields["qsn"].view(np.uint32), err_msg="qsn"
    )


@pytest.mark.gpu
@requires_gpu
def test_ruc_qsn_cuda_clamps_both_table_ends_and_rejects_contract_drift():
    import cupy as cp

    from gpuwm.core.ruc import ruc_saturation_table
    from gpuwm.core.ruc_gpu import ruc_qsn_cuda

    _, labels, fields = _qsn_oracle()
    table = cp.asarray(ruc_saturation_table())
    actual = cp.asnumpy(ruc_qsn_cuda(cp.asarray(fields["tn"]), table))
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_array_equal(
        actual.view(np.uint32), fields["qsn"].view(np.uint32)
    )
    # Below the first node WRF forces i=1, r=1 -> tbq(1); at or above the
    # last node it forces i=5000, r=5001 -> tbq(5001).
    host_table = cp.asnumpy(table)
    np.testing.assert_array_equal(actual[labels == "clamp_low"], host_table[0])
    np.testing.assert_array_equal(actual[labels == "clamp_high"], host_table[-1])
    np.testing.assert_array_equal(actual[labels == "node_low"], host_table[0])
    np.testing.assert_array_equal(actual[labels == "node_high"], host_table[-1])
    assert np.all(np.diff(actual[labels == "interior"]) > 0.0)

    shaped = ruc_qsn_cuda(cp.full((3, 4), np.float32(273.15)))
    assert shaped.shape == (3, 4) and shaped.dtype == np.float32
    np.testing.assert_array_equal(
        cp.asnumpy(shaped),
        cp.asnumpy(ruc_qsn_cuda(cp.asarray(np.float32(273.15)))),
    )
    with pytest.raises(ValueError, match="finite"):
        ruc_qsn_cuda(cp.asarray(np.float32("nan")))
    with pytest.raises(ValueError, match=r"shape \(5001,\)"):
        ruc_qsn_cuda(cp.asarray(np.float32(273.15)), table[:-1])


def _sice_oracle():
    path = ORACLE.with_name("sice.csv")
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(8, 9).T
        for name in rows[0]
        if name != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    return rows, cases, fields


def _sice_inputs(cp, fields, selection):
    from gpuwm.core.ruc import RUC_SEA_ICE_COLUMN_INPUTS

    values = {
        name: cp.asarray(np.ascontiguousarray(fields[name][:, selection]))
        for name in ("capice", "thdifice")
    }
    values["tso"] = cp.asarray(
        np.ascontiguousarray(fields["tso_before"][:, selection])
    )
    before = {"soilt": "soilt_before", "qvg": "qvg_before", "qsg": "qsg_before"}
    values.update({
        name: cp.asarray(
            np.ascontiguousarray(fields[before.get(name, name)][0, selection])
        )
        for name in RUC_SEA_ICE_COLUMN_INPUTS
    })
    return values


_SICE_PROFILE_OUTPUTS = (
    "tso", "soilmois", "soiliqw", "soilice", "smfrkeep", "keepfr",
)
_SICE_COLUMN_OUTPUTS = (
    "dew", "soilt", "qvg", "qsg", "qcg", "eeta", "qfx", "hfx",
    "s", "evapl", "prcpl", "fltot",
)


@pytest.mark.gpu
@requires_gpu
def test_ruc_sea_ice_step_cuda_matches_unmodified_wrf_bit_for_bit():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_sea_ice_step_cuda

    rows, cases, fields = _sice_oracle()
    assert len(rows) == 72
    assert cases == (
        "cold_thick_ice", "near_melt_cap", "thin_ice_warm_base",
        "strong_forcing", "dew_condensing", "myj_condensing",
        "myj_evaporating", "rain_on_ice",
    )
    for case in range(len(cases)):
        actual = ruc_sea_ice_step_cuda(
            _sice_inputs(cp, fields, [case]),
            delt=float(fields["delt"][0, case]),
            conflx=float(fields["conflx"][0, case]),
            myj=bool(fields["myj"][0, case] > 0.5),
            cw=float(fields["cw"][0, case]),
        )
        cp.cuda.get_current_stream().synchronize()
        for name in _SICE_PROFILE_OUTPUTS:
            expected = fields["tso_after" if name == "tso" else f"{name}_after"]
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(actual, name))[:, 0].view(np.uint32),
                expected[:, case].view(np.uint32),
                err_msg=f"{cases[case]}/{name}",
            )
        for name in _SICE_COLUMN_OUTPUTS:
            expected = fields.get(f"{name}_after", fields.get(name))
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(actual, name)).view(np.uint32),
                expected[0, case : case + 1].view(np.uint32),
                err_msg=f"{cases[case]}/{name}",
            )


@pytest.mark.gpu
@requires_gpu
def test_ruc_sea_ice_step_cuda_solves_independent_columns_in_one_launch():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_sea_ice_step_cuda

    _, cases, fields = _sice_oracle()
    # Every case that shares this lane's scalar arguments, batched together:
    # the per-case launches above cannot catch a column-stride defect.
    selection = [
        case for case in range(len(cases))
        if fields["conflx"][0, case] == np.float32(40.0)
        and fields["myj"][0, case] < np.float32(0.5)
    ]
    assert len(selection) == 5
    actual = ruc_sea_ice_step_cuda(
        _sice_inputs(cp, fields, selection),
        delt=60.0,
        conflx=40.0,
        myj=False,
        cw=4.183e6,
    )
    cp.cuda.get_current_stream().synchronize()
    for name in _SICE_PROFILE_OUTPUTS:
        expected = fields["tso_after" if name == "tso" else f"{name}_after"]
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            expected[:, selection].view(np.uint32),
            err_msg=name,
        )
    for name in _SICE_COLUMN_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            expected[0, selection].view(np.uint32),
            err_msg=name,
        )


@pytest.mark.gpu
@requires_gpu
def test_ruc_sea_ice_step_cuda_forces_ice_state_and_rejects_contract_drift():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_sea_ice_step_cuda

    _, cases, fields = _sice_oracle()
    assert cases[1] == "near_melt_cap"
    values = _sice_inputs(cp, fields, [1])
    originals = {
        name: cp.asnumpy(value).copy() for name, value in values.items()
    }
    actual = ruc_sea_ice_step_cuda(values, delt=60.0)
    cp.cuda.get_current_stream().synchronize()
    for name, expected in originals.items():
        np.testing.assert_array_equal(cp.asnumpy(values[name]), expected)

    # sice never writes the soil water arrays; sfctmp forces them to
    # 1/0/1/1/0 after both call sites and the port folds that in.
    np.testing.assert_array_equal(cp.asnumpy(actual.soilmois), np.float32(1.0))
    np.testing.assert_array_equal(cp.asnumpy(actual.soiliqw), np.float32(0.0))
    np.testing.assert_array_equal(cp.asnumpy(actual.soilice), np.float32(1.0))
    np.testing.assert_array_equal(cp.asnumpy(actual.smfrkeep), np.float32(1.0))
    np.testing.assert_array_equal(cp.asnumpy(actual.keepfr), np.float32(0.0))
    np.testing.assert_array_equal(cp.asnumpy(actual.qcg), np.float32(0.0))
    # icemelt absorbs the whole surface residual, so fltot cancels exactly.
    np.testing.assert_array_equal(cp.asnumpy(actual.fltot), np.float32(0.0))
    # The 271.4 K sea-ice cap binds on this regime and holds everywhere.
    assert bool(cp.all(actual.tso <= cp.float32(271.4)))
    np.testing.assert_array_equal(cp.asnumpy(actual.soilt), np.float32(271.4))

    bad = dict(values)
    bad["thdifice"] = bad["thdifice"][:-1]
    with pytest.raises(ValueError, match="expected"):
        ruc_sea_ice_step_cuda(bad, delt=60.0)
    bad = dict(values)
    bad["capice"] = bad["capice"][:-1]
    with pytest.raises(ValueError, match=r"shape \(6 or 9"):
        ruc_sea_ice_step_cuda(bad, delt=60.0)
    with pytest.raises(ValueError, match="positive"):
        ruc_sea_ice_step_cuda(values, delt=0.0)
    with pytest.raises(TypeError, match="myj"):
        ruc_sea_ice_step_cuda(values, delt=60.0, myj=1)
    incomplete = {
        name: value for name, value in values.items() if name != "rnet"
    }
    with pytest.raises(TypeError, match="missing RUC CUDA sice inputs: rnet"):
        ruc_sea_ice_step_cuda(incomplete, delt=60.0)


# --------------------------------------------------------------------------
# WRF v4.6.1 sfctmp snow preparation (phys/module_sf_ruclsm.F:1400-1766)
# --------------------------------------------------------------------------

SNOW_PREP_ORACLE = (
    Path(__file__).parents[1]
    / "gpuwm" / "data" / "ruc" / "oracle" / "sfctmp_prep.csv"
)
SNOW_PREP_NCASE = 17
SNOW_PREP_DATASETS = {15: "MODIFIED_IGBP_MODIS_NOAH", 24: "USGS"}
SNOW_PREP_BEFORE = {
    "snwe": "snwe_before",
    "snhei": "snhei_before",
    "snowfrac": "snowfrac_before",
    "rhosn": "rhosn_before",
    "rhosnfall": "rhosnfall_before",
    "cst": "cst_before",
    "alb": "alb_before",
    "emiss": "emiss_before",
    "znt": "znt_before",
}
SNOW_PREP_AFTER = {
    **{name: column.replace("_before", "_after")
       for name, column in SNOW_PREP_BEFORE.items()},
    "iland": "iland_after",
}


def _snow_prep_oracle():
    with SNOW_PREP_ORACLE.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == SNOW_PREP_NCASE * 9
    return {
        name: np.asarray(
            [float(row[name]) for row in rows], dtype=np.float32
        ).reshape(SNOW_PREP_NCASE, 9).T
        for name in rows[0]
        if name != "case"
    }


def _snow_prep_case(cp, field, case, column_inputs):
    isice = int(field["isice"][0, case])
    values = {"ts1d": cp.asarray(field["ts1d"][:, case : case + 1])}
    values.update({
        name: cp.asarray(field[SNOW_PREP_BEFORE.get(name, name)][0, case : case + 1])
        for name in column_inputs
    })
    keywords = {
        "delt": float(field["delt"][0, case]),
        "ivgtyp": cp.asarray([int(field["ivgtyp"][0, case])], dtype=cp.int32),
        "iland": cp.asarray(
            [int(field["iland_before"][0, case])], dtype=cp.int32
        ),
        "isice": isice,
        "c1sn": float(field["c1sn"][0, case]),
        "c2sn": float(field["c2sn"][0, case]),
        "mminlu": SNOW_PREP_DATASETS[isice],
    }
    return values, keywords


@pytest.mark.gpu
@requires_gpu
def test_snow_preparation_cuda_matches_unmodified_wrf_bit_for_bit():
    import cupy as cp

    from gpuwm.core.ruc import (
        RUC_SNOW_PREP_COLUMN_INPUTS,
        RUC_SNOW_PREP_COLUMN_OUTPUTS,
        RUC_SNOW_PREP_PROFILE_OUTPUTS,
    )
    from gpuwm.core.ruc_gpu import ruc_snow_preparation_cuda

    field = _snow_prep_oracle()
    assert list(dict.fromkeys(field["isice"][0])) == [15, 24]
    for case in range(SNOW_PREP_NCASE):
        values, keywords = _snow_prep_case(
            cp, field, case, RUC_SNOW_PREP_COLUMN_INPUTS
        )
        originals = {
            name: cp.asnumpy(value).copy() for name, value in values.items()
        }
        actual = ruc_snow_preparation_cuda(values, **keywords)
        cp.cuda.get_current_stream().synchronize()
        for name, expected in originals.items():
            np.testing.assert_array_equal(cp.asnumpy(values[name]), expected)
        for name in RUC_SNOW_PREP_PROFILE_OUTPUTS:
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(actual, name))[:, 0].view(np.uint32),
                field[name][:, case].view(np.uint32),
                err_msg=f"case {case + 1}/{name}",
            )
        for name in RUC_SNOW_PREP_COLUMN_OUTPUTS + ("iland",):
            if (
                name in ("keep_snow_albedo", "snowfrac2")
                and field["snhei_after"][0, case] <= 0.0
            ):
                # WRF leaves both undefined when the snhei>0. branch is
                # skipped, and the fixture records nan there.
                continue
            expected = field[SNOW_PREP_AFTER.get(name, name)][0, case]
            got = np.float32(cp.asnumpy(getattr(actual, name))[0])
            np.testing.assert_array_equal(
                got.view(np.uint32),
                expected.view(np.uint32),
                err_msg=f"case {case + 1}/{name}",
            )


@pytest.mark.gpu
@requires_gpu
def test_snow_preparation_cuda_matches_cpu_across_all_cover_options():
    """The two unverifiable snow-cover options must at least agree.

    ``module_sf_ruclsm.F:78`` pins ``isncovr_opt=2``, so options 1 and 3 have
    no oracle.  Holding the CPU and CUDA transcriptions bitwise-identical on
    them is the strongest available check.
    """

    import cupy as cp

    from gpuwm.core.ruc import (
        RUC_SNOW_PREP_COLUMN_INPUTS,
        RUC_SNOW_PREP_COLUMN_OUTPUTS,
        RUC_SNOW_PREP_PROFILE_OUTPUTS,
        ruc_snow_preparation,
    )
    from gpuwm.core.ruc_gpu import ruc_snow_preparation_cuda

    field = _snow_prep_oracle()
    for option in (1, 2, 3):
        for case in range(SNOW_PREP_NCASE):
            device, keywords = _snow_prep_case(
                cp, field, case, RUC_SNOW_PREP_COLUMN_INPUTS
            )
            host = {
                name: cp.asnumpy(value) for name, value in device.items()
            }
            host_keywords = dict(keywords)
            host_keywords["ivgtyp"] = cp.asnumpy(keywords["ivgtyp"])
            host_keywords["iland"] = cp.asnumpy(keywords["iland"])
            expected = ruc_snow_preparation(
                host, **host_keywords, isncovr_opt=option
            )
            actual = ruc_snow_preparation_cuda(
                device, **keywords, isncovr_opt=option
            )
            cp.cuda.get_current_stream().synchronize()
            for name in (
                RUC_SNOW_PREP_PROFILE_OUTPUTS
                + RUC_SNOW_PREP_COLUMN_OUTPUTS
                + ("iland",)
            ):
                np.testing.assert_array_equal(
                    cp.asnumpy(getattr(actual, name)),
                    np.asarray(getattr(expected, name)),
                    err_msg=f"option {option}/case {case + 1}/{name}",
                )


@pytest.mark.gpu
@requires_gpu
def test_snow_preparation_kernel_is_insensitive_to_fma_contraction():
    """Every a*b+c boundary in the kernel is pinned.

    The kernel is rebuilt with ``-fmad=true`` and with ``-fmad=false`` and the
    two must agree bit for bit.  If any expression were left un-pinned, letting
    the compiler contract it would move the result -- in a sibling RUC lane an
    un-pinned boundary moved one output by 559,240 ULP.
    """

    import cupy as cp

    from gpuwm.core.constants import CUDA_DEFINES
    from gpuwm.core.ruc import (
        RUC_SNOW_PREP_COLUMN_INPUTS,
        RUC_SNOW_PREP_COLUMN_OUTPUTS,
        RUC_SNOW_PREP_PROFILE_OUTPUTS,
    )
    from gpuwm.core.ruc_gpu import _snow_preparation_tables
    from gpuwm.core.state import DTYPE

    kernels = Path(__file__).parents[1] / "gpuwm" / "core" / "kernels"
    preamble = "\n".join(
        f"#define {key} {float(value)!r}f"
        for key, value in CUDA_DEFINES.items()
    )
    source = (
        preamble
        + "\n"
        + (kernels / "common.cuh").read_text()
        + "\n"
        + (kernels / "ruc.cu").read_text()
    )
    variants = {}
    for flag in ("-fmad=true", "-fmad=false"):
        module = cp.RawModule(code=source, options=("-std=c++17", flag))
        module.compile()
        variants[flag] = module.get_function("ruc_snow_preparation")

    field = _snow_prep_oracle()
    device = int(cp.cuda.runtime.getDevice())
    results = {}
    for flag, kernel in variants.items():
        collected = {}
        for case in range(SNOW_PREP_NCASE):
            isice = int(field["isice"][0, case])
            roughness, emissivity, urban, _ = _snow_preparation_tables(
                device, SNOW_PREP_DATASETS[isice]
            )
            ts1d = cp.ascontiguousarray(
                cp.asarray(field["ts1d"][:, case : case + 1], dtype=DTYPE)
            )
            columns = [
                cp.ascontiguousarray(
                    cp.asarray(
                        field[SNOW_PREP_BEFORE.get(name, name)][
                            0, case : case + 1
                        ],
                        dtype=DTYPE,
                    )
                )
                for name in RUC_SNOW_PREP_COLUMN_INPUTS
            ]
            profiles = {
                name: cp.empty((9, 1), dtype=DTYPE)
                for name in RUC_SNOW_PREP_PROFILE_OUTPUTS
            }
            outputs = {
                name: cp.empty((1,), dtype=DTYPE)
                for name in RUC_SNOW_PREP_COLUMN_OUTPUTS
            }
            land = cp.empty((1,), dtype=cp.int32)
            kernel(
                (1,),
                (128,),
                (
                    ts1d,
                    *columns,
                    cp.asarray([int(field["ivgtyp"][0, case])], dtype=cp.int32),
                    cp.asarray(
                        [int(field["iland_before"][0, case])], dtype=cp.int32
                    ),
                    roughness,
                    emissivity,
                    np.float32(field["delt"][0, case]),
                    np.float32(field["c1sn"][0, case]),
                    np.float32(field["c2sn"][0, case]),
                    np.int32(isice),
                    np.int32(urban),
                    np.int32(2),
                    *(profiles[name] for name in RUC_SNOW_PREP_PROFILE_OUTPUTS),
                    *(outputs[name] for name in RUC_SNOW_PREP_COLUMN_OUTPUTS),
                    land,
                    np.int32(1),
                ),
            )
            cp.cuda.get_current_stream().synchronize()
            for name, value in profiles.items():
                collected.setdefault(name, []).append(cp.asnumpy(value).ravel())
            for name, value in outputs.items():
                collected.setdefault(name, []).append(cp.asnumpy(value))
            collected.setdefault("iland", []).append(
                cp.asnumpy(land).astype(np.float32)
            )
        results[flag] = {
            name: np.concatenate(parts) for name, parts in collected.items()
        }

    for name, contracted in results["-fmad=true"].items():
        np.testing.assert_array_equal(
            contracted.view(np.uint32),
            results["-fmad=false"][name].view(np.uint32),
            err_msg=f"FMA contraction moved {name}",
        )


@pytest.mark.gpu
@requires_gpu
def test_snow_preparation_cuda_rejects_bad_contracts():
    import cupy as cp

    from gpuwm.core.ruc import RUC_SNOW_PREP_COLUMN_INPUTS
    from gpuwm.core.ruc_gpu import ruc_snow_preparation_cuda

    field = _snow_prep_oracle()
    values, keywords = _snow_prep_case(
        cp, field, 2, RUC_SNOW_PREP_COLUMN_INPUTS
    )
    with pytest.raises(ValueError, match="positive"):
        ruc_snow_preparation_cuda(values, **{**keywords, "delt": 0.0})
    with pytest.raises(ValueError, match="isncovr_opt"):
        ruc_snow_preparation_cuda(values, **keywords, isncovr_opt=0)
    with pytest.raises(TypeError, match="isice"):
        ruc_snow_preparation_cuda(values, **{**keywords, "isice": 15.0})
    with pytest.raises(ValueError, match="isice is outside"):
        ruc_snow_preparation_cuda(values, **{**keywords, "isice": 99})
    with pytest.raises(TypeError, match="missing RUC CUDA snow preparation"):
        ruc_snow_preparation_cuda(
            {name: value for name, value in values.items() if name != "znt"},
            **keywords,
        )
    bad = dict(values)
    bad["ts1d"] = bad["ts1d"][:-1]
    with pytest.raises(ValueError, match=r"shape \(6 or 9"):
        ruc_snow_preparation_cuda(bad, **keywords)


@pytest.mark.gpu
@requires_gpu
def test_snow_preparation_cuda_batches_a_two_dimensional_tile():
    """A 2x2 tile on the device must match the CPU tile bit for bit."""

    import cupy as cp

    from gpuwm.core.ruc import (
        RUC_SNOW_PREP_COLUMN_INPUTS,
        RUC_SNOW_PREP_COLUMN_OUTPUTS,
        RUC_SNOW_PREP_PROFILE_OUTPUTS,
        ruc_snow_preparation,
    )
    from gpuwm.core.ruc_gpu import ruc_snow_preparation_cuda

    field = _snow_prep_oracle()
    selected = [3, 5, 7, 8]
    assert list(dict.fromkeys(field["delt"][0, selected])) == [60.0]
    host = {"ts1d": field["ts1d"][:, selected].reshape(9, 2, 2)}
    host.update({
        name: field[SNOW_PREP_BEFORE.get(name, name)][0, selected].reshape(2, 2)
        for name in RUC_SNOW_PREP_COLUMN_INPUTS
    })
    vegetation = field["ivgtyp"][0, selected].astype(np.int32).reshape(2, 2)
    land = field["iland_before"][0, selected].astype(np.int32).reshape(2, 2)
    expected = ruc_snow_preparation(
        host, delt=60.0, ivgtyp=vegetation, iland=land, isice=15
    )
    actual = ruc_snow_preparation_cuda(
        {name: cp.asarray(value) for name, value in host.items()},
        delt=60.0,
        ivgtyp=cp.asarray(vegetation),
        iland=cp.asarray(land),
        isice=15,
    )
    cp.cuda.get_current_stream().synchronize()
    assert actual.tice.shape == (9, 2, 2)
    assert actual.alb.shape == (2, 2)
    for name in (
        RUC_SNOW_PREP_PROFILE_OUTPUTS
        + RUC_SNOW_PREP_COLUMN_OUTPUTS
        + ("iland",)
    ):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)),
            np.asarray(getattr(expected, name)),
            err_msg=name,
        )


# ---------------------------------------------------------------------------
# WRF v4.6.1 module_sf_ruclsm.F:3789-4526 snowseaice, CUDA half.
# ---------------------------------------------------------------------------

_SNOWSEAICE_BEFORE = {
    "snhei": "snhei_before",
    "snwe": "snwe_before",
    "rhosn": "rhosn_before",
    "emiss": "emiss_before",
    "alb": "alb_before",
    "znt": "znt_before",
    "soilt": "soilt_before",
    "soilt1": "soilt1_before",
    "tsnav": "tsnav_before",
    "qvg": "qvg_before",
    "qsg": "qsg_before",
    "snom": "snom_before",
    "s": "s_before",
}


def _snowseaice_oracle(filename: str, ncase: int):
    path = ORACLE.with_name(filename)
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(ncase, 9).T
        for name in rows[0]
        if name != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    assert len(rows) == ncase * 9
    return rows, cases, fields


def _snowseaice_inputs(cp, fields, selection):
    from gpuwm.core.ruc import (
        RUC_SNOW_SEA_ICE_COLUMN_INPUTS,
        RUC_SNOW_SEA_ICE_PROFILE_INPUTS,
    )

    values = {
        name: cp.asarray(np.ascontiguousarray(fields[name][:, selection]))
        for name in RUC_SNOW_SEA_ICE_PROFILE_INPUTS
        if name != "tso"
    }
    values["tso"] = cp.asarray(
        np.ascontiguousarray(fields["tso_before"][:, selection])
    )
    values.update({
        name: cp.asarray(
            np.ascontiguousarray(
                fields[_SNOWSEAICE_BEFORE.get(name, name)][0, selection]
            )
        )
        for name in RUC_SNOW_SEA_ICE_COLUMN_INPUTS
    })
    values["ilnb"] = cp.asarray(
        np.ascontiguousarray(
            fields["ilnb_before"][0, selection].astype(np.int32)
        )
    )
    return values


def _assert_snowseaice_cuda_bitwise(cp, fields, selection, actual) -> None:
    from gpuwm.core.ruc import RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS

    np.testing.assert_array_equal(
        cp.asnumpy(actual.tso).view(np.uint32),
        fields["tso_after"][:, selection].view(np.uint32),
        err_msg="tso",
    )
    np.testing.assert_array_equal(
        cp.asnumpy(actual.ilnb),
        fields["ilnb_after"][0, selection].astype(np.int32),
        err_msg="ilnb",
    )
    for name in RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            expected[0, selection].view(np.uint32),
            err_msg=name,
        )


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize(
    ("filename", "ncase"),
    [("snowseaice.csv", 5), ("snowseaice_contract.csv", 10)],
)
def test_ruc_snow_sea_ice_cuda_matches_unmodified_wrf_bit_for_bit(
    filename, ncase,
):
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_snow_sea_ice_step_cuda

    _, cases, fields = _snowseaice_oracle(filename, ncase)
    assert len(cases) == ncase
    for case in range(ncase):
        actual = ruc_snow_sea_ice_step_cuda(
            _snowseaice_inputs(cp, fields, [case]),
            delt=float(fields["delt"][0, case]),
            conflx=float(fields["conflx"][0, case]),
            myj=bool(fields["myj"][0, case] > 0.5),
            cw=float(fields["cw"][0, case]),
            xlv=float(fields["xlv"][0, case]),
        )
        cp.cuda.get_current_stream().synchronize()
        _assert_snowseaice_cuda_bitwise(cp, fields, [case], actual)


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_sea_ice_cuda_solves_independent_columns_in_one_launch():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_snow_sea_ice_step_cuda

    # Both oracle files, batched: fifteen columns whose scalar arguments
    # agree, spanning one-layer, two-layer, blended, melting, sublimated and
    # bare-ice regimes.  The per-case launches above cannot catch a
    # column-stride defect.
    for filename, ncase, expected in (
        ("snowseaice.csv", 5, 5),
        ("snowseaice_contract.csv", 10, 7),
    ):
        _, cases, fields = _snowseaice_oracle(filename, ncase)
        selection = [
            case for case in range(ncase)
            if fields["myj"][0, case] < np.float32(0.5)
            and fields["xlv"][0, case] == np.float32(2.5e6)
            and fields["delt"][0, case] == np.float32(60.0)
            and fields["conflx"][0, case] == np.float32(40.0)
        ]
        assert len(selection) == expected
        actual = ruc_snow_sea_ice_step_cuda(
            _snowseaice_inputs(cp, fields, selection),
            delt=60.0, conflx=40.0, myj=False, cw=4.183e6, xlv=2.5e6,
        )
        cp.cuda.get_current_stream().synchronize()
        _assert_snowseaice_cuda_bitwise(cp, fields, selection, actual)


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_sea_ice_cuda_agrees_with_the_cpu_transcription():
    import cupy as cp

    from gpuwm.core.ruc import (
        RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS,
        ruc_snow_sea_ice_step,
    )
    from gpuwm.core.ruc_gpu import ruc_snow_sea_ice_step_cuda

    for filename, ncase in (("snowseaice.csv", 5), ("snowseaice_contract.csv", 10)):
        _, cases, fields = _snowseaice_oracle(filename, ncase)
        for case in range(ncase):
            device_values = _snowseaice_inputs(cp, fields, [case])
            host_values = {
                name: cp.asnumpy(value)
                for name, value in device_values.items()
            }
            kwargs = dict(
                delt=float(fields["delt"][0, case]),
                conflx=float(fields["conflx"][0, case]),
                myj=bool(fields["myj"][0, case] > 0.5),
                cw=float(fields["cw"][0, case]),
                xlv=float(fields["xlv"][0, case]),
            )
            device = ruc_snow_sea_ice_step_cuda(device_values, **kwargs)
            cp.cuda.get_current_stream().synchronize()
            host = ruc_snow_sea_ice_step(host_values, **kwargs)
            np.testing.assert_array_equal(
                cp.asnumpy(device.tso).view(np.uint32),
                host.tso.view(np.uint32),
                err_msg=f"{cases[case]}/tso",
            )
            np.testing.assert_array_equal(
                cp.asnumpy(device.ilnb), host.ilnb,
                err_msg=f"{cases[case]}/ilnb",
            )
            for name in RUC_SNOW_SEA_ICE_COLUMN_OUTPUTS:
                np.testing.assert_array_equal(
                    cp.asnumpy(getattr(device, name)).view(np.uint32),
                    getattr(host, name).view(np.uint32),
                    err_msg=f"{cases[case]}/{name}",
                )


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_sea_ice_cuda_preserves_inputs_and_fails_closed():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_snow_sea_ice_step_cuda

    _, cases, fields = _snowseaice_oracle("snowseaice.csv", 5)
    assert cases[3] == "melting_on_ice"
    values = _snowseaice_inputs(cp, fields, [3])
    originals = {
        name: cp.asnumpy(value).copy() for name, value in values.items()
    }
    actual = ruc_snow_sea_ice_step_cuda(values, delt=60.0)
    cp.cuda.get_current_stream().synchronize()
    for name, expected in originals.items():
        np.testing.assert_array_equal(cp.asnumpy(values[name]), expected)

    # icemelt absorbs the whole surface residual, so fltot cancels exactly.
    np.testing.assert_array_equal(cp.asnumpy(actual.fltot), np.float32(0.0))
    np.testing.assert_array_equal(cp.asnumpy(actual.qcg), np.float32(0.0))
    np.testing.assert_array_equal(cp.asnumpy(actual.soilt), np.float32(273.15))
    assert bool(cp.all(actual.tso <= cp.float32(271.4)))

    bad = dict(values)
    bad["thdifice"] = bad["thdifice"][:-1]
    with pytest.raises(ValueError, match="expected"):
        ruc_snow_sea_ice_step_cuda(bad, delt=60.0)
    bad = dict(values)
    bad["capice"] = bad["capice"][:-1]
    with pytest.raises(ValueError, match=r"shape \(6 or 9"):
        ruc_snow_sea_ice_step_cuda(bad, delt=60.0)
    bad = dict(values)
    bad["rhosn"] = cp.zeros_like(bad["rhosn"])
    with pytest.raises(ValueError, match="rhosn must be positive"):
        ruc_snow_sea_ice_step_cuda(bad, delt=60.0)
    with pytest.raises(ValueError, match="delt must be finite and positive"):
        ruc_snow_sea_ice_step_cuda(values, delt=0.0)
    with pytest.raises(ValueError, match="xlv must be finite and positive"):
        ruc_snow_sea_ice_step_cuda(values, delt=60.0, xlv=0.0)
    with pytest.raises(TypeError, match="myj"):
        ruc_snow_sea_ice_step_cuda(values, delt=60.0, myj=1)
    incomplete = {
        name: value for name, value in values.items() if name != "rnet"
    }
    with pytest.raises(
        TypeError, match="missing RUC CUDA snowseaice inputs: rnet"
    ):
        ruc_snow_sea_ice_step_cuda(incomplete, delt=60.0)


_SNOWTEMP_FIXTURES = {
    "snowtemp.csv": (
        "shallow_fresh_1layer",
        "deep_aged_2layer",
        "melting_2layer",
        "thin_blended",
        "warm_ground_bottom_melt",
        "sublimating_all_snow",
    ),
    "snowtemp_contract.csv": (
        "melt_dense_dry_evap",
        "melt_trace_all_evap",
        "melt_blended_trace",
        "bottom_melt_2layer",
        "bottom_melt_trace",
        "lowdens_new_on_aged",
        "melt_dense_moist",
        "melt_hot_surface_1layer",
    ),
}
_SNOWTEMP_BEFORE = {
    "snwe": "snwe_before",
    "snhei": "snhei_before",
    "beta": "beta_before",
    "rhosn": "rhosn_before",
    "soilt": "soilt_before",
    "soilt1": "soilt1_before",
    "qvg": "qvg_before",
    "dew": "dew_before",
}
# WRF's local x is returned as `storage`; every other output keeps its name.
_SNOWTEMP_AFTER = {
    "soilt": "soilt_after",
    "soilt1": "soilt1_after",
    "tsnav": "tsnav_after",
    "qvg": "qvg_after",
    "qsg": "qsg_after",
    "qcg": "qcg_after",
    "dew": "dew_after",
    "snwe": "snwe_after",
    "snhei": "snhei_after",
    "rhosn": "rhosn_after",
    "beta": "beta_after",
    "smelt": "smelt",
    "snoh": "snoh",
    "snflx": "snflx",
    "s": "s",
    "rsm": "rsm",
    "snweprint": "snweprint",
    "snheiprint": "snheiprint",
    "storage": "x",
}


def _snowtemp_oracle(name="snowtemp.csv"):
    path = ORACLE.with_name(name)
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float32)
        .reshape(len(_SNOWTEMP_FIXTURES[name]), 9).T
        for key in rows[0]
        if key != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    return rows, cases, fields


def _snowtemp_inputs(cp, fields, selection):
    from gpuwm.core.ruc import RUC_SNOW_TEMPERATURE_COLUMN_INPUTS

    values = {
        name: cp.asarray(np.ascontiguousarray(fields[name][:, selection]))
        for name in ("cap", "thdif", "tranf")
    }
    values["tso"] = cp.asarray(
        np.ascontiguousarray(fields["tso_before"][:, selection])
    )
    values.update({
        name: cp.asarray(
            np.ascontiguousarray(
                fields[_SNOWTEMP_BEFORE.get(name, name)][0, selection]
            )
        )
        for name in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    })
    return values


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("fixture", sorted(_SNOWTEMP_FIXTURES))
def test_ruc_snow_temperature_cuda_matches_unmodified_wrf_bit_for_bit(fixture):
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_snow_temperature_step_cuda

    rows, cases, fields = _snowtemp_oracle(fixture)
    assert cases == _SNOWTEMP_FIXTURES[fixture]
    assert len(rows) == len(cases) * 9
    for case in range(len(cases)):
        actual = ruc_snow_temperature_step_cuda(
            _snowtemp_inputs(cp, fields, [case]),
            delt=float(fields["delt"][0, case]),
            conflx=float(fields["conflx"][0, case]),
            nroot=int(fields["nroot"][0, case]),
            ilnb=int(fields["ilnb_before"][0, case]),
            xlvm=float(fields["xlvm"][0, case]),
            cvw=float(fields["cvw"][0, case]),
        )
        cp.cuda.get_current_stream().synchronize()
        np.testing.assert_array_equal(
            cp.asnumpy(actual.tso)[:, 0].view(np.uint32),
            fields["tso_after"][:, case].view(np.uint32),
            err_msg=f"{cases[case]}/tso",
        )
        for name, column in _SNOWTEMP_AFTER.items():
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(actual, name)).view(np.uint32),
                fields[column][0, case : case + 1].view(np.uint32),
                err_msg=f"{cases[case]}/{name}",
            )
        assert int(cp.asnumpy(actual.ilnb)[0]) == int(
            fields["ilnb_after"][0, case]
        )


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("fixture", sorted(_SNOWTEMP_FIXTURES))
def test_ruc_snow_temperature_cuda_solves_independent_columns_in_one_launch(
    fixture,
):
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_snow_temperature_step_cuda

    _, cases, fields = _snowtemp_oracle(fixture)
    # Every regime shares this lane's scalar arguments, so one launch can
    # carry them all; the per-case launches above cannot catch a
    # column-stride defect.
    selection = list(range(len(cases)))
    assert np.all(fields["delt"][0, selection] == np.float32(60.0))
    assert np.all(fields["conflx"][0, selection] == np.float32(40.0))
    actual = ruc_snow_temperature_step_cuda(
        _snowtemp_inputs(cp, fields, selection),
        delt=60.0,
        conflx=40.0,
        nroot=cp.asarray(fields["nroot"][0, selection].astype(np.int32)),
        ilnb=cp.asarray(fields["ilnb_before"][0, selection].astype(np.int32)),
    )
    cp.cuda.get_current_stream().synchronize()
    np.testing.assert_array_equal(
        cp.asnumpy(actual.tso).view(np.uint32),
        fields["tso_after"][:, selection].view(np.uint32),
    )
    for name, column in _SNOWTEMP_AFTER.items():
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            fields[column][0, selection].view(np.uint32),
            err_msg=name,
        )
    np.testing.assert_array_equal(
        cp.asnumpy(actual.ilnb),
        fields["ilnb_after"][0, selection].astype(np.int32),
    )


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("fixture", sorted(_SNOWTEMP_FIXTURES))
def test_ruc_snow_temperature_cuda_agrees_with_the_cpu_transcription(fixture):
    import cupy as cp

    from gpuwm.core.ruc import ruc_snow_temperature_step
    from gpuwm.core.ruc_gpu import ruc_snow_temperature_step_cuda

    _, cases, fields = _snowtemp_oracle(fixture)
    selection = list(range(len(cases)))
    device = ruc_snow_temperature_step_cuda(
        _snowtemp_inputs(cp, fields, selection),
        delt=60.0,
        conflx=40.0,
        nroot=cp.asarray(fields["nroot"][0, selection].astype(np.int32)),
        ilnb=cp.asarray(fields["ilnb_before"][0, selection].astype(np.int32)),
    )
    cp.cuda.get_current_stream().synchronize()
    host_inputs = {
        name: cp.asnumpy(value)
        for name, value in _snowtemp_inputs(cp, fields, selection).items()
    }
    host = ruc_snow_temperature_step(
        host_inputs,
        delt=60.0,
        conflx=40.0,
        nroot=fields["nroot"][0, selection].astype(np.int32),
        ilnb=fields["ilnb_before"][0, selection].astype(np.int32),
    )
    for name in ("tso", *_SNOWTEMP_AFTER, "ilnb"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(device, name)),
            getattr(host, name),
            err_msg=name,
        )


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_temperature_cuda_preserves_inputs_and_fails_closed():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_snow_temperature_step_cuda

    _, cases, fields = _snowtemp_oracle()
    assert cases[2] == "melting_2layer"
    values = _snowtemp_inputs(cp, fields, [2])
    originals = {
        name: cp.asnumpy(value).copy() for name, value in values.items()
    }
    actual = ruc_snow_temperature_step_cuda(values, delt=60.0, nroot=6, ilnb=1)
    cp.cuda.get_current_stream().synchronize()
    for name, expected in originals.items():
        np.testing.assert_array_equal(cp.asnumpy(values[name]), expected)

    # The top-melt regime pins the skin at the freezing point and holds a
    # Koren liquid fraction in the pack, which densifies it.
    np.testing.assert_array_equal(cp.asnumpy(actual.soilt), np.float32(273.15))
    assert float(cp.asnumpy(actual.rsm)[0]) > 0.0
    assert float(cp.asnumpy(actual.rhosn)[0]) > float(
        cp.asnumpy(values["rhosn"])[0]
    )
    assert int(cp.asnumpy(actual.ilnb)[0]) == 2

    bad = dict(values)
    bad["thdif"] = bad["thdif"][:-1]
    with pytest.raises(ValueError, match="expected"):
        ruc_snow_temperature_step_cuda(bad, delt=60.0)
    bad = dict(values)
    bad["cap"] = bad["cap"][:-1]
    with pytest.raises(ValueError, match=r"shape \(6 or 9"):
        ruc_snow_temperature_step_cuda(bad, delt=60.0)
    with pytest.raises(ValueError, match="delt must be finite and positive"):
        ruc_snow_temperature_step_cuda(values, delt=0.0)
    with pytest.raises(ValueError, match="xlvm must be finite and positive"):
        ruc_snow_temperature_step_cuda(values, delt=60.0, xlvm=0.0)
    with pytest.raises(TypeError, match="ilnb must contain integer"):
        ruc_snow_temperature_step_cuda(values, delt=60.0, ilnb=1.0)
    # snhei == 0 is unreachable through LSMRUC, which gates the snow branch
    # at :1600, so the oracle builder cannot emit a column for it and the
    # launcher refuses the state rather than run unpinned arithmetic.
    zeroed = dict(values)
    zeroed["snhei"] = cp.zeros_like(values["snhei"])
    with pytest.raises(ValueError, match="snhei must be positive"):
        ruc_snow_temperature_step_cuda(zeroed, delt=60.0)
    incomplete = {
        name: value for name, value in values.items() if name != "rnet"
    }
    with pytest.raises(
        TypeError, match="missing RUC CUDA snowtemp inputs: rnet"
    ):
        ruc_snow_temperature_step_cuda(incomplete, delt=60.0)


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_temperature_cuda_has_no_unpinned_contraction():
    """Compiling with and without FMA contraction must agree bit for bit.

    NVRTC fuses ``a*b+c`` where gfortran does not, and ``snowtemp`` closes a
    two-pass melt iteration over a tridiagonal recurrence, so one fused
    rounding feeds the next.  Every boundary in the kernel is pinned with
    ``__fadd_rn``/``__fsub_rn``/``__fmul_rn``/``__fdiv_rn``; if any were
    missed, ``--fmad=false`` would move the result.  Compiled here rather
    than through ``gpuwm.core.kernels`` so the shared compile options for
    every other kernel are untouched.
    """

    import cupy as cp

    from gpuwm.core.constants import CUDA_DEFINES
    from gpuwm.core.ruc import RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    from gpuwm.core.ruc_gpu import _device_tbq

    kernels = Path(__file__).parents[1] / "gpuwm" / "core" / "kernels"
    source = (
        "\n".join(f"#define {k} {float(v)!r}f" for k, v in CUDA_DEFINES.items())
        + "\n"
        + (kernels / "common.cuh").read_text()
        + "\n"
        + (kernels / "ruc.cu").read_text()
    )
    scalars = (
        "soilt", "soilt1", "tsnav", "qvg", "qsg", "qcg", "dew", "snwe",
        "snhei", "rhosn", "beta", "smelt", "snoh", "snflx", "s", "rsm",
        "snweprint", "snheiprint", "storage",
    )
    results = []
    for options in (("-std=c++17",), ("-std=c++17", "--fmad=false")):
        module = cp.RawModule(code=source, options=options)
        module.compile()
        kernel = module.get_function("ruc_snow_temperature_step")
        collected = []
        for fixture in sorted(_SNOWTEMP_FIXTURES):
            _, cases, fields = _snowtemp_oracle(fixture)
            selection = list(range(len(cases)))
            values = _snowtemp_inputs(cp, fields, selection)
            ncolumn = len(selection)
            tso = cp.empty((9, ncolumn), dtype=cp.float32)
            outs = {name: cp.empty(ncolumn, dtype=cp.float32) for name in scalars}
            layers = cp.empty(ncolumn, dtype=cp.int32)
            kernel(
                (1,),
                (128,),
                (
                    values["cap"], values["thdif"], values["tranf"],
                    values["tso"],
                    *(values[n] for n in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS),
                    cp.asarray(fields["nroot"][0, selection].astype(np.int32)),
                    cp.asarray(
                        fields["ilnb_before"][0, selection].astype(np.int32)
                    ),
                    _device_tbq(int(cp.cuda.runtime.getDevice())),
                    np.float32(60.0),
                    # ``conflx`` is a POINTER, not a ``real``: it is
                    # ``0.5*dz8w(i,1,j)``, a per-column depth, and the column
                    # loop was batched so every leaf now carries the column
                    # axis.  This test builds the argument list by hand, so a
                    # scalar here is a garbage pointer and faults the context
                    # for every later test in the file.
                    cp.full(ncolumn, 40.0, dtype=cp.float32),
                    np.float32(2.835e6), np.float32(4.183e6),
                    tso, *(outs[name] for name in scalars), layers,
                    np.int32(ncolumn),
                ),
            )
            cp.cuda.get_current_stream().synchronize()
            collected.append(cp.asnumpy(tso).ravel())
            collected.extend(cp.asnumpy(outs[name]) for name in scalars)
            collected.append(cp.asnumpy(layers).astype(np.float32))
        results.append(np.concatenate(collected))
    assert results[0].size == 406
    np.testing.assert_array_equal(
        results[0].view(np.uint32), results[1].view(np.uint32)
    )


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_temperature_cuda_agrees_off_fixture():
    """CPU and CUDA must agree away from the fixtures, not just on them.

    Fixture parity does not imply agreement elsewhere: a sibling RUC snow
    lane found 3 ULP of ``thdif`` amplifying to 53,248 ULP of an energy
    residual while both halves were ``max_ulp 0`` against their fixtures.
    ``snowtemp`` calls no transcendental and this kernel pins every
    arithmetic boundary, so the two halves should be bit-identical on any
    state either accepts -- including a few-ULP perturbation, which is where
    an amplifying recurrence shows up, and signed zeros, where Python's
    ``max`` and CUDA's ``fmaxf`` could legally disagree.
    """

    import cupy as cp

    from gpuwm.core.ruc import (
        RUC_SNOW_TEMPERATURE_COLUMN_INPUTS,
        RUC_SNOW_TEMPERATURE_PROFILE_INPUTS,
        ruc_snow_temperature_step,
    )
    from gpuwm.core.ruc_gpu import ruc_snow_temperature_step_cuda

    rng = np.random.default_rng(20260724)
    positive = ("patm", "rho", "rhosn", "snhei", "snth", "deltsn")
    columns = tuple(_SNOWTEMP_AFTER)

    def host_inputs(fields, case):
        values = {
            name: np.ascontiguousarray(fields[name][:, case : case + 1]).copy()
            for name in ("cap", "thdif", "tranf")
        }
        values["tso"] = np.ascontiguousarray(
            fields["tso_before"][:, case : case + 1]
        ).copy()
        values.update({
            name: np.ascontiguousarray(
                fields[_SNOWTEMP_BEFORE.get(name, name)][0, case : case + 1]
            ).copy()
            for name in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
        })
        return values

    samples = []
    for fixture in sorted(_SNOWTEMP_FIXTURES):
        _, cases, fields = _snowtemp_oracle(fixture)
        for case in range(len(cases)):
            # multiplicative departures from the fixture state
            for _ in range(6):
                values = host_inputs(fields, case)
                for key, array in values.items():
                    factor = np.exp(
                        rng.normal(0.0, 0.12, size=array.shape)
                    ).astype(np.float32)
                    shifted = (array * factor).astype(np.float32)
                    if key in positive:
                        shifted = np.abs(shifted)
                    values[key] = np.ascontiguousarray(shifted)
                samples.append(values)
            # few-ULP departures, where an amplifying recurrence would show
            for key in ("thdif", "cap", "tso", "soilt"):
                for shift in (1, 3, 8):
                    values = host_inputs(fields, case)
                    bits = values[key].view(np.int32) + np.int32(shift)
                    values[key] = np.ascontiguousarray(
                        bits.view(np.float32).copy()
                    )
                    samples.append(values)
            # signed zeros into the min/max sites
            for key in ("dew", "newsnow", "prcpms", "vegfrac", "snwe"):
                for sign in (np.float32(0.0), np.float32(-0.0)):
                    values = host_inputs(fields, case)
                    values[key] = np.full_like(values[key], sign)
                    samples.append(values)

    kwargs = dict(delt=60.0, conflx=40.0, nroot=6, ilnb=1)
    keep = []
    rejected = 0
    for values in samples:
        try:
            host = ruc_snow_temperature_step(
                {k: v.copy() for k, v in values.items()}, **kwargs
            )
        except ValueError:
            # Both halves must fail closed on the same state.
            rejected += 1
            with pytest.raises(ValueError):
                ruc_snow_temperature_step_cuda(
                    {k: cp.asarray(v) for k, v in values.items()}, **kwargs
                )
            continue
        keep.append((values, host))
    assert len(keep) > 300, (len(keep), rejected)

    batch = {
        name: cp.asarray(
            np.concatenate([v[name] for v, _ in keep], axis=1)
        )
        for name in RUC_SNOW_TEMPERATURE_PROFILE_INPUTS
    }
    batch.update({
        name: cp.asarray(np.concatenate([v[name] for v, _ in keep]))
        for name in RUC_SNOW_TEMPERATURE_COLUMN_INPUTS
    })
    device = ruc_snow_temperature_step_cuda(batch, **kwargs)
    cp.cuda.get_current_stream().synchronize()

    tso = cp.asnumpy(device.tso)
    arrays = {name: cp.asnumpy(getattr(device, name)) for name in columns}
    ilnb = cp.asnumpy(device.ilnb)
    for slot, (_, host) in enumerate(keep):
        np.testing.assert_array_equal(
            tso[:, slot].view(np.uint32),
            host.tso[:, 0].view(np.uint32),
            err_msg=f"off-fixture sample {slot}/tso",
        )
        for name in columns:
            np.testing.assert_array_equal(
                arrays[name][slot : slot + 1].view(np.uint32),
                getattr(host, name).view(np.uint32),
                err_msg=f"off-fixture sample {slot}/{name}",
            )
        assert int(ilnb[slot]) == int(host.ilnb[0])


# ---------------------------------------------------------------------------
# WRF v4.6.1 phys/module_sf_ruclsm.F:3120-3786 snowsoil on the GPU.  Appended
# as one contiguous block for mergeability with the parallel snowtemp and
# snowseaice ports.
# ---------------------------------------------------------------------------

_SNOWSOIL_CASES = (
    "fresh_snow_cold_forest",
    "deep_aged_snow_grass",
    "melting_snow_rain",
    "thin_snow_frozen_crop",
)
_SNOWSOIL_PROFILE_OUTPUTS = (
    "tso", "soilmois", "smfrkeep", "keepfr", "soilice", "soiliqw",
)
_SNOWSOIL_COLUMN_OUTPUTS = (
    "cst", "dew", "soilt", "soilt1", "tsnav", "qvg", "qsg", "qcg",
    "snwe", "snhei", "rhosn", "snweprint", "snheiprint", "rsm", "smelt",
    "snoh", "snflx", "snom", "edir1", "ec1", "ett1", "eeta", "qfx",
    "hfx", "s", "sublim", "prcpl", "fltot", "runoff1", "runoff2",
    "mavail", "infiltrp",
)
_SNOWSOIL_ENTRY_NAMES = {
    "snhei": "snhei_before", "snwe": "snwe_before",
    "rhosn": "rhosn_before", "soilt": "soilt_before",
    "soilt1": "soilt1_before", "qvg": "qvg_before", "qsg": "qsg_before",
    "qcg": "qcg_before", "cst": "cst_before", "snom": "snom_before",
}


def _snowsoil_oracle():
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "snowsoil.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(len(_SNOWSOIL_CASES), 9).T
        for name in rows[0]
        if name != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    return rows, cases, fields


def _snowsoil_inputs(cp, fields, selection):
    from gpuwm.core.ruc import RUC_SNOW_SOIL_COLUMN_INPUTS

    values = {
        name: cp.asarray(np.ascontiguousarray(fields[key][:, selection]))
        for name, key in (
            ("soilmois", "soilmois_before"), ("tso", "tso_before"),
            ("smfrkeep", "smfrkeep_before"), ("keepfr", "keepfr_before"),
        )
    }
    values.update({
        name: cp.asarray(np.ascontiguousarray(
            fields[_SNOWSOIL_ENTRY_NAMES.get(name, name)][0, selection]
        ))
        for name in RUC_SNOW_SOIL_COLUMN_INPUTS
    })
    return values


def _snowsoil_call_cuda(cp, fields, selection):
    from gpuwm.core.ruc_gpu import ruc_snow_soil_step_cuda

    return ruc_snow_soil_step_cuda(
        _snowsoil_inputs(cp, fields, selection),
        cp.asarray(fields["iland"][0, selection].astype(np.int32)),
        nroot=cp.asarray(fields["nroot"][0, selection].astype(np.int32)),
        delt=float(fields["delt"][0, selection[0]]),
        conflx=float(fields["conflx"][0, selection[0]]),
        ilnb=cp.asarray(fields["ilnb_before"][0, selection].astype(np.int32)),
        cw=float(fields["cw"][0, selection[0]]),
    )


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_soil_step_cuda_matches_unmodified_wrf_bit_for_bit():
    import cupy as cp

    rows, cases, fields = _snowsoil_oracle()
    assert len(rows) == 36
    assert cases == _SNOWSOIL_CASES
    for case in range(len(cases)):
        actual = _snowsoil_call_cuda(cp, fields, [case])
        cp.cuda.get_current_stream().synchronize()
        for name in _SNOWSOIL_PROFILE_OUTPUTS:
            expected = fields.get(f"{name}_after", fields.get(name))
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(actual, name))[:, 0].view(np.uint32),
                expected[:, case].view(np.uint32),
                err_msg=f"{cases[case]}/{name}",
            )
        for name in _SNOWSOIL_COLUMN_OUTPUTS:
            expected = fields.get(f"{name}_after", fields.get(name))
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(actual, name)).view(np.uint32),
                expected[0, case : case + 1].view(np.uint32),
                err_msg=f"{cases[case]}/{name}",
            )
        assert int(cp.asnumpy(actual.ilnb)[0]) == int(
            fields["ilnb_after"][0, case]
        )


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_soil_step_cuda_solves_independent_columns_in_one_launch():
    import cupy as cp

    _, cases, fields = _snowsoil_oracle()
    # The per-case launches above cannot catch a column-stride defect; every
    # regime shares this lane's scalar arguments, so they batch together.
    selection = list(range(len(cases)))
    actual = _snowsoil_call_cuda(cp, fields, selection)
    cp.cuda.get_current_stream().synchronize()
    for name in _SNOWSOIL_PROFILE_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            expected[:, selection].view(np.uint32),
            err_msg=name,
        )
    for name in _SNOWSOIL_COLUMN_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            expected[0, selection].view(np.uint32),
            err_msg=name,
        )
    np.testing.assert_array_equal(
        cp.asnumpy(actual.ilnb), fields["ilnb_after"][0].astype(np.int32)
    )


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_soil_step_cuda_preserves_inputs_and_fails_closed():
    import cupy as cp

    from gpuwm.core.ruc_gpu import ruc_snow_soil_step_cuda

    _, cases, fields = _snowsoil_oracle()
    assert cases[2] == "melting_snow_rain"
    values = _snowsoil_inputs(cp, fields, [2])
    originals = {
        name: cp.asnumpy(value).copy() for name, value in values.items()
    }
    land = cp.asarray(np.asarray([10], dtype=np.int32))
    actual = _snowsoil_call_cuda(cp, fields, [2])
    cp.cuda.get_current_stream().synchronize()
    for name, expected in originals.items():
        np.testing.assert_array_equal(cp.asnumpy(values[name]), expected)

    # soilmoist never writes soiliqw back, so the returned partition still
    # describes the moisture state on entry.
    riw = np.float32(np.float32(900.0) * np.float32(1.0e-3))
    np.testing.assert_allclose(
        cp.asnumpy(actual.soiliqw) + cp.asnumpy(actual.soilice) * riw,
        fields["soilmois_before"][:, 2 : 3],
        rtol=1e-5, atol=1e-6,
    )
    np.testing.assert_array_equal(
        cp.asnumpy(actual.s), cp.asnumpy(actual.snflx)
    )

    with pytest.raises(ValueError, match="positive"):
        ruc_snow_soil_step_cuda(
            values, land, nroot=6, delt=0.0, conflx=40.0
        )
    with pytest.raises(ValueError, match="myj"):
        ruc_snow_soil_step_cuda(
            values, land, nroot=6, delt=60.0, conflx=40.0, myj=True
        )
    with pytest.raises(TypeError, match="ilnb"):
        ruc_snow_soil_step_cuda(
            values, land, nroot=6, delt=60.0, conflx=40.0, ilnb=1.0
        )
    bad = dict(values)
    bad["tso"] = bad["tso"][:-1]
    with pytest.raises(ValueError, match="expected"):
        ruc_snow_soil_step_cuda(
            bad, land, nroot=6, delt=60.0, conflx=40.0
        )
    incomplete = {
        name: value for name, value in values.items() if name != "rnet"
    }
    with pytest.raises(
        TypeError, match="missing RUC CUDA snowsoil inputs: rnet"
    ):
        ruc_snow_soil_step_cuda(
            incomplete, land, nroot=6, delt=60.0, conflx=40.0
        )


# ---------------------------------------------------------------------------
# snowsoil argument-contract fixture on the GPU.  See tests/test_ruc.py for why
# these three regimes exist: they kill the five argument mutants that
# snowsoil.csv alone cannot detect.
# ---------------------------------------------------------------------------

_SNOWSOIL_CONTRACT_CASES = (
    "partial_cover_two_layer",
    "sublimating_patch",
    "blended_retained_ilnb",
)


def _snowsoil_contract_oracle():
    path = (
        Path(__file__).parents[1]
        / "gpuwm" / "data" / "ruc" / "oracle" / "snowsoil_contract.csv"
    )
    with path.open(newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    fields = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float32)
        .reshape(len(_SNOWSOIL_CONTRACT_CASES), 9).T
        for name in rows[0]
        if name != "case"
    }
    cases = tuple(dict.fromkeys(row["case"] for row in rows))
    return rows, cases, fields


@pytest.mark.gpu
@requires_gpu
def test_ruc_snow_soil_step_cuda_contract_fixture_matches_wrf_bit_for_bit():
    import cupy as cp

    rows, cases, fields = _snowsoil_contract_oracle()
    assert len(rows) == 27
    assert cases == _SNOWSOIL_CONTRACT_CASES
    selection = list(range(len(cases)))
    actual = _snowsoil_call_cuda(cp, fields, selection)
    cp.cuda.get_current_stream().synchronize()
    for name in _SNOWSOIL_PROFILE_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            expected[:, selection].view(np.uint32),
            err_msg=name,
        )
    for name in _SNOWSOIL_COLUMN_OUTPUTS:
        expected = fields.get(f"{name}_after", fields.get(name))
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(actual, name)).view(np.uint32),
            expected[0, selection].view(np.uint32),
            err_msg=name,
        )
    np.testing.assert_array_equal(
        cp.asnumpy(actual.ilnb), fields["ilnb_after"][0].astype(np.int32)
    )
    for case in range(len(cases)):
        single = _snowsoil_call_cuda(cp, fields, [case])
        cp.cuda.get_current_stream().synchronize()
        for name in _SNOWSOIL_COLUMN_OUTPUTS:
            expected = fields.get(f"{name}_after", fields.get(name))
            np.testing.assert_array_equal(
                cp.asnumpy(getattr(single, name)).view(np.uint32),
                expected[0, case : case + 1].view(np.uint32),
                err_msg=f"{cases[case]}/{name}",
            )

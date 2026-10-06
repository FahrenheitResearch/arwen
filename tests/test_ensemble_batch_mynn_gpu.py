"""All-member MYNN native column leaves versus the same members alone."""

import dataclasses

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.physics_inventory import MYNN_SURFACE_OUTPUTS
from gpuwm.ensemble.batch_mynn import (
    prepare_mynn_pbl_column_batch, prepare_mynn_predict_column_batch,
    prepare_mynn_surface_column_batch)

pytestmark = [pytest.mark.gpu, requires_gpu]


def _surface_hosts(members):
    from test_mynn_surface_gpu import _oracle_fields, INPUT_ALIASES, INPUT_NAMES
    _, fields = _oracle_fields()
    hosts = [{name: fields[INPUT_ALIASES.get(name, name)].copy() for name in INPUT_NAMES}
             for _ in range(members)]
    for member, host in enumerate(hosts):
        host["u1"] += np.float32(member * 0.125)
        host["tsk"] += np.float32(member * 0.03125)
    return hosts, fields


@pytest.mark.parametrize("members", [2, 4, 8])
@pytest.mark.parametrize("variant", ["wrf_461", "gsl_wrf39"])
@pytest.mark.parametrize("aliased", [False, True])
def test_surface_one_launch_matches_all_outputs_and_carried_inputs(members, variant, aliased, monkeypatch):
    import cupy as cp
    from gpuwm.core.mynn_sfclay import (MynnSurfaceResult, launch_mynn_surface_layer,
                                      seed_mynn_surface_first_step)
    from gpuwm.core import mynn_sfclay
    native_kernel = mynn_sfclay.mynn_surface_kernel
    observed = []
    def observe_kernel(identity):
        kernel = native_kernel(identity)
        def launch(*args):
            observed.append(identity)
            return kernel(*args)
        return launch
    monkeypatch.setattr(mynn_sfclay, "mynn_surface_kernel", observe_kernel)
    hosts, fields = _surface_hosts(members)
    ny, nx = hosts[0]["u1"].shape
    inputs = {name: cp.asarray(np.concatenate([host[name] for host in hosts])) for name in hosts[0]}
    mol = cp.zeros_like(inputs["u1"])
    ustm = cp.zeros_like(inputs["u1"])
    outputs = {name: cp.zeros_like(mol) for name in MYNN_SURFACE_OUTPUTS}
    if aliased:
        outputs.update({name: inputs[name] for name in set(inputs) & set(outputs)})
        outputs.update(mol=mol, ustm=ustm)
    bound = prepare_mynn_surface_column_batch(inputs, mol=mol, ustm=ustm, outputs=outputs,
        members=members, ny=ny, nx=nx, dx=2250, available_bytes=1 << 24, variant=variant)
    ordinary_inputs = [{name: cp.asarray(host[name]) for name in host} for host in hosts]
    ordinary_mol = [cp.zeros((ny, nx), dtype=cp.float32) for _ in hosts]
    ordinary_ustm = [cp.zeros((ny, nx), dtype=cp.float32) for _ in hosts]
    ordinary_outputs = [{name: cp.zeros_like(ordinary_mol[member]) for name in MYNN_SURFACE_OUTPUTS}
                        for member in range(members)]
    if aliased:
        for member, result in enumerate(ordinary_outputs):
            result.update({name: ordinary_inputs[member][name] for name in set(inputs) & set(result)})
            result.update(mol=ordinary_mol[member], ustm=ordinary_ustm[member])
    for timestep in (1, 2, 3):
        if timestep == 1:
            seed_mynn_surface_first_step(inputs["u1"], inputs["v1"], inputs["qv1"],
                ust=inputs["ust"], mol=mol, qsfc=inputs["qsfc"], qstar=outputs["qstar"])
            for member in range(members):
                ordinary = ordinary_inputs[member]
                seed_mynn_surface_first_step(ordinary["u1"], ordinary["v1"], ordinary["qv1"],
                    ust=ordinary["ust"], mol=ordinary_mol[member], qsfc=ordinary["qsfc"],
                    qstar=ordinary_outputs[member]["qstar"])
        count_before = len(observed)
        bound(itimestep=timestep)
        assert len(observed) == count_before + 1
        for member in range(members):
            result = MynnSurfaceResult(**ordinary_outputs[member])
            launch_mynn_surface_layer(ordinary_inputs[member], ordinary_mol[member], ordinary_ustm[member],
                result, dx=2250, itimestep=timestep, variant=variant)
            stripe = slice(member * ny, (member + 1) * ny)
            for name in MYNN_SURFACE_OUTPUTS:
                assert outputs[name][stripe].get().tobytes() == getattr(result, name).get().tobytes(), (member, timestep, name)
            for name in inputs:
                assert inputs[name][stripe].get().tobytes() == ordinary_inputs[member][name].get().tobytes(), (member, timestep, name)
            assert mol[stripe].get().tobytes() == ordinary_mol[member].get().tobytes()
            assert ustm[stripe].get().tobytes() == ordinary_ustm[member].get().tobytes()
    assert bound.receipt["members"] == members and bound.receipt["launches_per_call"] == 1
    assert bound.receipt["variant"] == variant and len(bound.receipt["source_sha256"]) == 64


def _predict_hosts(members):
    from test_mynn_pbl import _predict_oracle
    from gpuwm.core.mynn_pbl import MYNN_PREDICT_INPUTS
    _, fields = _predict_oracle()
    values = {name: fields[name].copy() for name in
              ("dz", "rho", "dfq", "pdk", "pdt", "pdq", "pdc", "el", "s_aw", "s_awqke")}
    values.update({name: fields[name][:, 0].copy() for name in ("ust", "flt", "flq", "pmz", "phh", "delt")})
    values.update({name: fields[f"{name}_before"].copy() for name in ("qke", "tsq", "qsq", "cov")})
    assert set(values) == set(MYNN_PREDICT_INPUTS)
    hosts = [{name: value.copy() for name, value in values.items()} for _ in range(members)]
    for member, host in enumerate(hosts):
        host["delt"] += np.float32(member * 0.125)
    return hosts


@pytest.mark.parametrize("members", [2, 4, 8])
@pytest.mark.parametrize("version", ["wrf_461", "gsd_41"])
def test_tke_predictor_one_launch_matches_each_member_and_overwrites_workspace(members, version, monkeypatch):
    import cupy as cp
    from gpuwm.core.mynn_pbl_gpu import mynn_predict_default_cuda
    from gpuwm.core import mynn_pbl_gpu
    native_get = mynn_pbl_gpu.mynn_pbl_kernel
    observed = []
    def observe_kernel(entry, identity="wrf_461"):
        kernel = native_get(entry, identity)
        def launch(*args):
            observed.append(entry)
            return kernel(*args)
        return launch
    monkeypatch.setattr(mynn_pbl_gpu, "mynn_pbl_kernel", observe_kernel)
    hosts = _predict_hosts(members)
    values = {name: cp.asfortranarray(cp.asarray(np.concatenate([host[name] for host in hosts])))
              for name in hosts[0]}
    bound = prepare_mynn_predict_column_batch(values, members=members, ny=2, nx=2,
                                              available_bytes=1 << 24, bl_mynn_version=version)
    before = {name: array.get().tobytes() for name, array in values.items()}
    pointers = {name: int(array.data.ptr) for name, array in bound.storage.arrays.items()}
    for repeat in range(2):
        for array in bound.storage.arrays.values():
            array.fill(cp.float32(np.nan))
        count_before = len(observed)
        actual = bound()
        assert observed[count_before:] == ["mynn_predict_default_columns"]
        for member, host in enumerate(hosts):
            ordinary = mynn_predict_default_cuda(host, bl_mynn_version=version)
            for field in dataclasses.fields(actual):
                got = getattr(actual, field.name)[member * 4:(member + 1) * 4]
                expected = getattr(ordinary, field.name)
                assert got.get().tobytes() == expected.get().tobytes(), (member, field.name, repeat)
                assert bool(cp.all(cp.isfinite(got)))
        assert {name: array.get().tobytes() for name, array in values.items()} == before
        assert {name: int(array.data.ptr) for name, array in bound.storage.arrays.items()} == pointers
    assert bound.receipt["launches_per_call"] == 1
    assert bound.receipt["bl_mynn_version"] == version
    assert bound.storage.payload_bytes == 14 * members * 4 * 12 * 4


def test_surface_cross_field_alias_and_predictor_layout_refuse_before_mutation():
    import cupy as cp
    hosts, fields = _surface_hosts(2)
    inputs = {name: cp.asarray(np.concatenate([host[name] for host in hosts])) for name in hosts[0]}
    mol, ustm = cp.zeros_like(inputs["u1"]), cp.zeros_like(inputs["u1"])
    outputs = {name: cp.zeros_like(mol) for name in MYNN_SURFACE_OUTPUTS}
    outputs["ust"] = inputs["u1"]
    before = inputs["u1"].get().tobytes()
    with pytest.raises(ValueError, match="input u1 overlaps output ust"):
        prepare_mynn_surface_column_batch(inputs, mol=mol, ustm=ustm, outputs=outputs,
            members=2, ny=2, nx=3, dx=2250, available_bytes=1 << 24)
    assert inputs["u1"].get().tobytes() == before
    hosts = _predict_hosts(2)
    values = {name: cp.asfortranarray(cp.asarray(np.concatenate([host[name] for host in hosts])))
              for name in hosts[0]}
    values["dz"] = cp.ascontiguousarray(values["dz"])
    with pytest.raises(ValueError, match="F-contiguous"):
        prepare_mynn_predict_column_batch(values, members=2, ny=2, nx=2, available_bytes=1 << 24)


@pytest.mark.parametrize("members", [2, 4, 8])
@pytest.mark.parametrize("mixlength", [1, 2])
@pytest.mark.parametrize("version,unsquared,cloud_form", [
    ("wrf_461", False, "wrf_461"), ("gsd_41", False, "wrf_461"),
    ("gsd_41", False, "gsd_41"), ("gsd_41", True, "wrf_461"),
    ("gsd_41", True, "gsd_41")])
def test_full_pbl_wrapper_matches_each_ordinary_member_cold_and_carried_state(
        members, mixlength, version, unsquared, cloud_form, monkeypatch):
    import cupy as cp
    from test_mynn_pbl_runtime import _build
    from gpuwm.core.physics import _prepare_atmosphere
    from gpuwm.core.mynn_pbl_runtime import (
        _ATMOSPHERE_LAYERS, _COLUMN_FIELDS, _STATE_FIELD,
        MYNN_PBL_DIAGNOSTICS_2D, MYNN_PBL_DIAGNOSTICS_INT_2D, mynn_pbl_step)
    from gpuwm.core.mynn_pbl_scratch import MynnPblScratch
    from gpuwm.core import mynn_pbl_gpu
    native_get = mynn_pbl_gpu.mynn_pbl_kernel
    observed_predictors = []
    def observe_kernel(entry, identity="wrf_461"):
        kernel = native_get(entry, identity)
        if entry != "mynn_predict_default_columns":
            return kernel
        def launch(*args):
            observed_predictors.append((entry, identity))
            return kernel(*args)
        return launch
    monkeypatch.setattr(mynn_pbl_gpu, "mynn_pbl_kernel", observe_kernel)
    states, atmospheres, fields = [], [], []
    for member in range(members):
        state, cfg, driver = _build()
        driver.fields["tsk"] += cp.float32(member * 0.125)
        if version == "gsd_41":
            state.qc[:8] = cp.float32(0.0002 * (member + 1))
            state.qi[8:12] = cp.float32(0.0001 * (member + 1))
            driver.fields["qc_bl"][:8] = cp.float32(0.0001 * (member + 1))
            driver.fields["qi_bl"][8:12] = cp.float32(0.00005 * (member + 1))
            driver.fields["cldfra_bl"][:12] = cp.float32(0.375)
        atmosphere = _prepare_atmosphere(state)
        driver._run_sfclay(atmosphere, cfg)
        if version == "gsd_41":
            driver.fields["qfx"].fill(cp.float32(-0.0001 * (member + 1)))
        states.append(state)
        atmospheres.append(atmosphere)
        fields.append(driver.fields)
    names = {name for _, name in _ATMOSPHERE_LAYERS}
    packed_atmosphere = {name: cp.concatenate([atmosphere[name] for atmosphere in atmospheres], axis=1)
                         for name in names}
    names3 = set(_STATE_FIELD.values()) | {"exch_h", "exch_m"}
    names2 = {name for _, name in _COLUMN_FIELDS} | {"rmol", "pblh", "kpbl",
              *MYNN_PBL_DIAGNOSTICS_2D, *MYNN_PBL_DIAGNOSTICS_INT_2D}
    packed_fields = {name: cp.concatenate([field[name] for field in fields], axis=1 if name in names3 else 0)
                     for name in names3 | names2}
    packed_w = cp.concatenate([state.w for state in states], axis=1)
    options = {"bl_mynn_mixlength": mixlength, "bl_mynn_version": version,
               "bl_mynn_gsd41_unsquared_qtke": unsquared, "bl_mynn_cloud_tendency_form": cloud_form}
    bound = prepare_mynn_pbl_column_batch(packed_atmosphere, packed_fields, w=packed_w,
        members=members, ny=cfg.ny, nx=cfg.nx, dx=cfg.dx, mp_physics=cfg.mp_physics,
        column_chunk=members * cfg.ny * cfg.nx, available_bytes=1 << 30,
        options=options)
    before = {name: array.get().tobytes() for name, array in packed_atmosphere.items()}
    pointers = {name: int(array.data.ptr) for name, array in bound.storage.arrays.items()}
    # Poison only native write-before-read slots. Constant feeds retain the
    # zeros the ordinary holder gives its inactive branch inputs.
    slots = {name: array for name, array in bound.scratch_state.buffers.items()
             if not name.startswith("mynn_pbl_out_")}
    poison = MynnPblScratch(slots, chunk=members * cfg.ny * cfg.nx, nz=cfg.nz)
    for timestep in (1, 2, 3):
        poison.poison()
        count_before = len(observed_predictors)
        rates = bound(delt=cfg.dt, itimestep=timestep)
        assert len(observed_predictors) == count_before + 1
        assert observed_predictors[-1] == ("mynn_predict_default_columns", version)
        for member in range(members):
            ordinary = mynn_pbl_step(atmospheres[member], fields[member], w=states[member].w,
                dx=cfg.dx, delt=cfg.dt, itimestep=timestep, mp_physics=cfg.mp_physics,
                state=states[member], column_chunk=13, **options)
            stripe = slice(member * cfg.ny, (member + 1) * cfg.ny)
            for name, expected in ordinary.items():
                got = rates[name][:, stripe]
                assert got.get().tobytes() == expected.get().tobytes(), (member, timestep, name, mixlength)
                assert bool(cp.all(cp.isfinite(got)))
            for name in names3 | names2:
                got = packed_fields[name][:, stripe] if name in names3 else packed_fields[name][stripe]
                assert got.get().tobytes() == fields[member][name].get().tobytes(), (member, timestep, name, mixlength)
        assert {name: array.get().tobytes() for name, array in packed_atmosphere.items()} == before
        assert {name: int(array.data.ptr) for name, array in bound.storage.arrays.items()} == pointers
    assert bound.receipt["pieces_per_call"] == 1
    assert bound.receipt["members"] == members
    assert bound.receipt["options"]["bl_mynn_version"] == version
    assert bound.receipt["options"]["bl_mynn_gsd41_unsquared_qtke"] is unsquared
    assert bound.receipt["options"]["bl_mynn_cloud_tendency_form"] == cloud_form

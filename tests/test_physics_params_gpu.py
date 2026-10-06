"""Parameter table scaling and safety checks on the actual GPU path."""

import struct
from dataclasses import fields

import pytest

from conftest import requires_gpu


def _float32_words(values):
    return b"".join(struct.pack("<f", value) for value in values)


def test_empty_parameter_scaling_does_not_import_cupy(monkeypatch):
    import builtins
    from gpuwm.core.physics_param_gpu import scale_physics_param_values

    original = builtins.__import__

    def without_cupy(name, *args, **kwargs):
        if name == "cupy" or name.startswith("cupy."):
            pytest.fail("an empty parameter edit opened the CUDA path")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_cupy)
    assert scale_physics_param_values((), (), ()) == ()
    with pytest.raises(ValueError, match="differ in length"):
        scale_physics_param_values((), (1.0,), ())


@pytest.mark.gpu
@requires_gpu
def test_table_scaling_rounds_binary64_product_once():
    from gpuwm.core.physics_param_gpu import scale_physics_param_values

    # Values and factors remain binary64 until the product is rounded. The
    # first pair also detects casting either input to REAL before scaling.
    values = (0.2, 0.03, 0.0123456789123, 150.0, 0.14, 0.125)
    factors = (1.125, 1.25, 1.111111119, 1.125, 0.93, 1.01)
    actual = scale_physics_param_values(values, factors, (False,) * len(values))
    expected = tuple(value * factor for value, factor in zip(values, factors))
    assert _float32_words(actual) == _float32_words(expected)
    first_real = struct.unpack("<f", struct.pack("<f", values[0]))[0]
    assert _float32_words((first_real * factors[0],)) != _float32_words(actual[:1])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("value,factor,crop,reason", [
    (0.2, 0.625, True, "seasonal-crop roughness"),
    (0.2, 0.6, True, "seasonal-crop roughness"),
    (0.2, 0.0, False, "nonpositive"),
    (0.2, -1.0, False, "nonpositive"),
    (float("inf"), 1.0, False, "nonfinite"),
    (0.2, float("nan"), False, "nonfinite"),
    (1.0e300, 1.0e10, False, "nonfinite"),
])
def test_table_scaling_rejects_unsafe_results(value, factor, crop, reason):
    from gpuwm.core.physics_param_gpu import scale_physics_param_values

    with pytest.raises(ValueError, match=reason):
        scale_physics_param_values((value,), (factor,), (crop,))


@pytest.mark.gpu
@requires_gpu
def test_non_crop_roughness_below_crop_decrement_is_valid():
    from gpuwm.core.physics_param_gpu import scale_physics_param_values

    actual = scale_physics_param_values((0.2, 0.2), (0.6, 0.626), (False, True))
    assert _float32_words(actual) == _float32_words((0.12, 0.1252))


def _assert_bundle_cell_isolation(base, edited, pset):
    """Independent expected cells, checking every field of every row."""
    from gpuwm import physics_params as pp

    expected = {}
    for name in pset.changed():
        parameter = pp.registry()[name]
        for section, categories in parameter.categories.items():
            for category in categories:
                cell = (section, category, parameter.column)
                assert cell not in expected
                expected[cell] = (name, pset.value(name))
    changed = set()
    for section, before_table in base.vegetation.items():
        after_table = edited.vegetation[section]
        for field in fields(before_table):
            if field.name != "rows":
                assert getattr(after_table, field.name) == getattr(before_table, field.name)
        assert len(after_table.rows) == len(before_table.rows)
        for before, after in zip(before_table.rows, after_table.rows):
            for field in fields(before):
                cell = (section, before.category, field.name)
                old = getattr(before, field.name)
                new = getattr(after, field.name)
                if cell in expected:
                    _, factor = expected[cell]
                    assert _float32_words((new,)) == _float32_words((old * factor,)), cell
                    assert struct.pack("<d", new) != struct.pack("<d", old), cell
                    changed.add(cell)
                elif isinstance(old, float):
                    assert struct.pack("<d", new) == struct.pack("<d", old), cell
                else:
                    assert new == old, cell
    assert changed == set(expected)
    assert edited.soil is base.soil
    assert {key: value for key, value in edited.receipt.items()
            if key != "physics_params"} == dict(base.receipt)
    receipts = edited.receipt["physics_params"]["edits"]
    assert len(receipts) == len(expected)
    for entry in receipts:
        cell = (entry["section"], entry["category"], entry["column"])
        name, factor = expected[cell]
        before = getattr(base.vegetation[cell[0]].rows[cell[1] - 1], cell[2])
        after = getattr(edited.vegetation[cell[0]].rows[cell[1] - 1], cell[2])
        assert entry["constant"] == name and entry["multiplier"] == factor
        assert struct.pack("<d", entry["before"]) == struct.pack("<d", before)
        assert struct.pack("<d", entry["after"]) == struct.pack("<d", after)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", ("ruc.z0.tall", "ruc.z0.short", "ruc.rs"))
@pytest.mark.parametrize("bound", ("lower", "upper"))
def test_each_registered_ruc_row_endpoint_changes_only_named_table_cells(name, bound):
    from gpuwm import physics_params as pp
    from gpuwm.core.ruc import load_ruc_parameters
    from gpuwm.core.physics_param_gpu import scale_physics_param_values

    base = load_ruc_parameters()
    parameter = pp.registry()[name]
    pset = pp.make_set("endpoint", {name: getattr(parameter, bound)})
    edited = pp.apply_ruc_edits(base, pset, scale_physics_param_values)
    _assert_bundle_cell_isolation(base, edited, pset)
    assert load_ruc_parameters().vegetation == base.vegetation


@pytest.mark.gpu
@requires_gpu
def test_combined_ruc_columns_preserve_original_before_values_and_cumulative_edits():
    from gpuwm import physics_params as pp
    from gpuwm.core.ruc import load_ruc_parameters
    from gpuwm.core.physics_param_gpu import scale_physics_param_values

    base = load_ruc_parameters()
    pset = pp.make_set("combined", {
        "ruc.z0.tall": 1.5, "ruc.z0.short": 0.6, "ruc.rs": 1.5})
    edited = pp.apply_ruc_edits(base, pset, scale_physics_param_values)
    _assert_bundle_cell_isolation(base, edited, pset)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", ("ruc.z0.tall", "ruc.z0.short"))
@pytest.mark.parametrize("dataset,section", (
    ("MODIFIED_IGBP_MODIS_NOAH", "MODI-RUC"), ("USGS", "USGS-RUC")))
def test_roughness_row_reaches_live_surface_kernel_only_in_named_categories(
        name, dataset, section):
    import cupy as cp
    import numpy as np
    from gpuwm import physics_params as pp
    from gpuwm.core.ruc import load_ruc_parameters
    from gpuwm.core.ruc_gpu import ruc_surface_parameters_cuda
    from gpuwm.core.physics_param_gpu import scale_physics_param_values

    base = load_ruc_parameters()
    pset = pp.make_set("roughness", {name: 1.5})
    edited = pp.apply_ruc_edits(base, pset, scale_physics_param_values)
    count = len(base.vegetation[section].rows)
    vegetation = cp.arange(1, count + 1, dtype=cp.int32)
    inputs = (cp.ones(count, dtype=cp.int32), vegetation,
              *(cp.full(count, value, dtype=cp.float32)
                for value in (10.0, 80.0, 60.0, 0.05, 2.0)))
    default = ruc_surface_parameters_cuda(*inputs, mminlu=dataset)
    member = ruc_surface_parameters_cuda(*inputs, mminlu=dataset, parameters=edited)
    selected = np.zeros(count, dtype=bool)
    selected[np.asarray(pp.registry()[name].categories[section]) - 1] = True
    for field in fields(default):
        before = cp.asnumpy(getattr(default, field.name))
        after = cp.asnumpy(getattr(member, field.name))
        assert before.dtype == after.dtype
        if field.name == "znt":
            assert before[~selected].tobytes() == after[~selected].tobytes()
            assert np.all(before[selected].view(np.uint32) != after[selected].view(np.uint32))
            expected = [row.z0 for row in edited.vegetation[section].rows]
            assert after[selected].tobytes() == _float32_words(
                value for value, used in zip(expected, selected) if used)
        else:
            assert before.tobytes() == after.tobytes(), field.name


@pytest.mark.gpu
@requires_gpu
def test_rs_row_reaches_live_fused_sfctmp_and_preserves_unlisted_columns(monkeypatch):
    import cupy as cp
    import numpy as np
    from gpuwm import physics_params as pp
    from gpuwm.core.ruc_gpu import (
        ruc_sfctmp_full_width_fused, ruc_sfctmp_full_width_reference)
    from gpuwm.core.physics_param_gpu import scale_physics_param_values
    from test_ruc_sfctmp_fused import _capture, _resize

    values, keywords = _resize(*_capture(monkeypatch, "warm", 9), 37)
    base = keywords["parameters"]
    pset = pp.make_set("resistance", {"ruc.rs": 1.5})
    edited = pp.apply_ruc_edits(base, pset, scale_physics_param_values)
    # Grassland reads the edited RS row. Barren columns keep every original
    # table word and are the independent cell-isolation control.
    vegetation = cp.where(cp.arange(37) % 2 == 0, 10, 16).astype(cp.int32)
    keywords.update(ivgtyp=vegetation, iland=vegetation.copy())
    run = cp.ones(37, dtype=cp.bool_)
    baseline = ruc_sfctmp_full_width_fused(values, run=run, **keywords)
    tuned_keywords = dict(keywords, parameters=edited)
    member = ruc_sfctmp_full_width_fused(values, run=run, **tuned_keywords)
    reference = ruc_sfctmp_full_width_reference(values, run=run, **tuned_keywords)
    assert set(baseline) == set(member) == set(reference)
    selected = np.arange(37) % 2 == 0
    changed = []
    for name in member:
        before = cp.asnumpy(baseline[name])
        after = cp.asnumpy(member[name])
        expected = cp.asnumpy(reference[name])
        assert after.dtype == expected.dtype == before.dtype
        assert after.tobytes() == expected.tobytes(), name
        assert after[..., ~selected].tobytes() == before[..., ~selected].tobytes(), name
        if after[..., selected].tobytes() != before[..., selected].tobytes():
            changed.append(name)
    assert changed, "RS scaling reached no active grassland column"

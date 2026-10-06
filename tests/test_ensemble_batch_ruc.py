"""CPU contracts for member ownership around the original RUC arithmetic."""
import re

import numpy as np
import pytest

from gpuwm.core import ruc_memory
from gpuwm.core.ruc_tier import ruc_fused_source
from gpuwm.ensemble.batch_ruc import ENTRIES, packed_ruc_source, workspace_allocations


def _entries(source, prefix=""):
    result = {}
    for name in ENTRIES:
        match = re.search(r'extern "C" __global__ void ' + prefix + name
                          + r'\(.*?\)\s*\{', source, re.S)
        assert match, name
        start, depth, position = match.end(), 1, match.end()
        while depth:
            depth += (source[position] == "{") - (source[position] == "}")
            position += 1
        result[name] = (match.start(), start, position, source[start:position - 1])
    return result


@pytest.mark.parametrize("nzs", [6, 9])
@pytest.mark.parametrize("lineage", ["wrf_45", "wrf_461"])
@pytest.mark.parametrize("snow", ["wrf_45", "wrf_461"])
def test_every_original_science_body_and_other_source_byte_is_retained(nzs, lineage, snow):
    """All six bodies and every shared helper keep their source of record."""
    ordinary = ruc_fused_source(nzs, soilprop=lineage, snow=snow)
    packed = packed_ruc_source(nzs, soilprop=lineage, snow=snow)
    left, right = _entries(ordinary), _entries(packed, "packed_")
    for name in ENTRIES:
        body = left[name][3]
        assert right[name][3].endswith(body), name
        assert "blockIdx.y" in right[name][3][:-len(body)]
    # Restore each entry header and remove only its ownership binding.
    # The restored translation unit must be the complete ordinary source,
    # including default compiler macros, arithmetic helpers and constants.
    restored = packed[:packed.index('\nextern "C" __global__ void packed_ruc_bind_exner')]
    for name, (first, begin, end, body) in sorted(right.items(), key=lambda item: item[1][0], reverse=True):
        original_start, original_begin, original_end, _ = left[name]
        restored = restored[:first] + ordinary[original_start:original_end] + restored[end:]
    assert restored == ordinary


def test_flags_census_and_conditional_commit_are_member_private():
    source = packed_ruc_source()
    flags = ruc_memory.DRIVER_FLAG_SLOTS + ruc_memory.SFCTMP_FLAGS_SIZE
    assert source.count(f"flags += member * {flags * 2};") == 3
    assert len(re.findall(r"\bflags \+= member \* " + str(flags) + ";", source)) == 3
    assert f"sflags += member * {flags};" in source
    assert source.count("dt = member_dt[member]") == 2
    assert source.count("delt = member_dt[member]") == 3
    assert "ktau = member_ktau[member]" in source
    assert "qvg_air = member_qvg_air[member]" in source
    assert "irrigation = member_irrigation[member]" in source
    assert "log_profile = member_log_profile[member]" in source
    assert source.count("if(nlcat) landusef += member * nlcat * n;") == 2
    assert "for(int word=0;word<36;++word) if(flags[word]) return;" in source


@pytest.mark.parametrize("members", [1, 4, 8])
@pytest.mark.parametrize("nzs", [6, 9])
def test_inventory_prices_all_member_banks_and_science_outputs(members, nzs):
    actual = workspace_allocations(members, (3, 17), nzs)
    original = ruc_memory.driver_workspace_allocations(51, nzs)
    for name, (shape, dtype) in original.items():
        if name != "cptr":
            assert actual[name] == ((members,) + shape, dtype)
    assert actual["cptr"] == ((members, len(ruc_memory.DRIVER_TARGET_NAMES)), "uint64")
    for name, (shape, dtype) in ruc_memory.sfctmp_output_allocations(51, nzs).items():
        assert actual["sf_output:" + name] == ((members,) + shape, dtype)
    assert actual["exner_upload"] == ((members, 2, 51), "float32")
    assert actual["member_dt"] == ((members,), "float32")
    assert actual["member_ktau"] == ((members,), "int32")
    for name in ("member_qvg_air", "member_irrigation", "member_log_profile"):
        assert actual[name] == ((members,), "int32")
    assert actual["alive"] == ((members, 51), "bool")
    assert ruc_memory.allocation_bytes(actual) >= ruc_memory.allocation_bytes(actual, rounded=False)


@pytest.mark.parametrize("members,shape,nzs", [(0, (2, 3), 6), (2, (0, 3), 6),
                                             (2, (3,), 6), (2, (2, 3), 4)])
def test_invalid_bank_dimensions_refuse_before_allocation(members, shape, nzs):
    with pytest.raises(ValueError, match="positive member/grid extents"):
        workspace_allocations(members, shape, nzs)


@pytest.mark.parametrize("nzs", [6, 9])
def test_every_generated_scratch_pointer_has_the_original_type_offset_and_member_pitch(nzs):
    from gpuwm.core.ruc_sfctmp_layout import _SFCTMP_ARRAYS, _SFCTMP_OUTPUTS
    from gpuwm.ensemble.batch_ruc import _scratch_views
    n, members = 17, 4
    offsets, elements = ruc_memory.sfctmp_scratch_layout(n, nzs)
    slab = np.empty((members, elements), dtype=np.float32)
    views = _scratch_views(slab, n, nzs)
    assert {str(value.dtype) for value in views.values()} == {"float32", "int32", "bool"}
    for index, (binding, dtype, profile, slot) in enumerate(_SFCTMP_ARRAYS):
        if binding or index in _SFCTMP_OUTPUTS.values():
            assert index not in views
            continue
        view = views[index]
        assert view.dtype == np.dtype(dtype)
        assert view.shape == ((members, nzs, n) if profile else (members, n))
        assert view.strides[0] == slab.strides[0]
        assert view.ctypes.data == slab.ctypes.data + 4 * offsets[slot]
        assert view[1].ctypes.data - view[0].ctypes.data == slab.strides[0]
        assert view.strides[-1] == np.dtype(dtype).itemsize


def test_current_entry_headers_keep_each_original_argument_once_plus_private_selector_banks():
    original, packed = ruc_fused_source(6), packed_ruc_source(6)
    for name, extras in {
        "ruc_driver_prologue": ("member_dt", "member_ktau", "member_qvg_air"),
        "ruc_driver_epilogue": ("member_dt", "member_irrigation", "member_log_profile"),
    }.items():
        def arguments(source, prefix):
            header = re.search(r'extern "C" __global__ void ' + prefix + name + r'\((.*?)\)\s*\{', source, re.S).group(1)
            return tuple(re.search(r'(\w+)\s*$', item.strip()).group(1) for item in header.split(","))
        ordinary, joined = arguments(original, ""), arguments(packed, "packed_")
        assert joined == ordinary + extras
        assert len(joined) == len(set(joined))


def test_snow_lineage_changes_source_cache_identity_without_replacing_driver_arithmetic():
    import hashlib
    assert hashlib.sha256(packed_ruc_source(snow="wrf_45").encode()).digest() != hashlib.sha256(
        packed_ruc_source(snow="wrf_461").encode()).digest()


@pytest.mark.parametrize("name,value", [("soilprop", ["wrf_45", "wrf_461"]), ("snow", ["wrf_45", "wrf_461"])])
def test_mixed_compile_metadata_refuses_before_any_source_or_allocation(name, value):
    with pytest.raises(ValueError):
        packed_ruc_source(**{name: value})


def test_private_parameter_set_keeps_the_same_ordinary_ruc_source_and_only_private_bindings():
    from gpuwm import physics_params
    physics_params.reset_for_tests()
    try:
        physics_params.declare(physics_params.make_set("ruc-table-qualification",
            {"ruc.z0.short": 1.1, "ruc.rs": 0.9}), source="packed RUC source qualification")
        for snow in ("wrf_45", "wrf_461"):
            ordinary, packed = ruc_fused_source(6, snow=snow), packed_ruc_source(6, snow=snow)
            left, right = _entries(ordinary), _entries(packed, "packed_")
            for name in ENTRIES:
                assert right[name][3].endswith(left[name][3])
    finally:
        physics_params.reset_for_tests()


@pytest.mark.parametrize("members", [4, 8])
def test_selector_oracle_initializes_every_missing_carrier_before_divergent_carried_clocks(members):
    import numpy as np
    from test_ensemble_batch_ruc_gpu import _selector_counts
    initial = _selector_counts(1, members)
    assert initial.dtype == np.dtype("int32") and initial.shape == (members,)
    assert initial.tolist() == [1] * members
    previous = initial
    for call in (2, 3, 4):
        counts = _selector_counts(call, members)
        assert counts.shape == (members,) and counts.dtype == initial.dtype
        assert bool(np.all(counts > previous))
        assert len(set(counts.tolist())) == members
        previous = counts

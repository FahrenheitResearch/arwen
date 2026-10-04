"""Current metric ABI and integer-only compiled member addressing."""
import re
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core.kernels import module_source
from gpuwm.ensemble.batch_kernel import KernelSpec, PointerSpec, _close, _masked, _entry_parts
from gpuwm.ensemble.batch_mixing import (
    _GRID_MACRO_ENTRIES, _SMAG_PTX_ENTRIES, _member_smag_ptx,
    normalized_smag_source, metric_w_entry_family, required_scratch_slots,
)
from gpuwm.ensemble.batch_state import BatchStateUnsupported


def body(source, entry):
    masked = _masked(source)
    declarations = list(re.finditer(r"\bvoid\s+" + re.escape(entry) + r"\s*\(", masked))
    assert len(declarations) == 1
    opening = declarations[0].end() - 1
    parameters_end = _close(masked, opening, "(", ")")
    body_start = masked.index("{", parameters_end)
    body_end = _close(masked, body_start, "{", "}")
    return source[body_start:body_end + 1]


def test_every_installed_macro_signature_expands_and_preserves_original_cuda_body_bytes():
    source = module_source("smag2d")
    expanded = normalized_smag_source()
    names = re.findall(r"\bvoid\s+(\w+)\s*\(\s*WRF_SMAG_GRID_ARGS\b", source)
    assert len(names) == len(_GRID_MACRO_ENTRIES) == 23 and set(names) == _GRID_MACRO_ENTRIES
    for entry in names:
        assert body(source, entry).encode() == body(expanded, entry).encode(), entry
        signature = re.search(r"\bvoid\s+" + re.escape(entry) + r"\s*\(", expanded)
        end = _close(_masked(expanded), signature.end() - 1, "(", ")")
        assert "WRF_SMAG_GRID_ARGS" not in expanded[signature.start():end]


@pytest.mark.parametrize("options,entries", [
    (("-arch=compute_120",), ("wrf_smag_w_stress", "wrf_smag_hd_w_stress", "wrf_smag_w_primitives", "wrf_smag_hd_w_cached")),
    (("-arch=compute_90",), ("wrf_smag_w_primitives", "wrf_smag_hd_w_cached")),
    (("-arch=compute_120", "-DGPUWM_WRF_EXACT=1"), ("wrf_smag_hd_w",)),
])
def test_original_architecture_w_entries_have_complete_explicit_float_pointer_abi(options, entries):
    common = ("u", "v", "w", "php", "phb", "alt", "qv", "msft", "msfu", "msfv", "fnm", "fnp", "dn", "dnw")
    caches = ("cached_what", "cached_rdz", "cached_zx", "cached_zy")
    extra = {"wrf_smag_w_stress": ("km", "tx", "ty"), "wrf_smag_hd_w_stress": ("tx", "ty", "tend"),
             "wrf_smag_w_primitives": caches, "wrf_smag_hd_w_cached": ("km",) + caches + ("tend",),
             "wrf_smag_hd_w": ("km", "tend")}
    for entry in entries:
        spec = KernelSpec("smag2d", entry, tuple(PointerSpec(name, "member") for name in common + extra[entry]))
        names = _entry_parts(normalized_smag_source(), spec, options)[-2]
        assert names[:14] == common
        assert names[-6:] == ("nz", "ny", "nx", "phb3d", "boundary_x", "boundary_y")
    assert metric_w_entry_family(exact="GPUWM_WRF_EXACT" in " ".join(options), compute_capability="120" if "120" in options[0] else "90") == (
        ("wrf_smag_hd_w",) if "GPUWM_WRF_EXACT" in " ".join(options)
        else ("wrf_smag_w_stress", "wrf_smag_hd_w_stress") if "120" in options[0]
        else ("wrf_smag_w_primitives", "wrf_smag_hd_w_cached"))


def test_new_or_missing_macro_entry_is_not_silently_admitted(monkeypatch):
    source = module_source("smag2d")
    import gpuwm.core.kernels as kernels
    for changed in (source.replace("void wrf_smag_w_stress(", "void wrf_smag_future_stress("),
                    source + '\nextern "C" __global__ void wrf_smag_unreviewed(WRF_SMAG_GRID_ARGS) {}\n'):
        monkeypatch.setattr(kernels, "module_source", lambda name: changed)
        with pytest.raises(BatchStateUnsupported, match="signature inventory"):
            normalized_smag_source()


@pytest.mark.parametrize("entry", sorted(_SMAG_PTX_ENTRIES))
@pytest.mark.parametrize("pointer_form", [".u64 .ptr .align 4", ".u64"],
                         ids=["annotated-sm_120", "bare-sm_89"])
def test_admitted_entry_rebinding_adds_integer_addresses_and_preserves_float_instructions(entry, pointer_form):
    ptx = '''.version 8.0
.target sm_120
.address_size 64
.visible .entry ENTRY(
.param POINTER input,
.param POINTER output,
.param .f32 coefficient,
.param .u32 nx
) {
.reg .b32 %r<4>;
.reg .b64 %rd<4>;
.reg .f32 %f<4>;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
mov.u32 %r1, %ctaid.x;
ld.param.f32 %f1, [coefficient];
mul.f32 %f2, %f1, %f1;
st.global.f32 [%rd2], %f2;
ret;
}
'''.replace("ENTRY", entry).replace("POINTER", pointer_form)
    spec = KernelSpec("smag2d", entry, (PointerSpec("input", "shared"), PointerSpec("output", "member")))
    expanded, _ = _member_smag_ptx(ptx, spec, 20, ("input", "output", "coefficient", "nx"))
    floating = lambda value: [line.strip() for line in value.splitlines() if re.search(r"\.(?:f32|f64)\b", line)]
    assert floating(expanded) == floating(ptx)
    assert "add.u64 %rd2" in expanded and "add.u64 %rd1" not in expanded
    assert "div.u32" in expanded and "rem.u32" in expanded
    # A .u64 parameter that is not an audited pointer, or a pointer declared
    # as a scalar, is refused whichever annotation the target emits.
    from gpuwm.ensemble.batch_state import BatchStateUnsupported
    with pytest.raises(BatchStateUnsupported, match="parameter types differ"):
        _member_smag_ptx(ptx, spec, 20, ("input", "coefficient", "output", "nx"))


def test_primitive_cache_borrows_registered_step_dead_acoustic_slots():
    from gpuwm.core.preflight import scratch_slot_registry
    cfg = RunConfig(nx=13, ny=12, nz=5, dx=3000., dy=3000., ztop=9000., dt=3., run_seconds=30., km_opt=4, diff_opt=2)
    slots = required_scratch_slots(cfg)
    registry = scratch_slot_registry(cfg)
    assert {"acoustic_a", "acoustic_c2a"} <= slots.keys()
    assert registry["acoustic_a"] == (cfg.nz + 1, cfg.ny, cfg.nx)
    assert registry["acoustic_c2a"] == (cfg.nz, cfg.ny, cfg.nx)


@pytest.mark.parametrize("exact,architecture", [(False, "120"), (False, "90"), (True, "120")])
def test_w_family_constructor_binds_current_abi_and_original_launch_order(monkeypatch, exact, architecture):
    import gpuwm.core.dycore as original
    import gpuwm.ensemble.batch_mixing as mixing
    cfg = RunConfig(nx=13, ny=12, nz=5, dx=3000., dy=3000., ztop=9000., dt=3., run_seconds=30., km_opt=4, diff_opt=2)
    state = SimpleNamespace(cfg=cfg, storage=SimpleNamespace(specs={"w0": SimpleNamespace(shape=(6, 12, 13))}))
    common = tuple(object() for _ in range(14)) + tuple(np.float32(1) for _ in range(7)) + (np.int32(0),)
    dims = tuple(np.int32(value) for value in (5, 12, 13, 0, 0, 0))
    target = SimpleNamespace(device=SimpleNamespace(compute_capability=architecture))
    monkeypatch.setattr(original, "WRF_EXACT", exact)
    monkeypatch.setattr(mixing, "_array", lambda *args, **kwargs: object())
    seen = []
    def bind(state, module, entry, fields, arguments, grid):
        options = ("-arch=compute_" + architecture,) + (("-DGPUWM_WRF_EXACT=1",) if exact else ())
        spec = KernelSpec(module, entry, tuple(PointerSpec(name, "member") for name, _ in fields))
        names = _entry_parts(normalized_smag_source(), spec, options)[-2]
        assert len(names) == len(arguments)
        assert arguments[:len(common)] == common and arguments[-6:] == dims
        seen.append((entry, grid))
        return entry
    monkeypatch.setattr(mixing, "_raw", bind)
    result = mixing._smag_w_launches(state, common, dims, object(), target, object(), object())
    assert result == metric_w_entry_family(exact=exact, compute_capability=architecture)
    assert tuple(row[0] for row in seen) == result
    assert seen[-1][1] == (1, 12, 6)

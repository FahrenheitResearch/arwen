"""One family dispatch must retain all four scalar advection word sequences."""
import re

import numpy as np
import pytest

from conftest import requires_gpu


def test_family_source_n1_and_pointer_abi_are_closed():
    from gpuwm.core.kernels import module_source
    from gpuwm.ensemble.batch_kernel import _entry_parts, generate_batch_source
    from gpuwm.ensemble.batch_layout_trials import advection_family_source
    original = module_source("advection")
    for members in (1, 4, 10, 20, 40):
        for layout in ("outermost", "innermost"):
            for zero in (False, True):
                candidate, spec = advection_family_source(original, members, layout=layout,
                                                          zero_tendency=zero)
                if members == 1:
                    assert candidate == original
                    assert spec is None
                else:
                    _entry_parts(candidate, spec)
                    assert sum(p.role == "member" for p in spec.pointers) == 11
                    assert sum(p.role == "shared" for p in spec.pointers) == 7
                    if layout == "outermost":
                        generate_batch_source(candidate, spec, members)


def test_family_device_bodies_change_only_virtual_coordinates():
    from gpuwm.core.kernels import module_source
    from gpuwm.ensemble.batch_kernel import _entry_parts, _masked, _close
    from gpuwm.ensemble.batch_layout_trials import advection_family_source, flux_trial_source
    from gpuwm.ensemble.batch_operators import _FLUX_SPECS
    original = module_source("advection")
    for layout in ("outermost", "innermost"):
        family, _ = advection_family_source(original, 20, layout=layout, zero_tendency=True)
        masked_family = _masked(family)
        for spec in _FLUX_SPECS.values():
            single = flux_trial_source(original, spec, 20, layout=layout, zero_tendency=True)
            parts = _entry_parts(single, spec)
            body = _masked(single[parts[3] + 1:parts[4]])
            expected = re.sub(r"\bblockIdx\s*\.\s*([xyz])\b",
                              lambda match: "__trial_block_" + match[1], body)
            match = re.search(r"void __trial_" + spec.entry + r"\s*\(", masked_family)
            signature_end = _close(masked_family, match.end() - 1, "(", ")")
            start = masked_family.index("{", signature_end)
            end = _close(masked_family, start, "{", "}")
            actual = masked_family[start + 1:end]
            assert "".join(actual.split()) == "".join(expected.split())
        for helper in ("xface_cell_open", "xface_cell_per", "yface_cell_open", "yface_cell_per",
                       "zface_half", "w_lid_velocity"):
            assert len(re.findall(r"real\s+" + helper + r"\s*\(", masked_family)) == 1


def test_family_source_refuses_implicit_helper_coordinates():
    from gpuwm.core.kernels import module_source
    from gpuwm.ensemble.batch_kernel import BatchKernelUnsupported
    from gpuwm.ensemble.batch_layout_trials import advection_family_source
    source = module_source("advection").replace("int d = min(f, n - f);",
                                                 "int d = min(f, n - f) + blockIdx.x;", 1)
    with pytest.raises(BatchKernelUnsupported, match="helpers use implicit"):
        advection_family_source(source, 4)


def test_source_single_and_family_entries_retain_plain_c_linkage():
    from gpuwm.core.kernels import module_source
    from gpuwm.ensemble.batch_layout_trials import flux_trial_source, advection_family_source
    from gpuwm.ensemble.batch_operators import _FLUX_SPECS
    from gpuwm.ensemble.batch_kernel import generate_batch_source
    original = module_source("advection")
    for members in (4, 10, 20, 40):
        for layout in ("outermost", "innermost"):
            for spec in _FLUX_SPECS.values():
                candidate = flux_trial_source(original, spec, members, layout=layout, zero_tendency=True)
                emitted = generate_batch_source(candidate, spec, members) if layout == "outermost" else candidate
                assert re.search(r'extern\s+"C"\s+__global__\s+void\s+' + spec.entry + r"\s*\(", emitted)
            family, spec = advection_family_source(original, members, layout=layout)
            emitted = generate_batch_source(family, spec, members) if layout == "outermost" else family
            assert re.search(r'extern\s+"C"\s+__global__\s+void\s+flux_div_family\s*\(', emitted)
            assert not re.search(r'extern\s+"C"\s+__device__', emitted)


@pytest.mark.gpu
@requires_gpu
def test_n1_unpack_exact_contiguous_alias_is_a_word_preserving_noop(monkeypatch):
    import cupy as cp
    from gpuwm.ensemble.batch_layout_trials import pack_member_innermost, unpack_member_innermost
    bits = np.array([0, 0x80000000, 1, 0x3F800000, 0x7F800000, 0xFF800000,
                     0x7FC01234, 0x7FA05678], np.uint32)
    outer = cp.asarray(bits.view(np.float32).reshape(1, 2, 2, 2))
    packed = pack_member_innermost(outer)
    assert packed.data.ptr == outer.data.ptr
    def copied(*args, **kwargs):
        pytest.fail("an exact contiguous alias must return without a CUDA copy")
    monkeypatch.setattr(cp, "copyto", copied)
    assert unpack_member_innermost(packed, out=outer) is outer
    assert cp.asnumpy(outer).view(np.uint32).tobytes() == bits.tobytes()


@pytest.mark.gpu
@requires_gpu
def test_unpack_partial_overlap_still_refuses():
    import cupy as cp
    from gpuwm.ensemble.batch_layout_trials import unpack_member_innermost
    backing = cp.zeros(25, cp.float32)
    packed = backing[:24].reshape(2, 3, 4, 1)
    output = backing[1:].reshape(1, 2, 3, 4)
    with pytest.raises(ValueError, match="overlaps"):
        unpack_member_innermost(packed, out=output)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("members", (1, 4, 10, 20, 40))
@pytest.mark.parametrize("layout", ("outermost", "innermost"))
@pytest.mark.parametrize("core", ((1, 7, 128), (5, 9, 137)))
@pytest.mark.parametrize("boundary", ((False, False, False, False),
                                      (True, False, False, False),
                                      (True, True, True, True)))
def test_family_outputs_match_every_original_word_and_use_one_launch(
        members, layout, core, boundary, monkeypatch):
    import cupy as cp
    from gpuwm.ensemble import batch_kernel
    from gpuwm.ensemble.batch_layout_trials import (
        prepare_advection_family_trial, pack_member_innermost, unpack_member_innermost)
    from gpuwm.ensemble.batch_operators import prepare_flux_div
    nz, ny, nx = core
    mapped, open_x, open_y, specified = boundary
    rng = np.random.default_rng(19031)
    shapes = ((members, nz, ny, nx), (members, nz, ny, nx + 1),
              (members, nz, ny + 1, nx), (members, nz + 1, ny, nx))
    fields = tuple(cp.asarray(rng.uniform(-20, 35, shape).astype(np.float32)) for shape in shapes)
    flows = tuple(cp.asarray(rng.uniform(-60000, 80000, shape).astype(np.float32)) for shape in shapes[1:])
    rdnw = cp.asarray(np.linspace(-5, -1, nz, dtype=np.float32))
    rdn = rdnw.copy()
    fnm = cp.asarray(np.linspace(0.2, 0.8, nz, dtype=np.float32))
    fnp = cp.asarray(np.float32(1) - np.linspace(0.2, 0.8, nz, dtype=np.float32))
    maps = tuple(cp.asarray(rng.uniform(0.75, 1.5, shape[2:]).astype(np.float32)) for shape in shapes[:3])
    shared = (rdnw, rdnw, rdnw, rdn), (maps[0], maps[1], maps[2], maps[0])
    immutable = fields + flows + (rdnw, rdn, fnm, fnp) + maps
    snapshots = tuple(value.copy() for value in immutable)
    family_calls = []
    original_compiler = batch_kernel._compiled_source
    class ObservedKernel:
        def __init__(self, kernel):
            self.kernel = kernel
        def __getattr__(self, name):
            return getattr(self.kernel, name)
        def __call__(self, *args, **kwargs):
            family_calls.append(args[:2])
            return self.kernel(*args, **kwargs)
    def compiled(source, spec, *options):
        kernel, source_hash, compiled_hash = original_compiler(source, spec, *options)
        if spec.entry == "flux_div_family":
            kernel = ObservedKernel(kernel)
        return kernel, source_hash, compiled_hash
    monkeypatch.setattr(batch_kernel, "_compiled_source", compiled)
    options = dict(dx=950, dy=1375, open_x=open_x, open_y=open_y,
                   has_msf=mapped, specified=specified)
    for zero in (False, True):
        family_calls.clear()
        expected = tuple(cp.asarray(rng.uniform(-2, 3, shape).astype(np.float32)) for shape in shapes)
        for value in expected:
            value[:, 0, 0, :2] = cp.asarray(np.array([-0.0, 27.25], np.float32))
        actual = tuple(value.copy() for value in expected)
        working_fields = tuple(pack_member_innermost(value) for value in fields) if layout == "innermost" else fields
        working_flows = tuple(pack_member_innermost(value) for value in flows) if layout == "innermost" else flows
        working_out = tuple(pack_member_innermost(value) for value in actual) if layout == "innermost" else actual
        packed_snapshots = tuple(value.copy() for value in working_fields + working_flows) if layout == "innermost" else ()
        rows = tuple((field, tendency, spacing, msf, stagger) for field, tendency, spacing, msf, stagger in
                     zip(working_fields, working_out, shared[0], shared[1], ("", "x", "y", "z")))
        launch = prepare_advection_family_trial(rows, *working_flows, fnm, fnp,
                                                layout=layout, zero_tendency=zero, **options)
        originals = tuple(prepare_flux_div(
            fields[at][member:member + 1], *(value[member:member + 1] for value in flows),
            expected[at][member:member + 1], shared[0][at], fnm, fnp, shared[1][at],
            dx=950, dy=1375, open_x=open_x, open_y=open_y, has_msf=mapped,
            spec=specified, stagger=stagger)
            for at, stagger in enumerate(("", "x", "y", "z")) for member in range(members))
        for repeat in range(2):
            if zero:
                for value in expected:
                    value.fill(0)
            for original in originals:
                original()
            launch()
            if layout == "innermost":
                for value, output in zip(working_out, actual):
                    unpack_member_innermost(value, out=output)
            for at, (value, target) in enumerate(zip(actual, expected)):
                assert cp.asnumpy(value).view(np.uint32).tobytes() == cp.asnumpy(target).view(np.uint32).tobytes(), (
                    members, layout, core, boundary, zero, repeat, at)
        assert len(family_calls) == (0 if members == 1 else 2)
        for value, saved in zip(immutable, snapshots):
            assert cp.asnumpy(value).view(np.uint32).tobytes() == cp.asnumpy(saved).view(np.uint32).tobytes()
        for value, saved in zip(working_fields + working_flows, packed_snapshots):
            assert cp.asnumpy(value).view(np.uint32).tobytes() == cp.asnumpy(saved).view(np.uint32).tobytes()
        if members == 1:
            assert launch.metadata["n1_original_entries"]
        else:
            assert launch.metadata["family_kernel_launches_per_stage"] == 1


@pytest.mark.gpu
@requires_gpu
def test_family_refuses_cross_output_aliasing():
    import cupy as cp
    from gpuwm.ensemble.batch_layout_trials import prepare_advection_family_trial
    n, nz, ny, nx = 4, 5, 9, 11
    scalar = cp.ones((n, nz, ny, nx), cp.float32)
    u = cp.ones((n, nz, ny, nx + 1), cp.float32)
    v = cp.ones((n, nz, ny + 1, nx), cp.float32)
    w = cp.ones((n, nz + 1, ny, nx), cp.float32)
    outputs = tuple(cp.empty_like(value) for value in (scalar, u, v, w))
    profile = cp.ones(nz, cp.float32)
    maps = tuple(cp.ones(value.shape[2:], cp.float32) for value in (scalar, u, v))
    # One overlapping byte range is enough to make family writes unordered.
    backing = cp.empty(u.size, cp.float32)
    overlapping_scalar = backing[:scalar.size].reshape(scalar.shape)
    alias = backing.reshape(u.shape)
    rows = ((scalar, overlapping_scalar, profile, maps[0], ""), (u, alias, profile, maps[1], "x"),
            (v, outputs[2], profile, maps[2], "y"), (w, outputs[3], profile, maps[0], "z"))
    with pytest.raises(ValueError, match="overlaps"):
        prepare_advection_family_trial(rows, u, v, w, profile, profile, dx=1000, dy=1000)

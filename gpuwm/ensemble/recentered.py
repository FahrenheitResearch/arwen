"""Bounded source-member anomalies on already native-mapped physical fields.

This CUDA transform precedes native real initialization. It must be used for
both the initial atmosphere and every lateral-boundary knot. It does not add
uncorrelated prognostic noise, prescribe calibrated amplitudes, or replace the
native interpolation, hydrostatic integration and C-grid initialization.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from gpuwm.core.device_cache import cuda_cache

CONTRACT = "gpuwm-recentered-physical-fields.v1"
_CUDA_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false")

_SOURCE = r'''
extern "C" __global__ void recentered_physical(
    const float* base, const float* donors, const int* selected,
    float* out, unsigned long long cells, int population, int members,
    double amplitude, double bound, double lower, double upper) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cells * (unsigned long long)members) return;
    unsigned long long m = i / cells, cell = i % cells;
    float original = base[cell];
    // Existing interpolation undershoots belong to the base input. Keep
    // those words; the ensemble must not create new sub-floor values.
    if (amplitude == 0.0 || bound == 0.0 || (double)original < lower || (double)original > upper) {
        out[i] = original; return;
    }
    double mean = 0.0;
    for (int d = 0; d < population; ++d)
        mean = __dadd_rn(mean, (double)donors[(unsigned long long)d*cells + cell]);
    mean = __ddiv_rn(mean, (double)population);
    double low_exact = fmax(lower, __dsub_rn((double)original, bound));
    double high_exact = fmin(upper, __dadd_rn((double)original, bound));
    float low = __double2float_rn(low_exact), high = __double2float_rn(high_exact);
    // Inward representable endpoints also bound the final float32 rounding.
    if ((double)low < low_exact) low = nextafterf(low, __int_as_float(0x7f800000));
    if ((double)high > high_exact) high = nextafterf(high, -__int_as_float(0x7f800000));
    double scale = amplitude;
    for (int d = 0; d < population; ++d) {
        double delta = __dsub_rn((double)donors[(unsigned long long)d*cells+cell], mean);
        if (delta > 0.0) scale = fmin(scale, __ddiv_rn(__dsub_rn((double)high, (double)original), delta));
        if (delta < 0.0) scale = fmin(scale, __ddiv_rn(__dsub_rn((double)low, (double)original), delta));
    }
    double anomaly = __dsub_rn((double)donors[(unsigned long long)selected[m]*cells+cell], mean);
    if (scale == 0.0 || anomaly == 0.0) { out[i] = original; return; }
    out[i] = __double2float_rn(__dadd_rn((double)original, __dmul_rn(scale, anomaly)));
}
'''


@cuda_cache(maxsize=None)
def _cuda_module():
    """Preserve subnormal bounds and anomalies through the direct NVRTC route.

    RawModule appends FTZ=true after caller options. That makes valid tiny
    anomalies disappear and differs from the native CPU operation.
    """
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.kernel_compile_notice import observe_module_compile
    with observe_module_compile("gpuwm.ensemble.recentered"):
        binary, _ = compiler.compile_using_nvrtc(
            _SOURCE, _CUDA_OPTIONS, None, "ensemble_recenter.cu")
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary,str) else binary)
    return module


@dataclass(frozen=True)
class FieldBounds:
    """Physical units and explicit limits, never inferred from array names."""
    units: str
    max_increment: float
    lower: float
    upper: float
    amplitude: float = 1.0
    input_lower: float | None = None
    input_upper: float | None = None

    def __post_init__(self):
        if not self.units or any(not math.isfinite(value) for value in
                                 (self.max_increment, self.lower, self.upper, self.amplitude)):
            raise ValueError("physical field units and finite recentering bounds are required")
        if self.lower >= self.upper or self.max_increment < 0 or self.amplitude < 0:
            raise ValueError("recentered field needs ordered physical bounds and non-negative increment/amplitude")
        if self.input_lower is not None and (not math.isfinite(self.input_lower) or self.input_lower > self.lower):
            raise ValueError("input_lower must be finite and no greater than the physical lower bound")
        if self.input_upper is not None and (not math.isfinite(self.input_upper) or self.input_upper < self.upper):
            raise ValueError("input_upper must be finite and no less than the physical upper bound")

    @property
    def effective_input_lower(self):
        return self.lower if self.input_lower is None else self.input_lower

    @property
    def effective_input_upper(self):
        return self.upper if self.input_upper is None else self.input_upper


def recenter_field(base, donors, *, donor_ids, selected_ids, bounds: FieldBounds,
                   mapped_grid_sha256: str, array_module=None,
                   cpu_bridge=None, workers: int = 1):
    """Return member fields and a provenance receipt, with no input mutations.

    ``donor_ids`` identifies the complete fixed donor population. Its canonical
    order is mandatory: selecting fewer output members cannot change its mean
    or limiter. ``selected_ids`` may reorder or partition that population.
    A subset's mean anomaly need not be zero; recentering the subset again
    would break per-member independence from the requested member count.
    Native regridding must supply both arrays on the same physical coordinate
    levels, with the declared grid digest binding that preparation.
    """
    import numpy as np
    if array_module is None:
        import cupy as array_module
    xp = array_module
    donor_ids, selected_ids = tuple(donor_ids), tuple(selected_ids)
    if (len(donor_ids) < 2 or any(not isinstance(value, str) or not value for value in donor_ids)
            or len(set(donor_ids)) != len(donor_ids) or donor_ids != tuple(sorted(donor_ids))):
        raise ValueError("the complete donor population needs at least two unique IDs in canonical order")
    if (not selected_ids or any(not isinstance(value, str) or value not in donor_ids for value in selected_ids)
            or len(set(selected_ids)) != len(selected_ids)):
        raise ValueError("selected member IDs must be distinct members of the fixed donor population")
    if len(mapped_grid_sha256) != 64 or any(c not in "0123456789abcdef" for c in mapped_grid_sha256):
        raise ValueError("native-mapped grid identity needs its complete lowercase SHA-256")
    for name, array in (("base", base), ("donors", donors)):
        if not isinstance(array, xp.ndarray) or array.dtype != np.dtype("float32") or not array.flags.c_contiguous:
            raise ValueError(f"{name} must be contiguous device float32 storage from native field preparation")
        if not bool(xp.all(xp.isfinite(array))):
            raise ValueError(f"{name} contains non-finite source values; a bounded increment cannot repair missing atmosphere")
    if donors.shape != (len(donor_ids),) + base.shape or base.size == 0:
        raise ValueError("donor and base physical grids differ, so their anomaly cannot be applied")
    on_cpu = xp is np
    if not on_cpu and (base.device.id != donors.device.id or base.device.id != xp.cuda.runtime.getDevice()):
        raise ValueError("recentered inputs and launch must belong to the same device")
    if bool(xp.any(base.astype(xp.float64) < bounds.effective_input_lower)) or bool(xp.any(base.astype(xp.float64) > bounds.effective_input_upper)):
        raise ValueError("base physical field violates its declared bounds before recentering")
    indices = xp.asarray([donor_ids.index(value) for value in selected_ids], dtype=np.int32)
    out = xp.empty((len(selected_ids),) + base.shape, dtype=np.float32)
    if on_cpu:
        from gpuwm.ensemble.native_preparation import NativeEnsemblePreparation
        native = NativeEnsemblePreparation(cpu_bridge)
        native.recenter(base, donors, indices, out, bounds, workers=workers)
    else:
        kernel = _cuda_module().get_function("recentered_physical")
        kernel(((out.size + 255) // 256,), (256,),
               (base, donors, indices, out, np.uint64(base.size), np.int32(len(donor_ids)), np.int32(len(selected_ids)),
                np.float64(bounds.amplitude), np.float64(bounds.max_increment), np.float64(bounds.lower), np.float64(bounds.upper)))
        # Retain the selection buffer through this setup-only transform.
        xp.cuda.get_current_stream().synchronize()
    receipt = {"contract": CONTRACT, "donor_population": list(donor_ids),
               "selected_members": list(selected_ids), "mapped_grid_sha256": mapped_grid_sha256,
               "grid_identity_validation": "caller assertion; native coordinate/valid-time/unit receipts must be verified before this primitive",
               "bounds": asdict(bounds), "shape": list(base.shape),
               "kernel_sha256": hashlib.sha256(_SOURCE.encode()).hexdigest(),
               "calibration": "not calibrated", "stage": "physical fields before native real initialization",
               "rounding": "inward representable bounds; mean has final float32 roundoff",
               "boundary_rule": "same donor population and member at every valid-time knot"}
    receipt["backend"] = "rust-cpu" if on_cpu else "cuda"
    if on_cpu:
        from gpuwm.ensemble.physical_store import digest_file
        receipt["native_bridge_sha256"] = digest_file(native.path)
    else:
        receipt["compiler_backend"] = "direct-nvrtc"
        receipt["compiler_options"] = list(_CUDA_OPTIONS)
    return out, receipt

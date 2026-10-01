"""Cold and warm CUDA graph captures use the frozen eager base path."""
from pathlib import Path
import importlib.util

import numpy as np
import pytest


def _fixture(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A146: this frozen copy's divisions by a compile-time constant are spelled
# __fdiv_rn, as the production kernel has spelled them since that fix
# (NVRTC compiled them as reciprocal multiplies on Blackwell).  On sm_89
# both spellings are the same div.rn, so the copy is still the frozen base.
_BASE_PREFIX = r"""
// :4748-4855 (fallout1d), and :4859-5118 (Method I+II correction).
// Default option 18 uses fixed Atlas rain velocities, infall=irfall=4,
// adaptive first-order upwind substeps, and number in volumetric #/m3 while
// inside the scheme.  Each CUDA thread owns one complete vertical column.
#define NSSL2_KMAX_SHALLOW 64
#define NSSL2_KMAX_GENERIC 256

// WRF v4.6.1 NSSL option-18 driver support.
//
// The Registry arrays are gathered once into a 16-field internal slab. Number
// and volume moments remain in concentration space across every sedimentation
// category, then one final kernel scatters them to Registry mixing ratios.
// Numerical authority: module_mp_nssl_2mom.F:2650-3059, :4242-5118,
// :5168-5513 (calcnfromq), and :5546-5739 (calcnfromcuten).
enum Nssl2DriverField {
    NSSL2_QV = 0,
    NSSL2_QC = 1,
    NSSL2_QR = 2,
    NSSL2_QI = 3,
    NSSL2_QS = 4,
    NSSL2_QG = 5,
    NSSL2_QH = 6,
    NSSL2_NC = 7,
    NSSL2_NR = 8,
    NSSL2_NI = 9,
    NSSL2_NS = 10,
    NSSL2_NG = 11,
    NSSL2_NH = 12,
    NSSL2_NN = 13,
    NSSL2_VG = 14,
    NSSL2_VH = 15,
    NSSL2_DRIVER_FIELD_COUNT = 16,
};

__device__ __forceinline__ float* nssl2_driver_field(
    float* state, int field, int n)
{
    return state + (size_t)field * (size_t)n;
}

extern "C" __global__ void nssl2_driver_gather_initialize(
    const float* __restrict__ air_density,
    const float* __restrict__ qv,
    const float* __restrict__ qc,
    const float* __restrict__ qr,
    const float* __restrict__ qi,
    const float* __restrict__ qs,
    const float* __restrict__ qg,
    const float* __restrict__ qh,
    const float* __restrict__ qndrop,
    const float* __restrict__ qnr,
    const float* __restrict__ qni,
    const float* __restrict__ qns,
    const float* __restrict__ qng,
    const float* __restrict__ qnh,
    const float* __restrict__ qnn,
    const float* __restrict__ qvolg,
    const float* __restrict__ qvolh,
    const float* __restrict__ qrcuten,
    const float* __restrict__ qscuten,
    const float* __restrict__ qicuten,
    const float* __restrict__ qccuten,
    float* __restrict__ state,
    float dt,
    int first_step,
    int cu_used,
    int n,
    int predicted_ccn)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= n) return;

    const float rho = air_density[idx];
    const float cxmin = 1.0e-8f;
    const float qxmin_init = 1.0e-8f;
    const float qxmin_cloud = 1.0e-13f;
    const float qxmin_rain = 1.0e-12f;

    float vapor = qv[idx];
    float cloud = qc[idx];
    float rain = qr[idx];
    float ice = qi[idx];
    float snow = qs[idx];
    float graupel = qg[idx];
    float hail = qh[idx];

    // Exact driver denscale loop: all number and volume moments enter the
    // internal pipeline in concentration space once, before any processing.
    float cloud_number = qndrop[idx] * rho;
    float rain_number = qnr[idx] * rho;
    float ice_number = qni[idx] * rho;
    float snow_number = qns[idx] * rho;
    float graupel_number = qng[idx] * rho;
    float hail_number = qnh[idx] * rho;
    // module_mp_nssl_2mom.F:2719-2739 is the driver's CCN load.  With
    // predicted CCN (nssl_ccn_on=1 -> Registry package nssl_ccn_opt, so
    // f_qnn and therefore flag_ccn are true at :2464-2466) the prognostic
    // qnn field is loaded at :2727.  With nssl_ccn_on=0 the field does not
    // exist and WRF instead diagnoses the unactivated CCN every step from
    // the constant base concentration at :2734, taking the lccna==0 branch
    // because turn_on_ccna is gated on ccn_on==1 (:1403).  The subtraction
    // happens in per-mass units and only then is the whole slot scaled by
    // density at :2935-2944, so the product is formed after the difference.
    float ccn_number = predicted_ccn
        ? qnn[idx] * rho
        : (408163264.0f - qndrop[idx]) * rho;
    float graupel_volume = qvolg[idx] * rho;
    float hail_volume = qvolh[idx] * rho;

    if (first_step != 0) {
        if (cloud_number <= cxmin && cloud > qxmin_init) {
            const float qccn = 408163264.0f;
            const float cwmas_inverse = 327479132160.0f;
            cloud_number = fminf(qccn, cloud * cwmas_inverse) * rho;
            ccn_number -= cloud_number;
        } else if (cloud <= qxmin_cloud
                   || (cloud_number <= cxmin && cloud <= qxmin_init)) {
            vapor += cloud;
            cloud_number = 0.0f;
            cloud = 0.0f;
        }

        if (ice_number <= cxmin && ice > qxmin_init) {
            const float xims = 4.7123910329460728e-10f;
            ice_number = __fdiv_rn(rho * ice, xims);
        } else if (ice <= qxmin_cloud
                   || (ice_number <= cxmin && ice <= qxmin_init)) {
            vapor += ice;
            ice_number = 0.0f;
            ice = 0.0f;
        }

        if (rain_number <= 0.1f * cxmin && rain > qxmin_init) {
            const float zrfac = 3.9788734806922577e-11f;
            const double lambda_inverse = pow(
                (double)rho * (double)rain * (double)zrfac, 0.25);
            rain_number = (float)(
                lambda_inverse * (double)8000000.0f
                * (double)20.0f / (double)20.0f);
        } else if (rain <= qxmin_rain
                   || (rain_number <= cxmin && rain <= qxmin_init)) {
            vapor += rain;
            rain_number = 0.0f;
            rain = 0.0f;
        }

        if (snow_number <= 0.1f * cxmin && snow > qxmin_init) {
            const float zsfac = 1.0610329281846020e-9f;
            const double lambda_inverse = pow(
                (double)rho * (double)snow * (double)zsfac, 0.25);
            snow_number = (float)(
                lambda_inverse * (double)3000000.0f
                * (double)6.0000004768371582f / (double)20.0f);
        } else if (snow <= qxmin_cloud
                   || (snow_number <= cxmin && snow <= qxmin_init)) {
            vapor += snow;
            snow_number = 0.0f;
            snow = 0.0f;
        }

        if (graupel_number <= 0.1f * cxmin && graupel > qxmin_init) {
            if (graupel_volume <= 0.0f) {
                // Historical WRF quirk: this assignment follows denscale and
                // therefore is intentionally not multiplied by air density.
                graupel_volume = __fdiv_rn(graupel, 700.0f);
            }
            const float zhfac = 2.2736419413860176e-9f;
            const float xgms = 9.8960235561662557e-9f;
            const double lambda_inverse = pow(
                (double)rho * (double)graupel * (double)zhfac, 0.25);
            const double intercept_number =
                lambda_inverse * (double)200000.0f;
            const double maximum_number =
                (double)rho * (double)graupel / (double)xgms;
            const double diagnosed = fmin(intercept_number, maximum_number);
            if (diagnosed > (double)cxmin) {
                graupel_number = (float)diagnosed;
            } else {
                graupel = 0.0f;
                graupel_number = 0.0f;
                graupel_volume = 0.0f;
            }
        } else if (graupel <= qxmin_rain
                   || (graupel_number <= cxmin
                       && graupel <= qxmin_init)) {
            vapor += graupel;
            graupel = 0.0f;
        }

        if (hail_number <= 0.1f * cxmin && hail > qxmin_init) {
            if (hail_volume <= 0.0f) {
                hail_volume = __fdiv_rn(hail, 900.0f);
            }
            const float zhlfac = 8.8419414012719244e-9f;
            const double lambda_inverse = pow(
                (double)rho * (double)hail * (double)zhlfac, 0.25);
            hail_number = (float)(
                lambda_inverse * (double)40000.0f
                * (double)8.75f / (double)20.0f);
        } else if (hail <= qxmin_rain
                   || (hail_number <= cxmin && hail <= qxmin_init)) {
            vapor += hail;
            hail = 0.0f;
        }
    }

    if (cu_used != 0) {
        // calcnfromcuten diagnoses number only. Graupel/hail branches are
        // commented out in WRF 4.6.1, while all four live KF rates are used.
        const float cloud_increment = dt * qccuten[idx];
        const float ice_increment = dt * qicuten[idx];
        const float rain_increment = dt * qrcuten[idx];
        const float snow_increment = dt * qscuten[idx];
        // WRF 4.6.1 loads qccuten/qicuten into the cloud/ice *mass* slots,
        // but calcnfromcuten gates those branches on the corresponding empty
        // ancuten number slots. Thus both rates are consumed yet diagnose no
        // number increment. Preserve that exact official-source behavior.
        (void)cloud_increment;
        (void)ice_increment;
        if (rain_increment > qxmin_rain) {
            const float zrfac = 3.9788734806922577e-11f;
            const double lambda_inverse = pow(
                (double)rho * (double)rain_increment * (double)zrfac,
                0.25);
            rain_number += (float)(
                lambda_inverse * (double)8000000.0f
                * (double)20.0f / (double)20.0f);
        }
        if (snow_increment > qxmin_cloud) {
            const float zsfac = 1.0610329281846020e-9f;
            const double lambda_inverse = pow(
                (double)rho * (double)snow_increment * (double)zsfac,
                0.25);
            snow_number += (float)(
                lambda_inverse * (double)3000000.0f
                * (double)6.0000004768371582f / (double)20.0f);
        }
    }

    nssl2_driver_field(state, NSSL2_QV, n)[idx] = vapor;
    nssl2_driver_field(state, NSSL2_QC, n)[idx] = cloud;
    nssl2_driver_field(state, NSSL2_QR, n)[idx] = rain;
    nssl2_driver_field(state, NSSL2_QI, n)[idx] = ice;
    nssl2_driver_field(state, NSSL2_QS, n)[idx] = snow;
    nssl2_driver_field(state, NSSL2_QG, n)[idx] = graupel;
    nssl2_driver_field(state, NSSL2_QH, n)[idx] = hail;
    nssl2_driver_field(state, NSSL2_NC, n)[idx] = cloud_number;
    nssl2_driver_field(state, NSSL2_NR, n)[idx] = rain_number;
    nssl2_driver_field(state, NSSL2_NI, n)[idx] = ice_number;
    nssl2_driver_field(state, NSSL2_NS, n)[idx] = snow_number;
    nssl2_driver_field(state, NSSL2_NG, n)[idx] = graupel_number;
    nssl2_driver_field(state, NSSL2_NH, n)[idx] = hail_number;
    nssl2_driver_field(state, NSSL2_NN, n)[idx] = ccn_number;
    nssl2_driver_field(state, NSSL2_VG, n)[idx] = graupel_volume;
    nssl2_driver_field(state, NSSL2_VH, n)[idx] = hail_volume;
}

// WRF v4.6.1 module_mp_nssl_2mom.F:4242-4734 (sediment1d) and
// :6211-6498/:7333-7340 (default two-moment cloud-droplet velocity).
// Cloud mass and number use the same Stokes velocity.  Number remains in
// concentration space (#/m3), while cloud mass is a dry-air mixing ratio.

"""
_BASE_GATHER = r'''
def gather_initialize_and_sediment(
        air_density, dz,
        qv, qc, qr, qi, qs, qg, qh,
        qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
        dt_s: float, *, temperature_k,
        first_step: bool = False, cu_used: bool = False,
        qrcuten=None, qscuten=None, qicuten=None, qccuten=None,
        workspace: NSSL2DriverWorkspace | None = None,
        predicted_ccn: bool = True,
        ) -> NSSL2DriverWorkspace:
    """Gather once, initialize moments, diagnose KF numbers, and sediment.

    Volume fields use contiguous ``(nz, ny, nx)`` FP32 arrays.  Mass fields are
    kg/kg; number fields are #/kg; graupel/hail volume fields are m3/kg of dry
    air; density is kg/m3; ``temperature_k`` is absolute temperature; and
    ``dz`` is metres. Inputs are never mutated. The returned workspace keeps
    all moments in internal concentration units so GS and subsequent phases can
    run before the single final scatter.

    ``first_step`` selects WRF's exact ``itimestep == 1`` ``calcnfromq`` path.
    When ``cu_used`` is true, each supplied KF ``q*cuten`` rate is converted to
    a step mass increment and passed through exact ``calcnfromcuten`` number
    diagnosis.  Missing KF arrays are zero rates.  As in WRF, these rates do not
    add mass here: dynamics has already applied their mass tendencies.

    ``predicted_ccn`` is WRF's ``nssl_ccn_on``.  ``True`` is the resolved
    option-18 default and loads the prognostic ``qnn`` field.  ``False`` is
    the ``nssl_ccn_on=0`` variant (deprecated ``mp_physics=17``/``22``),
    where the Registry never allocates ``qnn`` and the module diagnoses the
    unactivated CCN from the base concentration every step
    (``module_mp_nssl_2mom.F:2734``).  ``qnn`` is then neither read nor
    written by the scheme.
    """
    if not isinstance(first_step, bool):
        raise TypeError("first_step must be bool")
    if not isinstance(cu_used, bool):
        raise TypeError("cu_used must be bool")
    if not isinstance(predicted_ccn, bool):
        raise TypeError("predicted_ccn must be bool")

    volume_fields = {
        "air_density": air_density,
        "temperature_k": temperature_k,
        "dz": dz,
        "qv": qv,
        "qc": qc,
        "qr": qr,
        "qi": qi,
        "qs": qs,
        "qg": qg,
        "qh": qh,
        "qndrop": qndrop,
        "qnr": qnr,
        "qni": qni,
        "qns": qns,
        "qng": qng,
        "qnh": qnh,
        "qnn": qnn,
        "qvolg": qvolg,
        "qvolh": qvolh,
    }
    rates = {
        "qrcuten": qrcuten,
        "qscuten": qscuten,
        "qicuten": qicuten,
        "qccuten": qccuten,
    }
    volume_fields.update({
        name: value for name, value in rates.items() if value is not None
    })
    (nz, ny, nx), size = _validate_volume_fields(volume_fields)
    step = _step32(dt_s)

    if workspace is not None:
        validate_nssl2_driver_workspace(workspace, (nz, ny, nx))
        if cu_used and any(value is None for value in rates.values()):
            raise ValueError(
                "a reusable NSSL workspace with cu_used=True requires all "
                "four KF rate arrays")

    # CuPy is imported lazily so CPU-only contract and lint tests remain usable.
    import cupy as cp

    state = (cp.empty((_FIELD_COUNT, nz, ny, nx), dtype=DTYPE)
             if workspace is None else workspace.state)
    zero_rate = None
    rate_args = []
    for value in rates.values():
        if value is None:
            if not cu_used:
                # The CUDA gather branch does not dereference KF pointers
                # when CU is disabled. Reuse a valid volume pointer instead
                # of allocating a full-volume zero field.
                rate_args.append(qv)
            else:
                if zero_rate is None:
                    zero_rate = cp.zeros_like(qv)
                rate_args.append(zero_rate)
        else:
            rate_args.append(value)

    element_blocks = (size + _ELEMENT_TPB - 1) // _ELEMENT_TPB
    get_kernel("nssl2_driver_support", "nssl2_driver_gather_initialize")(
        (element_blocks,), (_ELEMENT_TPB,),
        (air_density, qv, qc, qr, qi, qs, qg, qh,
         qndrop, qnr, qni, qns, qng, qnh, qnn, qvolg, qvolh,
         *rate_args, state, step, np.int32(first_step), np.int32(cu_used),
         np.int32(size), np.int32(1 if predicted_ccn else 0)))

    ncol = ny * nx
    column_blocks = (ncol + _COLUMN_TPB - 1) // _COLUMN_TPB
    if workspace is None:
        category_export = cp.empty((5, ny, nx), dtype=DTYPE)
        ignored_accumulator = cp.zeros((ny, nx), dtype=DTYPE)
    else:
        category_export = workspace.category_surface_export
        ignored_accumulator = workspace.ignored_accumulator
        # The five precipitating/ice kernels read-modify-write this temporary
        # WRF accumulator. Cloud fallout overwrites it with its own separate
        # export receipt after those launches.
        ignored_accumulator[...] = DTYPE(0.0)
    suffix = "64" if nz <= _SHALLOW_KMAX else "256"

    sediment_calls = (
        (f"nssl2_rain_sediment_{suffix}", _QR, _NR, None, 0),
        (f"nssl2_ice_sediment_{suffix}", _QI, _NI, None, 1),
        (f"nssl2_snow_sediment_{suffix}", _QS, _NS, None, 2),
        (f"nssl2_graupel_sediment_{suffix}", _QG, _NG, _VG, 3),
        (f"nssl2_hail_sediment_{suffix}", _QH, _NH, _VH, 4),
    )
    for kernel_name, mass_index, number_index, volume_index, export_index in (
            sediment_calls):
        arguments = [air_density, state[mass_index], state[number_index]]
        if volume_index is not None:
            arguments.append(state[volume_index])
        arguments.extend((
            dz, ignored_accumulator, category_export[export_index], step,
            np.int32(nz), np.int32(ny), np.int32(nx),
        ))
        get_kernel("nssl2_driver_support", kernel_name)(
            (column_blocks,), (_COLUMN_TPB,), tuple(arguments))

    # Cloud is disjoint from the other category states, so executing it after
    # the five standard exports is numerically identical to WRF's cloud-first
    # loop. Its bottom export is diagnosed but intentionally not reduced into
    # RAINNC, matching the official driver.
    get_kernel(
        "nssl2_driver_support", f"nssl2_cloud_sediment_{suffix}",
    )((column_blocks,), (_COLUMN_TPB,), (
        air_density, temperature_k, state[_QC], state[_NC], dz,
        ignored_accumulator, step,
        np.int32(nz), np.int32(ny), np.int32(nx),
    ))

    return NSSL2DriverWorkspace(
        state=state, category_surface_export=category_export,
        shape=(nz, ny, nx), ignored_accumulator=ignored_accumulator)



'''

@pytest.mark.parametrize("nz", [49, 65])
@pytest.mark.parametrize("warm", [False, True])
def test_graph_capture_matches_eager_base(nz, warm, monkeypatch):
    cp = pytest.importorskip("cupy")
    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            pytest.skip("CUDA device unavailable")
    except cp.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA device unavailable")
    from gpuwm.core import nssl2_driver_support as driver
    from gpuwm.core.kernels import _preamble, load_module

    references = _fixture("test_nssl2_speed_fused_sediment")
    base_module = cp.RawModule(
        code=_preamble() + _BASE_PREFIX + references._BASE_RAIN
        + references._BASE_SMALL + references._BASE_DENSE,
        options=("-std=c++17",))
    # Preflight may compile modules before capture. No eager driver call or
    # gamma table initialization is needed to capture fresh workspace buffers.
    load_module("nssl2_driver_support").compile()
    base_namespace = vars(driver).copy()
    base_namespace["get_kernel"] = lambda module, name: base_module.get_function(name)
    exec(_BASE_GATHER, base_namespace)
    base_gather = base_namespace["gather_initialize_and_sediment"]

    rng = np.random.default_rng(1810 + nz)
    shape = (nz, 1, 7)
    rho_host = rng.uniform(0.3, 1.2, shape).astype(np.float32)
    rho = cp.asarray(rho_host)
    dz = cp.asarray(rng.uniform(60, 400, shape).astype(np.float32))
    temperature = cp.asarray(rng.uniform(235, 285, shape).astype(np.float32))
    fields_host = []
    for index in range(16):
        if index < 7:
            values = rng.uniform(0, 0.003, shape).astype(np.float32)
            values[:, :, 0] = 0
        elif index < 14:
            values = rng.uniform(1, 2e4, shape).astype(np.float32)
        else:
            values = fields_host[5 if index == 14 else 6] / np.float32(500)
        fields_host.append(values)
    fields = [cp.asarray(value) for value in fields_host]
    rates = [cp.asarray(rng.uniform(0, 1e-6, shape).astype(np.float32)) for _ in range(4)]
    kwargs = dict(temperature_k=temperature, first_step=True, cu_used=True,
                  qrcuten=rates[0], qscuten=rates[1], qicuten=rates[2], qccuten=rates[3])
    def workspace():
        return driver.NSSL2DriverWorkspace(
            cp.empty((16, *shape), dtype=cp.float32),
            cp.empty((5, 1, 7), dtype=cp.float32), shape,
            cp.empty((1, 7), dtype=cp.float32))
    expected = workspace()
    actual = workspace()
    inputs = [rho, dz, temperature, *fields, *rates]
    inputs_before = [value.get().view(np.uint32).copy() for value in inputs]
    base_gather(rho, dz, *fields, 15.0, workspace=expected, **kwargs)
    driver._cached_velocity_gamma_table.cache_clear()
    if warm:
        with cp.cuda.Stream(non_blocking=True):
            driver._velocity_gamma_table(cp.cuda.runtime.getDevice())
    cp.cuda.runtime.deviceSynchronize()
    cached_before = driver._cached_velocity_gamma_table.cache_info()
    captured_names = []
    original_get_kernel = driver.get_kernel
    def get_kernel(module, name):
        if cp.cuda.get_current_stream().is_capturing():
            captured_names.append(name)
        return original_get_kernel(module, name)
    monkeypatch.setattr(driver, "get_kernel", get_kernel)
    cached_function = driver._cached_velocity_gamma_table
    def guarded_cache(device_id):
        assert not cp.cuda.get_current_stream().is_capturing()
        return cached_function(device_id)
    monkeypatch.setattr(driver, "_cached_velocity_gamma_table", guarded_cache)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        stream.begin_capture()
        try:
            driver.gather_initialize_and_sediment(
                rho, dz, *fields, 15.0, workspace=actual, **kwargs)
        finally:
            graph = stream.end_capture()
    assert not any("cached" in name or "parallel" in name or "gamma" in name
                   for name in captured_names)
    assert len(captured_names) == 7
    assert cached_function.cache_info() == cached_before
    for _ in range(2):
        with stream:
            for buffer in (actual.state, actual.category_surface_export, actual.ignored_accumulator):
                buffer.fill(np.float32(np.nan))
            graph.launch(stream=stream)
        stream.synchronize()
        for reference, observed in zip(
                (expected.state, expected.category_surface_export, expected.ignored_accumulator),
                (actual.state, actual.category_surface_export, actual.ignored_accumulator)):
            np.testing.assert_array_equal(reference.get().view(np.uint32), observed.get().view(np.uint32))
    eager = workspace()
    driver.gather_initialize_and_sediment(rho, dz, *fields, 15.0, workspace=eager, **kwargs)
    for reference, observed in zip(
            (expected.state, expected.category_surface_export, expected.ignored_accumulator),
            (eager.state, eager.category_surface_export, eager.ignored_accumulator)):
        np.testing.assert_array_equal(reference.get().view(np.uint32), observed.get().view(np.uint32))
    for before, after in zip(inputs_before, inputs):
        np.testing.assert_array_equal(before, after.get().view(np.uint32))
    cached_function.cache_clear()

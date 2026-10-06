"""Bitwise native Noah startup parity against the unchanged host authority."""
import ctypes
import math
import numpy as np
import pytest

from gpuwm.core.noah import NoahParams, SOIL_COLS, load_tables, pack_params, noah_frh2o, sh2o_init
from gpuwm import noah_init_bridge as bridge

def _frh2o_reference(tkelv: float, smc: float, sh2o: float, smcmax: float,
                bexp: float, psis: float) -> float:
    """WRF Noah ``FRH2O`` supercooled-liquid-water solve in float64.

    This is the setup-time CPU twin of ``noah_frh2o`` in ``noah.cu`` and
    follows ``module_sf_noahlsm.F:1447-1585``: the CK=8 log-form Newton
    iteration is bounded to ten iterations, with the CK=0 explicit fallback.
    """
    ck, blim, error = 8.0, 5.5, 0.005
    hlice, gs, t0 = 3.335e5, 9.81, 273.15
    bx = bexp if bexp <= blim else blim
    nlog = 0
    kcount = 0
    if tkelv > (t0 - 1.0e-3):
        return smc
    swl = smc - sh2o
    if swl > (smc - 0.02):
        swl = smc - 0.02
    if swl < 0.0:
        swl = 0.0
    while (nlog < 10) and (kcount == 0):
        nlog += 1
        df = (math.log((psis * gs / hlice) * ((1.0 + ck * swl) ** 2.0)
                       * (smcmax / (smc - swl)) ** bx)
              - math.log(-(tkelv - t0) / tkelv))
        denom = 2.0 * ck / (1.0 + ck * swl) + bx / (smc - swl)
        swlk = swl - df / denom
        if swlk > (smc - 0.02):
            swlk = smc - 0.02
        if swlk < 0.0:
            swlk = 0.0
        dswl = abs(swlk - swl)
        swl = swlk
        if dswl <= error:
            kcount += 1
    free = smc - swl
    if kcount == 0:
        fk = (((hlice / (gs * (-psis))) * ((tkelv - t0) / tkelv))
              ** (-1.0 / bx)) * smcmax
        if fk < 0.02:
            fk = 0.02
        free = min(fk, smc)
    return free

def _sh2o_reference(smois, tslb, isltyp, params: NoahParams) -> np.ndarray:
    """SH2O from SMOIS/TSLB exactly as LSMINIT (module_sf_noahdrv.F):
    Explicit first guess, then the FRH2O iteration."""
    smois = np.asarray(smois, np.float64)
    tslb = np.asarray(tslb, np.float64)
    if smois.shape != tslb.shape or smois.ndim == 0:
        raise ValueError("smois and tslb must be same-shape soil profiles")
    column_shape = smois.shape[1:]
    soil_type = np.asarray(isltyp)
    try:
        soil_type = np.broadcast_to(soil_type, column_shape)
    except ValueError as exc:
        raise ValueError("isltyp must match the soil-profile columns") from exc
    if (not np.isfinite(soil_type).all()
            or np.any(soil_type != np.floor(soil_type))):
        raise ValueError("isltyp must contain finite integer categories")

    out = smois.copy()
    blim, hlice, grav, t0 = 5.5, 3.335e5, 9.81, 273.15
    # LSMINIT compares a stored FP32 soil temperature against this FP32
    # literal. Evaluating the literal as FP64 would send its own FP32
    # boundary word through the cold solve instead of the warm copy
    # (module_sf_noahdrv.F:1931,1955).
    cold_threshold = float(np.float32(273.149))
    for column in np.ndindex(column_shape):
        category = int(soil_type[column])
        if category < 1 or category > params.slcats:
            raise ValueError(f"isltyp category {category} is outside table")
        row = params.soil[category - 1]
        bx = row[SOIL_COLS.index("bexp")]
        smcmax = row[SOIL_COLS.index("smcmax")]
        psisat = row[SOIL_COLS.index("psisat")]
        if not (bx > 0.0 and smcmax > 0.0 and psisat > 0.0):
            continue
        bx = min(bx, blim)
        for k in range(smois.shape[0]):
            index = (k, *column)
            if tslb[index] >= cold_threshold:
                continue
            fk = (((hlice / (grav * (-psisat)))
                   * ((tslb[index] - t0) / tslb[index]))
                  ** (-1.0 / bx)) * smcmax
            if fk < 0.02:
                fk = 0.02
            guess = min(fk, smois[index])
            out[index] = _frh2o_reference(
                tslb[index], smois[index], guess, smcmax, bx, psisat)
    return out


@pytest.fixture(scope="module")
def params():
    bridge.load()
    return pack_params(load_tables())


def _assert_bits(left, right):
    assert np.asarray(left, np.float64).tobytes() == np.asarray(right, np.float64).tobytes()


def test_all_table_categories_and_worker_counts_keep_reference_bits(params):
    random = np.random.default_rng(80231)
    categories = np.broadcast_to(np.arange(1, params.slcats + 1)[:, None], (params.slcats, 73))
    moisture = random.uniform(0.025, 0.40, (4, *categories.shape))
    temperature = random.uniform(225.0, 299.0, moisture.shape)
    reference = _sh2o_reference(moisture, temperature, categories, params)
    table = params.soil[:, [SOIL_COLS.index(name) for name in ("bexp", "smcmax", "psisat")]]
    for workers in (1, 2):
        actual = bridge.initialize(moisture, temperature, categories, table, workers=workers)
        _assert_bits(actual, reference)
    _assert_bits(sh2o_init(moisture, temperature, categories, params), reference)


def test_real_retained_coastal_value_pins(params):
    # Retained FP32 soil temperatures/category pairs and f64 output pins
    # from tests/test_noah.py's archived initialization regression.
    temperatures = np.asarray([[272.3423, 271.7954, 270.2837, 268.39307, 273.09937]], np.float32)
    categories = np.asarray([[6, 6, 2, 3, 6]], np.float64)
    moisture = np.ones((4, 1, 5), np.float64)
    heat = np.broadcast_to(temperatures, moisture.shape).astype(np.float64)
    expected = np.asarray([[float.fromhex(value) for value in (
        "0x1.399bacbd85316p-2", "0x1.1f6e0c311fd4ap-2", "0x1.06c59ec6a23f4p-3",
        "0x1.666b1430be558p-3", "0x1.e6e4e2bad21b4p-2")]], np.float64)
    _assert_bits(_sh2o_reference(moisture, heat, categories, params)[0], expected)
    _assert_bits(sh2o_init(moisture, heat, categories, params)[0], expected)


def test_float32_cutoff_and_strided_inputs(params):
    cutoff = float(np.float32(273.149))
    temperatures = np.asarray([
        np.nextafter(cutoff, -np.inf), cutoff, np.nextafter(cutoff, np.inf),
        273.149, np.nextafter(273.149, -np.inf), 273.15, 274.0,
    ], np.float64)
    moisture = np.full((4, 2, temperatures.size), 0.3, np.float64)
    heat = np.broadcast_to(temperatures, moisture.shape)
    _assert_bits(sh2o_init(moisture[:, ::-1, ::-1], heat[:, ::-1, ::-1], 6, params),
                 _sh2o_reference(moisture[:, ::-1, ::-1], heat[:, ::-1, ::-1], 6, params))
    liquid = sh2o_init(moisture, heat, 6, params)
    _assert_bits(liquid[:, :, 1:], moisture[:, :, 1:])


@pytest.mark.parametrize("categories,message", [
    ([0.0, 6.0], "isltyp category 0 is outside table"),
    ([6.0, 20.0], "isltyp category 20 is outside table"),
    ([0.0, np.nan], "isltyp must contain finite integer categories"),
    ([6.5, 6.0], "isltyp must contain finite integer categories"),
])
def test_category_errors_keep_reference_precedence(params, categories, message):
    moisture = np.full((4, 1, 2), 0.3)
    heat = np.full(moisture.shape, 270.0)
    for operation in (sh2o_init, _sh2o_reference):
        with pytest.raises(ValueError) as error:
            operation(moisture, heat, np.asarray([categories]), params)
        assert str(error.value) == message


def test_empty_profiles_and_wholly_liquid_invalid_table_rows(params):
    _assert_bits(sh2o_init(np.empty((0, 2)), np.empty((0, 2)), 6, params), np.empty((0, 2)))
    _assert_bits(sh2o_init(np.empty((4, 0)), np.empty((4, 0)), np.empty((0,)), params), np.empty((4, 0)))
    moisture = np.asarray([[0.0, -0.0, 0.3]])
    heat = np.full(moisture.shape, 270.0)
    _assert_bits(sh2o_init(moisture, heat, 14, params), moisture)


def test_earlier_column_math_error_precedes_later_category_range(params):
    moisture = np.full((4, 1, 2), 0.3)
    heat = np.full(moisture.shape, 270.0)
    heat[:, 0, 0] = -1.0
    with np.errstate(all="ignore"):
        for operation in (sh2o_init, _sh2o_reference):
            with pytest.raises(ValueError) as error:
                operation(moisture, heat, np.asarray([[6.0, 20.0]]), params)
            assert "category" not in str(error.value)
            with pytest.raises(ValueError) as error:
                operation(moisture, heat, np.asarray([[20.0, 6.0]]), params)
            assert str(error.value) == "isltyp category 20 is outside table"


def test_scalar_frh2o_keeps_python_and_numpy_bits(params):
    random = np.random.default_rng(20779)
    for _ in range(350):
        row = params.soil[int(random.integers(0, 13))]
        args = (random.uniform(240.0, 274.0), random.uniform(0.03, 0.45),
                random.uniform(0.01, 0.4), row[SOIL_COLS.index("smcmax")],
                row[SOIL_COLS.index("bexp")], row[SOIL_COLS.index("psisat")])
        for converted in (tuple(float(value) for value in args), tuple(np.float64(value) for value in args)):
            _assert_bits(bridge.frh2o(*converted), _frh2o_reference(*converted))


def test_scalar_domain_error_and_python_zero_division():
    for operation in (bridge.frh2o, _frh2o_reference):
        with pytest.raises(ValueError):
            operation(270.0, 0.3, 0.2, 0.4, 4.0, -0.3)
        with pytest.raises(ZeroDivisionError):
            operation(0.0, 0.3, 0.2, 0.4, 4.0, 0.3)
        with pytest.raises(TypeError):
            operation(270.0, 0.3, 0.2, -0.4, 4.5, 0.3)


def test_native_scalar_iteration_and_fallback_branches(params):
    library = bridge.load()
    function = getattr(library, bridge.NOAH_FRH2O_ENTRY)
    random = np.random.default_rng(9907)
    iterations_seen = set()
    fallback_seen = False
    for _ in range(2500):
        moisture = float(10.0 ** random.uniform(-3, 3))
        args = (float(random.uniform(220, 273.149)), moisture,
                moisture * float(random.uniform(0, 1)), float(10.0 ** random.uniform(-3, 2)),
                float(10.0 ** random.uniform(-3, math.log10(5.5))),
                float(10.0 ** random.uniform(-7, 3)))
        result, iterations, fallback = ctypes.c_double(), ctypes.c_size_t(), ctypes.c_uint32()
        code = function(*args, 0, ctypes.byref(result), ctypes.byref(iterations), ctypes.byref(fallback))
        assert code == 0
        _assert_bits(result.value, _frh2o_reference(*args))
        iterations_seen.add(iterations.value)
        fallback_seen |= bool(fallback.value)
    assert {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} <= iterations_seen
    # A finite scalar stress control forces the bounded iteration to stop.
    # Its moisture is outside a physical soil column; it tests the fallback,
    # not scientific admissibility of an initialization field.
    args = tuple(float.fromhex(value) for value in (
        "0x1.fe6c2ef7cb6cap+7", "0x1.13065ccc8e8d1p+35", "0x1.19a951f88d2bap+34",
        "0x1.296073c835e4ep+22", "0x1.ee4fbb871ffc3p-17", "0x1.c2ac7428df2a3p-32"))
    result, iterations, fallback = ctypes.c_double(), ctypes.c_size_t(), ctypes.c_uint32()
    code = function(*args, 0, ctypes.byref(result), ctypes.byref(iterations), ctypes.byref(fallback))
    assert code == 0 and iterations.value == 10 and fallback.value == 1
    _assert_bits(result.value, _frh2o_reference(*args))


def test_extended_category_precision_and_special_values(params):
    moisture = np.full((4, 1, 2), 0.3)
    heat = np.full(moisture.shape, 270.0)
    extended = np.longdouble
    fraction = extended(6) + extended(2) ** extended(-60)
    for categories in (
        np.asarray([[6, 14]], dtype=extended),
        np.asarray([[fraction, 6]], dtype=extended),
        np.asarray([[np.inf, 6]], dtype=extended),
        np.asarray([[np.nan, 6]], dtype=extended),
        np.asarray([[0, 6]], dtype=extended),
        np.asarray([[extended("1e400"), 6]], dtype=extended),
    ):
        for array in (categories, categories.astype(categories.dtype.newbyteorder(">"))):
            try:
                expected = _sh2o_reference(moisture, heat, array, params)
            except ValueError as original:
                with pytest.raises(ValueError) as actual:
                    sh2o_init(moisture, heat, array, params)
                assert str(actual.value) == str(original)
            else:
                _assert_bits(sh2o_init(moisture, heat, array, params), expected)


def test_missing_abi_probe_names_native_cold_init_remedy(monkeypatch):
    monkeypatch.setattr(bridge.ctypes, "CDLL", lambda path: object())
    bridge._load.cache_clear()
    with pytest.raises(bridge.NoahInitUnavailable) as error:
        bridge._load("missing-probe-test")
    assert "gpuwm_preprocess_cpu_abi_version" in str(error.value)
    assert "gpuwm fetch-bridges" in str(error.value)


def test_mixed_scalar_kinds_keep_exception_order():
    args = (0.0, 0.3, 0.2, 0.4, np.float64(4.0), 0.3)
    for operation in (bridge.frh2o, _frh2o_reference):
        with pytest.raises(ZeroDivisionError):
            operation(*args)


def test_float32_and_mixed_scalar_arithmetic_keeps_bits():
    random = np.random.default_rng(64420)
    for _ in range(250):
        values = (random.uniform(235, 273.148), random.uniform(.03, .4),
            random.uniform(.01, .35), random.uniform(.25, .48),
            random.uniform(2, 11), random.uniform(.04, .7))
        for kinds in ((np.float32,) * 6,
                      (float, np.float32, np.float32, float, np.float64, np.float32),
                      (np.float32, float, np.float32, np.float64, float, np.float32)):
            args = tuple(kind(value) for kind, value in zip(kinds, values))
            _assert_bits(bridge.frh2o(*args), _frh2o_reference(*args))


def test_scalar_qualification_authority_preserves_extended_input_types():
    for dtype in (np.float16, np.longdouble):
        args = tuple(dtype(value) for value in (270.0, .3, .2, .4, 4.0, .3))
        with np.errstate(all="ignore"):
            try:
                expected = _frh2o_reference(*args)
            except (ValueError, TypeError, ZeroDivisionError, OverflowError) as original:
                with pytest.raises(type(original)) as actual:
                    noah_frh2o(*args)
                assert str(actual.value) == str(original)
            else:
                _assert_bits(noah_frh2o(*args), expected)


def test_ieee_quad_category_scan_reads_actual_raw_precision():
    function = getattr(bridge.load(), bridge.NOAH_CATEGORY_SCAN_ENTRY)
    six = (16385 << 112) | (1 << 111)
    controls = ((six, 0, 6.0), (six | 1, 40, None),
        (0, 0, 0.0), (1 << 127, 0, -0.0),
        (32767 << 112, 40, None), ((32767 << 112) | 1, 40, None))
    for endian in (0, 1):
        for word, expected_code, expected_value in controls:
            raw = np.frombuffer(word.to_bytes(16, "big" if endian else "little"), dtype=np.uint8)
            output = ctypes.c_double()
            error = ctypes.c_size_t()
            code = function(raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                1, 1, endian, ctypes.byref(output), ctypes.byref(error))
            assert code == expected_code
            if code == 0:
                _assert_bits(output.value, expected_value)

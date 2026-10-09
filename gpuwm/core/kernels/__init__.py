from __future__ import annotations
from functools import lru_cache
from gpuwm.core.device_cache import cuda_cache
from pathlib import Path
from types import MappingProxyType

from gpuwm.core.constants import CUDA_DEFINES

_KDIR = Path(__file__).parent

# Every read of a .cu/.cuh in this directory is UTF-8, named explicitly.
#
# These sources are the input to nvrtc AND to the pinned hash the kernel
# manifest records (certify/kernel_manifest.py::record_module digests exactly
# the string module_source returns).  Path.read_text() with no encoding
# decodes with the host's locale, which is cp1252 on a stock Windows box, and
# cp1252 does not fail on a UTF-8 em dash -- it silently turns the three bytes
# into three characters.  acoustic.cu, advection.cu and coriolis_map.cu all
# carry U+2014 in comments, so the same checkout produced two different
# source_sha256 values depending on the host, with nothing raising to say so.
# The Thompson translation units happen to be pure ASCII today, which is luck,
# not a property anyone is maintaining.
_ENCODING = "utf-8"

# Explicit, closed allow-list of modules that receive an extra device header
# prepended between the preamble and their own source.  There is no #include
# path under cupy.RawModule, so this is how the six aerosol-aware Thompson
# (mp_physics=28) translation units share one set of __device__ helpers.
#
# The mechanism is deliberately inert for everything else: a module absent
# from this dict contributes the empty string and therefore assembles a
# BYTE-IDENTICAL source to what it assembled before this hook existed.  That
# is what keeps gpuwm/core/kernels/thompson.cu's compiled source string -- and
# so its PTX, register allocation and FP contraction -- unchanged by
# construction rather than by measurement.  tests/test_kernel_loader_inert.py
# proves it for every .cu file in this directory.
#
# This table must stay a literal name -> filenames mapping.  Do not give it
# filesystem probing, globbing, or any implicit fallback.
_EXTRA_HEADERS: dict[str, tuple[str, ...]] = {
    "acoustic": ("glibc_trig_flt32.cuh",),
    "upper_wind_limiter": ("glibc_flt32.cuh",),
    # Reuse the scalar high-order helpers without moving the order-3 unit.
    "pd_vertical_sl": ("pd_advection.cu",),
    "sfire_coupling": ("glibc_trig_flt32.cuh",),
    "sfire_ideal": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh", "sfire_libm.cuh"),
    "sfire_ideal_atmos": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    # Reuse the existing FRH2O device function for cold-start soil water.
    # The forecast's noah module remains unlisted and byte-identical.
    "noah_init": ("noah.cu",),
    "horizontal": ("portable_libm64.cuh",),
    "portable_libm64_grade": ("portable_libm64.cuh",),
    "vert_interp": ("glibc_flt32.cuh",),
    "thompson_cold_start": ("glibc_flt32.cuh", "portable_libm64.cuh"),
    "chem_prep": ("glibc_flt32.cuh",),
    "chem_dust": ("glibc_flt32.cuh",),
    "chem_seasalt": ("glibc_flt32.cuh",),
    "chem_rrtmgp_aerosol": ("glibc_flt32.cuh",),
    "chem_sulfur": ("glibc_flt32.cuh",),
    "chem_ageing": ("glibc_flt32.cuh",),
    "chem_settling": ("glibc_flt32.cuh",),
    "chem_drydep_gocart": ("glibc_flt32.cuh",),
    "chem_optics": ("glibc_flt32.cuh",),
    # The mp=28 units evaluate WRF's EXP/LOG/LOG10/** through WOOF's own
    # float32 and binary64 libm words (thompson_aerosol_libm.cuh),
    # measured equal to the gfortran oracle host's words; CUDA's
    # builtins are different functions and left no column of the 0 ULP
    # column oracle bit-identical (tools/thompson_aerosol_column_oracle).
    "thompson_aerosol_probe": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_state": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_sat": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_cold": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_warm": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    "thompson_aerosol_sed": ("glibc_flt32.cuh", "glibc_flt64.cuh",
                               "thompson_aerosol_libm.cuh",
                               "thompson_aerosol_common.cuh"),
    # The LW solver derives the Planck sources itself instead of loading
    # what rrtmgp_planck_sources wrote; the helpers it needs live in a
    # header whose every FP op is pinned (HOWTO 13.6j).  rrtmgp_gas is
    # deliberately NOT listed -- it keeps its own helpers verbatim so its
    # assembled source and PTX stay byte-identical.
    "rrtmgp_rte": ("rrtmgp_planck_common.cuh",),
    # glibc 2.39's own float32 expf/logf/powf words, which every kernel
    # graded bitwise against a gfortran oracle must call instead of CUDA's
    # builtins.  gf.cu owned the only copy until New Tiedtke (cu_physics=16)
    # needed the same three functions; the alternative was a THIRD
    # transcription beside gf's gfk_* and noahmp_leaves.cu's r_log/r_exp/
    # r_pow.  Listing gf here costs it the by-construction inertness the
    # unlisted modules keep, so it is bought with a measurement instead:
    # the lift leaves all seven gf entry points at byte-identical
    # local_size_bytes/num_regs/const_size_bytes, and the gf parity suites
    # still grade at max_ulp 0.  See glibc_flt32.cuh's header.
    # Noah mosaic uses scalar glibc float32 words for the WRF column oracle.
    "noah_mosaic": ("glibc_flt32.cuh",),
    # RUC: every EXP, LOG, LOG10, TANH and REAL**REAL of module_sf_ruclsm.F
    # takes WOOF's float32 words (gfk_* here, the log10/expm1/tanh routines in
    # ruc.cu) and the soil-resistance COS takes glibc_cosf, graded bitwise
    # against WRF v4.6.1 by tools/ruc_lsm_gpu_oracle.
    "ruc": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "lake": ("glibc_flt32.cuh", "lake_support.cuh", "lake_wrf.cuh"),
    "gf": ("glibc_flt32.cuh",),
    # WRF-Chem Wesely gas dry deposition: rc/depvel read exp, log and pow,
    # and glibc's float forms are what the gfortran column oracle ran.
    "chem_drydep_wesely": ("glibc_flt32.cuh",),
    # New Tiedtke: scale_fac reads log(dxref/dx), and glibc's logf is not
    # CUDA's.  Prep stage only so far; cumastrn will add exp and pow.
    "ntiedtke": ("glibc_flt32.cuh",),
    # The single-layer urban canopy model (sf_urban_physics=1): EXP, ALOG
    # and every REAL**REAL in module_sf_urban.F are glibc calls in its WRF
    # v4.7.1 column oracle, graded bitwise.
    "urban_ucm": ("glibc_flt32.cuh",),
    # Urban BEP (sf_urban_physics 2/3), graded bitwise against gfortran/glibc
    # WRF v4.7.1 column oracles: the column uses logf/powf and six trig
    # functions (glibc_trig_flt32.cuh), the surface coupling uses powf.
    "urban_bep": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    "urban_bep_couple": ("glibc_flt32.cuh",),
    # MYJ under BEP (module_bl_myjurb.F), graded against gfortran/glibc:
    # EXP and REAL powers are glibc's expf/powf.
    "myjurb": ("glibc_flt32.cuh",),
    # topo_wind arm (ysu_column_topo): glibc powf for the paj TKE profile
    # and the Beljaars convective velocity, as bl_ysu.F90 evaluates them,
    # and the arm's own pieces (get_pblh, the 10 m blend), kept out of
    # ysu.cu so its cited line numbers stand.
    "ysu": ("glibc_flt32.cuh", "ysu_topo.cuh"),
    # Shin-Hong takes glibc's powf/expf (gfk_pow/gfk_exp) since
    # lane/parity-pbl-libm, graded by tests/test_shinhong_wrf461_parity.py.
    "shinhong": ("glibc_flt32.cuh",),
    # The UW moist-turbulence PBL (bl_pbl_physics=9) computes in binary64
    # like the CAM code it transcribes: glibc's own binary64 exp/log/pow,
    # the rounding-pinned R8 vocabulary, then the CAM modules in call order
    # (saturation lookups, the implicit diffusion solver, exacol/zisocl/
    # compute_cubic, caleddy, compute_eddy_diff with trbintd/sfdiag and the
    # camuwpbl column driver).  A new module, so no existing unit moves.
    "uwpbl": ("glibc_flt64.cuh", "uwpbl_common.cuh", "uwpbl_wvsat.cuh",
              "uwpbl_vdiff.cuh", "uwpbl_zisocl.cuh", "uwpbl_caleddy.cuh",
              "uwpbl_eddy.cuh", "uwpbl_driver.cuh"),
    # WRF swint_opt = 1 (module_radiation_driver.F radconst/calc_coszen,
    # update_swinterp_parameters, interp_sw_radiation of the operational
    # HRRR fork): LOG and ** are glibc's logf/powf, SIN/COS/ASIN glibc's
    # sinf/cosf/asinf, graded bitwise against the fork's gfortran/glibc
    # Fortran by tests/test_swint_interpolation.py.  A new module, so no
    # existing unit moves.
    "swint": ("glibc_flt32.cuh", "glibc_trig_flt32.cuh"),
    # WRF aer_opt = 3 shortwave optics (gt_aod, calc_aerosol_rrtmg_sw of
    # the operational HRRR fork): EXP is glibc's expf, graded bitwise
    # against the fork's gfortran/glibc Fortran by
    # tests/test_rrtmg_aerosol_optics.py.  A new module.
    "rrtmg_aer3": ("glibc_flt32.cuh",),
    "real_init": ("real_init_common.cuh",),
    # REAL's float64 thermodynamics use the CPU portable library's bits.
    "real_init_math": ("real_init_common.cuh", "portable_libm64.cuh"),
    "chem_fire": ("glibc_flt32.cuh",),
    "chem_plumerise": ("glibc_flt32.cuh",),
    # The km_opt=2/3 path (tke_km, calc_l_scale, tke_dissip, calculate_N2 and
    # phy_prep's p8w/t8w) takes REAL**REAL, EXP and LOG as gfk_pow/gfk_exp/
    # gfk_log, graded bitwise by tools/tke_km2_wrf461_oracle against the WRF
    # v4.6.1 Fortran.  CUDA's powf/expf/logf flipped calculate_N2's saturated
    # predicate there and moved the TKE coefficients by up to 34,746 ULP.
    "smag2d": ("glibc_flt32.cuh",),
    "myjsfc": ("glibc_flt32.cuh", "flt32_expf_fma.cuh"),
    "mynn_surface": ("mynn_libm.cuh", "surface_subnormal.cuh"),
    "sfclay": ("glibc_flt32.cuh", "sfclay_classic.cuh"),
}

#: Read-only view for tests and freeze receipts.
EXTRA_HEADERS = MappingProxyType(_EXTRA_HEADERS)


def _preamble(kernel_dir: Path = _KDIR) -> str:
    lines = [f"#define {k} {float(v)!r}f" for k, v in CUDA_DEFINES.items()]
    lines.append((Path(kernel_dir) / "common.cuh").read_text(encoding=_ENCODING))
    return "\n".join(lines) + "\n"


def _extra_header_text(name: str, kernel_dir: Path = _KDIR) -> str:
    """Return the allow-listed headers for ``name``, or ``''`` for any other.

    The empty-string return for an unlisted module is the whole point: it
    makes the assembled source byte-identical to the pre-hook string.
    """
    headers = _EXTRA_HEADERS.get(name, ())
    from gpuwm.wrf_exact import ENABLED, DIAGNOSTICS_ENABLED, DIFFUSION_ENABLED
    if DIAGNOSTICS_ENABLED and name == "diagnostics":
        headers += ("glibc_flt32.cuh",)
    return "".join((Path(kernel_dir) / header).read_text(encoding=_ENCODING)
                   for header in headers)


def _unit_text(name: str, kernel_dir: Path = _KDIR) -> str:
    """Read a CUDA unit and apply only the active set's registered literals."""
    from gpuwm.physics_params import edit_kernel_source
    return edit_kernel_source(
        name, (Path(kernel_dir) / f"{name}.cu").read_text(encoding=_ENCODING))


def module_source(name: str, *, kernel_dir: Path = _KDIR) -> str:
    """The exact source string :func:`load_module` hands to nvrtc.

    ``kernel_dir`` composes the same unit from another tree's kernel files:
    tools/literal_division_census.py gates the tree it scans, which need not
    be the imported package (A193).
    """
    return (_preamble(kernel_dir) + _extra_header_text(name, kernel_dir)
            + _unit_text(name, kernel_dir))


#: Units graded word for word against a WRF column oracle, which needs NVRTC
#: not to contract multiply-adds (WRF's reference is gfortran -O0, no FMA).
#: lake: the CLM lake oracle.  ysu and shinhong: the WRF 4.6.1 YSU and
#: Shin-Hong oracles (tests/test_ysu_wrf461_parity.py,
#: tests/test_shinhong_wrf461_parity.py), together with glibc's powf/expf
#: (gfk_pow/gfk_exp) in place of CUDA's.  mynn_pbl and its DMP sibling: the
#: WRF 4.6.1 MYNN oracles (tests/test_mynn_wrf461_exact_gpu.py), through
#: both loaders (the gsd_41 generation compiles mynn_pbl with an integer
#: define, :func:`load_module_int_defines`).
_NO_FMAD_MODULES = frozenset({"lake", "ysu", "shinhong", "mynn_pbl",
                              "mynn_dmp_sibling"})


_NO_FTZ_MODULES = frozenset({"sfclay", "myjsfc", "mynn_surface", "mynn_pbl"})


def module_options(name: str) -> tuple[str, ...]:
    """Compile options shared by runtime, oracle and division census."""
    if name in _NO_FTZ_MODULES:
        return ("-std=c++17", "--fmad=false", "--ftz=false")
    if name in _DIFFUSION_MODULES:
        return DIFFUSION_OPTIONS
    return (("-std=c++17", "--fmad=false") if name in _NO_FMAD_MODULES
            else ("-std=c++17",))


_DIFFUSION_MODULES = frozenset(("smag2d", "diffusion", "diff_opt1", "diff6", "diff6_seam"))

DIFFUSION_OPTIONS = ("-std=c++17", "--fmad=false", "--ftz=false",
                     "--prec-div=true", "--prec-sqrt=true",
                     "-DGPUWM_WRF_EXACT_C_DIFFUSION=1")
# Only the coefficient and w solve use this route in the acoustic unit.
# Other acoustic symbols keep their existing compiled module.
_DIFFUSION_FUNCTIONS = {
    "acoustic": frozenset(("calc_coefs", "advance_w_phi", "advance_w_phi_msf")),
    "dycore": frozenset(("w_damp", "w_cfl_stat")),
}


def diffusion_kernel(name: str, function: str) -> bool:
    # The default acoustic arm failed the real-column exactness gate.
    # Retain its existing strict opt-in compiler and the qualified diffusion
    # modules; an ordinary acoustic launch keeps its published arithmetic.
    if name == "acoustic":
        from gpuwm.wrf_exact import ENABLED
        if not ENABLED:
            return False
    return name in _DIFFUSION_MODULES or function in _DIFFUSION_FUNCTIONS.get(name, ())


def function_options(name: str, function: str, options) -> tuple[str, ...]:
    """Keep scalar and batch diffusion on the same arithmetic route."""
    if not diffusion_kernel(name, function):
        return tuple(options)
    overridden = {"std", "fmad", "ftz", "prec-div", "prec-sqrt", "use_fast_math",
                  "DGPUWM_WRF_EXACT_C_DIFFUSION"}
    return tuple(o for o in options if o.lstrip("-").split("=", 1)[0] not in overridden) + DIFFUSION_OPTIONS


@cuda_cache(maxsize=None)
def compile_diffusion_source(source: str, key: str, options):
    """Load and record an IEEE diffusion image, including batch witnesses."""
    import cupy as cp
    from cupy.cuda import compiler
    from gpuwm.kernel_compile_notice import observe_module_compile
    from gpuwm.certify.kernel_manifest import record_module
    from types import SimpleNamespace
    from gpuwm.wrf_exact import ENABLED, effective_options
    if ENABLED:
        options = effective_options(options)
    with observe_module_compile(key):
        binary, _ = compiler.compile_using_nvrtc(source, options=options)
        module = cp.cuda.function.Module()
        module.load(binary.encode() if isinstance(binary, str) else binary)
    image = binary.encode() if isinstance(binary, str) else binary
    kind = "cubin" if image.startswith(b"\x7fELF") else "ptx"
    record_module(key, source=source, options=options,
                  module=SimpleNamespace(**{kind: image}))
    return module


@cuda_cache(maxsize=None)
def _load_diffusion_module(name: str, defines=()):
    """Compile with IEEE arithmetic at NVRTC's final boundary.

    RawModule appends FTZ after caller options. Direct NVRTC keeps the
    requested --ftz=false and records the actual image and option tuple.
    The ordinary and integer-tier loaders share this one compile site.
    """
    from gpuwm.wrf_exact import ENABLED, DIFFUSION_ENABLED, effective_options
    from gpuwm.physics_params import note_compiled
    source = module_source_int_defines(name, defines) if defines else module_source(name)
    options = DIFFUSION_OPTIONS
    if not DIFFUSION_ENABLED:
        options = tuple(o for o in options if "DGPUWM_WRF_EXACT_C_DIFFUSION" not in o)
    if ENABLED:
        options = effective_options(options)
    tier = ",".join(f"{key}={value}" for key, value in defines)
    key = f"{MODULE_KEY_ROOT}:{name}" + (f"[{tier}]" if tier else "")
    module = compile_diffusion_source(source, key, options)
    note_compiled(name)
    return module


def compile_noftz_module(name: str, src: str, key: str):
    """Compile the requested image without CuPy's appended FTZ option."""
    import cupy as cp
    from gpuwm import nvrtc_ptx_cache as compiler
    from gpuwm.kernel_compile_notice import observe_module_compile
    from gpuwm.certify.kernel_manifest import record_module
    from gpuwm.physics_params import note_compiled
    options = module_options(name)
    # Bind strict macros into the direct cache key as well as the compiler
    # request. Otherwise a default cached image could enter a strict run.
    from gpuwm import wrf_exact
    if wrf_exact.ENABLED:
        options = wrf_exact.effective_options(options)
    with observe_module_compile(key):
        image, _ = compiler.compile_using_nvrtc(src, options, None, name + ".cu")
        module = cp.cuda.function.Module()
        module.load(image.encode() if isinstance(image, str) else image)
    note_compiled(name)
    record_module(key, source=src, options=options, module=None)
    return module


def _load_module_without_fmad(name: str, src: str, options: tuple[str, ...]):
    """The CLM lake's own compile site: the loader's tuple plus --fmad=false.

    Its WRF column oracle is graded word for word, which needs NVRTC not to
    contract multiply-adds.  A separate site keeps load_module's call a
    literal tuple, so the FTZ route inventory still reads route R1's options.

    The site records what it compiled here, beside the compile, with the
    same literal tuple.  The breakage this prevents: the kernel manifest's
    audit pairs each compile with a record in the same function and
    compares their arguments as written; a record left in load_module named
    a variable, so the audit could no longer tell that the lake's manifest
    row states the options NVRTC was given.
    """
    import cupy as cp
    if options != ("-std=c++17", "--fmad=false"):
        raise ValueError(f"no kernel compile site takes options {options!r}")
    mod = cp.RawModule(code=src, options=("-std=c++17", "--fmad=false"),
                       name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    # The physics-parameter guard must see these units compile too:
    # mynn_pbl and mynn_dmp_sibling carry registered literals
    # (gpuwm/physics_params_registry_v1.json), and a set declared after they
    # compiled would otherwise run under the set's name with the default
    # constants, the breakage physics_params.declare refuses.
    from gpuwm.physics_params import note_compiled
    note_compiled(name)
    from gpuwm.certify.kernel_manifest import record_module
    record_module(f"{MODULE_KEY_ROOT}:{name}", source=src,
                  options=("-std=c++17", "--fmad=false"), module=mod)
    return mod


@cuda_cache(maxsize=None)
def load_module(name: str):
    import cupy as cp
    if name in _DIFFUSION_MODULES:
        return _load_diffusion_module(name)
    if name.startswith("noahmp_"):
        from gpuwm.core.noahmp_kernel_sources import (
            NOAHMP_TRANSLATION_UNITS, compile_runtime_unit)
        # A standalone Noah-MP unit compiles through the one Noah-MP
        # RawModule site, so the source string a forecast hands NVRTC is
        # the one its frame recording was read from.  Fragments (which
        # fail alone, and must keep failing alone) and the generic C++17
        # VEGE_FLUX census stay on the plain route below: the runtime
        # VEGE_FLUX unit is C++14 without the preamble and has its own
        # factory.  tests/test_kernel_loader_inert.py asserts the two
        # routes assemble byte-identical source for every unit this
        # branch takes.
        if (name in NOAHMP_TRANSLATION_UNITS
                and len(NOAHMP_TRANSLATION_UNITS[name]) == 1
                and name != "noahmp_vegeflux"):
            return compile_runtime_unit(name, module_key=f"{MODULE_KEY_ROOT}:{name}")
    src = module_source(name)
    if name in _NO_FTZ_MODULES:
        return compile_noftz_module(name, src, f"{MODULE_KEY_ROOT}:{name}")
    options = module_options(name)
    if options != ("-std=c++17",):
        return _load_module_without_fmad(name, src, options)
    # The loader's literal tuple: tools/ftz_receipt reads it from this
    # call (route R1) and probes NVRTC's float behaviour under it.
    mod = cp.RawModule(code=src, options=("-std=c++17",), name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    from gpuwm.physics_params import note_compiled
    note_compiled(name)
    from gpuwm.certify.kernel_manifest import record_module
    record_module(f"{MODULE_KEY_ROOT}:{name}",
                  source=src, options=("-std=c++17",), module=mod)
    return mod


@cuda_cache(maxsize=None)
def load_module_int_defines(
        name: str, defines: tuple[tuple[str, int], ...]):
    """Compile one kernel source with a small, identity-bound integer tier.

    This keeps compile-time local-array bounds specialized without enlarging
    the common kernel.  Only uppercase C-preprocessor identifiers and positive
    integer values are accepted; callers cannot inject arbitrary source text.
    """
    import re
    import cupy as cp

    normalized = tuple((str(key), int(value)) for key, value in defines)
    if normalized != defines:
        raise TypeError("kernel integer defines must be canonical (str, int) pairs")
    for key, value in normalized:
        if re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None:
            raise ValueError(f"invalid CUDA preprocessor identifier {key!r}")
        if isinstance(value, bool) or value < 1:
            raise ValueError(
                f"CUDA integer define {key} must be a positive integer")
    prefix = "\n".join(f"#define {key} {value}" for key, value in normalized)
    src = module_source_int_defines(name, normalized, prefix=prefix)
    if name in _NO_FTZ_MODULES:
        tier = ",".join(f"{key}={value}" for key, value in normalized)
        return compile_noftz_module(name, src, f"{MODULE_KEY_ROOT}:{name}[{tier}]")
    if name in _DIFFUSION_MODULES:
        return _load_diffusion_module(name, normalized)
    if name in _NO_FMAD_MODULES:
        return _load_module_int_defines_without_fmad(name, src, normalized)
    mod = cp.RawModule(code=src, options=("-std=c++17",),
                       name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    from gpuwm.physics_params import note_compiled
    note_compiled(name)
    from gpuwm.certify.kernel_manifest import record_module
    tier = ",".join(f"{key}={value}" for key, value in normalized)
    record_module(f"{MODULE_KEY_ROOT}:{name}[{tier}]",
                  source=src, options=("-std=c++17",), module=mod)
    return mod


def _load_module_int_defines_without_fmad(name: str, src: str, normalized):
    """An integer-define tier of a no-FMA unit: the same literal tuple as
    :func:`_load_module_without_fmad`.

    The breakage this prevents: the gsd_41 MYNN generation compiles
    mynn_pbl through this loader; with the plain tuple NVRTC would contract
    its multiply-adds while the default build (load_module) does not, and
    the two builds of one source would round differently.  The record sits
    beside the compile with the same literal tuple, as the lake's does.
    """
    import cupy as cp
    mod = cp.RawModule(code=src, options=("-std=c++17", "--fmad=false"),
                       name_expressions=None)
    _compile_observed(mod, f"{MODULE_KEY_ROOT}:{name}")
    from gpuwm.physics_params import note_compiled
    note_compiled(name)
    from gpuwm.certify.kernel_manifest import record_module
    tier = ",".join(f"{key}={value}" for key, value in normalized)
    record_module(f"{MODULE_KEY_ROOT}:{name}[{tier}]",
                  source=src, options=("-std=c++17", "--fmad=false"),
                  module=mod)
    return mod


#: Manifest namespace for the translation units the loaders above compile.
#: Declared below them because the FTZ receipt pins their RawModule lines.
MODULE_KEY_ROOT = "gpuwm.core.kernels"


def module_source_int_defines(
        name: str, defines: tuple[tuple[str, int], ...],
        *, prefix: str | None = None, kernel_dir: Path = _KDIR) -> str:
    """The exact source :func:`load_module_int_defines` hands to nvrtc.

    The allow-listed header, when present, goes immediately after the
    preamble, exactly as in :func:`module_source`; for every module absent
    from ``_EXTRA_HEADERS`` the inserted text is empty and the string is
    byte-identical to the pre-hook assembly.
    """
    if prefix is None:
        prefix = "\n".join(f"#define {key} {value}" for key, value in defines)
    return (_preamble(kernel_dir) + _extra_header_text(name, kernel_dir)
            + prefix + "\n"
            + _unit_text(name, kernel_dir))


@cuda_cache(maxsize=None)
def get_kernel(name: str, func: str):
    """Return one stable CuPy function wrapper per raw-kernel symbol."""
    if diffusion_kernel(name, func) and func in _DIFFUSION_FUNCTIONS.get(name, ()):
        return _load_diffusion_module(name).get_function(func)
    return load_module(name).get_function(func)


@cuda_cache(maxsize=None)
def get_kernel_int_defines(
        name: str, func: str, defines: tuple[tuple[str, int], ...]):
    """Return a cached kernel compiled with validated integer definitions."""
    if diffusion_kernel(name, func) and func in _DIFFUSION_FUNCTIONS.get(name, ()):
        # Validate through the same public loader before selecting a tier.
        # The tier loader's validation remains the sole definition contract.
        if defines != tuple((str(key), int(value)) for key, value in defines):
            raise TypeError("kernel integer defines must be canonical (str, int) pairs")
        import re
        for key, value in defines:
            if re.fullmatch(r"[A-Z][A-Z0-9_]*", key) is None or isinstance(value, bool) or value < 1:
                raise ValueError(f"invalid CUDA integer define {key}={value!r}")
        return _load_diffusion_module(name, defines).get_function(func)
    return load_module_int_defines(name, defines).get_function(func)


def _compile_observed(module, module_key: str) -> None:
    """Compile ``module``, telling a watching run when it really compiled.

    :func:`gpuwm.kernel_compile_notice.observe_module_compile` costs nothing
    when no run is watching; when one is, a module that wrote to the kernel
    cache (a compile, not a cache load) is published as progress.
    """
    from gpuwm.kernel_compile_notice import observe_module_compile
    with observe_module_compile(module_key):
        module.compile()

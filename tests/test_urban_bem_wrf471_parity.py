"""BEP+BEM (``sf_urban_physics = 3``) against WRF v4.7.1's own ``BEP_BEM``.

``tools/urban_wrf471_oracle/run_bep_bem.F90`` drives the byte-unmodified
``phys/module_sf_bep_bem.F`` and ``phys/module_sf_bem.F`` (pinned by
``SOURCES.sha256``) exactly as ``module_sf_noahdrv.F:1636-1677`` does, after
``module_sf_urban.F``'s own ``urban_param_init`` read the table and its own
``urban_var_init`` built the option-3 initial state.  It dumps every
``BEP_BEM`` input, every in/out state array before the first and after the
last call, every output of every call, and every ``module_sf_urban`` table
array ``BEP_BEM`` reads.  The four fixtures under
``gpuwm/data/urban/oracle/bem/`` (see ``PROVENANCE.md`` there) are:

=========  ===================================================================
``stock``  ``URBPARM.TBL`` as WRF ships it, ``use_wudapt_lcz = 0``: classes
           1-3, the stock ``ISURBAN -> 2`` mapping, the table-fraction arm
           (``FRC_URB2D = 0`` handed in), FRC 0.01 / 0.5 / 0.99 / 1.0, day,
           night, dusk and the 24 h wrap of the local hour, wet and dry,
           stable and unstable, default and gridded (``HGT_URB2D > 0``)
           morphology, two rural columns BEP_BEM must not touch.
``lcz``    ``URBPARM_LCZ.TBL``, ``use_wudapt_lcz = 1``: all eleven LCZ rows
           (``ISURBAN -> 5``); LCZ 11 carries ``SW_COND = 0``, ``PWIN = 0``.
``gr1pv``  the stock table with the green roof on (``GR_FLAG 1``, ``GR_TYPE
           1``), roof photovoltaics, irrigation hours, air conditioning off
           outside ``TIME_ON..TIME_OFF`` and ``SW_COND = 0`` on class 2.
``gr2``    green roof ``GR_TYPE 2`` with irrigation every hour.
``long``   ``stock`` for 30 consecutive calls (half an hour at 60 s), so the
           prognostic wall, roof, floor, window and indoor layers are
           graded after they have evolved, not one step from their
           initial values.
=========  ===================================================================

The last two tables are test inputs derived from the pinned one by the
``sed`` lines recorded in PROVENANCE.md; WRF ships the arms they reach
switched off, and a switch no fixture reaches is a switch no gate checks.

The measured spread of WRF's own compilers on these fixtures is the
tolerance a port may claim: the same sources built at WRF's own
``-O2 -ftree-vectorize -funroll-loops`` (which links glibc's libmvec
``_ZGVbN4vv_powf``) differ from the ``-O0`` reference in exactly two fields,
``grdflx_urb`` (up to 80 ULP, on the 30th call of ``long``) and ``rl_up``
(up to 4 ULP), and are bit-identical in every other array of every step --
including every prognostic layer after 30 calls, so neither difference
feeds back into the state.  The port is asserted
against the ``-O0`` reference, field by field, at the measured table below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.core.fp32_ulp import fp32_ulp_distance

FIXTURE_DIR = (Path(__file__).resolve().parents[1]
               / "tests" / "data" / "oracles" / "urban" / "bem")
VARIANTS = ("stock", "lcz", "gr1pv", "gr2", "long")

#: BEP_BEM's per-call outputs on mass levels kts..kte (sf also kte+1).
LEVEL_OUTPUTS = ("a_u", "a_v", "a_t", "a_e", "b_u", "b_v", "b_t", "b_e",
                 "b_q", "dlg", "dl_u", "vl")
SURFACE_OUTPUTS = ("rl_up", "rs_abs", "emiss", "grdflx_urb")
BEM_2D_OUTPUTS = ("sf_ac_urb3d", "lf_ac_urb3d", "cm_ac_urb3d",
                  "sfvent_urb3d", "lfvent_urb3d", "ep_pv_urb3d",
                  "qgr_urb3d", "tgr_urb3d", "draingr_urb3d")
NDM_OUTPUTS = ("sfg_urb3d", "dg_urb3d", "lfg_urb3d")

#: Every BEP_BEM INTENT(INOUT) state array (module_sf_bep_bem.F:95-117).
STATE_ARRAYS = (
    "trb_urb4d", "tw1_urb4d", "tw2_urb4d", "tgb_urb4d",
    "tlev_urb3d", "qlev_urb3d", "tw1lev_urb3d", "tw2lev_urb3d",
    "tglev_urb3d", "tflev_urb3d", "sf_ac_urb3d", "lf_ac_urb3d",
    "cm_ac_urb3d", "sfvent_urb3d", "lfvent_urb3d",
    "sfwin1_urb3d", "sfwin2_urb3d",
    "sfw1_urb3d", "sfw2_urb3d", "sfr_urb3d", "sfg_urb3d",
    "ep_pv_urb3d", "t_pv_urb3d",
    "trv_urb4d", "qr_urb4d", "qgr_urb3d", "tgr_urb3d",
    "drain_urb4d", "draingr_urb3d", "sfrv_urb3d",
    "lfrv_urb3d", "dgr_urb3d", "dg_urb3d", "lfr_urb3d", "lfg_urb3d",
)
MORPHOLOGY = ("frc_urb2d", "utype_urb2d", "lp_urb2d", "lb_urb2d",
              "hgt_urb2d", "hi_urb2d")
ATMOSPHERE_3D = ("dz8w", "u_phy", "v_phy", "th_phy", "rho", "p_phy",
                 "qv_phy")
ATMOSPHERE_2D = ("swdown", "glw", "swddir", "swddif", "rainbl",
                 "cosz_urb2d", "omg_urb2d", "xlat", "xlong")

#: Worst ULP distance from the port to WRF over all five fixtures, every
#: step, every column: zero for every field, measured on an RTX 4090 (sm_89)
#: and an RTX 5090 (sm_120).  A measurement, asserted for equality.
#:
#: This gate has fired.  Before the generator emitted every FP operator as
#: its IEEE intrinsic, NVRTC 13.3 for sm_120 compiled flux_flat's
#: ``c = c*ch/cm`` as ``c*(ch/cm)`` (one ULP) and the 5090 was up to 4552 ULP
#: off in b_t while the 4090 passed; spelling that one line as ``c*(ch/cm)``
#: by hand still makes the 4090 fail (b_t 4, b_q 16 ULP).  Before the trig
#: helpers used glibc_trig_flt32.cuh, a double-then-round acosf/asinf/tanf
#: gave different words on the two cards.
EXPECTED_MAX_ULP: dict[str, int] = {}


def load_fixture(variant: str) -> dict[str, np.ndarray]:
    """``{dump name: array}``; arrays keep the Fortran index order."""
    with np.load(FIXTURE_DIR / f"bep_bem_{variant}.npz") as npz:
        return {k.replace("__", "/"): npz[k] for k in npz.files}


def to_gpuwm(a: np.ndarray) -> np.ndarray:
    """WRF ``(ims:ime, [k,] jms:jme)`` with one j row -> gpuwm (k, ny, nx)."""
    if a.ndim == 2:            # (ncol, 1) -> (1, ncol)
        return np.ascontiguousarray(a.T)
    if a.ndim == 3:            # (ncol, k, 1) -> (k, 1, ncol)
        return np.ascontiguousarray(a.transpose(1, 2, 0))
    raise ValueError(a.shape)


def table_inputs(fx: dict[str, np.ndarray]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in fx.items():
        if not key.startswith("tbl/"):
            continue
        name = key[4:]
        out[name] = (value.reshape(()).item() if value.size == 1
                     and name in ("icate", "gr_flag_tbl", "gr_type_tbl")
                     else value)
    return out


def run_port(fx: dict[str, np.ndarray]) -> tuple[dict, dict]:
    """Drive the port through the fixture; return per-step and final arrays."""
    import cupy as cp

    from gpuwm.core import urban_bem as ub

    nz = int(fx["nz"][0])
    nsteps = int(fx["nsteps"][0])
    num_urban_hi = int(fx["num_urban_hi"][0])
    tblf, tbli = ub.pack_bem_tables(table_inputs(fx))
    cls = ub.build_class_tables(tblf, tbli)

    dev: dict[str, object] = {}
    for name in STATE_ARRAYS + MORPHOLOGY:
        dev[name] = cp.asarray(to_gpuwm(fx[f"state0/{name}"]))
    for name in LEVEL_OUTPUTS:
        dev[name] = cp.asarray(to_gpuwm(fx[f"state0/{name}_bep"])[:nz])
    dev["sf"] = cp.asarray(to_gpuwm(fx["state0/sf_bep"])[:nz + 1])
    for name in SURFACE_OUTPUTS:
        dev[name] = cp.zeros_like(dev["frc_urb2d"])

    steps: dict[int, dict[str, np.ndarray]] = {}
    for step in range(1, nsteps + 1):
        tag = f"step{step}"
        for name in ATMOSPHERE_3D:
            dev[name] = cp.asarray(to_gpuwm(fx[f"{tag}/{name}"])[:nz])
        for name in ATMOSPHERE_2D:
            dev[name] = cp.asarray(to_gpuwm(fx[f"{tag}/{name}"]))
        # module_sf_noahdrv.F:1639-1647: zeroed on every column before
        # every call (run_bep_bem.F90 does the same).
        for name in SURFACE_OUTPUTS:
            dev[name][...] = 0.0
        dev["b_q"][...] = 0.0
        ub.launch_bep_bem_columns(
            dev, cls,
            gmt=float(fx[f"{tag}/gmt"][0]),
            julday=int(fx[f"{tag}/julday"][0]),
            declin_urb=float(fx[f"{tag}/declin_urb"][0]),
            dt=float(fx[f"{tag}/dt"][0]),
            itimestep=int(fx[f"{tag}/itimestep"][0]),
            num_urban_hi=num_urban_hi)
        steps[step] = {k: cp.asnumpy(v) for k, v in dev.items()
                       if hasattr(v, "shape")}
    return steps, steps[nsteps]


def measure(fx: dict[str, np.ndarray]) -> dict[str, int]:
    """Worst ULP per field name over every step of one fixture."""
    nz = int(fx["nz"][0])
    steps, final = run_port(fx)
    worst: dict[str, int] = {}

    def note(name: str, got: np.ndarray, want: np.ndarray) -> None:
        assert got.shape == want.shape, (name, got.shape, want.shape)
        if want.dtype.kind == "i":
            d = int(np.abs(got.astype(np.int64) - want).max())
        else:
            d = int(fp32_ulp_distance(got, want).max())
        worst[name] = max(worst.get(name, 0), d)

    for step, got in steps.items():
        tag = f"step{step}/out"
        for name in LEVEL_OUTPUTS:
            note(name, got[name], to_gpuwm(fx[f"{tag}/{name}"])[:nz])
        note("sf", got["sf"], to_gpuwm(fx[f"{tag}/sf"])[:nz + 1])
        for name in SURFACE_OUTPUTS + BEM_2D_OUTPUTS + NDM_OUTPUTS:
            note(name, got[name], to_gpuwm(fx[f"{tag}/{name}"]))
    for name in STATE_ARRAYS:
        note(f"final/{name}", final[name], to_gpuwm(fx[f"final/{name}"]))
    return worst


@pytest.fixture(scope="module")
def measured() -> dict[str, dict[str, int]]:
    return {v: measure(load_fixture(v)) for v in VARIANTS}


def test_fixtures_are_the_ones_provenance_names():
    """Each fixture is the dump its PROVENANCE row describes."""
    for variant in VARIANTS:
        fx = load_fixture(variant)
        assert int(fx["nsteps"][0]) == (30 if variant == "long" else 4)
        assert int(fx["ncol"][0]) == 12
        assert int(fx["use_wudapt_lcz"][0]) == (1 if variant == "lcz" else 0)
        assert int(fx["tbl/gr_flag_tbl"][0]) == (
            1 if variant in ("gr1pv", "gr2") else 0)


@requires_gpu
def test_bep_bem_matches_wrf471(measured):
    table: dict[str, int] = {}
    for per_variant in measured.values():
        for name, d in per_variant.items():
            table[name] = max(table.get(name, 0), d)
    assert table == {name: EXPECTED_MAX_ULP.get(name, 0) for name in table}, (
        {v: {k: d for k, d in m.items() if d} for v, m in measured.items()})


@requires_gpu
def test_rural_columns_are_never_written():
    """BEP_BEM runs only where FRC_URB2D > 0 (module_sf_bep_bem.F:710)."""
    fx = load_fixture("stock")
    _, final = run_port(fx)
    rural = np.flatnonzero(to_gpuwm(fx["state0/frc_urb2d"])[0] <= 0.0)
    assert rural.size == 2
    for name in STATE_ARRAYS:
        np.testing.assert_array_equal(
            final[name][..., rural], to_gpuwm(fx[f"state0/{name}"])[..., rural],
            err_msg=name)


#: urban_var_init's option-2/3 multi-layer PBL arrays (module_sf_urban.F:2974-2992).
PBL_ARRAYS = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
              "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
              "dlg_bep", "dl_u_bep", "sf_bep", "vl_bep")


@pytest.mark.parametrize("variant", VARIANTS)
def test_option3_cold_start_is_urban_var_init(variant):
    """The option-3 initial state gpuwm builds is WRF's, word for word.

    infra's :func:`gpuwm.core.urban_state.urban_var_init_host` writes the
    whole option-3 block of ``urban_var_init`` (module_sf_urban.F:2912-2992)
    -- on every column, rural ones included, where the building temperatures
    take the top soil layer instead of ``TBLEND_TBL`` -- and this module's
    ``init_state`` adds nothing to it.  Graded here against the state WRF's
    own ``urban_var_init`` left before BEP_BEM's first call.

    The gridded-morphology inputs (``HGT/LP/LB/HI_URB2D``, two columns of
    every fixture) are not compared: gpuwm does not ingest gridded
    morphology yet (DESIGN 2, named follow-up), so its cold start takes
    WRF's default-morphology arm there, and those columns are driven with
    the fixture's own morphology in the column tests above.
    """
    from gpuwm.core import urban_bem as ub
    from gpuwm.core.urban_state import (option_spec, resolve_dimensions,
                                        urban_var_init_host)
    from gpuwm.core.urban_tables import UrbanCategories, load_urban_params

    fx = load_fixture(variant)
    lcz = int(fx["use_wudapt_lcz"][0])
    nz = int(fx["nz"][0])
    # run_bep_bem.F90 hands urban_var_init ISURBAN = 13 and LCZ_1..11 = 51..61.
    categories = UrbanCategories(isurban=13, natural=14,
                                 lcz=tuple(range(51, 62)))
    out = urban_var_init_host(
        option=3, use_wudapt_lcz=lcz, params=load_urban_params(3, lcz),
        categories=categories, ivgtyp=to_gpuwm(fx["init/ivgtyp"]),
        tsk=to_gpuwm(fx["init/tsk"]), tslb=to_gpuwm(fx["init/tslb"]),
        tmn=to_gpuwm(fx["init/tmn"]), smois=to_gpuwm(fx["init/smois"]),
        frc_urb2d=to_gpuwm(fx["init/frc_urb2d_in"]),
        num_urban_hi=int(fx["num_urban_hi"][0]), nz=nz,
        dims=resolve_dimensions(3, ub), spec=option_spec(3, ub))
    for name in STATE_ARRAYS + PBL_ARRAYS + ("frc_urb2d", "utype_urb2d"):
        want = to_gpuwm(fx[f"state0/{name}"])
        if name in PBL_ARRAYS:
            want = want[:nz + 1 if name == "sf_bep" else nz]
        got = out[name]
        assert got.dtype == want.dtype, name
        np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32),
                                      err_msg=name)


def test_tables_pack_in_fortran_order_with_category_padding():
    """A three-class table fills the first three of eleven class slots."""
    from gpuwm.core import urban_bem as ub

    fx = load_fixture("stock")
    tbl = table_inputs(fx)
    tblf, tbli = ub.pack_bem_tables(tbl)
    offset, shape = ub.BEM_TABLE_FLOAT_LAYOUT["height_bin_tbl"]
    unpacked = tblf[offset:offset + int(np.prod(shape))].reshape(
        shape, order="F")
    np.testing.assert_array_equal(unpacked[:, :3], tbl["height_bin_tbl"])
    assert not unpacked[:, 3:].any()
    for layout, words in ((ub.BEM_TABLE_FLOAT_LAYOUT, tblf),
                          (ub.BEM_TABLE_INT_LAYOUT, tbli)):
        used = np.zeros(words.size, np.int32)
        for offset, shape in layout.values():
            used[offset:offset + int(np.prod(shape, dtype=int))] += 1
        assert (used == 1).all()


def test_a_missing_or_transposed_table_is_named():
    from gpuwm.core import urban_bem as ub

    tbl = table_inputs(load_fixture("lcz"))
    del tbl["capb_tbl"]
    with pytest.raises(KeyError, match="capb_tbl"):
        ub.pack_bem_tables(tbl)
    tbl = table_inputs(load_fixture("lcz"))
    tbl["street_direction_tbl"] = tbl["street_direction_tbl"].T.copy()
    with pytest.raises(ValueError, match="street_direction_tbl"):
        ub.pack_bem_tables(tbl)


@pytest.mark.parametrize("variant,lcz", [("stock", 0), ("lcz", 1)])
def test_the_vendored_tables_pack_to_the_words_wrf_read(variant, lcz):
    """gpuwm's URBPARM parse feeds BEP_BEM the words WRF's own parse did.

    ``after_lsm`` packs its tables from ``load_urban_params`` (infra's
    transcription of ``urban_param_init``); the fixtures hold what WRF's
    ``urban_param_init`` left in module_sf_urban.  Every word BEP_BEM reads
    must agree bit for bit, or the column proof above would be a proof about
    a different table.
    """
    from gpuwm.core import urban_bem as ub
    from gpuwm.core.urban_tables import load_urban_params

    ours = ub.pack_bem_tables(ub.tables_from_params(load_urban_params(3, lcz)))
    wrf = ub.pack_bem_tables(table_inputs(load_fixture(variant)))
    for got, want in zip(ours, wrf):
        np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32))


def test_state_spec_and_dimensions_agree_with_infra():
    """infra's option_spec / resolve_dimensions accept this module's rows."""
    from gpuwm.core import urban_bem as ub
    from gpuwm.core.urban_state import (BEM_SPEC, BEP_SPEC, option_spec,
                                        resolve_dimensions)

    spec = option_spec(3, ub)
    for name in ub.STATE_SPEC:
        assert name in spec
    assert set(ub.STATE_SPEC) == set(BEP_SPEC) | set(BEM_SPEC)
    dims = resolve_dimensions(3, ub)
    for key, value in ub.DIMENSIONS.items():
        assert dims[key] == value
    for key in ("zrd", "zwd", "gd", "zd", "zdf", "bd", "wd", "gbd", "fbd",
                "zgrd"):
        assert dims[f"urban_map_{key}"] > 0


@requires_gpu
def test_after_lsm_runs_wrfs_column_then_the_shared_couple():
    """The DESIGN 3.4 hook wires WRF's arguments to the column it proved.

    infra's cold start builds the UrbanState from the fixture's own LSM
    fields; ``after_lsm`` then zeroes what the LSM driver zeroes, runs
    BEP_BEM with its inputs mapped from ``_prepare_atmosphere`` names, the
    driver fields and the held solar geometry, and hands the column to the
    BEP lane's couple.  What the column hands over (the surface words and
    the BEM outputs) must be WRF's first call, bit for bit, and the couple
    must have weighted the source terms by the urban fraction, as
    module_sf_noahdrv.F:1684-1697 does above level 1.
    """
    import types

    import cupy as cp

    from gpuwm.core import urban_bem as ub
    from gpuwm.core.urban_state import UrbanSolar, init_urban_state
    from gpuwm.core.urban_tables import UrbanCategories, load_urban_params

    fx = load_fixture("stock")
    nz = int(fx["nz"][0])
    tag = "step1"
    params = load_urban_params(3, 0)
    fields = {
        "ivgtyp": cp.asarray(to_gpuwm(fx["init/ivgtyp"])),
        "tsk": cp.asarray(to_gpuwm(fx["init/tsk"])),
        "tslb": cp.asarray(to_gpuwm(fx["init/tslb"])),
        "tmn": cp.asarray(to_gpuwm(fx["init/tmn"])),
        "smois": cp.asarray(to_gpuwm(fx["init/smois"])),
    }
    cfg = types.SimpleNamespace(sf_urban_physics=3, use_wudapt_lcz=0,
                                num_urban_hi=int(fx["num_urban_hi"][0]))
    state = init_urban_state(
        cfg, params, UrbanCategories(isurban=13, natural=14,
                                     lcz=tuple(range(51, 62))),
        fields, nz=nz, frc_urb2d=cp.asarray(to_gpuwm(fx["init/frc_urb2d_in"])),
        module=ub)
    # gpuwm does not ingest gridded morphology yet; hand the column the
    # fixture's, as WRF had it.
    for name in ("lp_urb2d", "lb_urb2d", "hgt_urb2d", "hi_urb2d"):
        state.fields[name][...] = cp.asarray(to_gpuwm(fx[f"state0/{name}"]))
    ny, nx = state.fields["frc_urb2d"].shape
    for name in ("swdown", "glw", "swddir", "swddif", "rainbl"):
        fields[name] = cp.asarray(to_gpuwm(fx[f"{tag}/{name}"]))
    rural = {"ust": 0.3, "hfx": 50.0, "qfx": 2.0e-5, "lh": 50.0,
             "grdflx": 10.0, "albedo": 0.2, "emiss": 0.95}
    for name, value in rural.items():
        fields[name] = cp.full((ny, nx), value, cp.float32)
    atmosphere = {
        key: cp.asarray(to_gpuwm(fx[f"{tag}/{name}"])[:nz])
        for name, key in ub.ATMOSPHERE_SOURCES.items()}
    solar = UrbanSolar(
        declin=float(fx[f"{tag}/declin_urb"][0]),
        coszen=cp.asarray(to_gpuwm(fx[f"{tag}/cosz_urb2d"])),
        hrang=cp.asarray(to_gpuwm(fx[f"{tag}/omg_urb2d"])),
        xlat=cp.asarray(to_gpuwm(fx[f"{tag}/xlat"])),
        xlong=cp.asarray(to_gpuwm(fx[f"{tag}/xlong"])),
        gmt=float(fx[f"{tag}/gmt"][0]), julday=int(fx[f"{tag}/julday"][0]))

    ub.after_lsm(state, params, lsm=2, fields=fields, atmosphere=atmosphere,
                 dt=float(fx[f"{tag}/dt"][0]), itimestep=1, solar=solar,
                 cfg=cfg)

    for name, key in ub.SURFACE_OUTPUTS.items():
        got = cp.asnumpy(state.bep_out[key])
        want = to_gpuwm(fx[f"{tag}/out/{name}"])
        np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32),
                                      err_msg=name)
    for name in BEM_2D_OUTPUTS:
        got = cp.asnumpy(state.fields[name])
        want = to_gpuwm(fx[f"{tag}/out/{name}"])
        np.testing.assert_array_equal(got.view(np.int32), want.view(np.int32),
                                      err_msg=name)
    frc = to_gpuwm(fx["state0/frc_urb2d"])
    raw = to_gpuwm(fx[f"{tag}/out/b_t"])[:nz]
    weighted = cp.asnumpy(state.pbl_terms["b_t_bep"])
    np.testing.assert_array_equal(weighted[1:], (raw[1:] * frc).astype(np.float32))
    for name in rural:
        assert bool(cp.isfinite(fields[name]).all()), name


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("pbl", ["ysu", "myj"])
def test_option3_runs_in_the_physics_driver(pbl):
    """``sf_urban_physics = 3`` end to end: cold start, the surface step's
    hand-over, BEP_BEM, the couple, the 2 m overrides and the PBL (YSU with
    WRF's flag_bep arm, or MYJ routed to myjurb), three steps on a small
    domain whose first two columns are the stock MODIS urban category."""
    import sys

    import cupy as cp

    sys.path.insert(0, str(Path(__file__).parent))
    from test_urban_default_off_identity import _small_driver

    extra = ({} if pbl == "ysu"
             else {"bl_pbl_physics": 2, "sf_sfclay_physics": 2})
    state, cfg, driver = _small_driver(sf_urban_physics=3, **extra)
    urban = driver.urban
    assert urban is not None and urban.option == 3
    assert urban.pbl_terms is not None
    tlev0 = cp.asnumpy(driver.fields["tlev_urb3d"]).copy()
    for _ in range(3):
        driver.compute(state, cfg)
    f = driver.fields
    city = cp.asnumpy(f["utype_urb2d"]) > 0
    assert city[:, :2].all() and not city[:, 2:].any()
    for name in ("tsk", "hfx", "qfx", "ust", "t2", "q2", "u10", "v10",
                 "swddir", "swddif", "ts_urb2d", "sh_urb2d", "rn_urb2d"):
        assert np.isfinite(cp.asnumpy(f[name])).all(), name
    for name in STATE_ARRAYS:
        assert np.isfinite(cp.asnumpy(f[name])).all(), name
    for name, arr in urban.pbl_terms.items():
        assert np.isfinite(cp.asnumpy(arr)).all(), name
    # the building energy model ran on the city and only there: indoor air
    # moved off its cold-start value, and no rural column was touched
    tlev = cp.asnumpy(f["tlev_urb3d"])
    assert not np.array_equal(tlev[..., city], tlev0[..., city])
    np.testing.assert_array_equal(tlev[..., ~city], tlev0[..., ~city])
    assert (cp.asnumpy(f["sh_urb2d"])[city] != 0).all()
    assert (cp.asnumpy(f["sh_urb2d"])[~city] == 0).all()
    vl = cp.asnumpy(urban.pbl_terms["vl_bep"])
    assert (vl[0][city] < 1).all() and (vl[0][~city] == 1).all()
    # module_surface_driver.F:3028-3032: city T2/TH2 are level-1 theta
    th1 = cp.asnumpy(state.total_theta())[0]
    assert np.array_equal(cp.asnumpy(f["th2"])[city], th1[city])


@pytest.mark.gpu
@requires_gpu
def test_the_composed_bem_unit_compiles_within_the_priced_frame():
    """Keep BEP+BEM's real local-memory store inside its preflight price.

    The standalone kernel sweep cannot compile urban_bep_bem.cu, so it
    cannot detect this unit's frame drift.  The previous 4,144 B price
    undercharged NVRTC 12.9.86's 5,128 B frame on sm_120 by 984 B per
    resident thread.  Compile through the loader that launches the unit
    and read both exports on the actual device.
    """
    import cupy as cp

    from gpuwm.core import preflight as pf
    from gpuwm.core import urban_bem as ub

    row = pf.CHAINED_TRANSLATION_UNIT_FRAMES["urban_bem_composed"]
    module = ub._bem_module()
    frames = {name: int(module.get_function(name).local_size_bytes)
              for name in ("bep_bem_columns", "bep_bem_class_init")}
    frame = max(frames.values())
    profile = pf.local_memory_profile_from_device(cp)
    unpriced = ((frame - row.max_local_size_bytes)
                * profile.resident_thread_capacity)
    assert frame <= row.max_local_size_bytes, (
        f"urban_bem_composed compiles to {frames} B per thread on "
        f"{profile.name}, NVRTC {cp.cuda.nvrtc.getVersion()}, against its "
        f"{row.max_local_size_bytes} B preflight price, undercharging the "
        f"local-memory reservation by up to {unpriced / 1024 ** 2:.1f} MiB. "
        "Remedy: raise CHAINED_TRANSLATION_UNIT_FRAMES to this reading "
        "and record its compiler and architecture in "
        "CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW.")

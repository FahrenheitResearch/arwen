"""WRF aer_opt = 3 (gpuwm.core.rrtmg_aerosol_optics): fork oracle, twin,
kernel, and the shortwave engine consuming the optics.

The oracle is the operational HRRR fork's own Fortran (NOAA-EMC/HRRR
v4.1.21: module_radiation_driver.F gt_aod, module_ra_aerosol.F
calc_aerosol_rrtmg_sw and its helpers, module_mp_thompson.F RSLF, all
extracted verbatim and compiled with gfortran -O0 -ffp-contract=off against
glibc by tools/hrrr_radiation_driver_oracle), stored in
tests/data/aer3_oracle/aer3_oracle.npz: 320 columns x 50 layers x 14 bands.

1. the NumPy twin against the fork, word for word;
2. the CUDA kernel against the fork, word for word, in the batched SW
   engine's (column, band, layer) layout with the layer above the model
   top at 0/0/1 (GPU);
3. the batched SW engine with aer_opt = 3 optics against the per-column
   NumPy rrtmg_sw composition (itself held to WRF's Fortran at max ULP 0
   with zero aerosol) fed the same optics, word for word on every output,
   over the WRF fixture deck's day columns (GPU); and aer_opt = 0 through
   the new plumbing equal to before;
4. both compositions directly against the operational fork's compiled
   rrtmg_sw with nonzero per-band aerosol, using the same supplied
   coefficient tables and McICA inputs (sw_flux_oracle.npz). This checks
   the numerical composition, independently of the WRF wrapper and its
   coefficient initialization.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core import rrtmg_aerosol_optics as ao

F = np.float32
REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "data" / "aer3_oracle" / "aer3_oracle.npz"
FIXTURE_SHA256 = REPO / "tests" / "data" / "aer3_oracle" / "SHA256"
SW_FIXDIR = REPO / "tools" / "rrtmg_wrf461_oracle" / "sw_fixtures"
SW_FIX_FILES = ("fixtures_real.npz", "fixtures_synth.npz", "fixtures_tall.npz")
FLUX_FIXTURE = REPO / "tests/data/aer3_oracle/sw_flux_oracle.npz"
FLUX_OUTPUTS = ("swuflx", "swdflx", "swhr", "swuflxc", "swdflxc", "swhrc",
                "sibvisdir", "sibvisdif", "sibnirdir", "sibnirdif", "swdkdir",
                "swdkdif", "swdkdirc")
OUTS = ("tauaer", "ssaaer", "asyaer")


def bits_equal(name, got, want):
    got = np.asarray(got, np.float32)
    want = np.asarray(want, np.float32)
    assert got.shape == want.shape, (name, got.shape, want.shape)
    differ = got.view(np.uint32) != want.view(np.uint32)
    assert not differ.any(), (
        f"{name}: {int(differ.sum())} of {got.size} words differ; first "
        f"got {got[differ][:4]} want {want[differ][:4]}")
    return got.size


@pytest.fixture(scope="module")
def oracle():
    want = FIXTURE_SHA256.read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    assert got == want, f"aer3 oracle fixture sha256 {got} != pinned {want}"
    data = np.load(FIXTURE)
    receipt = json.loads(str(data["receipt"]))
    assert receipt["commit"] == "40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827"
    assert receipt["sources"]["module_ra_aerosol.F"] == (
        "9931482abac91768fd00674c23b8c953e69ce49fa026c2c026b123845785dc42")
    return {k: data[k] for k in data.files if k != "receipt"}


def _inputs(o):
    return [o[f"in/{k}"] for k in ("p", "t", "qv", "dz8w", "nwfa", "nifa")]


def test_tables_hash_is_pinned():
    assert ao.tables_sha256() == ao.AER3_TABLES_SHA256


def test_fixture_reaches_the_branches(oracle):
    """gt_aod's three RH-index branches and both clamps, every t_idx,
    the number caps, and the Bolton RH clamps occur in the fixture."""
    o = oracle
    p, t, qv = o["in/p"], o["in/t"], o["in/qv"]
    rh_gt = np.minimum(98.0, np.maximum(10.1, qv / ao.rslf(p, t) * 100.0))
    assert (rh_gt < 60).any() and ((rh_gt >= 60) & (rh_gt < 80)).any()
    assert (rh_gt >= 80).any()
    assert (rh_gt == F(10.1)).any() and (rh_gt == F(98.0)).any()
    t_idx = np.clip(ao._nint(F(10.999) - F(0.0333) * t), 1, 4)
    assert set(np.unique(t_idx)) == {1, 2, 3, 4}
    assert (o["in/nwfa"] > 99999.e6).any() and (o["in/nwfa"] < 1.0).any()
    assert (o["in/nifa"] > 9999.e6).any() and (o["in/nifa"] < 0.01).any()
    rh_b = ao.relative_humidity(p, t, qv)
    assert (rh_b == F(0.0)).any() and (rh_b == F(99.0)).any()


def test_twin_matches_the_fork_word_for_word(oracle):
    o = oracle
    tau, ssa, asy, taod = ao.aer3_sw_optics(*_inputs(o))
    words = 0
    for name, got in zip(OUTS, (tau, ssa, asy)):
        words += bits_equal(name, got, o[f"out/{name}"])
    words += bits_equal("taod5503d", taod, o["out/taod5503d"])
    # the driver's column sum, k order
    col = np.zeros(taod.shape[0], np.float32)
    for k in range(taod.shape[1]):
        col = (col + taod[:, k]).astype(np.float32)
    words += bits_equal("taod5502d", col, o["out/taod5502d"])
    print(f"aer3 twin vs fork: {words} words, max ULP 0")


def test_engine_layout_keeps_the_layer_above_the_top_neutral(oracle):
    o = oracle
    tau, ssa, asy, _ = ao.aer3_sw_optics(*_inputs(o))
    nz = tau.shape[1]
    ztaua, zasya, zomga = ao.sw_engine_layout(tau, ssa, asy, nz + 1)
    assert ztaua.shape == (tau.shape[0], 14, nz + 1)
    assert np.all(ztaua[:, :, nz] == 0) and np.all(zasya[:, :, nz] == 0)
    assert np.all(zomga[:, :, nz] == 1)
    bits_equal("layout tau", ztaua[:, :, :nz], tau.transpose(0, 2, 1))
    bits_equal("layout ssa", zomga[:, :, :nz], ssa.transpose(0, 2, 1))
    bits_equal("layout asy", zasya[:, :, :nz], asy.transpose(0, 2, 1))


def test_the_optics_keep_the_sw_division_numerator_in_the_normal_range(oracle):
    """The shortwave engine divides ``pasya * pomga * ptaua`` by ``zomcc``
    on the hardware division (module_ra_rrtmg_sw.F:8448), and the kernel
    header's invariant requires every operand there to be zero or a normal
    float32.  The aer_opt = 3 optics meet it by construction, measured
    here: the Lagrange factors have positive floors over the whole clamped
    RH range, gt_aod's number floors and smallest table entries bound the
    layer AOD below by the layer mass, and the layer above the model top
    is exactly zero."""
    tiny = float(np.finfo(np.float32).tiny)
    # every RH the Bolton clamp can hand the interpolation, the nodes and
    # their float32 neighbours included
    rh = np.concatenate([
        np.linspace(0.0, 99.0, 99001), ao.SPEC_RHS,
        np.nextafter(ao.SPEC_RHS, F(200.0)),
        np.nextafter(ao.SPEC_RHS, F(-1.0))]).astype(np.float32)
    rh = np.clip(rh, F(0.0), F(99.0))
    raod = float(ao._lagrange(rh, ao.RAOD_RURAL).min())
    ssa = float(ao._lagrange(rh, ao.SSA_RURAL).min())
    asy = float(ao._lagrange(rh, ao.ASY_RURAL).min())
    assert raod >= 0.058 and ssa >= 0.55 and asy >= 0.61, (raod, ssa, asy)
    # gt_aod: MAX(1., nwfa) and MAX(0.01, nifa) times the smallest
    # per-particle extinction, per unit layer mass dz8w * rhoa (kg m^-2);
    # its RH weighting is a convex combination of two table entries
    bext = float(ao.GT_LOOKUP[..., 0].min()) * 1.0         + float(ao.GT_LOOKUP[..., 1].min()) * 0.01
    numerator_per_mass = asy * ssa * raod * bext
    assert numerator_per_mass >= 6.3e-16, numerator_per_mass
    # normal for any layer heavier than this, which no model layer is not
    lightest_layer = tiny / numerator_per_mass
    assert lightest_layer <= 1.9e-23, lightest_layer
    # and on the fork's own outputs for the fixture columns, the edge
    # columns with zero and negative aerosol numbers included
    o = oracle
    numerator = (o["out/asyaer"] * o["out/ssaaer"]) * o["out/tauaer"]
    assert float(numerator.min()) >= tiny * 1.0e20, float(numerator.min())
    mass = o["in/dz8w"] * (o["in/p"] / (F(287.0) * o["in/t"]))
    assert np.all(o["out/taod5503d"] >= F(bext * 0.999) * mass)
    tau, ssa_t, asy_t, _ = ao.aer3_sw_optics(*_inputs(o))
    nz = tau.shape[1]
    ztaua, zasya, zomga = ao.sw_engine_layout(tau, ssa_t, asy_t, nz + 1)
    top = (zasya[:, :, nz] * zomga[:, :, nz]) * ztaua[:, :, nz]
    assert np.all(top.view(np.uint32) == 0)


@pytest.mark.gpu
def test_kernel_matches_the_fork_word_for_word(oracle):
    import cupy as cp
    o = oracle
    dev = [cp.asarray(a) for a in _inputs(o)]
    nz = dev[0].shape[1]
    ztaua, zasya, zomga, taod = ao.aer3_sw_optics_device(*dev, nz + 1)
    want = ao.sw_engine_layout(o["out/tauaer"], o["out/ssaaer"],
                               o["out/asyaer"], nz + 1)
    words = 0
    for name, got, w in zip(("ztaua", "zasya", "zomga"),
                            (ztaua, zasya, zomga), want):
        words += bits_equal(name, cp.asnumpy(got), w)
    words += bits_equal("taod5503d", cp.asnumpy(taod), o["out/taod5503d"])
    # two extra layers above the top (a wider engine) stay neutral too
    wide = ao.aer3_sw_optics_device(*dev, nz + 3)
    bits_equal("wide tau model layers", cp.asnumpy(wide[0])[:, :, :nz],
               want[0][:, :, :nz])
    assert np.all(cp.asnumpy(wide[2])[:, :, nz:] == 1)
    print(f"aer3 kernel vs fork: {words} words, max ULP 0")


# ---------------------------------------------------------------------------
# The SW engine consuming the optics (GPU).
# ---------------------------------------------------------------------------

def _sw_deck():
    fix = {}
    for name in SW_FIX_FILES:
        f = np.load(SW_FIXDIR / name)
        fix.update({k: f[k] for k in f.files})
    cases = sorted(k.split("/")[0] for k in fix
                   if k.endswith("/night") and int(fix[k]) == 0)
    groups = {}
    for c in cases:
        key = tuple(int(fix[f"{c}/entry/{n}"]) for n in
                    ("icld", "inflgsw", "iceflgsw", "liqflgsw", "dyofyr",
                     "nlay"))
        groups.setdefault(key, []).append(c)
    return fix, groups


def _optics_for(nlay, ncol, seed):
    """Physical per-band optics on the model layers, neutral above."""
    rng = np.random.default_rng(seed)
    nz = nlay - 1
    tau = np.zeros((ncol, nlay, 14), F)
    ssa = np.ones((ncol, nlay, 14), F)
    asy = np.zeros((ncol, nlay, 14), F)
    tau[:, :nz] = rng.uniform(0.0, 0.08, (ncol, nz, 14)).astype(F)
    ssa[:, :nz] = rng.uniform(0.55, 0.999, (ncol, nz, 14)).astype(F)
    asy[:, :nz] = rng.uniform(0.6, 0.82, (ncol, nz, 14)).astype(F)
    return tau, ssa, asy


@pytest.fixture(scope="module")
def flux_oracle():
    digest = hashlib.sha256(FLUX_FIXTURE.read_bytes()).hexdigest()
    want = (FLUX_FIXTURE.parent / "SW_FLUX_SHA256").read_text().split()[0]
    assert digest == want
    data = dict(np.load(FLUX_FIXTURE))
    receipt = json.loads(str(data.pop("receipt")))
    assert receipt["sources"]["module_ra_rrtmg_sw.F"] == (
        "04e97b7e3cea6984979fd4b59ea0293d3ed15e2ecd2ba7a7dd81979bd86f3f39")
    assert receipt["coefficient_sha256"] == hashlib.sha256(
        (SW_FIXDIR / "sw_tables.npz").read_bytes()).hexdigest()
    return data


def _flux_args(fix, c):
    e = lambda key: fix[f"{c}/entry/{key}"]
    scalars = {"tsfc", "asdir", "asdif", "aldir", "aldif", "coszen", "adjes", "scon"}
    integers = {"nlay", "icld", "dyofyr", "inflgsw", "iceflgsw", "liqflgsw"}
    keys = ("nlay", "icld", "play", "plev", "tlay", "tlev", "tsfc", "h2ovmr",
            "o3vmr", "co2vmr", "ch4vmr", "n2ovmr", "o2vmr", "asdir", "asdif",
            "aldir", "aldif", "coszen", "adjes", "dyofyr", "scon", "inflgsw",
            "iceflgsw", "liqflgsw", "cldfmcl", "taucmcl", "ssacmcl", "asmcmcl",
            "fsfcmcl", "ciwpmcl", "clwpmcl", "cswpmcl", "reicmcl", "relqmcl", "resnmcl")
    return tuple(int(e(k)) if k in integers else F(e(k)) if k in scalars else e(k)
                 for k in keys)


def test_nonzero_aerosol_fluxes_match_the_fork(flux_oracle):
    """The fork's untouched composition with supplied coefficients and McICA
    inputs, covering all/clear-sky fluxes, heating and direct/diffuse parts.
    The producer's optics are graded separately against verbatim Fortran."""
    from gpuwm.core import rrtmg_sw as sw
    fix, groups = _sw_deck()
    tables = sw.tables_from_dump(dict(np.load(SW_FIXDIR / "sw_tables.npz")))
    words = 0
    for cases in groups.values():
        for c in cases:
            aerosol = [flux_oracle[f"{c}/in/{name}"] for name in ("tau", "ssa", "asy")]
            result = sw.rrtmg_sw(tables, *_flux_args(fix, c), *aerosol, aer_opt=3)
            for name in FLUX_OUTPUTS:
                words += bits_equal(f"fork/{c}/{name}", result[name],
                                    flux_oracle[f"{c}/out/{name}"])
    print(f"aerosol SW composition vs fork: {words} words, max ULP 0")


@pytest.mark.gpu
def test_nonzero_aerosol_device_fluxes_match_the_fork(flux_oracle):
    import cupy as cp
    from gpuwm.core import rrtmg_sw as sw
    fix, groups = _sw_deck()
    tables = sw.tables_from_dump(dict(np.load(SW_FIXDIR / "sw_tables.npz")))
    cuda = sw.CudaSW(tables)
    words = 0
    for key, cs in groups.items():
        icld, inflg, iceflg, liqflg, dyofyr, nlay = key
        col = lambda k: np.stack([fix[f"{c}/entry/{k}"] for c in cs])
        scalar = lambda k: np.asarray([fix[f"{c}/entry/{k}"] for c in cs], F)
        mc = lambda k: np.stack([fix[f"{c}/entry/{k}"] for c in cs], axis=1)
        aerosol = tuple(cp.asarray(np.ascontiguousarray(np.stack([
            flux_oracle[f"{c}/in/{name}"] for c in cs]).transpose(0, 2, 1)))
            for name in ("tau", "asy", "ssa"))
        args = (len(cs), nlay, icld, col("play"), col("plev"), col("tlay"), col("tlev"),
                scalar("tsfc"), col("h2ovmr"), col("o3vmr"), col("co2vmr"), col("ch4vmr"),
                col("n2ovmr"), col("o2vmr"), scalar("asdir"), scalar("asdif"), scalar("aldir"),
                scalar("aldif"), scalar("coszen"), scalar("adjes"), dyofyr, scalar("scon"),
                inflg, iceflg, liqflg, mc("cldfmcl"), mc("taucmcl"), mc("ssacmcl"),
                mc("asmcmcl"), mc("fsfcmcl"), mc("ciwpmcl"), mc("clwpmcl"), mc("cswpmcl"),
                col("reicmcl"), col("relqmcl"), col("resnmcl"))
        result = sw.sw_batched_to_host(cuda.rrtmg_sw_batched_device(
            *args, aer_opt=3, aerosol=aerosol))
        for i, c in enumerate(cs):
            for name in FLUX_OUTPUTS:
                words += bits_equal(f"fork device/{c}/{name}", result[name][i],
                                    flux_oracle[f"{c}/out/{name}"])
    print(f"aerosol SW device composition vs fork: {words} words, max ULP 0")


@pytest.mark.gpu
def test_batched_sw_engine_with_aerosol_matches_the_numpy_composition():
    import cupy as cp
    from gpuwm.core import rrtmg_sw as sw
    fix, groups = _sw_deck()
    tables = sw.tables_from_dump(dict(np.load(SW_FIXDIR / "sw_tables.npz")))
    cuda = sw.CudaSW(tables)
    words = 0
    keys = ("swuflx", "swdflx", "swhr", "swuflxc", "swdflxc", "swhrc",
            "sibvisdir", "sibvisdif", "sibnirdir", "sibnirdif", "swdkdir",
            "swdkdif", "swdkdirc")
    for gi, (key, cs) in enumerate(sorted(groups.items())):
        icld, inflg, iceflg, liqflg, dyofyr, nlay = key
        e = lambda c, k: fix[f"{c}/entry/{k}"]
        col = lambda k: np.stack([np.asarray(e(c, k), F) for c in cs])
        scal = lambda k: np.asarray([e(c, k) for c in cs], F)
        mc = lambda k: np.stack([np.asarray(e(c, k), F) for c in cs], axis=1)
        tau, ssa, asy = _optics_for(nlay, len(cs), 7 + gi)
        aerosol = tuple(cp.asarray(np.ascontiguousarray(a.transpose(0, 2, 1)))
                        for a in (tau, asy, ssa))
        args = (len(cs), nlay, icld, col("play"), col("plev"), col("tlay"),
                col("tlev"), scal("tsfc"), col("h2ovmr"), col("o3vmr"),
                col("co2vmr"), col("ch4vmr"), col("n2ovmr"), col("o2vmr"),
                scal("asdir"), scal("asdif"), scal("aldir"), scal("aldif"),
                scal("coszen"), scal("adjes"), dyofyr, scal("scon"),
                inflg, iceflg, liqflg, mc("cldfmcl"), mc("taucmcl"),
                mc("ssacmcl"), mc("asmcmcl"), mc("fsfcmcl"), mc("ciwpmcl"),
                mc("clwpmcl"), mc("cswpmcl"), col("reicmcl"),
                col("relqmcl"), col("resnmcl"))
        out = sw.sw_batched_to_host(cuda.rrtmg_sw_batched_device(
            *args, aer_opt=3, aerosol=aerosol))
        for i, c in enumerate(cs):
            ref = sw.rrtmg_sw(
                tables, nlay, icld, e(c, "play"), e(c, "plev"), e(c, "tlay"),
                e(c, "tlev"), F(e(c, "tsfc")), e(c, "h2ovmr"),
                e(c, "o3vmr"), e(c, "co2vmr"), e(c, "ch4vmr"),
                e(c, "n2ovmr"), e(c, "o2vmr"), F(e(c, "asdir")),
                F(e(c, "asdif")), F(e(c, "aldir")), F(e(c, "aldif")),
                F(e(c, "coszen")), F(e(c, "adjes")), int(e(c, "dyofyr")),
                F(e(c, "scon")), inflg, iceflg, liqflg, e(c, "cldfmcl"),
                e(c, "taucmcl"), e(c, "ssacmcl"), e(c, "asmcmcl"),
                e(c, "fsfcmcl"), e(c, "ciwpmcl"), e(c, "clwpmcl"),
                e(c, "cswpmcl"), e(c, "reicmcl"), e(c, "relqmcl"),
                e(c, "resnmcl"), tau[i], ssa[i], asy[i], aer_opt=3)
            for k in keys:
                words += bits_equal(f"{c}/{k}", out[k][i], ref[k])
        # the aerosol moves the fluxes (the optics are not ignored)
        zero = sw.sw_batched_to_host(cuda.rrtmg_sw_batched_device(*args))
        assert not np.array_equal(zero["swdkdir"], out["swdkdir"])
        # aer_opt = 0 through the new plumbing equals neutral optics
        # handed in explicitly
        neutral = (cp.zeros_like(aerosol[0]), cp.zeros_like(aerosol[1]),
                   cp.ones_like(aerosol[2]))
        same = sw.sw_batched_to_host(cuda.rrtmg_sw_batched_device(
            *args, aer_opt=3, aerosol=neutral))
        for k in keys:
            bits_equal(f"neutral/{k}", same[k], zero[k])
    print(f"batched SW with aerosol vs NumPy composition: {words} words, "
          "max ULP 0")


@pytest.mark.gpu
def test_batched_sw_refuses_aerosol_it_cannot_honour():
    import cupy as cp
    from gpuwm.core import rrtmg_sw as sw
    tables = sw.tables_from_dump(dict(np.load(SW_FIXDIR / "sw_tables.npz")))
    cuda = sw.CudaSW(tables)
    dummy = [None] * 36
    with pytest.raises(ValueError, match="aer_opt=3 needs"):
        cuda.rrtmg_sw_batched_device(*dummy, aer_opt=3)
    with pytest.raises(NotImplementedError, match="not transcribed"):
        cuda.rrtmg_sw_batched_device(*dummy, aer_opt=2)
    with pytest.raises(ValueError, match="aer_opt=0"):
        cuda.rrtmg_sw_batched_device(*dummy, aer_opt=0,
                                     aerosol=(cp.zeros(1),) * 3)

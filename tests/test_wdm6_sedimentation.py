"""Column mass and number telescope across the actual sedimentation substep."""
import ctypes
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest


@pytest.fixture(scope="module")
def transport(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("the CPU mirror needs a C++ compiler")
    source = Path(__file__).resolve().parents[1] / "gpuwm/core/kernels/wdm6.cu"
    directory = tmp_path_factory.mktemp("rain-flux")
    wrapper = directory / "flux.cpp"
    wrapper.write_text(
        '#define WDM6_CPU_MIRROR\n#include "' + source.as_posix() + '"\n'
        'extern "C" void flux(float*q,float*n,const float*rho,const float*dz,'
        'const float*vr,const float*vn,int nz,float dt,float*out) {'
        'auto budget=wdm6_rain_substep(q,n,rho,dz,vr,vn,nz,dt);'
        'out[0]=budget.mass;out[1]=budget.number;}\n')
    library = directory / "flux.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-fPIC", "-shared",
                    "-ffp-contract=off", str(wrapper), "-o", str(library)],
                   check=True, capture_output=True, text=True)
    loaded = ctypes.CDLL(str(library))
    function = loaded.flux
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    function.argtypes = [array] * 6 + [ctypes.c_int, ctypes.c_float, array]
    function.restype = None

    def move(q, n, rho, dz, vr, vn, dt):
        values = [np.array(value, dtype=np.float32, copy=True) for value in (q, n, rho, dz, vr, vn)]
        surface = np.zeros(2, np.float32)
        function(*values, len(values[0]), dt, surface)
        return values[0], values[1], surface
    return move


def budget(values, weights):
    return np.sum(np.asarray(values, np.float64) * np.asarray(weights, np.float64))


def assert_balance(before, after, surface):
    # A single-precision stock/flux update and a double-precision sum.
    # The bound is proportional to represented inventory, not a rain-loss allowance.
    bound = 8. * np.finfo(np.float32).eps * max(abs(before), np.finfo(np.float32).tiny)
    assert abs(before - after - surface) <= bound


def test_balance_instrument_accepts_a_known_budget_and_rejects_missing_water():
    assert_balance(10., 8., 2.)
    with pytest.raises(AssertionError):
        assert_balance(10., 7.5, 2.)


def test_two_cell_solution_has_the_true_surface_mass_and_number_flux(transport):
    q, n, surface = transport([.001, .002], [100., 200.], [1., 1.], [100., 100.],
                              [.002, .002], [.001, .001], 100.)
    np.testing.assert_allclose(q, [.0012, .0016], rtol=2.e-7)
    np.testing.assert_allclose(n, [110., 180.], rtol=2.e-7)
    np.testing.assert_allclose(surface, [.02, 1000.], rtol=2.e-7)


@pytest.mark.parametrize("courant", [0., .2, .75, 1., 1.5])
def test_mass_and_number_close_on_unequal_density_and_depth(transport, courant):
    rng = np.random.default_rng(716)
    for _ in range(32):
        q = rng.uniform(0., .003, 40).astype(np.float32)
        n = rng.uniform(10., 10000., 40).astype(np.float32)
        rho = rng.uniform(.3, 1.3, 40).astype(np.float32)
        dz = rng.uniform(10., 150., 40).astype(np.float32)
        dt = 10.
        vr = np.full(40, courant / dt, np.float32)
        vn = vr * np.float32(.47)
        after_q, after_n, surface = transport(q, n, rho, dz, vr, vn, dt)
        assert (after_q >= 0).all() and (after_n >= 0).all()
        assert_balance(budget(q, rho.astype(float) * dz),
                       budget(after_q, rho.astype(float) * dz), surface[0])
        assert_balance(budget(n, dz), budget(after_n, dz), surface[1])


def test_top_only_rain_moves_instead_of_disappearing(transport):
    q, n, surface = transport([0., 0., .001], [0., 0., 1000.],
                              [1., 1., 1.], [50., 50., 50.],
                              [.015] * 3, [.0075] * 3, 50.)
    np.testing.assert_allclose(q, [0., .00075, .00025], rtol=2.e-7)
    np.testing.assert_allclose(n, [0., 375., 625.], rtol=2.e-7)
    assert surface.tolist() == [0., 0.]


@pytest.fixture(scope="module")
def evolving_transport(tmp_path_factory):
    """Compile the entire production rain block, including its time schedule."""
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("the CPU mirror needs a C++ compiler")
    source = Path(__file__).resolve().parents[1] / "gpuwm/core/kernels/wdm6.cu"
    text = source.read_text()
    start = text.index("        int mstep = 1;", text.index("// ---- rain mass+number"))
    end = text.index("        // ---- snow + graupel", start)
    block = text[start:end]
    loop = "        for (int n = 1; n <= mstep; ++n) {"
    assert block.count(loop) == 1
    block = block.replace(loop, loop + "\nfor(int j=0;j<nz;++j) "
        "out[2]=fmax(out[2],(double)fmaxf(work1r[j],workn[j])*dtcld/mstep);")
    block = block.replace("fall_r_sfc += fallout.mass / delz[0] / dtcld;",
        "fall_r_sfc += fallout.mass / delz[0] / dtcld; out[1]+=fallout.number;")
    directory = tmp_path_factory.mktemp("rain-evolution")
    wrapper = directory / "evolution.cpp"
    wrapper.write_text('#define WDM6_CPU_MIRROR\n#include "' + source.as_posix() + '"\n'
        'extern "C" void evolve(float*qr,float*nr,const float*den,const float*delz,'
        'int nz,float dtcld,double*out){float work1r[80],workn[80];\n' + block +
        '\nout[0]=fall_r_sfc*delz[0]*dtcld;out[3]=mstep;}\n'
        'extern "C" void rates(float*q,float*n,float*rho,float*dz,int nz,float*rm,float*rn){'
        'for(int k=0;k<nz;++k){auto s=wdm6_rain_slope(q[k],n[k],rho[k],sqrtf(1.28f/rho[k]));'
        'rm[k]=s.vt/dz[k];rn[k]=s.vtn/dz[k];}}')
    library = directory / "evolution.so"
    subprocess.run([compiler, "-std=c++17", "-O2", "-fPIC", "-shared", "-ffp-contract=off",
                    str(wrapper), "-o", str(library)], check=True, capture_output=True, text=True)
    loaded = ctypes.CDLL(str(library))
    array = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")
    loaded.evolve.argtypes = [array] * 4 + [ctypes.c_int, ctypes.c_float,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")]
    loaded.rates.argtypes = [array] * 4 + [ctypes.c_int, array, array]
    return loaded


def evolving_input(empty):
    return [np.array(x, np.float32) for x in (
        [0., .001] if empty else [.001, .003],
        [0., 1000.] if empty else [636.619812, 1909.859375], [1., 1.], [1., 100.])]


def evolve(library, values, seconds):
    q, n, rho, dz = [x.copy() for x in values]
    result = np.zeros(4, np.float64)
    library.evolve(q, n, rho, dz, len(q), seconds, result)
    return q, n, result


def refined_simultaneous(library, values, seconds, cfl):
    """Independent vector flux update with a geometry-wide rate bound."""
    q, n, rho, dz = [x.copy() for x in values]
    rm, rn = np.empty_like(q), np.empty_like(q)
    bound = 1.25 * 2998.49272 * 1.e-3 ** .8 * np.max(np.sqrt(1.28 / rho.astype(float)) / dz)
    steps = int(np.ceil(seconds * bound / cfl))
    dt = seconds / steps
    surface = np.zeros(2)
    weights = rho.astype(float) * dz
    for _ in range(steps):
        library.rates(q, n, rho, dz, len(q), rm, rn)
        mass, number = q.astype(float) * weights, n.astype(float) * dz
        fm, fn = mass * rm.astype(float) * dt, number * rn.astype(float) * dt
        q = ((mass - fm + np.r_[fm[1:], 0.]) / weights).astype(np.float32)
        n = ((number - fn + np.r_[fn[1:], 0.]) / dz).astype(np.float32)
        surface += [fm[0], fn[0]]
    return q, n, surface


@pytest.mark.parametrize("empty", [False, True])
def test_evolving_rates_stay_inside_the_transport_budget(evolving_transport, empty):
    values = evolving_input(empty)
    q, n, result = evolve(evolving_transport, values, 60.)
    assert result[2] <= .4 * (1. + 8. * np.finfo(np.float32).eps)
    assert (q >= 0).all() and (n >= 0).all()
    # Accumulated FP32 fallout remains independently covered by the
    # complete production-call water test; do not relax that test here.


@pytest.mark.parametrize("empty", [False, True])
def test_rain_profile_agrees_with_temporally_refined_transport(evolving_transport, empty):
    values = evolving_input(empty)
    q, n, result = evolve(evolving_transport, values, 60.)
    coarse = refined_simultaneous(evolving_transport, values, 60., .1)
    fine = refined_simultaneous(evolving_transport, values, 60., .05)
    # This is a local first-order temporal accuracy check, separate from
    # water/number conservation. The doubled reference resolves its own
    # remaining time error much more tightly than the production check.
    np.testing.assert_allclose(coarse[0], fine[0], rtol=3.e-4)
    np.testing.assert_allclose(coarse[1], fine[1], rtol=3.e-4)
    np.testing.assert_allclose(q, fine[0], rtol=3.e-3)
    np.testing.assert_allclose(n, fine[1], rtol=3.e-3)
    np.testing.assert_allclose(result[:2], fine[2], rtol=3.e-3)


def test_first_arrival_converges_under_actual_operator_interval_refinement(evolving_transport):
    values = evolving_input(True)
    reference = refined_simultaneous(evolving_transport, values, .25, .025)
    errors = []
    for splits in (1, 32, 64):
        q, n, rho, dz = [x.copy() for x in values]
        for _ in range(splits):
            q, n, _ = evolve(evolving_transport, [q, n, rho, dz], .25 / splits)
        errors.append(np.array([abs(float(q[0]) - reference[0][0]),
                                abs(float(n[0]) - reference[1][0])]))
    # An initially dry receiver must approach the independent arrival
    # profile as the actual operator interval is refined, for both moments.
    assert (errors[1] < errors[0] * .4).all()
    assert (errors[2] < errors[1] * .6).all()

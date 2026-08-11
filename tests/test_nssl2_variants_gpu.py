"""Column smoke for the NSSL option-18 variant family on a real card.

Drives the shipped ``run_nssl2_production_step`` seam once per ported
variant and asserts what the scheme is supposed to conserve and exclude,
plus a mutation control per switch: the same variant re-run with the
kernel's switch forced back to the default lane.  If a switch did nothing,
the two arms would agree and the control assertions fail, which is what
stops these from being restatements of the code.

Two later sections go past that seam.  One drives the SHIPPED route --
RunConfig, validate_run_config, DomainState, initialize_physics,
gpuwm.core.microphysics.apply -- to pin the absent-category contract where
production actually runs rather than where a test arranges it.  The other
isolates the nucleation-pool half of diagnosed CCN from the
low-temperature limiter, on a column warm enough that the limiter cannot
fire at all.

The selector resolution and refusal contract is CPU-only and lives in
``tests/test_nssl2_variants.py``.

EVIDENCE SCOPE.  This is self-consistency and treatment, measured against
gpuwm's own default lane.  There is NO oracle comparison against WRF Fortran
for any variant path here: no WRF run, no matched trajectory, no ULP
measurement.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gpuwm.core.nssl2_contract import (
    DEFAULT_MODE,
    pinned_zero_fields,
    require_ported_nssl2_mode,
    resolve_nssl2_mode,
)


_GPU_MODES = (
    ("mp18-default", {}),
    ("mp17-no-ccn", {"nssl_ccn_on": 0}),
    ("mp18-no-hail", {"nssl_hail_on": 0}),
    ("mp22-no-hail-no-ccn", {"nssl_hail_on": 0, "nssl_ccn_on": 0}),
)


def _riming_updraft(nz=16, ny=2, nx=2):
    """Centimetre-scale graupel riming in supercooled water under 20 m/s.

    The seeded moist column behind the mp18 digest baseline never reaches
    the graupel-to-hail conversion -- its wet-growth diameter stays at the
    0.1501 m ceiling with an empty tail above it -- so a hail-switch test
    written on that profile silently proves nothing.  This regime puts the
    conversion in range.
    """
    shape = (nz, ny, nx)
    column = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None]
    pressure = np.broadcast_to(
        70000.0 - 40000.0 * column, shape).astype(np.float32)
    exner = ((pressure / 100000.0) ** (287.04 / 1004.0)).astype(np.float32)
    temperature = np.full(shape, 263.0, dtype=np.float32)
    theta = (temperature / exner).astype(np.float32)
    rho = (pressure / (287.04 * temperature)).astype(np.float32)

    fields = {
        "qv": np.full(shape, 2.0e-3, dtype=np.float32),
        "qc": np.full(shape, 5.0e-3, dtype=np.float32),
        "qr": np.full(shape, 1.0e-3, dtype=np.float32),
        "qi": np.full(shape, 2.0e-4, dtype=np.float32),
        "qs": np.full(shape, 5.0e-4, dtype=np.float32),
        "qg": np.full(shape, 6.0e-3, dtype=np.float32),
        "qh": np.zeros(shape, dtype=np.float32),
    }
    fields["qndrop"] = (fields["qc"] * 1.0e11).astype(np.float32)
    fields["qnr"] = (fields["qr"] * 1.0e7).astype(np.float32)
    fields["qni"] = (fields["qi"] * 1.0e9).astype(np.float32)
    fields["qns"] = (fields["qs"] * 5.0e7).astype(np.float32)
    fields["qng"] = np.full(
        shape, 100.0 / float(rho.mean()), dtype=np.float32)
    fields["qnh"] = np.zeros(shape, dtype=np.float32)
    fields["qnn"] = np.full(shape, 4.0e8, dtype=np.float32)
    fields["qvolg"] = (fields["qg"] / 500.0).astype(np.float32)
    fields["qvolh"] = np.zeros(shape, dtype=np.float32)

    environment = {
        "theta": theta, "rho": rho, "pressure": pressure, "exner": exner,
        "temperature": temperature,
        "w": np.full((nz + 1, ny, nx), 20.0, dtype=np.float32),
        "dz": np.full(shape, 250.0, dtype=np.float32),
    }
    return fields, environment


def _moist_column(nz=32, ny=2, nx=2):
    """A mixed-phase column with clear, condensing and glaciated cells.

    The CCN switch changes the nucleation POOL, so it only shows up where
    droplet number is activation-limited.  The riming regime above is the
    opposite case -- every cell is already dense cloud whose droplet number
    the mean-mass bounds pin -- so the CCN proof needs this profile and the
    hail proof needs that one.  Neither regime alone tests both switches,
    which is worth saying out loud: a single-regime version of this file
    would have reported a passing treatment proof for a switch it never
    exercised.
    """
    rng = np.random.default_rng(1974)
    shape = (nz, ny, nx)
    column = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None]

    pressure = np.broadcast_to(
        100000.0 - 85000.0 * column, shape).astype(np.float32)
    pressure = pressure + rng.uniform(-50.0, 50.0, shape).astype(np.float32)
    exner = ((pressure / 100000.0) ** (287.04 / 1004.0)).astype(np.float32)
    theta = np.broadcast_to(300.0 + 45.0 * column, shape).astype(np.float32)
    theta = theta + rng.uniform(-0.5, 0.5, shape).astype(np.float32)
    temperature = (theta * exner).astype(np.float32)
    rho = (pressure / (287.04 * temperature)).astype(np.float32)

    saturation_proxy = (0.018 * np.exp(-4.0 * column)).astype(np.float32)
    fields = {
        "qv": (saturation_proxy
               * rng.uniform(0.6, 1.05, shape)).astype(np.float32),
        "qc": np.where(rng.uniform(size=shape) < 0.4,
                       rng.uniform(0.0, 2.0e-3, shape), 0.0
                       ).astype(np.float32),
        "qr": np.where(rng.uniform(size=shape) < 0.3,
                       rng.uniform(0.0, 4.0e-3, shape), 0.0
                       ).astype(np.float32),
        "qi": np.where((column > 0.4) & (rng.uniform(size=shape) < 0.4),
                       rng.uniform(0.0, 1.0e-3, shape), 0.0
                       ).astype(np.float32),
        "qs": np.where((column > 0.35) & (rng.uniform(size=shape) < 0.4),
                       rng.uniform(0.0, 2.5e-3, shape), 0.0
                       ).astype(np.float32),
        "qg": np.where((column > 0.25) & (rng.uniform(size=shape) < 0.35),
                       rng.uniform(0.0, 3.0e-3, shape), 0.0
                       ).astype(np.float32),
        "qh": np.where((column > 0.25) & (rng.uniform(size=shape) < 0.2),
                       rng.uniform(0.0, 2.0e-3, shape), 0.0
                       ).astype(np.float32),
    }
    fields["qndrop"] = (fields["qc"] * 1.0e11).astype(np.float32)
    fields["qnr"] = (fields["qr"] * 1.0e8).astype(np.float32)
    fields["qni"] = (fields["qi"] * 1.0e9).astype(np.float32)
    fields["qns"] = (fields["qs"] * 5.0e7).astype(np.float32)
    fields["qng"] = (fields["qg"] * 2.0e7).astype(np.float32)
    fields["qnh"] = (fields["qh"] * 5.0e6).astype(np.float32)
    fields["qnn"] = np.maximum(
        0.0, 408163264.0 - fields["qndrop"]).astype(np.float32)
    fields["qvolg"] = (fields["qg"] / 500.0).astype(np.float32)
    fields["qvolh"] = (fields["qh"] / 800.0).astype(np.float32)

    environment = {
        "theta": theta, "rho": rho, "pressure": pressure, "exner": exner,
        "temperature": temperature,
        "w": rng.uniform(-2.0, 8.0, (nz + 1, ny, nx)).astype(np.float32),
        "dz": np.full(shape, 250.0, dtype=np.float32),
    }
    return fields, environment


def _advance(mode, fields, environment, steps=2):
    """One production step per ``steps`` at the shipped coordinator seam.

    The mode decides every variant switch, exactly as production does: the
    coordinator owns the CCN load and store, and the fused-GS callback owns
    the hail switch.  Nothing here can set a switch the resolved mode did
    not ask for, which is why the controls below re-run the SAME inputs
    under the default mode instead of poking a kernel argument.
    """
    import cupy as cp

    from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace
    from gpuwm.core.nssl2_fused_gs import launch_fused_gs
    from gpuwm.core.nssl2_nucond import launch_nucond
    from gpuwm.core.nssl2_production_coordinator import (
        NSSL2PrecipitationFields,
        NSSL2ProductionHooks,
        NSSL2RegistryFields,
        run_nssl2_production_step,
    )

    device = {name: cp.asarray(value) for name, value in fields.items()}
    shape = fields["qv"].shape
    nz, ny, nx = shape
    theta = cp.asarray(environment["theta"])
    rho = cp.asarray(environment["rho"])
    pressure = cp.asarray(environment["pressure"])
    exner = cp.asarray(environment["exner"])
    w = cp.asarray(environment["w"])
    dz = cp.asarray(environment["dz"])
    dt = 4.0

    registry = NSSL2RegistryFields(**device)
    precipitation = NSSL2PrecipitationFields(**{
        name: cp.zeros((ny, nx), dtype=cp.float32)
        for name in ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                     "graupelncv", "hailnc", "hailncv", "sr")})

    temperature_k = cp.asarray(environment["temperature"])
    ss_scratch = cp.empty(shape, dtype=cp.float32)
    fused_temperature = cp.empty(shape, dtype=cp.float32)
    primary_ice = cp.empty(shape, dtype=cp.float32)

    def fused_gs(workspace: NSSL2DriverWorkspace) -> None:
        launch_fused_gs(
            workspace, theta, rho, pressure, exner, w,
            fused_temperature, primary_ice, dz, dt, hail_on=mode.hail)

    def nucond_qvexcess(workspace: NSSL2DriverWorkspace) -> None:
        launch_nucond(
            theta, rho, pressure, exner, w,
            workspace.field("qv"), workspace.field("qc"),
            workspace.field("qr"), workspace.field("qi"),
            workspace.field("qs"), workspace.field("qndrop"),
            workspace.field("qnr"), workspace.field("qni"),
            workspace.field("qns"), workspace.field("qnn"),
            dt, supersaturation_scratch=ss_scratch,
            concentration_space=True, predicted_ccn=mode.predicted_ccn,
            validate_values=False)

    hooks = NSSL2ProductionHooks(
        fused_gs=fused_gs,
        nucond_qvexcess=nucond_qvexcess,
        moist_physics_finish=lambda _workspace: None,
    )

    for step_index in range(steps):
        cp.multiply(theta, exner, out=temperature_k)
        run_nssl2_production_step(
            rho, dz, registry, precipitation, hooks, dt,
            first_step=(step_index == 0),
            temperature_k=temperature_k,
            validate_values=False,
            mode=mode)

    outputs = {name: cp.asnumpy(value) for name, value in device.items()}
    outputs["theta"] = cp.asnumpy(theta)
    for name in ("rainnc", "graupelnc", "hailnc", "snownc", "sr"):
        outputs[name] = cp.asnumpy(getattr(precipitation, name))
    return outputs


def _zeroed_for(mode, fields):
    """The inputs a variant's Registry packages would actually give it."""
    prepared = {name: value.copy() for name, value in fields.items()}
    for name in pinned_zero_fields(mode):
        if name != "qnn":
            prepared[name] = np.zeros_like(prepared[name])
    return prepared


@pytest.mark.gpu
@pytest.mark.parametrize("label,selectors", _GPU_MODES)
def test_every_ported_variant_runs_the_production_seam(label, selectors):
    """Finite, bounded, and absent categories stay absent."""
    mode = require_ported_nssl2_mode(resolve_nssl2_mode(**selectors))
    fields, environment = _riming_updraft()
    prepared = _zeroed_for(mode, fields)
    outputs = _advance(mode, prepared, environment)

    for name, value in outputs.items():
        assert np.all(np.isfinite(value)), f"{label}: {name} not finite"
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qh",
                 "qndrop", "qnr", "qni", "qns", "qng", "qnh",
                 "qvolg", "qvolh"):
        assert np.all(outputs[name] >= 0.0), f"{label}: {name} went negative"
    assert np.all(outputs["theta"] > 100.0)
    assert np.all(outputs["theta"] < 1000.0)

    # A category the variant does not carry must be exactly absent, not
    # merely small: these are the fields WRF's Registry never allocates.
    for name in pinned_zero_fields(mode):
        if name == "qnn":
            continue
        assert np.max(np.abs(outputs[name])) == 0.0, (
            f"{label}: {name} is pinned zero but the run produced mass")

    # WRF skips the CCN store entirely when CCN are not predicted
    # (module_mp_nssl_2mom.F:3283), so the caller's array is untouched.
    if not mode.predicted_ccn:
        assert np.array_equal(outputs["qnn"], prepared["qnn"])


@pytest.mark.gpu
def test_hail_switch_actually_suppresses_the_conversion():
    """Treatment proof for nssl_hail_on, with its stub control.

    Arm A is the ported hail-off variant.  Arm B is the DEFAULT mode on
    byte-identical inputs -- the port's variant path stubbed out, since
    "hail off does nothing" and "hail off was never implemented" produce
    exactly the same run.  The control arm is what proves the regime
    reaches the conversion at all; without it, an inert switch and a
    conversion that never fires are indistinguishable, which is how the
    first version of this proof passed while testing nothing.
    """
    mode = require_ported_nssl2_mode(resolve_nssl2_mode(nssl_hail_on=0))
    fields, environment = _riming_updraft()
    prepared = _zeroed_for(mode, fields)

    ported = _advance(mode, prepared, environment)
    control = _advance(DEFAULT_MODE, prepared, environment)

    # The conversion is live on this regime: under the default mode, and
    # starting from zero hail, graupel really does convert.
    assert np.max(control["qh"]) > 0.0, (
        "the riming regime no longer reaches the graupel-to-hail "
        "conversion, so this test proves nothing about the hail switch")
    # And with hail off, it does not -- exactly zero, not merely small.
    assert np.max(np.abs(ported["qh"])) == 0.0
    assert np.max(np.abs(ported["qnh"])) == 0.0
    assert np.max(np.abs(ported["qvolh"])) == 0.0
    # The graupel the conversion would have consumed stays graupel.
    assert not np.array_equal(ported["qg"], control["qg"])


@pytest.mark.gpu
def test_ccn_switch_changes_the_nucleation_pool_and_skips_the_store():
    """Treatment proof for nssl_ccn_on, with its stub control.

    Arm B is again the DEFAULT mode on byte-identical inputs, which is what
    "the CCN variant was never implemented" looks like from outside.
    """
    mode = require_ported_nssl2_mode(resolve_nssl2_mode(nssl_ccn_on=0))
    fields, environment = _moist_column()
    prepared = _zeroed_for(mode, fields)

    ported = _advance(mode, prepared, environment, steps=3)
    control = _advance(DEFAULT_MODE, prepared, environment, steps=3)

    # With CCN predicted, the scheme reads and writes the field.
    assert not np.array_equal(control["qnn"], prepared["qnn"])
    # Without, WRF skips the store (module_mp_nssl_2mom.F:3283) and the
    # caller's array comes back exactly as it went in.
    assert np.array_equal(ported["qnn"], prepared["qnn"])
    # The variant is not merely a suppressed write: renucfrac rises to 1.0
    # (:2555-2557), changing the nucleation pool at :10116, so the droplet
    # number the scheme produces diverges too.
    assert not np.array_equal(ported["qndrop"], control["qndrop"])


@pytest.mark.gpu
def test_default_lane_is_unchanged_by_the_variant_parameters():
    """Passing the resolved default mode equals passing nothing at all."""
    fields, environment = _riming_updraft()
    explicit = _advance(DEFAULT_MODE, fields, environment)
    resolved = _advance(
        require_ported_nssl2_mode(resolve_nssl2_mode(
            nssl_2moment_on=-1, nssl_hail_on=-1, nssl_ccn_on=-1,
            nssl_density_on=-1, nssl_3moment=0)),
        fields, environment)
    for name, value in explicit.items():
        assert np.array_equal(value, resolved[name]), name


# --------------------------------------------------------------------------
# The SHIPPED route, not the probe
# --------------------------------------------------------------------------

def _shipped_hail_off_domain():
    """RunConfig -> validate -> DomainState -> initialize_physics, hail off.

    Deliberately the real entry points: nothing here arranges the
    pinned-zero precondition the way ``_zeroed_for`` does above, and the
    hail fields are RE-SEEDED after construction so the per-step pin has
    to be the thing holding them at zero.
    """
    import cupy as cp

    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.state import DomainState

    cfg = validate_run_config(RunConfig(
        nx=4, ny=4, nz=8, dx=1000.0, dy=1000.0, ztop=8000.0,
        dt=2.0, run_seconds=8.0, moist=True, mp_physics=18,
        nssl_hail_on=0))
    state = DomainState(cfg)
    state.thb[...] = cp.asarray(
        np.linspace(299.0, 320.0, cfg.nz), dtype=cp.float32)
    state.phb[...] = cp.asarray(
        np.linspace(0.0, 8000.0, cfg.nz + 1) * 9.81, dtype=cp.float32)
    state.thp[...] = cp.float32(0.0)
    state.php[...] = cp.float32(0.0)
    state.p[...] = cp.float32(85000.0)
    state.alt[...] = cp.float32(0.8)
    state.w[...] = cp.float32(1.0)
    for name, value in (
            ("qv", 0.006), ("qc", 5.0e-4), ("qr", 2.0e-5), ("qi", 1.0e-6),
            ("qs", 2.0e-6), ("qg", 1.0e-5), ("qh", 1.0e-6),
            ("qndrop", 1.0e8), ("qnr", 1.0e5), ("qni", 1.0e5),
            ("qns", 1.0e4), ("qng", 1.0e4), ("qnh", 0.0), ("qnn", 4.0e8),
            ("qvolg", 1.0e-5 / 500.0), ("qvolh", 1.0e-6 / 900.0)):
        getattr(state, name)[...] = cp.float32(value)
    return cfg, state, initialize_physics(state, cfg)


@pytest.mark.gpu
def test_hail_off_stays_hail_free_through_the_shipped_microphysics_route():
    """The pinned-zero contract holds where production actually runs.

    WRF's no-hail packages never allocate qh/qnh/qvolh, so there is
    nothing for dynamics to advect, nothing for the scheme to grow and
    nothing for the writer to publish.  gpuwm allocates the full field set
    for every mode, so that has to be enforced -- and it must be enforced
    on the SHIPPED path, not only where a test arranges zeros.

    Reproduction of the defect this pins (fix defeated by stubbing
    pin_absent_nssl2_fields to a no-op, same script otherwise):
        after initialize_physics: qh=1.000e-06 qnh=0.000e+00 qvolh=1.111e-09
        after 6 steps:            qh=3.570e-07 qnh=1.623e+00 qvolh=3.967e-10
    qnh GREW from nothing, in a configuration whose hail category WRF
    cannot represent at all.
    """
    import cupy as cp

    from gpuwm.core import microphysics

    cfg, state, driver = _shipped_hail_off_domain()

    # Construction alone clears the seeded hail: an output or restart
    # written before the first microphysics call must not carry it.
    for name in ("qh", "qnh", "qvolh"):
        assert float(cp.abs(getattr(state, name)).max()) == 0.0, name

    # Re-seed after construction, which is what a DA increment, a nest
    # feedback or a hand-edited restart would do.
    state.qh[...] = cp.float32(1.0e-6)
    state.qvolh[...] = cp.float32(1.0e-6 / 900.0)
    state.qnh[...] = cp.float32(5.0e3)

    for _ in range(6):
        driver.accept_microphysics(
            microphysics.apply(state, cfg, cfg.dt, refl_10cm_due=False))
    cp.cuda.Stream.null.synchronize()

    for name in ("qh", "qnh", "qvolh"):
        assert float(cp.abs(getattr(state, name)).max()) == 0.0, (
            f"{name} survived a hail-off run on the shipped route")
    # And the run is still a run: the categories the variant DOES carry
    # are finite and the scheme did something.
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "qndrop", "qng",
                 "qvolg", "qnn"):
        assert bool(cp.isfinite(getattr(state, name)).all()), name
    assert driver.microphysics_updates == 6


@pytest.mark.gpu
def test_the_default_lane_pins_nothing_on_the_shipped_route():
    """The enforcement is variant-only: mp18's default lane keeps its hail.

    Zeroing a category the run DOES carry would be a far worse bug than the
    one above, so the no-op half is pinned too.
    """
    import cupy as cp

    from gpuwm.core.nssl2_runtime import pin_absent_nssl2_fields
    from gpuwm.core.state import DomainState

    assert pinned_zero_fields(DEFAULT_MODE) == ()

    cfg, _state, _driver = _shipped_hail_off_domain()
    state = DomainState(dataclasses.replace(cfg, nssl_hail_on=-1))
    for name, value in (
            ("qv", 0.006), ("qc", 5.0e-4), ("qh", 1.0e-6), ("qnh", 5.0e3)):
        getattr(state, name)[...] = cp.float32(value)
    assert pin_absent_nssl2_fields(state, DEFAULT_MODE) == ()
    assert float(cp.abs(state.qh).max()) == pytest.approx(1.0e-6)
    assert float(cp.abs(state.qnh).max()) == pytest.approx(5.0e3)


# --------------------------------------------------------------------------
# The CCN nucleation POOL, isolated from the low-temperature limiter
# --------------------------------------------------------------------------

_BASE_CCN_PER_KG = 408163264.0   # nssl_cccn=0.5e9 over rho00=1.225, :1997


def _warm_activating_column(nz=8, ny=2, nx=2):
    """Supersaturated cloud above 265 K with the CCN pool half depleted.

    Purpose-built to isolate ONE half of the diagnosed-CCN port.  Above
    265 K the low-temperature limiter (:10120-10127) never arms, so the
    only thing left that ``nssl_ccn_on=0`` changes is the nucleation pool
    at :10116 -- ``cnuc = ccnc`` (the available CCN) instead of the
    default lane's ``Max(ccnc, cwnccn)`` (the background floor).

    ``qnn`` is seeded to EXACTLY the value the diagnosed branch computes,
    ``408163264 - qndrop`` (:2734), so both arms gather the same CCN
    number to the bit and the pool expression is the whole difference.
    The droplet load puts the available CCN just above half the
    background, which is where the two spellings are furthest apart: the
    renucleation cap ``pool - (background - ccn)`` is ``ccn`` under the
    default lane and only ``2*ccn - background`` under the variant.
    """
    shape = (nz, ny, nx)
    column = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None]
    pressure = np.broadcast_to(
        90000.0 - 20000.0 * column, shape).astype(np.float32)
    exner = ((pressure / 100000.0) ** (287.04 / 1004.0)).astype(np.float32)
    temperature = np.full(shape, 280.0, dtype=np.float32)
    theta = (temperature / exner).astype(np.float32)
    rho = (pressure / (287.04 * temperature)).astype(np.float32)

    # Bolton saturation, only to place qv a couple of percent above
    # saturation; the kernel uses its own table and the exact figure does
    # not matter as long as 0.5% < supersaturation < 20%.
    es = 611.2 * np.exp(17.67 * (temperature - 273.15)
                        / (temperature - 29.65))
    qvs = (0.622 * es / (pressure - es)).astype(np.float32)

    fields = {
        "qv": (1.02 * qvs).astype(np.float32),
        "qc": np.full(shape, 1.0e-3, dtype=np.float32),
        "qr": np.zeros(shape, dtype=np.float32),
        "qi": np.zeros(shape, dtype=np.float32),
        "qs": np.zeros(shape, dtype=np.float32),
        "qg": np.zeros(shape, dtype=np.float32),
        "qh": np.zeros(shape, dtype=np.float32),
        "qndrop": np.full(shape, 2.0e8, dtype=np.float32),
    }
    fields["qnr"] = np.zeros(shape, dtype=np.float32)
    fields["qni"] = np.zeros(shape, dtype=np.float32)
    fields["qns"] = np.zeros(shape, dtype=np.float32)
    fields["qng"] = np.zeros(shape, dtype=np.float32)
    fields["qnh"] = np.zeros(shape, dtype=np.float32)
    fields["qnn"] = (np.float32(_BASE_CCN_PER_KG)
                     - fields["qndrop"]).astype(np.float32)
    fields["qvolg"] = np.zeros(shape, dtype=np.float32)
    fields["qvolh"] = np.zeros(shape, dtype=np.float32)

    environment = {
        "theta": theta, "rho": rho, "pressure": pressure, "exner": exner,
        "temperature": temperature,
        "w": np.full((nz + 1, ny, nx), 3.0, dtype=np.float32),
        "dz": np.full(shape, 250.0, dtype=np.float32),
    }
    return fields, environment


@pytest.mark.gpu
def test_diagnosed_ccn_substitutes_the_pool_not_just_the_limiter():
    """The pool half of nssl_ccn_on=0, with the limiter provably inert.

    This is the half the earlier test did not reach: with the limiter
    disarmed by temperature, reverting `diagnostic_cnuc = ccn_number` to
    the default lane's `fmaxf(ccn_number, background_ccn)` makes the two
    arms BITWISE IDENTICAL here, because the seeded qnn equals the
    diagnosed CCN so both arms gather the same number and the pool
    expression is all that is left.  Measured on this column (seed qndrop
    max 200000000.0), with that one line reverted:

        ported qndrop max 352174912.0 == control 352174912.0, arrays equal

    and with the port as shipped:

        ported qndrop max 208093040.0 <  control 352174912.0

    so the assertion below is exactly the pool substitution and nothing
    else.  The already-shipped
    ``test_ccn_switch_changes_the_nucleation_pool_and_skips_the_store``
    stays green under that same revert, which is why this test exists.
    """
    mode = require_ported_nssl2_mode(resolve_nssl2_mode(nssl_ccn_on=0))
    fields, environment = _warm_activating_column()

    # The precondition the whole design rests on: nothing here is cold
    # enough to arm the :10121 limiter, so it cannot be the source of a
    # difference.
    assert float(environment["temperature"].min()) > 265.0

    ported = _advance(mode, fields, environment, steps=1)
    control = _advance(DEFAULT_MODE, fields, environment, steps=1)

    # Activation happened at all -- otherwise the pool is irrelevant and
    # the test proves nothing.
    assert np.max(control["qndrop"]) > np.max(fields["qndrop"])
    # And the variant activated FEWER droplets, because its pool is the
    # depleted CCN rather than the background floor.
    assert not np.array_equal(ported["qndrop"], control["qndrop"])
    assert np.all(ported["qndrop"] <= control["qndrop"])
    assert np.max(ported["qndrop"]) < np.max(control["qndrop"])

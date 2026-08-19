"""Does the Noah-MP cold start actually launch its CUDA kernel?

``tests/test_noahmp_device_wiring.py`` asks this of the per-timestep column
solver and the answer is yes.  It has never been asked of the **one-time**
cold start, and the answer was no: ``noahmp_cold_start`` ran a Python double
loop over every column of the grid calling
``gpuwm.core.noahmp_driver.noahmp_init_column`` -- 528 lines of pure numpy --
on every ``sf_surface_physics=4`` run this project has ever started, while
``gpuwm/core/noahmp_driver_gpu.py`` and ``kernels/noahmp_driver.cu`` held a
bitwise-verified device port with no production caller.  ``preflight.py``
priced ``noahmp_driver`` into the scheme-4 compile inventory the whole time,
so every Noah-MP run paid to compile a kernel it never launched.

**A timing gate cannot catch this.**  The host loop is correct, merely slow
(18.4-38.8 us per column measured, 6.6-14.0 s at a 360,000-column nest), so
nothing fails -- there is just a start-up cost nobody can attribute.  What
catches it is making the host loop *impossible* to call and requiring a real
cold start to survive anyway:

* :func:`test_the_cold_start_runs_on_the_device_by_default` replaces both host
  cold-start routines with raising tripwires and builds a real Noah-MP
  configuration through ``initialize_physics``.  If a single column were
  initialised on the host the build could not finish.
* :func:`test_the_trap_fires_when_the_device_arm_is_withheld` and
  :func:`test_the_env_override_selects_the_host_cold_start` withhold the
  device arm the two supported ways and require the tripwire to fire, so the
  pass above is evidence rather than a gate that has never been able to fail.
* :func:`test_the_two_cold_start_arms_agree_bit_for_bit` runs both arms over
  five real configurations -- bare ground, a two-layer pack, a three-layer
  pack, the 2000 mm SWE cap and the glacier fill -- and requires byte-for-byte
  equal carriers.  The numpy loop is the paired scalar authority; the device
  arm has to reproduce it exactly, at the same ``max_ulp == 0`` bar
  ``tests/test_noahmp_driver_cuda.py`` holds the kernel to against the WRF
  oracle.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from conftest import requires_gpu

import gpuwm.core.noahmp_driver as driver_module
import gpuwm.core.noahmp_runtime as runtime_module

from test_noahmp_runtime import _ICE, _build


class HostColdStartTripwire(RuntimeError):
    """Raised if any column is cold-started by a CPython routine."""


def _booby_trap(monkeypatch):
    """Make every host cold-start entry point impossible to call.

    Both are trapped, not just the outer one: ``noahmp_init_column`` is what
    the loop calls today, and ``snow_init_column`` is the routine underneath
    it, so a future arm that reached the snow ladder directly would still be
    caught.
    """
    def trip(*args, _name="noahmp_init_column", **kwargs):
        raise HostColdStartTripwire(
            f"the Noah-MP cold start answered a column with {_name!r} on the "
            "host; the CUDA driver kernel did not run")

    monkeypatch.setattr(runtime_module, "noahmp_init_column", trip)
    monkeypatch.setattr(
        driver_module, "snow_init_column",
        lambda *a, **k: trip(_name="snow_init_column"))


#: What ``noahmp_cold_start`` writes.  Asserted against the implementation in
#: :func:`test_the_trap_covers_the_whole_written_set` so that "both arms
#: agree" cannot quietly become "both arms agree about nine of them".
_COLD_START_WRITES = (
    "snow", "snowh", "canwat", "tslb", "smois", "sh2o", "lai",
    "isnowxy", "zsnsoxy", "tsnoxy", "snicexy", "snliqxy",
    "tvxy", "tgxy", "canicexy", "canliqxy", "eahxy", "tahxy",
    "cmxy", "chxy", "fwetxy", "sneqvoxy", "alboldxy",
    "qsnowxy", "qrainxy", "wslakexy", "waxy", "xsaixy",
)

#: 2140: ``ALBOLDXY = 0.65`` is assigned on every column of the grid with no
#: predicate at all, so it is the one value whose presence everywhere proves
#: the cold start ran over the whole slab rather than over a subset.
_ALBOLD_COLD = np.float32(0.65)


def _digest(driver) -> dict[str, str]:
    """sha256 per written carrier, dtype included."""
    import cupy as cp

    out = {}
    for name in _COLD_START_WRITES:
        array = np.ascontiguousarray(cp.asnumpy(driver.fields[name]))
        out[name] = (f"{array.dtype}:"
                     + hashlib.sha256(array.tobytes()).hexdigest())
    return out


def test_the_trap_covers_the_whole_written_set():
    """The written set this file compares must be the one the code writes."""
    assert set(_COLD_START_WRITES) == set(runtime_module.COLD_START_WRITES), (
        "the cold start writes a carrier this file neither traps nor "
        "compares, so the parity gate below has a hole in it")


@requires_gpu
def test_the_cold_start_runs_on_the_device_by_default(monkeypatch):
    """A bare default Noah-MP cold start, with the numpy loop made fatal."""
    import cupy as cp

    monkeypatch.delenv(runtime_module.HOST_COLD_START_ENV, raising=False)
    monkeypatch.delenv(runtime_module.HOST_LEAVES_ENV, raising=False)
    _booby_trap(monkeypatch)
    _state, _cfg, driver = _build(nx=6, ny=4, water_columns=1,
                                  snow_mm=45.0, snow_depth_m=0.16)

    albold = cp.asnumpy(driver.fields["alboldxy"])
    assert albold.shape == (4, 6)
    assert (albold == _ALBOLD_COLD).all(), (
        "ALBOLDXY is not 0.65 on every column, so the cold start did not run "
        "over the whole slab and the trap was never in a position to fire")
    assert (cp.asnumpy(driver.fields["tvxy"]) > 200.0).all(), (
        "TV is still at the allocation zero; 0 K is not a cold state")
    assert (cp.asnumpy(driver.fields["isnowxy"]) < 0).any(), (
        "no column built a snow layer, so the snow ladder never ran")


@requires_gpu
def test_the_trap_fires_when_the_device_arm_is_withheld(monkeypatch):
    """The same build with the host arm bound MUST trip.

    Without this, the test above would pass just as well against a trap
    installed somewhere nothing ever calls.
    """
    monkeypatch.delenv(runtime_module.HOST_COLD_START_ENV, raising=False)
    monkeypatch.delenv(runtime_module.HOST_LEAVES_ENV, raising=False)
    _booby_trap(monkeypatch)
    monkeypatch.setattr(runtime_module, "COLD_START_EVALUATOR",
                        runtime_module.cold_start_on_host)
    with pytest.raises(HostColdStartTripwire):
        _build(nx=6, ny=4, water_columns=1, snow_mm=45.0, snow_depth_m=0.16)


@requires_gpu
def test_the_env_override_selects_the_host_cold_start(monkeypatch):
    """``GPUWM_NOAHMP_HOST_COLD_START=1`` is the other way to the host arm.

    It is the switch a parity or timing run uses, so it has to be shown
    selecting the path it claims to.
    """
    monkeypatch.delenv(runtime_module.HOST_LEAVES_ENV, raising=False)
    _booby_trap(monkeypatch)
    monkeypatch.setenv(runtime_module.HOST_COLD_START_ENV, "1")
    with pytest.raises(HostColdStartTripwire):
        _build(nx=6, ny=4, water_columns=1)


@requires_gpu
def test_the_process_wide_host_switch_also_reaches_the_cold_start(monkeypatch):
    """``GPUWM_NOAHMP_HOST_LEAVES=1`` asks for the host authority everywhere.

    A run that set it and still got a device cold start would be reporting
    host numbers for a state the device produced.
    """
    monkeypatch.delenv(runtime_module.HOST_COLD_START_ENV, raising=False)
    _booby_trap(monkeypatch)
    monkeypatch.setenv(runtime_module.HOST_LEAVES_ENV, "1")
    with pytest.raises(HostColdStartTripwire):
        _build(nx=6, ny=4, water_columns=1)


#: Five configurations, chosen for the branches the cold start has, not for
#: variety: no snow at all, a two-layer pack (0.10 < SNODEP <= 0.25), a
#: three-layer pack (SNODEP > 0.45), the 2000 mm SWE cap at 2037-2040, and the
#: glacier fill at 2074-2082 with its 10 mm SWE floor.
_PARITY_CASES = {
    "bare": dict(water_columns=1),
    "two_layer_pack": dict(water_columns=1, snow_mm=45.0, snow_depth_m=0.16),
    "three_layer_pack": dict(water_columns=1, snow_mm=200.0,
                             snow_depth_m=0.60),
    "swe_cap": dict(water_columns=1, snow_mm=2500.0, snow_depth_m=6.0),
    "glacier": dict(water_columns=1, vegtyp=_ICE, xice=0.0, snow_mm=5.0,
                    snow_depth_m=0.03),
}


@requires_gpu
@pytest.mark.parametrize("case", sorted(_PARITY_CASES))
def test_the_two_cold_start_arms_agree_bit_for_bit(monkeypatch, case):
    """The device arm must reproduce the numpy authority exactly.

    Byte equality, not a tolerance: ``tests/test_noahmp_driver_cuda.py``
    already holds this kernel to ``max_ulp == 0`` against the WRF v4.6.1
    oracle, and a cold start that agreed only to a tolerance would seed every
    forecast from a state no fixture pins.
    """
    kwargs = dict(_PARITY_CASES[case], nx=6, ny=4)
    columns = kwargs["nx"] * kwargs["ny"]
    monkeypatch.delenv(runtime_module.HOST_COLD_START_ENV, raising=False)
    monkeypatch.delenv(runtime_module.HOST_LEAVES_ENV, raising=False)

    # Count the scalar-authority calls rather than trusting the binding.  An
    # A/B whose two arms were secretly the same arm would agree perfectly and
    # prove nothing -- the failure mode
    # ``project-ab-arms-prove-treatment`` was written for.
    calls = []
    real = runtime_module.noahmp_init_column
    monkeypatch.setattr(
        runtime_module, "noahmp_init_column",
        lambda *a, **k: (calls.append(1), real(*a, **k))[1])

    monkeypatch.setattr(runtime_module, "COLD_START_EVALUATOR",
                        runtime_module.cold_start_on_device)
    device = _digest(_build(**kwargs)[2])
    assert calls == [], (
        f"{case}: the device arm called the CPython column {len(calls)} "
        "times; the two arms are not distinct")

    monkeypatch.setattr(runtime_module, "COLD_START_EVALUATOR",
                        runtime_module.cold_start_on_host)
    host = _digest(_build(**kwargs)[2])
    assert len(calls) == columns, (
        f"{case}: the host arm answered {len(calls)} of {columns} columns; "
        "the scalar authority did not run")

    differ = sorted(name for name in _COLD_START_WRITES
                    if device[name] != host[name])
    assert differ == [], (
        f"{case}: the CUDA cold start and the numpy authority disagree on "
        f"{differ}")

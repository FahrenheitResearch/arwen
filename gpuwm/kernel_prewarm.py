"""Compile the uncached dycore diffusion modules while the input is read.

The defect (NV recipe dry run, D5): the first step of NCAR's CONUS 12 km
restart leg took 22 to 37 s on a WARM kernel cache, against a 50 to 57 ms
steady step.  A wall-clock sampler of the worker put 32 s of the 33.5 s
first step inside one NVRTC call: ``compile_diffusion_source`` compiling
``smag2d`` and ``diff_opt1`` on the first ``launch_wrf_smag2d_km``.  The
diffusion units compile through CuPy's direct NVRTC entry point (it keeps
``--ftz=false``, which ``RawModule`` would override), and that entry point
has no disk cache, so every run pays the compile on step 1.

Two changes remove it.  ``compile_diffusion_source`` now keeps its image in
the direct route's persistent exact-image cache
(:mod:`gpuwm.nvrtc_ptx_cache`, the store the no-FTZ units already use), so
a warm kernel cache no longer compiles these units at all.  And on a cold
cache, the worker starts these exact loader calls on a background thread
before it reads the input (the restart read is 10 to 42 s with the card
idle), so step 1 finds the module built or being built.  The image is the one the
loader would have built on step 1: same function, same source, same
options, same process-wide cache entry (``cuda_cache`` keys by device and
serialises first use, so step 1 waits for a compile still running instead
of starting a second one).

Only modules that carry no physics-parameter literals are warmed here:
their source cannot depend on a parameter set declared later
(:func:`gpuwm.physics_params.declare` refuses a set that arrives after an
affected module compiled, and none of these is affected).
"""

from __future__ import annotations

import os
import threading
import time

#: The units the direct, uncached NVRTC route compiles for the WRF
#: dycore, in the order the first step asks for them.
DIFFUSION_ROUTE_MODULES = ("smag2d", "diff_opt1", "diff6")

#: ``GPUWM_KERNEL_PREWARM=0`` turns the background compile off (step 1
#: then compiles, as before).  Default on.
ENV = "GPUWM_KERNEL_PREWARM"


def modules_for(experiment) -> tuple[str, ...]:
    """The diffusion-route units this experiment's dycore launches."""

    domains = tuple(getattr(experiment, "domains", ()) or ())
    runs = [getattr(domain, "run", None) for domain in domains]
    runs = [run for run in runs if run is not None]
    if not runs:
        return ()
    wanted = []
    if any(int(getattr(run, "diff_opt", 0) or 0) >= 1
           or int(getattr(run, "km_opt", 0) or 0) != 0 for run in runs):
        wanted.append("smag2d")
    if any(int(getattr(run, "diff_opt", 0) or 0) == 1 for run in runs):
        wanted.append("diff_opt1")
    if any(int(getattr(run, "diff_6th_opt", 0) or 0) != 0 for run in runs):
        wanted.append("diff6")
    return tuple(name for name in DIFFUSION_ROUTE_MODULES if name in wanted)


class Prewarm:
    """One background compile of ``modules``; :meth:`join` reports it."""

    def __init__(self, modules):
        self.modules = tuple(modules)
        self.seconds: dict[str, float] = {}
        self.errors: dict[str, str] = {}
        self.started = time.perf_counter()
        self.finished: float | None = None
        self._thread = threading.Thread(target=self._run, name="kernel-prewarm",
                                        daemon=True)

    def start(self) -> "Prewarm":
        self._thread.start()
        return self

    def _run(self) -> None:
        try:
            import cupy as cp
            from gpuwm.core import kernels
            from gpuwm.physics_params import _affected_modules

            affected = set(_affected_modules())
            unknown = set(self.modules) - set(kernels._DIFFUSION_MODULES)
            if unknown:
                raise ValueError(f"not diffusion-route units: {sorted(unknown)}")
            # The worker's card is device 0 of its mask, the device the run
            # opens; the cache entry is keyed by it.
            with cp.cuda.Device(0):
                for name in self.modules:
                    if name in affected:
                        self.errors[name] = "carries physics-parameter literals; not warmed"
                        continue
                    began = time.perf_counter()
                    try:
                        # The diffusion loader itself, not load_module: the
                        # cache lock is per loader, and holding load_module's
                        # through a 31 s compile stalled every other module
                        # the restore loads (V12 probe with a 10 s read:
                        # init 3.9 s became 36.6 s).  load_module(name) for
                        # these units is this call, so step 1 hits its entry.
                        kernels._load_diffusion_module(name)
                    except Exception as error:  # noqa: BLE001 - step 1 reports it
                        self.errors[name] = f"{type(error).__name__}: {error}"
                    self.seconds[name] = round(time.perf_counter() - began, 3)
        except Exception as error:  # noqa: BLE001 - nothing here may fail a run
            self.errors["*"] = f"{type(error).__name__}: {error}"
        finally:
            self.finished = time.perf_counter()

    def join(self, timeout: float | None = None) -> dict:
        self._thread.join(timeout)
        return self.receipt()

    def receipt(self) -> dict:
        return {"modules": list(self.modules), "seconds": dict(self.seconds),
                "errors": dict(self.errors),
                "wall_seconds": (None if self.finished is None
                                 else round(self.finished - self.started, 3))}


def start_for(experiment) -> Prewarm | None:
    """Start the background compile for ``experiment``, or return ``None``.

    ``None`` when it is turned off, when no unit applies, when CuPy or a
    card is absent, or when the worker sees more than one card (the run's
    device is then not known to be device 0).
    """

    if os.environ.get(ENV, "1").strip().lower() in ("0", "false", "off", "no"):
        return None
    modules = modules_for(experiment)
    if not modules:
        return None
    try:
        from gpuwm.local_gpu import no_local_gpu

        if no_local_gpu():
            return None
        import cupy as cp

        if cp.cuda.runtime.getDeviceCount() != 1:
            return None
    except Exception:  # noqa: BLE001 - no card, no warm-up
        return None
    return Prewarm(modules).start()


__all__ = ["DIFFUSION_ROUTE_MODULES", "ENV", "Prewarm", "modules_for", "start_for"]

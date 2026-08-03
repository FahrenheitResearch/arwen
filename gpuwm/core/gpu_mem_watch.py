"""Continuously observed GPU-memory high-water marks for run receipts.

The forecast runners used to assemble ``gpu_peak_used_bytes_observed``
from samples taken only inside step-boundary callbacks (the history
handler and the per-period progress callback).  ``execute_experiment``
trims the CuPy pool per STEP op and again at period commit BEFORE the
progress callback fires, so every sample landed after the intra-step
transient working set had already been released: the receipt reported
19.41 GiB against a 22.34 GiB true high-water mark on the four-domain
tree shape (-13%).  A receipt that understates VRAM is worse than no
receipt -- it invites launching a shape that demotes or OOMs.

:class:`GpuPeakMemoryWatcher` closes the gap with a lightweight daemon
thread that folds probe readings into running maxima every
``interval_seconds``, alongside the explicit boundary and end-of-run
``sample()`` calls the runners already made.  Probes are injected so
the accounting path is testable CPU-side, and each probe carries an
honest ``scope`` label saying WHAT its number measures -- the
``cudaMemGetInfo`` view is not the whole card on a WDDM host (see
:func:`gpuwm.core.preflight.device_wide_used_bytes`), and the CuPy pool
views are in-process only.

Overhead: one probe pass per tick.  The watcher machinery itself
measures in single-digit microseconds per pass CPU-side; at the default
20 Hz the device-facing cost is one ``cudaMemGetInfo`` plus two pool
counter reads per tick -- the same queries the boundary callbacks
already issued, now on a fixed, low cadence instead of per event.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import threading


#: Poll cadence for the background sampler.  Root periods run for
#: seconds on nested shapes, and the transients worth catching persist
#: for a meaningful fraction of a step, so 50 ms resolves them while
#: keeping the query load negligible.
DEFAULT_INTERVAL_SECONDS = 0.05


@dataclass(frozen=True)
class MemoryProbe:
    """One byte counter plus the honest description of its scope."""

    name: str
    scope: str
    read: Callable[[], int]


def default_cupy_probes() -> tuple[MemoryProbe, ...]:
    """The three views a CuPy-backed forecast can report.

    Imports CuPy lazily so that constructing probes (and every CPU-side
    test of the watcher) never touches the GPU stack.
    """
    import cupy as cp

    def cuda_device_used() -> int:
        free, total = cp.cuda.runtime.memGetInfo()
        return int(total - free)

    def pool_total() -> int:
        return int(cp.get_default_memory_pool().total_bytes())

    def pool_used() -> int:
        return int(cp.get_default_memory_pool().used_bytes())

    return (
        MemoryProbe(
            name="cuda_device_used",
            scope=(
                "cudaMemGetInfo total-free: device bytes the CUDA runtime "
                "sees as used.  NOT the whole card on a WDDM host -- other "
                "processes' residency (e.g. the desktop compositor) is "
                "invisible to it, though its deltas track the NVML "
                "device-wide series closely "
                "(gpuwm.core.preflight.device_wide_used_bytes)"),
            read=cuda_device_used),
        MemoryProbe(
            name="cupy_pool_total",
            scope="in-process CuPy default-pool bytes held (live + cached)",
            read=pool_total),
        MemoryProbe(
            name="cupy_pool_used",
            scope="in-process CuPy default-pool bytes live",
            read=pool_used),
    )


class GpuPeakMemoryWatcher:
    """Running maxima over injected byte probes, thread- and event-fed.

    ``sample()`` is the strict path the runners call at step boundaries
    and at end of run: probe errors propagate, exactly as the old
    ``record_memory`` closures failed loud.  The daemon thread is the
    lenient path: a probe that breaks mid-run must not kill a forecast,
    so its first error is recorded for the receipt and that probe is
    retired while its siblings keep sampling.  Either way the receipt
    can say what was measured, how often, and whether observation was
    complete.
    """

    _MECHANISM = "background-polling-thread+boundary-samples"

    def __init__(self, probes: Iterable[MemoryProbe], *,
                 interval_seconds: float = DEFAULT_INTERVAL_SECONDS):
        probes = tuple(probes)
        if not probes:
            raise ValueError("at least one memory probe is required")
        names = [probe.name for probe in probes]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate probe names: {names}")
        if not interval_seconds > 0.0:
            raise ValueError("interval_seconds must be positive")
        self._probes = probes
        self._interval = float(interval_seconds)
        self._lock = threading.Lock()
        self._peaks = {probe.name: 0 for probe in probes}
        self._counts = {probe.name: 0 for probe in probes}
        self._errors: dict[str, str] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- sampling ---------------------------------------------------------

    def _fold(self, name: str, value: int) -> None:
        with self._lock:
            if value > self._peaks[name]:
                self._peaks[name] = value
            self._counts[name] += 1

    def sample(self) -> None:
        """Strict pass over every probe; the first error propagates.

        Healthy probes are folded before the error is raised, so a
        boundary sample never discards information it already read.
        """
        first_error: BaseException | None = None
        for probe in self._probes:
            try:
                value = int(probe.read())
            except BaseException as error:  # noqa: BLE001 - re-raised below
                if first_error is None:
                    first_error = error
                continue
            self._fold(probe.name, value)
        if first_error is not None:
            raise first_error

    def _sample_lenient(self) -> None:
        for probe in self._probes:
            with self._lock:
                retired = probe.name in self._errors
            if retired:
                continue
            try:
                value = int(probe.read())
            except BaseException as error:  # noqa: BLE001 - receipt-recorded
                with self._lock:
                    self._errors[probe.name] = (
                        f"{type(error).__name__}: {error}")
                continue
            self._fold(probe.name, value)

    def _loop(self) -> None:
        while not self._stop_event.wait(self._interval):
            self._sample_lenient()
            with self._lock:
                if len(self._errors) == len(self._probes):
                    return  # every probe retired; nothing left to watch

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("the memory watcher was already started")
        self._thread = threading.Thread(
            target=self._loop, name="gpu-mem-watch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Idempotent; safe whether or not the thread ever started."""
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join()

    # -- reporting --------------------------------------------------------

    def peak_bytes(self, name: str) -> int:
        with self._lock:
            return self._peaks[name]

    def summary(self) -> dict[str, object]:
        """Receipt-facing provenance: mechanism, cadence, per-probe
        peaks with their honest scope labels, sample counts, and any
        mid-run observation failure."""
        with self._lock:
            return {
                "mechanism": self._MECHANISM,
                "interval_seconds": self._interval,
                "probes": {
                    probe.name: {
                        "peak_bytes": self._peaks[probe.name],
                        "scope": probe.scope,
                        "samples": self._counts[probe.name],
                        "error": self._errors.get(probe.name),
                    }
                    for probe in self._probes
                },
            }

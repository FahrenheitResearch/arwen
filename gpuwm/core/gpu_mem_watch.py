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
accurate ``scope`` label saying WHAT its number measures -- the
``cudaMemGetInfo`` view is not the whole card on a WDDM host (see
:func:`gpuwm.core.preflight.device_wide_used_bytes`), and the CuPy pool
views are in-process only.

WHOSE BYTES.  Every view above is either in-process (the pool) or
device-wide (``cudaMemGetInfo`` and NVML's ``memory.used`` both answer
for the whole card, every process included).  A receipt built from
those alone cannot tell a run that grew from a card that filled up
underneath it: a 5 h pair of nested runs on a 32 GB card was read as
"one leg climbs to 29.8 GB and runs 2.3x slower", when the leg itself sat at
8.2-8.6 GB throughout and a second process was holding 15-17.8 GB of the
same card and running kernels on it for 98% of the leg.  The
:func:`nvidia_smi_process_probes` views split the card by process from
NVML's per-process accounting (``nvidia-smi --query-compute-apps``): the
bytes THIS process holds, the bytes every OTHER process holds, and the
device-wide figure, plus -- through the watcher's time-weighted
``nonzero_seconds`` -- how long the card was shared.  They are sampled
on their own slow cadence (one ``nvidia-smi`` call every few seconds)
and are never strict: a host without ``nvidia-smi`` on PATH, or a WDDM
driver that reports per-process memory as N/A, records that fact in the
receipt and the forecast runs on.

Overhead: one probe pass per tick.  The watcher machinery itself
measures in single-digit microseconds per pass CPU-side; at the default
20 Hz the device-facing cost is one ``cudaMemGetInfo`` plus two pool
counter reads per tick -- the same queries the boundary callbacks
already issued, now on a fixed, low cadence instead of per event.  The
NVML views add one ``nvidia-smi`` subprocess pair per
:data:`NVIDIA_SMI_INTERVAL_SECONDS` from the daemon thread.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
import subprocess
import threading
import time


#: Poll cadence for the background sampler.  Root periods run for
#: seconds on nested shapes, and the transients worth catching persist
#: for a meaningful fraction of a step, so 50 ms resolves them while
#: keeping the query load negligible.
DEFAULT_INTERVAL_SECONDS = 0.05

#: Cadence of the per-process NVML views.  ``nvidia-smi`` is a
#: subprocess costing tens of milliseconds; a foreign process that
#: shares the card does so for minutes, so a reading every two seconds
#: resolves it while costing well under one percent of a host core.
NVIDIA_SMI_INTERVAL_SECONDS = 2.0

_MIB = 1024 ** 2


@dataclass(frozen=True)
class MemoryProbe:
    """One byte counter plus the accurate description of its scope.

    ``interval_seconds`` is the probe's own cadence; ``None`` reads it on
    every watcher tick and every boundary sample.  ``strict`` probes
    fail a boundary ``sample()`` loud (the pool and runtime views, which
    cannot fail on a working device); a non-strict probe's error is
    recorded for the receipt and the probe retired, on either path.
    """

    name: str
    scope: str
    read: Callable[[], int]
    interval_seconds: float | None = None
    strict: bool = True


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


class ProcessMemoryUnavailable(RuntimeError):
    """NVML cannot attribute device memory to processes on this host."""


def _run_nvidia_smi_text(arguments: list[str]) -> str:
    result = subprocess.run(
        ["nvidia-smi", *arguments], check=False, capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=20)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:200]
        raise ProcessMemoryUnavailable(
            f"nvidia-smi {' '.join(arguments)} exited "
            f"{result.returncode}: {detail}")
    return result.stdout


def _mib_field(text: str, *, what: str) -> int:
    value = text.strip()
    if not value or value.upper() in ("[N/A]", "N/A", "[NOT SUPPORTED]"):
        shown = repr(value) if value else "empty"
        raise ProcessMemoryUnavailable(
            f"nvidia-smi reports {what} as {shown}; NVML does not "
            "attribute device memory per process on this host (WDDM "
            "drivers answer N/A here)")
    return int(float(value))


@dataclass(frozen=True)
class NvidiaSmiReading:
    """One consistent snapshot of NVML's per-process and device views."""

    this_process_bytes: int
    other_processes_bytes: int
    device_used_bytes: int
    other_pids: tuple[int, ...]

    @property
    def shared(self) -> bool:
        return self.other_processes_bytes > 0


def parse_nvidia_smi_views(compute_apps_csv: str, gpus_csv: str, *,
                           pid: int) -> NvidiaSmiReading:
    """Split NVML's process list by ``pid`` on the card that carries it.

    ``compute_apps_csv`` is ``--query-compute-apps=pid,used_memory,gpu_uuid``
    and ``gpus_csv`` is ``--query-gpu=uuid,memory.used``, both
    ``--format=csv,noheader,nounits`` (MiB).  A process appears on the
    GPU it holds a context on; before this process has created its
    context (or on a host that lists nothing) the first GPU stands in,
    so a foreign process already resident there is still reported.
    """
    apps: list[tuple[int, int, str]] = []
    for line in compute_apps_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise ProcessMemoryUnavailable(
                f"unexpected nvidia-smi compute-apps row {line!r}")
        row_pid = int(parts[0])
        used = _mib_field(parts[1], what="a process's used_memory")
        uuid = parts[2] if len(parts) > 2 else ""
        apps.append((row_pid, used, uuid))
    devices: list[tuple[str, int]] = []
    for line in gpus_csv.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise ProcessMemoryUnavailable(
                f"unexpected nvidia-smi gpu row {line!r}")
        devices.append((parts[0], _mib_field(parts[1], what="memory.used")))
    if not devices:
        raise ProcessMemoryUnavailable("nvidia-smi listed no GPU")
    ours = [row for row in apps if row[0] == pid]
    card = ours[0][2] if ours else devices[0][0]
    on_card = [row for row in apps if row[2] == card or not row[2]]
    this_bytes = sum(used for row_pid, used, _ in on_card if row_pid == pid)
    others = [(row_pid, used) for row_pid, used, _ in on_card
              if row_pid != pid]
    device_used = next(
        (used for uuid, used in devices if uuid == card), devices[0][1])
    return NvidiaSmiReading(
        this_process_bytes=this_bytes * _MIB,
        other_processes_bytes=sum(used for _, used in others) * _MIB,
        device_used_bytes=device_used * _MIB,
        other_pids=tuple(sorted({row_pid for row_pid, _ in others})))


class _NvidiaSmiSnapshot:
    """One nvidia-smi pass shared by the three per-process probes.

    The watcher reads the probes of a pass within microseconds of each
    other; caching the snapshot for ``max_age_seconds`` keeps that pass
    at one pair of subprocess calls instead of three.
    """

    def __init__(self, *, pid: int, run: Callable[[list[str]], str],
                 max_age_seconds: float = 0.5):
        self._pid = int(pid)
        self._run = run
        self._max_age = float(max_age_seconds)
        self._lock = threading.Lock()
        self._reading: NvidiaSmiReading | None = None
        self._read_at = float("-inf")

    def reading(self) -> NvidiaSmiReading:
        with self._lock:
            now = time.perf_counter()
            if self._reading is None or now - self._read_at > self._max_age:
                apps = self._run([
                    "--query-compute-apps=pid,used_memory,gpu_uuid",
                    "--format=csv,noheader,nounits"])
                gpus = self._run([
                    "--query-gpu=uuid,memory.used",
                    "--format=csv,noheader,nounits"])
                self._reading = parse_nvidia_smi_views(
                    apps, gpus, pid=self._pid)
                self._read_at = now
            return self._reading


def nvidia_smi_process_probes(
        *, pid: int | None = None,
        interval_seconds: float = NVIDIA_SMI_INTERVAL_SECONDS,
        run: Callable[[list[str]], str] | None = None,
) -> tuple[MemoryProbe, ...]:
    """The per-process split of the card, from NVML via ``nvidia-smi``.

    ``run`` is injectable for CPU-side tests; ``pid`` defaults to this
    process.  The probes are non-strict and share one slow cadence.
    """
    snapshot = _NvidiaSmiSnapshot(
        pid=os.getpid() if pid is None else int(pid),
        run=_run_nvidia_smi_text if run is None else run)
    return (
        MemoryProbe(
            name="nvml_this_process_used",
            scope=("NVML per-process accounting (nvidia-smi "
                   "--query-compute-apps): device bytes THIS process holds"),
            read=lambda: snapshot.reading().this_process_bytes,
            interval_seconds=interval_seconds, strict=False),
        MemoryProbe(
            name="nvml_other_processes_used",
            scope=("NVML per-process accounting: device bytes held by "
                   "every OTHER process on the same card; its "
                   "nonzero_seconds is how long the card was shared"),
            read=lambda: snapshot.reading().other_processes_bytes,
            interval_seconds=interval_seconds, strict=False),
        MemoryProbe(
            name="nvml_device_used",
            scope=("NVML device-wide memory.used of the card this process "
                   "runs on: every process included"),
            read=lambda: snapshot.reading().device_used_bytes,
            interval_seconds=interval_seconds, strict=False),
    )


class GpuPeakMemoryWatcher:
    """Running maxima over injected byte probes, thread- and event-fed.

    ``sample()`` is the strict path the runners call at step boundaries
    and at end of run: a strict probe's error propagates, exactly as the
    old ``record_memory`` closures failed loud.  The daemon thread is the
    lenient path: a probe that breaks mid-run must not kill a forecast,
    so its first error is recorded for the receipt and that probe is
    retired while its siblings keep sampling.  Non-strict probes take
    the lenient path everywhere.  Either way the receipt can say what
    was measured, how often, and whether observation was complete.

    Beside the peak, every probe accumulates ``nonzero_seconds``: the
    wall time over which its last reading was above zero, weighted by
    the gap to the next reading.  For a view of OTHER processes' bytes
    that is the time the card was shared.
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
        for probe in probes:
            if (probe.interval_seconds is not None
                    and not probe.interval_seconds > 0.0):
                raise ValueError(
                    f"probe {probe.name!r}: interval_seconds must be "
                    "positive")
        self._probes = probes
        self._interval = float(interval_seconds)
        self._lock = threading.Lock()
        self._peaks = {probe.name: 0 for probe in probes}
        self._counts = {probe.name: 0 for probe in probes}
        self._last_value: dict[str, int] = {}
        self._last_at: dict[str, float] = {}
        self._nonzero_seconds = {probe.name: 0.0 for probe in probes}
        self._errors: dict[str, str] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # -- sampling ---------------------------------------------------------

    def _fold(self, name: str, value: int) -> None:
        now = time.perf_counter()
        with self._lock:
            if value > self._peaks[name]:
                self._peaks[name] = value
            self._counts[name] += 1
            previous_at = self._last_at.get(name)
            if (previous_at is not None
                    and self._last_value.get(name, 0) > 0):
                self._nonzero_seconds[name] += max(0.0, now - previous_at)
            self._last_value[name] = value
            self._last_at[name] = now

    def _due(self, probe: MemoryProbe) -> bool:
        if probe.interval_seconds is None:
            return True
        with self._lock:
            last = self._last_at.get(probe.name)
        return last is None or (
            time.perf_counter() - last >= probe.interval_seconds)

    def _read_lenient(self, probe: MemoryProbe) -> None:
        with self._lock:
            retired = probe.name in self._errors
        if retired:
            return
        try:
            value = int(probe.read())
        except BaseException as error:  # noqa: BLE001 - receipt-recorded
            with self._lock:
                self._errors[probe.name] = (
                    f"{type(error).__name__}: {error}")
            return
        self._fold(probe.name, value)

    def sample(self) -> None:
        """Strict pass over every due probe; the first strict error
        propagates.

        Healthy probes are folded before the error is raised, so a
        boundary sample never discards information it already read.
        """
        first_error: BaseException | None = None
        for probe in self._probes:
            if not self._due(probe):
                continue
            if not probe.strict:
                self._read_lenient(probe)
                continue
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
            if self._due(probe):
                self._read_lenient(probe)

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

    def observed(self, name: str) -> bool:
        """True once the probe has folded at least one reading."""
        with self._lock:
            return self._counts[name] > 0

    def peak_bytes_observed(self, name: str) -> int | None:
        """The peak, or ``None`` for a probe that never read anything --
        a receipt must not print 0 bytes for "could not be measured"."""
        with self._lock:
            return self._peaks[name] if self._counts[name] > 0 else None

    def nonzero_seconds(self, name: str) -> float:
        with self._lock:
            return self._nonzero_seconds[name]

    def nonzero_seconds_observed(self, name: str) -> float | None:
        with self._lock:
            if self._counts[name] == 0:
                return None
            return self._nonzero_seconds[name]

    def summary(self) -> dict[str, object]:
        """Receipt-facing provenance: mechanism, cadence, per-probe
        peaks with their accurate scope labels, sample counts, time above
        zero, and any mid-run observation failure."""
        with self._lock:
            return {
                "mechanism": self._MECHANISM,
                "interval_seconds": self._interval,
                "probes": {
                    probe.name: {
                        "peak_bytes": self._peaks[probe.name],
                        "scope": probe.scope,
                        "samples": self._counts[probe.name],
                        "interval_seconds": (
                            self._interval if probe.interval_seconds is None
                            else probe.interval_seconds),
                        "nonzero_seconds": round(
                            self._nonzero_seconds[probe.name], 3),
                        "last_bytes": self._last_value.get(probe.name),
                        "error": self._errors.get(probe.name),
                    }
                    for probe in self._probes
                },
            }


def process_memory_receipt(watch: GpuPeakMemoryWatcher) -> dict[str, object]:
    """The receipt rows that say WHOSE bytes the card carried.

    ``None`` where NVML could not attribute memory per process (no
    ``nvidia-smi``, or a WDDM driver answering N/A); the probe's error
    string sits in the watcher summary beside it.
    """
    shared_seconds = watch.nonzero_seconds_observed(
        "nvml_other_processes_used")
    return {
        "this_process_peak_bytes_nvml": watch.peak_bytes_observed(
            "nvml_this_process_used"),
        "other_processes_peak_bytes_nvml": watch.peak_bytes_observed(
            "nvml_other_processes_used"),
        "device_wide_peak_bytes_nvml": watch.peak_bytes_observed(
            "nvml_device_used"),
        "card_shared_seconds": (
            None if shared_seconds is None else round(shared_seconds, 1)),
        "card_shared": (
            None if shared_seconds is None else bool(shared_seconds > 0.0)),
    }

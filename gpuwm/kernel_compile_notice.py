"""One honest sentence before the first run's invisible kernel compile.

The first GPU forecast on a machine pays on the order of two minutes of
NVRTC compilation -- every physics module's ``cupy.RawModule`` and the
direct-NVRTC radiation translation units compile for the local card
before a single model step runs.  CuPy writes the compiled binaries to
its on-disk kernel cache, so every later run loads them in seconds; but
on the run that pays, nothing said so.  The first field run of the
published wheel watched the forecast phase sit at a stale status for a
hundred seconds and reasonably wondered whether it had hung.

This module is the one place that knows how to ask "is that about to
happen?".  It asks it two ways, because the first way provably missed:

**Cold cache.**  The CuPy kernel cache directory (``CUPY_CACHE_DIR``
when set, else ``~/.cupy/kernel_cache`` -- CuPy's own documented
resolution) with no compiled entries in it means every kernel this run
needs is about to be built from source.

**A cache for a different card.**  CuPy's cache key includes the target
architecture (``cupy/cuda/compiler.py`` mixes ``arch`` into the hash it
names the file after), so entries compiled for one card cannot be
loaded by another -- a card swap recompiles everything against a cache
that is not empty at all.  MEASURED, 2026-08-16: this project's
reference box had 7,164 sm_120 entries when its 5090 was moved out and
a 3080 moved in; the cold-cache test said "warm", no notice was
printed, and 51 seconds of sm_86 compilation ran silently inside the
wall clock of model step 1.  That is the case
:func:`kernel_cache_state` exists to catch.

The architecture is read from the entry itself.  A CuPy cache file is
the 40-character SHA1 of the blob followed by the blob, and for a cubin
that blob is a CUDA ELF whose ``e_flags`` carries the SM version in its
second byte.  An entry that does not decode -- a PTX fallback, a
truncated write, a future CuPy layout -- counts as **possibly this
card's**, never as evidence against it: announcing a two-minute compile
on the strength of a file we could not read is a false positive that
teaches a reader to ignore the line.

No CUDA import happens in any predicate here; every question is
answered from the filesystem alone, so callers may ask before touching
the device.  The one function that does need the device is
:func:`current_compute_capability`, which is separate, optional, and
returns ``None`` rather than raising when there is no card to ask.
"""

from __future__ import annotations

import dataclasses
import os
import struct
from pathlib import Path

#: Environment variable CuPy consults for its kernel cache location.
CUPY_CACHE_ENV = "CUPY_CACHE_DIR"

#: progress.json status published while the first-run compile is likely
#: under way; ``gpuwm go`` relays it verbatim in its heartbeat lines.
COMPILING_STATUS = "COMPILING_GPU_KERNELS"

#: :attr:`KernelCacheState.reason` -- nothing is cached at all.
COLD_CACHE = "cold_cache"

#: :attr:`KernelCacheState.reason` -- entries are cached, none of them
#: for this card's compute capability.
ARCHITECTURE_MISSING = "architecture_missing"

#: Bytes CuPy writes before the compiled blob: the blob's SHA1, hex.
_HASH_PREFIX_BYTES = 40

#: Offset of ``e_flags`` inside a 64-bit ELF header.
_ELF_FLAGS_OFFSET = 48

#: The smallest prefix that can answer "which architecture is this?".
_PROBE_BYTES = _HASH_PREFIX_BYTES + _ELF_FLAGS_OFFSET + 4


def cupy_kernel_cache_dir() -> Path:
    """Where CuPy keeps compiled kernels, by CuPy's own resolution order."""

    override = os.environ.get(CUPY_CACHE_ENV, "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cupy" / "kernel_cache"


def current_compute_capability() -> str | None:
    """This process's card as CuPy spells it (``"86"``, ``"120"``), or None.

    THE ONLY function in this module that touches the device.  It is
    called by the runners, which have already initialised CUDA by the
    time they ask; every predicate below takes the answer as an
    argument, so a caller that must not initialise CUDA simply does not
    call this.

    ``None`` for every failure -- no CuPy, no driver, no card, a driver
    that refuses the query -- because "which card" being unanswerable
    is a reason to fall back to the cold-cache test, never a reason to
    fail a forecast in its telemetry.
    """

    try:
        import cupy

        return str(cupy.cuda.Device().compute_capability)
    except Exception:  # noqa: BLE001 - telemetry never fails a run
        return None


def _entry_architecture(path: str) -> str | None:
    """The SM version one cache entry was compiled for, or ``None``.

    ``None`` means "could not tell", and every caller treats that as
    "possibly mine".  See the module docstring for the layout.
    """

    try:
        with open(path, "rb") as handle:
            head = handle.read(_PROBE_BYTES)
    except OSError:
        return None
    blob = head[_HASH_PREFIX_BYTES:]
    if len(blob) < _ELF_FLAGS_OFFSET + 4 or blob[:4] != b"\x7fELF":
        return None
    flags = struct.unpack_from("<I", blob, _ELF_FLAGS_OFFSET)[0]
    architecture = (flags >> 8) & 0xFF
    return str(architecture) if architecture else None


@dataclasses.dataclass(frozen=True)
class KernelCacheState:
    """What the cache says about the compile this run is about to pay.

    ``reason`` is ``None`` when nothing needs saying.  Everything else
    is the evidence behind it, kept so a receipt can carry WHY a notice
    fired rather than only that one did.
    """

    reason: str | None
    compute_capability: str | None
    entries: int
    entries_for_capability: int
    undecodable: int
    architectures: dict[str, int]

    @property
    def notice(self) -> str | None:
        """The one line to print, or ``None`` when silence is right."""

        if self.reason == COLD_CACHE:
            return (
                "forecast: first run on this machine -- compiling GPU "
                "kernels for the local card (typically 1-3 minutes with "
                "no visible progress; the compiled kernels are cached, "
                "so later runs skip this)")
        if self.reason == ARCHITECTURE_MISSING:
            others = ", ".join(
                f"sm_{name}" for name in sorted(self.architectures))
            return (
                f"forecast: the kernel cache holds {self.entries} entry(s), "
                f"none of them for this card -- compiling GPU kernels for "
                f"sm_{self.compute_capability} (the cache carries {others}; "
                "a card swap recompiles everything, typically 1-3 minutes "
                "with no visible progress, and the result is cached for "
                "later runs)")
        return None


def scan_kernel_cache(cache_dir: Path | None = None
                      ) -> tuple[int, int, dict[str, int]]:
    """Census one cache directory: entries, undecodable, per-architecture.

    A missing or unreadable directory censuses as empty, which is what
    every caller wants: in that state the coming run compiles from
    source either way.
    """

    directory = cupy_kernel_cache_dir() if cache_dir is None else cache_dir
    entries = 0
    undecodable = 0
    architectures: dict[str, int] = {}
    try:
        with os.scandir(directory) as scan:
            for entry in scan:
                if not entry.is_file():
                    continue
                entries += 1
                architecture = _entry_architecture(entry.path)
                if architecture is None:
                    undecodable += 1
                else:
                    architectures[architecture] = (
                        architectures.get(architecture, 0) + 1)
    except OSError:
        return 0, 0, {}
    return entries, undecodable, architectures


def kernel_cache_state(cache_dir: Path | None = None, *,
                       compute_capability: str | None = None,
                       census: tuple[int, int, dict[str, int]] | None = None
                       ) -> KernelCacheState:
    """Whether the coming run compiles, and on what evidence.

    ``compute_capability`` is this run's card, as
    :func:`current_compute_capability` spells it.  ``None`` means "not
    asked / no card to ask", and then only the cold-cache half applies
    -- the architecture half cannot be answered without knowing which
    architecture, and a guess there would announce a compile that never
    happens.

    ``census`` is a :func:`scan_kernel_cache` result taken EARLIER.
    Pass it, and pass it from before the run touched the GPU.  MEASURED
    on the reference box while building this: staging 200 sm_120 entries
    against an sm_86 card and running the chain, the notice stayed
    silent even with the architecture test in place -- because by the
    time the runner asks, its own first kernels have already been
    compiled INTO the cache it is asking about, and a cache with 160
    fresh sm_86 entries is warm.  The question is "was this cache usable
    when the run started", and it can only be answered with a reading
    from when the run started.
    """

    if census is None:
        entries, undecodable, architectures = scan_kernel_cache(cache_dir)
    else:
        entries, undecodable, architectures = census
    mine = (0 if compute_capability is None
            else architectures.get(str(compute_capability), 0))
    if entries == 0:
        reason = COLD_CACHE
    elif (compute_capability is not None and mine == 0
          and undecodable == 0 and architectures):
        # Positive evidence only: entries exist, every one of them
        # decoded, and not one decoded to this card.
        reason = ARCHITECTURE_MISSING
    else:
        reason = None
    return KernelCacheState(
        reason=reason,
        compute_capability=(None if compute_capability is None
                            else str(compute_capability)),
        entries=entries, entries_for_capability=mine,
        undecodable=undecodable, architectures=dict(architectures))


def kernel_cache_is_cold(cache_dir: Path | None = None) -> bool:
    """True when the CuPy kernel cache holds no compiled entries at all.

    A missing directory, an unreadable one, and an existing-but-empty
    one all count as cold: in every one of those states the coming run
    compiles from source.  Any regular file in the directory counts as
    warmth -- this predicate has no business knowing the entries'
    naming scheme.

    UNCHANGED, deliberately, now that :func:`kernel_cache_state` answers
    the larger question.  "Is anything cached at all" is still a real
    question with a cheap answer, and widening it in place would have
    made every existing caller's meaning depend on which card was
    plugged in.
    """

    directory = cupy_kernel_cache_dir() if cache_dir is None else cache_dir
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_file():
                    return False
    except OSError:
        return True
    return True


def kernel_compile_notice(cache_dir: Path | None = None, *,
                          compute_capability: str | None = None
                          ) -> str | None:
    """The one line to print before compilation, or ``None``.

    ``None`` means nothing needs saying and saying anything would be
    noise.  Pass ``compute_capability`` (from
    :func:`current_compute_capability`, which the runners call) to get
    the changed-card case as well as the first-run one.
    """

    return kernel_cache_state(
        cache_dir, compute_capability=compute_capability).notice


__all__ = [
    "ARCHITECTURE_MISSING", "COLD_CACHE", "COMPILING_STATUS",
    "CUPY_CACHE_ENV", "KernelCacheState", "cupy_kernel_cache_dir",
    "current_compute_capability", "kernel_cache_is_cold",
    "kernel_cache_state", "kernel_compile_notice", "scan_kernel_cache",
]

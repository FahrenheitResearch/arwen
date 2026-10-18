"""Compile a run's whole CUDA kernel set at once, on every core, before step 1.

THE DEFECT (2026-10-16: "no single-CPU path on the critical path").  On
a cold kernel cache -- a new card architecture, a fresh box, a wiped cache --
a forecast sat at 100 % of ONE core with its card at 0 % for minutes before
step 1, compiling 160-220 NVRTC images one after another as each kernel was
first reached (x1.40962 on an RTX 4090: step 1 111.5 s against 0.3 s steady
steps; on a 5090 214-301 s; route-B MPAS physics on an RTX PRO 6000: ~10 min).

WHAT THIS DOES.  Every compile route in gpuwm and gpuwm-hex -- CuPy's cached
``RawModule``/``RawKernel``/``ElementwiseKernel`` route, the engine's direct
exact-image cache (``gpuwm.nvrtc_ptx_cache``), the in-memory physics carriers
and every unit that calls ``compile_using_nvrtc`` itself -- ends in one CuPy
function, ``cupy.cuda.compiler._compile_using_nvrtc_no_warning``.  This module
wraps that one function:

* **Plan.**  Each request a run makes (source, options, architecture, program
  name, name expressions) is recorded into a plan file for this run's
  configuration (``tag``) and compiler identity.  The plan is the run's kernel
  set, in the very keys the compiler is called with.
* **Compile ahead.**  At the next start the plan is submitted to a thread pool
  as wide as the cores this process may use, longest compile first (NVRTC
  releases the GIL: 8 threads compiled 8 modules 5.4x faster than one on a
  loaded host).  This runs while the host reads and prepares its inputs.
* **Single flight.**  A request that arrives while its compile is running
  waits for it; one still queued is taken back and compiled by the caller, as
  before; one already finished is returned at once.  Two threads never
  compile the same request twice.

WHY THE OUTPUT IS UNCHANGED.  The wrapper returns exactly what NVRTC returned
for the identical inputs, and every route still writes its own cache entry,
under its own key, from those bytes -- nothing else about a route changes.
Two inputs are not part of the match when they cannot change the image: the
program name and whether CuPy wrote the source to a temporary file (measured
on NVRTC 13: one image for every name and path).  When they CAN change it --
the source uses ``assert`` or ``__FILE__``, or the options ask for line or
debug info -- they are part of the match.  (Such images embed CuPy's random
temporary path, so two serial compiles of them already differ.)  The
architecture is recorded as "the device's" and resolved on the card the run
uses, so a plan made on one card architecture compiles for another.

A ROW THE ROUTE WILL FIND IN ITS CACHE IS NOT COMPILED AHEAD (2026-10-17).
The plan holds the requests that reached NVRTC, i.e. cache misses when they
were recorded.  Replaying them all at the next start recompiled kernels the
routes would load from disk anyway: on an RTX 4090 with a warm cache and plans
present, ``gpuwm warm-kernels`` took 122.5 s and 78 NVRTC compiles against
2.5 s and 0 without the plans, and the second run on any box paid the same.
So each row records, per architecture, the disk entry its route stores the
image under: CuPy's cached route names its program after that entry
(``<sha1>.cubin.cu``), and a route with its own disk cache names its entry
around the compile (:func:`cache_entry`) and may name a probe that computes
the entry for another architecture.  At replay a row whose entry exists for
this card is skipped (``pool_skipped_cached``); rows without one (in-memory
routes, an architecture never seen) compile ahead as before.  A CuPy row
recorded on one card finds its entry on another by CuPy's own key form, used
only after it reproduces the entry recorded on the first card.

``GPUWM_KERNEL_PLAN=off`` turns it off; ``GPUWM_KERNEL_PLAN_WORKERS`` caps the
pool; ``GPUWM_KERNEL_PLAN_DIR`` moves the plans.  The same file ships in
gpuwm (``gpuwm.kernel_plan``) and gpuwm-hex (``hexcore.kernel_plan``); the
first one installed in a process owns the wrapper.
"""

from __future__ import annotations

import atexit
import concurrent.futures
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

SCHEMA = "gpuwm.kernel-plan/v1"
ENVIRONMENT = "GPUWM_KERNEL_PLAN"
WORKERS_ENVIRONMENT = "GPUWM_KERNEL_PLAN_WORKERS"
DIRECTORY_ENVIRONMENT = "GPUWM_KERNEL_PLAN_DIR"
MAX_ENTRIES = 4000
#: Entries not requested in this many runs of their tag leave the plan.
KEEP_RUNS = 4
POOL_PREFIX = "kernel-plan"
_MARK = "__gpuwm_kernel_plan__"

_lock = threading.Lock()
_flight: dict[str, concurrent.futures.Future] = {}
_requested: dict[str, dict[str, Any]] = {}
_state: dict[str, Any] = {"installed": False, "plan": None, "tag": None, "tag_text": None,
                          "device_arch": None, "replay": None, "counts": {}}


def enabled() -> bool:
    return os.environ.get(ENVIRONMENT, "").strip().lower() not in ("off", "0", "no", "false")


def _compiler():
    from cupy.cuda import compiler

    return compiler


def compiler_identity() -> dict[str, Any]:
    import cupy

    compiler = _compiler()
    return {"cupy": str(cupy.__version__), "nvrtc": list(compiler._get_nvrtc_version()),
            "cache_key": str(compiler._get_cupy_cache_key()),
            "use_ptx": bool(getattr(compiler, "_use_ptx", False)),
            "use_pch": bool(getattr(compiler, "_use_pch", False))}


def default_tag_text(argv: Iterable[str] | None = None) -> str:
    """The run's configuration as far as kernels can see it.

    Command-line words that name a file or carry a number (paths, dates,
    hours, mesh sizes) are dropped: they change the run, not its kernel set.
    The numerics and exactness switches that change compile options stay.
    """

    words = [str(word) for word in (sys.argv[1:] if argv is None else argv)]
    kept = [word for word in words if "/" not in word and "\\" not in word
            and not any(ch.isdigit() for ch in word)]
    for name in ("GPUWM_HEX_NUMERICS", "GPUWM_WRF_EXACT", "CUPY_CUDA_COMPILE_WITH_DEBUG"):
        if os.environ.get(name):
            kept.append(f"{name}={os.environ[name]}")
    return " ".join(kept)


def _device_arch() -> str:
    arch = _state["device_arch"]
    if arch is None:
        arch = str(_compiler()._get_arch())
        _state["device_arch"] = arch
    return arch


def _arguments(source, options, arch, filename, name_expressions, cache_in_memory, method):
    """The recorded form of one request (architecture relative to the card)."""

    options = tuple(str(option) for option in (options or ()))
    device = _device_arch()
    return {"source": source, "options": list(options),
            "arch": "device" if arch is None or str(arch) == device else str(arch),
            "filename": str(filename), "cache_in_memory": bool(cache_in_memory),
            "name_expressions": None if name_expressions is None else [str(n) for n in name_expressions],
            "method": None if method is None else str(method)}


def _key(arguments: dict[str, Any]) -> str:
    """What decides the image: the plan's entry digest and the flight key."""

    source, options = arguments["source"], tuple(arguments["options"])
    material = {"source": source, "options": list(options), "arch": arguments["arch"],
                "name_expressions": arguments["name_expressions"], "method": arguments["method"],
                # NVRTC embeds the basename in internal symbols even without __FILE__, asserts or
                # debug options (WSM5/WSM7's portable libm tables). Serving wsm5.cu for kern.cu
                # changed the exact-mode cubin in a serial GPU battery. The in-memory route also
                # changes the program name CuPy hands NVRTC.
                "filename": arguments["filename"], "cache_in_memory": arguments["cache_in_memory"]}
    return hashlib.sha256(json.dumps(material, sort_keys=True).encode("utf-8")).hexdigest()


def _current_device() -> int | None:
    try:
        import cupy

        return int(cupy.cuda.runtime.getDevice())
    except Exception:
        return None


def _use(device: int | None) -> None:
    """Make ``device`` current on this thread (a no-op without CuPy)."""

    if device is None:
        return
    import cupy

    cupy.cuda.Device(int(device)).use()


#: CuPy's cached route compiles ``<cache entry>.cu``: the entry is a SHA-1 of its key plus the extension.
_CUPY_PROGRAM = re.compile(r"^([0-9a-f]{40}\.(?:cubin|ltoir))\.cu$")
CUPY_ROUTE = "cupy-disk"
_SHARED = "_gpuwm_kernel_plan_shared"


def _shared() -> dict[str, Any]:
    """State both packages' copies of this module share (kept on CuPy's compiler module)."""

    return vars(_compiler()).setdefault(_SHARED, {"local": threading.local()})


@contextlib.contextmanager
def cache_entry(route: str, path: str | Path):
    """Name the disk entry a compile made inside this block is stored under.

    ``route`` is ``"package.module:function"`` naming a probe
    ``function(arguments, arch) -> path | None`` that computes the entry of a
    recorded request for another architecture (or any other text for none).
    """

    try:
        local = _shared()["local"]
    except Exception:
        yield
        return
    previous = getattr(local, "entry", None)
    local.entry = (str(route), str(path))
    try:
        yield
    finally:
        local.entry = previous


def _cache_root() -> Path | None:
    """CuPy's resolved disk cache directory (relative entries live under it)."""

    compiler = _compiler()
    for name in ("get_cache_dir", "_get_cache_dir"):
        getter = getattr(compiler, name, None)
        if callable(getter):
            try:
                return Path(getter())
            except Exception:
                return None
    disk_type = getattr(compiler, "_DiskKernelCacheBackend", None)
    backend = getattr(compiler, "_kernel_cache_backend", None)
    if isinstance(disk_type, type) and isinstance(backend, disk_type):
        return Path(backend._cache_dir)
    return None


def _entry_of(filename, cache_in_memory) -> tuple[str, str] | None:
    """(route, entry) of the request being compiled, entry relative to the cache root when under it."""

    entry = getattr(_shared()["local"], "entry", None)
    if entry is None:
        match = None if cache_in_memory else _CUPY_PROGRAM.match(os.path.basename(str(filename)))
        if match is None:
            return None
        return CUPY_ROUTE, match.group(1)
    route, path = entry
    root = _cache_root()
    if root is not None and os.path.isabs(path):
        try:
            path = Path(path).relative_to(root).as_posix()
        except ValueError:
            pass
    return route, path


def _probe(route: str):
    if ":" not in route:
        return None
    module, _, function = route.partition(":")
    try:
        return getattr(importlib.import_module(module), function)
    except Exception:
        return None


def _cupy_extra_sources() -> tuple[Any, ...]:
    """The ``extra_source`` values CuPy hashes into its key: none, or its own header text (RawModule,
    elementwise kernels).  NVRTC never reads it, so the funnel cannot record it; the replay tries both."""

    try:
        from cupy._core import core

        return (None, core._get_header_source())
    except Exception:
        return (None,)


def _cupy_entry(arguments: dict[str, Any], arch: str, extra_source: Any = None) -> str:
    """The disk entry CuPy's cached route names for this request on ``arch``.

    Forms CuPy's own key (``_compile_with_cache_cuda``) from the options the
    funnel received, which already carry what that route appended.  The caller
    trusts it only after it reproduces the entry recorded on another card, so
    a CuPy release that keys differently (or a request with extra_source)
    compiles ahead as before instead of being skipped.
    """

    compiler = _compiler()
    options = tuple(arguments["options"])
    env = ((str(arch), options, compiler._get_nvrtc_version(), "nvrtc")
           + tuple(compiler._get_arch_for_options_for_nvrtc(str(arch))))
    cache = compiler._empty_file_preprocess_cache
    base = cache.get(env)
    if base is None:
        base = compiler._preprocess("", options, str(arch), "nvrtc")
        cache[env] = base
    key_src = "%s %s %s %s %s" % (env, base, arguments["source"], extra_source, compiler._get_cupy_cache_key())
    names = arguments["name_expressions"]
    if names:
        key_src += " ne%d " % compiler._cache_payload_version
        key_src += "\0".join(sorted(names))
    return compiler._hash_hexdigest(key_src.encode("utf-8")) + (".ltoir" if "-dlto" in options else ".cubin")


def _cupy_entry_elsewhere(arguments: dict[str, Any], entries: dict, target: str) -> str | None:
    """A CuPy row's entry for an architecture it was not recorded on, once the key form is proven."""

    for arch, name in entries.items():
        if not isinstance(name, str):
            return None
        try:
            for extra in _cupy_extra_sources():
                if _cupy_entry(arguments, str(arch), extra) == name:
                    return _cupy_entry(arguments, target, extra)
        except Exception:
            return None
        return None
    return None


def _cached(row: dict[str, Any], device_arch: str, root: Path | None) -> bool:
    """Whether this row's route will load its image from disk on this card."""

    arguments = row["arguments"]
    target = device_arch if arguments["arch"] == "device" else str(arguments["arch"])
    entries = row.get("cache_entries")
    entry = entries.get(target) if isinstance(entries, dict) else None
    if not isinstance(entry, str) and row.get("route") == CUPY_ROUTE and isinstance(entries, dict):
        entry = _cupy_entry_elsewhere(arguments, entries, target)
    if not isinstance(entry, str):
        probe = _probe(str(row.get("route") or ""))
        if probe is None:
            return False
        try:
            entry = probe(arguments, target)
        except Exception:
            return False
        if not entry:
            return False
    path = Path(entry)
    if not path.is_absolute():
        if root is None:
            return False
        path = root / path
    return path.is_file()


def _retarget_cupy_program(row: dict[str, Any], device_arch: str) -> dict[str, Any]:
    """Replay a CuPy disk request with the program name this card's caller will use."""
    arguments = row["arguments"]
    entries = row.get("cache_entries")
    if (row.get("route") != CUPY_ROUTE or arguments["arch"] != "device"
            or not isinstance(entries, dict) or device_arch in entries):
        return row
    entry = _cupy_entry_elsewhere(arguments, entries, device_arch)
    if entry is None:
        return row
    arguments = dict(arguments, filename=str(Path(arguments["filename"]).with_name(entry + ".cu")))
    return dict(row, arguments=arguments, sha256=_key(arguments),
                cache_entries={**entries, device_arch: entry})


def _count(name: str, amount: float = 1) -> None:
    counts = _state["counts"]
    counts[name] = counts.get(name, 0) + amount


def _copy(result):
    image, mapping = result
    return image, (dict(mapping) if isinstance(mapping, dict) else mapping)


def install(plan_dir: str | Path | None = None, tag_text: str | None = None) -> bool:
    """Wrap the NVRTC funnel (once per process) and bind this run's plan file."""

    if _state["installed"]:
        return True
    if not enabled():
        return False
    compiler = _compiler()
    original = compiler._compile_using_nvrtc_no_warning
    if getattr(original, _MARK, False):
        return False  # the other package's copy already owns the funnel
    directory = plan_dir or os.environ.get(DIRECTORY_ENVIRONMENT)
    if directory is None:
        directory = Path.home() / ".cache" / "gpuwm" / "kernel-plans"
    text = default_tag_text() if tag_text is None else str(tag_text)
    try:
        identity = compiler_identity()
    except Exception:
        return False
    identity_digest = hashlib.sha256(json.dumps([SCHEMA, identity], sort_keys=True).encode()).hexdigest()[:20]
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    _state.update(identity=identity, root=Path(directory).expanduser() / identity_digest,
                  tag=tag, tag_text=text, original=original)
    _state["plan"] = _state["root"] / f"{tag}.json"

    def funnel(source, options=(), arch=None, filename="kern.cu", name_expressions=None,
               log_stream=None, cache_in_memory=False, jitify=None, method=None):
        def call():
            return original(source, options, arch, filename, name_expressions, log_stream,
                            cache_in_memory, jitify, method)

        capturing = _capturing()
        if log_stream is not None or jitify or not isinstance(source, str):
            if capturing:
                raise _Captured()
            return call()
        try:
            arguments = _arguments(source, options, arch, filename, name_expressions,
                                   cache_in_memory, method)
            key = _key(arguments)
        except Exception:
            if capturing:
                raise _Captured() from None
            return call()
        if capturing:
            # A static-plan capture (:func:`ahead`): hand back the request, compile nothing.
            raise _Captured(arguments, key)
        try:
            entry = _entry_of(filename, cache_in_memory)
            target = _device_arch() if arguments["arch"] == "device" else arguments["arch"]
        except Exception:
            entry = None
        # A pool thread's request is a warm-up, not the run's own: it is not
        # recorded into the plan.  Any thread takes back a queued compile, so
        # a pool thread never waits on a job queued behind it.
        in_pool = POOL_PREFIX in threading.current_thread().name
        owner = None
        with _lock:
            if not in_pool:
                row = _requested.setdefault(key, {"arguments": arguments, "seconds": None})
                if entry is not None:
                    row.setdefault("cache_entries", {})[target] = entry[1]
                    row["route"] = entry[0]
            pending = _flight.get(key)
            if pending is None or pending.cancel():
                if pending is not None:
                    _count("taken_back")
                owner = concurrent.futures.Future()
                owner.set_running_or_notify_cancel()
                _flight[key] = owner
        if owner is None:
            try:
                result = pending.result()
            except Exception:
                _count("failed_then_inline")
                return call()
            _count("served")
            return _copy(result)
        started = time.perf_counter()
        try:
            result = call()
        except BaseException as error:
            with _lock:
                _flight.pop(key, None)
            owner.set_exception(error)
            raise
        seconds = time.perf_counter() - started
        owner.set_result(result)
        with _lock:
            _count("inline_compiles")
            _count("inline_seconds", seconds)
            if key in _requested:
                _requested[key]["seconds"] = round(seconds, 3)
        return _copy(result)

    setattr(funnel, _MARK, True)
    funnel.__wrapped__ = original  # type: ignore[attr-defined]
    compiler._compile_using_nvrtc_no_warning = funnel
    _state["installed"] = True
    atexit.register(save)
    return True


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or document.get("compiler") != _state.get("identity")):
        return []
    rows = document.get("entries")
    if not isinstance(rows, list) or len(rows) > MAX_ENTRIES:
        return []
    good = []
    for row in rows:
        arguments = row.get("arguments") if isinstance(row, dict) else None
        if not isinstance(arguments, dict) or not isinstance(arguments.get("source"), str):
            return []
        try:
            if row.get("sha256") != _key(arguments):
                return []
        except Exception:
            return []
        good.append(row)
    return good


def load_plan() -> tuple[list[dict[str, Any]], str]:
    """This tag's plan, else the union of every plan for this compiler."""

    path = _state["plan"]
    rows = _read(path) if path is not None and path.is_file() else []
    if rows:
        return rows, "tag"
    merged: dict[str, dict[str, Any]] = {}
    root = _state.get("root")
    if root is not None and root.is_dir():
        for other in sorted(root.glob("*.json")):
            for row in _read(other):
                merged.setdefault(row["sha256"], row)
    return list(merged.values()), ("union" if merged else "none")


def _workers() -> int:
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    limit = os.environ.get(WORKERS_ENVIRONMENT, "").strip()
    if limit.isdigit() and int(limit) > 0:
        cores = min(cores, int(limit))
    return max(1, cores)


def start(device_id: int | None = None) -> dict[str, Any] | None:
    """Open a pool as wide as the usable cores and submit the plan; returns at once.

    :func:`submit` adds a package's own warm-up jobs to the same pool before
    :func:`close`; a job that compiles goes through the funnel, where it joins
    the single-flight table like any other request.
    """

    if not _state["installed"] or _state["replay"] is not None:
        return _state["replay"]
    device = _current_device() if device_id is None else int(device_id)
    if device is not None and _state["device_arch"] is None:
        import cupy

        with cupy.cuda.Device(device):
            _device_arch()
    arch = _device_arch()
    rows, source = load_plan()
    rows = [_retarget_cupy_program(row, arch) for row in rows]
    workers = _workers()
    root = _cache_root()
    pending = [row for row in rows if not _cached(row, arch, root)]
    record: dict[str, Any] = {"plan": str(_state["plan"]), "plan_source": source, "tag_text": _state["tag_text"],
                              "entries": len(rows), "device_arch": arch, "workers": workers,
                              "pool_skipped_cached": len(rows) - len(pending),
                              "pool_compiled": 0, "pool_failed": 0, "pool_seconds": 0.0,
                              "started": time.monotonic(), "last_done": None}
    rows = pending
    _state["replay"] = record
    _state["device"] = device
    original = _state["original"]
    rows.sort(key=lambda row: -(row.get("seconds") or 0.0))

    def compile_row(arguments: dict[str, Any]):
        _use(device)
        started = time.perf_counter()
        try:
            result = original(arguments["source"], tuple(arguments["options"]),
                              arch if arguments["arch"] == "device" else arguments["arch"],
                              arguments["filename"], arguments["name_expressions"], None,
                              arguments["cache_in_memory"], None, arguments["method"])
        except BaseException:
            with _lock:
                record["pool_failed"] += 1
                record["last_done"] = time.monotonic()
            raise
        with _lock:
            record["pool_compiled"] += 1
            record["pool_seconds"] = round(record["pool_seconds"] + time.perf_counter() - started, 3)
            record["last_done"] = time.monotonic()
        return result

    pool = concurrent.futures.ThreadPoolExecutor(workers, thread_name_prefix=POOL_PREFIX)
    _state["pool"] = pool
    with _lock:
        for row in rows:
            if row["sha256"] not in _flight:
                _flight[row["sha256"]] = pool.submit(compile_row, row["arguments"])
    return record


#: The thread that is capturing a static-plan request (:func:`ahead`).
_capture = threading.local()


class _Captured(BaseException):
    """A capturing thread's request, raised by the funnel instead of compiling it.

    A ``BaseException`` so no route's ``except Exception`` fallback can catch it
    and compile anyway.  ``key`` is None when nothing is to be compiled ahead:
    the route found its image on disk (:func:`refuse_while_capturing`) or the
    request is one the plan never replays (a log stream, jitify).
    """

    def __init__(self, arguments: dict[str, Any] | None = None, key: str | None = None):
        super().__init__("kernel-plan capture")
        self.arguments = arguments
        self.key = key


def _capturing() -> bool:
    return bool(getattr(_capture, "on", False))


def refuse_while_capturing() -> None:
    """Stop a capturing thread before it records anything a real load records.

    Called first by the per-process records a loader writes after its image
    exists (the physics-parameter guard's compiled set, the kernel manifest).
    A loader whose route found its image on disk reaches them without passing
    the funnel; under :func:`ahead` it must stop there, so the plan's warm-up
    never marks a module compiled or adds a manifest row the run did not make.
    """

    if _capturing():
        raise _Captured()


def ahead(call, *args: Any, **kwargs: Any) -> str:
    """Compile, on this thread, the NVRTC request ``call(*args, **kwargs)`` would make.

    THE STATIC PLAN (2026-10-17).  A recorded plan only exists from a
    configuration's second start; the first forecast on a fresh box still
    compiled its kernels one after another inside step 1 on one core (box
    door test on an RTX PRO 6000: noah 15.6 s, smag2d 12.5 s + 11.7 s,
    ysu_480 3.8 s, all serial).  ``call`` is a loader's own uncached factory
    (e.g. ``load_module.__wrapped__``): it runs on this thread up to its NVRTC
    request, where the funnel hands the request back instead of compiling
    (:class:`_Captured`).  This thread then compiles exactly that request and
    leaves the result in the single-flight table, where the run's own call --
    same source, same options, same architecture, hence the same key -- finds
    it.  The run's loader still writes its cache entry, its manifest row and
    its physics-parameter record itself, from the same bytes, at the moment it
    always did.  A request the run never makes is compiled for nothing; one
    whose bytes differ at run time (a parameter set declared later) misses the
    table and compiles inline as before.

    Returns ``"compiled"``, ``"known"`` (already compiled or compiling),
    ``"cached"`` (the route found its image; nothing to compile),
    ``"failed"`` or ``"off"``.
    """

    if not _state["installed"]:
        return "off"
    previous = _capturing()
    _capture.on = True
    captured = None
    try:
        call(*args, **kwargs)
    except _Captured as request:
        captured = request
    except Exception:
        return "failed"
    finally:
        _capture.on = previous
    record = _state["replay"]
    if captured is None or captured.key is None:
        with _lock:
            if record is not None:
                record["static_cached"] = record.get("static_cached", 0) + 1
        return "cached"
    arguments, key = captured.arguments, captured.key
    with _lock:
        if key in _flight:
            if record is not None:
                record["static_known"] = record.get("static_known", 0) + 1
            return "known"
        owner = concurrent.futures.Future()
        owner.set_running_or_notify_cancel()
        _flight[key] = owner
    arch = _device_arch()
    started = time.perf_counter()
    try:
        result = _state["original"](arguments["source"], tuple(arguments["options"]),
                                    arch if arguments["arch"] == "device" else arguments["arch"],
                                    arguments["filename"], arguments["name_expressions"], None,
                                    arguments["cache_in_memory"], None, arguments["method"])
    except BaseException as error:
        with _lock:
            if _flight.get(key) is owner:
                _flight.pop(key, None)
            if record is not None:
                record["static_failed"] = record.get("static_failed", 0) + 1
        owner.set_exception(error)
        return "failed"
    owner.set_result(result)
    with _lock:
        if record is not None:
            record["static_compiled"] = record.get("static_compiled", 0) + 1
            record["static_seconds"] = round(record.get("static_seconds", 0.0)
                                             + time.perf_counter() - started, 3)
    return "compiled"


def submit(job, *args: Any) -> concurrent.futures.Future | None:
    """Run ``job(*args)`` on the plan's pool (on the run's card); None if no pool."""

    pool = _state.get("pool")
    if pool is None:
        return None
    device = _state.get("device")

    def run():
        _use(device)
        try:
            return job(*args)
        finally:
            with _lock:
                if _state["replay"] is not None:
                    _state["replay"]["last_done"] = time.monotonic()

    return pool.submit(run)


def close() -> None:
    """No more jobs: the pool finishes what it holds and its threads exit."""

    pool = _state.pop("pool", None)
    if pool is not None:
        pool.shutdown(wait=False)


def save() -> None:
    """Merge this run's requests into its tag's plan (atomic rewrite)."""

    try:
        path = _state["plan"]
        if path is None:
            return
        with _lock:
            fresh = {key: dict(value) for key, value in _requested.items()}
        if not fresh:
            return
        document_runs = 0
        old: dict[str, dict[str, Any]] = {}
        if path.is_file():
            try:
                document_runs = int(json.loads(path.read_text(encoding="utf-8")).get("runs", 0))
            except (OSError, ValueError, TypeError, AttributeError):
                document_runs = 0
            old = {row["sha256"]: row for row in _read(path)}
        run = document_runs + 1
        entries: dict[str, dict[str, Any]] = {}
        for key, row in old.items():
            if run - int(row.get("last_run", 0)) < KEEP_RUNS:
                entries[key] = row
        for key, value in fresh.items():
            seconds = value["seconds"]
            if seconds is None and key in old:
                seconds = old[key].get("seconds")
            entries[key] = {"sha256": key, "arguments": value["arguments"], "seconds": seconds,
                            "last_run": run}
            # Each architecture's cache entry is kept: a plan baked on two cards skips on both.
            known = old.get(key, {}).get("cache_entries")
            merged = dict(known) if isinstance(known, dict) else {}
            merged.update(value.get("cache_entries") or {})
            route = value.get("route") or old.get(key, {}).get("route")
            if merged:
                entries[key]["cache_entries"] = merged
            if route:
                entries[key]["route"] = route
        rows = sorted(entries.values(), key=lambda row: -int(row["last_run"]))[:MAX_ENTRIES]
        document = {"schema": SCHEMA, "compiler": _state["identity"], "tag_text": _state["tag_text"],
                    "runs": run, "entries": rows}
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(prefix=".kernel-plan.", dir=path.parent)
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        pass


def rebind(tag_text: str) -> bool:
    """Save this process's plan and bind a new configuration tag (one worker, several profiles).

    The breakage this prevents: a ``warm-kernels`` worker that warmed several
    profiles wrote every profile's kernels under its first profile's tag
    (11 plan files for 68 profile runs on a 4090), so a later run of the other
    profiles found no plan of its own.
    """

    if not _state["installed"]:
        return False
    text = str(tag_text)
    if text == _state["tag_text"]:
        return True
    save()
    close()
    with _lock:
        _requested.clear()
        _state["counts"] = {}
    tag = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    _state.update(tag=tag, tag_text=text, plan=_state["root"] / f"{tag}.json", replay=None)
    return True


def receipt() -> dict[str, Any]:
    with _lock:
        replay = None if _state["replay"] is None else {
            key: value for key, value in _state["replay"].items() if key not in ("started", "last_done")}
        if replay is not None and _state["replay"]["last_done"] is not None:
            replay["pool_wall_seconds"] = round(_state["replay"]["last_done"] - _state["replay"]["started"], 3)
        counts = dict(_state["counts"])
        requested = len(_requested)
    if "inline_seconds" in counts:
        counts["inline_seconds"] = round(counts["inline_seconds"], 3)
    return {"schema": SCHEMA, "enabled": enabled(), "installed": _state["installed"],
            "requested_this_run": requested, "replay": replay, "calls": counts}


__all__ = ["CUPY_ROUTE", "SCHEMA", "ahead", "cache_entry", "close", "compiler_identity", "default_tag_text", "enabled",
           "install", "load_plan", "rebind", "receipt", "refuse_while_capturing", "save", "start", "submit"]

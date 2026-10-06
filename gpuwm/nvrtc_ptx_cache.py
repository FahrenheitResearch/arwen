"""Persistent images for the direct NVRTC route, with its options unchanged.

The legacy radiation units bypass RawModule to retain ``--ftz=false``.
That bypass also skips CuPy's disk cache. Store exactly the image returned
by ``compile_using_nvrtc`` (PTX or cubin), keyed by source, options, target
and the resolved compiler build. A missing, corrupt or unwritable cache
falls back to the original compiler call.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

_SCHEMA = "gpuwm.direct-nvrtc.v1"
_MAGIC = (_SCHEMA + "\n").encode("ascii")
_MAX_HEADER_BYTES = 1024 * 1024
_EVENT_LOCK = threading.Lock()
_EVENTS: list[dict] = []


def _external_dependencies(source: str, options: tuple[str, ...]) -> bool:
    # Preprocessing splices lines and treats comments as whitespace before
    # reading directives. Match the ordinary and digraph forms of '#'.
    directive_source = re.sub(r"\\\r?\n", "", source)
    directive_source = re.sub(r"/\*.*?\*/", " ", directive_source,
                              flags=re.DOTALL)
    if re.search(r"^\s*(?:#|%:)\s*(?:include|include_next|import)\b",
                 directive_source, flags=re.MULTILINE):
        return True
    # Include paths alone do not read a header, but accepting them would
    # make this key depend on unrecorded external search directories.
    return any(re.match(
        r"^(?:-I|-include|-imacros|-isystem|-iquote|-idirafter|"
        r"--(?:include-path|pre-include)(?:=|\s|$))", option)
        for option in options)


def cache_events(*, clear: bool = False) -> list[dict]:
    """Snapshot lookup and compile timings for an external startup receipt."""
    with _EVENT_LOCK:
        result = [dict(row) for row in _EVENTS]
        if clear:
            _EVENTS.clear()
    return result


def _compiler_runtime():
    import cupy
    from cupy.cuda import compiler
    return cupy, compiler


def _cache_directory(compiler):
    """Resolve CuPy's selected storage without inventing a directory default."""
    for name in ("get_cache_dir", "_get_cache_dir"):
        getter = getattr(compiler, name, None)
        if callable(getter):
            return Path(getter())
    # CuPy 14.2 moved directory selection into its pluggable storage backend.
    # The active disk instance also preserves an explicit programmatic path.
    disk_type = getattr(compiler, "_DiskKernelCacheBackend", None)
    backend = getattr(compiler, "_kernel_cache_backend", None)
    if isinstance(disk_type, type) and isinstance(backend, disk_type):
        return Path(backend._cache_dir)
    raise ValueError("CuPy has no resolved disk cache directory")


@lru_cache(maxsize=None)
def _compile_environment(compiler, arch: str, options: tuple[str, ...]):
    """CuPy's own full-build discriminator and effective target selector."""
    version = tuple(compiler._get_nvrtc_version())
    target = tuple(compiler._get_arch_for_options_for_nvrtc(arch))
    # CuPy uses this empty translation unit's banner in its own disk key:
    # getVersion alone cannot distinguish two NVRTC patch builds.
    banner = compiler._preprocess("", options, arch, "nvrtc")
    if not banner or "Compiler" not in banner:
        raise ValueError("the full NVRTC build is unresolved")
    compiler_source = Path(compiler.__file__).read_bytes()
    return {
        "nvrtc_version": list(version),
        "nvrtc_banner": banner,
        "target": list(target),
        "cupy_cache_key": str(compiler._get_cupy_cache_key()),
        "compiler_sha256": hashlib.sha256(compiler_source).hexdigest(),
    }


def _cache_identity(cupy, compiler, source, options, arch, filename):
    selected_arch = str(compiler._get_arch() if arch is None else arch)
    environment = _compile_environment(compiler, selected_arch, options)
    return {
        "schema": _SCHEMA,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "options": list(options),
        "arch": selected_arch,
        "filename": filename,
        "cupy_version": str(cupy.__version__),
        "driver_version": int(cupy.cuda.runtime.driverGetVersion()),
        **environment,
    }


def _entry_key(identity):
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _read_entry(path: Path, identity):
    try:
        with path.open("rb") as handle:
            if handle.read(len(_MAGIC)) != _MAGIC:
                return None
            raw_size = handle.read(4)
            if len(raw_size) != 4:
                return None
            size = struct.unpack("<I", raw_size)[0]
            if size > _MAX_HEADER_BYTES:
                return None
            header = json.loads(handle.read(size).decode("utf-8"))
            image = handle.read()
        if (header.get("identity") != identity or not image
                or header.get("image_sha256")
                != hashlib.sha256(image).hexdigest()):
            return None
        mapping = header.get("mapping")
        if mapping is not None and (not isinstance(mapping, dict) or any(
                not isinstance(k, str) or not isinstance(v, str)
                for k, v in mapping.items())):
            return None
        if header.get("image_type") == "str":
            image = image.decode("utf-8")
        elif header.get("image_type") != "bytes":
            return None
        return image, mapping
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def _write_entry(path: Path, identity, image, mapping):
    if not isinstance(image, (bytes, str)) or not image:
        return False
    if mapping is not None and (not isinstance(mapping, dict) or any(
            not isinstance(k, str) or not isinstance(v, str)
            for k, v in mapping.items())):
        return False
    image_bytes = image.encode("utf-8") if isinstance(image, str) else image
    header = json.dumps({
        "identity": identity,
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_type": "str" if isinstance(image, str) else "bytes",
        "mapping": mapping,
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(_MAGIC + struct.pack("<I", len(header)) + header)
            handle.write(image_bytes)
        # One replacement publishes header and image together. Readers never
        # observe a partly written entry, including concurrent first starts.
        os.replace(temporary, path)
        temporary = None
        return True
    except OSError:
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def compile_using_nvrtc(source, options=(), arch=None, filename="kern.cu",
                        name_expressions=None, log_stream=None,
                        cache_in_memory=False, jitify=None, method=None):
    """CuPy's direct call with an exact-image persistent cache, default-on.

    The forecast units use self-contained source and no template expressions.
    Other compiler modes and source including external headers take CuPy's
    original route because their additional dependencies are not keyed here.
    """
    from gpuwm.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu
    if no_local_gpu():
        raise RuntimeError(
            f"{NO_LOCAL_GPU_ENV} forbids local NVRTC compilation")
    cupy, compiler = _compiler_runtime()
    options = tuple(options)
    began = time.perf_counter()
    event = {"filename": filename, "status": "bypass", "key": None,
             "identity_seconds": 0.0, "lookup_seconds": 0.0,
             "compile_seconds": 0.0, "write_seconds": 0.0,
             "cache_written": False}
    path = identity = None
    compatible = (name_expressions is None and jitify is None
                  and method is None and not cache_in_memory
                  and os.environ.get("CUPY_CACHE_IN_MEMORY", "0") != "1"
                  and not _external_dependencies(source, options)
                  and "__FILE__" not in source
                  and "__DATE__" not in source and "__TIME__" not in source)
    if compatible:
        try:
            identity = _cache_identity(
                cupy, compiler, source, options, arch, filename)
            event["key"] = _entry_key(identity)
            path = _cache_directory(compiler) / "gpuwm-direct-nvrtc" / (
                event["key"] + ".image")
        except Exception:
            # Identity is optional; compilation remains the original route
            # when a CuPy release changes identity or storage accessors.
            path = identity = None
    event["identity_seconds"] = time.perf_counter() - began
    if path is not None:
        lookup = time.perf_counter()
        hit = _read_entry(path, identity)
        event["lookup_seconds"] = time.perf_counter() - lookup
        if hit is not None:
            event["status"] = "hit"
            result = hit
        else:
            event["status"] = "miss"
            result = None
    else:
        result = None
    if result is None:
        started = time.perf_counter()
        # Preserve the source, options and four positional arguments exactly.
        # RawModule would inject -ftz=true, which changes subnormal results.
        kwargs = {}
        for name, value in (("name_expressions", name_expressions),
                            ("log_stream", log_stream), ("jitify", jitify),
                            ("method", method)):
            if value is not None:
                kwargs[name] = value
        if cache_in_memory:
            kwargs["cache_in_memory"] = cache_in_memory
        result = compiler.compile_using_nvrtc(
            source, options, arch, filename, **kwargs)
        event["compile_seconds"] = time.perf_counter() - started
        if path is not None:
            started = time.perf_counter()
            event["cache_written"] = _write_entry(path, identity, *result)
            event["write_seconds"] = time.perf_counter() - started
    event["image_bytes"] = len(result[0])
    event["total_seconds"] = time.perf_counter() - began
    with _EVENT_LOCK:
        _EVENTS.append(event)
    return result


__all__ = ["cache_events", "compile_using_nvrtc"]

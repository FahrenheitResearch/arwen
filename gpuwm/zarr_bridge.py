"""Orchestration for the native regular-grid Zarr acquisition bridge."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from gpuwm.bridges import (accept_resolved, artifact_remedy, default_bridge_dir,
                           executable_name, packaged_bridge_dir)

CRATE_RELATIVE = "tools/zarr_bridge"
BRIDGE_ENV = "GPUWM_RW_ZARR"
ABI_MARKER = b"arwen.regular-forcing-record.v1"


def resolve_zarr_bin() -> Path:
    filename = executable_name("rw_zarr")
    override = os.environ.get(BRIDGE_ENV)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GPUWM_RW_ZARR names a missing file: {path}")
        return accept_resolved(path.resolve())
    root = Path(__file__).resolve().parent.parent
    for path in (root / CRATE_RELATIVE / "target/release" / filename,
                 root / CRATE_RELATIVE / "target/debug" / filename,
                 root / "libexec/bridges" / filename,
                 packaged_bridge_dir() / filename,
                 default_bridge_dir() / filename):
        if path.is_file():
            return accept_resolved(path.resolve())
    raise FileNotFoundError(
        "The native Zarr reader rw_zarr is not installed. "
        + artifact_remedy(env_var=BRIDGE_ENV, filename=filename,
            subject="the native Zarr reader", crate_relative=CRATE_RELATIVE, artifact="rw_zarr"))


def extract_regular_zarr(request: dict, *, request_path: Path,
                         output: Path, progress=print) -> dict:
    """Run native acquisition with bounded lifetime and progressive status."""
    executable = resolve_zarr_bin()
    request_path.write_text(json.dumps(request, sort_keys=True) + "\n", encoding="utf-8")
    report_path = request_path.with_suffix(".result.json")
    with report_path.open("w", encoding="utf-8") as report:
        process = subprocess.Popen([os.fspath(executable), "extract",
            os.fspath(request_path), os.fspath(output)],
            stdout=report, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 24 * 3600
        shown = 0
        try:
            while True:
                try:
                    _, status = process.communicate(timeout=15)
                    done = True
                except subprocess.TimeoutExpired as error:
                    status = error.stderr or ""
                    done = False
                if isinstance(status, bytes):
                    status = status.decode("utf-8", errors="replace")
                complete = status if done else status[:status.rfind("\n") + 1]
                for line in complete[shown:].splitlines():
                    progress(f"fetch: {line}")
                shown = len(complete)
                if done:
                    if process.returncode:
                        raise ValueError(f"Native Zarr acquisition failed: {status[-4000:].strip()}")
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("Native Zarr acquisition exceeded 24 hours")
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()
    document = json.loads(report_path.read_text(encoding="utf-8"))
    if document.get("schema") != "arwen.regular-forcing.v1":
        raise ValueError("Native Zarr reader returned an incompatible output schema")
    return document


@contextmanager
def regular_netcdf_record(path: Path, index: int, variables):
    """Expose one native-decoded time record without rereading the whole window."""
    names = tuple(variables)
    with tempfile.TemporaryDirectory(prefix="arwen-forcing-record-") as temporary:
        root = Path(temporary)
        result = subprocess.run(
            [os.fspath(resolve_zarr_bin()), "dump-record", os.fspath(path),
             str(index), os.fspath(root), *names],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise ValueError(f"Native forcing record decode failed: {result.stderr[-4000:].strip()}")
        document = json.loads(result.stdout)
        if (document.get("schema") != "arwen.regular-forcing-record.v1"
                or document.get("index") != index):
            raise ValueError("Native forcing record has an incompatible schema or time index")
        records = {}
        for item in document.get("variables", []):
            name, leaf = item.get("name"), item.get("file")
            shape = item.get("shape")
            if (name not in names or name in records or not isinstance(leaf, str)
                    or Path(leaf).name != leaf or item.get("dtype") != "<f8"
                    or not isinstance(shape, list)
                    or any(type(size) is not int or size <= 0 for size in shape)):
                raise ValueError("Native forcing record has invalid variable metadata")
            records[name] = (root / leaf, tuple(shape))
        if set(records) != set(names):
            raise ValueError("Native forcing record omitted requested variables")
        yield records

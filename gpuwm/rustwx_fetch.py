"""Locate and drive the vendored Rust fetch backbone (``rw_fetch``).

``gpuwm fetch --engine rust`` moves bytes through ``rw_fetch``, the
sixth binary of the vendored ``tools/rustwx`` workspace, instead of
through the stdlib ``urllib`` transports in :mod:`tools`.  What that
buys is machinery ArWen would otherwise have to reimplement:

* **16 MiB parallel whole-file range GETs** -- one 700 MB serial TCP
  stream becomes tens of concurrent ones (``wx-core client.rs:723``
  ``get_bytes_parallel_whole``);
* **``.idx`` range coalescing** -- a 561-record selection collapses to a
  handful of GETs (``idx.rs:376`` ``byte_ranges`` plus coalescing);
* **the cross-process NOMADS rate governor** (``client.rs:178-232``) --
  a lock file and shared state enforcing a 2.5 s minimum inter-request
  gap and a 15-minute node-wide cooldown when Akamai's malformed
  over-rate-limit response is fingerprinted.  It is shared state, so a
  ``rw_fetch`` subprocess and any other rusty-weather process on the
  box cooperate rather than racing each other into an IP block;
* **a two-tier disk cache** keyed by URL and byte range.

Everything durable stays in Python.  ``rw_fetch`` reports what it did;
:mod:`gpuwm.fetch` decides what that means -- the
``gpuwm-fetch-manifest-v1`` manifest, the request-identity resume
guard, quarantine-not-delete, and the record-count bars.

Like the GRIB bridges and the batch renderer, the pip wheel ships no
compiled Rust: the binary is built once from the vendored workspace
(``cargo build --release --locked --offline``) and then *pointed at*,
with the same resolution ladder as :mod:`gpuwm.bridges`:

1. the ``GPUWM_RW_FETCH`` environment variable naming the built file
   (a missing file it names is a hard error, never silently skipped);
2. a source checkout's ``tools/rustwx/target/{release,debug}``;
3. ``<root>/libexec/bridges`` beside the package;
4. ``~/.gpuwm/bridges``.

Nothing here runs cargo; resolution is read-only so ``gpuwm doctor``
can report the estate without side effects.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from gpuwm.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           cargo_build_one_liner, default_bridge_dir,
                           ensure_executable, executable_name, launchable,
                           packaged_bridge_dir, quiet_loader_errors)

#: Environment variable naming a prebuilt fetch backbone.
FETCH_ENV = "GPUWM_RW_FETCH"

#: Executable base name of the vendored fetch backbone.
FETCH_NAME = "rw_fetch"

#: The one-liner that builds it, from a checkout root.  Same workspace
#: as the batch renderer, so one cargo invocation produces both.
#: Shell-correct for this platform (no ``&&`` on Windows PowerShell).
CARGO_BUILD_HINT = cargo_build_one_liner(RUSTWX_CRATE_RELATIVE)

#: Schema of the document ``rw_fetch fetch`` prints on stdout.
FETCH_RECORD_SCHEMA = "gpuwm-rw-fetch-record-v1"

#: Schema of the document ``rw_fetch probe`` prints on stdout.
PROBE_REPORT_SCHEMA = "gpuwm-rw-fetch-probe-v1"

#: The byte transports ``--mode`` selects between.  ``auto`` is the
#: probe rule: object present and its ``.idx`` absent, malformed, or
#: provably shorter than the object => take the whole file.  No time
#: constants are involved, and both named modes are first-class.
FETCH_MODES = ("auto", "full-file", "idx-subset")

#: ``rw_fetch --abi``.  The exact-ABI marker also compiled into the
#: binary, so a stale build that still answers ``--help`` but no longer
#: emits one of these record keys is caught by
#: ``gpuwm.native_wrf_distribution`` before a distribution ships.
FETCH_ABI_MARKER = (
    "gpuwm-rw-fetch-record-v1\tmode\tmode_reason\tsource\tgrib_url\t"
    "idx_url\tidx_sha256\tidx_record_count\tselected_record_count\t"
    "ranges\tsha256")

_PROBE_TIMEOUT_S = 20


def crate_dir() -> Path:
    """The vendored Rusty Weather workspace of a source checkout."""

    return Path(__file__).resolve().parent.parent / "tools" / "rustwx"


def fetch_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for the backbone, best first."""

    filename = executable_name(FETCH_NAME)
    candidates: list[Path] = []
    override = os.environ.get(FETCH_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def find_fetch_bin() -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error:
    explicit configuration must fail loudly, not fall through to the
    Python transports and leave the operator wondering why the download
    was slow.
    """

    override = os.environ.get(FETCH_ENV)
    for candidate in fetch_candidates():
        if candidate.is_file():
            return ensure_executable(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{FETCH_ENV} names a missing file: {candidate}")
    return None


def fetch_remedy() -> str:
    """The remedy for a missing backbone, true for THIS install."""

    return artifact_remedy(
        env_var=FETCH_ENV, filename=executable_name(FETCH_NAME),
        subject="the rust fetch backbone",
        crate_relative=RUSTWX_CRATE_RELATIVE,
        one_liner=CARGO_BUILD_HINT)


def probe_fetch_bin(path: Path) -> tuple[bool, str]:
    """Launch ``path --version`` once; can this binary execute at all?

    ``rw_fetch --version`` prints its name and version and exits 0.
    That observable separates a runnable executable from an empty,
    truncated, or wrong-platform file, which refuses to launch
    (OSError) or dies with an abnormal status and no output.  The
    ``--abi`` check that follows is the stale-build guard: a binary that
    launches but emits a different record ABI is worse than a missing
    one, because it fails after the download rather than before it.

    The header gate in front of the launch is the same one every probe
    in this package uses, for the same reason: on Windows a corrupt
    image header can hang ``CreateProcess`` itself, where no timeout
    reaches.  See :func:`gpuwm.bridges.native_executable_format`.
    """

    ok, evidence = launchable(path)
    if not ok:
        return False, f"{evidence} -- corrupt, stale, or built for " \
                      "another platform"
    try:
        with quiet_loader_errors():
            probe = subprocess.run(
                [str(path), "--version"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except OSError as error:
        return False, f"exists but failed to execute: {error}"
    except subprocess.TimeoutExpired:
        return False, (f"probe invocation did not exit within "
                       f"{_PROBE_TIMEOUT_S} s")
    transcript = f"{probe.stdout or ''}{probe.stderr or ''}"
    if probe.returncode != 0 or not transcript.startswith("rw_fetch "):
        return False, (f"probe --version exited {probe.returncode} without "
                       "its version line -- corrupt, stale, or built for "
                       "another platform")
    try:
        with quiet_loader_errors():
            abi = subprocess.run(
                [str(path), "--abi"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"executes but --abi failed: {error}"
    observed = (abi.stdout or "").strip()
    if abi.returncode != 0 or observed != FETCH_ABI_MARKER:
        return False, ("executes but reports a different fetch-record ABI "
                       "than this gpuwm expects -- rebuild it: "
                       f"{CARGO_BUILD_HINT}")
    return True, (f"{transcript.strip()} -- --abi matches the "
                  "fetch-record contract")


def _run(command: list[str], *, what: str) -> dict:
    """Run one ``rw_fetch`` subcommand and parse its JSON document."""

    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace")
    except OSError as error:
        raise RuntimeError(
            f"rw_fetch failed to launch: {error}") from error
    if result.returncode != 0:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip() and not line.startswith("usage:")
                and not line.startswith(" ")]
        detail = tail[0] if tail else f"exit {result.returncode}"
        raise RuntimeError(f"rw_fetch {what}: {detail}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"rw_fetch {what} did not print a JSON record: {error}"
        ) from error


def _window(*, model: str, date: str, cycle: int,
            hours: tuple[int, ...] | None, product: str,
            source: str | None) -> list[str]:
    command = ["--model", model, "--date", date, "--cycle", f"{cycle:02d}",
               "--product", product]
    if hours is not None:
        command += ["--hours", ",".join(str(hour) for hour in hours)]
    if source is not None:
        command += ["--source", source]
    return command


def run_fetch(binary: Path, *, model: str, date: str, cycle: int,
              hours: tuple[int, ...], product: str, out: Path,
              mode: str = "auto", source: str | None = None,
              patterns: tuple[str, ...] = (),
              pattern_file: Path | None = None,
              exclusions: tuple[str, ...] = (),
              cache_dir: Path | None = None,
              keep_idx: bool = False) -> dict:
    """Download one model-run window; return the parsed fetch record.

    ``patterns`` are exact ``VAR:LEVEL`` selectors.  A selection of any
    size beyond a handful should go through ``pattern_file`` instead --
    561 selectors on a command line exceeds the Windows argv limit, and
    the file form is what the HRRR lane uses.  ``exclusions`` are
    substrings tested against the whole raw index line, which is how the
    accumulated twin of a surface field (``0-1 hr acc fcst``) is kept
    out of an otherwise exact selection.
    """

    command = [str(binary), "fetch"]
    command += _window(model=model, date=date, cycle=cycle, hours=hours,
                       product=product, source=source)
    command += ["--mode", mode, "--out", str(out)]
    for pattern in patterns:
        command += ["--var-pattern", pattern]
    if pattern_file is not None:
        command += ["--var-pattern-file", str(pattern_file)]
    for excluded in exclusions:
        command += ["--exclude-forecast-contains", excluded]
    if cache_dir is not None:
        command += ["--cache-dir", str(cache_dir)]
    if keep_idx:
        command.append("--keep-idx")
    record = _run(command, what="fetch")
    if record.get("schema") != FETCH_RECORD_SCHEMA:
        raise RuntimeError(
            f"rw_fetch printed schema {record.get('schema')!r}, expected "
            f"{FETCH_RECORD_SCHEMA!r}")
    return record


def run_probe(binary: Path, *, model: str, date: str, cycle: int,
              hours: tuple[int, ...], product: str,
              mode: str = "auto", source: str | None = None,
              cache_dir: Path | None = None) -> dict:
    """Report the transport decision per hour, moving no payload."""

    command = [str(binary), "probe"]
    command += _window(model=model, date=date, cycle=cycle, hours=hours,
                       product=product, source=source)
    command += ["--mode", mode]
    if cache_dir is not None:
        command += ["--cache-dir", str(cache_dir)]
    report = _run(command, what="probe")
    if report.get("schema") != PROBE_REPORT_SCHEMA:
        raise RuntimeError(
            f"rw_fetch printed schema {report.get('schema')!r}, expected "
            f"{PROBE_REPORT_SCHEMA!r}")
    return report


def write_pattern_file(path: Path, patterns: tuple[str, ...]) -> Path:
    """Write exact selectors one per line for ``--var-pattern-file``."""

    path.write_text(
        "".join(f"{pattern}\n" for pattern in patterns),
        encoding="utf-8", newline="\n")
    return path


__all__ = [
    "CARGO_BUILD_HINT", "FETCH_ABI_MARKER", "FETCH_ENV", "FETCH_MODES",
    "FETCH_NAME", "FETCH_RECORD_SCHEMA", "PROBE_REPORT_SCHEMA",
    "crate_dir", "fetch_candidates", "fetch_remedy", "find_fetch_bin",
    "probe_fetch_bin", "run_fetch", "run_probe", "write_pattern_file",
]

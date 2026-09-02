"""Locate and drive ``rw_ensbatch`` and ``rw_obsgrid``.

The render law says weather-field product plots come from the real Rust
renderer.  Two families of weather field had no Rust renderer to come
from, so three modules in this tree drew them with matplotlib and said so
in their own docstrings:

* **ensemble products** -- ``gpuwm/da/enprod.py``: *"the vendored rust
  renderer's catalog has no ensemble entries ... there is no second
  engine to switch to"*;
* **gridded radar observations** -- ``tools/da_level2_render.py`` and
  ``tools/da_nowcast_render.py``, which already borrow the Rust
  renderer's assets and palette while re-implementing its fill in
  matplotlib.

``rw_ensbatch`` and ``rw_obsgrid`` are those two engines.  This module is
the seam the three callers reach them through, built exactly like
:mod:`gpuwm.rustwx` and :mod:`gpuwm.rustwx_fetch`: the same five-rung
resolution ladder, the same ``--abi`` contract handshake (a binary that
launches is not a binary that speaks your grammar), the same
event-line parsing.

Neither binary is a replacement for ``rw_wrfbatch``: its
one-positional-wrfout contract is untouched, and a caller with one
deterministic run still drives it.  What these add is the two input
shapes that contract cannot express -- N members at ONE valid time, and
a file of observations that is not a forecast.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from gpuwm import bridges
from gpuwm.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           cargo_build_one_liner, default_bridge_dir,
                           executable_name, packaged_bridge_dir)

#: Environment variables naming prebuilt binaries.
ENSEMBLE_ENV = "GPUWM_RW_ENSBATCH"
OBSGRID_ENV = "GPUWM_RW_OBSGRID"

#: Executable base names.
ENSEMBLE_NAME = "rw_ensbatch"
OBSGRID_NAME = "rw_obsgrid"

#: The one-liner that builds them.  Same workspace as the batch renderer,
#: so one cargo invocation produces all three.
CARGO_BUILD_HINT = cargo_build_one_liner(RUSTWX_CRATE_RELATIVE)

#: The exact ``--abi`` lines these wrappers were written against.
#:
#: Same discipline as :data:`gpuwm.rustwx.RENDERER_ABI_MARKER`, and for
#: the same reason: "does the binary start" is not a contract check.  What
#: is listed is the vocabulary the PYTHON half parses -- the product names
#: it may pass and the event words it reads -- so changing either changes
#: the literal and every older build fails the handshake instead of
#: quietly answering the old grammar.
ENSEMBLE_ABI_MARKER = (
    "gpuwm-rw-ensbatch-products-v1\tmean\tspread\tprob\tpmm\tpaintball\t"
    "gpuwm-rw-ensbatch-events-v1\tRENDERED\tSKIPPED\tFAILED\t"
    "gpuwm-rw-ensbatch-vocabulary-v1\tMEMBERS\tCOVERAGE\tTIES")

OBSGRID_ABI_MARKER = (
    "gpuwm-rw-obsgrid-products-v1\tz-composite\tcoverage-depth\t"
    "radar-overlap\tvr-lowest\tradar-contribution\t"
    "gpuwm-rw-obsgrid-events-v1\tRENDERED\tSKIPPED\tFAILED\t"
    "gpuwm-rw-obsgrid-vocabulary-v1\tOBSERVED\tSITES")

#: Ensemble products, in the order ``all`` expands to.
ENSEMBLE_PRODUCTS = ("mean", "spread", "prob", "pmm", "paintball")

#: Ensemble fields the engine can reduce.
ENSEMBLE_FIELDS = ("refl", "uh", "precip", "t2", "wspd10")

#: Observation-grid products.
OBSGRID_PRODUCTS = ("z-composite", "coverage-depth", "radar-overlap",
                    "vr-lowest", "radar-contribution")

_PROBE_TIMEOUT_S = 20


def crate_dir() -> Path:
    """The vendored Rusty Weather workspace of a source checkout."""

    return Path(__file__).resolve().parent.parent / "tools" / "rustwx"


def _candidates(env_var: str, name: str) -> tuple[Path, ...]:
    filename = executable_name(name)
    candidates: list[Path] = []
    override = os.environ.get(env_var)
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


def _find(env_var: str, name: str) -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error
    naming the three ways out, exactly as :func:`gpuwm.rustwx
    .find_renderer` does: explicit configuration must fail loudly rather
    than fall through to a matplotlib fallback the operator did not ask
    for.
    """

    override = os.environ.get(env_var)
    for candidate in _candidates(env_var, name):
        if candidate.is_file():
            return bridges.accept_resolved(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{env_var} names a missing file: {candidate}.  Point it "
                f"at a built {name} binary, unset {env_var} to use the "
                f"vendored resolution ladder, or build it with: "
                f"{CARGO_BUILD_HINT}")
    return None


def find_ensemble_bin() -> Path | None:
    return _find(ENSEMBLE_ENV, ENSEMBLE_NAME)


def find_obsgrid_bin() -> Path | None:
    return _find(OBSGRID_ENV, OBSGRID_NAME)


def ensemble_remedy() -> str:
    return artifact_remedy(
        env_var=ENSEMBLE_ENV, filename=executable_name(ENSEMBLE_NAME),
        subject="the rust ensemble product engine",
        crate_relative=RUSTWX_CRATE_RELATIVE, one_liner=CARGO_BUILD_HINT)


def obsgrid_remedy() -> str:
    return artifact_remedy(
        env_var=OBSGRID_ENV, filename=executable_name(OBSGRID_NAME),
        subject="the rust observation-grid engine",
        crate_relative=RUSTWX_CRATE_RELATIVE, one_liner=CARGO_BUILD_HINT)


def _probe(path: Path, marker: str, name: str) -> tuple[bool, str]:
    """``--abi``: is this binary runnable, and does it speak our grammar?"""

    launchable, why = bridges.launchable(path)
    if not launchable:
        return False, why
    try:
        probe = subprocess.run([str(path), "--abi"], capture_output=True,
                               text=True, errors="replace",
                               timeout=_PROBE_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as error:
        return False, f"{name} --abi did not run: {error}"
    answer = (probe.stdout or "").strip()
    if probe.returncode != 0 or answer != marker:
        detail = answer or f"exit {probe.returncode} with no --abi line"
        return False, (
            f"launches, but --abi does not match the contract this gpuwm "
            f"expects ({detail}) -- it is a build from another checkout.  "
            f"REBUILD it from this one: {CARGO_BUILD_HINT}")
    return True, "--abi matches the contract"


def probe_ensemble_bin(path: Path) -> tuple[bool, str]:
    return _probe(path, ENSEMBLE_ABI_MARKER, ENSEMBLE_NAME)


def probe_obsgrid_bin(path: Path) -> tuple[bool, str]:
    return _probe(path, OBSGRID_ABI_MARKER, OBSGRID_NAME)


def _renderer_env() -> dict[str, str]:
    """Subprocess environment: the basemap assets, as the renderer needs."""

    from gpuwm import rustwx

    return rustwx.renderer_env()


def _read_events(stdout: str, stderr: str, subject: str
                 ) -> tuple[list[Path], list[str], list[tuple[str, str]]]:
    """``RENDERED`` / ``SKIPPED`` / ``FAILED``, all three read.

    All three, because reading only two is exactly the defect
    :func:`gpuwm.rustwx.run_renderer`'s docstring records: a product the
    engine honestly skipped became invisible to every caller, and a
    reader saw one image fewer than the catalog with nothing saying why.
    """

    written: list[Path] = []
    failures: list[str] = []
    skipped: list[tuple[str, str]] = []
    for line in (stdout or "").splitlines():
        if line.startswith("RENDERED "):
            _, _, rest = line.partition(" ")
            _, _, path = rest.partition(" ")
            if path:
                written.append(Path(path))
        elif line.startswith("SKIPPED "):
            slug, _, reason = line[len("SKIPPED "):].partition(" ")
            skipped.append((slug, f"{subject}: {reason or 'no reason given'}"))
    for line in (stderr or "").splitlines():
        if line.startswith("FAILED "):
            failures.append(f"{subject}: {line[len('FAILED '):]}")
    return written, failures, skipped


def run_ensemble_renderer(
        engine: Path, manifest: Path, *, store_root: Path, out_dir: Path,
        field: str = "refl", products: str = "mean,spread,prob,pmm,paintball",
        threshold: float | None = None, neighborhood_km: float = 0.0,
        frames: int | None = None, nan_policy: str = "mask",
        pmm_tie_rule: str = "flat-index",
        accept_status: str | None = None,
        width: int = 1200, height: int = 900,
        source_label: str | None = None,
        overlays: Path | None = None, annotate: Path | None = None,
        members: dict[int, Path] | None = None,
        ) -> tuple[list[Path], list[str], list[tuple[str, str]], dict]:
    """One ensemble batch; ``(written, failures, skipped, report)``.

    ``report`` carries what the panels stamp and a caller may want in a
    provenance record: the member roster, the missingness/coverage
    counts, the PMM tie statistics, and the paintball member->colour
    map.  It is read off the engine's own stdout rather than recomputed,
    so a caption and a record cannot disagree.

    ``members`` values may be one wrfout or the member DIRECTORY holding
    its frames; a member written with ``frames_per_outfile = 1`` is a
    series of single-frame files and all of them are that member, which
    is what ``--frames`` indexes into.
    """

    command = [
        str(engine),
        "--store-root", str(store_root),
        "--out-dir", str(out_dir),
        "--manifest", str(manifest),
        "--field", field,
        "--products", products,
        "--nan-policy", nan_policy,
        "--pmm-tie-rule", pmm_tie_rule,
        "--width", str(width),
        "--height", str(height),
    ]
    if threshold is not None:
        command.extend(("--threshold", repr(float(threshold))))
    if neighborhood_km:
        command.extend(("--neighborhood-km", repr(float(neighborhood_km))))
    if frames is not None:
        command.extend(("--frames", str(frames)))
    if accept_status:
        command.extend(("--accept-status", accept_status))
    if source_label:
        command.extend(("--source-label", source_label))
    if overlays is not None:
        command.extend(("--overlays", str(overlays)))
    if annotate is not None:
        command.extend(("--annotate", str(annotate)))
    for number, path in sorted((members or {}).items()):
        command.extend(("--member", f"{number}={path}"))

    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", env=_renderer_env())
    except OSError as error:
        return [], [f"{manifest}: {ENSEMBLE_NAME} failed to launch: {error}"], \
            [], {}
    written, failures, skipped = _read_events(
        result.stdout, result.stderr, str(manifest))
    report: dict = {}
    for line in (result.stdout or "").splitlines():
        if line.startswith("MEMBERS "):
            report["members"] = line[len("MEMBERS "):]
        elif line.startswith("COVERAGE "):
            report["coverage"] = line[len("COVERAGE "):]
        elif line.startswith("TIES "):
            report["pmm_ties"] = line[len("TIES "):]
        elif line.startswith("PAINTBALL_LEGEND\t"):
            report["paintball_legend"] = line.split("\t", 1)[1]
    if result.returncode != 0 and not failures:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        failures.append(
            f"{manifest}: {tail[-1] if tail else f'exit {result.returncode}'}")
    return written, failures, skipped, report


def run_obsgrid_renderer(
        engine: Path, obs: Path, *, out_dir: Path,
        products: str = "all", radar: int | None = None,
        rings_km: str | None = None, sites: bool = False,
        width: int = 1200, height: int = 900,
        source_label: str | None = None,
        overlays: Path | None = None, annotate: Path | None = None,
        ) -> tuple[list[Path], list[str], list[tuple[str, str]], list[dict]]:
    """One observation-grid batch; ``(written, failures, skipped, sites)``.

    The fourth element is the site roster the engine read out of the
    file -- ``[{"index", "id", "lat", "lon"}, ...]`` -- so a caller
    composing a gallery can label panels without re-opening the NetCDF.
    """

    command = [
        str(engine),
        "--obs", str(obs),
        "--out-dir", str(out_dir),
        "--products", products,
        "--width", str(width),
        "--height", str(height),
    ]
    if radar is not None:
        command.extend(("--radar", str(radar)))
    if rings_km:
        command.extend(("--rings-km", rings_km))
    if sites:
        command.append("--sites")
    if source_label:
        command.extend(("--source-label", source_label))
    if overlays is not None:
        command.extend(("--overlays", str(overlays)))
    if annotate is not None:
        command.extend(("--annotate", str(annotate)))

    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", env=_renderer_env())
    except OSError as error:
        return [], [f"{obs}: {OBSGRID_NAME} failed to launch: {error}"], [], []
    written, failures, skipped = _read_events(
        result.stdout, result.stderr, str(obs))
    roster: list[dict] = []
    for line in (result.stdout or "").splitlines():
        if line.startswith("SITES\t"):
            parts = line.split("\t")
            if len(parts) == 5:
                roster.append({"index": int(parts[1]), "id": parts[2],
                               "lat": float(parts[3]),
                               "lon": float(parts[4])})
    if result.returncode != 0 and not failures:
        tail = [line for line in (result.stderr or "").splitlines()
                if line.strip()]
        failures.append(
            f"{obs}: {tail[-1] if tail else f'exit {result.returncode}'}")
    return written, failures, skipped, roster


__all__ = [
    "CARGO_BUILD_HINT", "ENSEMBLE_ABI_MARKER", "ENSEMBLE_ENV",
    "ENSEMBLE_FIELDS", "ENSEMBLE_NAME", "ENSEMBLE_PRODUCTS",
    "OBSGRID_ABI_MARKER", "OBSGRID_ENV", "OBSGRID_NAME", "OBSGRID_PRODUCTS",
    "crate_dir", "ensemble_remedy", "find_ensemble_bin", "find_obsgrid_bin",
    "obsgrid_remedy", "probe_ensemble_bin", "probe_obsgrid_bin",
    "run_ensemble_renderer", "run_obsgrid_renderer",
]

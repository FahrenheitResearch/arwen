"""Cross-platform native HRRR preparation and stock-WRF export wrapper."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from gpuwm import explain
from gpuwm.physics_compat import (
    SINGLE_DOMAIN_PHYSICS_PROFILES,
    KESSLER_PROFILE_ID,
    MORRISON_PROFILE_ID,
    MYNN_NOAHMP_PROFILE_ID,
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID,
    MYNN_PROFILE_ID,
    MYNN_RTE_RRTMGP_PROFILE_ID,
    MYNN_RUC_PROFILE_ID,
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    NSSL2_LEGACY_RRTMG_PROFILE_ID,
    NSSL2_PROFILE_ID,
    RUC_PROFILE_ID,
    THOMPSON_LEGACY_RRTMG_PROFILE_ID,
    THOMPSON_PROFILE_ID,
    THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID,
    WSM6_PROFILE_ID,
)
from gpuwm.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
from gpuwm.hrrr_forecast import (
    HRRR_TIME_FORMAT, hrrr_source_window, resolve_cycle_flags)
from gpuwm.hrrr_native_static import sha256_file
from gpuwm.hrrr_prepared_bundle import (
    EXPERIMENT_CONFIG_NAME as PORTABLE_EXPERIMENT_CONFIG_NAME,
    HrrrBundleError, publish_hrrr_prepared_bundle)
from gpuwm.namelist_seal import namelist_extension_invariant


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# The decode pipeline owns this bound and the sealed decoder binary
# enforces it; restating a number here is how this wrapper came to
# advertise 1..64 while the pipeline refused anything over 13 -- and to
# refuse it only after the static build, minutes in.
from tools.hrrr_pipeline import MAX_PIPELINE_WORKERS  # noqa: E402


def _pipeline_workers(value: str) -> str:
    """Parse-time validation for ``--pipeline-workers``.

    argparse raises on the spelling itself, so an unrunnable request
    costs a usage message rather than a static build.
    """
    text = str(value).strip().lower()
    if text == "auto":
        return text
    try:
        workers = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"pipeline-workers must be 1..{MAX_PIPELINE_WORKERS} or auto"
        ) from None
    if workers not in range(1, MAX_PIPELINE_WORKERS + 1):
        raise argparse.ArgumentTypeError(
            f"pipeline-workers must be 1..{MAX_PIPELINE_WORKERS} or auto "
            f"(the HRRR decoder accepts no more than "
            f"{MAX_PIPELINE_WORKERS})")
    return str(workers)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _write_json_create(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _link_file_create(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError as exc:
        raise RuntimeError(
            "sealed HRRR extension requires same-filesystem hard-link reuse; "
            "refusing a quadratic copy") from exc


def _manifest_entries(path: Path) -> dict[str, str]:
    """Parse one sha256sum document without forgiving unsafe inventory."""
    entries: dict[str, str] = {}
    for line_number, raw in enumerate(
            Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw:
            continue
        try:
            digest, name = raw.split(None, 1)
        except ValueError as exc:
            raise ValueError(
                f"malformed SHA256SUMS line {line_number}: {path}") from exc
        digest = digest.lower()
        name = name.removeprefix("*").removeprefix("./")
        candidate = Path(name)
        if (len(digest) != 64
                or any(character not in "0123456789abcdef"
                       for character in digest)
                or candidate.is_absolute() or ".." in candidate.parts
                or candidate.as_posix() != name or name in entries):
            raise ValueError(
                f"unsafe or duplicate SHA256SUMS entry on line "
                f"{line_number}: {path}")
        entries[name] = digest
    if not entries:
        raise ValueError(f"SHA256SUMS inventory is empty: {path}")
    return entries


def _verify_manifest_payloads(root: Path, entries: dict[str, str]) -> None:
    for name, digest in entries.items():
        payload = root / Path(name)
        if not payload.is_file():
            raise FileNotFoundError(payload)
        if _sha256(payload) != digest:
            raise ValueError(f"manifest payload changed: {payload}")


def _source_manifest_extension(*, predecessor: Path, extended: Path,
                               source_root: Path, old_hours, new_hours,
                               cycle: datetime) -> dict[str, object]:
    old_entries = _manifest_entries(predecessor)
    new_entries = _manifest_entries(extended)
    _verify_manifest_payloads(source_root, new_entries)
    changed = sorted(
        name for name, digest in old_entries.items()
        if new_entries.get(name) != digest)
    if changed:
        raise ValueError(
            "extended source manifest changes its sealed prefix: "
            + ", ".join(changed))
    added = sorted(set(new_entries) - set(old_entries))
    lead = int(new_hours[-1])
    expected_added = sorted((
        f"hrrr.t{cycle:%H}z.wrfnatf{lead:02d}.grib2",
        f"hrrr.t{cycle:%H}z.soilf{lead:02d}.grib2",
    ))
    if added != expected_added or list(new_hours) != list(old_hours) + [lead]:
        raise ValueError(
            "extended source manifest must add exactly the next atmosphere "
            "and soil objects")
    return {
        "schema": "gpuwm-source-manifest-prefix-extension-v1",
        "predecessor_sha256": _sha256(predecessor),
        "extended_sha256": _sha256(extended),
        "old_source_forecast_hours": list(old_hours),
        "new_source_forecast_hours": list(new_hours),
        "suffix_source_forecast_hours": [int(old_hours[-1]), lead],
        "retained_entries": len(old_entries),
        "retained_entries_sha256": hashlib.sha256(
            _canonical(old_entries).encode("utf-8")).hexdigest(),
        "added_entries": [
            {"path": name, "sha256": new_entries[name]}
            for name in added
        ],
    }


_BRIDGE_HOUR = re.compile(r"^(atmosphere|soil)-f([0-9]{2})/")


def _bridge_manifest_extension(*, predecessor: Path, suffix: Path,
                               output: Path, old_hours, new_hours
                               ) -> dict[str, object]:
    """Publish a full bridge by linking the prefix and one suffix hour."""
    predecessor_manifest = predecessor / "SHA256SUMS"
    suffix_manifest = suffix / "SHA256SUMS"
    old_digest = _sha256(predecessor_manifest)
    suffix_digest = _sha256(suffix_manifest)
    old_entries = _manifest_entries(predecessor_manifest)
    suffix_entries = _manifest_entries(suffix_manifest)
    _verify_manifest_payloads(predecessor, old_entries)
    _verify_manifest_payloads(suffix, suffix_entries)
    old_hours = list(old_hours)
    new_hours = list(new_hours)
    lead = int(new_hours[-1])
    if new_hours != old_hours + [lead]:
        raise ValueError("bridge extension is not exactly one next hour")
    retained = {}
    for name, digest in old_entries.items():
        match = _BRIDGE_HOUR.match(name)
        if match is None or int(match.group(2)) not in old_hours:
            continue
        retained[name] = digest
    appended = {}
    for name, digest in suffix_entries.items():
        match = _BRIDGE_HOUR.match(name)
        if match is not None and int(match.group(2)) == lead:
            appended[name] = digest
    atmosphere_count = sum(
        name.startswith(f"atmosphere-f{lead:02d}/") for name in appended)
    soil_count = sum(
        name.startswith(f"soil-f{lead:02d}/") for name in appended)
    if atmosphere_count != 22 or soil_count != 2:
        raise ValueError(
            "suffix bridge lacks the exact 22 atmosphere and 2 soil fields")
    if set(retained) & set(appended):
        raise ValueError("suffix bridge overwrites a retained payload")
    output.mkdir(parents=True, exist_ok=False)
    for name in sorted(retained):
        _link_file_create(predecessor / name, output / name)
    for name in sorted(appended):
        _link_file_create(suffix / name, output / name)
    prior_gate = dict(
        raw.split("\t", 1)
        for raw in (predecessor / "gate.txt").read_text(
            encoding="utf-8").splitlines() if raw)
    suffix_gate = dict(
        raw.split("\t", 1)
        for raw in (suffix / "gate.txt").read_text(
            encoding="utf-8").splitlines() if raw)
    if (prior_gate.get("forecast_hours")
            != ",".join(map(str, old_hours))
            or suffix_gate.get("forecast_hours")
            != ",".join(map(str, [old_hours[-1], lead]))):
        raise ValueError("bridge gates do not describe the proven windows")
    for key in set(prior_gate) & set(suffix_gate) - {
            "forecast_hours", "series_count"}:
        if prior_gate[key] != suffix_gate[key]:
            raise ValueError(f"suffix bridge changes immutable gate field {key}")
    prior_gate["forecast_hours"] = ",".join(map(str, new_hours))
    prior_gate["series_count"] = str(len(new_hours))
    gate_path = output / "gate.txt"
    gate_path.write_text(
        "".join(f"{key}\t{value}\n" for key, value in prior_gate.items()),
        encoding="utf-8", newline="\n")
    authority = {
        "schema": "gpuwm-native-hrrr-bridge-prefix-extension-v1",
        "predecessor_sha256": old_digest,
        "suffix_sha256": suffix_digest,
        "old_source_forecast_hours": old_hours,
        "new_source_forecast_hours": new_hours,
        "suffix_source_forecast_hours": [old_hours[-1], lead],
        "retained_entries_sha256": hashlib.sha256(
            _canonical(retained).encode("utf-8")).hexdigest(),
        "added_entries_sha256": hashlib.sha256(
            _canonical(appended).encode("utf-8")).hexdigest(),
    }
    _write_json_create(output / "extension-authority.json", authority)
    output_entries = {**retained, **appended}
    output_entries["gate.txt"] = _sha256(gate_path)
    output_entries["extension-authority.json"] = _sha256(
        output / "extension-authority.json")
    manifest_path = output / "SHA256SUMS"
    with manifest_path.open("x", encoding="utf-8", newline="\n") as stream:
        for name in sorted(output_entries):
            stream.write(f"{output_entries[name]}  ./{name}\n")
    merged_digest = _sha256(manifest_path)
    _verify_manifest_payloads(output, output_entries)
    if (_sha256(predecessor_manifest) != old_digest
            or _sha256(suffix_manifest) != suffix_digest):
        raise ValueError("bridge authority changed during extension")
    return {
        "schema": "gpuwm-bridge-manifest-prefix-extension-v1",
        "predecessor_sha256": old_digest,
        "suffix_sha256": suffix_digest,
        "extended_sha256": merged_digest,
        "old_source_forecast_hours": old_hours,
        "new_source_forecast_hours": new_hours,
        "suffix_source_forecast_hours": [old_hours[-1], lead],
        "retained_entries": len(retained),
        "retained_entries_sha256": authority[
            "retained_entries_sha256"],
        "added_entries": [
            {"path": name, "sha256": appended[name]}
            for name in sorted(appended)
        ],
    }

_HRRR_COLD_START_CONTRACT = {
    WSM6_PROFILE_ID: ((), {}),
    KESSLER_PROFILE_ID: ((), {}),
    MYNN_PROFILE_ID: ((), {}),
    MYNN_RTE_RRTMGP_PROFILE_ID: ((), {}),
    MYNN_RUC_PROFILE_ID: ((), {}),
    MYNN_RUC_RTE_RRTMGP_PROFILE_ID: ((), {}),
    RUC_PROFILE_ID: ((), {}),
    NOAHMP_PROFILE_ID: ((), {}),
    MYNN_NOAHMP_PROFILE_ID: ((), {}),
    MYNN_NOAHMP_RTE_RRTMGP_PROFILE_ID: ((), {}),
    THOMPSON_PROFILE_ID: (
        ("QNICE", "QNRAIN"),
        {"ni": (0.0, 0), "nr": (0.0, 0)},
    ),
    MORRISON_PROFILE_ID: (
        ("QNRAIN", "QNICE", "QNSNOW", "QNGRAUPEL"),
        {
            "nc": (0.0, 0),
            "nr": (0.0, 0),
            "ni": (0.0, 0),
            "ns": (0.0, 0),
            "ng": (0.0, 0),
        },
    ),
    NSSL2_PROFILE_ID: (
        (
            "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
            "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
        ),
        {
            "qh": (0.0, 0),
            "qndrop": (0.0, 0),
            "qnr": (0.0, 0),
            "qni": (0.0, 0),
            "qns": (0.0, 0),
            "qng": (0.0, 0),
            "qnh": (0.0, 0),
            "qnn": (408163264.0, 1304600734),
            "qvolg": (0.0, 0),
            "qvolh": (0.0, 0),
        },
    ),
}
_HRRR_COLD_START_CONTRACT[NSSL2_LEGACY_RRTMG_PROFILE_ID] = (
    _HRRR_COLD_START_CONTRACT[NSSL2_PROFILE_ID])
# The cold-start species contract is a microphysics property, not a
# radiation one (B4-ROUTE-QUALIFICATION.md section 1, item 4), so the
# legacy-RRTMG Thompson profile reuses the Thompson entry verbatim.
_HRRR_COLD_START_CONTRACT[THOMPSON_LEGACY_RRTMG_PROFILE_ID] = (
    _HRRR_COLD_START_CONTRACT[THOMPSON_PROFILE_ID])
# The same costing one component further: the gray-zone sibling selects a
# different PBL closure and changes nothing about which species the source
# supplies or what an absent one cold-starts to.
_HRRR_COLD_START_CONTRACT[THOMPSON_SHINHONG_LEGACY_RRTMG_PROFILE_ID] = (
    _HRRR_COLD_START_CONTRACT[THOMPSON_PROFILE_ID])


def _run(command: list[str], env: dict[str, str],
         cwd: Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=REPO if cwd is None else cwd,
                   env=env)


#: The vendored decoder workspace.  Its ``.cargo/config.toml`` is what
#: replaces crates.io with ``vendor/crates-io``, and cargo discovers that
#: file by walking UP from the working directory -- never from
#: ``--manifest-path``.  Building this crate from the repository root
#: therefore bypasses the vendored registry entirely and resolves against
#: crates.io, which on an air-gapped box fails with ``no matching package
#: named '<crate>' found; location searched: crates.io index``.  Every
#: documented build (README, install.sh, install.ps1, CONTRIBUTING) already
#: ``cd``s here; this one now does too.
#:
#: It exists in a source checkout and NOT in a wheel install: a wheel's
#: ``REPO`` is ``site-packages``, which has a ``tools`` package (this
#: file is in it) and no Rust workspace at all.  Building here is
#: therefore the developer path only; see :func:`_decoder`.
BRIDGE_CRATE = REPO / "tools" / "grib1_bridge"


def _decoder(env: dict[str, str]) -> Path:
    """The HRRR decoder this preparation will launch.

    This used to resolve exactly one path --
    ``<REPO>/tools/grib1_bridge/target/release/hrrr_grib2_bridge`` --
    and, when it was absent, run ``cargo build`` in a directory that a
    wheel install does not have.  On a pip install ``REPO`` is
    ``site-packages``, so the bridge ``gpuwm setup`` had already staged
    in ``~/.gpuwm/bridges`` was never even looked at and a healthy
    machine refused with a path nobody had ever been told to create.

    The order now is the project's one bridge ladder
    (:func:`gpuwm.bridges.resolve_source_decoder`): the
    ``GPUWM_HRRR_DECODER`` override first and fail-loud, then a
    checkout's own build, then ``libexec/bridges``, then
    ``~/.gpuwm/bridges``.  ``gpuwm doctor`` calls the same function, so
    a green report and a runnable preparation are now the same claim --
    and since 1.8.9 that includes the contract check, which the resolver
    performs itself.  It used to sit outside, in doctor only, so a
    binary built before this release's contract passed this door and
    failed the report; a 12-byte text file planted at the override path
    was accepted here.  A stale bridge now refuses at both.

    Building from source stays available and stays last: it is the
    developer convenience it always was, and it can only run where the
    crate actually exists.

    ``env`` is the environment for that build subprocess.  Resolution
    itself reads the PROCESS environment, through the shared ladder --
    which is the point: ``gpuwm doctor``, ingest and this wrapper must
    all be answering about the same machine, and a private copy of the
    environment is how three answers to one question start.
    """

    from gpuwm import bridges

    # An explicit override is the operator's decision and never falls
    # through to anything -- least of all to a build, which would answer
    # a question about one exact file by producing a different one.
    overridden = bool(os.environ.get(bridges.BRIDGE_ENV["hrrr_grib2_bridge"]))
    try:
        return bridges.resolve_source_decoder("hrrr")
    except FileNotFoundError:
        if overridden or not BRIDGE_CRATE.is_dir():
            raise
    # A checkout that has simply not built the crate yet: build it where
    # its vendored registry lives, then resolve through the same ladder
    # so the answer is the one every other consumer would have got.
    _run([
        "cargo", "build", "--release", "--locked", "--offline",
        "--bin", "hrrr_grib2_bridge",
    ], env, cwd=BRIDGE_CRATE)
    return bridges.resolve_source_decoder("hrrr")


#: The dispositions :func:`_stock_wrf_export` can RETURN.  "REFUSED" left
#: this tuple with the 2026-08-04 fail-closed ruling: a requested export
#: that fails now raises instead of returning a receipt, so a PASS report
#: can never again stand over missing WRF-arm outputs.
STOCK_WRF_EXPORT_STATES = ("PASS", "SKIPPED")


def _verified_export_outputs(output: Path) -> dict[str, object]:
    """The export's declared outputs, existing and hash-manifested.

    A PASS claim without its artifacts is a lying receipt: the 2026-08-04
    battery case run lost its WRF-arm inputs to an EMFILE inside the
    converter subprocess while the step's receipt said PASS-shaped things
    around it (out/node19-shakedown/b04/B04-EXPORT-REFUSAL.txt).  So PASS
    is now earned: the manifest must exist, every file it declares must
    exist, and the manifest's own digest rides in the receipt so a later
    reader can bind the file set this step vouched for.
    """

    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "stock-WRF export exited 0 but left no manifest at "
            f"{manifest_path}; refusing to report PASS for outputs that "
            "do not exist")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest.get("files")
    if not isinstance(declared, dict) or not declared:
        raise RuntimeError(
            f"stock-WRF export manifest {manifest_path} declares no files")
    missing = sorted(name for name in declared
                     if not (output / name).is_file())
    if missing:
        raise RuntimeError(
            "stock-WRF export manifest declares file(s) missing on disk: "
            f"{missing} under {output}")
    return {
        "output": str(output),
        "manifest_sha256": _sha256(manifest_path),
        "files": sorted(declared),
    }


def _stock_wrf_export(command: list[str], env: dict[str, str], *,
                      skip: bool, required: bool,
                      output: Path) -> dict[str, object]:
    """Run the stock-WRF export: SKIPPED on request, else PASS or refuse.

    Two postures met here and the second replaced the first.  The
    original unconditional ``check=True`` took the whole wrapper down on
    any converter refusal, discarding a completed preparation; the fix
    for that swung to warn-and-continue -- and the 2026-08-04 battery
    case run measured the cost: the converter subprocess died on EMFILE
    (an environmental failure, not a representability gap), one warning
    scrolled past, the preparation PASS-logged, and the WRF arm had no
    inputs and no manifest
    (out/node19-shakedown/b04/B04-EXPORT-REFUSAL.txt).  The subprocess
    boundary flattens honest typed refusals and environmental deaths
    into one returncode, so no middle posture can tell them apart.

    Fail-closed, ruled: a REQUESTED export either PASSes with its
    declared outputs verified on disk and hash-manifested
    (:func:`_verified_export_outputs`), or this step raises.  The
    prepared cache and preparation work remain on disk either way --
    nothing re-runs the expensive part -- and the deliberate opt-out is
    ``--skip-stock-wrf-export``, recorded in the receipt as SKIPPED
    rather than dressed as success.
    """

    if skip:
        return {
            "status": "SKIPPED",
            "reason": "--skip-stock-wrf-export",
            "required": False,
        }
    try:
        _run(command, env)
    except subprocess.CalledProcessError as error:
        returncode = error.returncode
    else:
        # PASS is earned, not assumed: the declared outputs must exist
        # and hash-manifest, or this step refuses loudly.
        return {"status": "PASS", "required": required,
                **_verified_export_outputs(output)}
    # A requested export that fails is FATAL (2026-08-04 ruling, after
    # the battery case run): the converter died on EMFILE -- an
    # environmental failure a retry with a higher fd limit fixes, not a
    # representability gap -- and the old warn-and-continue posture
    # PASS-logged a preparation that had produced no WRF-arm inputs and
    # no manifest (out/node19-shakedown/b04/B04-EXPORT-REFUSAL.txt).
    # The subprocess boundary flattens the converter's honest typed
    # refusals and environmental deaths into one returncode, so the only
    # fail-closed reading of a nonzero exit is refusal.  The explicit
    # opt-outs remain: --skip-stock-wrf-export skips the attempt up
    # front and says so in the receipt; a caller whose product really
    # is only the prepared cache passes it deliberately.  The converter
    # writes its own refusal to this process's stderr, so the reason is
    # already on the terminal, verbatim; the prepared cache and the
    # preparation work above are complete and remain on disk either way.
    raise RuntimeError(
        f"stock-WRF export was requested and failed (exit {returncode}); "
        "refusing to PASS a preparation whose declared WRF-arm outputs "
        "were not produced.  The prepared cache and preparation report "
        "above are complete and stand on disk; fix the converter's "
        "refusal (quoted on stderr) and re-run, or pass "
        "--skip-stock-wrf-export to deliberately not produce WRF-arm "
        "inputs")


def _namelist_start_time(path: Path) -> datetime | None:
    """The ``&time_control`` start instant of a WRF namelist, if it has one."""

    from gpuwm.namelist_import import parse_namelist

    try:
        control = parse_namelist(path).get("time_control", {})
        parts = {
            key: int(control[f"start_{key}"][0])
            for key in ("year", "month", "day", "hour", "minute", "second")}
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None
    try:
        return datetime(**parts)
    except ValueError:
        return None


def _namelist_start_advisory(path: Path, model_start_time: datetime,
                             *, cycle: datetime, lead: int) -> None:
    """Say when the namelist's own start time disagrees with model time zero.

    The GPU single-domain route does not read this field -- it builds its
    experiment from ``--cycle`` plus ``--forecast-start-hour``, and the
    stock-WRF export below is handed the derived model start -- so a
    disagreement here changes nothing about THIS run and is not refused.
    It matters to the two consumers that do read it: a stock ``wrf.exe``
    driven from the same namelist, and ``gpuwm.hrrr_hierarchy_direct``,
    which requires the namelist start to BE model time zero and refuses
    otherwise.  A run at lead K whose namelist still says the cycle hour
    is inconsistent by exactly K hours for both of them, so it is said
    out loud, once, with the value that would fix it.
    """

    observed = _namelist_start_time(path)
    if observed is None or observed == model_start_time:
        return
    remedy = (f"Re-emit it with `gpuwm domain --forecast-start-hour {lead}`, "
              "or set its start_* keys to "
              f"{model_start_time:%Y-%m-%d %H:%M:%S}." if lead else
              "Set its start_* keys to "
              f"{model_start_time:%Y-%m-%d %H:%M:%S}.")
    explain.warn(
        f"{path.name} starts at {observed:%Y-%m-%d %H:%M:%S}, but this "
        f"preparation's model time zero is "
        f"{model_start_time:%Y-%m-%d %H:%M:%S} (cycle "
        f"{cycle:%Y-%m-%d %H:%M:%S} + {lead} h).  This run is unaffected "
        "-- the GPU route takes its clock from the cycle and the lead, "
        "not from that field -- but a stock wrf.exe driven from this "
        "namelist would start at the wrong hour, and "
        "gpuwm.hrrr_hierarchy_direct requires the two to agree and will "
        f"refuse.  {remedy}",
        "The single-domain preparer has always ignored this field, so "
        "nothing that ran before starts failing now; before leads were "
        "reachable from the wizard the two instants could not disagree.")


def _positive_finite_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "history-interval-seconds must be a number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise argparse.ArgumentTypeError(
            "history-interval-seconds must be finite and positive")
    return value


def _validated_worker_receipts(
        preparation_report: dict[str, object], *, selected_backend: str,
        requested_preprocess_workers: int | None,
        requested_pipeline_workers: str,
        final_hour: int) -> tuple[dict, dict, dict]:
    """Validate the public CPU budget and independent decoder receipt."""

    try:
        preprocess_receipt = preparation_report["preparation"][
            "preprocess_backend"]
        preprocess_worker_budget = preparation_report["preparation"][
            "preprocess_worker_budget"]
        pipeline_worker_receipt = preparation_report["pipeline"]["workers"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            "HRRR preparation omitted a worker/backend receipt") from exc
    if (preparation_report.get("status") != "PASS"
            or not isinstance(preprocess_receipt, dict)
            or preprocess_receipt.get("backend") not in {"cpu", "cuda"}):
        raise RuntimeError("HRRR preparation backend receipt is malformed")
    if (selected_backend != "auto"
            and preprocess_receipt["backend"] != selected_backend):
        raise RuntimeError(
            "HRRR preparation backend differs from the public selector")
    if (requested_preprocess_workers is not None
            and preprocess_receipt["backend"] == "cpu"
            and preprocess_receipt.get("workers")
            != requested_preprocess_workers):
        raise RuntimeError(
            "HRRR CPU preparation worker receipt differs from the request")
    if (not isinstance(preprocess_worker_budget, dict)
            or preprocess_worker_budget.get("schema")
            != "gpuwm-preprocess-worker-budget-v1"
            or preprocess_worker_budget.get("backend")
            != preprocess_receipt["backend"]
            or preprocess_worker_budget.get("applicable")
            is not (preprocess_receipt["backend"] == "cpu")
            or preprocess_worker_budget.get(
                "pipeline_decoder_workers_included") is not False):
        raise RuntimeError("HRRR preprocessing worker-budget receipt is malformed")
    if preprocess_receipt["backend"] == "cpu":
        effective_total = preprocess_worker_budget.get(
            "effective_total_native_workers")
        peak_native = preprocess_worker_budget.get(
            "peak_active_native_workers")
        jobs = preprocess_worker_budget.get("effective_allocation_per_job")
        if (isinstance(effective_total, bool)
                or not isinstance(effective_total, int)
                or effective_total < 1
                or isinstance(peak_native, bool)
                or not isinstance(peak_native, int)
                or peak_native < 1
                or effective_total != preprocess_receipt.get("workers")
                or not isinstance(jobs, list)
                or len(jobs) != 1 + 2 * final_hour
                or any(
                    not isinstance(job, dict)
                    or isinstance(job.get("effective_native_workers"), bool)
                    or not isinstance(
                        job.get("effective_native_workers"), int)
                    or not 1 <= job["effective_native_workers"] <= effective_total
                    for job in jobs)
                or peak_native > effective_total):
            raise RuntimeError(
                "HRRR CPU preprocessing worker budget was not honored")
        if (requested_preprocess_workers is not None
                and preprocess_worker_budget.get(
                    "requested_total_native_workers")
                != requested_preprocess_workers):
            raise RuntimeError(
                "HRRR CPU preprocessing budget differs from the request")
    elif preprocess_worker_budget.get("peak_active_native_workers") != 0:
        raise RuntimeError("CUDA preprocessing reported native CPU workers")
    selected_pipeline_workers = (
        pipeline_worker_receipt.get("selected")
        if isinstance(pipeline_worker_receipt, dict) else None)
    if (not isinstance(pipeline_worker_receipt, dict)
            or pipeline_worker_receipt.get("requested")
            != str(requested_pipeline_workers).strip().lower()
            or isinstance(selected_pipeline_workers, bool)
            or selected_pipeline_workers not in range(
                1, MAX_PIPELINE_WORKERS + 1)):
        raise RuntimeError("HRRR decoder worker receipt is malformed")
    return (preprocess_receipt, preprocess_worker_budget,
            pipeline_worker_receipt)


#: The receipt schema this wrapper's own producer emits.  It is a single
#: constant rather than a set because ``main`` never validates a stored
#: artifact: it runs ``tools/hrrr_single_domain_benchmark.py --prepare-only``
#: itself (below) and reads back the ``report.json`` that invocation just
#: wrote, from the same source tree.  An older-schema receipt therefore
#: cannot reach this consumer, and accepting one would only widen the gate
#: for a case that does not exist.  A prepared cache restored from an older
#: tree does not change that: the benchmark recomputes this receipt from the
#: restored state on every run, so the schema is always the running code's.
HRRR_INITIALIZATION_SCHEMA = "gpuwm-hrrr-microphysics-initialization-v4"

#: Every hydrometeor species the native HRRR analysis decodes.  The producer
#: must account for each of them exactly once, as either retained (mapped to
#: a state field) or discarded under a source-bound WRF-real policy.
_HRRR_DECODED_SOURCE_SPECIES = ("QC", "QR", "QI", "QS", "QG")

_HRRR_FINGERPRINT_KEYS = frozenset({
    "shape", "dtype", "sha256", "nonzero_mask_sha256", "nonzero_count",
    "minimum", "maximum",
})


def _validated_retention_evidence(
        initialization: dict[str, object]) -> dict[str, object]:
    """Bind the producer's analyzed-hydrometeor correspondence evidence.

    Validated as fields, not as a version string: the species partition, the
    per-species decoded-source and initialized-state fingerprints, and the
    self-declared evidentiary strength of each retention claim.  Vacuity is
    reported, never refused -- a cloud-free analysis is a legitimate
    preparation, and it is the receipt's job to say that its retention claim
    for that species was not earned.
    """

    correspondence = initialization.get("source_to_state_correspondence")
    discarded = initialization.get("discarded_source_species")
    masses = initialization.get("state_mass_fields")
    evidence = initialization.get("retention_evidence")
    summary = initialization.get("retention_evidence_summary")
    disposition = initialization.get("vertical_disposition")
    if (initialization.get("source_mass_fields")
            != list(_HRRR_DECODED_SOURCE_SPECIES)
            or not isinstance(correspondence, dict)
            or not isinstance(discarded, dict)
            or not isinstance(masses, dict)
            or not isinstance(evidence, dict)
            or not isinstance(summary, dict)
            or not isinstance(disposition, dict)
            or set(correspondence) | set(discarded)
            != set(_HRRR_DECODED_SOURCE_SPECIES)
            or set(correspondence) & set(discarded)
            or set(evidence) != set(correspondence)
            or not correspondence):
        raise RuntimeError(
            "HRRR preparation omitted analyzed-hydrometeor correspondence "
            "evidence")
    from gpuwm.ingest.real import (
        validate_hrrr_hydrometeor_vertical_disposition,
    )

    vacuous = []
    proven = []
    partially_excluded = []
    excluded = []
    source_fingerprints = {}
    initialized_fingerprints = {}
    for source_name, state_name in sorted(correspondence.items()):
        mass = masses.get(state_name)
        species = evidence[source_name]
        if (not isinstance(state_name, str)
                or not isinstance(mass, dict)
                or not _HRRR_FINGERPRINT_KEYS <= set(mass)
                or mass.get("decoded_source_field") != source_name
                or not isinstance(mass.get("decoded_source"), dict)
                or not _HRRR_FINGERPRINT_KEYS <= set(mass["decoded_source"])
                or not isinstance(species, dict)
                or species.get("state_field") != state_name
                or species.get("strength") not in (
                    "PROVEN", "PARTIALLY_WRF_EXCLUDED", "WRF_EXCLUDED",
                    "VACUOUS")
                or int(species.get("source_nonzero_count", -1)) < 0
                or int(species.get("target_influencing_source_count", -1)) < 0
                or int(species.get("wrf_excluded_source_count", -1)) < 0
                or int(species.get("state_nonzero_count", -1)) < 0
                or not isinstance(
                    species.get("vertical_disposition_schema"), str)
                or not isinstance(
                    species.get("vertical_disposition_labels_sha256"), str)):
            raise RuntimeError(
                "HRRR preparation omitted analyzed-hydrometeor "
                "correspondence evidence")
        source_nonzero = int(species["source_nonzero_count"])
        if (source_nonzero != int(mass["decoded_source"]["nonzero_count"])
                or int(species["state_nonzero_count"])
                != int(mass["nonzero_count"])):
            raise RuntimeError(
                "HRRR preparation analyzed-hydrometeor retention evidence "
                f"for {source_name} disagrees with its own fingerprints")
        influencing = int(species["target_influencing_source_count"])
        wrf_excluded = int(species["wrf_excluded_source_count"])
        if influencing + wrf_excluded != source_nonzero:
            raise RuntimeError(
                "HRRR preparation analyzed-hydrometeor retention strength "
                f"for {source_name} contradicts its source partition")
        if influencing > 0 and int(mass["nonzero_count"]) == 0:
            raise RuntimeError(
                "HRRR preparation lost every nonzero analyzed "
                f"{source_name} on the way into state.{state_name}")
        if influencing == 0 and int(mass["nonzero_count"]) != 0:
            raise RuntimeError(
                "HRRR preparation created analyzed hydrometeor mass without "
                f"target support for {source_name}->state.{state_name}")
        source_fingerprints[source_name] = mass["decoded_source"]
        initialized_fingerprints[source_name] = {
            key: mass[key] for key in _HRRR_FINGERPRINT_KEYS}
        if species["strength"] == "VACUOUS":
            vacuous.append(source_name)
        elif species["strength"] == "PROVEN":
            proven.append(source_name)
        elif species["strength"] == "PARTIALLY_WRF_EXCLUDED":
            partially_excluded.append(source_name)
        else:
            excluded.append(source_name)

    try:
        validated_disposition = \
            validate_hrrr_hydrometeor_vertical_disposition(
                source_fingerprints, initialized_fingerprints, disposition)
    except ValueError as exc:
        raise RuntimeError(
            "HRRR preparation hydrometeor vertical disposition evidence "
            f"is invalid: {exc}") from exc
    for source_name, species in evidence.items():
        validated = validated_disposition["species"][source_name]
        if (species["strength"] != validated["strength"]
                or species["target_influencing_source_count"]
                != validated["target_influencing_source_count"]
                or species["wrf_excluded_source_count"]
                != validated["wrf_excluded_source_count"]
                or species["vertical_disposition_schema"]
                != validated_disposition["schema"]
                or species["vertical_disposition_labels_sha256"]
                != validated["labels_sha256"]):
            raise RuntimeError(
                "HRRR preparation analyzed-hydrometeor retention evidence "
                f"for {source_name} disagrees with its WRF support partition")
    if partially_excluded or (excluded and proven):
        expected_strength = "PARTIALLY_WRF_EXCLUDED"
    elif excluded:
        expected_strength = "WRF_EXCLUDED"
    elif not proven:
        expected_strength = "VACUOUS"
    elif vacuous:
        expected_strength = "PARTIALLY_VACUOUS"
    else:
        expected_strength = "PROVEN"
    if (summary.get("vacuous_species") != vacuous
            or summary.get("proven_species") != proven
            or summary.get("partially_excluded_species")
            != partially_excluded
            or summary.get("excluded_species") != excluded
            or summary.get("strength") != expected_strength):
        raise RuntimeError(
            "HRRR preparation analyzed-hydrometeor retention summary "
            "disagrees with its per-species evidence")
    return summary


def _validated_physics_receipt(
        preparation_report: dict[str, object], *,
        requested_profile: str) -> dict[str, object]:
    """Retain one exact runner profile and its cold-start evidence."""

    physics = preparation_report.get("physics")
    if (not isinstance(physics, dict)
            or physics.get("schema") != "gpuwm-prepared-physics-profile-v1"
            or physics.get("profile") != requested_profile):
        raise RuntimeError(
            "HRRR preparation physics receipt differs from the request")
    initialization = physics.get("hrrr_initialization")
    expected_wrf_fields, expected_state_fields = \
        _HRRR_COLD_START_CONTRACT[requested_profile]
    if (not isinstance(initialization, dict)
            or initialization.get("schema") != HRRR_INITIALIZATION_SCHEMA
            or initialization.get("source_absent_wrf_fields")
            != list(expected_wrf_fields)
            or not isinstance(
                initialization.get("state_source_absent_fields"), dict)
            or set(initialization["state_source_absent_fields"])
            != set(expected_state_fields)):
        raise RuntimeError(
            "HRRR preparation omitted deterministic cold-start evidence")
    _validated_retention_evidence(initialization)
    for name, (expected_float32, expected_uint32_bits) \
            in expected_state_fields.items():
        observed = initialization["state_source_absent_fields"][name]
        if (not isinstance(observed, dict)
                or set(observed) != {
                    "expected_float32", "expected_uint32_bits",
                    "all_exact_expected"}
                or isinstance(observed.get("expected_float32"), bool)
                or not isinstance(
                    observed.get("expected_float32"), (int, float))
                or float(observed["expected_float32"]) != expected_float32
                or isinstance(observed.get("expected_uint32_bits"), bool)
                or observed.get("expected_uint32_bits")
                != expected_uint32_bits
                or observed.get("all_exact_expected") is not True):
            raise RuntimeError(
                "HRRR preparation omitted deterministic cold-start evidence")
    return physics


def _sealed_extension(args, *, valid_time: datetime,
                      source_forecast_hours, output: Path,
                      env: dict[str, str], decoder: Path,
                      started: float,
                      namelist_invariant: dict[str, str]) -> int:
    """Prepare only the shared terminal/new-hour slab and append it."""
    from gpuwm.ingest.prepared_cache import (
        PreparedCacheReader, extend_prepared_cache,
    )

    prior = args.extend_root_preparation.resolve()
    required_prior = {
        "wrapper": prior / "public-wrapper-result.json",
        "cache": prior / "native" / "prepared-cache",
        "report": prior / "native" / "preparation-report" / "report.json",
        "bridge": prior / "native" / "native-bridge",
        "source_snapshot": prior / "native" / "source-manifest.snapshot",
        "static": prior / "native-static.npz",
        "static_receipt": prior / "native-static-receipt.json",
        "geometry": prior / "native-geometry-receipt.json",
    }
    for path in required_prior.values():
        if not path.exists():
            raise FileNotFoundError(path)
    prior_wrapper = json.loads(
        required_prior["wrapper"].read_text(encoding="utf-8"))
    old_hours = prior_wrapper.get("source_forecast_hours")
    new_hours = list(source_forecast_hours)
    if (prior_wrapper.get("status") != "PASS"
            or prior_wrapper.get("prepared_cache_contract", {}).get("mode")
            != "sealed-prefix-v1"
            or not isinstance(old_hours, list) or len(old_hours) < 2
            or old_hours != list(range(len(old_hours)))
            or new_hours != old_hours + [len(old_hours)]
            or args.forecast_start_hour != 0
            or args.run_seconds != new_hours[-1] * 3600):
        raise ValueError(
            "sealed root extension must append exactly the next hour to a "
            "zero-based sealed predecessor")
    if (prior_wrapper.get("source_cycle") != valid_time.isoformat()
            or prior_wrapper.get("physics", {}).get("profile")
            != args.physics_profile
            or float(prior_wrapper.get("history_interval_seconds", -1.0))
            != float(args.history_interval_seconds)):
        raise ValueError("sealed predecessor belongs to another run contract")
    source_manifest = args.source_manifest.resolve()
    observed_source_sha = _sha256(source_manifest)
    if observed_source_sha != args.source_manifest_sha256.lower():
        raise ValueError("source manifest digest differs from the request")
    source_proof = _source_manifest_extension(
        predecessor=required_prior["source_snapshot"],
        extended=source_manifest, source_root=args.source_root.resolve(),
        old_hours=old_hours, new_hours=new_hours, cycle=valid_time)
    prior_header = json.loads(
        (required_prior["cache"] / "header.json").read_text(
            encoding="utf-8"))
    hydrometeor_initialization = prior_header.get("metadata", {}).get(
        "hydrometeor_initialization")
    vertical_disposition = (
        hydrometeor_initialization.get("vertical_disposition")
        if isinstance(hydrometeor_initialization, dict) else None)
    if (not isinstance(hydrometeor_initialization, dict)
            or hydrometeor_initialization.get("schema")
            != "gpuwm-real-hydrometeor-correspondence-v2"
            or not isinstance(vertical_disposition, dict)
            or vertical_disposition.get("schema")
            != "gpuwm-wrf-real-hydrometeor-vertical-disposition-v1"):
        raise ValueError(
            "sealed predecessor lacks exhaustive WRF hydrometeor "
            "disposition evidence; rebuild it with the current GPUWM "
            "before extending")
    prior_identity = prior_header.get("identity")
    if (not isinstance(prior_identity, dict)
            or prior_header.get("metadata", {}).get(
                "forcing_extension_mode") != "sealed-prefix-v1"):
        raise ValueError("predecessor prepared cache is not prefix sealed")
    PreparedCacheReader(
        required_prior["cache"], expected_identity=prior_identity).verify_all()
    prior_namelist_invariant = prior_identity.get(
        "namelist_extension_invariant")
    if prior_namelist_invariant is None:
        raise ValueError(
            "sealed predecessor lacks the namelist extension invariant; "
            "rebuild it with the current GPUWM")
    if namelist_invariant != prior_namelist_invariant:
        raise ValueError(
            "native namelist changes immutable fields from the sealed "
            "predecessor")
    if _sha256(required_prior["static"]) != prior_identity.get(
            "static_cache_sha256"):
        raise ValueError("static cache differs from sealed predecessor")
    if args.domain_spec is None:
        raise ValueError("sealed HRRR extension requires --domain-spec")

    output.mkdir(parents=True, exist_ok=False)
    for role in ("static", "static_receipt", "geometry"):
        _link_file_create(required_prior[role], output / required_prior[role].name)
    native = output / "native"
    native.mkdir()
    snapshot = native / "source-manifest.snapshot"
    with snapshot.open("xb") as stream:
        stream.write(source_manifest.read_bytes())
    if _sha256(snapshot) != observed_source_sha:
        raise ValueError("source manifest changed while it was snapshotted")
    extension_work = native / "extension-work"
    extension_work.mkdir()
    suffix_start = old_hours[-1]
    lead = new_hours[-1]
    series = extension_work / f"hrrr-f{suffix_start:02d}-f{lead:02d}-series.tsv"
    rows = []
    for hour in (suffix_start, lead):
        atmosphere = args.source_root.resolve() / (
            f"hrrr.t{valid_time:%H}z.wrfnatf{hour:02d}.grib2")
        soil = args.source_root.resolve() / (
            f"hrrr.t{valid_time:%H}z.soilf{hour:02d}.grib2")
        for path in (atmosphere, soil):
            if not path.is_file():
                raise FileNotFoundError(path)
        rows.append(f"{hour}\t{atmosphere}\t{soil}\n")
    series.write_text("".join(rows), encoding="utf-8", newline="\n")
    suffix_bridge = extension_work / "native-bridge"
    suffix_cache = extension_work / "prepared-cache"
    suffix_report_root = extension_work / "preparation-report"
    benchmark = [
        sys.executable, str(REPO / "tools" / "hrrr_single_domain_benchmark.py"),
        "--bridge", str(suffix_bridge),
        "--valid-time", args.valid_time,
        "--forecast-start-hour", str(suffix_start),
        "--forecast-end-hour", str(lead),
        "--pipeline-series", str(series),
        "--pipeline-decoder", str(decoder),
        "--pipeline-signals", str(extension_work / "pipeline-signals"),
        "--pipeline-workers", str(args.pipeline_workers),
        "--source-root", str(args.source_root.resolve()),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", observed_source_sha,
        "--static-cache", str(required_prior["static"]),
        "--static-receipt", str(required_prior["static_receipt"]),
        "--namelist-input", str(args.namelist_input.resolve()),
        "--physics-profile", args.physics_profile,
        "--prepared-cache", str(suffix_cache),
        "--prepare-only", "--run-seconds", "3600",
        "--history-interval-seconds", str(args.history_interval_seconds),
        "--namelist-extension-suffix",
        "--outdir", str(suffix_report_root),
        "--domain-spec", str(args.domain_spec.resolve()),
        "--preprocess-backend", args.preprocess_backend,
    ]
    for acknowledgement in args.ack:
        benchmark.extend(("--ack", acknowledgement))
    if args.prepare_workers is not None:
        benchmark.extend(("--prepare-workers", str(args.prepare_workers)))
    if args.preprocess_workers is not None:
        benchmark.extend(("--preprocess-workers", str(args.preprocess_workers)))
    if args.cpu_preprocess_bridge is not None:
        benchmark.extend((
            "--cpu-preprocess-bridge",
            str(args.cpu_preprocess_bridge.resolve())))
    prepare_started = time.perf_counter()
    _run(benchmark, env)
    prepare_seconds = time.perf_counter() - prepare_started
    suffix_report_path = suffix_report_root / "report.json"
    suffix_report = json.loads(suffix_report_path.read_text(encoding="utf-8"))
    if (suffix_report.get("status") != "PASS"
            or suffix_report.get("source_forecast_hours")
            != [suffix_start, lead]
            or suffix_report.get("model_forcing_hours") != [0, 1]):
        raise RuntimeError("suffix preparation report has another time window")
    preprocess_receipt, preprocess_budget, pipeline_receipt = \
        _validated_worker_receipts(
            suffix_report, selected_backend=args.preprocess_backend,
            requested_preprocess_workers=args.preprocess_workers,
            requested_pipeline_workers=args.pipeline_workers, final_hour=1)
    physics_receipt = _validated_physics_receipt(
        suffix_report, requested_profile=args.physics_profile)
    merged_bridge = native / "native-bridge"
    bridge_proof = _bridge_manifest_extension(
        predecessor=required_prior["bridge"], suffix=suffix_bridge,
        output=merged_bridge, old_hours=old_hours, new_hours=new_hours)
    suffix_header = json.loads(
        (suffix_cache / "header.json").read_text(encoding="utf-8"))
    suffix_identity = suffix_header.get("identity")
    if not isinstance(suffix_identity, dict):
        raise ValueError("suffix cache has no identity")
    new_identity = deepcopy(prior_identity)
    new_identity["bridge_manifest_sha256"] = bridge_proof[
        "extended_sha256"]
    new_identity["source_manifest_sha256"] = observed_source_sha
    new_identity["namelist_sha256"] = _sha256(
        args.namelist_input.resolve())
    new_identity["namelist_extension_invariant"] = namelist_invariant
    new_identity["forcing_hours"] = list(range(lead + 1))
    new_identity["domain_config"]["run"]["run_seconds"] = float(
        args.run_seconds)
    new_source_identity = deepcopy(prior_identity["source_identity"])
    new_source_identity["source_forecast_hours"] = new_hours
    new_source_identity["model_forcing_hours"] = list(range(lead + 1))
    new_identity["source_identity"] = new_source_identity
    prior_user = deepcopy(prior_header["metadata"]["user"])
    prior_user.update({
        "last_valid_time": (valid_time + timedelta(hours=lead)).isoformat(),
        "source_forecast_hours": new_hours,
        "model_forcing_hours": list(range(lead + 1)),
        "forcing_hours": list(range(lead + 1)),
        "source_manifest_extension": source_proof,
        "bridge_manifest_extension": bridge_proof,
    })
    cache_receipt = extend_prepared_cache(
        native / "prepared-cache", predecessor=required_prior["cache"],
        suffix=suffix_cache, identity=new_identity, metadata=prior_user,
        source_manifest_extension=source_proof,
        bridge_manifest_extension=bridge_proof)
    evidence = native / "extension-evidence"
    evidence.mkdir()
    _link_file_create(
        suffix_cache / "header.json", evidence / "suffix-cache-header.json")
    _link_file_create(
        suffix_report_path, evidence / "suffix-preparation-report.json")
    _write_json_create(evidence / "source-manifest-extension.json", source_proof)
    _write_json_create(evidence / "bridge-manifest-extension.json", bridge_proof)
    suffix_record = dict(cache_receipt["suffix"])
    suffix_record.pop("path", None)
    suffix_record.update({
        "retained": False,
        "header_snapshot": str(
            (evidence / "suffix-cache-header.json").resolve()),
    })
    cache_receipt["suffix"] = suffix_record
    _write_json_create(evidence / "prepared-cache-extension.json", cache_receipt)
    prior_report = json.loads(
        required_prior["report"].read_text(encoding="utf-8"))
    combined_report = deepcopy(prior_report)
    # A predecessor report is useful only as a base for fields whose
    # authority genuinely did not change.  Its input block names the old
    # bridge, old source digest, and shorter forcing window, so copying it
    # would publish a self-contradictory PASS report.  Reconstruct every
    # input identity from the newly published artifacts instead.
    combined_input = {
        "bridge": str(merged_bridge.resolve()),
        "bridge_manifest_sha256": bridge_proof["extended_sha256"],
        "source_manifest_sha256": observed_source_sha,
        "source_cycle": valid_time.isoformat(),
        "model_start_time": valid_time.isoformat(),
        "source_forecast_hours": new_hours,
        "model_forcing_hours": list(range(lead + 1)),
        "forcing_hours": list(range(lead + 1)),
        "native_static_cache": str((output / "native-static.npz").resolve()),
        "native_static_cache_sha256": _sha256(output / "native-static.npz"),
        "native_static_receipt": str(
            (output / "native-static-receipt.json").resolve()),
        "namelist_input_sha256": _sha256(args.namelist_input.resolve()),
        "native_physics_profile": args.physics_profile,
    }
    combined_report.update({
        "status": "PASS",
        "source_cycle": valid_time.isoformat(),
        "model_start_time": valid_time.isoformat(),
        "source_forecast_hours": new_hours,
        "model_forcing_hours": list(range(lead + 1)),
        "forcing_hours": list(range(lead + 1)),
        "source_identity": new_source_identity,
        "input": combined_input,
        "prepared_cache": cache_receipt,
        "extension": {
            "schema": "gpuwm-root-preparation-extension-v1",
            "predecessor": str(prior),
            "source_manifest": source_proof,
            "bridge_manifest": bridge_proof,
            "suffix_cache_content_sha256": suffix_header[
                "content_sha256"],
        },
        "physics": suffix_report["physics"],
        "pipeline": suffix_report.get("pipeline"),
        "preparation": suffix_report.get("preparation"),
    })
    preparation_report_path = native / "preparation-report" / "report.json"
    _write_json_create(preparation_report_path, combined_report)
    export_command = [
        sys.executable, "-m", "gpuwm.wrf_direct",
        "--prepared-cache", str(native / "prepared-cache"),
        "--static-cache", str(output / "native-static.npz"),
        "--geometry-receipt", str(output / "native-geometry-receipt.json"),
        "--output", str(output / "wrf-native-input"),
        "--valid-time", valid_time.strftime("%Y-%m-%d_%H:%M:%S"),
        "--physics-profile", args.physics_profile,
    ]
    for acknowledgement in args.ack:
        export_command.extend(("--ack", acknowledgement))
    export_started = time.perf_counter()
    stock_wrf_export = _stock_wrf_export(
        export_command, env, skip=args.skip_stock_wrf_export,
        required=args.require_stock_wrf_export,
        output=output / "wrf-native-input")
    export_seconds = time.perf_counter() - export_started
    # The two-frame cache was construction scratch.  Its header/report are
    # retained above; every old payload and the new decoded hour are already
    # hard-linked into the immutable products, so retaining the full scratch
    # state would waste one complete root state per leg.
    shutil.rmtree(extension_work)
    result = {
        "status": "PASS",
        "stock_wrf_export": stock_wrf_export,
        "wrf_input": (str(output / "wrf-native-input")
                      if stock_wrf_export["status"] == "PASS" else None),
        "source_cycle": valid_time.isoformat(),
        "model_start_time": valid_time.isoformat(),
        "source_forecast_hours": new_hours,
        "model_forcing_hours": list(range(lead + 1)),
        "forcing_hours": list(range(lead + 1)),
        "history_interval_seconds": float(args.history_interval_seconds),
        "static_seconds": 0.0,
        "prepare_seconds": prepare_seconds,
        "export_seconds": export_seconds,
        "total_seconds": time.perf_counter() - started,
        "preprocess_backend": preprocess_receipt,
        "preprocess_worker_budget": preprocess_budget,
        "pipeline_decoder_workers": pipeline_receipt,
        "preparation_report": str(preparation_report_path),
        "physics": physics_receipt,
        "prepared_cache_contract": {
            "mode": "sealed-prefix-v1",
            "operation": "extend-one-hour",
            "predecessor": str(prior),
            "content_sha256": cache_receipt["content_sha256"],
        },
    }
    _write_json_create(output / "public-wrapper-result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    static = parser.add_mutually_exclusive_group(required=True)
    static.add_argument("--geog-root", type=Path)
    static.add_argument("--static-cache", type=Path)
    parser.add_argument("--static-receipt", type=Path)
    parser.add_argument("--domain-spec", type=Path)
    parser.add_argument("--namelist-input", type=Path, required=True)
    parser.add_argument(
        "--wps-namelist", type=Path,
        help=("YOUR &geogrid namelist.wps, bound into the portable "
              "bundle in place of the one this route renders from the "
              "domain it was given.  The portable bundle -- proof.json, "
              "a role-keyed source manifest, the experiment authority "
              "and namelist.wps -- is published on every run, so this "
              "flag changes WHICH namelist is bound, never WHETHER the "
              "bundle exists"))
    parser.add_argument(
        "--physics-profile",
        choices=SINGLE_DOMAIN_PHYSICS_PROFILES,
        default=ROUTE_DEFAULT_PHYSICS_PROFILE,
        help="explicit GPUWM HRRR physics/runtime contract",
    )
    parser.add_argument(
        "--ack", action="append", default=[],
        help="registry-owned expert physics acknowledgement id; repeatable")
    parser.add_argument(
        "--cycle",
        help="the HRRR CYCLE this window comes from, YYYY-MM-DD_HH:MM:SS "
             "(UTC).  Model time zero is cycle + --forecast-start-hour and "
             "is never typed")
    parser.add_argument(
        "--valid-time",
        help="deprecated spelling of --cycle (it always meant the cycle "
             "on this command); accepted unchanged for v1.4.0 scripts")
    parser.add_argument(
        "--forecast-start-hour", type=int, default=0,
        help="absolute HRRR cycle-relative source lead used for model time zero")
    parser.add_argument(
        "--forecast-end-hour", type=int,
        help="inclusive absolute source lead; must cover --run-seconds exactly")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--sealed-prepared-cache", action="store_true",
        help=("opt in to a prefix-sealed root preparation for operational "
              "one-hour extension"))
    parser.add_argument(
        "--extend-root-preparation", type=Path,
        help=("sealed predecessor root; decode only its terminal overlap and "
              "the one newly arrived hour"))
    parser.add_argument("--run-seconds", type=int, default=43_200)
    parser.add_argument(
        "--history-interval-seconds", type=_positive_finite_seconds,
        default=300.0,
        help=("future GPUWM history cadence bound into the prepared-cache "
              "identity; preparation itself writes no history frames"),
    )
    parser.add_argument(
        "--pipeline-workers", type=_pipeline_workers, default="8",
        help=(f"HRRR decode workers: 1..{MAX_PIPELINE_WORKERS} or auto "
              "(the decoder's own bound)"))
    parser.add_argument("--prepare-workers", type=int)
    parser.add_argument(
        "--preprocess-backend", choices=("cuda", "cpu", "auto"),
        default="cuda")
    parser.add_argument("--preprocess-workers", type=int)
    parser.add_argument("--cpu-preprocess-bridge", type=Path)
    export = parser.add_mutually_exclusive_group()
    export.add_argument(
        "--skip-stock-wrf-export", action="store_true",
        help="do not write the stock-WRF wrfinput/wrfbdy export at all.  "
             "The GPU hierarchy consumes the prepared cache directly and "
             "never reads that export; skipping it is the right answer "
             "whenever stock wrf.exe is not the consumer")
    export.add_argument(
        "--require-stock-wrf-export", action="store_true",
        help="accepted for compatibility; a requested export that fails "
             "is ALWAYS this command's failure since the 2026-08-04 "
             "fail-closed ruling (a warn-and-continue posture PASS-logged "
             "a preparation with no WRF-arm outputs).  The explicit "
             "opt-out is --skip-stock-wrf-export")
    explain.add_explain_flag(parser)
    return parser


def _require_microphysics_tables(profile: str) -> None:
    """Refuse an unstaged lookup-table set before any paid stage runs.

    Presence only, in one sentence naming ``gpuwm fetch-tables``.  The
    exact sizes and SHA-256 digests are re-checked by the root
    preparation's own profile binding
    (``tools/hrrr_single_domain_benchmark._microphysics_table_
    authority``) and again at load, so this is the early half of a gate
    that still fails closed later, not a second opinion.
    """

    from gpuwm.core.thompson_contract import (
        CLASSIC_TABLE_ASSETS, MP_PHYSICS as THOMPSON_MP_PHYSICS)
    from gpuwm.physics_compat import single_domain_runtime_switches
    from gpuwm.table_assets import require_thompson_tables

    switches = single_domain_runtime_switches(profile)
    if int(switches["mp_physics"]) == THOMPSON_MP_PHYSICS:
        require_thompson_tables(assets=CLASSIC_TABLE_ASSETS)


def main(argv: list[str] | None = None) -> int:
    """This tool's front door; the body is :func:`_prepare_from_argv`.

    Same reason as `gpuwm.cli.main`: `--explain` lives in module state
    because library warnings have no `args` in reach, and this function
    is called in-process (the tests import this module and call it), so
    a stamp that outlives the call leaks an explanation layer into the
    next one.  `explain_scope` bounds it to this invocation.
    """

    with explain.explain_scope(False):
        return _prepare_from_argv(argv)


def _prepare_from_argv(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    explain.set_explain(explain.explain_enabled(args))
    cycle, _legacy = resolve_cycle_flags(
        args.cycle, args.valid_time,
        tool="prepare_hrrr_wrf", legacy_means="the HRRR cycle",
        warn=explain.warn)
    # Kept under its old name inside this function: every receipt key and
    # every downstream flag below already spells this instant the cycle.
    valid_time = cycle
    source_forecast_hours = hrrr_source_window(
        cycle=valid_time, start_hour=args.forecast_start_hour,
        run_seconds=args.run_seconds, end_hour=args.forecast_end_hour)
    model_forcing_hours = tuple(range(len(source_forecast_hours)))
    final_hour = model_forcing_hours[-1]
    model_start_time = valid_time + timedelta(
        hours=source_forecast_hours[0])
    # The selected suite's microphysics lookup tables, resolved and
    # byte-checked BEFORE the decoder runs and before a single source
    # file is read.  This preparation used to resolve no tables at all --
    # its only table gate lived in the benchmark runner and fired for one
    # profile behind two environment variables -- so an mp8 suite reached
    # GPU setup unstaged, minutes of paid work later.  Naming it here also
    # removed the reason this route's default had to be a suite that
    # needs no tables, which is why the default could be flipped to a
    # full-radiation one at all.
    _require_microphysics_tables(args.physics_profile)
    if args.prepare_workers is not None and args.prepare_workers not in range(1, 33):
        raise ValueError("prepare-workers must be between 1 and 32")
    if args.preprocess_workers is not None and args.preprocess_workers < 1:
        raise ValueError("preprocess-workers must be positive")
    if args.preprocess_backend == "cuda" and args.preprocess_workers is not None:
        raise ValueError(
            "preprocess-workers requires preprocess-backend cpu or auto")
    if (args.preprocess_backend != "cpu"
            and args.cpu_preprocess_bridge is not None):
        raise ValueError(
            "cpu-preprocess-bridge requires preprocess-backend cpu")
    if args.extend_root_preparation is not None \
            and not args.sealed_prepared_cache:
        raise ValueError(
            "extend-root-preparation requires --sealed-prepared-cache")
    if args.sealed_prepared_cache and args.forecast_start_hour != 0:
        raise ValueError(
            "sealed prepared-cache orchestration requires forecast start 0")
    # --pipeline-workers is validated by its argparse type, at parse
    # time, against the decode pipeline's own ceiling.
    if len(args.source_manifest_sha256) != 64:
        raise ValueError("source-manifest-sha256 must contain 64 hexadecimal digits")
    int(args.source_manifest_sha256, 16)
    source_root = args.source_root.resolve()
    source_manifest = args.source_manifest.resolve()
    namelist_input = args.namelist_input.resolve()
    for path in (source_root, source_manifest, namelist_input):
        if not path.exists():
            raise FileNotFoundError(path)
    namelist_invariant = (
        namelist_extension_invariant(
            namelist_input, cycle=valid_time, run_seconds=args.run_seconds)
        if args.sealed_prepared_cache else None)
    if args.geog_root is not None and args.domain_spec is None:
        raise ValueError("domain-spec is required with geog-root")
    if args.static_cache is not None and args.static_receipt is None:
        raise ValueError("static-receipt is required with static-cache")
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output root: {output}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    decoder = _decoder(env)
    started = time.perf_counter()
    if args.sealed_prepared_cache:
        observed_manifest = _sha256(source_manifest)
        if observed_manifest != args.source_manifest_sha256.lower():
            raise ValueError("source manifest digest differs from the request")
    if args.extend_root_preparation is not None:
        if args.wps_namelist is not None:
            # A sealed extension joins one more hour onto a root that
            # already published its own authorities; binding a second,
            # different namelist beside them would leave two proofs
            # describing one bundle.  Refuse rather than silently drop
            # the flag.
            raise ValueError(
                "--wps-namelist binds a namelist into a NEW preparation's "
                "portable bundle; a sealed extension inherits the root's "
                "own authorities and must not republish them")
        return _sealed_extension(
            args, valid_time=valid_time,
            source_forecast_hours=source_forecast_hours,
            output=output, env=env, decoder=decoder, started=started,
            namelist_invariant=namelist_invariant)
    output.mkdir(parents=True)

    geometry_receipt = output / "native-geometry-receipt.json"
    if args.geog_root is not None:
        static_cache = output / "native-static.npz"
        static_receipt = output / "native-static-receipt.json"
        _run([
            sys.executable, str(REPO / "tools" / "hrrr_build_native_static.py"),
            "--geog-root", str(args.geog_root.resolve()),
            "--domain-spec", str(args.domain_spec.resolve()),
            "--output", str(static_cache), "--receipt", str(static_receipt),
        ], env)
    else:
        static_cache = args.static_cache.resolve()
        static_receipt = args.static_receipt.resolve()
        for path in (static_cache, static_receipt):
            if not path.is_file():
                raise FileNotFoundError(path)
        # A prefix-sealed root is a self-contained predecessor authority.
        # Extension cannot depend on the operator retaining two external
        # paths which were not part of that root, and copying them once per
        # hourly generation would defeat the hard-link-only storage contract.
        # Publish the exact verified files under the same canonical names the
        # geog-built route uses.  The same-filesystem refusal is intentional:
        # every future generation links these bytes again.
        #
        # A PORTABLE bundle needs the same thing for a different reason:
        # the forecast front door resolves the static cache relative to
        # --prepared-root, and a root whose static geography lives on
        # some other path the operator happened to pass is not a bundle
        # anyone else can bind.  Same link, same canonical names, same
        # bytes -- and a reused external static stays reused.
        #
        # Unconditional, because the portable bundle is published on
        # every run now.  It used to be gated on the two flags that
        # implied a portable product, which left the default --static-cache
        # run publishing nothing: the bundle writer refuses a static cache
        # outside the root ("is outside the bundle root"), so the one
        # decision that made the authority publishable was taken only when
        # someone had already asked for it by name.
        sealed_static = output / "native-static.npz"
        sealed_receipt = output / "native-static-receipt.json"
        _link_file_create(static_cache, sealed_static)
        _link_file_create(static_receipt, sealed_receipt)
        static_cache = sealed_static
        static_receipt = sealed_receipt
    geometry_command = [
        sys.executable,
        str(REPO / "tools" / "write_hrrr_native_geometry_receipt.py"),
        "--static-cache", str(static_cache),
        "--hrrr-static-receipt", str(static_receipt),
        "--output", str(geometry_receipt),
    ]
    if args.domain_spec is not None:
        geometry_command.extend(("--domain-spec", str(args.domain_spec.resolve())))
    _run(geometry_command, env)
    static_seconds = time.perf_counter() - started

    native = output / "native"
    native.mkdir()
    series = native / (
        f"hrrr-f{source_forecast_hours[0]:02d}-"
        f"f{source_forecast_hours[-1]:02d}-series.tsv")
    rows = []
    for hour in source_forecast_hours:
        atmosphere = source_root / (
            f"hrrr.t{valid_time:%H}z.wrfnatf{hour:02d}.grib2")
        soil = source_root / f"hrrr.t{valid_time:%H}z.soilf{hour:02d}.grib2"
        for path in (atmosphere, soil):
            if not path.is_file():
                raise FileNotFoundError(path)
        rows.append(f"{hour}\t{atmosphere}\t{soil}\n")
    series.write_text("".join(rows), encoding="utf-8")
    _namelist_start_advisory(
        namelist_input, model_start_time, cycle=valid_time,
        lead=source_forecast_hours[0])
    benchmark = [
        sys.executable, str(REPO / "tools" / "hrrr_single_domain_benchmark.py"),
        "--bridge", str(native / "native-bridge"),
        "--cycle", valid_time.strftime(HRRR_TIME_FORMAT),
        "--forecast-start-hour", str(source_forecast_hours[0]),
        "--forecast-end-hour", str(source_forecast_hours[-1]),
        "--pipeline-series", str(series),
        "--pipeline-decoder", str(decoder),
        "--pipeline-signals", str(native / "pipeline-signals"),
        "--pipeline-workers", str(args.pipeline_workers),
        "--source-root", str(source_root),
        "--source-manifest", str(source_manifest),
        "--source-manifest-sha256", args.source_manifest_sha256.lower(),
        "--static-cache", str(static_cache),
        "--static-receipt", str(static_receipt),
        "--namelist-input", str(namelist_input),
        "--physics-profile", args.physics_profile,
        "--prepared-cache", str(native / "prepared-cache"),
        "--prepare-only", "--run-seconds", str(args.run_seconds),
        "--history-interval-seconds", str(args.history_interval_seconds),
        "--outdir", str(native / "preparation-report"),
    ]
    # Unconditional.  Only the benchmark holds the experiment tables this
    # route builds in code, so only it can render the authority a
    # config-driven stage will bind -- and every prepared tree needs one,
    # because every prepared tree is something `gpuwm sim` should be able
    # to run.  It used to be published only when --wps-namelist was
    # supplied, which is how a finished tree could exist that no forecast
    # front door would accept.
    benchmark.extend((
        "--publish-experiment-config",
        str(output / PORTABLE_EXPERIMENT_CONFIG_NAME)))
    if args.sealed_prepared_cache:
        benchmark.append("--sealed-prepared-cache")
    for acknowledgement in args.ack:
        benchmark.extend(("--ack", acknowledgement))
    if args.prepare_workers is not None:
        benchmark.extend(("--prepare-workers", str(args.prepare_workers)))
    benchmark.extend(("--preprocess-backend", args.preprocess_backend))
    if args.preprocess_workers is not None:
        benchmark.extend(("--preprocess-workers", str(args.preprocess_workers)))
    if args.cpu_preprocess_bridge is not None:
        benchmark.extend((
            "--cpu-preprocess-bridge",
            str(args.cpu_preprocess_bridge.resolve())))
    if args.domain_spec is not None:
        benchmark.extend(("--domain-spec", str(args.domain_spec.resolve())))
    prepare_started = time.perf_counter()
    _run(benchmark, env)
    prepare_seconds = time.perf_counter() - prepare_started
    preparation_report_path = native / "preparation-report" / "report.json"
    preparation_report = json.loads(
        preparation_report_path.read_text(encoding="utf-8"))
    if args.sealed_prepared_cache:
        snapshot = native / "source-manifest.snapshot"
        with snapshot.open("xb") as stream:
            stream.write(source_manifest.read_bytes())
        if _sha256(snapshot) != args.source_manifest_sha256.lower():
            raise ValueError(
                "source manifest changed during sealed preparation")
    observed_history_interval = preparation_report.get(
        "history_interval_seconds")
    if (isinstance(observed_history_interval, bool)
            or not isinstance(observed_history_interval, (int, float))
            or float(observed_history_interval)
            != float(args.history_interval_seconds)):
        raise RuntimeError(
            "HRRR prepared-cache history cadence differs from the request")
    (preprocess_receipt, preprocess_worker_budget,
     pipeline_worker_receipt) = _validated_worker_receipts(
         preparation_report,
         selected_backend=args.preprocess_backend,
         requested_preprocess_workers=args.preprocess_workers,
         requested_pipeline_workers=args.pipeline_workers,
         final_hour=final_hour)
    physics_receipt = _validated_physics_receipt(
        preparation_report, requested_profile=args.physics_profile)

    bridge_manifest = native / "native-bridge" / "SHA256SUMS"
    export_started = time.perf_counter()
    export_command = [
        sys.executable, "-m", "gpuwm.wrf_direct",
        "--prepared-cache", str(native / "prepared-cache"),
        "--static-cache", str(static_cache),
        "--geometry-receipt", str(geometry_receipt),
        "--output", str(output / "wrf-native-input"),
        # gpuwm.wrf_direct's --valid-time is the valid time of the state
        # being exported, which IS model time zero.  That flag is not part
        # of the cycle/model-start collision: an exported wrfinput has
        # exactly one valid time and no lead to be read against.
        "--valid-time", model_start_time.strftime(HRRR_TIME_FORMAT),
        "--physics-profile", args.physics_profile,
    ]
    for acknowledgement in args.ack:
        export_command.extend(("--ack", acknowledgement))
    stock_wrf_export = _stock_wrf_export(
        export_command, env,
        skip=args.skip_stock_wrf_export,
        required=args.require_stock_wrf_export,
        output=output / "wrf-native-input")
    export_seconds = time.perf_counter() - export_started

    # ---- the portable half, published on every run ---------------------
    # Everything above this line is the certified native preparation and
    # is untouched.  This adds the documents a config-driven forecast
    # stage binds, from values already on disk: the cache's own source
    # identity, the preparation report's preprocessing receipt, and
    # digests of files this wrapper just wrote.
    #
    # It used to be gated on --wps-namelist, because the caller's
    # namelist was the one authority this route could not produce.  It
    # can now: the route holds the experiment those numbers come from,
    # and the bundle writer renders a namelist.wps from it and checks it
    # with the forecast stage's own geometry validator.  So publication
    # is the DEFAULT, and --wps-namelist means "bind mine instead of the
    # rendered one".  MEASURED 2026-08-18: a prep that exited 0 with
    # every wrfinput/wrfbdy on disk left a tree `gpuwm sim` refused,
    # because nothing here had published the digests it binds.
    #
    # A publication that cannot be completed does not fail the
    # preparation -- the native tree it would have described is finished
    # and valid, and the native benchmark runs it.  The reason is
    # recorded and printed instead, and `gpuwm sim` reads it back off
    # this receipt when it refuses the tree, so the chain says the same
    # thing at both ends.
    portable_bundle = None
    portable_bundle_refusal = None
    try:
        # Read inside the guard on purpose: every input the publication
        # needs is an input the publication can fail on, and a run that
        # cannot publish still has a native tree worth keeping.
        cache_header = json.loads(
            (native / "prepared-cache" / "header.json").read_text(
                encoding="utf-8"))
        portable_bundle = publish_hrrr_prepared_bundle(
            output_root=output,
            prepared_cache=native / "prepared-cache",
            static_cache=static_cache, static_receipt=static_receipt,
            geometry_receipt=geometry_receipt,
            bridge_manifest=bridge_manifest,
            namelist_input=namelist_input,
            wps_namelist=(None if args.wps_namelist is None
                          else args.wps_namelist.resolve()),
            source_manifest=source_manifest,
            experiment_config=output / PORTABLE_EXPERIMENT_CONFIG_NAME,
            source_cycle=valid_time,
            source_forecast_hours=source_forecast_hours,
            model_forcing_hours=model_forcing_hours,
            # The VALIDATED receipt, not the raw report field: this is
            # the one the wrapper already proved describes the backend
            # that was asked for.
            preprocessing=preprocess_receipt,
            source_identity=cache_header["identity"]["source_identity"],
            physics_profile=args.physics_profile,
            expert_acknowledgements=tuple(args.ack),
            domain_spec=(args.domain_spec.resolve()
                         if args.domain_spec is not None else None),
            namelist_extension_invariant=namelist_invariant)
    except (HrrrBundleError, OSError, TypeError, ValueError, KeyError) as error:
        portable_bundle_refusal = (
            f"{type(error).__name__}: {error}")
        print(
            "prepare-hrrr-wrf: the native preparation finished, and the "
            "portable authorities a config-driven forecast binds "
            f"(proof.json, {PORTABLE_EXPERIMENT_CONFIG_NAME}, namelist.wps) "
            f"were NOT published: {portable_bundle_refusal}\n"
            "  what still works: the native bundle in this output root, run "
            "by tools/hrrr_single_domain_benchmark.py --prepared-cache\n"
            "  what does not: `gpuwm sim` on this root, which binds those "
            "authorities and will say so",
            file=sys.stderr)

    result = {
        "status": "PASS",
        "stock_wrf_export": stock_wrf_export,
        # The three pins a forecast stage passes beside --prepared-root,
        # or None with the reason beside it when the publication could
        # not be completed.  Same rule as bridge_manifest_sha256 below:
        # each stage publishes the digests the stage after it must bind
        # -- and when it cannot, it records why, here, where `gpuwm sim`
        # reads it back so both ends of the chain say the same thing.
        "portable_bundle": portable_bundle,
        "portable_bundle_refusal": portable_bundle_refusal,
        # Named only when it exists.  A path in a receipt is a claim
        # that the artifact is there; a skipped or refused export must
        # not leave one behind for a reader to trip over.
        "wrf_input": (str(output / "wrf-native-input")
                      if stock_wrf_export["status"] == "PASS" else None),
        "source_cycle": valid_time.isoformat(),
        "model_start_time": model_start_time.isoformat(),
        # What the single-domain FORECAST stage asks for next and nothing
        # printed.  Same rule the hierarchy follows with its own
        # preparation_receipt_sha256: each stage publishes the digest the
        # stage after it must bind, so the printed chain can be walked
        # without hashing a file by hand.  Named only when the artifact is
        # there -- a digest in a receipt is a claim that the file exists.
        "bridge_manifest_sha256": (
            sha256_file(bridge_manifest) if bridge_manifest.is_file()
            else None),
        "source_forecast_hours": list(source_forecast_hours),
        "model_forcing_hours": list(model_forcing_hours),
        # Backward-compatible alias: exporter/cache forcing is model-relative.
        "forcing_hours": list(model_forcing_hours),
        "history_interval_seconds": float(args.history_interval_seconds),
        "static_seconds": static_seconds,
        "prepare_seconds": prepare_seconds,
        "export_seconds": export_seconds,
        "total_seconds": time.perf_counter() - started,
        "preprocess_backend": preprocess_receipt,
        "preprocess_worker_budget": preprocess_worker_budget,
        "pipeline_decoder_workers": pipeline_worker_receipt,
        "preparation_report": str(preparation_report_path),
        "physics": physics_receipt,
    }
    if args.sealed_prepared_cache:
        prepared_header = json.loads(
            (native / "prepared-cache" / "header.json").read_text(
                encoding="utf-8"))
        if prepared_header.get("metadata", {}).get(
                "forcing_extension_mode") != "sealed-prefix-v1":
            raise RuntimeError(
                "benchmark omitted requested prepared-cache prefix seal")
        result["prepared_cache_contract"] = {
            "mode": "sealed-prefix-v1",
            "operation": "initial",
            "content_sha256": prepared_header.get("content_sha256"),
        }
    (output / "public-wrapper-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

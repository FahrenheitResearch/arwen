"""The mapped-engine parity SWEEP: both engines through the public doors.

The pytest battery (``tests/test_mapped_engine_parity.py``) holds each
engine to the measured golden through the engine seam directly.  This
driver is the other half the decode-vendor design names: it drives the
PUBLIC frozen entries -- ``decode_mapped_source`` / ``inspect_mapped_source``
routed by ``GPUWM_MAPPED_ENGINE`` exactly as ``gpuwm prep``'s
``--mapped-engine`` flag routes them -- twice per staged source, and
compares what a caller actually receives:

* the parity digest of the returned frames (field arrays by SHA-256 of
  their ``<f8`` bytes, axes, shapes, missing counts, headers);
* the frame-evidence receipt (``mapped_frame_receipt``), canonical JSON,
  no mask needed -- it carries no engine identity by construction;
* the inspection document, canonical JSON under the battery's one
  enumerated mask (``decoders``: which binaries read the bytes -- the
  only thing that differs BY DESIGN);
* refusals: the same Python exception TYPE and the same message, on a
  battery of driven refusal cases (broken mapping grammar, missing
  input, manifest drift, corrupt GRIB bytes, and the real
  selector-identity twin the staging tree carries).

A second arm drives ``decode_composed_source`` -- the entry a mapped
PREPARATION actually runs -- the same way, and compares the composed
parity digest AND ``mapped_composition_receipt``, the receipt a prepared
run publishes as its evidence.

Wall time for both runs is recorded per source and per arm, so the sweep
is also the measured Python-versus-Rust decode and compose bench.

An unexplained diff is FAIL and exits nonzero; a source routed to the
Python engine on both sides (a declared capability gap: compose, GRIB1,
NetCDF) is reported as ``ROUTED-PYTHON`` -- identical by construction,
and counted separately so the summary can never dress a gap up as a
comparison.

    python tools/mapped_engine_parity_sweep.py --output DIR
    python tools/mapped_engine_parity_sweep.py --source rap-awip32 --output DIR
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from gpuwm import mapped_engine_bridge as engine_bridge  # noqa: E402
from gpuwm.mapped_source import (decode_mapped_source,  # noqa: E402
                                 inspect_mapped_source,
                                 mapped_frame_receipt)
try:
    from test_mapped_engine_parity import (  # noqa: E402 - path set above
        COMPOSED_SOURCES, INSPECTION_MASK, STAGED_SOURCES, _composed_staged,
        _composed_path_strings, _format_of, _mapping_of, _sha256, _staged,
        _python_tools, canonical, compose_digest, compose_refusal_digest,
        composed_recipe, machine_independent, parity_digest, refusal_digest)
except ImportError as error:  # pragma: no cover - wheel-install route
    raise SystemExit(
        "mapped_engine_parity_sweep runs from a gpuwm checkout only: it "
        "imports the mapped parity battery and drives the staged sources it "
        "declares (tests/test_mapped_engine_parity.py), and tests/ does not "
        f"ship in the wheel ({error})") from error


SCHEMA = "gpuwm-mapped-parity-sweep-v1"


def _engine_env(engine: str | None) -> None:
    """Select the engine the way the front door's flag does."""

    if engine is None:
        os.environ.pop(engine_bridge.ENGINE_ENV, None)
    else:
        os.environ[engine_bridge.ENGINE_ENV] = engine


def _run_one_engine(source: str, engine: str | None) -> dict[str, object]:
    """One decode + receipt + inspection through the public doors."""

    entry = STAGED_SOURCES[source]
    mapping = _mapping_of(entry)
    files = entry["files"]
    _engine_env(engine)
    result: dict[str, object] = {"engine_request": engine or "default"}

    started = time.perf_counter()
    try:
        frames = decode_mapped_source(mapping, files)
    except Exception as error:  # noqa: BLE001 - the refusal IS the answer
        result["decode"] = refusal_digest(source, mapping, error)
        result["decode_seconds"] = time.perf_counter() - started
    else:
        result["decode"] = parity_digest(source, mapping, frames)
        result["decode_seconds"] = time.perf_counter() - started
        result["receipt"] = mapped_frame_receipt(mapping, files, frames)
        result["cells"] = int(sum(
            field.values.size
            for frame in frames for field in frame.fields.values()))
        del frames

    started = time.perf_counter()
    try:
        document = dict(inspect_mapped_source(mapping, files))
    except Exception as error:  # noqa: BLE001
        result["inspect"] = {
            "verdict": "REFUSED",
            "refusal": {"type": type(error).__name__,
                        "message": str(error)},
        }
    else:
        for key in INSPECTION_MASK:
            document.pop(key, None)
        result["inspect"] = {"verdict": "INSPECTED", "document": document}
    result["inspect_seconds"] = time.perf_counter() - started
    return result


def _messages_agree(python_message: str, rust_message: str) -> bool:
    """The battery's documented rule: the engine appends its remedy
    sentence to the shared message, so prefix agreement is agreement."""

    return (python_message == rust_message
            or rust_message.startswith(python_message)
            or python_message.startswith(rust_message))


def _same_answer(python_side, rust_side) -> bool:
    """Canonical equality, with the refusal prefix rule applied."""

    if canonical(python_side) == canonical(rust_side):
        return True
    if not (isinstance(python_side, dict) and isinstance(rust_side, dict)):
        return False
    p_refusal = python_side.get("refusal")
    r_refusal = rust_side.get("refusal")
    if not (p_refusal and r_refusal):
        return False
    if p_refusal["type"] != r_refusal["type"]:
        return False
    if not _messages_agree(p_refusal["message"], r_refusal["message"]):
        return False
    trimmed_p = {k: v for k, v in python_side.items() if k != "refusal"}
    trimmed_r = {k: v for k, v in rust_side.items() if k != "refusal"}
    return canonical(trimmed_p) == canonical(trimmed_r)


def sweep_source(source: str) -> dict[str, object]:
    """Both engines, one verdict."""

    entry = STAGED_SOURCES[source]
    source_format = _format_of(entry)
    supported = engine_bridge.ENGINE_CAPABILITIES["decode"]
    routed_python = supported is not None and source_format not in supported

    python_run = _run_one_engine(source, engine_bridge.ENGINE_PYTHON)
    rust_run = _run_one_engine(source, None)   # the bare default
    _engine_env(None)

    same_decode = _same_answer(python_run["decode"], rust_run["decode"])
    same_receipt = canonical(python_run.get("receipt")) == canonical(
        rust_run.get("receipt"))
    same_inspect = _same_answer(python_run["inspect"], rust_run["inspect"])
    if routed_python:
        verdict = "ROUTED-PYTHON" if (
            same_decode and same_receipt and same_inspect) else "FAIL"
    else:
        verdict = "PASS" if (
            same_decode and same_receipt and same_inspect) else "FAIL"
    return {
        "source": source,
        "format": source_format,
        "files": len(entry["files"]),
        "bytes": int(sum(Path(p).stat().st_size for p in entry["files"])),
        "verdict": verdict,
        "decode_equal": same_decode,
        "receipt_equal": same_receipt,
        "inspect_equal": same_inspect,
        "decode_verdict": python_run["decode"]["verdict"],
        "cells": python_run.get("cells"),
        "python_seconds": python_run["decode_seconds"],
        "rust_seconds": rust_run["decode_seconds"],
        "python_inspect_seconds": python_run["inspect_seconds"],
        "rust_inspect_seconds": rust_run["inspect_seconds"],
        "python": python_run,
        "rust": rust_run,
    }


# --------------------------------------------------------------------
# The COMPOSE arm: the subcommand a preparation actually runs.
# --------------------------------------------------------------------
#
# `decode_mapped_source` above is the decode UNDER a preparation.  What
# `gpuwm prep` runs is `decode_composed_source` -- terrain composition,
# the coordinate-subset solve, the cross-source borrow -- and that is a
# different answer with two evidence products of its own.  This arm
# drives that public frozen entry twice per composed source, routed by
# the documented engine spelling exactly as `--mapped-engine` routes it,
# and compares what a preparation actually publishes:
#
#   * the composed parity digest (frames, alignment receipt,
#     contributing-source records, soil contract, terrain provenance);
#   * `mapped_composition_receipt(bundle)` -- the receipt a prepared run
#     carries as its evidence, canonical JSON, machine paths masked the
#     way every other row is masked.
#
# The receipt is compared as well as the digest because it is what a
# reader of a prepared case sees: a bundle whose arrays agreed while its
# published receipt did not would be a preparation nobody could audit.


def _compose_one_engine(source: str, engine: str | None,
                        work: Path) -> dict[str, object]:
    """One composed preparation through the public frozen entry."""

    from gpuwm.mapped_authoring import author_input_manifest
    from gpuwm.mapped_composition import (decode_composed_source,
                                          mapped_composition_receipt)

    recipe = composed_recipe(source)
    paths = _composed_path_strings(recipe)
    _engine_env(engine)
    # The Python arm names the subprocess decoder tools EXPLICITLY, which
    # is how a caller pins that engine and is what the front door's
    # `--mapped-engine python` resolves for them.  It is not redundant
    # with the environment spelling: `mapped_authoring` asks the
    # capability table about `decode` while the composition asks about
    # `compose`, so both halves have to be pinned or the manifest seals
    # one binary and the composition verifies against another.
    tools = ({} if engine != engine_bridge.ENGINE_PYTHON
             else _python_tools(str(recipe["format"])))
    result: dict[str, object] = {"engine_request": engine or "default"}
    manifest = work / f"input-manifest-{engine or 'default'}.json"
    if manifest.exists():
        manifest.unlink()
    # Authored per run, not shared: the manifest seals WHICH BINARY reads
    # the bytes, so the two routes seal different decoder rows on purpose
    # and a shared manifest would refuse one of them.
    author_input_manifest(
        manifest,
        mapping_path=recipe["mapping"], composition_path=recipe["composition"],
        primary_files=recipe["primary"],
        supplement_files=recipe["supplements"],
        provenance_files=recipe["provenance"], **tools)
    started = time.perf_counter()
    try:
        bundle = decode_composed_source(
            recipe["composition"], recipe["mapping"], recipe["primary"],
            recipe["supplements"], recipe["provenance"],
            input_manifest=manifest, input_manifest_sha256=_sha256(manifest),
            contributing_mappings=recipe["contributing"] or None, **tools)
    except Exception as error:  # noqa: BLE001 - the refusal IS the answer
        result["compose"] = compose_refusal_digest(recipe, error)
        result["compose_seconds"] = time.perf_counter() - started
        return result
    result["compose"] = compose_digest(recipe, bundle)
    result["compose_seconds"] = time.perf_counter() - started
    result["receipt"] = machine_independent(
        source, mapped_composition_receipt(bundle), paths=paths)
    result["cells"] = int(sum(
        field.values.size
        for frame in bundle.frames for field in frame.fields.values()))
    del bundle
    return result


#: What a composition receipt carries that the two routes differ on BY
#: DESIGN, enumerated rather than diffed away.  Everything else -- the
#: mapping and composition identities, the valid times, the frame
#: digests, the alignment receipt, the terrain products and provenance,
#: the soil contract -- is compared.
#:
#:   * ``decoders`` -- WHICH BINARY read the bytes.  The engine route
#:     seals one in-process row; the Python route seals the subprocess
#:     pair.  That is the whole point of recording it, and it is the same
#:     member the inspection document masks for the same reason.
#:   * ``input_manifest`` -- authored per route, because the manifest
#:     SEALS the decoder inventory above.  Two routes cannot share one
#:     manifest without one of them refusing it, so its path and digest
#:     differ by construction.
#:   * ``receipt_content_sha256`` -- the digest OVER the two members
#:     above, so it moves with them.
COMPOSITION_RECEIPT_MASK = ("decoders", "input_manifest",
                            "receipt_content_sha256")


def _masked_receipt(receipt):
    if not isinstance(receipt, dict):
        return receipt
    return {key: value for key, value in receipt.items()
            if key not in COMPOSITION_RECEIPT_MASK}


def sweep_composed(source: str, work: Path) -> dict[str, object]:
    """Both engines, one composed preparation, one verdict."""

    recipe = composed_recipe(source)
    source_format = str(recipe["format"])
    routed_python = not engine_bridge.engine_supports("compose", source_format)

    work.mkdir(parents=True, exist_ok=True)
    python_run = _compose_one_engine(source, engine_bridge.ENGINE_PYTHON, work)
    rust_run = _compose_one_engine(source, None, work)   # the bare default
    _engine_env(None)

    same_compose = _same_answer(python_run["compose"], rust_run["compose"])
    same_receipt = canonical(
        _masked_receipt(python_run.get("receipt"))) == canonical(
            _masked_receipt(rust_run.get("receipt")))
    if routed_python:
        verdict = "ROUTED-PYTHON" if (same_compose and same_receipt) else "FAIL"
    else:
        verdict = "PASS" if (same_compose and same_receipt) else "FAIL"
    return {
        "source": source,
        "format": source_format,
        "files": len(recipe["primary"]),
        "verdict": verdict,
        "compose_equal": same_compose,
        "receipt_equal": same_receipt,
        "receipt_mask": list(COMPOSITION_RECEIPT_MASK),
        "compose_verdict": python_run["compose"]["verdict"],
        "bindings": len(python_run["compose"].get("contributing_sources", ())),
        "cells": python_run.get("cells"),
        "python_seconds": python_run["compose_seconds"],
        "rust_seconds": rust_run["compose_seconds"],
        "python": python_run,
        "rust": rust_run,
    }


# --------------------------------------------------------------------
# The driven refusal battery: same exception type, same message.
# --------------------------------------------------------------------

def _refusal(callable_, *args, **kwargs):
    try:
        callable_(*args, **kwargs)
    except Exception as error:  # noqa: BLE001 - the refusal IS the answer
        return {"type": type(error).__name__, "message": str(error)}
    return {"type": None, "message": "NO REFUSAL: the call succeeded"}


def refusal_cases(work: Path) -> dict[str, tuple[Path, tuple[Path, ...]]]:
    """Driven refusal inputs, built beside real staged bytes.

    Every case is decoded through BOTH engines by the sweep; the grammar
    cases fail before any engine runs (Python owns mapping validation on
    both routes), and the byte cases reach whichever engine the route
    chooses, so the comparison covers the whole refusal path a user hits.
    """

    real = STAGED_SOURCES["rap-awip32"]
    real_mapping = _mapping_of(real)
    real_files = tuple(real["files"])

    work.mkdir(parents=True, exist_ok=True)
    broken = work / "broken-grammar.mapping.json"
    document = json.loads(real_mapping.read_text(encoding="utf-8"))
    document.pop("fields", None)
    broken.write_text(json.dumps(document), encoding="utf-8")

    truncated = work / "truncated.grib2"
    payload = Path(real_files[0]).read_bytes()
    truncated.write_bytes(payload[: max(64, len(payload) // 7)])

    nomads = STAGED_SOURCES["aigfs-nomads"]
    twin_files = tuple(
        Path(str(path)).with_name(
            Path(str(path)).name.replace("NOMADS.", "S3."))
        for path in nomads["files"])

    return {
        "mapping_grammar": (broken, real_files),
        "missing_input": (real_mapping,
                          (work / "never-fetched.grib2",)),
        "corrupt_grib": (real_mapping,
                         (truncated, *real_files[1:])),
        "selector_identity_twin": (_mapping_of(nomads), twin_files),
    }


def sweep_refusals(work: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, (mapping, files) in sorted(refusal_cases(work).items()):
        if any(not Path(f).is_file() for f in files) \
                and name != "missing_input":
            rows.append({"case": name, "verdict": "SKIPPED",
                         "reason": "staged twin bytes absent"})
            continue
        _engine_env(engine_bridge.ENGINE_PYTHON)
        python_answer = _refusal(decode_mapped_source, mapping, files)
        _engine_env(None)
        rust_answer = _refusal(decode_mapped_source, mapping, files)
        same_type = python_answer["type"] == rust_answer["type"]
        # The engine appends its remedy sentence to the shared message;
        # prefix agreement is agreement.  One EXPLAINED exception: on
        # undecodable bytes the Python route quotes its subprocess tool
        # ("GRIB2 inventory failed for X: Error: ...") while the engine
        # speaks in-process -- the difference IS the masked decoder
        # identity -- so there the shared grib-core diagnostic
        # ("truncated GRIB2 envelope N at byte B") must match instead.
        p_message, r_message = python_answer["message"], rust_answer["message"]
        same_message = _messages_agree(p_message, r_message)
        explained = False
        if not same_message:
            diagnostic = re.compile(
                r"truncated GRIB2 envelope \d+ at byte \d+")
            p_found = diagnostic.search(p_message)
            r_found = diagnostic.search(r_message)
            if p_found and r_found and p_found.group() == r_found.group():
                same_message = True
                explained = "shared grib-core diagnostic; wrappers differ " \
                    "by decoder identity (subprocess tool vs in-process)"
        rows.append({
            "case": name,
            "verdict": "PASS" if (same_type and same_message
                                  and python_answer["type"]) else "FAIL",
            "explained": explained,
            "python": python_answer,
            "rust": rust_answer,
        })
    return rows


def report(message: str) -> None:
    """One row of the running verdict, on the terminal AND in the log NOW.

    The sweep is a twenty-minute job whose whole reason for printing per
    row is that somebody -- a person, or a watcher grepping the redirect
    -- can see where it is.  Python line-buffers stdout only when it is a
    terminal; redirect it to a file, which is exactly what a queued leg
    does, and it becomes an 8 KiB block buffer.  Measured at the 2.5.0 RC
    verify: the log stayed 0 bytes for the entire run and every row
    appeared at once at exit.

    That is the recorded buffered-log silent-green trap. A watcher
    reading that file sees no FAIL rows because it sees NO rows, and
    reads a crashed sweep as a quiet one -- it cannot distinguish "still
    running, all good" from "died on row 3" until the process is over,
    which is after the only moment the information was worth anything.

    So every row goes out flushed, and :func:`main` additionally puts the
    stream itself in line-buffered mode so a traceback or a library print
    lands in the same order rather than after the rows it preceded.
    """

    print(message, flush=True)


def main(argv: list[str] | None = None) -> int:
    # Line buffering for anything that does not go through report():
    # tracebacks, and any print a called module makes.  Guarded because a
    # caller may hand us a stdout that is not a TextIOWrapper (a capture
    # buffer, a pipe wrapper); report()'s explicit flush is what the row
    # verdicts actually rely on, and it works on any stream.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError, OSError):
        pass

    parser = argparse.ArgumentParser(
        prog="mapped_engine_parity_sweep",
        description="Both engines through the public doors, one verdict.")
    parser.add_argument("--source", action="append", dest="sources",
                        help="sweep one source; repeat for several")
    parser.add_argument("--output", type=Path, required=True,
                        help="directory for the sweep record (created)")
    parser.add_argument("--skip-refusals", action="store_true")
    parser.add_argument("--skip-decode", action="store_true",
                        help="skip the decode arm; sweep compose only")
    parser.add_argument("--skip-compose", action="store_true",
                        help="skip the compose arm; sweep decode only")
    args = parser.parse_args(argv)

    names = sorted(args.sources or STAGED_SOURCES)
    composed_names = sorted(
        args.sources or COMPOSED_SOURCES) if not args.skip_compose else []
    composed_names = [n for n in composed_names if n in COMPOSED_SOURCES]
    if args.skip_decode:
        names = []
    unknown = [
        n for n in (args.sources or ())
        if n not in STAGED_SOURCES and n not in COMPOSED_SOURCES
    ]
    if unknown:
        print(f"unknown source(s): {unknown}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures = 0
    for name in names:
        entry = STAGED_SOURCES[name]
        if not _staged(entry):
            rows.append({"source": name, "verdict": "SKIPPED",
                         "reason": "staged bytes absent"})
            report(f"{name}: SKIPPED (staged bytes absent)")
            continue
        try:
            row = sweep_source(name)
        except Exception:  # noqa: BLE001 - a crash is not a refusal
            failures += 1
            rows.append({"source": name, "verdict": "ERROR",
                         "trace": traceback.format_exc()})
            report(f"{name}: ERROR (see record)")
            continue
        rows.append(row)
        if row["verdict"] == "FAIL":
            failures += 1
        report(f"{name}: {row['verdict']} "
              f"({row['decode_verdict']}; python {row['python_seconds']:.1f}s"
              f" vs rust {row['rust_seconds']:.1f}s)")

    composed_rows: list[dict[str, object]] = []
    for name in composed_names:
        if not _composed_staged(name):
            composed_rows.append({"source": name, "verdict": "SKIPPED",
                                  "reason": "staged composed bytes absent"})
            report(f"compose {name}: SKIPPED (staged bytes absent)")
            continue
        try:
            row = sweep_composed(name, args.output / "compose-work")
        except Exception:  # noqa: BLE001 - a crash is not a refusal
            failures += 1
            composed_rows.append({"source": name, "verdict": "ERROR",
                                  "trace": traceback.format_exc()})
            report(f"compose {name}: ERROR (see record)")
            continue
        composed_rows.append(row)
        if row["verdict"] == "FAIL":
            failures += 1
        report(f"compose {name}: {row['verdict']} "
              f"({row['compose_verdict']}, {row['bindings']} binding(s); "
              f"python {row['python_seconds']:.1f}s"
              f" vs rust {row['rust_seconds']:.1f}s)")

    refusals: list[dict[str, object]] = []
    if not args.skip_refusals:
        refusals = sweep_refusals(args.output / "refusal-work")
        for row in refusals:
            if row["verdict"] == "FAIL":
                failures += 1
            report(f"refusal {row['case']}: {row['verdict']}")

    record = {
        "schema": SCHEMA,
        "engine_default": engine_bridge.DEFAULT_ENGINE,
        "sources": rows,
        "composed": composed_rows,
        "refusals": refusals,
    }
    (args.output / "sweep.json").write_text(
        canonical(record) + "\n", encoding="utf-8", newline="\n")
    report(f"record: {args.output / 'sweep.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

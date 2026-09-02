"""Seam constants, frameset codec and launcher for the Rust mapped engine.

This module is the Python half of the ``gpuwm_mapped_engine`` seam.  The
normative contract text is ``docs/dev/decode-vendor-design.md`` and the
Rust half is ``tools/rw_wps/crates/mapped-engine/src/main.rs``; all three
spell ONE contract and the parity battery
(``tests/test_mapped_engine_parity.py``) holds them to it.

What lives here, and why each piece is on the Python side of the ruling
(Python orchestrates, Rust decodes):

* the resolution ladder and the ABI-marker handshake -- install-state
  knowledge the engine cannot have about itself;
* the ``gpuwm-mapped-frameset-v1`` codec: :func:`write_frameset` is what
  the PYTHON engine emits so its output is comparable to the Rust
  engine's byte for byte, and :func:`read_frameset` is what both routes
  read back.  One codec, two writers, one reader -- a divergence between
  the engines shows up as a hash mismatch rather than as two readers
  that disagree quietly;
* :func:`run_engine`, which launches the exe, drains its progress
  stream, and turns a refusal object into the exception type the Python
  engine raises for that same condition today.

Engine selection is :func:`resolve_engine`.  The Python engine stays
reachable through ``GPUWM_MAPPED_ENGINE=python`` / ``--mapped-engine
python`` and is documented AS A WORKAROUND (fixed means default).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Sequence as _ABCSequence
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gpuwm.bridges import (default_bridge_dir, accept_resolved,
                           executable_name, packaged_bridge_dir)
from gpuwm.ingest.source_coverage import \
    ForcingSeriesRefusal as _ForcingSeriesRefusal

#: Executable basename, resolved through the standard bridge-ladder
#: shape (env override, this checkout's build, staged copies) the way
#: :mod:`gpuwm.netcdf_bridge` resolves ``rw_netcdf``, and held to the
#: ABI marker below exactly like every other bridge.
ENGINE_NAME = "gpuwm_mapped_engine"

#: Environment variable naming a prebuilt engine binary.  A set
#: override naming a missing file is a hard error at resolution time:
#: explicit configuration fails loudly, never falls through.
ENGINE_PATH_ENV = "GPUWM_MAPPED_ENGINE_BIN"

#: Workspace directory of the engine crate, relative to a checkout root.
ENGINE_CRATE_RELATIVE = "tools/rw_wps"

#: The compiled-in contract handshake.  This is the frameset schema
#: name: it changes exactly when the frameset contract changes, so a
#: stale staged binary fails the static handshake instead of writing a
#: shape the Python side no longer reads.  Lane 3 adds this literal to
#: :data:`gpuwm.bridges.BRIDGE_ABI_MARKERS` when it wires the route.
ABI_MARKER = b"gpuwm-mapped-frameset-v1"

#: Output schemas the engine writes.
FRAMESET_SCHEMA = "gpuwm-mapped-frameset-v1"
PROGRESS_SCHEMA = "gpuwm-mapped-engine-progress-v1"
REFUSAL_SCHEMA = "gpuwm-mapped-refusal-v1"

#: Engine selection: default is the Rust engine once lane 3 lands
#: (fixed means default); the Python engine remains reachable ONLY as a
#: documented workaround through this environment variable or the
#: matching front-door flag.
ENGINE_ENV = "GPUWM_MAPPED_ENGINE"
ENGINE_RUST = "rust"
ENGINE_PYTHON = "python"
ENGINES = (ENGINE_RUST, ENGINE_PYTHON)

#: The engine a bare run uses.  ONE constant, so the flip named in the
#: design's §5 is one edit with one test behind it.
#:
#: ``rust`` since integration: a bare ``gpuwm prep --source mapped`` run
#: decodes its bytes in the Rust engine, and the Python engine below is
#: reachable only through the documented workaround spelling.  Setting
#: this back to :data:`ENGINE_PYTHON` means writing a blocker into
#: :data:`DEFAULT_ENGINE_BLOCKER` that names the concrete breakage;
#: ``tests/test_mapped_engine_parity.py`` fails if the two disagree.
DEFAULT_ENGINE = ENGINE_RUST

#: Why :data:`DEFAULT_ENGINE` is not ``rust``; ``None`` while it is.
DEFAULT_ENGINE_BLOCKER = None

#: What the Rust engine implements, by mapped source format, per
#: subcommand.  ``None`` means every format.
#:
#: This is NOT an optimisation and NOT a fallback: the engine refuses
#: the entries missing below with class ``not_implemented`` in its own
#: words, so without this table a bare run of a composed source -- which
#: is how most staged sources reach a complete canonical frame -- would
#: refuse where it used to decode, and Drew's refusal law is that a
#: refusal has to name breakage it PREVENTS.  An unported path prevents
#: nothing; it is unfinished work, and unfinished work does not get to
#: break a route that already ships.
#:
#: So the default engine is Rust and the still-unported paths are named
#: HERE, in one table, checked against the real binary by
#: ``test_the_capability_table_matches_the_built_engine`` -- which reads
#: the artifact's own declaration rather than trusting this comment.
#: Every mapped call records which engine actually ran, so nothing about
#: the split is silent.  NO entry is outstanding: every subcommand
#: declares every source format this repo reads.
#:
#: ``compose`` was the last one out, and it left on the evidence the
#: other two left on -- real staged bytes, measured against the Python
#: engine of record.  All eleven registered ``mapped_composition_v1``
#: sources with staged bytes reproduce their compose golden through the
#: built binary, byte for byte: the frames, the alignment receipt (all
#: three terrain clock rules) and the per-binding contributing-source
#: records (both cross-source borrows), plus the one source whose
#: staged ladder makes it refuse, refusing with the same sentence.
#: ``tests/test_mapped_engine_parity.py``'s
#: ``test_the_rust_engine_reproduces_the_compose_golden`` is that gate,
#: and it runs through ``decode_composed_source`` -- the real front
#: door -- rather than by driving the exe by hand.
#:
#: Per format, precisely, because the eleven measured rows are all
#: GRIB2 and that is the whole registry of packaged composed profiles:
#:
#:   * ``grib2`` -- eleven registered sources measured byte for byte,
#:     plus the front-door dual run in
#:     ``tools/mapped_engine_parity_sweep.py``'s compose arm.
#:   * ``netcdf`` -- no composed NetCDF source has staged bytes on any
#:     box that has run this (``20crv3-cf`` is the written-down
#:     exemption in the registry-coverage test), so it has no compose
#:     GOLDEN.  It is declared anyway, and not as a guess: NetCDF has no
#:     subprocess decoder tool, so an undeclared ``compose`` beside a
#:     declared ``decode`` makes the two capability questions name
#:     DIFFERENT decoder inventories and ``_verify_manifest`` refuses a
#:     correct preparation.  That asymmetry is measured by
#:     ``test_the_two_questions_name_one_decoder_inventory``.  What the
#:     declaration rests on otherwise is that the composition layer is
#:     format-independent -- it joins DECODED collections, and the
#:     per-format work under it is ``decode``, which NetCDF passes on
#:     its own golden and on the whole Python NetCDF suite.
#:   * ``grib1`` -- no packaged composed GRIB1 profile exists, so it is
#:     reachable only through the generic ``--source mapped`` door and
#:     has no golden for the same reason.  Declared to keep the two
#:     entries equal, which is the shape the manifest seal wants;
#:     leaving it out would make a generic GRIB1 composed prep depend on
#:     the door forwarding ``grib1_bridge`` to keep the answers in step.
#:
#: Every decode format this repo reads now decodes in process.  The two
#: that left this list before ``compose`` left it on real-bytes
#: evidence too:
#:
#:   * GRIB1 left when ``mapped-engine``'s ``grib1`` module landed: the
#:     ERA5 1974 reference object decodes to the Python engine's own
#:     forty-two array digests, grid fingerprint and materialization
#:     refusal through the built binary
#:     (``tests/test_mapped_engine_parity.py``'s ``era5-1974-grib1``
#:     row), so a bare GRIB1 run no longer needs the Python route.
#:   * NetCDF left once BOTH causes of its hold-back were named and
#:     fixed: ``tools/rw_wps`` was linking the stock crates.io
#:     ``hdf5-reader`` rather than the hardened vendored copy, and
#:     NetCDF-4 coordinate variables are HDF5 dimension scales that
#:     netcrust's variable index omits.  The whole Python NetCDF test
#:     set passes under ``GPUWM_MAPPED_ENGINE=rust`` on the same
#:     fixtures the Python engine passes, which is the evidence that
#:     declaration rests on -- not the single crate golden that was
#:     green throughout.
#: ``inventory`` is the raw per-record product-identity surface -- the
#: engine's answer to the question ``grib2_inventory`` used to be
#: resolved for on the 20CRv3 member route.  Declared for GRIB2 alone
#: because it renders GRIB2 section octets and decodes nothing; the
#: subprocess tool it replaces read the same edition and nothing else.
ENGINE_CAPABILITIES: Mapping[str, frozenset[str] | None] = {
    "decode": frozenset({"grib1", "grib2", "netcdf"}),
    "inspect": frozenset({"grib1", "grib2", "netcdf"}),
    "compose": frozenset({"grib1", "grib2", "netcdf"}),
    "inventory": frozenset({"grib2"}),
}

#: The subcommand ``gpuwm.mapped_direct`` -- the module every mapped
#: preparation runs, packaged profiles and the generic route alike --
#: actually asks the engine for.
#:
#: It is ``compose`` on EVERY call, contributing mappings or not:
#: ``prepare_mapped_wrf`` runs ``gpuwm.mapped_composition``'s byte work
#: (terrain composition, bound fields, subset indices) to reach a
#: canonical frame, and that is the ``compose`` entry above.
#:
#: Named here, beside the table it indexes, because the front door and
#: the route have to ask ONE question.  They did not: ``source_cli``
#: asked which engine was the DEFAULT and concluded "Rust, so no
#: subprocess tools are wanted", while the route asked the table for
#: ``compose`` and got the Python engine -- so the door composed a
#: command with the Python engine's work to do and none of the Python
#: engine's tools, and a bare prep of any composed source died inside
#: the decoder contract.  Declaring ``compose`` was the ONE edit the
#: port needed in this module; every consulting site followed.
MAPPED_ROUTE_SUBCOMMAND = "compose"

#: The unported paths, spelled for humans (docs, doctor, receipts).
#:
#: EMPTY, and kept as a tuple so the next unported door is DECLARED
#: rather than excused: ``gpuwm doctor`` and the CLI reference read this
#: table, and ``test_a_door_that_forwards_decoder_tools_is_a_named_gap``
#: holds it in both directions over the whole registry -- a door that
#: forwards a decoder tool with no entry here fails, and an entry with
#: no forwarding door fails with it.
#:
#: The last entry out was the 20CRv3 member route (``gpuwm prep --source
#: 20crv3``), and it left carrying both of the gates that held it back
#: rather than dropping them:
#:
#:   * ENSEMBLE IDENTITY.  The 20CRv3 PDT carries no member, so the
#:     verified filename member is now an EXPLICIT binding in the
#:     composition input manifest (``member``/``member_identity``), and
#:     ``compose`` -- both engines -- stamps it onto every canonical
#:     frame and into the sealed alignment receipt.  Held on the private
#:     member bytes by ``tests/test_twentycrv3_direct.py::``
#:     ``test_the_member_survives_the_generic_compose_route``.
#:   * PRODUCT IDENTITY.  ``twentycrv3_direct._verify_archive_inventory``
#:     keeps its exact every-member GRIB2 product contract and its own
#:     refusal wording; its measurement instrument on the bare default
#:     is the engine's raw record-inventory surface (the ``inventory``
#:     subcommand, :func:`engine_record_inventory`), with the subprocess
#:     ``grib2_inventory`` still answering on the documented
#:     Python-engine workaround.
ENGINE_GAPS: tuple[str, ...] = ()


#: Schema of the `capabilities` subcommand's document.
CAPABILITIES_SCHEMA = "gpuwm-mapped-engine-capabilities-v1"

#: Schema of the `inventory` subcommand's document: the raw per-record
#: GRIB2 product identity of each input, value spellings identical to the
#: subprocess `grib2_inventory` TSV so a product-identity gate reads ONE
#: spelling whichever instrument measured it.
RECORD_INVENTORY_SCHEMA = "gpuwm-mapped-record-inventory-v1"


def engine_record_inventory(
    files: Sequence[str | Path],
    *,
    engine: str | Path | None = None,
) -> dict[Path, list[dict[str, str]]]:
    """Raw per-record product identity, read by the engine in process.

    The engine twin of running ``grib2_inventory`` over each file: one
    row per GRIB2 message carrying every identity octet (authority,
    process, time semantics, level pair, member octet, grid definition,
    packing), as strings in the subprocess tool's own spellings.  This
    is the surface the archive product-identity gates consume on the
    bare default, so those gates keep their exact contract and their
    own refusal wording while the measurement instrument moves in
    process.  Returns ``{resolved_path: rows}``.
    """

    import tempfile

    binary = Path(engine) if engine is not None else require_engine()
    paths = [Path(path).resolve() for path in files]
    if not paths:
        raise ValueError("record inventory requires at least one input file")
    with tempfile.TemporaryDirectory(
            prefix="gpuwm-mapped-inventory-") as work:
        input_list = Path(work) / "inputs.txt"
        input_list.write_text(
            "".join(f"{path}\n" for path in paths), encoding="utf-8")
        command = [str(binary), "inventory", "--input-list", str(input_list)]
        completed = subprocess.run(
            command, capture_output=True, text=True, check=False,
        )
    if completed.returncode != 0:
        refusal = parse_refusal(completed.stderr or "")
        if refusal is None:
            tail = (completed.stderr or "").strip().splitlines()
            raise RuntimeError(
                f"{ENGINE_NAME} inventory exited {completed.returncode} "
                f"without a {REFUSAL_SCHEMA} object on its last stderr "
                "line, so there is no class to map and no remedy to "
                "relay: " + (tail[-1] if tail else "it printed nothing"))
        raise refusal_error(refusal, command)
    document = None
    for line in reversed((completed.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            document = json.loads(line)
            break
    if not isinstance(document, dict) \
            or str(document.get("schema")) != RECORD_INVENTORY_SCHEMA:
        raise ValueError(
            f"{binary} inventory answered "
            f"{None if document is None else document.get('schema')!r}; "
            f"this release reads {RECORD_INVENTORY_SCHEMA!r} -- rebuild "
            "the engine from a matching checkout")
    result: dict[Path, list[dict[str, str]]] = {}
    for entry in document["files"]:
        rows = [
            {str(key): str(value) for key, value in row.items()}
            for row in entry["records"]
        ]
        result[Path(str(entry["path"])).resolve()] = rows
    missing = [path for path in paths if path not in result]
    if missing:
        raise ValueError(
            f"{binary} inventory omitted requested input(s) {missing}; the "
            "answer does not cover the question, so nothing downstream may "
            "bind a contract to it")
    return result


def declared_capabilities(engine: Path | None = None) -> dict[str, list[str]]:
    """What the BUILT engine says it implements, per subcommand.

    Asked by running the artifact, because the answer is a property of
    the binary in hand and not of this release's notes: a checkout whose
    engine is older or newer than this file implements a different set,
    and a table that guessed would misroute silently.
    """

    import subprocess

    binary = Path(engine) if engine is not None else require_engine()
    completed = subprocess.run(
        [str(binary), "capabilities"],
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        refusal = parse_refusal(completed.stderr or "")
        raise refusal_error(refusal) if refusal else RuntimeError(
            f"{binary} capabilities exited {completed.returncode}: "
            f"{(completed.stderr or '').strip()[-400:]}")
    document = json.loads(completed.stdout)
    schema = str(document.get("schema"))
    if schema != CAPABILITIES_SCHEMA:
        raise ValueError(
            f"{binary} capabilities declares schema {schema!r}; this "
            f"release reads {CAPABILITIES_SCHEMA!r}")
    return {
        str(name): [str(value) for value in formats]
        for name, formats in dict(document["subcommands"]).items()
    }


def engine_supports(subcommand: str, source_format: str | None) -> bool:
    """Can the Rust engine do this subcommand for this source format?

    ``source_format`` is the mapping's declared ``format``; ``None``
    asks about the subcommand alone.  Unknown subcommands answer
    ``False`` rather than raising: a caller asking about something this
    release has never heard of must not be routed at the engine.
    """

    formats = ENGINE_CAPABILITIES.get(subcommand)
    if formats is None:
        return subcommand in ENGINE_CAPABILITIES
    if not formats:
        return False
    return source_format is not None and source_format in formats

#: Refusal classes the engine may emit, mapped to the exception type the
#: Python engine raises for the same condition today.  The parity
#: battery asserts CLASS AND REMEDY equality case by case; an engine
#: refusal with a class not in this table is itself a defect (the
#: bridge re-raises it as ``RuntimeError`` naming the unknown class).
REFUSAL_CLASSES: Mapping[str, type[Exception]] = {
    # argv/contract misuse; also covers the skeleton's `usage` refusal.
    "usage": ValueError,
    # Skeleton-only: subcommand not yet implemented (lane 2 removes it).
    "not_implemented": NotImplementedError,
    # A named input, mapping, composition, or decoder path is absent.
    "missing_input": FileNotFoundError,
    # Mapping/composition document invalid (schema, grammar, closed
    # catalogs, duplicate keys, non-finite JSON numbers).
    "mapping_invalid": ValueError,
    # Input manifest verification failed (hash or inventory drift).
    "manifest_mismatch": ValueError,
    # No selector matched / selector identity ambiguity on real bytes.
    "selector_unmatched": ValueError,
    # Observed GRIB grid octets contradict the mapping's declaration.
    "grid_mismatch": ValueError,
    # GRIB/NetCDF byte-level decode failure (grib-core / netcrust).
    "decode_failed": ValueError,
    # Canonical-frame invariant violated (axes, units, monotonicity,
    # missing-count accounting, soil column policy).
    "frame_invalid": ValueError,
    # The staged valid times cannot bound a forecast.  Split out of
    # frame_invalid when the preparation front door promoted the same
    # condition to its own class (a ValueError subclass, so an existing
    # `except ValueError` net is unchanged): the door prints this one as
    # sentences with the staging remedy instead of a traceback, and the
    # engines must agree on the class a caller catches.
    "forcing_series": _ForcingSeriesRefusal,
    # Authority file changed hash mid-run.
    "authority_moved": RuntimeError,
}


def engine_candidates() -> tuple[Path, ...]:
    """Deterministic candidate paths for the engine binary, best first.

    The same ladder shape :func:`gpuwm.netcdf_bridge.netcdf_candidates`
    uses; lane 3 registers the name in :mod:`gpuwm.bridges` proper so
    ``gpuwm doctor`` reports the estate.
    """

    filename = executable_name(ENGINE_NAME)
    candidates: list[Path] = []
    override = os.environ.get(ENGINE_PATH_ENV)
    if override:
        candidates.append(Path(override))
    root = Path(__file__).resolve().parent.parent
    crate = root / ENGINE_CRATE_RELATIVE
    candidates.extend((
        crate / "target" / "release" / filename,
        crate / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        packaged_bridge_dir() / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def find_engine() -> Path | None:
    """First existing candidate, or ``None``.

    A set environment override naming a missing file raises: explicit
    configuration fails loudly rather than falling through the ladder.
    Marker enforcement happens at the call site exactly as
    ``gpuwm.mapped_source._build_grib2_tools`` does for the GRIB2
    tools: a found-but-stale binary is treated as unresolved and the
    refusal names the staleness and the remedy.
    """

    override = os.environ.get(ENGINE_PATH_ENV)
    for candidate in engine_candidates():
        if candidate.is_file():
            return accept_resolved(candidate.resolve())
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{ENGINE_PATH_ENV} names a missing file: {candidate}.  "
                f"Point it at a built {ENGINE_NAME} binary, or unset "
                f"{ENGINE_PATH_ENV} to use the resolution ladder.")
    return None


class EngineUnavailable(FileNotFoundError):
    """The engine binary is absent or speaks a contract this release does not.

    A ``FileNotFoundError`` because that is what every other unresolved
    gpuwm artifact raises, so console scripts already print it as one
    line with a remedy rather than a traceback.
    """


def engine_remedy(reason: str) -> str:
    """The refusal text for an unusable engine: breakage, then remedy.

    One paragraph, because this is what an EXCEPTION carries: a console
    script prints it as a single line and a reader is being told what
    went wrong.  Doctor's remedy field wants the other shape -- see
    :func:`engine_remedy_lines`.
    """

    from gpuwm import bridges

    route = bridges.install_aware_one_line_hint(
        f"build it with `cargo build --release --locked --offline "
        f"--manifest-path {ENGINE_CRATE_RELATIVE}/Cargo.toml`",
        ENGINE_CRATE_RELATIVE, ENGINE_NAME)
    return (
        f"{ENGINE_NAME} is unusable: {reason}.  To fix it, {route}; or "
        f"point {ENGINE_PATH_ENV} at a built copy; or run the documented "
        f"workaround {ENGINE_ENV}=python (equivalently --mapped-engine "
        "python), which decodes on the slower Python engine."
    )


def engine_remedy_lines(reason: str) -> str:
    """The same routes, shaped so the block survives being pasted.

    Doctor's closing contract is that every remedy LINE it prints is a
    command or a ``#`` comment, so a reader can select a whole gap
    report and paste it into a shell.  The paragraph above is prose in
    the middle of that block: pasting it runs ``gpuwm_mapped_engine`` as
    a command that does not exist, and the reader gets a shell error on
    top of the gap they were already looking at.  Both spellings exist
    because the two callers are different: an exception is read, a
    doctor remedy is pasted.

    Install-aware through the one mechanism every bundled artifact
    shares, :func:`gpuwm.bridges.artifact_remedy`: in a CHECKOUT the
    crate is present and the cargo build is a real one-liner; on a WHEEL
    the prebuilt bundle this release published leads, and the
    clone-and-build route follows it commented out, because that route
    is the only one on a platform with no published bundle.  The engine
    is named to :func:`gpuwm.bridges.prebuilt_bundle_offer` rather than
    merely asking whether a bundle exists, so the download is offered
    only when the pinned bundle actually carries it.

    It did not always compose that way, and the reason is exactly what
    changed.  ``gpuwm_mapped_engine`` was in no bundle roster, so this
    function printed a comment-only block on a wheel: leading with
    ``gpuwm fetch-bridges`` would have sent a reader to a command that
    could not supply what they were missing, which is the ``rw_mrms``
    failure :func:`gpuwm.bridges.prebuilt_bundle_offer` records.  The
    measured cost of that state was the whole DEFAULT decode path -- a
    fresh wheel install reported ``MISSING mapped decode engine ...
    blocks every mapped source`` and no gpuwm command could close it,
    leaving a clone and a cargo build as a wheel user's only route.  The
    engine is now in :data:`gpuwm.bridge_assets.BUNDLED_ARTIFACTS`,
    which is what makes the shared helper the truthful answer here.

    The documented workaround is named as a workaround, after both
    routes, on either install.
    """

    from gpuwm import bridges

    return "\n".join([
        f"# {ENGINE_NAME} is unusable: {reason}.",
        bridges.artifact_remedy(
            env_var=ENGINE_PATH_ENV,
            filename=executable_name(ENGINE_NAME),
            subject="the mapped decode engine",
            crate_relative=ENGINE_CRATE_RELATIVE,
            artifact=ENGINE_NAME),
        "# ...or take the documented WORKAROUND, which decodes on the "
        "slower Python",
        f"#   engine (equivalently --mapped-engine python):  "
        f"{ENGINE_ENV}=python",
    ])


def require_engine() -> Path:
    """Resolve the engine and prove it speaks this release's contract.

    The marker check is the same static handshake every bridge gets: a
    binary built before the frameset contract would write a shape this
    module no longer reads, and it would do it silently -- the 1.1.0 GFS
    series-file failure class.  Here it refuses before a byte is decoded.
    """

    from gpuwm.bridges import bridge_abi_matches

    engine = find_engine()
    if engine is None:
        raise EngineUnavailable(engine_remedy(
            "no binary was found through the resolution ladder "
            f"({ENGINE_PATH_ENV}, this checkout's "
            f"{ENGINE_CRATE_RELATIVE}/target, libexec/bridges, the "
            "packaged copy, then ~/.gpuwm/bridges)"))
    matches, detail = bridge_abi_matches(ENGINE_NAME, engine)
    if not matches:
        raise EngineUnavailable(engine_remedy(f"{engine} {detail}"))
    return engine


def resolve_engine(explicit: str | None = None) -> str:
    """Which engine a call uses: explicit argument, environment, default.

    Explicit beats environment beats :data:`DEFAULT_ENGINE`.  An
    unknown spelling refuses instead of falling back, because a silent
    fallback here would report a Rust-engine run that was a Python one.
    """

    for value, origin in (
        (explicit, "the --mapped-engine argument"),
        (os.environ.get(ENGINE_ENV), ENGINE_ENV),
    ):
        if value is None:
            continue
        chosen = str(value).strip().lower()
        if chosen not in ENGINES:
            raise ValueError(
                f"{origin} names engine {value!r}; choose one of "
                f"{', '.join(ENGINES)} ({ENGINE_PYTHON} is the "
                "documented workaround)")
        return chosen
    return DEFAULT_ENGINE


# --------------------------------------------------------------------
# gpuwm-mapped-frameset-v1
# --------------------------------------------------------------------
#
# One little-endian float64 stream (`frames.f64`) plus one metadata
# document (`frames.json`).  Field arrays ride the stream in manifest
# order; the 1-D axes ride the JSON, each beside the sha256 of its `<f8`
# bytes so a lossy number round-trip on either side is a refusal rather
# than a quiet perturbation of the grid.

#: `numpy` spelling of the only dtype the stream carries.
STREAM_DTYPE = "<f8"

#: Basenames inside a frameset directory.
FRAMES_DOCUMENT = "frames.json"
FRAMES_STREAM = "frames.f64"

#: `compose` writes one further document beside the frameset: the two
#: pieces of composition evidence that exist ONLY as a product of the
#: byte work, and which Python therefore cannot recompute on its own --
#: the terrain/bound-field alignment receipt and the per-binding
#: contributing-source records.
#:
#: An ADDENDUM to the design's §3.2, which specified the frameset and
#: was silent on composition evidence.  The named breakage without it:
#: `MappedSourceBundle` requires `alignment_receipt` and
#: `contributing_sources`, so a `compose` that wrote only a frameset
#: could not produce a bundle at all, and the composition receipt --
#: the evidence a cross-source preparation is judged on -- would have
#: to be fabricated on the Python side from values nobody measured.
COMPOSITION_DOCUMENT = "composition.json"
COMPOSITION_SCHEMA = "gpuwm-mapped-composition-evidence-v1"


def read_composition_evidence(directory: str | Path) -> dict[str, object]:
    """Read `compose`'s evidence document out of a frameset directory."""

    directory = Path(directory)
    path = directory / COMPOSITION_DOCUMENT
    if not path.is_file():
        raise ValueError(
            f"{ENGINE_NAME} compose wrote no {COMPOSITION_DOCUMENT} beside "
            f"its frameset in {directory}, so the composition has no "
            "alignment receipt and no contributing-source records; "
            "rebuild the engine from a matching checkout")
    document = json.loads(path.read_text(encoding="utf-8"))
    schema = str(document.get("schema"))
    if schema != COMPOSITION_SCHEMA:
        raise ValueError(
            f"{path} declares schema {schema!r}; this release reads "
            f"{COMPOSITION_SCHEMA!r}")
    for key in ("alignment_receipt", "contributing_sources"):
        if key not in document:
            raise ValueError(f"{path} carries no {key!r}")
    return document


def _axis_bytes(values: Any) -> bytes:
    import numpy as np

    return np.ascontiguousarray(
        np.asarray(values, dtype=np.float64)).astype(STREAM_DTYPE).tobytes()


def _stream_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _axis_document(values: Any) -> dict[str, object]:
    payload = _axis_bytes(values)
    return {
        "values": [float(value) for value in values],
        "count": len(payload) // 8,
        "sha256": _stream_sha256(payload),
    }


def _axis_values(document: Mapping[str, object], label: str):
    """Parse an axis, refusing a round-trip that moved a single bit."""

    import numpy as np

    values = np.asarray(document["values"], dtype=np.float64)
    if values.size != int(document["count"]):
        raise ValueError(
            f"frameset {label} axis declares {document['count']} values "
            f"and carries {values.size}")
    observed = _stream_sha256(_axis_bytes(values))
    if observed != str(document["sha256"]):
        raise ValueError(
            f"frameset {label} axis did not survive its JSON round trip: "
            f"the engine recorded {document['sha256']}, these numbers "
            f"hash to {observed}")
    return values


def _descriptor_document(value: object) -> object:
    """`asdict`-shaped JSON for a source-frame descriptor tree."""

    from dataclasses import asdict, is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value


def _header_document(header: object) -> dict[str, object]:
    return dict(_descriptor_document(header))            # type: ignore[arg-type]


def _header_from_document(document: Mapping[str, object]):
    """Rebuild a :class:`SourceFrameHeader` from its `asdict` form.

    Tuple-declared members are restored as tuples so a header rebuilt
    from the engine is indistinguishable from one the Python engine
    constructed -- including under `dataclasses.replace` and equality,
    which the composition path uses.
    """

    from gpuwm.source_frame import (FieldDescriptor, GridDescriptor,
                                    SourceFrameHeader, TimeDescriptor,
                                    VerticalDescriptor)

    grid = dict(document["grid"])                        # type: ignore[arg-type]
    grid["parameters"] = dict(grid.get("parameters") or {})
    vertical = {}
    for name, entry in (document.get("vertical_coordinates") or {}).items():
        entry = dict(entry)
        for key in ("level_values", "a_coefficients", "b_coefficients"):
            entry[key] = tuple(entry.get(key) or ())
        vertical[str(name)] = VerticalDescriptor(**entry)
    fields = []
    for entry in document.get("fields") or ():
        entry = dict(entry)
        entry["time"] = TimeDescriptor(**dict(entry["time"]))
        entry["dimensions"] = tuple(entry.get("dimensions") or ())
        entry["shape"] = tuple(int(size) for size in entry.get("shape") or ())
        fields.append(FieldDescriptor(**entry))
    return SourceFrameHeader(
        source_id=str(document["source_id"]),
        source_cycle=str(document["source_cycle"]),
        grid=GridDescriptor(**grid),
        vertical_coordinates=vertical,
        fields=tuple(fields),
        initialization_policies=dict(
            document.get("initialization_policies") or {}),
        schema=str(document.get("schema") or ""),
    )


def _emit_frameset(frames: Sequence[Any], sink) -> dict[str, object]:
    """Write the stream into ``sink``; return the metadata document.

    One routine builds both spellings (a directory and an in-memory
    buffer) so the two can never describe the stream differently.  Field
    arrays go out C-contiguous in document order, which is what makes
    the per-field ``offset``/``length`` a plain slice on the far side,
    and they are written one at a time: a real frameset runs to
    gigabytes and joining it first would triple its peak footprint.
    """

    import numpy as np

    from gpuwm.mapped_source import _array_sha256

    stream_digest = hashlib.sha256()
    offset = 0
    frame_documents: list[dict[str, object]] = []
    for frame in frames:
        field_documents: list[dict[str, object]] = []
        for name, field in frame.fields.items():
            array = np.ascontiguousarray(
                np.asarray(field.values, dtype=np.float64),
            ).astype(STREAM_DTYPE, copy=False)
            payload = memoryview(array).cast("B")
            stream_digest.update(payload)
            sink.write(payload)
            length = array.nbytes
            field_documents.append({
                "name": str(name),
                "units": field.units,
                "axes": list(field.axes),
                "location": field.location,
                "staggering": field.staggering,
                "shape": [int(size) for size in array.shape],
                "dtype": STREAM_DTYPE,
                "offset": offset,
                "length": length,
                "sha256": _array_sha256(array),
                "missing_count": int(field.missing_count),
                "source_references": list(field.source_references),
            })
            offset += length
        frame_documents.append({
            "valid_time": frame.valid_time.isoformat(),
            "member": frame.member,
            "source_cycle": frame.source_cycle.isoformat(),
            "latitude": _axis_document(frame.latitude),
            "longitude": _axis_document(frame.longitude),
            "vertical_kind": frame.vertical_kind,
            "vertical_units": frame.vertical_units,
            "vertical_values": _axis_document(frame.vertical_values),
            "grid_fingerprint": frame.grid_fingerprint,
            "mapping_sha256": frame.mapping_sha256,
            "input_sha256": {
                str(key): str(value)
                for key, value in sorted(frame.input_sha256.items())
            },
            "header": _header_document(frame.header),
            "fields": field_documents,
        })
    return {
        "schema": FRAMESET_SCHEMA,
        "stream": {
            "path": FRAMES_STREAM,
            "dtype": STREAM_DTYPE,
            "bytes": offset,
            "sha256": stream_digest.hexdigest(),
        },
        "frames": frame_documents,
    }


def frameset_document(frames: Sequence[Any]) -> tuple[dict[str, object], bytes]:
    """``(frames.json content, frames.f64 bytes)`` for in-memory callers."""

    import io

    sink = io.BytesIO()
    document = _emit_frameset(frames, sink)
    return document, sink.getvalue()


def write_frameset(directory: str | Path, frames: Sequence[Any]) -> Path:
    """Write a frameset directory; return it.

    The Python engine writes through this function so that "compare the
    two engines" is a comparison of two files in one format, not of a
    Rust artifact against an in-memory Python object graph.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / FRAMES_STREAM).open("wb") as sink:
        document = _emit_frameset(frames, sink)
    (directory / FRAMES_DOCUMENT).write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    return directory


def _fields_tile_stream(document: Mapping[str, object],
                        declared_bytes: int) -> bool:
    """Do the field extents cover the stream exactly, once each?

    True when the fields, in document order, start at 0, abut with no
    gap and end at the declared length -- which is what both writers
    emit (§3.2: "fields packed row-major in `frames.json` order").  When
    it holds, verifying every field's own sha256 verifies every byte of
    the stream, and the whole-stream digest is a second pass over the
    same bytes for the same answer.
    """

    # A document this cannot walk answers False, which costs one
    # whole-stream hash and leaves the real complaint to
    # :func:`_frames_from_document`, where it is already spelled.
    cursor = 0
    try:
        for entry in document["frames"]:                 # type: ignore[index]
            for field_document in entry["fields"]:
                start = int(field_document["offset"])
                length = int(field_document["length"])
                if start != cursor or length < 0:
                    return False
                cursor += length
    except (KeyError, TypeError, ValueError):
        return False
    return cursor == declared_bytes


class FrameSet(_ABCSequence):
    """The frames of a written frameset, materialized ONE at a time.

    Named breakage: reading a frameset whole makes a preparation's host
    memory scale with the NUMBER OF VALID TIMES.  Measured on real RRFS
    bytes (3 km CONUS, 45 pressure levels, one 300x300 target), a bare
    default prep peaked at 35.2 GiB for two valid times and 67.0 GiB for
    four -- 15.9 GiB per additional time, of which the frames read whole
    are the largest single term -- so the seven-time preparation the
    source's own forecast cadence asks for needed ~114 GiB and died on
    every box smaller than that.  Nothing downstream ever needs two
    valid times at once: the initialize loop takes them one at a time
    and keeps only the perimeter frames.

    So this reads the DOCUMENT once -- schema, stream length, the
    scalars and per-field digests of every frame -- and reads a frame's
    arrays only when that frame is asked for, keeping the most recent
    one so a caller that touches the same index twice does not pay
    twice.  What a frame costs is unchanged, and so are its bytes: the
    arrays are the same little-endian float64 values the memory-mapped
    read produced, verified against the same per-field digests.

    What moves is WHEN a corrupt array is caught.  It is caught when its
    frame is read rather than before the first frame is built.  No
    number from an unverified frame reaches a preparation either way --
    a mapped preparation publishes its output tree atomically at the end
    -- and every frame the route consumes is verified, because the route
    consumes them all.
    """

    def __init__(self, directory: str | Path, *, retain: object = None):
        self._directory = Path(directory)
        document = json.loads(
            (self._directory / FRAMES_DOCUMENT).read_text(encoding="utf-8"))
        schema = str(document.get("schema"))
        if schema != FRAMESET_SCHEMA:
            raise ValueError(
                f"{self._directory / FRAMES_DOCUMENT} declares schema "
                f"{schema!r}; this release reads {FRAMESET_SCHEMA!r}")
        stream_document = dict(document["stream"])
        self._stream = self._directory / str(
            stream_document.get("path", FRAMES_STREAM))
        size = self._stream.stat().st_size
        self._declared_bytes = int(stream_document["bytes"])
        if size != self._declared_bytes:
            raise ValueError(
                f"{self._stream} carries {size} bytes; the frameset "
                f"declares {stream_document['bytes']}")
        # The whole-stream digest is verified ONLY when the fields leave
        # bytes it alone would cover.  Both writers pack the fields
        # contiguously in document order, so :func:`_fields_tile_stream`
        # normally holds and every byte of the stream is inside exactly
        # one field -- whose own sha256 is checked at materialization,
        # over the same bytes, and binds dtype and shape as well.  The
        # breakage the fallback still prevents: a stream carrying bytes
        # no field claims -- padding, a gap, a truncated last field --
        # which per-field hashing would never look at.  It streams
        # through a fixed buffer, so it costs one pass and no residency.
        if not _fields_tile_stream(document, self._declared_bytes):
            digest = hashlib.sha256()
            with self._stream.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            observed = digest.hexdigest()
            if observed != str(stream_document["sha256"]):
                raise ValueError(
                    f"{self._stream} hashes to {observed}; the frameset "
                    f"declares {stream_document['sha256']}")
        self._entries = list(document["frames"])
        #: The object whose lifetime the stream file needs -- the
        #: engine's scratch directory handle.  Held here so the frames
        #: cannot outlive the bytes they read from.
        self._retain = retain
        self._cached_index: int | None = None
        self._cached_frame: Any = None

    # -- the document, without touching an array -------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def valid_times(self) -> tuple[datetime, ...]:
        return tuple(
            datetime.fromisoformat(str(entry["valid_time"]))
            for entry in self._entries)

    @property
    def members(self) -> tuple[str | None, ...]:
        return tuple(
            None if entry["member"] is None else str(entry["member"])
            for entry in self._entries)

    @property
    def mapping_sha256s(self) -> tuple[str, ...]:
        return tuple(str(entry["mapping_sha256"]) for entry in self._entries)

    def input_sha256(self, index: int) -> dict[str, str]:
        return {
            str(key): str(value)
            for key, value in dict(
                self._entries[index]["input_sha256"]).items()}

    def field_names(self, index: int) -> tuple[str, ...]:
        return tuple(
            str(field["name"]) for field in self._entries[index]["fields"])

    def field_digest(self, index: int, name: str) -> str:
        """The declared ``_array_sha256`` of one field.

        The same digest :meth:`_materialize` checks the read bytes
        against, so a receipt built from it names the array that
        actually reached the preparation -- an engine that wrote a
        digest its own bytes do not match refuses when the frame is
        read, before the tree is published.
        """

        for field in self._entries[index]["fields"]:
            if str(field["name"]) == name:
                return str(field["sha256"])
        raise KeyError(f"frame {index} carries no field {name!r}")

    def field_count(self, index: int) -> int:
        return len(self._entries[index]["fields"])

    def header(self, index: int) -> Any:
        return _header_from_document(self._entries[index]["header"])

    # -- one frame's arrays ----------------------------------------

    def field(self, index: int, name: str) -> Any:
        """One field of one frame, read and verified on its own.

        For the checks that need a single array -- the vertical
        coverage of the source column, the canonical terrain -- reading
        one field instead of a whole valid time is the difference
        between one array and the twenty a frame carries.
        """

        if self._cached_index == index:
            return self._cached_frame.fields[name]
        for document in self._entries[index]["fields"]:
            if str(document["name"]) == name:
                return self._read_field(index, document)
        raise KeyError(f"frame {index} carries no field {name!r}")

    def _read_field(self, index: int, document: Mapping[str, Any]) -> Any:
        import numpy as np

        from gpuwm.mapped_source import CanonicalField, _array_sha256

        name = str(document["name"])
        start = int(document["offset"])
        length = int(document["length"])
        shape = tuple(int(size) for size in document["shape"])
        dtype = str(document["dtype"])
        if dtype != STREAM_DTYPE:
            raise ValueError(
                f"frame {index} field {name!r} declares dtype "
                f"{dtype!r}; the stream carries {STREAM_DTYPE!r}")
        if start < 0 or start + length > self._declared_bytes:
            raise ValueError(
                f"frame {index} field {name!r} claims bytes "
                f"[{start}, {start + length}) of a "
                f"{self._declared_bytes}-byte stream")
        array = np.empty(shape, dtype=np.dtype(STREAM_DTYPE))
        buffer = memoryview(array).cast("B")
        if len(buffer) != length:
            raise ValueError(
                f"frame {index} field {name!r} declares {length} bytes "
                f"for shape {list(shape)}, which needs {len(buffer)}")
        with self._stream.open("rb") as handle:
            handle.seek(start)
            read = handle.readinto(buffer)
        if read != length:
            raise ValueError(
                f"frame {index} field {name!r} ends at byte "
                f"{start + int(read or 0)} of a stream that declares "
                f"{self._declared_bytes}")
        observed = _array_sha256(array)
        declared = str(document["sha256"])
        if observed != declared:
            raise ValueError(
                f"frame {index} field {name!r} hashes to {observed}; "
                f"the frameset declares {declared}")
        return CanonicalField(
            name=name,
            units=str(document["units"]),
            axes=tuple(str(axis) for axis in document["axes"]),
            location=str(document["location"]),
            staggering=str(document["staggering"]),
            values=array,
            missing_count=int(document["missing_count"]),
            source_references=tuple(
                str(value) for value in document["source_references"]),
        )

    def _materialize(self, index: int) -> Any:
        from gpuwm.mapped_source import MappedSourceFrame

        entry = self._entries[index]
        fields = {
            str(document["name"]): self._read_field(index, document)
            for document in entry["fields"]
        }
        return MappedSourceFrame(
            valid_time=datetime.fromisoformat(str(entry["valid_time"])),
            member=None if entry["member"] is None else str(entry["member"]),
            source_cycle=datetime.fromisoformat(str(entry["source_cycle"])),
            latitude=_axis_values(entry["latitude"], "latitude"),
            longitude=_axis_values(entry["longitude"], "longitude"),
            vertical_kind=str(entry["vertical_kind"]),
            vertical_units=str(entry["vertical_units"]),
            vertical_values=_axis_values(entry["vertical_values"], "vertical"),
            fields=fields,
            mapping_sha256=str(entry["mapping_sha256"]),
            input_sha256=self.input_sha256(index),
            grid_fingerprint=str(entry["grid_fingerprint"]),
            header=_header_from_document(entry["header"]),
        )

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _FrameSetSlice(self, range(*index.indices(len(self))))
        position = int(index)
        if position < 0:
            position += len(self._entries)
        if not 0 <= position < len(self._entries):
            raise IndexError(position)
        if self._cached_index == position:
            return self._cached_frame
        # The previous frame is dropped BEFORE the next is read, so the
        # peak is one valid time and not two.
        self._cached_index = None
        self._cached_frame = None
        frame = self._materialize(position)
        self._cached_index = position
        self._cached_frame = frame
        return frame

    def close(self) -> None:
        """Drop the retained frame and the engine scratch it read from."""

        self._cached_index = None
        self._cached_frame = None
        retain, self._retain = self._retain, None
        cleanup = getattr(retain, "cleanup", None)
        if cleanup is not None:
            cleanup()


class _FrameSetSlice(_ABCSequence):
    """A contiguous window on a :class:`FrameSet`, still one at a time."""

    def __init__(self, frames: FrameSet, positions: range):
        self._frames = frames
        self._positions = positions

    def __len__(self) -> int:
        return len(self._positions)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return _FrameSetSlice(
                self._frames, self._positions[index])
        return self._frames[self._positions[int(index)]]


def open_frameset(directory: str | Path, *, retain: object = None) -> FrameSet:
    """Open a frameset directory without reading a single array.

    ``retain`` is the object whose lifetime the stream file needs -- the
    engine's scratch directory handle -- so a caller can hand ownership
    over instead of deleting the bytes the frames still read from.
    """

    return FrameSet(directory, retain=retain)


def read_frameset(directory: str | Path) -> tuple[Any, ...]:
    """Rebuild EVERY canonical frame from a frameset directory.

    Every array is re-hashed against the document and the rebuilt
    dataclasses re-run their own validators, so an engine that wrote an
    array its manifest does not describe is refused rather than
    believed.  The whole set is resident when this returns; a route that
    consumes one valid time at a time wants :func:`open_frameset`.
    """

    return tuple(FrameSet(directory))




# --------------------------------------------------------------------
# Launching the engine
# --------------------------------------------------------------------

def engine_command(
    subcommand: str,
    *,
    engine: Path,
    mapping: Path,
    input_list: Path,
    output: Path,
    composition: Path | None = None,
    supplements: Mapping[str, Sequence[Path]] | None = None,
    provenance: Mapping[str, Path] | None = None,
    contributing_mappings: Mapping[str, Path] | None = None,
    input_manifest: Path | None = None,
    input_manifest_sha256: str | None = None,
) -> list[str]:
    """The exact argv the seam contract defines, in a stable order.

    Built as its own function so a test can read the command line a
    route would run without running it -- and so the hand-run in the
    verification law is a copy of this list, not a paraphrase.

    ``--contributing-mapping ROLE=PATH`` is an ADDENDUM to the argv line
    printed in the design's §3.1, which enumerated supplements and
    provenance but not the third role-bound binding a cross-source
    composition carries: each contributing source's own sealed mapping
    document, whose bytes the composition pins by SHA-256.  ``compose``
    cannot decode a cross-source composition without it -- the named
    breakage is that every ``field_sources`` binding would resolve to no
    mapping and refuse -- so it is spelled here in the same ROLE=PATH
    grammar as its siblings and recorded for the engine lane.
    """

    if subcommand not in ("decode", "compose", "inspect"):
        raise ValueError(
            f"unknown mapped-engine subcommand {subcommand!r}; the "
            "contract defines decode, compose and inspect")
    command = [
        str(engine), subcommand,
        "--mapping", str(mapping),
        "--input-list", str(input_list),
        "--output", str(output),
    ]
    if composition is not None:
        command.extend(("--composition", str(composition)))
    for role in sorted(supplements or {}):
        for path in (supplements or {})[role]:
            command.extend(("--supplement", f"{role}={path}"))
    for role in sorted(provenance or {}):
        command.extend(("--provenance", f"{role}={(provenance or {})[role]}"))
    for role in sorted(contributing_mappings or {}):
        command.extend((
            "--contributing-mapping",
            f"{role}={(contributing_mappings or {})[role]}",
        ))
    if input_manifest is not None:
        command.extend(("--input-manifest", str(input_manifest)))
        command.extend(("--input-manifest-sha256", str(input_manifest_sha256)))
    return command


def parse_refusal(stderr: str) -> dict[str, str] | None:
    """The refusal object, or ``None`` when the last line is not one.

    The contract puts it on the LAST stderr line so a decoder's own
    chatter above it cannot be mistaken for the verdict.
    """

    for line in reversed(stderr.splitlines()):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            return None
        try:
            document = json.loads(line)
        except ValueError:
            return None
        if not isinstance(document, dict):
            return None
        if str(document.get("schema")) != REFUSAL_SCHEMA:
            return None
        return {
            "class": str(document.get("class", "")),
            "message": str(document.get("message", "")),
            "remedy": str(document.get("remedy", "")),
        }
    return None


def refusal_error(refusal: Mapping[str, str], command: Sequence[str]) -> Exception:
    """Map an engine refusal onto the exception the Python engine raises.

    An unlisted class is itself a contract defect: it means the engine
    grew a refusal the Python side was never taught, and the two would
    then disagree about what a caller may catch.  That re-raises as
    ``RuntimeError`` naming the unknown class rather than being widened
    into whatever exception looks closest.
    """

    name = str(refusal.get("class", ""))
    message = str(refusal.get("message", "")).strip()
    remedy = str(refusal.get("remedy", "")).strip()
    text = message if not remedy else f"{message}; {remedy}"
    exception = REFUSAL_CLASSES.get(name)
    if exception is None:
        return RuntimeError(
            f"{ENGINE_NAME} refused with unknown class {name!r}: {text}.  "
            "This release's REFUSAL_CLASSES table does not carry it, so "
            "the engine and gpuwm speak different contracts; rebuild the "
            "engine from a matching checkout")
    return exception(text)


def _drain_progress(stdout: str, on_progress: Callable[[dict], None] | None):
    """Parse the JSON-lines progress stream; return its last object."""

    receipt: dict[str, object] | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            document = json.loads(line)
        except ValueError:
            continue
        if not isinstance(document, dict):
            continue
        if str(document.get("schema")) != PROGRESS_SCHEMA:
            continue
        receipt = document
        if on_progress is not None:
            on_progress(document)
    return receipt


def run_engine(
    subcommand: str,
    *,
    mapping: str | Path,
    files: Sequence[str | Path],
    output: str | Path,
    composition: str | Path | None = None,
    supplements: Mapping[str, Sequence[str | Path]] | None = None,
    provenance: Mapping[str, str | Path] | None = None,
    contributing_mappings: Mapping[str, str | Path] | None = None,
    input_manifest: str | Path | None = None,
    input_manifest_sha256: str | None = None,
    engine: str | Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> dict[str, object]:
    """Run one engine subcommand; refusals become Python exceptions.

    Returns ``{"output", "receipt", "stdout", "command"}``.  The caller
    reads frames with :func:`read_frameset` (``decode``/``compose``) or
    the inspection document off stdout (``inspect``); this function does
    not choose for it, because the three subcommands have three
    different products and one wrapper that guessed would be a place for
    them to drift apart.

    The input list is written for every call -- never an argv of file
    paths.  A field-per-file source runs to hundreds of inputs and
    Windows caps a command line at 32 KB; the list file is the contract's
    only transport for that reason.
    """

    binary = Path(engine) if engine is not None else require_engine()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    input_list = output / "inputs.txt"
    input_list.write_text(
        "".join(f"{Path(path)}\n" for path in files), encoding="utf-8")
    command = engine_command(
        subcommand,
        engine=binary,
        mapping=Path(mapping),
        input_list=input_list,
        output=output,
        composition=None if composition is None else Path(composition),
        supplements={
            str(role): tuple(Path(path) for path in paths)
            for role, paths in (supplements or {}).items()
        },
        provenance={
            str(role): Path(path)
            for role, path in (provenance or {}).items()
        },
        contributing_mappings={
            str(role): Path(path)
            for role, path in (contributing_mappings or {}).items()
        },
        input_manifest=(
            None if input_manifest is None else Path(input_manifest)),
        input_manifest_sha256=input_manifest_sha256,
    )
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        refusal = parse_refusal(completed.stderr or "")
        if refusal is None:
            tail = (completed.stderr or "").strip().splitlines()
            raise RuntimeError(
                f"{ENGINE_NAME} {subcommand} exited "
                f"{completed.returncode} without a "
                f"{REFUSAL_SCHEMA} object on its last stderr line, so "
                "there is no class to map and no remedy to relay: "
                + (tail[-1] if tail else "it printed nothing"))
        raise refusal_error(refusal, command)
    receipt = _drain_progress(completed.stdout or "", on_progress)
    return {
        "output": output,
        "receipt": receipt,
        "stdout": completed.stdout or "",
        "command": command,
    }


__all__ = [
    "ABI_MARKER",
    "DEFAULT_ENGINE",
    "DEFAULT_ENGINE_BLOCKER",
    "ENGINES",
    "ENGINE_CAPABILITIES",
    "ENGINE_GAPS",
    "ENGINE_CRATE_RELATIVE",
    "ENGINE_ENV",
    "ENGINE_NAME",
    "ENGINE_PATH_ENV",
    "ENGINE_PYTHON",
    "ENGINE_RUST",
    "EngineUnavailable",
    "FRAMESET_SCHEMA",
    "FRAMES_DOCUMENT",
    "FRAMES_STREAM",
    "FrameSet",
    "MAPPED_ROUTE_SUBCOMMAND",
    "PROGRESS_SCHEMA",
    "RECORD_INVENTORY_SCHEMA",
    "REFUSAL_SCHEMA",
    "REFUSAL_CLASSES",
    "STREAM_DTYPE",
    "engine_command",
    "engine_record_inventory",
    "engine_supports",
    "engine_remedy",
    "find_engine",
    "frameset_document",
    "open_frameset",
    "parse_refusal",
    "read_frameset",
    "refusal_error",
    "require_engine",
    "resolve_engine",
    "run_engine",
    "write_frameset",
]

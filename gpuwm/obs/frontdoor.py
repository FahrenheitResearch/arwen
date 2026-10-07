"""Locate and drive one of the vendored observation front doors.

:mod:`gpuwm.obs.nexrad` established the shape: resolution ladder, ``--abi``
probe, JSON record with a checked schema, for one binary. The battery adds
four more (``rw_mrms``, ``rw_stage4``, ``rw_asos``, ``rw_goes``), and four
more copies of that shape is four places for the ladder to drift. So the
shape lives here once and each instrument module supplies its own names.

``gpuwm obs`` (:mod:`gpuwm.obs.cli`) is the product door onto these.
Until it existed this module resolved four binaries and refused for
four, and a whole-tree search for its own name returned no importers:
it shipped, it resolved, it refused, and nobody could invoke it.

The ladder, unchanged:

1. the instrument's environment variable naming the built file (a missing
   file it names is a hard error, never silently skipped);
2. a source checkout's ``tools/rustwx/target/{release,debug}``;
3. ``<root>/libexec/bridges`` beside the package;
4. ``~/.gpuwm/bridges``.

Nothing here runs cargo.  Resolution has one side effect and one
only: an artifact found in ``~/.gpuwm/bridges`` that is not the one this
release pinned is re-fetched before it is handed to a door
(:func:`gpuwm.bridges.require_release_pin`).  ``gpuwm doctor`` resolves
inside :func:`gpuwm.bridges.inspection_only`, where there is no side
effect at all, so the report still says what the estate IS.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from dataclasses import dataclass

from gpuwm.bridges import (RUSTWX_CRATE_RELATIVE, artifact_remedy,
                           default_bridge_dir,
                           legacy_bridge_candidates, lazy_build_hints,
                           rustwx_build_hint,
                           accept_resolved, executable_name,
                           packaged_bridge_dir)

_PROBE_TIMEOUT_S = 20

#: ``CARGO_BUILD_HINT``: the one-liner that builds the front doors, from
#: a checkout root, spelled for the shell rule when it is read.
__getattr__ = lazy_build_hints(
    __name__, CARGO_BUILD_HINT=RUSTWX_CRATE_RELATIVE)


def _repo_root() -> Path:
    """The directory containing the ``gpuwm`` package."""

    return Path(__file__).resolve().parent.parent.parent


def crate_dir() -> Path:
    """The vendored Rust workspace of a source checkout (may not exist)."""

    return _repo_root() / "tools" / "rustwx"


@dataclass(frozen=True)
class FrontDoor:
    """One observation front door: its names, its ABI, its remedy.

    ``abi_marker`` is the exact ``--abi`` line the Python wrapper was
    written against. It is not a version number: it names the record
    contract, so a rebuilt-but-unchanged binary still matches and a binary
    whose records changed shape does not.
    """

    name: str
    env_var: str
    subject: str
    abi_marker: str

    def candidates(self) -> tuple[Path, ...]:
        """Deterministic candidate paths, best first."""

        filename = executable_name(self.name)
        candidates: list[Path] = []
        override = os.environ.get(self.env_var)
        if override:
            candidates.append(Path(override))
        candidates.extend((
            crate_dir() / "target" / "release" / filename,
            crate_dir() / "target" / "debug" / filename,
            _repo_root() / "libexec" / "bridges" / filename,
            packaged_bridge_dir() / filename,
            default_bridge_dir() / filename,
            *legacy_bridge_candidates(filename),
        ))
        return tuple(candidates)

    def find(self) -> Path | None:
        """First existing candidate, or None.

        An environment override naming a missing file is a hard error:
        explicit configuration must fail loudly rather than fall through to
        a different executable.
        """

        override = os.environ.get(self.env_var)
        for candidate in self.candidates():
            if candidate.is_file():
                return accept_resolved(candidate.resolve())
            if override and candidate == Path(override):
                raise FileNotFoundError(
                    f"{self.env_var} names a missing file: {candidate}")
        return None

    def require(self) -> Path:
        """The binary, or a RuntimeError carrying this install's remedy."""

        found = self.find()
        if found is None:
            raise RuntimeError(
                f"{self.subject} ({self.name}) is not built or not found.\n"
                f"{self.remedy()}")
        return found

    def remedy(self) -> str:
        """The remedy for a missing front door, true for THIS install."""

        return artifact_remedy(
            env_var=self.env_var, filename=executable_name(self.name),
            subject=self.subject, crate_relative=RUSTWX_CRATE_RELATIVE,
            one_liner=rustwx_build_hint(), artifact=self.name)

    def probe(self, path: Path) -> tuple[bool, str]:
        """``--version`` then ``--abi``: is this the binary we expect?"""

        try:
            version = subprocess.run(
                [str(path), "--version"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as error:
            return False, f"{path} did not run: {error}"
        if version.returncode != 0:
            return False, f"{path} --version exited {version.returncode}"
        transcript = (version.stdout or "").strip()
        try:
            abi = subprocess.run(
                [str(path), "--abi"], capture_output=True, text=True,
                errors="replace", timeout=_PROBE_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as error:
            return False, f"{path} --abi did not run: {error}"
        from gpuwm import provenance_gate

        # A GPUWM_BRIDGE_SOURCE_REV line after the contract is not part of
        # it (rw_ensbatch's whole-answer compare refused its own checkout's
        # build); it is judged by the shared stamp check.
        contract, revision = provenance_gate.split_abi_answer(abi.stdout)
        if abi.returncode != 0 or contract != self.abi_marker:
            return False, (f"{transcript} -- --abi does not match the record "
                           "contract this gpuwm expects; rebuild it: "
                           f"{rustwx_build_hint()}")
        if revision is not None:
            refusal = provenance_gate.abi_revision_refusal(
                path, revision, env_var=self.env_var)
            if refusal is not None:
                return False, (f"{transcript} -- --abi matches the record "
                               f"contract, but {refusal}; rebuild it: "
                               f"{rustwx_build_hint()}")
        return True, f"{transcript} -- --abi matches the record contract"

    def run(self, subcommand: str, arguments: list[str], *,
            schema: str, binary: Path | None = None) -> dict:
        """Run one subcommand and parse its JSON record.

        A non-zero exit, unparseable output, or a record declaring a schema
        other than ``schema`` are all failures of the same kind: this is not
        the binary this wrapper was written against, and continuing would
        read a record shape nobody checked.
        """

        # A reviewed caller may bind the already-probed executable so a
        # companion resolver cannot change the provider between stages.
        binary = self.require() if binary is None else Path(binary)
        command = [str(binary), subcommand, *arguments]
        try:
            result = subprocess.run(
                command, capture_output=True, text=True, errors="replace")
        except OSError as error:
            raise RuntimeError(
                f"{self.name} failed to launch: {error}") from error
        if result.returncode != 0:
            tail = [line for line in (result.stderr or "").splitlines()
                    if line.strip() and not line.startswith("usage:")
                    and not line.startswith(" ")]
            detail = tail[0] if tail else f"exit {result.returncode}"
            raise RuntimeError(f"{self.name} {subcommand}: {detail}")
        try:
            record = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"{self.name} {subcommand} did not print a JSON record: "
                f"{error}") from error
        if not isinstance(record, dict):
            raise RuntimeError(f"{self.name} {subcommand} did not print a JSON object")
        if record.get("schema") != schema:
            raise RuntimeError(
                f"{self.name} {subcommand} printed schema "
                f"{record.get('schema')!r}, expected {schema!r}")
        return record


#: Composite reflectivity and one-hour precipitation with coverage masks.
MRMS = FrontDoor(
    name="rw_mrms",
    env_var="GPUWM_RW_MRMS",
    subject="the MRMS front door",
    abi_marker=(
        "gpuwm-obs.mrms-fetch.v1\tproduct\twindow\tbucket\tfiles\tbytes\t"
        "sha256\tgpuwm-obs.obs-grid.v1\tcomposite_reflectivity\tdBZ\t"
        "precipitation_accumulation\tmm\taccumulation_seconds"),
)

#: The Stage-IV precipitation front door.
STAGE4 = FrontDoor(
    name="rw_stage4",
    env_var="GPUWM_RW_STAGE4",
    subject="the Stage-IV precipitation front door",
    abi_marker=(
        "gpuwm-obs.stage4-fetch.v1\taccumulation\twindow\tarchive\tfiles\t"
        "bytes\tsha256\tgpuwm-obs.obs-grid.v1\t"
        "precipitation_accumulation\tmm"),
)

#: The ASOS/METAR surface front door.
ASOS = FrontDoor(
    name="rw_asos",
    env_var="GPUWM_RW_ASOS",
    subject="the surface observation front door",
    abi_marker=(
        "gpuwm-obs.asos-surface.v2\tstations\treports\tprovenance\t"
        "observation_time\ttemperature_2m\tdewpoint_2m\twind_speed_10m\t"
        "mslp\tK\tm s-1\tPa\t"
        # The neutral-table subcommands (table, networks, awc) the global
        # observation streams call; appended after the v2 line so the
        # surface contract above is unchanged.
        "gpuwm-obs.asos-table.v1\tgpuwm-obs.table.v2\t"
        # fetch/decode --product asos1min: a v2 record may carry the
        # one-minute provenance, so a binary without the route is stale.
        "iem-asos-1min"),
)

#: The GOES ABI cloud-product front door.
#:
#: The satellite twin of ``rw_nexrad``: it writes the ``.goespack``
#: files :mod:`gpuwm.obs.goes_pack` reads.  It was the one producer in
#: the obs estate with no resolver at all -- the reader shipped, the
#: writer was reachable only by name from a checkout with a Rust
#: toolchain -- so a wheel user could parse a GOES pack and had no way
#: to obtain one.  Its ``--abi`` names the fetch record and both pack
#: schemas, the same three-part contract the other doors pin.
GOES = FrontDoor(
    name="rw_goes",
    env_var="GPUWM_RW_GOES",
    subject="the GOES cloud-product front door",
    abi_marker=("gpuwm-obs.goes-fetch.v1\tgpuwm-obs.goes-cwp.v2\t"
                "gpuwm-obs.goes-cloudtop.v2\t"
                # The Level 1b radiance family and the clear-sky forward
                # operator (bt, colocate, superobs, quicklook, forward).
                "gpuwm-obs.goes-bt.v1\tgpuwm-da.abi-forward.v1\tgpuwm-obs.goes-aod.v1"),
)

#: The European radar composite front door.
#:
#: ``corner_check`` is in the marker on purpose. The composite is published
#: on an *ellipsoidal* Lambert azimuthal equal-area grid and says so in its
#: own ``projdef``; inverting it on a sphere displaces the northern corners
#: by about nine cells on a 1 km grid. The frame states its four corner
#: coordinates, so the georeference has an oracle inside it, and the front
#: door proves itself against that oracle on every decode. A build without
#: that check is a different instrument and must not answer to this name.
#:
#: ``no_coverage`` and ``no_echo`` are in it for the same reason: the archive
#: marks *unobserved* and *observed empty* with different sentinels, and a
#: build that collapsed them would delete every correct negative a skill
#: score is built on while still printing a plausible record.
OPERA = FrontDoor(
    name="rw_opera",
    env_var="GPUWM_RW_OPERA",
    subject="the European radar composite front door",
    abi_marker=(
        "gpuwm-obs.opera-fetch.v1\tcollection\twindow\tlinks\tbytes\tsha256\t"
        "gpuwm-obs.obs-grid.v1\tcomposite_reflectivity\tdBZ\tno_coverage\t"
        "no_echo\tcorner_check"),
)

#: The European per-site polar-volume front door.
#:
#: The composite's counterpart and not a variant of it: ``rw_opera`` serves a
#: 2-D reflectivity grid that can grade a forecast, and ``rw_odim`` serves the
#: per-site polar scan that can be *assimilated*, because it carries radial
#: velocity and the elevation geometry an observation operator needs.
#:
#: Four things are in the marker because a build that dropped any of them
#: would still print a plausible record:
#:
#: ``undetect``/``nodata`` -- ODIM reserves two sentinels per moment and they
#: are opposite claims. ``undetect`` is the radar reporting it looked and
#: found nothing, which is a clear-air observation and the most common true
#: one on any scan; ``nodata`` is the radar reporting it did not look. A build
#: that collapsed them would delete every correct negative while still
#: decoding.
#:
#: ``sentinel_ambiguous`` -- Finnish ``VRADH`` declares both sentinels as the
#: same raw value, so some gates are genuinely undecidable. They get their own
#: code and are reported as not observed. Measured at 721,898 gates in one
#: volume, so this is not a corner case.
#:
#: ``nyquist_granularity`` -- ODIM has one ``NI`` per cut and no per-ray
#: Nyquist. The pack says so rather than broadcasting one number across the
#: radials, which is what lets the dealiaser tell "this format never had
#: per-radial originals" from "they were dropped".
#:
#: ``assembled``/``manifest_sha256`` -- some national feeds publish one file
#: per elevation and quantity, and a pack built from many files cannot quote a
#: file digest. A build without the assembly leg cannot serve Germany at all.
ODIM = FrontDoor(
    name="rw_odim",
    env_var="GPUWM_RW_ODIM",
    subject="the European polar-volume front door",
    abi_marker=(
        "gpuwm-obs.odim-pack.v1\tgpuwm-obs.odim-nyquist.v1\t"
        "gpuwm-obs.odim-volumes.v1\tgpuwm-obs.radar-sweeps.v3\t"
        "site\telangle\tnbins\tnrays\trscale\trstart\tazimuth\tnyquist_ms\t"
        "nyquist_source\tnyquist_granularity\tcensor\tmeasured\tundetect\t"
        "nodata\tsentinel_ambiguous\tassembled\tmanifest_sha256\tstamp"),
)

#: NCEP's prepbufr, the conventional-observation front door.
#:
#: It decodes the file's own dictionary and every report in Rust and
#: writes ``gpuwm-obs.table.v2`` rows with GSI's use rule on NCEP's
#: quality marks applied inside the door, so a reader of its table never
#: sees a value whose mark is 4 or more.  ``mark-limit-4`` and
#: ``pressure-mark`` are in the marker because a build without them would
#: still print a plausible table of rejected reports.  ``surface-gsdqc-2``
#: (GSI's surface dewpoint and calm mesonet wind rules) and
#: ``error-table`` (each row's error is GSI's, from the conventional error
#: table) are in it because a build without them writes rows GSI does not
#: use and errors GSI does not assign.  It is not in
#: :data:`FRONT_DOORS` yet: joining it needs a ``gpuwm obs prepbufr``
#: subcommand, its line in the doctor's door names and a bundle entry.
PREPBUFR = FrontDoor(
    name="rw_prepbufr",
    env_var="GPUWM_RW_PREPBUFR",
    subject="the conventional-observation (prepbufr) front door",
    abi_marker=(
        "gpuwm-obs.prepbufr-fetch.v1\tgpuwm-obs.prepbufr-decode.v1\t"
        "gpuwm-obs.prepbufr-table.v2\tgpuwm-obs.table.v2\t"
        "gpuwm-obs.prepbufr-listing.v1\tmark-limit-4\tpressure-mark\t"
        "upper-moisture-9-to-2\tvad-superob\tprofiler-400-hpa\t"
        "surface-gsdqc-2\terror-table"),
)

#: Every front door the battery drives, by instrument id.
#:
#: ``goes`` is the satellite twin of ``nexrad``: it writes the ``.goespack``
#: files :mod:`gpuwm.obs.goes_pack` reads.
#:
#: ``opera`` is the composite-reflectivity grader for domains MRMS does not
#: cover. It serves the same quantity in the same units through the same pack
#: schema, so a consumer picks a grader by region rather than by code path.
#:
#: ``odim`` is the assimilable European route and pairs with ``nexrad`` in the
#: same way: both write ``gpuwm-obs.radar-sweeps`` packs that the one sweeps
#: reader accepts, so a superob picks a decoder by which network covers the
#: domain rather than by which code path it has to take.
AIRNOW = FrontDoor(
    name="rw_airnow", env_var="GPUWM_RW_AIRNOW", subject="the AirNow front door",
    abi_marker="gpuwm-obs.airnow-fetch.v1\tgpuwm-obs.airnow-table.v1\tgpuwm-obs.table.v2",
)

FRONT_DOORS = {"mrms": MRMS, "stage4": STAGE4, "asos": ASOS, "goes": GOES,
               "opera": OPERA, "odim": ODIM, "airnow": AIRNOW}


__all__ = ["AIRNOW", "ASOS", "CARGO_BUILD_HINT", "FRONT_DOORS", "GOES", "MRMS", "ODIM",
           "OPERA", "PREPBUFR",
           "STAGE4", "FrontDoor", "crate_dir"]

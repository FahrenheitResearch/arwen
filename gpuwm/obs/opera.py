"""Drive the European radar-composite front door (``rw_opera``).

The wrapper is deliberately thin, because the shape it needs already exists:
:class:`gpuwm.obs.frontdoor.FrontDoor` owns the resolution ladder, the
``--abi`` probe and the schema-checked JSON record, and this module supplies
the names and the six subcommand signatures. Nothing here decodes anything.

**What this route is, and what it is not.** ``rw_opera`` serves the EUMETNET
OPERA *composite* — one 1 km column-maximum reflectivity grid over Europe,
the direct counterpart of the MRMS composite this pipeline already scores
against, and the reason a European case can be graded at all. It is not a
polar volume: there is no radial velocity in it, no elevation geometry, and
therefore no path from it into the superob layer. That gap is real, it is
upstream, and :func:`polar_volume_support` states it in one place so a caller
asking for velocity is told why rather than handed reflectivity.

**Two sentinels.** The archive marks *no radar coverage* and *no echo* with
different values, and they are opposite claims: the first is unobserved, the
second is the network reporting that it looked and found nothing. The second
is the most common true observation on any frame and every correct negative a
skill score is built on. The front door separates them and counts both; this
module's contribution is to pin that in the ABI marker, so a binary that
stopped separating them cannot be probed as if it still did.
"""

from __future__ import annotations

from pathlib import Path

from gpuwm.obs.frontdoor import OPERA

#: Record schemas the six subcommands print.
COVERAGE_SCHEMA = "gpuwm-obs.opera-coverage.v1"
NEAREST_SCHEMA = "gpuwm-obs.opera-nearest.v1"
FETCH_SCHEMA = "gpuwm-obs.opera-fetch.v1"
DECODE_SCHEMA = "gpuwm-obs.opera-decode.v1"
GRID_SCHEMA = "gpuwm-obs.opera-grid.v1"
VERIFY_SCHEMA = "gpuwm-obs.opera-verify.v1"

#: The pack schema ``decode`` writes -- the same one ``rw_mrms`` writes, which
#: is what lets one consumer read a European frame and a North-American one.
PACK_SCHEMA = "gpuwm-obs.obs-grid.v1"

#: The composite's published cadence, measured live on 2026-08-12: one frame
#: every five minutes, stamped on the minute.
CADENCE_SECONDS = 300

#: How far from a requested valid time a frame may sit and still be treated
#: as coincident. Half the cadence would refuse a frame that is merely the
#: other side of the interval, so this is the cadence itself: at most one
#: frame can be nearer, and nothing further away than one full interval is
#: ever admitted.
DEFAULT_MATCH_SECONDS = CADENCE_SECONDS

#: The source label a decoded frame carries into the seam's provenance.
SOURCE_LABEL = "opera"

#: The product label the front door stamps on a decoded composite.
PRODUCT_LABEL = "opera-comp-dbzh-max"

def _window(*, start: str, end: str, limit: int | None) -> list[str]:
    command = ["--start", start, "--end", end]
    if limit is not None:
        command += ["--limit", str(limit)]
    return command


def run_coverage(*, start: str, end: str, limit: int | None = None) -> dict:
    """Which composite frames a window resolves to, moving no payload."""

    return OPERA.run("coverage", _window(start=start, end=end, limit=limit),
                     schema=COVERAGE_SCHEMA)


def run_nearest(*, valid_time: str,
                window_seconds: int | None = None) -> dict:
    """The one frame nearest ``valid_time``, or a refusal naming the gap."""

    command = ["--valid-time", valid_time]
    if window_seconds is not None:
        command += ["--window-seconds", str(int(window_seconds))]
    return OPERA.run("nearest", command, schema=NEAREST_SCHEMA)


def run_fetch(*, start: str, end: str, out: Path,
              limit: int | None = None) -> dict:
    """Download the frames in the window into ``out``, one sha256 apiece."""

    return OPERA.run(
        "fetch", [*_window(start=start, end=end, limit=limit),
                  "--out", str(out)],
        schema=FETCH_SCHEMA)


def _decode_args(*, file: Path, out: Path,
                 bbox: tuple[float, float, float, float] | None,
                 no_echo_dbz: float | None) -> list[str]:
    command = ["--file", str(file), "--out", str(out)]
    if bbox is not None:
        command += ["--bbox", ",".join(f"{value:g}" for value in bbox)]
    if no_echo_dbz is not None:
        command += ["--no-echo-dbz", f"{no_echo_dbz:g}"]
    return command


def run_decode(*, file: Path, out: Path,
               bbox: tuple[float, float, float, float] | None = None,
               no_echo_dbz: float | None = None) -> dict:
    """Turn one ODIM frame into a ``gpuwm-obs.obs-grid.v1`` pack.

    ``bbox`` is ``(west, south, east, north)`` in degrees. Passing one is
    strongly advised rather than required: the published grid is 3800x4400,
    which is 134 MB of float64 values and 267 MB of coordinates per frame.
    """

    return OPERA.run(
        "decode",
        _decode_args(file=file, out=out, bbox=bbox, no_echo_dbz=no_echo_dbz),
        schema=DECODE_SCHEMA)


def run_grid(*, file: Path, out: Path,
             bbox: tuple[float, float, float, float] | None = None) -> dict:
    """Write the frame's latitude/longitude once, as a ``obs-geo.v1`` pack.

    The same ``bbox`` the values were decoded with must be passed here: the
    geometry pack and the value pack are one grid, and a consumer that read
    two different subsets of it would score each cell against the wrong
    coordinates.
    """

    return OPERA.run("grid",
                     _decode_args(file=file, out=out, bbox=bbox,
                                  no_echo_dbz=None),
                     schema=GRID_SCHEMA)


def run_verify(*, pack: Path) -> dict:
    """Re-prove a pack's header, array index and payload digest."""

    return OPERA.run("verify", ["--pack", str(pack)], schema=VERIFY_SCHEMA)


def polar_volume_support() -> tuple[bool, str]:
    """Can this route serve per-site polar volumes? ``(True, how)``.

    Stated as a function with a reason rather than left as an absence,
    because a caller planning a velocity-assimilating cycle needs to be told
    which of "Europe has no velocity" and "nothing here can read it" is
    true. For most of this pipeline's life the honest answer was the second
    one and this function said so, naming the vendored composite decoder that
    required a rank-2 ``/dataset1/data1/data`` and a ``/where`` ``projdef``
    and therefore refused every polar scan at the projection check.

    That is no longer the gap. :mod:`gpuwm.obs.odim` drives ``rw_odim``,
    which reads ODIM ``PVOL`` and ``SCAN`` objects and writes the same
    ``gpuwm-obs.radar-sweeps.v3`` pack the superob layer already consumes.
    **This route is still not the one that serves them** -- ``rw_opera``
    decodes the composite and only the composite, and adding polar decoding
    to it would give one binary two products with two geometries. The pair is
    the answer, and this function points at the other half rather than
    claiming a capability it does not itself have.
    """

    return True, (
        "per-site polar volumes are served by gpuwm.obs.odim (rw_odim), not "
        "by this composite route: rw_opera decodes the 2D LAEA grid and "
        "nothing else. rw_odim reads ODIM PVOL and SCAN objects -- including "
        "the split single-sweep files Germany publishes, assembled by "
        "nominal time -- and writes gpuwm-obs.radar-sweeps.v3, which is the "
        "pack gpuwm.obs.sweeps, superob, radar_grid and the LETKF adapter "
        "already read. Radial velocity from Europe is therefore assimilated "
        "by exactly the code that assimilates it from a NEXRAD volume. "
        "Reach it with `gpuwm obs radar volumes` and `gpuwm obs radar pack`")


__all__ = ["CADENCE_SECONDS", "COVERAGE_SCHEMA", "DECODE_SCHEMA",
           "DEFAULT_MATCH_SECONDS", "FETCH_SCHEMA", "GRID_SCHEMA",
           "NEAREST_SCHEMA", "OPERA", "PACK_SCHEMA", "PRODUCT_LABEL",
           "SOURCE_LABEL", "VERIFY_SCHEMA", "polar_volume_support",
           "run_coverage", "run_decode", "run_fetch", "run_grid",
           "run_nearest", "run_verify"]

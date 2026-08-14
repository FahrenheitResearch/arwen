"""Drive the European polar-volume front door (``rw_odim``).

This is the other half of :mod:`gpuwm.obs.opera`. That module serves the
EUMETNET OPERA *composite*: one 1 km column-maximum reflectivity grid over
Europe, which is a grader and not an assimilable observation, because a
composite has no radial velocity, no elevation geometry and no path into the
superob layer. This module serves the **per-site polar volume**, which has
all three.

The wrapper is thin for the same reason ``opera.py`` is:
:class:`gpuwm.obs.frontdoor.FrontDoor` owns the resolution ladder, the
``--abi`` probe and the schema-checked JSON record, and this module supplies
the names and the three subcommand signatures. Nothing here decodes anything.

**What the pack it writes joins.** ``rw_odim pack`` writes
``gpuwm-obs.radar-sweeps.v3``, the same container ``rw_nexrad decode``
writes, read by the same :func:`gpuwm.obs.sweeps.read_sweep_pack`. Everything
below that seam -- dealias, superob, ``radar_grid``, ``da.obs_radar``, LETKF
-- is not American and needs no European variant. That is the whole design:
one new pack writer, so that a European radial velocity is assimilated by
exactly the code that assimilates an American one.

**Two shapes of volume, and they are not interchangeable.** The Netherlands
and Romania publish one ``PVOL`` file per volume. Germany publishes one
``SCAN`` file per (elevation, quantity) pair, so a ten-elevation volume
carrying ``DBZH`` and ``VRADH`` arrives as twenty files that mean nothing
apart. :func:`run_pack` reads the first shape and :func:`run_pack_dir` the
second; :func:`run_volumes` says which shape a directory holds, and is what a
caller runs before deciding. A directory holding two nominal times is refused
rather than resolved by recency, because a volume glued out of two scans has
a wind field from the wrong minute in some of its columns and nothing
downstream can see it.

**Antenna altitude comes from the file.** ``/where/height`` is in every ODIM
volume, so the pack answers site altitude without consulting
``radar_sites_odim.json`` and :func:`gpuwm.obs.radar_sites.require_assimilable`
never has to guess one.
"""

from __future__ import annotations

from pathlib import Path

from gpuwm.obs.frontdoor import ODIM

#: The records the three subcommands print.
PACK_SCHEMA = "gpuwm-obs.odim-pack.v1"
NYQUIST_SCHEMA = "gpuwm-obs.odim-nyquist.v1"
VOLUMES_SCHEMA = "gpuwm-obs.odim-volumes.v1"

#: The sweep-pack schema ``pack`` writes. Held here as well as in
#: :mod:`gpuwm.obs.sweeps` so a caller can check what it is about to get
#: without decoding anything.
SWEEP_PACK_SCHEMA = "gpuwm-obs.radar-sweeps.v3"

#: The source label a decoded volume carries into the seam's provenance.
SOURCE_LABEL = "odim"

#: What ``nyquist_granularity`` says for every ODIM sweep, and why it is not
#: a parameter: ODIM declares one ``/datasetN/how/NI`` for a whole cut and has
#: no per-ray Nyquist anywhere in its model. The pack therefore omits the
#: per-radial array that a NEXRAD pack carries rather than broadcasting one
#: number across 360 radials, which would assert that originals were kept
#: when there was only ever one original.
NYQUIST_GRANULARITY = "sweep"


def _pack_options(*, quantities: list[str] | None,
                  max_elevation_deg: float | None,
                  max_range_km: float | None) -> list[str]:
    command: list[str] = []
    if quantities:
        command += ["--quantities", ",".join(quantities)]
    if max_elevation_deg is not None:
        command += ["--max-elevation-deg", f"{float(max_elevation_deg):g}"]
    if max_range_km is not None:
        command += ["--max-range-km", f"{float(max_range_km):g}"]
    return command


def run_pack(*, file: Path, out: Path,
             quantities: list[str] | None = None,
             max_elevation_deg: float | None = None,
             max_range_km: float | None = None) -> dict:
    """Decode one whole-volume ODIM file into a sweep pack.

    ``quantities`` are ODIM spellings (``DBZH``, ``VRADH``); omitting it
    carries every quantity the volume holds, which is a lot of payload for a
    volume that has nine of them.
    """

    return ODIM.run(
        "pack",
        ["--file", str(file), "--out", str(out),
         *_pack_options(quantities=quantities,
                        max_elevation_deg=max_elevation_deg,
                        max_range_km=max_range_km)],
        schema=PACK_SCHEMA)


def run_pack_dir(*, directory: Path, out: Path, stamp: str | None = None,
                 quantities: list[str] | None = None,
                 max_elevation_deg: float | None = None,
                 max_range_km: float | None = None) -> dict:
    """Assemble a directory of single-sweep files into one sweep pack.

    ``stamp`` names the nominal time to assemble, in the spelling
    :func:`run_volumes` reports (``YYYYmmddTHHMMSSZ``). It may be omitted only
    when the directory holds exactly one volume; a directory holding several
    is a hard error naming them, never the newest one silently.

    The record's ``source_sha256`` is then a **manifest** digest -- the
    SHA-256 of the sorted ``"<sha256>  <name>"`` lines of the members -- not
    any file's digest, because an assembled volume has no bytes of its own.
    The ``assembled`` block carries the members so a reader can still name the
    actual bytes.
    """

    command = ["--dir", str(directory), "--out", str(out)]
    if stamp is not None:
        command += ["--stamp", str(stamp)]
    command += _pack_options(quantities=quantities,
                             max_elevation_deg=max_elevation_deg,
                             max_range_km=max_range_km)
    return ODIM.run("pack", command, schema=PACK_SCHEMA)


def run_volumes(*, directory: Path) -> dict:
    """Which volumes a directory of ODIM files holds, grouped by nominal time.

    Reads every file's header and no payload, so surveying twenty 30 MB scans
    does not decode 20 million gates to answer a question about timestamps.
    """

    return ODIM.run("volumes", ["--dir", str(directory)],
                    schema=VOLUMES_SCHEMA)


def run_nyquist(*, file: Path) -> dict:
    """The dealias handoff: per-sweep Nyquist interval and its provenance.

    A folded velocity is not an observation until it is paired with the
    interval it folded at, and this is the record that says whether the file
    declared one (``Declared``), whether it was derived from a single PRF
    (``DerivedSinglePrf``), or whether it could not be established at all
    (``Unavailable``) -- in which case the sweep's velocities are not
    assimilable and the dealiaser refuses them by name rather than unfolding
    against a guess.
    """

    return ODIM.run("nyquist", ["--file", str(file)], schema=NYQUIST_SCHEMA)


__all__ = ["NYQUIST_GRANULARITY", "NYQUIST_SCHEMA", "ODIM", "PACK_SCHEMA",
           "SOURCE_LABEL", "SWEEP_PACK_SCHEMA", "VOLUMES_SCHEMA",
           "run_nyquist", "run_pack", "run_pack_dir", "run_volumes"]

"""The out-of-coverage refusal, as a class a front door can own.

A source grid that does not reach the requested domain is a complete,
well-named refusal: :func:`outside_source_grid_message` prints the first
uncovered target point, the source index it maps to, and the window the
source actually covers, which is what separates "the crop is too small"
from "this source does not reach the target" -- two problems with
opposite remedies.

It was raised as a bare ``ValueError`` from inside the interpolator, so
the preparation adapters, which call the interpolator with no handler,
relayed it as ten lines of internal call stack with the sentence at the
bottom.  A user read ``gpuwm/ingest/horiz.py`` line numbers before
reading what to do about their domain.  Giving the refusal its own class
lets every preparation door catch exactly this and nothing else: a real
defect in the same call still raises and still prints its stack.

The class stays a ``ValueError`` subclass on purpose -- library callers
that already guard interpolation with ``except ValueError`` keep working
unchanged.

Nothing here is specific to a source.  Any regional model -- any grid
that simply does not extend to where the user put their domain -- lands
on the same class, the same message and the same remedy, so a model
added as table data inherits the behaviour with no new code.
"""
from __future__ import annotations

import functools
import sys

import numpy as np

#: Exit status of a door-owned coverage refusal.  ``sysexits.h``'s
#: ``EX_CONFIG`` (78), the same code :mod:`gpuwm.source_cli` returns for
#: "this configuration cannot run", so the preparation stage answers a
#: geometry mismatch with one number no matter which adapter met it.
#: :mod:`gpuwm.source_cli` relays the adapter's status unchanged, so this
#: is also what ``gpuwm prep`` exits with.
SOURCE_COVERAGE_EXIT_CODE = 78

#: The same status under the name that covers every owned refusal, not
#: only the geometry one.  ``gpuwm prep`` answers "the inputs you staged
#: cannot make this run" with one number whatever the reason was.
PREPARATION_REFUSAL_EXIT_CODE = SOURCE_COVERAGE_EXIT_CODE

#: An explicitly selected identity pairing: a target that IS a declared
#: source grid, with the same mass dimensions and every point on its own
#: source cell, copies index for index instead of interpolating.
#: Maximum residual in source cells against either declared lattice.
#: This covers rounding of the geographic anchor, rather than an
#: arbitrary shift of the target.
LATTICE_IDENTITY_ANCHOR_CELLS = 1.0e-4
#: ... and every point must project strictly closer than this to its own
#: lattice position, the geometric limit past which a point belongs to
#: another cell.  A grid spelled on WPS's sphere and the same grid's GRIB
#: header on another sphere drift apart with distance from the anchor
#: (MEASURED on the 1799 x 1059 3 km CONUS grid: +0.347 / +0.204 cells in
#: x / y at its far corner, 6,370 km against 6,371.229 km).
LATTICE_IDENTITY_LIMIT_CELLS = 0.5


def lattice_identity(y_index, x_index, *, nx: int, ny: int,
                     sphere_scale: float = 1.0):
    """The source lattice indices of a target that IS the source grid.

    Callers select this geometric check only for sources whose metadata
    declares identity pairing. Unmarked sources keep their prior indices.

    ``y_index``/``x_index`` are a target staggering's zero-based
    fractional source indices, as the declared projection maps them; the
    source grid has ``ny x nx`` mass points.  The target is the source
    grid's own mass points, its u faces (face ``i`` at ``i - 1/2``) or its
    v faces -- the whole grid, or an index-aligned window of it (a crop of
    ``ny_t x nx_t`` mass points whose first point is source cell ``(j0,
    i0)``) -- with every point within
    :data:`LATTICE_IDENTITY_ANCHOR_CELLS` of either the exact lattice or
    that lattice scaled by the declared source sphere's radius divided by
    the WPS radius (``sphere_scale``).  This allows rounded source anchors
    and the two declared sphere conventions, without accepting an
    arbitrary spacing change, shift or local distortion.  Every point must
    also lie strictly within :data:`LATTICE_IDENTITY_LIMIT_CELLS` of its
    own cell.  The exact indices are returned, the outermost faces (half a
    cell past the grid's edge) clamped onto the edge cell, so mass points
    copy their own cell and an interior face reads its two neighbours.
    ``None`` for every other target, which keeps the projected indices it
    came with.

    Breakage it removes: the native grid as a target was refused by every
    coverage guard, because its outermost faces and cell corners sit half
    a cell past the source's outermost points, and the advice was a one-
    row trim that moved the boundary relaxation zone one row inward of
    the grid's own.

    Breakage the window form removes (hour-1 spin-up, 2026-10-06): a crop
    of the HRRR lattice -- including the 1797 x 1057 HRRR door itself,
    which is one cell inside the 1799 x 1059 source -- failed this test on
    shape alone and fell through to bilinear interpolation at the
    GRIB-sphere drift (0.15 to 0.35 of a cell), which smeared every start
    field: graupel peaks 8% lower at the median, 25 to 32% more nonzero
    cloud-ice and graupel cells than the analysis had.  An index-aligned
    window copies cell for cell exactly as the whole grid does.
    """

    y_index = np.asarray(y_index, dtype=np.float64)
    x_index = np.asarray(x_index, dtype=np.float64)
    shape = tuple(y_index.shape)
    nx, ny = int(nx), int(ny)
    if (len(shape) != 2 or tuple(x_index.shape) != shape
            or shape[0] < 1 or shape[1] < 1
            or shape[0] > ny + 1 or shape[1] > nx + 1):
        return None
    if not (np.isfinite(y_index).all() and np.isfinite(x_index).all()):
        return None
    if not np.isfinite(sphere_scale) or sphere_scale <= 0.0:
        return None
    rows, cols = np.indices(shape, dtype=np.float64)
    # The staggering is read from the target's own indices: a mass window
    # sits on whole cells, a u-face window half a cell west, a v-face
    # window half a cell south.  The anchor (j0, i0) is the source cell
    # under the window's first point; the whole grid is the window at
    # (0, 0), for which this reduces term for term to the earlier test.
    for offset_y, offset_x in ((0.0, 0.0), (0.0, -0.5), (-0.5, 0.0)):
        # The window may reach one point past the grid only along a
        # staggered axis, where that point is the outermost face half a
        # cell past the edge (clamped onto the edge cell below).  A mass
        # window must sit wholly on the grid's own cells: a window whose
        # last row or column is a full cell past the edge is not the
        # grid, and returning it here would copy the edge cell into the
        # overhang and silence the coverage refusal that names the gap.
        max_rows = ny + (1 if offset_y else 0)
        max_cols = nx + (1 if offset_x else 0)
        for scale in (1.0, float(sphere_scale)):
            j0 = float(np.rint(y_index[0, 0] / scale - offset_y))
            i0 = float(np.rint(x_index[0, 0] / scale - offset_x))
            if j0 < 0.0 or i0 < 0.0 or j0 + shape[0] > max_rows \
                    or i0 + shape[1] > max_cols:
                continue
            lattice_y = rows + offset_y + j0
            lattice_x = cols + offset_x + i0
            residual = max(
                float(np.abs(y_index - lattice_y * scale).max()),
                float(np.abs(x_index - lattice_x * scale).max()))
            if residual > LATTICE_IDENTITY_ANCHOR_CELLS:
                continue
            drift = max(float(np.abs(y_index - lattice_y).max()),
                        float(np.abs(x_index - lattice_x).max()))
            if not drift < LATTICE_IDENTITY_LIMIT_CELLS:
                return None
            return (np.clip(lattice_y, 0.0, float(ny - 1)),
                    np.clip(lattice_x, 0.0, float(nx - 1)))
    return None


#: What to DO about it.  Both branches are named because the message
#: above distinguishes them and they do not share a fix.
SOURCE_COVERAGE_REMEDY = (
    "remedy: if the window above is a CROP of a wider grid, re-fetch the "
    "source with a margin that contains the whole domain -- the "
    "interpolation stencil reaches one cell beyond every corner.  If the "
    "window IS the source's whole extent, no crop reaches this domain: "
    "move the domain inside the window, or prepare it from a source whose "
    "grid covers it (--list-sources names every source this install runs)."
)

#: What to DO about a series too short to bound a forecast.  One valid
#: time is an initial condition and nothing else, so there is no second
#: state to relax the domain edges toward and no cadence to relax on.
FORCING_SERIES_REMEDY = (
    "remedy: stage the whole window this run needs, not just its first "
    "time.  The first valid time is the initial condition and every later "
    "one is a lateral boundary, so a bounded run needs at least two on a "
    "single uniform cadence -- `gpuwm domain` prints the acquisition step "
    "for the window it sized, and `gpuwm prep --show-source NAME` names "
    "the products every one of those times must carry. A single analysis "
    "may instead be supplied to mapped preparation with --initial-inputs; "
    "its primary forcing window still needs at least two times."
)


class PreparationRefusal(ValueError):
    """The staged inputs cannot make the run that was asked for.

    Not a defect: a complete, well-named statement that this
    configuration of BYTES and DOMAIN has no forecast in it.  The door
    that a user typed catches this class and prints it as sentences;
    anything else keeps its traceback, because anything else is ours to
    fix rather than theirs.

    Each subclass carries the remedy its own breakage has, because two
    refusals with the same delivery can still have opposite fixes.  It
    stays a ``ValueError`` subclass so every library caller that already
    guards these calls with ``except ValueError`` is unchanged.
    """

    #: Overridden per subclass; the door prints this under the message.
    remedy = (
        "remedy: `gpuwm prep --show-source NAME` names what this route "
        "requires of the inputs you staged.")

    #: Folders the message and remedy name as the place to act (a
    #: scratch folder to make room in).  A page that hides machine paths
    #: still shows these: without them the remedy names nowhere.
    folders: tuple[str, ...] = ()

    def __init__(self, message: str, *, remedy: str | None = None,
                 folders=()) -> None:
        super().__init__(message)
        if remedy is not None:
            self.remedy = remedy
        if folders:
            self.folders = tuple(str(folder) for folder in folders)


class SourceCoverageRefusal(PreparationRefusal):
    """The source grid does not cover the requested target domain."""

    remedy = SOURCE_COVERAGE_REMEDY


class SourceProjectionRefusal(PreparationRefusal):
    """The source declares a projection this install cannot pair against.

    A declared projection is a promise about what the source's own
    coordinate arrays MEAN, and every route pairs the target into that
    plane before it touches them.  A family the transform does not
    evaluate therefore has no safe reading at all: taking the arrays as
    degrees is exactly the defect that put a domain sitting inside a
    Lambert grid 3,353 columns off its west edge.  Its own class, and a
    refusal rather than a traceback, because the bytes are the answer --
    nothing in this install fixes them.
    """

    remedy = (
        "remedy: this install pairs a declared lambert_conformal source "
        "through its own projection; any other declared family has to be "
        "regridded to a regular latitude/longitude grid before it is "
        "staged (`gpuwm prep --show-source NAME` names what the route "
        "requires of the inputs).")


class DecoderInventoryRefusal(PreparationRefusal):
    """The decoders this route needs are not the ones it was handed.

    The Python engine reads GRIB through subprocess tools and the Rust
    engine reads it in process, so a route knows exactly which
    executables its work requires.  When it is handed a different set --
    none at all, or subprocess tools on a route that decodes in process
    -- there is nothing to decode WITH, and that is a statement about
    the install, not a defect in the bytes.

    It reached users as ``ValueError: grib2 decoder inventory differs
    from the contract`` nineteen frames deep out of
    ``gpuwm/mapped_composition.py``, on a bare default
    ``gpuwm prep --source <any composed source>``.  The refusal was
    right and its delivery threw it away.

    The remedy is passed per raise rather than fixed here: what to DO
    depends on whether this install can stage a bridge, build one, or
    only be told to stop pinning one, and only the raising site knows
    which.  The class default names the estate command that answers it
    on every install.
    """

    remedy = (
        "remedy: `gpuwm doctor` prints this machine's decoder estate and "
        "the exact command that fills the gap it finds.")


class RunInputRefusal(PreparationRefusal):
    """A path this run was handed does not point at what its flag requires.

    The mapped door used to answer a missing ``--experiment-config``
    file with a bare ``FileNotFoundError`` holding nothing but the
    path: no flag name, no statement of what the file was FOR, no way
    to tell a typo from a pasted relative path resolved against the
    wrong working directory.  The 2.5.0 persona walks met it as the
    first of the door's three raw tracebacks (UX finding N6).

    The message names the FLAG and the RESOLVED path together, because
    the resolved path is what exposes the working-directory mistake a
    pasted ``prep-command.txt`` line makes.  The remedy is passed per
    raise when a missing flag has its own writer to name (the
    experiment config has two); the class default covers the rest.
    """

    remedy = (
        "remedy: every flag above must name an existing file.  Paths "
        "resolve from the directory this command runs in, so a pasted "
        "relative path needs the working directory it was written from."
    )


def existing_output_root_refusal(output_root) -> PreparationRefusal | None:
    """The mapped preparer's refusal for an ``--output-root`` that exists.

    One check with two callers: the preparer, before it builds anything,
    and ``gpuwm prep --source-root``, before it writes the folder's input
    manifest, so a run that is going to be refused here changes nothing
    on disk first.  ``None`` when nothing is at the path.
    """

    import os

    if not os.path.lexists(output_root):
        return None
    from gpuwm.ingest.boundary_stream import unfinished_tree_reason

    if unfinished_tree_reason(output_root) is not None:
        # A chained head whose producer failed, was stopped or went silent
        # is the preparer's own unfinished product, which it removes and
        # builds again (boundary_stream.remove_unfinished_tree); refusing
        # it here would stop the rebuild the preparer exists to do.
        return None
    return PreparationRefusal(
        f"refusing to overwrite mapped output {output_root}: a "
        "prepared tree is published atomically, and a folder that "
        "already exists may be a finished run something else reads",
        remedy=(
            "remedy: pass a fresh --output-root, or move or remove "
            "the old folder yourself first; prep never deletes an "
            "output tree."))


class VerticalLadderRefusal(PreparationRefusal):
    """The experiment's vertical ladder cannot drive the mapped target.

    Two bare exceptions used to share this breakage and CIRCLE (UX
    finding N6): an imported WRF config's level count met the mapping's
    reference count as ``ValueError: mapped target vertical levels
    differ``, and matching the count then raised ``explicit eta_levels
    has shape (0,)`` -- demanding an explicit ladder that
    ``import-namelist`` never writes and no stock WRF namelist carries,
    because WRF's real.exe generates the ladder itself.  No edit the
    first message suggested could terminate.

    The class carries no useful default remedy on purpose: what to DO
    depends on whether the config carries a ladder at all, so every
    raising site names its own doors.
    """

    remedy = (
        "remedy: declare an explicit [shared] eta_levels ladder in the "
        "experiment config; `gpuwm domain` authors a config carrying "
        "the certified reference ladder to copy from."
    )


class ForcingSeriesRefusal(PreparationRefusal):
    """The staged valid times cannot bound a forecast.

    Deliberately generic: "fewer than two times", "times not increasing",
    "cadence not uniform" are one question -- can these bytes drive the
    domain edges -- and every route asks it of its own series.  Four
    routes each raised their own bare ``ValueError`` for it, so the same
    user mistake arrived as four different tracebacks.
    """

    remedy = FORCING_SERIES_REMEDY


#: The variable that places the decode engine's scratch; see
#: :func:`gpuwm.mapped_composition._compose_scratch_base`.
COMPOSE_SCRATCH_ENV = "GPUWM_COMPOSE_SCRATCH"


def compose_scratch_folder(destination):
    """The folder a preparation writing ``destination`` stages its frame stream in.

    The placement :func:`gpuwm.mapped_composition._compose_scratch_base`
    makes, asked without creating or refusing anything, so a plan can
    measure that folder's disk before the download:
    ``GPUWM_COMPOSE_SCRATCH`` when it is set, else the destination's
    parent, else None (the system temp, for a caller with no output).
    """

    import os
    from pathlib import Path

    override = os.environ.get(COMPOSE_SCRATCH_ENV)
    if override:
        return Path(override)
    if destination is None:
        return None
    return Path(destination).resolve().parent


def compose_scratch_override_refusal():
    """The refusal for a ``GPUWM_COMPOSE_SCRATCH`` that names no existing folder, or None.

    The preparation refuses such a variable by name rather than fall back
    to the system temp it was set to steer away from
    (:func:`gpuwm.mapped_composition._compose_scratch_base`), but only
    when it starts composing, which is after the whole download.  A plan
    asks this first, so the same mistake is refused while nothing has
    been spent.  None when the variable is unset or names a folder.
    """

    import os
    from pathlib import Path

    override = os.environ.get(COMPOSE_SCRATCH_ENV)
    if not override or Path(override).is_dir():
        return None
    return (f"{COMPOSE_SCRATCH_ENV}={override} does not name an existing folder, and this "
            "run's preparation stages its decoded frame stream there, so it would stop as it "
            "starts composing rather than put the stream on the system temp this variable "
            f"steers away from.  Create {override}, or set "
            f"{COMPOSE_SCRATCH_ENV} to an existing folder on a disk with room for the stream, "
            "or unset it to stage the stream beside the run's preparation")


#: What to DO when the scratch disk cannot hold the frame stream.
SCRATCH_DISK_REMEDY = (
    f"remedy: set {COMPOSE_SCRATCH_ENV} to an existing directory on a disk "
    "with room for the bytes named above, and the preparation stages its "
    "frame stream there instead; or free that much space on the disk "
    "that holds the folder named above.")


class ScratchDiskRefusal(PreparationRefusal):
    """The disk that holds the decode scratch cannot hold the frame stream.

    The engine stages every decoded valid time on disk before the
    preparation reads it back, tens of GB for a global source over two
    days, so the scratch disk is a resource the request has to fit, like
    the card.  The message names the folder, the bytes the stream needs
    when they are known and the space the disk has.  Its own class
    because its remedy is space: raised as ``FileNotFoundError``, a full
    disk told the user to supply a file the preparation already had.
    """

    remedy = SCRATCH_DISK_REMEDY


def scratch_disk_folder(base) -> str:
    """The folder a preparation's scratch is made in: ``base``, or the system temp."""

    import tempfile

    return tempfile.gettempdir() if base is None else str(base)


def scratch_disk_refusal(refusal, base) -> ScratchDiskRefusal:
    """The engine's scratch refusal, re-said for the folder the next attempt stages in.

    The engine names its own temporary folder, deleted as the
    preparation fails; the remedy and the folder a page shows are where
    the NEXT attempt would stage again (:func:`scratch_disk_remedy`).
    """

    return ScratchDiskRefusal(str(refusal), remedy=scratch_disk_remedy(base),
                              folders=(scratch_disk_folder(base),))


def scratch_disk_remedy(base) -> str:
    """The remedy naming the folder a preparation's scratch is made in.

    The refusal's own path is the engine's temporary folder, deleted as
    the preparation fails, so the remedy names where the NEXT attempt
    would stage again: ``base``, or the system temp when it is None.
    """

    where = scratch_disk_folder(base)
    return (
        f"remedy: the preparation stages its frame stream in {where}.  Set "
        f"{COMPOSE_SCRATCH_ENV} to an existing directory on a disk with room "
        "for the bytes named above and it stages there instead, or free that "
        "much space on this disk.")


def outside_source_grid_message(latitude, longitude, target_lat, target_lon,
                                y, x, outside, *, axis_space=None,
                                target_geographic=None) -> str:
    """Name the first uncovered target point, its index, and the source span.

    ``target points fall outside the source grid`` on its own named no
    coordinate and no window, so a user could not tell a genuinely
    undersized crop from a source axis that does not reach the target --
    the two have opposite remedies.  The numbers here are the same ones
    the native-route coverage refusal prints.

    ``axis_space`` names the plane the pairing happened in whenever the
    source is NOT geographic.  A projected source's coordinate arrays are
    its own projection axes, so printing them as ``lon 0..53.94`` and
    ``lat 0..31.74`` -- degrees, and a box off the coast of Africa --
    describes a source that does not exist.  Given the name, both the
    target point and the source span are printed in that plane instead.

    ``target_geographic`` is the same target points in degrees.  In the
    plane, the point that misses is stated in coordinates the user never
    typed, so the refusal is printed with the domain's own lat/lon beside
    it -- that is the number in the namelist, and the number the remedy
    is applied to.
    """

    first = int(np.argmax(np.asarray(outside).ravel()))
    index = np.unravel_index(first, np.shape(outside))
    point = (f"{np.asarray(target_lat).ravel()[first]:.4f}, "
             f"{np.asarray(target_lon).ravel()[first]:.4f}")
    if axis_space is None:
        where = f"at lat/lon ({point})"
        span = (f"(lon {longitude[0]:g}..{longitude[-1]:g})",
                f"(lat {latitude[0]:g}..{latitude[-1]:g})")
    else:
        where = f"at {axis_space} (y, x) = ({point})"
        if target_geographic is not None:
            geographic_lat, geographic_lon = target_geographic
            where += (
                f", lat/lon ({np.asarray(geographic_lat).ravel()[first]:.4f}, "
                f"{np.asarray(geographic_lon).ravel()[first]:.4f})")
        span = (f"({axis_space} x {longitude[0]:g}..{longitude[-1]:g})",
                f"({axis_space} y {latitude[0]:g}..{latitude[-1]:g})")
    return (
        "target points fall outside the source grid: target point "
        f"{tuple(int(value) for value in index)} {where} maps to source "
        f"index x={np.asarray(x).ravel()[first]:.3f} "
        f"y={np.asarray(y).ravel()[first]:.3f}, and the source covers "
        f"x=0..{longitude.size - 1} {span[0]} "
        f"y=0..{latitude.size - 1} {span[1]}")


#: The file a caller that launched a preparation asks its door to record
#: the refusal in, as well as printing it.  Read back by
#: :func:`recorded_preparation_refusal`; inherited by every child the
#: preparation launches, so a refusal met in the adapter process reaches
#: the caller whole instead of as an exit status.
PREPARATION_REFUSAL_RECORD_ENV = "GPUWM_PREPARATION_REFUSAL_RECORD"
PREPARATION_REFUSAL_RECORD_SCHEMA = "gpuwm-preparation-refusal-v1"


def _record_preparation_refusal(refusal: PreparationRefusal, remedy: str) -> None:
    import json
    import os

    path = os.environ.get(PREPARATION_REFUSAL_RECORD_ENV)
    if not path:
        return
    document = {"schema": PREPARATION_REFUSAL_RECORD_SCHEMA,
                "class": type(refusal).__name__, "message": str(refusal),
                "remedy": remedy, "folders": list(getattr(refusal, "folders", ()) or ())}
    try:
        with open(path, "w", encoding="utf-8") as output:
            json.dump(document, output)
    except OSError:
        # The printed refusal is still the refusal; the record is a copy.
        pass


def _refusal_class(name: str) -> type[PreparationRefusal]:
    pending = [PreparationRefusal]
    while pending:
        cls = pending.pop()
        if cls.__name__ == name:
            return cls
        pending.extend(cls.__subclasses__())
    return PreparationRefusal


class recorded_preparation_refusal:
    """Collect the refusal a preparation door reports, in this process or a child.

    Used as ``with recorded_preparation_refusal() as refused: ...`` around a
    preparation; afterwards ``refused()`` is the refusal the door printed,
    rebuilt as its own class with its message, remedy and folders and the
    door's exit status on ``exit_code``, or None when it printed none.
    Named breakage: a chain that ran the preparation saw only the status,
    so a scratch disk too small for the frame stream reached a run's
    failure notice as "prepare failed (exit 78)."
    """

    def __enter__(self):
        import os
        import tempfile

        handle, self._path = tempfile.mkstemp(prefix="gpuwm-prep-refusal-", suffix=".json")
        os.close(handle)
        os.unlink(self._path)
        self._previous = os.environ.get(PREPARATION_REFUSAL_RECORD_ENV)
        os.environ[PREPARATION_REFUSAL_RECORD_ENV] = self._path
        self._refusal = None
        return lambda: self._refusal

    def __exit__(self, *exc) -> bool:
        import json
        import os

        if self._previous is None:
            os.environ.pop(PREPARATION_REFUSAL_RECORD_ENV, None)
        else:
            os.environ[PREPARATION_REFUSAL_RECORD_ENV] = self._previous
        try:
            with open(self._path, encoding="utf-8") as record:
                document = json.load(record)
        except (OSError, ValueError):
            document = None
        finally:
            try:
                os.unlink(self._path)
            except OSError:
                pass
        if (isinstance(document, dict)
                and document.get("schema") == PREPARATION_REFUSAL_RECORD_SCHEMA):
            refusal = _refusal_class(str(document.get("class")))(
                str(document.get("message", "")),
                remedy=str(document.get("remedy") or "") or None,
                folders=tuple(document.get("folders") or ()))
            refusal.exit_code = PREPARATION_REFUSAL_EXIT_CODE
            self._refusal = refusal
        return False


def report_preparation_refusal(refusal: PreparationRefusal, *,
                               stream=None) -> int:
    """Print the refusal and ITS remedy, and return the door's status.

    Two lines on stderr and nothing on stdout: a caller piping the
    adapter's JSON proof gets an empty pipe and a non-zero status rather
    than a parse error.  The remedy comes off the refusal because two
    refusals delivered the same way can still have opposite fixes.  A
    caller that asked for a record (:func:`recorded_preparation_refusal`)
    gets the same refusal as data.
    """

    stream = sys.stderr if stream is None else stream
    remedy = getattr(refusal, "remedy", PreparationRefusal.remedy)
    print(f"prep: REFUSED: {refusal}", file=stream)
    print(remedy, file=stream)
    _record_preparation_refusal(refusal, remedy)
    return PREPARATION_REFUSAL_EXIT_CODE


#: The name this function had when the only owned refusal was the
#: coverage one.  Kept so existing callers and suites are unchanged.
report_source_coverage_refusal = report_preparation_refusal


def owns_source_coverage_refusal(main):
    """Wrap a preparation ``main`` so the door answers its own refusal.

    Applied to every module ``gpuwm prep`` launches.  The whole body is
    inside the handler rather than one call, because a refusal fires
    wherever the staged inputs first meet the target -- statics, primary
    decode, a supplement, or the valid-time series -- and a user must get
    the same two lines from all of them.

    It catches :class:`PreparationRefusal`, the whole family, rather than
    the coverage member alone: a route that refuses a one-time series
    correctly and then relays it as a 28-line traceback has written the
    sentence and thrown it away.  Anything that is NOT that family still
    raises with its stack, because that is ours to fix.
    """

    @functools.wraps(main)
    def door(*args, **kwargs) -> int:
        try:
            return main(*args, **kwargs)
        except PreparationRefusal as refusal:
            return report_preparation_refusal(refusal)

    door.owns_source_coverage_refusal = True
    return door


__all__ = [
    "FORCING_SERIES_REMEDY",
    "PREPARATION_REFUSAL_EXIT_CODE",
    "PREPARATION_REFUSAL_RECORD_ENV",
    "PREPARATION_REFUSAL_RECORD_SCHEMA",
    "SOURCE_COVERAGE_EXIT_CODE",
    "SOURCE_COVERAGE_REMEDY",
    "DecoderInventoryRefusal",
    "ForcingSeriesRefusal",
    "PreparationRefusal",
    "RunInputRefusal",
    "SCRATCH_DISK_REMEDY",
    "ScratchDiskRefusal",
    "SourceCoverageRefusal",
    "SourceProjectionRefusal",
    "VerticalLadderRefusal",
    "compose_scratch_folder",
    "compose_scratch_override_refusal",
    "existing_output_root_refusal",
    "lattice_identity",
    "outside_source_grid_message",
    "owns_source_coverage_refusal",
    "recorded_preparation_refusal",
    "report_preparation_refusal",
    "report_source_coverage_refusal",
    "scratch_disk_folder",
    "scratch_disk_refusal",
    "scratch_disk_remedy",
]

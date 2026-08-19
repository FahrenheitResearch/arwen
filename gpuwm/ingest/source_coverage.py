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
    "the products every one of those times must carry.  A single time can "
    "only be run with lateral boundaries switched off in the target "
    "contract, which is an idealized case, not a forecast."
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

    def __init__(self, message: str, *, remedy: str | None = None) -> None:
        super().__init__(message)
        if remedy is not None:
            self.remedy = remedy


class SourceCoverageRefusal(PreparationRefusal):
    """The source grid does not cover the requested target domain."""

    remedy = SOURCE_COVERAGE_REMEDY


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


def outside_source_grid_message(latitude, longitude, target_lat, target_lon,
                                y, x, outside) -> str:
    """Name the first uncovered target point, its index, and the source span.

    ``target points fall outside the source grid`` on its own named no
    coordinate and no window, so a user could not tell a genuinely
    undersized crop from a source axis that does not reach the target --
    the two have opposite remedies.  The numbers here are the same ones
    the native-route coverage refusal prints.
    """

    first = int(np.argmax(np.asarray(outside).ravel()))
    index = np.unravel_index(first, np.shape(outside))
    return (
        "target points fall outside the source grid: target point "
        f"{tuple(int(value) for value in index)} at lat/lon "
        f"({np.asarray(target_lat).ravel()[first]:.4f}, "
        f"{np.asarray(target_lon).ravel()[first]:.4f}) maps to source "
        f"index x={np.asarray(x).ravel()[first]:.3f} "
        f"y={np.asarray(y).ravel()[first]:.3f}, and the source covers "
        f"x=0..{longitude.size - 1} (lon {longitude[0]:g}..{longitude[-1]:g}) "
        f"y=0..{latitude.size - 1} (lat {latitude[0]:g}..{latitude[-1]:g})")


def report_preparation_refusal(refusal: PreparationRefusal, *,
                               stream=None) -> int:
    """Print the refusal and ITS remedy, and return the door's status.

    Two lines on stderr and nothing on stdout: a caller piping the
    adapter's JSON proof gets an empty pipe and a non-zero status rather
    than a parse error.  The remedy comes off the refusal because two
    refusals delivered the same way can still have opposite fixes.
    """

    stream = sys.stderr if stream is None else stream
    print(f"prep: REFUSED: {refusal}", file=stream)
    print(getattr(refusal, "remedy", PreparationRefusal.remedy), file=stream)
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
    "SOURCE_COVERAGE_EXIT_CODE",
    "SOURCE_COVERAGE_REMEDY",
    "DecoderInventoryRefusal",
    "ForcingSeriesRefusal",
    "PreparationRefusal",
    "RunInputRefusal",
    "SourceCoverageRefusal",
    "VerticalLadderRefusal",
    "outside_source_grid_message",
    "owns_source_coverage_refusal",
    "report_preparation_refusal",
    "report_source_coverage_refusal",
]

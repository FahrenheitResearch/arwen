"""Where a rendered product PNG goes -- the one answer, for both engines.

Every picture a run drew used to land in ONE directory.  A three-nest
run of the rust catalog at hourly output puts five figures of files
there, of every product and every valid time, and the only way to find
one is to read filenames::

    out/case/png/arwen_wrf_20260520_18z_f000_d02-3km_composite_reflectivity.png
    out/case/png/arwen_wrf_20260520_18z_f000_d02-3km_2m_temperature.png
    out/case/png/arwen_wrf_20260520_18z_f000_d04-100m_composite_reflectivity.png
    ... 10,869 more

The layout below replaces that.  It is Drew's 2026-08-06 ruling -- case
folder, then domain, then product subfolders, organised AT RENDER TIME
rather than tidied afterwards -- with the reporter's timestamp request
slotted into it as the leaf grouping::

    <--out>/<domain-token>/<product>/<valid-day>/<filename>.png

Read as a sentence: *which nest, which chart, which day.*  The case
folder is ``--out`` itself, which every front door already sets per
case (``gpuwm go`` gives it ``<case>/png``), so this module never
invents a case name -- it could not, and a case name in generic code is
a rule this project has paid for twice.

Four properties are the contract, and each is pinned by a test in
``tests/test_render_layout.py``:

**It is the default.**  Not a flag, not an opt-in.  ``--layout flat``
exists only so a consumer written against the old directory has
somewhere to stand while it is updated, and it reproduces the v2.4.1
spelling byte for byte.  A correctness remedy that ships off is a
workaround; this one ships on.

**It is predictable.**  The path is a pure function of the three facts
in it, so a script can compute where a frame will be BEFORE it exists
and watch that one file, instead of globbing a directory and diffing
listings.  There is no adaptive bucketing, no "split when it gets big":
those cannot be predicted, which defeats the point.

**It never loses a file.**  Every segment has a defined value even when
the fact behind it cannot be read: an unidentifiable domain is
``native_grid`` (the spelling both engines already use), an unreadable
valid time is :data:`UNDATED`, an unparseable engine filename is
:data:`UNCLASSIFIED`.  A picture is always somewhere nameable, never
dropped and never left loose at the root.  Path LENGTH is part of that
promise on Windows, and it is answered TWICE.

:func:`delivered_name` answers the half that is ours: a frame filed
under ``<domain>/<product>/`` used to carry those same two tokens inside
its own filename, and a delivery measured on disk reached 310 characters
because of it.  The pair comes off at the organisation step -- the same
layout is then 226 characters under a typical case root -- and
:func:`engine_name` puts it back, so nothing is lost and no two frames
in a folder collapse onto one name.

:func:`fs_path` answers the half that is the caller's: a case root deep
enough still passes ``MAX_PATH``, the move into the tree fails, and the
picture is left flat at the root -- the ruling inverted for exactly the
products whose names are longest, while their shorter neighbours file
correctly and the directory looks almost right.

**One walker reads it.**  :func:`iter_rendered` is what every in-tree
consumer of a render directory uses, so a reader cannot be written that
sees only half the tree -- and it skips dot-directories, because the
early render's scratch is a dot-prefixed sibling of the pictures and a
naive recursive read would publish a half-finished run's temporaries.

The day is the day the frame is VALID, not the day its run was
initialised: a 21z cycle at f+06 files under the next morning, which is
the date a forecaster is looking for.
"""

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path

#: Windows' classic path ceiling.  A path of this length or longer is
#: rejected by the ordinary Win32 entry points -- ``mkdir``, ``replace``,
#: ``open`` and the directory walk alike -- with ERROR_PATH_NOT_FOUND,
#: which reads as "the folder is not there" rather than "the name is too
#: long".
_MAX_PATH = 260

#: The extended-length prefix.  A fully-qualified path wearing it is
#: handed to the filesystem verbatim, with no MAX_PATH ceiling and no
#: normalisation -- which is why :func:`fs_path` resolves the path first.
_LONG_PREFIX = "\\\\?\\"

#: Domain / product / valid-day subfolders.  The default.
NESTED = "nested"

#: Every PNG directly under ``--out``: what v2.4.1 and earlier wrote.
FLAT = "flat"

#: The vocabulary of ``gpuwm render --layout``.
LAYOUTS = (NESTED, FLAT)

#: What ``--layout`` is when nobody says otherwise.
DEFAULT_LAYOUT = NESTED

#: The day segment for a frame whose valid time could not be read.
UNDATED = "undated"

#: The product segment for an engine output whose slug could not be read.
UNCLASSIFIED = "unclassified"

#: The domain segment for a file that proves no domain identity.  Spelled
#: as ``gpuwm.render.NATIVE_GRID_SLUG`` and as the rust engine's
#: ``native_grid`` spell it -- one word for one fact.
NATIVE_GRID = "native_grid"

#: ``YYYY-MM-DD`` at the head of a WRF ``Times`` record
#: (``1974-04-03_18:00:00``), its filename-safe form
#: (``1974-04-03_18-00-00``) or an ISO instant.  All three begin with the
#: date, which is the only part this module needs.
_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T_ ]|$)")

#: The HEAD of the rust engine's output filename: everything through the
#: forecast-hour marker.  It is the frame's own identity -- model, cycle
#: date, cycle hour, whole-hour lead -- and nothing in it is repeated by
#: any folder the layout builds, so it is exactly the part that survives
#: :func:`delivered_name`.
#:
#: ``model`` is non-greedy so the first eight-digit run is the date.
#:
#: The cycle hour is ONE OR TWO digits, and the one-digit form is not a
#: tolerance: it is what the engine writes.  ``store_render.rs`` formats
#: its ``cycle_utc: u8`` with a plain ``{}``, so a 06Z run is ``_6z_``
#: -- ten of the twenty-four cycle hours, GFS 00Z and 06Z among them.  A
#: parser that demanded two digits returned None for all of them, and
#: every frame of those runs was left flat under a front door printing
#: ``layout nested``.
_HEAD = (r"(?:arwen|rustwx)_(?P<model>.+?)_(?P<date>\d{8})"
         r"_(?P<cycle>\d{1,2})z_f(?P<lead>\d{3})")

#: The rust engine's output filename, as
#: ``rustwx-products``/``derived.rs`` formats it and
#: ``gpuwm.render._rebrand_engine_output`` rebrands it::
#:
#:     arwen_<model>_<YYYYMMDD>_<H>z_f<NNN>_<domain-slug>_<product-slug>.png
#:
#: The tail is split into domain and product separately, because both
#: halves contain underscores and only their grammar tells them apart.
_ENGINE_NAME = re.compile(rf"^{_HEAD}_(?P<tail>.+)$")

#: The same grammar with the tail made OPTIONAL, which is what a
#: delivered name has: :func:`delivered_name` takes the tail off, so the
#: only thing left after ``f{NNN}`` is the exact-time suffix, or nothing
#: at all.  Spelled from the same ``_HEAD`` fragment as ``_ENGINE_NAME``
#: so the two cannot drift into disagreeing about what a frame is.
_DELIVERED_NAME = re.compile(rf"^{_HEAD}(?P<rest>.*)$")

#: A domain slug at the head of that tail: ``d02-3km``, ``d05-111m``, a
#: bare ``d02``, or the anonymous ``native_grid``.  The same three
#: degradation steps :func:`gpuwm.render.domain_token` produces, which
#: are ``rw-wrfbatch::native_domain_slug``'s.
_TAIL = re.compile(
    r"^(?P<domain>native_grid|d\d{2}(?:-\d+(?:\.\d+)?(?:km|m))?)"
    r"_(?P<product>.+)$")

#: The engine's EXACT-TIME suffix, at the end of a product slug.
#:
#: ``f{NNN}`` in the filename counts whole hours, so two frames of one
#: product inside the same hour would collide.  The vendored engine
#: settles that itself: ``rusty-weather/src/render_all.rs`` builds
#:
#:     valid_{YYYY}{MM}{DD}_{HH}{MM}{SS}z_lead_{HHH}h{MM}m{SS}s
#:
#: with ``{lead_hours:03}`` (a MINIMUM width -- a run past 999 h writes
#: more digits) and two-digit lead minutes and seconds, and appends it.
#:
#: It identifies a FRAME, so it must not reach the product folder: a
#: parser that read it as part of the product name gave every frame a
#: folder of its own, which is the flat directory this module exists to
#: prevent wearing a nested costume, and on Windows pushed the path past
#: MAX_PATH so the frame was left flat outright.
#:
#: Anchored at the end and spelled out in full on purpose: a product
#: legitimately named ``valid_hours_since_analysis`` keeps its name.
_EXACT_TIME = re.compile(
    r"^(?P<product>.+)_valid_(?P<stamp>\d{8}_\d{6})z"
    r"_lead_\d{3,}h\d{2}m\d{2}s$")


def fs_path(path, *, descend: bool = False) -> str:
    """``path`` spelled so a filesystem call accepts it at ANY length.

    On Windows a path at or past :data:`_MAX_PATH` is refused by the
    ordinary API, and every one of this module's callers turns that
    refusal into the same degradation: the picture is left where the
    engine dropped it, flat at the render root.  That is Drew's
    2026-08-06 layout ruling inverted by nothing but arithmetic, and it
    is SELECTIVE -- it takes the longest product names first, so one
    frame of a set escapes the tree while its neighbours file correctly
    and the directory looks almost right.

    The remedy is the extended-length spelling, applied only when the
    ordinary one would fail: below the ceiling the caller's own path
    comes back unchanged, so error messages, receipts and anything a
    reader compares against stay in the spelling they typed.  Verbatim
    paths are not normalised by the OS, so the path is resolved to an
    absolute, ``..``-free, backslash form BEFORE the prefix goes on --
    a raw ``a/b/../c`` behind the prefix would name a directory called
    ``..``.

    ``descend=True`` is for a path that is about to be WALKED rather
    than opened.  A root's own length says nothing about its
    descendants', and the ceiling is enforced on the FULL path of each
    entry: a 200-character render directory holding a 269-character
    product file enumerates the directory happily and then reports the
    file as not a file.  A walk that started verbatim sees all of it.

    Not Windows, already prefixed, or short enough: unchanged.  This is
    a spelling, never a different file.
    """

    text = os.fspath(Path(path))
    if os.name != "nt" or text.startswith(_LONG_PREFIX):
        return text
    absolute = os.path.abspath(text)
    if not descend and len(absolute) < _MAX_PATH:
        return text
    if absolute.startswith("\\\\"):
        # A UNC share: \\server\share\... becomes \\?\UNC\server\share\...
        return _LONG_PREFIX + "UNC" + absolute[1:]
    return _LONG_PREFIX + absolute


def valid_day(stamp: str | None) -> str | None:
    """``1974-04-03`` from any stamp spelling this tree writes, or None.

    ``None`` is not an error: it means the caller must use
    :data:`UNDATED`, which :func:`place` does for it.  Guessing a day
    from a stamp that does not carry one would file a frame under a date
    it has no evidence for, and a wrong date is worse than an honest
    ``undated`` because a reader believes it.
    """

    if not stamp:
        return None
    match = _DAY.match(str(stamp).strip())
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime.date(year, month, day).isoformat()
    except ValueError:
        # A syntactically well-formed date that does not exist
        # (``1974-02-31``) is no evidence at all.
        return None


def product_dir(*, domain: str | None, product: str | None,
                day: str | None) -> Path:
    """The relative directory one product frame belongs in.

    Relative on purpose: the root is the caller's ``--out``, and a
    function that joined it would be one that could be handed the wrong
    root without anyone noticing.
    """

    return (Path(domain or NATIVE_GRID) / (product or UNCLASSIFIED)
            / (day or UNDATED))


def place(root, *, domain: str | None, product: str | None,
          day: str | None, filename: str,
          layout: str = DEFAULT_LAYOUT) -> Path:
    """The full path for one rendered PNG, under ``root``.

    ``layout=FLAT`` returns ``root / filename`` -- the v2.4.1 spelling,
    unchanged, so a consumer pinned to it keeps working while it is
    updated.
    """

    root = Path(root)
    if layout == FLAT:
        return root / filename
    if layout != NESTED:
        raise ValueError(
            f"unknown render layout {layout!r}; choose from "
            f"{', '.join(LAYOUTS)}")
    return root / product_dir(domain=domain, product=product,
                              day=day) / filename


def _folder_tokens(name: str, *, domain: str | None,
                   product: str | None) -> tuple[str, str, str] | None:
    """``(head, repeated, rest)`` for one frame, or None if it does not fit.

    ``repeated`` is the exact ``<domain>_<product>`` string the two
    folders above the frame spell.  ``None`` means this name and these
    folders do not line up -- an engine name the grammar cannot read, or
    the honest fallback where neither the caller's token nor the slug
    grammar could split the tail -- and every caller then leaves the
    name exactly as it found it rather than cutting a guess out of it.
    """

    if not domain or not product:
        return None
    stem = Path(name).stem
    match = _DELIVERED_NAME.match(stem)
    if match is None:
        return None
    return stem[:match.start("rest")], f"{domain}_{product}", match.group(
        "rest")


def delivered_name(name: str, *, domain: str | None,
                   product: str | None) -> str:
    """``name`` with the tokens its own folders already carry removed.

    The engine writes ``arwen_<model>_<date>_<cycle>z_f<NNN>_<domain>_
    <product>[_<exact-time>].png`` into one flat directory, where every
    token has to be there because nothing else tells two files apart.
    Filed into the layout, two of those tokens become the names of the
    folders the frame is sitting in, and the picture then carries them
    twice::

        .../d01-12km/var_geopotential_height_700hpa_38d0bbbc4b4b7e87/
            2026-08-20/arwen_wrf_20260820_0z_f000_d01-12km_var_
            geopotential_height_700hpa_38d0bbbc4b4b7e87_valid_...png

    That is not untidiness, it is a delivery defect.  A real one measured
    310 characters, and while :func:`fs_path` lets ArWen write and read
    past MAX_PATH, it does nothing for the tools the delivery is OPENED
    with: Explorer, ``tar``, and the readers a recipient's own script
    imports all refuse the path.  The picture is filed correctly and
    cannot be opened, which is the same lost picture by a third route.

    So the repeated pair comes off HERE, at the Python organisation step,
    and not in the engine: the vendored crate stays byte-identical to its
    campaign builds, and ``--layout flat`` -- where the folders spell
    nothing and every token is load-bearing -- keeps the v2.4.1 name
    byte for byte.

    What survives is the frame's own identity (model, cycle date, cycle
    hour, lead) and the engine's exact-time suffix, which no folder
    carries.  Names therefore stay collision-free: :func:`engine_name`
    rebuilds the engine's name from the delivered one and its two
    folders exactly, so two frames that differed before differ after.

    A name the grammar cannot read, or one whose folders do not spell
    what it carries, comes back unchanged.
    """

    tokens = _folder_tokens(name, domain=domain, product=product)
    if tokens is None:
        return Path(name).name
    head, repeated, rest = tokens
    if not rest.startswith(f"_{repeated}"):
        return Path(name).name
    return head + rest[len(repeated) + 1:] + Path(name).suffix


def engine_name(name: str, *, domain: str | None,
                product: str | None) -> str:
    """The engine's own filename, rebuilt from a delivered one.

    The exact inverse of :func:`delivered_name` given the same two
    folders, and the reason the shortening is safe to do at all: it is
    information-preserving, so a reader that wants the v2.4.1 spelling
    -- ``--pair``'s matching key is the one in this tree -- computes it
    instead of being handed a name it can no longer recognise.

    A name that already carries the pair is already the engine's, and
    comes back unchanged; so does one the grammar cannot read.
    """

    tokens = _folder_tokens(name, domain=domain, product=product)
    if tokens is None:
        return Path(name).name
    head, repeated, rest = tokens
    if rest.startswith(f"_{repeated}"):
        return Path(name).name
    return f"{head}_{repeated}{rest}{Path(name).suffix}"


def parse_engine_output(name: str, *, domain: str | None = None
                        ) -> tuple[str, str, str] | None:
    """``(domain, product, valid_day)`` for one rust-engine filename.

    ``None`` when the name is not one of the engine's at all -- the
    caller then leaves the file where the engine put it rather than
    filing it under a guess.

    ``domain`` is the token the caller read from the wrfout itself
    (:func:`gpuwm.render.domain_token`).  It is a HINT for splitting the
    tail -- domain slugs and product slugs both contain underscores, so
    knowing one end tells you where the other begins -- and the fallback
    when the filename carries no parseable slug.  It does not override
    the filename's own slug, because the folder a picture lands in must
    be spelled the same way as the token inside its filename: a
    ``d02-3km`` file under a ``d02/`` folder is a reader asking which of
    the two is the lie, every time.  In a real render they are the same
    string; they differ only when one side could read less of the file
    than the other, and the more specific answer is the useful one.

    The day is the frame's VALID day: cycle date + cycle hour + lead
    hours.  A 21z run at f+06 is the next morning, and filing it under
    the initialisation date would put the interesting frames of every
    evening run in the wrong folder.
    """

    match = _ENGINE_NAME.match(Path(name).stem)
    if match is None:
        return None
    tail = match.group("tail")
    product: str | None = None
    if domain and tail.startswith(f"{domain}_"):
        product = tail[len(domain) + 1:]
    else:
        split = _TAIL.match(tail)
        if split is not None:
            domain = split.group("domain")
            product = split.group("product")
    if not product:
        # A tail neither the caller's token nor the slug grammar can
        # split.  The domain is still known (or honestly anonymous), and
        # the whole tail is a truthful, if ugly, product name -- better
        # than dropping the file at the root where it is invisible.
        product = tail
    # A sub-hourly frame carries the engine's exact-time suffix.  It
    # names the FRAME, so it comes off the product -- and it carries the
    # frame's own valid stamp, which is better evidence of the valid day
    # than cycle + whole-hour lead, the only thing available without it.
    exact = _EXACT_TIME.match(product)
    if exact is not None:
        product = exact.group("product")
        try:
            stamp = datetime.datetime.strptime(
                exact.group("stamp"), "%Y%m%d_%H%M%S")
        except ValueError:
            # A well-formed but impossible stamp is evidence of nothing,
            # and the same call `valid_day` makes for one.  The suffix
            # still comes off: the engine wrote it, and it is still not
            # the product's name.
            return (domain or NATIVE_GRID), product, UNDATED
        return (domain or NATIVE_GRID), product, stamp.date().isoformat()
    try:
        # The cycle hour is zero-padded HERE, not required of the name:
        # the engine writes ``6z``, and handing strptime a nine-digit
        # string would let ``%H`` steal a date digit.
        cycle = datetime.datetime.strptime(
            f"{match.group('date')}{int(match.group('cycle')):02d}",
            "%Y%m%d%H")
        valid = cycle + datetime.timedelta(hours=int(match.group("lead")))
    except ValueError:
        return (domain or NATIVE_GRID), product, UNDATED
    return (domain or NATIVE_GRID), product, valid.date().isoformat()


def iter_rendered(root) -> list[Path]:
    """Every rendered PNG under ``root``, flat layout or nested.

    THE reader for a render directory, and the reason there is only one:
    a consumer that globbed ``*.png`` saw nothing after this layout
    landed, and a consumer that recursed naively saw the early render's
    ``.first-products-scratch`` temporaries as though they had been
    published.  Both mistakes are made once, here.

    Sorted by path so two directories walk in the same order, which is
    what ``--pair`` needs to line frames up.

    The walk goes through :func:`fs_path` and the results come back in
    the CALLER's spelling.  A reader that could not see as deep as the
    placement can write is the same lost picture by another route: the
    file would be correctly filed, invisible to the early-render
    publisher, and reported as "produced no picture".
    """

    root = Path(root)
    walk_root = Path(fs_path(root, descend=True))
    if not walk_root.is_dir():
        return []
    found = []
    for path in walk_root.rglob("*.png"):
        relative = path.relative_to(walk_root)
        if any(part.startswith(".") for part in relative.parts[:-1]):
            continue
        if not path.is_file():
            continue
        found.append(root / relative)
    return sorted(found, key=lambda path: path.as_posix())


def describe(root: str = "<--out>", *, sep: str | None = None) -> str:
    """The one sentence a script author needs, for --help and docs.

    ``sep`` defaults to THIS platform's separator, which is what the
    line a render PRINTS needs: that line pastes the reader's own
    ``--out`` -- ``mycase-out\\png\\run-...Z`` on Windows -- in front of
    this template, and a forward-slash template behind a backslash path
    read as two paths glued together (UX finding N24).

    Committed documentation and ``--help`` pass ``sep="/"`` instead,
    because those strings are shared by every reader on every platform:
    a page regenerated on Windows must not ship backslashes to somebody
    running Linux.
    """

    mark = os.sep if sep is None else sep
    return (f"{root}{mark}<domain>{mark}<product>{mark}<valid-day>{mark}"
            f"<file>.png (domain as d02-3km / d05-111m / native_grid, "
            f"valid-day as YYYY-MM-DD)")


__all__ = [
    "DEFAULT_LAYOUT", "FLAT", "LAYOUTS", "NATIVE_GRID", "NESTED",
    "UNCLASSIFIED", "UNDATED", "delivered_name", "describe",
    "engine_name", "fs_path", "iter_rendered", "parse_engine_output",
    "place", "product_dir", "valid_day",
]

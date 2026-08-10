"""What every front door does with :mod:`gpuwm.provenance`: say it, or refuse.

:mod:`gpuwm.provenance` answers *which tree is executing*.  It judges
nothing and prints nothing, deliberately -- it is the instrument.  This
module is the POLICY built on it, and it is exactly three sentences
long:

1. **Announce.**  Every front door prints the one-line form at start and
   records :func:`receipt_block` in whatever receipt it already writes.
   A run whose output nobody can trace to a tree is how this project
   spent eight days executing a branch that had diverged 2,370 commits
   earlier with nothing anywhere announcing it.

2. **Refuse a version that contradicts its own source.**  Two numbers
   describing one tree that disagree is not a cosmetic defect: it is the
   shape of "a user reports plots labelled 1.6.2 while believing they
   installed 1.8", and nothing in the product could adjudicate it.
   :func:`version_identity_refusal`.

3. **Refuse a renderer bridge this tree did not build and did not ask
   for.**  :func:`renderer_bridge_refusal`, which is the real fix for
   the bridge being gated on EXISTENCE: ``rustwx.find_renderer()``
   returns the first candidate that is a file, and its last candidate is
   the shared staging directory ``~/.gpuwm/bridges`` -- which belongs to
   no tree at all.  Measured on the reference box while this was
   written: a worktree at ``faa1ce286db8`` (1.8.7) with no local build
   resolved the shared staging copy of ``rw_wrfbatch.exe``, whose
   embedded ``GPUWM_BRIDGE_SOURCE_REV`` is ``6a3cbaaf051a`` -- the
   1.6.0 release-prep commit.  Three minor releases of product catalog
   silently substituted into a 1.8.7 run, which is how a render produced
   153 products where the matching engine produced 168.

WHAT MUST NOT REFUSE
--------------------
A refusal that fires on a normal install is worse than the defect it
guards, so every gate here is written to be *unanswerable-means-silent*:

* A plain ``pip install gpuwm`` wheel: its code and its metadata were
  written together by pip, so the version question has one answer and
  cannot disagree; it has no git anywhere, no vendored Rust workspace,
  and its renderer can only ever come from the staging directory.  Both
  gates pass without printing a word.
* A source tree nobody installed: ``gpuwm.__version__`` honestly reports
  ``0+unknown``, and an honest "I do not know" is not a contradicting
  claim.  No refusal.
* A checkout whose renderer is built in its own ``tools/rustwx/target``:
  the binary is inside the executing tree by construction.  No refusal,
  and no bytes are read to establish it.
* A checkout that NAMES its renderer through ``GPUWM_RW_WRFBATCH``:
  explicit configuration is a declaration.  The verdict is recorded in
  the receipt; nothing is refused.

Only the silent fall-through refuses -- the case where nobody chose the
binary that ran.

The refusals are composed with :func:`gpuwm.explain.layered` and raised
as ``ValueError``, which is the convention every front door in this
package already prints as one sentence at exit 2 (see
``gpuwm.cli.main``'s refusal boundary).  The voice is the
nocturnal-radiation guard's: name what was refused, name the remedy, and
put the mechanism in the ``--explain`` half.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from gpuwm.explain import layered
from gpuwm.provenance import UNKNOWN_VERSION, Provenance, resolve

#: Set to ``0`` to suppress the startup banner.  The banner goes to
#: stderr, so a piped stdout (JSON reports, the run-plan event stream,
#: ``--show-capabilities``) is already clean; this exists for the caller
#: who wants a silent log, and for the test that pins the quiet form.
BANNER_ENV = "GPUWM_PROVENANCE_BANNER"

#: Front doors that must never be blocked by a provenance refusal:
#: they are the commands a reader runs *because* the install is
#: confusing.  ``gpuwm version`` exists to report exactly the
#: disagreement below; ``doctor`` reports it as a check; ``report``
#: packages it for someone else to read; ``update`` prints the command
#: that fixes it.  A gate that refuses its own diagnostic is a trap.
DIAGNOSTIC_COMMANDS = frozenset({
    "version", "doctor", "report", "update", "cases"})

#: The schema of :func:`receipt_block`.  Versioned because receipts
#: embed it, exactly like ``gpuwm-provenance-v1`` inside it.
RECEIPT_SCHEMA = "gpuwm-provenance-receipt-v1"


# ---------------------------------------------------------------------------
# The truthful version
# ---------------------------------------------------------------------------

def executing_version(prov: Provenance | None = None) -> str:
    """The version of the code that is RUNNING, not the version claimed.

    ``gpuwm.__version__`` asks distribution metadata BY NAME, so a
    process whose code no distribution provides still receives a
    confident number from some other ``.dist-info``.  Anything stamped
    onto an artifact has to prefer the code's own declaration over that
    borrowed one, because the stamp is what a reader quotes back months
    later when they ask why two runs differ.

    The order is "closest to the bytes first":

    1. the version the CODE declares -- ``pyproject.toml`` in a source
       tree or an editable install, and for a wheel the metadata pip
       wrote beside the code it installed (there is no second
       declaration in a wheel, and saying so is honest);
    2. failing that, the metadata of the distribution that actually
       provides this package;
    3. failing that, whatever ``gpuwm.__version__`` produced -- but only
       when provenance did not find it BORROWED;
    4. :data:`gpuwm.provenance.UNKNOWN_VERSION`, which is the honest
       answer and the one this project already ships for an uninstalled
       tree.
    """

    prov = resolve() if prov is None else prov
    if prov.code_version:
        return str(prov.code_version)
    if prov.metadata_version:
        return str(prov.metadata_version)
    if prov.reported_version and not prov.metadata_is_borrowed:
        return str(prov.reported_version)
    return UNKNOWN_VERSION


def receipt_block(prov: Provenance | None = None) -> dict:
    """The provenance block a receipt embeds.

    :meth:`gpuwm.provenance.Provenance.as_dict` plus three derived
    fields a reader of the receipt should not have to re-derive.
    Everything is JSON-safe; nothing here can raise.

    ``consistent`` and ``version_identity_refused`` are deliberately
    two fields rather than one, because they answer different
    questions and this run is a live example of them differing.  The
    resolver calls a BORROWED version inconsistent even when its digits
    happen to be right -- correctly: the number describes a different
    install.  The GATE still lets that run, because nothing about it is
    wrong.  Collapsing the two would either make a receipt claim an
    install was clean when its version was on loan, or make it claim a
    refusal that never happened.
    """

    prov = resolve() if prov is None else prov
    block = prov.as_dict()
    block["schema"] = RECEIPT_SCHEMA
    block["provenance_schema"] = prov.as_dict()["schema"]
    block["executing_version"] = executing_version(prov)
    block["consistent"] = bool(prov.is_consistent)
    block["version_identity_refused"] = (
        version_identity_refusal(prov) is not None)
    return block


# ---------------------------------------------------------------------------
# Refusal (a): the version contradicts the source it claims to describe
# ---------------------------------------------------------------------------

def version_identity_refusal(prov: Provenance | None = None) -> str | None:
    """Why this install's two version claims cannot both be true, or None.

    The comparison is between:

    * what this process REPORTS -- ``gpuwm.__version__``, which is what
      reaches every receipt, every stamped artifact, and every sentence
      a user quotes; and
    * what the running CODE DECLARES -- ``[project].version`` of the
      ``pyproject.toml`` beside the package, which is the only version
      written down by hand in a source tree.

    Both must be known, and the reported one must not be the honest
    ``0+unknown`` sentinel.  That exemption is load-bearing: a plain
    ``git clone`` with nothing installed reports ``0+unknown`` against a
    tree that declares 1.8.7, and refusing THAT would refuse the
    project's own development flow over a non-defect.

    A wheel cannot reach this refusal at all: it ships no
    ``pyproject.toml``, so its "declaration" IS its metadata (see
    :func:`gpuwm.provenance.describe_provenance`) and the two are equal
    by construction.

    What DOES reach it is the failure this program was opened for: an
    editable install whose ``.dist-info`` says one release while the
    tree it points at has been moved to another.  That is the shape of
    the field report -- plots labelled 1.6.2 from a tree the reader
    believed was 1.8 -- and until now nothing in the product compared
    the two numbers at all.
    """

    prov = resolve() if prov is None else prov
    claimed = prov.reported_version or prov.metadata_version
    declared = prov.code_version
    if not claimed or not declared:
        return None
    if claimed == UNKNOWN_VERSION or claimed == declared:
        return None

    where = prov.code_version_source or f"{prov.source_root}/pyproject.toml"
    if prov.metadata_is_borrowed:
        # No distribution provides this code, yet a number arrived
        # anyway.  Naming the borrowing is the whole remedy: the reader
        # has to know the number describes somebody else's tree.
        action = (
            f"this process reports gpuwm {claimed!r}, but NO installed "
            f"distribution provides the code it is running "
            f"({prov.package_path}) -- that number was read out of "
            f"another install's metadata, while the tree actually "
            f"executing declares {declared!r} in {where}.  Run the "
            f"installed copy, or bind this tree to its own metadata "
            f"with: pip install -e {prov.source_root}")
        why = (
            "gpuwm.__version__ asks importlib.metadata for the version of "
            "the DISTRIBUTION NAMED 'gpuwm', not for the version of the "
            "package that was imported.  When the two are different trees "
            "the call still succeeds and still returns a confident "
            "number, which then travels into every receipt and every "
            "stamped artifact this run writes.  It is at its most "
            "dangerous when the digits happen to match, which is why the "
            "check is on provenance rather than on digits, and why it "
            "only refuses when they DIFFER -- a borrowed number that "
            "agrees is recorded in the receipt and allowed to run.")
    else:
        action = (
            f"the distribution providing this code "
            f"({prov.distribution_name} {prov.metadata_version!r}) and the "
            f"source tree it points at ({where}, {declared!r}) name "
            f"different releases, so nothing can say which one is "
            f"executing.  Re-bind them with: pip install -e "
            f"{prov.source_root}")
        why = (
            "An editable install's .pth entry shadows every wheel pip "
            "writes, so `pip install --upgrade gpuwm` reports success, "
            "rewrites the .dist-info, and leaves the old source tree "
            "serving every import.  The metadata then describes a "
            "release that is not running.  Every receipt this run would "
            "write, and every version a user would quote back, comes "
            "from that metadata -- which is how a reader ends up holding "
            "products labelled with a release they never executed.")
    return layered(f"REFUSING: version identity is ambiguous -- {action}", why)


def require_consistent_version(command: str,
                               prov: Provenance | None = None) -> None:
    """Raise the version refusal, unless ``command`` is a diagnostic.

    ``command`` is the front door's own name (``run``, ``render``,
    ``rw-wps``, ...).  The members of :data:`DIAGNOSTIC_COMMANDS` are
    exempt on purpose -- they are what a reader runs to understand the
    refusal, and blocking them would leave no way out of it.
    """

    if command in DIAGNOSTIC_COMMANDS:
        return
    refusal = version_identity_refusal(prov)
    if refusal is not None:
        raise ValueError(refusal)


# ---------------------------------------------------------------------------
# The shared config load: where every front door inherits refusal (a)
# ---------------------------------------------------------------------------

def borrowed_version_origin() -> str | None:
    """Which install supplied the number ``gpuwm.__version__`` reports.

    Only meaningful when provenance found the version BORROWED: the
    resolver proved no distribution provides the executing code, yet a
    number arrived anyway, so it came from whatever distribution answers
    to the package's published name.  This asks the exact question
    ``gpuwm/__init__.py`` asked -- ``importlib.metadata`` by
    ``DISTRIBUTION_NAME`` -- and names where THAT answer lives, so the
    warning can say whose number is being worn.  Never raises; ``None``
    means the origin could not be named, and the warning degrades to
    saying so.
    """

    try:
        from importlib import metadata

        from gpuwm import DISTRIBUTION_NAME

        dist = metadata.distribution(DISTRIBUTION_NAME)
    except Exception:                                   # noqa: BLE001
        return None
    from gpuwm.provenance import direct_url

    info = direct_url(dist)
    editable = bool(info.get("dir_info", {}).get("editable", False))
    url = info.get("url") if editable else None
    location: str | None = None
    try:
        location = str(Path(dist.locate_file("")).resolve())
    except Exception:                                   # noqa: BLE001
        location = None
    try:
        named = f"{DISTRIBUTION_NAME} {str(dist.metadata['Version'])!r}"
    except Exception:                                   # noqa: BLE001
        named = DISTRIBUTION_NAME
    if url:
        suffix = f" (metadata in {location})" if location else ""
        return f"the editable install of {named} declared at {url}{suffix}"
    if location:
        return f"the installed {named} metadata in {location}"
    return f"an installed distribution named {named}"


#: One borrowed-version warning per process, like the banner: the
#: wizard's candidate loop loads dozens of configs, and a warning per
#: load would train readers to skip it.
_BORROWED_WARNED = False


def warn_borrowed_version(prov: Provenance | None = None, *,
                          stream=None) -> str | None:
    """One non-fatal line when the running code wears a borrowed number.

    BORROWED-and-agreeing is the state of every worktree beside an
    editable install -- 260 of them on the reference box -- so it must
    NEVER refuse: a refusal here bricks development.  But it is worth
    exactly one line, because the agreement is luck rather than binding,
    and the line names which install the number actually came from so a
    reader knows whose claim their receipts are wearing.

    Returns the line whenever the state warrants one -- even when the
    once-per-process latch or :data:`BANNER_ENV` ``=0`` kept it off the
    stream -- and ``None`` when there is nothing to warn about.  A
    borrowed number that DISAGREES is refusal (a)'s case and returns
    ``None`` here: a warning under a refusal would be a second, softer
    spelling of it.
    """

    global _BORROWED_WARNED
    prov = resolve() if prov is None else prov
    if not prov.metadata_is_borrowed:
        return None
    if version_identity_refusal(prov) is not None:
        return None
    origin = borrowed_version_origin() or (
        "distribution metadata found elsewhere on this interpreter's path")
    line = (f"provenance: WARNING -- gpuwm {prov.reported_version!r} is a "
            f"borrowed version: it came from {origin}, which does not "
            f"provide the code executing at {prov.source_root}; the "
            f"numbers agree, so the run continues")
    if not _BORROWED_WARNED and banner_enabled():
        _BORROWED_WARNED = True
        target = sys.stderr if stream is None else stream
        try:
            print(line, file=target, flush=True)
        except Exception:                               # noqa: BLE001
            # A warning must never be the reason a run dies.
            pass
    return line


def require_version_identity(source: str,
                             prov: Provenance | None = None) -> None:
    """Refusal (a) plus the borrowed warning, at the shared config load.

    ``gpuwm.experiment.build_experiment`` calls this beside the
    nocturnal-radiation guard -- the one load every front door shares
    (run/go/check, both prepared runners, the DA drivers, the wizard's
    candidate loop) -- so a door nobody remembered to wire still
    inherits the policy.  Same seam, same voice, same prefix: the
    refusal names the config that was loading when it fired, then both
    versions, both locations, and the remedy.

    The diagnostic doors need no exemption here: none of ``version``,
    ``doctor``, ``report`` or ``update`` loads an experiment config, so
    the way out of the refusal stays open even when it fires everywhere
    else.
    """

    prov = resolve() if prov is None else prov
    refusal = version_identity_refusal(prov)
    if refusal is not None:
        raise ValueError(f"experiment config {source}: {refusal}")
    warn_borrowed_version(prov)


# ---------------------------------------------------------------------------
# Refusal (b): a renderer bridge from a tree that is not this one
# ---------------------------------------------------------------------------

#: Verdicts :func:`bridge_tree_match` can return.  Four of the five are
#: passes; only ``foreign`` refuses.
BRIDGE_VERDICTS = (
    "absent",        # nothing resolved; the caller's own error path owns it
    "in-tree",       # the binary lives inside the executing source root
    "declared",      # the caller named it through the environment override
    "stamp-matches",  # built from the exact commit this tree is on
    "unanswerable",  # no tree to match against (a wheel, or no git)
    "foreign",       # discovered silently, and from another tree
)


@dataclass(frozen=True)
class BridgeMatch:
    """Whether the renderer that would run belongs to the engine's tree."""

    bridge: str | None
    verdict: str
    matched: bool
    engine_commit: str | None
    bridge_source_rev: str | None
    basis: str

    def as_dict(self) -> dict:
        """JSON-safe, for the render receipt."""

        return {
            "bridge": self.bridge,
            "verdict": self.verdict,
            "matched": self.matched,
            "engine_commit": self.engine_commit,
            "bridge_source_rev": self.bridge_source_rev,
            "basis": self.basis,
        }


def _declared_bridge(env_var: str) -> Path | None:
    """The path the caller explicitly named, resolved, or None."""

    named = os.environ.get(env_var)
    if not named:
        return None
    try:
        return Path(named).resolve()
    except (OSError, ValueError):
        return None


def bridge_tree_match(bridge, *, env_var: str,
                      prov: Provenance | None = None) -> BridgeMatch:
    """Does ``bridge`` belong to the tree that is about to invoke it?

    Answered in the cheapest decisive order, and every step that cannot
    decide hands back a PASS rather than a suspicion:

    1. ``bridge is None`` -- nothing resolved.  Not this gate's business;
       the caller already has a "not built" message with a build
       one-liner in it.
    2. The binary is inside the executing source root.  That is what
       ``tools/rustwx/target/release`` and ``<root>/libexec/bridges``
       both are, and no bytes need reading to establish it.
    3. ``env_var`` names this exact file.  Explicit configuration is a
       declaration; the project already treats this variable that way
       (``rustwx.find_renderer`` makes an override naming a MISSING file
       a hard error rather than a fall-through).
    4. There is no tree to match against -- an installed wheel, or a
       source tree with no git identity (an unpacked sdist, a tarball, a
       vendored copy).  Unanswerable, therefore silent.  This is the
       path every ordinary ``pip install`` takes.
    5. The binary carries exactly one ``GPUWM_BRIDGE_SOURCE_REV`` stamp
       equal to this checkout's HEAD.  The stamp is the project's own
       existing proof of "built from this commit" -- the release cut
       verifies bundles with it (``gpuwm.bridge_assets
       .verify_source_revision``) -- so run time reuses that standard
       rather than inventing a second one.
    6. Otherwise the binary was discovered silently and came from
       somewhere else.  ``foreign``.
    """

    prov = resolve() if prov is None else prov
    if bridge is None:
        return BridgeMatch(None, "absent", False, None, None,
                           "no renderer resolved")
    try:
        bridge_path = Path(bridge).resolve()
    except (OSError, ValueError):
        bridge_path = Path(bridge)
    engine_commit = (prov.git or {}).get("commit_full")

    try:
        source_root = Path(prov.source_root)
        if bridge_path == source_root or bridge_path.is_relative_to(
                source_root):
            return BridgeMatch(
                str(bridge_path), "in-tree", True, engine_commit, None,
                f"the renderer is inside the executing source root "
                f"{source_root}")
    except (OSError, ValueError):
        pass

    declared = _declared_bridge(env_var)
    if declared is not None and declared == bridge_path:
        return BridgeMatch(
            str(bridge_path), "declared", True, engine_commit, None,
            f"{env_var} names this file explicitly")

    if prov.install_kind == "wheel" or not engine_commit:
        return BridgeMatch(
            str(bridge_path), "unanswerable", True, engine_commit, None,
            "the engine is not a git checkout, so there is no tree "
            "revision to match the renderer against")

    revisions: tuple[str, ...] = ()
    try:
        from gpuwm.bridge_assets import embedded_source_revisions

        revisions = embedded_source_revisions(bridge_path.read_bytes())
    except Exception as error:                          # noqa: BLE001
        return BridgeMatch(
            str(bridge_path), "foreign", False, engine_commit, None,
            f"the renderer's source-revision stamp could not be read "
            f"({type(error).__name__}: {error})")
    stamp = revisions[0] if len(revisions) == 1 else None
    if stamp == engine_commit:
        return BridgeMatch(
            str(bridge_path), "stamp-matches", True, engine_commit, stamp,
            "the renderer's embedded GPUWM_BRIDGE_SOURCE_REV is this "
            "checkout's HEAD")
    if not revisions:
        detail = "it carries no GPUWM_BRIDGE_SOURCE_REV stamp at all"
    elif len(revisions) > 1:
        detail = (f"it carries {len(revisions)} distinct source-revision "
                  f"stamps ({', '.join(revisions)})")
    else:
        detail = f"it was built from {revisions[0]}"
    return BridgeMatch(
        str(bridge_path), "foreign", False, engine_commit,
        revisions[0] if revisions else None, detail)


def renderer_bridge_refusal(bridge, *, env_var: str | None = None,
                            prov: Provenance | None = None,
                            match: BridgeMatch | None = None) -> str | None:
    """Why this renderer may not be used by this tree, or None.

    Only a ``foreign`` verdict refuses -- a binary nobody chose, from a
    tree that is not the one running.  Everything else, including every
    shape a plain wheel install can produce, returns ``None``.
    """

    if env_var is None:
        from gpuwm.rustwx import RENDERER_ENV

        env_var = RENDERER_ENV
    prov = resolve() if prov is None else prov
    match = bridge_tree_match(bridge, env_var=env_var,
                              prov=prov) if match is None else match
    if match.matched or match.verdict == "absent":
        return None

    from gpuwm.bridges import RUSTWX_CRATE_RELATIVE, cargo_build_one_liner

    build = cargo_build_one_liner(RUSTWX_CRATE_RELATIVE)
    engine = match.engine_commit or "an unknown commit"
    action = (
        f"the rust render engine at {match.bridge} does not belong to the "
        f"tree invoking it: {match.basis}, while this engine is running "
        f"from {prov.source_root} at {engine[:12]}.  Nothing chose that "
        f"binary -- it was the last candidate that happened to exist.  "
        f"Build the matching engine in this tree: {build}   ... or, if "
        f"that binary IS the one you mean to use, say so: set "
        f"{env_var}={match.bridge}")
    why = (
        "The renderer is resolved by taking the first candidate path "
        "that is a file, and the last candidate is the shared staging "
        "directory every gpuwm on this machine writes into.  A checkout "
        "with no local build therefore silently borrows whichever "
        "engine some other tree staged there, and the two do not render "
        "the same catalog: a stale engine drew 153 products where the "
        "matching one drew 168, with nothing in the output saying which "
        "engine had run.  This gate is deliberately narrow -- it asks "
        "nothing of an installed wheel (which has no tree to match and "
        "no way to build one), nothing of a renderer inside the "
        "executing tree, and nothing of a renderer the caller named "
        "through the environment.  It refuses exactly the case where "
        "nobody chose.")
    return layered(f"REFUSING: renderer bridge is from another tree -- "
                   f"{action}", why)


def require_matched_renderer(bridge, *, env_var: str | None = None,
                             prov: Provenance | None = None) -> BridgeMatch:
    """Raise the bridge refusal when it applies; return the verdict.

    Returned rather than discarded so the caller can put the verdict in
    its receipt: "this render used the in-tree engine" is a fact worth
    recording even on the passing path, and it is the fact that was
    missing when a foreign engine's output had to be explained after
    the fact.
    """

    if env_var is None:
        from gpuwm.rustwx import RENDERER_ENV

        env_var = RENDERER_ENV
    prov = resolve() if prov is None else prov
    match = bridge_tree_match(bridge, env_var=env_var, prov=prov)
    refusal = renderer_bridge_refusal(bridge, env_var=env_var, prov=prov,
                                      match=match)
    if refusal is not None:
        raise ValueError(refusal)
    return match


# ---------------------------------------------------------------------------
# Announcement
# ---------------------------------------------------------------------------

#: One announcement per process.  A front door that dispatches into
#: another front door (``gpuwm go`` runs the prepared runners in-process)
#: must not print the same line twice.
_ANNOUNCED = False


def banner_enabled() -> bool:
    """Is the startup banner wanted?  Anything but ``0`` means yes."""

    return os.environ.get(BANNER_ENV, "1").strip() != "0"


def reset_announcement() -> None:
    """Forget that this process has announced (tests, and only tests).

    Clears both once-per-process latches -- the banner's and the
    borrowed-version warning's -- because a test that resets one and
    not the other inherits the previous test's silence.
    """

    global _ANNOUNCED, _BORROWED_WARNED
    _ANNOUNCED = False
    _BORROWED_WARNED = False


def announce(command: str, *, stream=None,
             prov: Provenance | None = None) -> Provenance:
    """Print the one-line provenance form once, then enforce refusal (a).

    stderr, not stdout: several of these front doors write JSON or an
    event stream to stdout and a banner in it would be a parse error for
    the program reading it.  The one line always names the command, so a
    log holding several runs can be read back.

    Returns the resolved provenance so a caller that is about to write a
    receipt does not resolve it a second time.
    """

    global _ANNOUNCED
    prov = resolve() if prov is None else prov
    if not _ANNOUNCED and banner_enabled():
        _ANNOUNCED = True
        target = sys.stderr if stream is None else stream
        try:
            print(f"{command}: {prov.banner()}", file=target, flush=True)
        except Exception:                               # noqa: BLE001
            # A banner must never be the reason a run dies -- a closed
            # or replaced stderr is somebody's legitimate setup.
            pass
    require_consistent_version(command.split()[-1], prov)
    return prov


def announce_for_main(command: str, *, explain: bool = False,
                      stream=None) -> str | None:
    """:func:`announce` for an entry point with no refusal boundary.

    ``gpuwm.cli.main`` catches ``ValueError`` and prints it as one
    sentence at exit 2; the standalone console scripts
    (``rw-wps``/``gpuwm-wrf-init``, the two prepared runners,
    ``gpuwm-wrf-runtime-check``, ``gpuwm-mapped-inspect``) have no such
    boundary and would turn a refusal into a traceback.  So this one
    RETURNS the message, already rendered at the requested layer, and
    the caller prints it with its own prefix and returns 2.

    ``None`` means "announced, nothing to refuse", which is every
    healthy install.
    """

    from gpuwm.explain import render as render_layer

    try:
        announce(command, stream=stream)
    except ValueError as refusal:
        return render_layer(str(refusal), explain=explain, command=command)
    return None


__all__ = [
    "BANNER_ENV", "BRIDGE_VERDICTS", "BridgeMatch", "DIAGNOSTIC_COMMANDS",
    "RECEIPT_SCHEMA", "announce", "announce_for_main", "banner_enabled",
    "borrowed_version_origin", "bridge_tree_match",
    "executing_version", "receipt_block", "renderer_bridge_refusal",
    "require_consistent_version", "require_matched_renderer",
    "require_version_identity",
    "reset_announcement", "version_identity_refusal",
    "warn_borrowed_version",
]

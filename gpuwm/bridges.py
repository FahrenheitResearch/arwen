"""Locate the built Rust bridge executables outside a source checkout.

The pip wheel deliberately ships no compiled Rust: the fail-closed GRIB
decoders and the CPU preprocessing library are built once from the
vendored ``tools/grib1_bridge`` workspace (``cargo build --release
--locked --offline``) and then *pointed at*.  This module is the single
resolution mechanism shared by ingest (:func:`gpuwm.ingest.grib
.build_rust_bridge`), ``gpuwm doctor``, and documentation:

1. an explicit per-executable environment variable
   (:data:`BRIDGE_ENV`) naming the built file;
2. a source checkout's own ``tools/grib1_bridge/target/{release,debug}``
   build tree (the developer path -- ingest may also *build* there);
3. ``<root>/libexec/bridges`` beside the package (the sealed runtime
   archive layout);
4. the user-level default directory :func:`default_bridge_dir`
   (``~/.gpuwm/bridges``), where a wheel user copies the built
   executables once.

Nothing here runs cargo; resolution is read-only so ``gpuwm doctor``
can report the estate without side effects.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil

#: Bridge executable -> the environment variable that names a prebuilt
#: copy.  The same variables drive the sealed-runtime decoder binding in
#: :mod:`gpuwm.source_cli`, so one mechanism serves both install modes.
BRIDGE_ENV = {
    "grib1_bridge": "GPUWM_GRIB1_BRIDGE",
    "gfs_grib2_bridge": "GPUWM_GFS_GRIB2_BRIDGE",
    "hrrr_grib2_bridge": "GPUWM_HRRR_DECODER",
    "grib2_inventory": "GPUWM_GRIB2_INVENTORY",
    "grib2_dump": "GPUWM_GRIB2_DUMP",
}

#: The bridge crate's path inside a checkout.
CRATE_RELATIVE = "tools/grib1_bridge"

#: True when the shell a remedy will be pasted into is Windows
#: PowerShell rather than a POSIX shell.  Windows PowerShell 5.1 -- the
#: in-box shell the project's own PowerShell install route reaches --
#: has no ``&&`` and no ``. "$HOME/..."``, so a remedy written for one
#: shell is a parse error in the other.
WINDOWS_SHELL = os.name == "nt"


def _shell_path(*parts: str) -> str:
    """A relative path spelled for the shell the remedy targets."""

    joined = "/".join(part.strip("/") for part in parts if part)
    return joined.replace("/", "\\") if WINDOWS_SHELL else joined


def _parent_hops(*parts: str) -> str:
    """The ``..`` walk that undoes ``_shell_path(*parts)``, exactly.

    One ``..`` per component the ``cd`` descended, counted rather than
    hardcoded: ``tools/rustwx`` is two, ``gpuwm/tools/rustwx`` is three,
    and a remedy that guesses walks the reader somewhere else.
    """

    depth = sum(len([c for c in part.strip("/").split("/") if c])
                for part in parts if part)
    return _shell_path(*[".."] * depth)


#: The renderer/fetch-backbone crate's path inside a checkout.  Declared
#: here so all three artifacts share one shell-correctness rule.
RUSTWX_CRATE_RELATIVE = "tools/rustwx"


def cargo_build_one_liner(crate_relative: str = CRATE_RELATIVE) -> str:
    """``cd <crate>``, the offline build, and the ``cd`` back -- one line.

    The separator is the shell's, not a habit: ``&&`` is a parse error in
    Windows PowerShell 5.1, so there it is ``;``.

    It returns to the directory it started in.  ``gpuwm doctor`` prints
    one remedy block per gap and says they run in the order printed, so
    a block that leaves the shell two levels down inside the crate is a
    block that breaks the next one's relative paths -- and on a fresh
    machine there are always several.  The tidier POSIX spelling is a
    subshell, ``(cd X && cargo build ...)``, which the README uses; it
    has no Windows PowerShell 5.1 equivalent (there is no subshell that
    contains a location change there), so adopting it would give the two
    shells different SHAPES rather than one shape with two separators.
    The explicit ``cd`` back is honest and identical in both, and is
    already the form the README documents for PowerShell.
    """

    separator = ";" if WINDOWS_SHELL else " &&"
    return (f"cd {_shell_path(crate_relative)}{separator} "
            f"cargo build --release --locked --offline{separator} "
            f"cd {_parent_hops(crate_relative)}")


#: The one-liner that builds every bridge, run from a source checkout's
#: own root.  ``--offline`` works because the crate vendors its
#: dependencies (``tools/grib1_bridge/vendor/crates-io`` plus that
#: workspace's ``.cargo/config.toml``): no network, no registry.
CARGO_BUILD_HINT = cargo_build_one_liner(CRATE_RELATIVE)

#: Where a pip user gets the sources the wheel does not carry.  Same URL
#: and same clone directory as README's install section, so the two
#: cannot drift into telling different stories.
REPOSITORY_URL = "https://github.com/FahrenheitResearch/arwen"
CLONE_DIR = "gpuwm"


def cargo_is_installed() -> bool:
    """Is a Rust toolchain on PATH?  Read-only, runs nothing."""

    return shutil.which("cargo") is not None


def rust_toolchain_install_command() -> str:
    """The command that installs Rust here.  A command, and nothing else.

    No label, no parenthetical.  ``install Rust: winget ... (or
    https://rustup.rs)`` reads fine to someone who already knows what a
    shell is and is a syntax error to everyone the remedy exists for;
    anything that is not the command belongs on its own ``#`` line.
    """

    if WINDOWS_SHELL:
        return "winget install --id Rustlang.Rustup -e --source winget"
    return ("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
            "| sh -s -- -y")


def cargo_activation_command() -> str:
    """Put a freshly installed cargo on PATH in the CURRENT shell.

    rustup edits the login profile, which does nothing for the shell
    already running -- so a bootstrap that installs Rust and then calls
    ``cargo`` fails on the very machine the bootstrap is for, with
    ``cargo: command not found`` two lines after installing cargo.
    """

    if WINDOWS_SHELL:
        return '$env:Path = "$env:USERPROFILE\\.cargo\\bin;$env:Path"'
    return '. "$HOME/.cargo/env"'


def build_from_clone_hint(
        crate_relative: str = CRATE_RELATIVE) -> tuple[str, ...]:
    """The whole bootstrap, for an install with no Rust sources in it.

    The pip wheel ships no crates, so on a pip-only machine NOTHING can
    decode GRIB until these are built -- and the short
    :data:`CARGO_BUILD_HINT` names ``tools/grib1_bridge``, a directory
    that does not exist there.  A remedy pointing at a missing directory
    is worse than no remedy: it reads as a broken install rather than a
    missing step, and the v1.0.0 field report says exactly that.

    Every line below is either a command that runs as printed, in order,
    from any working directory -- in the shell this platform actually
    has -- or a ``#`` comment, which is inert if pasted with the rest.
    Nothing is a mixture of the two.  The measured cost of the whole
    chain is roughly two minutes on a warm machine, dominated by the
    compile.

    Two properties the *sequence* needs, which no single line shows.
    ``git clone`` fails when the directory is already there, and a
    pip-only machine gaps every bridge at once, so doctor prints this
    same clone six or seven times in one report: the note above it says
    to skip the line rather than leaving the reader to read an error as
    a failure.  And the last line walks back out of the crate, because
    the next block starts where this one ends -- ``cd`` with no return
    is how a paste that "runs in the order printed" stops doing so
    after the first block.
    """

    lines: list[str] = []
    if not cargo_is_installed():
        lines += [
            "  # Rust is not on PATH.  These two lines install it and make",
            "  # cargo usable in THIS shell (rustup only edits the profile,",
            "  # which the already-running shell never re-reads).",
            f"  {rust_toolchain_install_command()}",
            f"  {cargo_activation_command()}",
        ]
    lines += [
        f"  # skip the clone if {CLONE_DIR}/ already exists -- doctor prints",
        "  # one block per gap and they all start from the same clone, so",
        "  # this line repeats; a second clone into it just errors.",
        f"  git clone {REPOSITORY_URL} {CLONE_DIR}",
        f"  cd {_shell_path(CLONE_DIR, crate_relative)}",
        "  cargo build --release --locked --offline",
        "  # back out to where you started, so the next block's relative",
        "  # paths still mean what they say.",
        f"  cd {_parent_hops(CLONE_DIR, crate_relative)}",
    ]
    return tuple(lines)


def sources_present(crate_relative: str = CRATE_RELATIVE) -> bool:
    """Does this install carry the Rust sources for ``crate_relative``?

    False on a pip install, where every ``cd tools/...`` instruction
    names a directory that does not exist.  It answers about the crate
    the caller is actually going to name: this used to answer for
    ``tools/grib1_bridge`` whatever it was asked, so a tree carrying one
    crate and not the other got a ``cd`` into the missing one.
    """

    return (_package_parent() / Path(crate_relative)).is_dir()


def install_aware_build_hint(one_liner: str,
                             crate_relative: str = CRATE_RELATIVE) -> str:
    """A cargo one-liner, or the whole bootstrap when there is no crate.

    One call site, two true answers.  Handing a pip user
    ``cd tools/rustwx && cargo build`` sends them to a directory that
    does not exist, which reads as a broken install rather than a
    missing step.

    Neither answer carries a leading newline.  The bootstrap used to,
    which meant every caller that wrote its own headline --
    ``"# rebuild it:\\n" + hint`` -- printed a blank line and then a
    command at column 0, out from under the label it belonged to.  A
    caller that wants the hint on its own line adds the newline it is
    asking for.
    """

    if sources_present(crate_relative):
        return one_liner
    return "\n".join(build_from_clone_hint(crate_relative))


def install_aware_one_line_hint(one_liner: str,
                                crate_relative: str = CRATE_RELATIVE) -> str:
    """The same two truths, for a caller that may emit only ONE line.

    ``install_aware_build_hint``'s pip answer is the whole bootstrap --
    Rust install, PATH activation, clone, build -- and a caller bound to
    a single physical line cannot print it.  Inlining the checkout
    one-liner anyway is the failure this exists to avoid: it names a
    directory a wheel install does not have.

    So the honest one-line composition is a pointer.  ``gpuwm doctor``
    already assembles the full bootstrap for this exact machine, and
    naming it is true on every install, where ``cd tools/rustwx`` is
    true on only some.
    """

    if sources_present(crate_relative):
        return one_liner
    return ("run `gpuwm doctor`, which prints the build steps for this "
            "install (a wheel carries no Rust sources, so this one needs "
            "a clone)")


def default_bridge_dir() -> Path:
    """User-level directory for prebuilt bridges: ``~/.gpuwm/bridges``."""

    return Path.home() / ".gpuwm" / "bridges"


def _package_parent() -> Path:
    """The directory containing the ``gpuwm`` package.

    A source checkout's repository root, or ``site-packages`` for an
    installed wheel (where the crate does not exist).
    """

    return Path(__file__).resolve().parent.parent


def crate_dir() -> Path:
    """The vendored Rust workspace of a source checkout (may not exist)."""

    return _package_parent() / "tools" / "grib1_bridge"


def executable_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def artifact_candidates(env_var: str, filename: str) -> tuple[Path, ...]:
    """Deterministic candidate paths for one built artifact, best first.

    THE resolution order for everything ``tools/grib1_bridge`` builds
    (bridge executables and the CPU preprocessing library alike):
    environment override, checkout release, checkout debug, ``libexec``
    beside the package, user-level default directory.  The environment
    override comes first; a missing file it names is the caller's error
    to raise (never silently skipped).
    """

    candidates: list[Path] = []
    override = os.environ.get(env_var)
    if override:
        candidates.append(Path(override))
    root = _package_parent()
    candidates.extend((
        crate_dir() / "target" / "release" / filename,
        crate_dir() / "target" / "debug" / filename,
        root / "libexec" / "bridges" / filename,
        default_bridge_dir() / filename,
    ))
    return tuple(candidates)


def find_artifact(env_var: str, filename: str) -> Path | None:
    """First existing candidate, or None.

    An environment override that names a missing file is a hard error:
    explicit configuration must fail loudly, not fall through to a
    different executable or library.
    """

    override = os.environ.get(env_var)
    for candidate in artifact_candidates(env_var, filename):
        if candidate.is_file():
            return candidate.resolve()
        if override and candidate == Path(override):
            raise FileNotFoundError(
                f"{env_var} names a missing file: {candidate}")
    return None


def bridge_candidates(name: str) -> tuple[Path, ...]:
    """Deterministic candidate paths for bridge ``name``, best first."""

    if name not in BRIDGE_ENV:
        raise ValueError(f"unknown bridge executable {name!r}; known: "
                         f"{sorted(BRIDGE_ENV)}")
    return artifact_candidates(BRIDGE_ENV[name], executable_name(name))


def find_bridge(name: str) -> Path | None:
    """First existing candidate for bridge ``name``, or None.

    See :func:`find_artifact` for the fail-loud override contract.
    """

    if name not in BRIDGE_ENV:
        raise ValueError(f"unknown bridge executable {name!r}; known: "
                         f"{sorted(BRIDGE_ENV)}")
    return find_artifact(BRIDGE_ENV[name], executable_name(name))


def artifact_remedy(*, env_var: str, filename: str, subject: str,
                    crate_relative: str = CRATE_RELATIVE,
                    one_liner: str = "") -> str:
    """The remedy for one missing built artifact, true for THIS install.

    Two installs, two different true answers.  In a source checkout the
    crate is right there and the fix is one ``cargo build`` -- with the
    real destination path, not a ``<clone>`` the reader has to expand.
    On a pip install there is no crate at all, so the remedy starts from
    the clone; printing the checkout's one-liner there names a directory
    that does not exist and reads as a broken install.

    Every continuation line is a command or a ``#`` comment, so the
    whole block survives being pasted in one go.  The guidance that used
    to trail as bare prose (``then EITHER set ...``) is commented for
    exactly that reason: a reader who selects the block should not have
    to prune it first.

    One implementation for the bridges, the renderer and the fetch
    backbone.  Three copies is how two of them kept ``<clone>`` and the
    "exact copy-pasteable" claim after the third stopped saying it.
    """

    crate = _package_parent() / Path(crate_relative)
    built = crate / "target" / "release" / filename
    if crate.is_dir():
        return (
            f"# build {subject} once, from this checkout's root:\n"
            f"  {one_liner or CARGO_BUILD_HINT}\n"
            f"  # that produces {built},\n"
            f"  # which gpuwm then finds on its own (or set {env_var} "
            f"to it).")
    steps = "\n".join(build_from_clone_hint(crate_relative))
    clone_built = _shell_path(CLONE_DIR, crate_relative,
                              "target/release") + \
        ("\\" if WINDOWS_SHELL else "/") + filename
    return (
        f"# this install carries no Rust sources -- the wheel ships none --\n"
        f"# so {subject} must be built from a clone.  About two minutes, "
        f"once:\n"
        f"{steps}\n"
        f"  # then EITHER point gpuwm at the build:\n"
        f"  #   {env_var}={clone_built}\n"
        f"  #   (relative to the directory you ran git clone in)\n"
        f"  # OR copy the built executables into {default_bridge_dir()},\n"
        f"  # where gpuwm looks by default.")


def bridge_remedy(name: str) -> str:
    """The remedy for a missing GRIB bridge, true for THIS install."""

    return artifact_remedy(
        env_var=BRIDGE_ENV[name], filename=executable_name(name),
        subject="the GRIB bridges")


__all__ = [
    "BRIDGE_ENV", "CARGO_BUILD_HINT", "CLONE_DIR", "CRATE_RELATIVE",
    "REPOSITORY_URL", "RUSTWX_CRATE_RELATIVE", "WINDOWS_SHELL",
    "artifact_candidates", "cargo_build_one_liner",
    "artifact_remedy", "bridge_candidates", "bridge_remedy",
    "build_from_clone_hint", "cargo_activation_command",
    "cargo_is_installed", "crate_dir", "default_bridge_dir",
    "executable_name", "find_artifact", "find_bridge",
    "install_aware_build_hint", "rust_toolchain_install_command",
    "install_aware_one_line_hint",
    "sources_present",
]

"""Locate the built Rust bridge executables outside a source checkout.

A platform wheel SHIPS the compiled Rust: the fail-closed GRIB decoders
and the CPU preprocessing library are built from the vendored
``tools/grib1_bridge`` workspace (``cargo build --release --locked
--offline``) and staged into :func:`packaged_bridge_dir` before the
wheel is built, so ``pip install gpuwm`` lands a complete install on
every platform a bundle exists for.  This module is the single
resolution mechanism shared by ingest (:func:`gpuwm.ingest.grib
.build_rust_bridge`), ``gpuwm doctor``, and documentation:

1. an explicit per-executable environment variable
   (:data:`BRIDGE_ENV`) naming the built file;
2. a source checkout's own ``tools/grib1_bridge/target/{release,debug}``
   build tree (the developer path -- ingest may also *build* there);
3. ``<root>/libexec/bridges`` beside the package (the sealed runtime
   archive layout);
4. :func:`packaged_bridge_dir` -- ``gpuwm/libexec/bridges`` INSIDE the
   installed package, which is what a platform wheel carries.  It sits
   below the checkout rungs so a developer's own build still wins, and
   above ``~/.gpuwm/bridges`` because bytes shipped with this version
   are version-matched by construction while a fetched bundle is only
   as fresh as the last ``gpuwm fetch-bridges``;
5. the user-level default directory :func:`default_bridge_dir`
   (``~/.gpuwm/bridges``), which ``gpuwm fetch-bridges``
   (:mod:`gpuwm.bridge_assets`) stages the release's prebuilt bundle
   into, and where a wheel user otherwise copies their own build once.

The ``py3-none-any`` fallback wheel -- the one pip resolves on a
platform with no published bundle -- carries no rung 4, and every
consumer of a missing artifact must refuse BY NAME with
:func:`artifact_remedy` rather than degrade into a Python
reimplementation of the decoder.

Nothing here runs cargo; resolution is read-only so ``gpuwm doctor``
can report the estate without side effects.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
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
# `gpuwm_mapped_engine` is deliberately NOT in this map, for the same
# reason `rw_netcdf` is not: this map's consumers resolve through
# :func:`crate_dir`, which is the grib1_bridge crate, and the mapped
# engine builds in `tools/rw_wps`.  An entry here would make
# :func:`find_bridge` miss a checkout build and answer from a staged
# copy instead -- a stale-binary answer wearing a fresh-checkout face.
# Its ladder lives in :mod:`gpuwm.mapped_engine_bridge`; its contract
# marker is in :data:`BRIDGE_ABI_MARKERS` below, which is keyed by
# artifact and not by this map.

#: The bridge crate's path inside a checkout.
CRATE_RELATIVE = "tools/grib1_bridge"

#: Bridge executable -> a byte marker of the CONTRACT it speaks, which
#: must appear in the built binary.
#:
#: A bridge is not "whatever executable has the right basename".  The
#: wheel ships no Rust, so the binaries on a machine were built from
#: some checkout at some time, and an upgrade of the Python half does
#: not touch them.  1.1.0 changed the GFS series file from two columns
#: to three; a 1.0.1 bridge still launched, still printed its usage
#: diagnostic, and `gpuwm doctor` therefore reported it `ok` -- and then
#: every preparation died with `series line 1 must be HOUR<TAB>GRIB2`,
#: a message that blames the series file gpuwm had just written
#: correctly.  A node-7 validation run found the cause only by diffing
#: two git tags.
#:
#: Each marker is a literal the CURRENT contract compiles into the
#: binary and the previous one did not, so a stale build fails the
#: handshake statically -- no execution, no new bridge CLI surface, and
#: it works on the already-built binaries a user has on disk today.
#: This is the same mechanism `gpuwm.native_wrf_distribution` applies
#: before sealing a distribution, applied one step earlier: at the
#: doctor report, which is where a user looks before they burn a run.
#:
#: Adding a marker is part of changing a bridge contract.  Choose the
#: literal that spells the contract out (the series grammar, the usage
#: line naming the argument vector), never a version number, which a
#: rebuild bumps whether or not anything changed.
BRIDGE_ABI_MARKERS = {
    "grib1_bridge": b"usage: grib1_bridge INPUT.grb OUTPUT_DIR",
    "gfs_grib2_bridge": (
        b" must be HOUR<TAB>GRIB2[<TAB>FORECAST_PROCESS_ID]"),
    "hrrr_grib2_bridge": (
        b"usage: hrrr_grib2_bridge WRFNAT_F00 WRFNAT_F01 SOIL_F00 "
        b"SOIL_F01 OUTPUT_DIR EXPECTED_CYCLE I_START I_END J_START "
        b"J_END"),
    # The inventory contract grew the ensemble-identity columns
    # (typeOfEnsembleForecast, encoded ensemble size, derived-forecast
    # statistic code) beside the perturbation number it always carried,
    # and then the pv column (Section 4's coordinate octets -- the
    # hybrid A/B coefficient channel).  The marker is the header tail,
    # which only a binary speaking the grown contract contains: a stale
    # build would inventory a hybrid model-level file without the
    # coefficients that price its pressure ladder, and the decode gate
    # would refuse it at run time with a rebuild remedy -- this catches
    # it statically instead.
    "grib2_inventory": (
        b"minimum\tmaximum\t"
        b"ensemble_type\tensemble_size\tderived_forecast\tpv"),
    "grib2_dump": (
        b"parameter\tcenter\tsubcenter\tmaster_table_version\t"
        b"local_table_version\tlevel_type"),
    # A library, so the literal is an exported symbol name rather than a
    # usage line: `bw_dealias_rift_v1` is the refinement entry point, and
    # it is the contract that matters here because the default engine
    # runs refinement.  A build predating it exports `bw_dealias` alone,
    # loads cleanly, answers the legacy ABI probe with 1, and then fails
    # inside the first refined solve -- which is exactly the class of
    # stale build this table exists to catch statically.
    "region_global_dealias": b"bw_dealias_rift_v1",
    # The NetCDF writer cdylib behind the DEFAULT wrfout engine AND the
    # DEFAULT wrfinput/wrfbdy export.  A library, so the literal is an
    # exported symbol name, and it names the newest capability a default
    # path depends on: it was `gpuwm_ncwrite_write_record` while the
    # record dimension was that capability (a build predating it exports
    # every other entry point, loads cleanly, answers the version probe
    # -- and then cannot write a `Times` variable), and it is now the
    # read-back sweep, because the wrfinput export verifies every float
    # variable for finiteness before it publishes.  Spelled to match
    # gpuwm.io.nc_writer_bridge.ABI_MARKER; a test binds the two.
    "netcdf_writer": b"gpuwm_ncwrite_scan_nonfinite",
    # The mapped decode engine behind `gpuwm prep --source mapped`.  The
    # marker is its OUTPUT SCHEMA name rather than a usage line, because
    # that is the literal which changes exactly when the frameset
    # contract changes: a binary built before a frameset change still
    # launches, still refuses politely, and would then write a directory
    # `gpuwm.mapped_engine_bridge.read_frameset` no longer reads -- the
    # 1.1.0 GFS series-file failure class, one layer further in.  Spelled
    # to match gpuwm.mapped_engine_bridge.ABI_MARKER; a test binds the
    # two.
    "gpuwm_mapped_engine": b"gpuwm-mapped-frameset-v1",
    # The static-field builder cdylib (tools/rustwx/crates/static-fields),
    # the default engine for the WPS-geogrid-equivalent statics from the
    # static-rust-port lanes on.  A library, so the literal is an
    # exported symbol name: a build that loads and answers the version
    # probe but predates the field build cannot produce a single static
    # field.  Spelled to match gpuwm.static.rust_bridge.ABI_MARKER; a
    # test binds the two.
    "static_fields": b"gpuwm_static_build_fields",
    # The observation remap cdylib (tools/rustwx/crates/obs-regrid),
    # behind the DEFAULT plan build of the observation battery.  A
    # library, so the literal is an exported symbol name: a build that
    # answers the version probe but predates the plan builder cannot
    # produce a single remap, and the battery would fall back to scipy
    # -- whose tie-breaking is traversal order -- while reporting the
    # Rust engine as present.
    "obs_regrid": b"gpuwm_obsregrid_build_plan",
}

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


def prebuilt_bundle_offer(artifact: str | None = None
                          ) -> tuple[str, ...] | None:
    """The ``gpuwm fetch-bridges`` lead-in, or None when it would lie.

    Returned only when this install can actually do it: a platform key
    for this OS/architecture, and a bundle pinned for that platform in
    the pins document *this wheel carries*.  Both are properties of
    artifacts on disk, so the offer is never a promise about a release
    that has not happened -- a tree whose pins name no platform gets the
    build-from-source remedy it has always got, unchanged.

    ``artifact``, when given, adds the third question, and it is the one
    whose absence cost a whole wave.  ``gpuwm.obs.frontdoor`` refused
    with this block at its head -- "the MRMS front door (rw_mrms) is not
    built or not found.  gpuwm fetch-bridges ..." -- because *a* bundle
    existed for the platform.  It never asked whether ``rw_mrms`` was
    IN that bundle, and it was not; running the offered command printed
    "all artifacts already staged and pin-valid" and the refusal
    repeated verbatim.  A remedy that cannot supply what the refusal is
    about is worse than no remedy: it reads as a broken machine rather
    than a missing feature, and it costs the reader the download before
    it tells them nothing changed.  Named here, once, so no caller can
    reintroduce it: an artifact the pinned bundle does not carry gets
    the build-from-source route, which is true.

    Every line is a command or a ``#`` comment, because doctor prints
    these verbatim and claims exactly that of them.
    """

    try:
        from gpuwm import bridge_assets

        pins = bridge_assets.load_pins()
        platform = bridge_assets.host_platform()
        bundle = pins.bundle_for(platform)
    except Exception:  # a broken/absent pins document is not an offer
        return None
    if bundle is None:
        return None
    if artifact is not None and not any(
            pin.artifact == artifact for pin in bundle.binaries):
        return None
    mib = bundle.bytes / (1024 * 1024)
    return (
        "  gpuwm fetch-bridges",
        f"  # one {mib:.0f} MiB download: the prebuilt {bundle.platform} "
        "bundle,",
        "  # every artifact verified against the SHA-256 pins packaged "
        "with",
        f"  # this release before it is staged into {default_bridge_dir()}.",
        "  # --from DIR stages the same bundle from a local directory, "
        "offline.",
    )


def _offer_for(artifact: str | None) -> tuple[str, ...] | None:
    """:func:`prebuilt_bundle_offer`, called the way it has always been.

    The artifact-aware parameter is an ADDITION, so a caller with
    nothing to say about a specific artifact must still reach the
    function through its original zero-argument shape.  ``gpuwm doctor``
    substitutes a stand-in for this function in its own tests, and a
    stand-in written against the old signature is not a stale test --
    it is every out-of-tree caller that ever wrapped it.  Passing
    ``None`` positionally would break all of them for no gain.
    """

    if artifact is None:
        return prebuilt_bundle_offer()
    return prebuilt_bundle_offer(artifact)


def _as_comments(block: str) -> str:
    """Comment out a build block so a paste does not also run it.

    When the prebuilt bundle is offered first, the source build is the
    alternative rather than the next step: a reader who selects the whole
    report must not clone 2.5 GB and compile for two minutes to obtain
    files the line above already staged.  The block's own ``#`` notes
    stay as they are; only its commands are demoted, indented one level
    so it is visible which lines are the ones to uncomment.
    """

    lines: list[str] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lines.append(f"  {stripped}" if stripped.startswith("#")
                     else f"  #   {stripped}")
    return "\n".join(lines)


def _bundle_first(build_block: str, artifact: str | None = None) -> str:
    """``gpuwm fetch-bridges`` first, the source build commented after it."""

    offer = _offer_for(artifact)
    if offer is None:
        return build_block
    return "\n".join(offer + (
        "  # Or build the artifacts from source instead -- the route that",
        "  # works on every platform, including ones with no published",
        "  # bundle.  Uncomment these to take it:",
        _as_comments(build_block),
    ))


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
                             crate_relative: str = CRATE_RELATIVE,
                             artifact: str | None = None) -> str:
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
    return _bundle_first("\n".join(build_from_clone_hint(crate_relative)),
                         artifact)


def install_aware_one_line_hint(one_liner: str,
                                crate_relative: str = CRATE_RELATIVE,
                                artifact: str | None = None) -> str:
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
    if _offer_for(artifact) is not None:
        return ("run `gpuwm fetch-bridges`, which stages this platform's "
                "prebuilt artifacts under the SHA-256 pins packaged with "
                "this release (`gpuwm doctor` prints the source-build "
                "route as well)")
    return ("run `gpuwm doctor`, which prints the build steps for this "
            "install (a wheel carries no Rust sources, so this one needs "
            "a clone)")


#: The two native executable formats this project runs, by magic bytes.
#: An ELF header is four bytes; a PE is ``MZ`` plus a signature at the
#: offset stored at 0x3c, so 0x40 bytes is always enough to classify.
_EXECUTABLE_HEADER_BYTES = 0x40


def native_executable_format(path: Path) -> tuple[str | None, str]:
    """``('elf'|'pe'|None, evidence)`` from the file's own header.

    A cheap read, and the ONLY safe way to ask "can this be executed"
    about a file of unknown provenance -- because on Windows, asking the
    operating system can hang forever.

    ``subprocess.run(..., timeout=...)`` bounds the *wait*, never
    ``CreateProcess`` itself.  A file with a corrupt or absent PE header
    can make the image loader raise a modal error dialog inside
    ``CreateProcess``, and in a session with no interactive desktop
    there is nothing to dismiss it: the call never returns, the timeout
    never starts, and the process is unkillable-by-timeout.  A release
    battery froze twice at exactly that call, on a probe of a file whose
    contents were sixteen bytes of ASCII.

    So every probe reads the header first and refuses non-executables
    itself.  The answer for a real bridge is unchanged -- a genuine ELF
    or PE still goes on to be launched -- and the answer for the file
    that used to hang is now a finding, in microseconds.
    """

    try:
        with Path(path).open("rb") as stream:
            head = stream.read(_EXECUTABLE_HEADER_BYTES)
    except OSError as error:
        return None, f"cannot be read: {error}"
    if head[:4] == b"\x7fELF":
        return "elf", "ELF executable header"
    if head[:2] == b"MZ" and len(head) >= 0x40:
        offset = int.from_bytes(head[0x3c:0x40], "little")
        # The signature lives past this window; its OFFSET is what a
        # truncated or fabricated MZ stub gets wrong, and a plausible
        # one is all that is needed to make launching safe to attempt.
        if 0 < offset < (1 << 24):
            return "pe", "PE executable header"
        return None, ("has an MZ stub whose PE signature offset is out of "
                      "range -- truncated or not an executable")
    return None, (f"is not a native executable ({len(head)} byte(s) read, "
                  "no ELF or PE header)")


def launchable(path: Path) -> tuple[bool, str]:
    """Is ``path`` safe and sensible to hand to the operating system?

    Format first, then the POSIX execute bit.  Both are properties of
    the file, so neither can hang, and together they cover every way a
    probe used to discover the answer the expensive way.
    """

    path = Path(path)
    if not path.is_file():
        return False, "does not exist"
    binary_format, evidence = native_executable_format(path)
    if binary_format is None:
        return False, f"exists but {evidence}"
    expected = "pe" if os.name == "nt" else "elf"
    if binary_format != expected:
        return False, (f"exists but is a {binary_format.upper()} binary on a "
                       f"host that runs {expected.upper()} -- built for "
                       "another platform")
    if os.name != "nt" and not os.access(path, os.X_OK):
        return False, "exists but is not marked executable"
    return True, evidence


class quiet_loader_errors:  # noqa: N801 - a context manager, used as a verb
    """Make the Windows image loader FAIL instead of prompting.

    ``SetErrorMode`` is per-process and inherited by children, so
    setting it around a probe covers the loader dialog that would
    otherwise appear inside ``CreateProcess``.  This is the backstop
    behind :func:`launchable`, not a replacement for it: a header check
    cannot know about a missing DLL, and a missing-DLL dialog hangs
    exactly the same way.

    A no-op everywhere but Windows.
    """

    _FAIL_FAST = 0x0001 | 0x0002 | 0x8000  # CRITICALERRORS|GPFAULT|OPENFILE

    def __enter__(self):
        self._previous = None
        if os.name != "nt":
            return self
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            self._previous = kernel32.SetErrorMode(self._FAIL_FAST)
            # SetErrorMode replaces rather than merges, so restore the
            # union: another library may have asked for its own bits.
            kernel32.SetErrorMode(self._previous | self._FAIL_FAST)
        except (AttributeError, OSError):  # pragma: no cover - not Windows
            self._previous = None
        return self

    def __exit__(self, *_exception):
        if self._previous is None:
            return False
        try:
            import ctypes

            ctypes.windll.kernel32.SetErrorMode(self._previous)
        except (AttributeError, OSError):  # pragma: no cover
            pass
        return False


def default_bridge_dir() -> Path:
    """User-level directory for prebuilt bridges: ``~/.gpuwm/bridges``."""

    return Path.home() / ".gpuwm" / "bridges"


#: Directory INSIDE the package that a platform wheel stages its
#: prebuilt Rust artifacts into.  Named once here because six separate
#: resolution ladders consume it (this module, :mod:`gpuwm.rustwx`,
#: :mod:`gpuwm.rustwx_fetch`, :mod:`gpuwm.obs.nexrad`,
#: :mod:`gpuwm.obs.frontdoor` and :mod:`gpuwm.obs.dealias_region`) and a
#: seventh copy of the string is how a rung goes missing on one door.
PACKAGED_BRIDGE_SUBDIR = ("libexec", "bridges")


def ensure_executable(path: Path) -> Path:
    """Give a wheel-shipped artifact back its executable bit.

    ``pip`` does not preserve unix modes for package data.  It applies
    them only to entries under the wheel's ``.data/scripts/``
    directory; everything else lands 0644 however the wheel recorded it.
    So the binaries this package ships arrive on Linux and macOS present,
    correct, and unrunnable -- measured, not theorised: a clean-venv
    install of the manylinux wheel put all eleven artifacts in place and
    then died with ``PermissionError: [Errno 13] Permission denied`` on
    the first ``subprocess.run``.

    The repair is done at resolution rather than asked of the user,
    because "fixed" means a bare ``pip install gpuwm`` works: a flag or a
    documented ``chmod +x`` would be a workaround, and this defect is
    invisible until a door is already being opened.

    Only files inside :func:`packaged_bridge_dir` are touched -- a
    checkout build, a ``~/.gpuwm/bridges`` copy or an environment
    override belongs to the user, and silently re-permissioning those
    would be changing something this package does not own.  A no-op on
    Windows, where execution does not consult a mode bit.
    """

    if os.name == "nt":
        return path
    try:
        packaged = packaged_bridge_dir()
        if not path.is_relative_to(packaged):
            return path
        mode = path.stat().st_mode
        if mode & 0o111:
            return path
        # Mirror the read bits: a file the user can read, they may run.
        path.chmod(mode | ((mode & 0o444) >> 2))
    except OSError as error:
        raise PermissionError(
            f"{path} is not executable and its mode could not be repaired "
            f"({error}).  pip does not preserve executable bits on package "
            f"data, so a wheel-installed bridge needs one of:\n"
            f"  chmod +x {path}\n"
            f"  # or point gpuwm at a copy you control via its environment "
            f"variable") from None
    return path


def packaged_bridge_dir() -> Path:
    """The wheel-bundled bridge directory inside this installation.

    ``<site-packages>/gpuwm/libexec/bridges`` for an installed wheel and
    ``<checkout>/gpuwm/libexec/bridges`` in a source tree, where
    ``tools/stage_wheel_bridges.py`` puts the built artifacts before a
    platform wheel is built.  Distinct from ``<root>/libexec/bridges``,
    which is BESIDE the package (the sealed-runtime archive layout) and
    therefore cannot be package data.
    """

    return Path(__file__).resolve().parent.joinpath(*PACKAGED_BRIDGE_SUBDIR)


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
    beside the package, the wheel-bundled :func:`packaged_bridge_dir`,
    user-level default directory.  The environment override comes first;
    a missing file it names is the caller's error to raise (never
    silently skipped).
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
        packaged_bridge_dir() / filename,
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
            return ensure_executable(candidate.resolve())
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


#: Public data source -> the bridge executable its preparation route
#: launches.  These are product names (the data a user asks for), never
#: case names.
SOURCE_DECODERS = {
    "hrrr": "hrrr_grib2_bridge",
    "gfs": "gfs_grib2_bridge",
    "era5": "grib1_bridge",
}


class DecoderContractError(RuntimeError):
    """A decoder is installed but does not speak this release's contract.

    Distinct from ``FileNotFoundError`` because the two have different
    remedies -- install one versus replace this one -- and because a
    caller that catches "missing" to offer a build must not silently
    swallow "wrong".
    """


class BridgeBuildError(RuntimeError):
    """A cargo build that could not produce a bridge, named by CLASS.

    A third state beside "installed and wrong" and "not installed at
    all": the sources are here, the build was attempted, and the build
    itself failed -- for a reason that is almost never about the code.
    Its own class because the remedy differs from both neighbours and
    because a caller relaying it must be able to say "this was a build,
    not your data".

    ``failure_class`` carries the short slug
    :func:`classify_cargo_failure` assigned, so a caller can branch on
    the CLASS without re-parsing English.
    """

    def __init__(self, message: str, *, failure_class: str = "build-failed"):
        super().__init__(message)
        self.failure_class = failure_class


#: Cargo/linker failure CLASSES, as ``(slug, needles, sentence)`` rows.
#:
#: A TABLE, so a newly-observed failure mode is a row and not another
#: branch.  Each ``needles`` entry is matched case-insensitively against
#: cargo's combined output; the first row that matches names the class.
#: Order matters only where one output could match two rows, and the
#: specific rows are therefore first.
#:
#: The first row is the one that cost the reproduction: on Windows a
#: cdylib that any live process has mapped cannot be replaced, so cargo
#: fails at the *link* step with a filesystem error, several screens
#: below a wall of unrelated compiler warnings.  It is not a code
#: failure and re-running it changes nothing until the holder exits.
CARGO_FAILURE_CLASSES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("artifact-held-open",
     ("being used by another process", "os error 32", "access is denied",
      "os error 5", "text file busy", "os error 26", "permission denied"),
     "a build artifact could not be replaced because another process on "
     "this machine has it open -- a running gpuwm, another worktree's "
     "build, a debugger or an antivirus scan.  Nothing about the source "
     "is wrong and re-running changes nothing until the holder exits"),
    ("build-lock-held",
     ("blocking waiting for file lock",),
     "another cargo is already building in this target directory and "
     "this one could not take the lock"),
    ("lockfile-out-of-date",
     ("the lock file needs to be updated", "--locked"),
     "the checked-in Cargo.lock does not match the manifests, and this "
     "build runs --locked so it will not silently update it"),
    ("vendor-incomplete",
     ("no matching package", "failed to select a version",
      "unable to get packages from source", "not in the vendored sources"),
     "the vendored dependency set does not satisfy this lockfile, and "
     "this build runs --offline so it cannot fetch the difference"),
)

#: What a caller sees when no row matched.  Still names the class -- a
#: build -- because the defect being guarded is a build error read as
#: something else entirely.
_UNCLASSIFIED_CARGO_FAILURE = (
    "build-failed",
    "the cargo build did not complete; its own error lines are below")

#: How many of cargo's error lines are relayed as evidence.  The point
#: of classifying is that the reader does not have to read a compiler's
#: warning wall to find the cause, so this is a tail and not a dump.
_CARGO_EVIDENCE_LINES = 6


def classify_cargo_failure(output: str) -> tuple[str, str]:
    """``(slug, sentence)`` for cargo's own output.  Never raises."""

    lowered = (output or "").lower()
    for slug, needles, sentence in CARGO_FAILURE_CLASSES:
        if any(needle in lowered for needle in needles):
            return slug, sentence
    return _UNCLASSIFIED_CARGO_FAILURE


def _cargo_evidence(output: str) -> str:
    """Cargo's error lines, without the warning wall around them.

    ``warning:`` blocks are the bulk of a normal build's output and they
    are why the reproduction's real cause -- ``failed to remove file`` --
    arrived twelve lines down.  Prefer the ``error``/``Caused by`` lines;
    fall back to the tail only when there are none.
    """

    lines = [line.rstrip() for line in (output or "").splitlines()
             if line.strip()]
    keep: list[str] = []
    in_error = False
    for line in lines:
        stripped = line.strip().lower()
        if stripped.startswith(("error", "caused by")):
            in_error = True
        elif stripped.startswith(("warning:", "warning[", "note:", "help:",
                                  "= note:", "= help:", "compiling ",
                                  "finished ", "checking ")):
            in_error = False
            continue
        if in_error:
            keep.append(line)
    tail = keep or lines
    return "\n".join(f"    {line}" for line in tail[-_CARGO_EVIDENCE_LINES:])


def cargo_build_refusal(artifact: str, crate_relative: str, *,
                        returncode: int, output: str,
                        one_liner: str | None = None) -> str:
    """The sentence a failed ``cargo build`` for ``artifact`` deserves.

    Three parts, in the order a stuck reader needs them: WHAT failed (a
    build, named), WHY (the class), and the REMEDY this install can
    actually take -- the staged bundle where one exists, the build
    one-liner where the sources do.
    """

    slug, sentence = classify_cargo_failure(output)
    remedy = install_aware_one_line_hint(
        one_liner or cargo_build_one_liner(crate_relative),
        crate_relative, artifact)
    if slug == "artifact-held-open":
        remedy = ("close whatever holds the file (the path is in the "
                  "evidence below), then re-run this command -- or use a "
                  "copy that is already built: " + remedy)
    elif slug == "build-lock-held":
        remedy = ("wait for the other build to finish and re-run this "
                  "command -- or use a copy that is already built: "
                  + remedy)
    return (
        f"the Rust bridge `{artifact}` is not built here and building it "
        f"in {crate_relative} FAILED (cargo exited {returncode}).\n"
        f"  why: {sentence}.\n"
        f"  remedy: {remedy}\n"
        f"  cargo said:\n{_cargo_evidence(output)}")


def cargo_missing_refusal(artifact: str, crate_relative: str) -> str:
    """``cargo`` itself could not be started.  A refusal, not an OSError.

    The build path used to let ``FileNotFoundError: [WinError 2]`` out
    of ``subprocess.run`` untouched, so an install with sources and no
    Rust toolchain ended in a traceback naming ``cargo`` with no hint
    that a toolchain is what is missing.
    """

    return (
        f"the Rust bridge `{artifact}` is not built here, and `cargo` "
        "could not be started to build it -- no Rust toolchain is on "
        "PATH.\n"
        "  why: this install carries the crate sources but nothing that "
        "can compile them, so no route to the decoder exists until one "
        "of the two is supplied.\n"
        "  remedy: " + install_aware_one_line_hint(
            cargo_build_one_liner(crate_relative), crate_relative, artifact)
        + "\n  # install Rust first if that route is the one taken:\n"
        f"  {rust_toolchain_install_command()}\n"
        f"  {cargo_activation_command()}")


def resolve_source_decoder(source: str) -> Path:
    """THE decoder ``source``'s preparation will launch, or a refusal.

    One function, called by the preparation wrapper AND by ``gpuwm
    doctor``, because the alternative was two resolvers with different
    answers: the wrapper resolved a *source-tree cargo workspace*
    (``<root>/tools/grib1_bridge/target/release/...``) that exists only
    in a checkout, while doctor resolved the shared ladder in this
    module.  On a wheel install doctor therefore reported the bridge
    ``gpuwm setup`` had staged in ``~/.gpuwm/bridges`` -- correctly --
    and the preparation then went looking under ``site-packages/tools``
    and refused.  "No gaps" followed by a missing file is the exact
    failure a pre-flight exists to prevent, and it can only be prevented
    by asking the same question through the same code.

    Resolution is :func:`bridge_candidates`' order: the environment
    override (a missing file it names is a hard error, never a silent
    fall-through), then a checkout's own build, then ``libexec/bridges``
    beside the package, then the user-level ``~/.gpuwm/bridges`` that
    ``gpuwm fetch-bridges`` stages into.  A checkout's build coming
    before the staged copy is deliberate: a developer's rebuild must win
    over whatever they downloaded last month.

    Existence is not the whole question, so it is not the whole answer.
    A binary built before this release's decoder contract is present,
    executable, and wrong, and until 1.8.9 this function returned it:
    ``gpuwm doctor`` asked :func:`bridge_abi_matches` afterwards and the
    production preparation did not, so a stale bridge passed the door
    that runs it while the report that checks it said green.  A 12-byte
    text file planted at each override path was returned by all three
    sources.  The gate lives INSIDE the resolver now -- one question,
    one answer, no second call for a caller to forget.

    Nothing here runs cargo, so it is safe to call from a report.
    """

    if source not in SOURCE_DECODERS:
        raise ValueError(
            f"no decoder is declared for source {source!r}; known: "
            f"{sorted(SOURCE_DECODERS)}")
    name = SOURCE_DECODERS[source]
    found = find_bridge(name)
    if found is not None:
        ok, evidence = bridge_abi_matches(name, found)
        if ok:
            return found
        raise DecoderContractError(
            f"the {source} route's decoder at {found} {evidence}.\n"
            + install_aware_build_hint(CARGO_BUILD_HINT))
    raise FileNotFoundError(
        f"the {source} route's decoder ({executable_name(name)}) is not "
        "installed here.  Searched, in order: "
        + ", ".join(str(candidate) for candidate in bridge_candidates(name))
        + "\n" + bridge_remedy(name))


def bridge_abi_matches(name: str, path: Path) -> tuple[bool, str]:
    """Does the built bridge at ``path`` speak the contract gpuwm uses?

    Read-only and static: it searches the binary for the marker rather
    than adding a ``--abi`` subcommand the already-built binaries on
    every user's disk would not have.  That is the whole point -- the
    skew this catches is precisely a binary that predates the change,
    and a handshake only new builds can answer would report the stale
    ones as broken rather than as stale, or not at all.

    A bridge with no declared marker answers ``True``: the absence of a
    marker means nobody has yet named this bridge's contract, which is
    not evidence of skew.  Fail closed on the check that IS declared,
    never on the one that is not.
    """

    marker = BRIDGE_ABI_MARKERS.get(name)
    if marker is None:
        return True, "no declared contract marker for this bridge"
    try:
        payload = Path(path).read_bytes()
    except OSError as error:
        return False, f"cannot read it to check its contract: {error}"
    if marker in payload:
        return True, "speaks this release's contract"
    return False, (
        "was built from a checkout that predates this release's "
        f"{name} contract, so it will refuse inputs gpuwm writes "
        "correctly; rebuild it from a matching checkout")


def artifact_remedy(*, env_var: str, filename: str, subject: str,
                    crate_relative: str = CRATE_RELATIVE,
                    one_liner: str = "", artifact: str | None = None) -> str:
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

    On a pip install there is now a third true answer, when the release
    published a bundle this platform can run: one download, verified
    against the packaged pins.  It leads, and the clone-and-build route
    follows it commented out -- still printed, because it is the only
    route on a platform with no bundle, and commented because a reader
    who pastes the report must not also compile what the line above
    already staged.

    ``artifact`` is the bundle name of the thing being remedied, and
    passing it is how a caller says "check that the bundle carries THIS
    one" rather than "check that a bundle exists".  See
    :func:`prebuilt_bundle_offer` for the wave that cost.  Callers that
    omit it get the old behaviour, which is correct for the artifacts
    that have always been in the bundle and is what the checkout branch
    above does anyway.
    """

    crate = _package_parent() / Path(crate_relative)
    built = crate / "target" / "release" / filename
    if crate.is_dir():
        # The build line must name the crate this remedy is ABOUT.  The
        # fallback used to be CARGO_BUILD_HINT, which is frozen to
        # CRATE_RELATIVE (tools/grib1_bridge): every caller that passed
        # a different `crate_relative` -- rw_netcdf lives in
        # tools/rustwx -- printed `cd tools/grib1_bridge` one line above
        # a "that produces .../rustwx/target/release/rw_netcdf.exe"
        # promise the command cannot keep.  Building grib1_bridge
        # produces no rw_netcdf, so a reader who pasted the block landed
        # exactly where they started, on the refusal that sent them.
        # This is the same defect
        # test_the_bridge_remedy_is_a_real_bootstrap_on_a_pip_install
        # already records being fixed once for the six bridges; it
        # survived here because this fallback ignores its own argument.
        return (
            f"# build {subject} once, from this checkout's root:\n"
            f"  {one_liner or cargo_build_one_liner(crate_relative)}\n"
            f"  # that produces {built},\n"
            f"  # which gpuwm then finds on its own (or set {env_var} "
            f"to it).")
    steps = "\n".join(build_from_clone_hint(crate_relative))
    clone_built = _shell_path(CLONE_DIR, crate_relative,
                              "target/release") + \
        ("\\" if WINDOWS_SHELL else "/") + filename
    source_route = (
        f"{steps}\n"
        f"  # building it is not the same as wiring it, so finish the job:\n"
        + install_into_default_bridge_dir(clone_built) + "\n"
        f"  # OR, instead of copying, point gpuwm at the build in place:\n"
        f"  #   {env_var}={clone_built}\n"
        f"  #   (relative to the directory you ran git clone in)")
    offer = _offer_for(artifact)
    if offer is None:
        return (
            "# this install carries no Rust sources -- the wheel ships "
            "none --\n"
            f"# so {subject} must be built from a clone.  About two "
            "minutes, once:\n"
            f"{source_route}")
    return (
        "# this install carries no Rust sources -- the wheel ships none --\n"
        f"# so there are two routes to {subject}: the prebuilt bundle\n"
        "# this release published, or a clone and a build.  The download\n"
        "# is one command:\n"
        + "\n".join(offer) + "\n"
        "  # Or build it from source instead -- the route that works on\n"
        "  # every platform, including ones with no published bundle.\n"
        "  # Uncomment these to take it:\n"
        + _as_comments(source_route))


def install_into_default_bridge_dir(built: str) -> str:
    """Copy one built artifact where gpuwm looks by default: COMMANDS.

    The wiring step used to be offered as two ``#`` alternatives -- copy
    here, or export that -- and a reader who did exactly what the
    contract promises ("every line is a command to run as printed, in
    the order printed, or a ``#`` comment") ran the whole report and
    still ended at six MISSING bridges, because the only step that
    finishes the job was commented out on both branches.  A choice
    between two alternatives is real, but one of them can be the
    default: this is the copy, as commands, with the environment
    variable demoted to the comment beneath it.

    The destination is :func:`default_bridge_dir` spelled out in full
    rather than through ``$HOME``/``$env:USERPROFILE``, so the path the
    reader pastes is the path :func:`artifact_candidates` searches --
    even on a machine whose ``HOME`` disagrees with its passwd entry,
    which is exactly the environment a scratch-HOME validation run has.
    """

    destination = str(default_bridge_dir())
    if WINDOWS_SHELL:
        return (f'  New-Item -ItemType Directory -Force "{destination}"\n'
                f'  Copy-Item "{built}" "{destination}"')
    return (f'  mkdir -p "{destination}"\n'
            f'  cp "{built}" "{destination}"')


def bridge_remedy(name: str) -> str:
    """The remedy for a missing GRIB bridge, true for THIS install.

    Deliberately does NOT pass ``artifact``, unlike
    :func:`cpu_bridge_remedy` and :meth:`gpuwm.obs.frontdoor.FrontDoor
    .remedy`.  The five GRIB decoders have been in every bundle this
    project has published, so the membership question the parameter asks
    has one answer for them and adding it would change no output -- while
    it WOULD change the arity of the ``prebuilt_bundle_offer`` call, and
    ``gpuwm doctor``'s tests substitute a zero-argument stand-in for that
    function.  Breaking a stand-in to produce identical text is a bad
    trade.  The doctor lane widening that stand-in is the handoff that
    unblocks it; until then this is the historical call, unchanged.
    """

    return artifact_remedy(
        env_var=BRIDGE_ENV[name], filename=executable_name(name),
        subject="the GRIB bridges")


#: The bundled name and environment variable of the parallel CPU
#: preprocessing library.  Declared here beside every other artifact's
#: because :mod:`gpuwm.ingest.cpu_backend` is a numpy-importing module
#: and its resolver's REMEDY has to be composable from this one, which
#: imports nothing but the standard library.
CPU_BRIDGE_ARTIFACT = "gpuwm_preprocess_cpu"
CPU_BRIDGE_ENV = "GPUWM_CPU_PREPROCESS_BRIDGE"


def cpu_bridge_remedy(filename: str) -> str:
    """The remedy for a missing CPU preprocessing library.

    ``ingest/cpu_backend.py``'s resolver was the one refusal in the
    estate whose message never said how to fix it: it listed the paths
    it searched and stopped.  Every path it lists is a rung of the same
    ladder ``gpuwm fetch-bridges`` stages into, and the library IS in
    the bundle -- so the answer existed the whole time and the message
    just never carried it.  Composed from the shared builder so it says
    what this install can actually do, exactly like the GRIB bridges'.
    """

    return artifact_remedy(
        env_var=CPU_BRIDGE_ENV, filename=filename,
        subject="the parallel CPU preprocessing library",
        artifact=CPU_BRIDGE_ARTIFACT)


#: A bridge's out-of-range refusal, as it reaches Python: the field, the
#: decoded value, and the range it left.  Matched only to explain the
#: refusal -- never to suppress it.
_BOUND_REFUSAL = re.compile(
    r"(?P<field>\S+) value (?P<value>\S+) outside \[(?P<range>[^\]]*)\]")


def decode_failure_message(subject: str, stderr: str) -> str:
    """A decoder's refusal, with the reason it gives and what to do next.

    The bridges fail closed and say why on stderr; Python's job is to
    carry that verbatim and add the sentence a user cannot be expected
    to supply -- what the number means, and whether re-running can help.

    An out-of-range refusal earns a specific remedy, because the obvious
    reading of it is wrong.  A field value a hair outside a physical
    bound is GRIB2 packing, not bad data, and the bridge now clamps
    those against the record's own quantization step; so a refusal that
    survives that says the value is further out than the encoding can
    explain.  That points at the bytes, which a re-fetch can fix, rather
    than at the bound, which no one should widen.

    Every remedy line is a command or a ``#`` comment, so the block
    survives being pasted whole.
    """

    detail = stderr.strip()
    message = f"{subject} failed: {detail}" if detail else f"{subject} failed"
    match = _BOUND_REFUSAL.search(detail)
    if match is None:
        return message
    field = match.group("field")
    value = match.group("value")
    bounds = match.group("range")
    return message + (
        "\n  remedy:"
        f"\n  # {field} decoded {value}, outside its physical range "
        f"[{bounds}]."
        "\n  # A value that merely touches a bound is expected -- GRIB2"
        "\n  # packing rounds onto a fixed grid, so a cell encoded AT a"
        "\n  # limit can decode a step past it -- and the bridge already"
        "\n  # clamps those, within that record's own packing step."
        "\n  # This one is further out than the encoding can account for,"
        "\n  # so the suspect is the source record, not the range."
        "\n  # Re-fetch the cycle and re-run; a truncated or corrupted"
        "\n  # download is the usual cause and is not detectable any"
        "\n  # earlier than here.  If it survives a clean re-fetch, the"
        "\n  # published record is bad and the refusal is correct.")


__all__ = [
    "decode_failure_message",
    "SOURCE_DECODERS", "resolve_source_decoder", "DecoderContractError",
    "launchable", "native_executable_format", "quiet_loader_errors",
    "BRIDGE_ABI_MARKERS", "bridge_abi_matches",
    "BRIDGE_ENV", "CARGO_BUILD_HINT", "CLONE_DIR", "CRATE_RELATIVE",
    "CPU_BRIDGE_ARTIFACT", "CPU_BRIDGE_ENV", "cpu_bridge_remedy",
    "REPOSITORY_URL", "RUSTWX_CRATE_RELATIVE", "WINDOWS_SHELL",
    "artifact_candidates", "cargo_build_one_liner",
    "artifact_remedy", "bridge_candidates", "bridge_remedy",
    "BridgeBuildError", "CARGO_FAILURE_CLASSES", "cargo_build_refusal",
    "cargo_missing_refusal", "classify_cargo_failure",
    "build_from_clone_hint", "cargo_activation_command",
    "cargo_is_installed", "crate_dir", "default_bridge_dir",
    "executable_name", "find_artifact", "find_bridge",
    "install_aware_build_hint", "install_into_default_bridge_dir",
    "rust_toolchain_install_command",
    "install_aware_one_line_hint",
    "prebuilt_bundle_offer",
    "sources_present",
]

"""Platform tagging for the wheel, decided by what the wheel CARRIES.

Everything about this distribution is declared in ``pyproject.toml``.
This file exists for one thing setuptools cannot express declaratively:
a wheel that ships prebuilt Rust artifacts is not installable on every
platform, and must not claim to be.

The rule is mechanical, and it reads the tree rather than a flag:

* ``gpuwm/libexec/bridges`` is absent -> the wheel is pure Python and is
  tagged ``py3-none-any``, exactly as every release through 2.3.3 was.
  This is still published, and it is what pip resolves on a platform no
  bundle exists for (macOS, aarch64).  On it, every Rust door refuses BY
  NAME with a build-from-source remedy; nothing degrades into Python.
* ``gpuwm/libexec/bridges`` is present -> the wheel carries native code
  for exactly one platform and is tagged ``py3-none-<platform>``
  (``py3-none-win_amd64``, ``py3-none-manylinux_2_28_x86_64``).  The
  platform comes from the staged ``BUNDLE.json``, never from the machine
  running the build, so a cross-staged bundle cannot be mis-tagged as
  the builder's own platform.

The ABI tag stays ``none`` and the Python tag stays ``py3`` on purpose.
The artifacts are executables and cdylibs launched or ``ctypes``-loaded
by path -- they are not CPython extension modules, they link no libpython
and they see no ``Py_`` symbol.  Letting bdist_wheel derive a
``cp313-cp313`` tag from ``has_ext_modules`` would multiply the release
matrix by every supported Python minor to describe a coupling that does
not exist.

``tools/stage_wheel_bridges.py`` is what creates that directory; see its
docstring for what is staged and what deliberately is not.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.sdist import sdist as _Sdist
from setuptools.dist import Distribution

try:  # setuptools >= 70.1 vendors bdist_wheel; older trees use the wheel package
    from setuptools.command.bdist_wheel import bdist_wheel as _BdistWheel
except ImportError:  # pragma: no cover - older setuptools
    from wheel.bdist_wheel import bdist_wheel as _BdistWheel

_HERE = Path(__file__).resolve().parent
_STAGED = _HERE / "gpuwm" / "libexec" / "bridges"
_MANIFEST = _STAGED / "BUNDLE.json"

#: Platform key (gpuwm's spelling) -> the wheel platform tag it installs on.
#:
#: manylinux_2_28 rather than a plain ``linux_x86_64``: PyPI rejects the
#: bare tag outright, and 2.28 is the glibc floor the release workflow's
#: ubuntu-24.04 runner builds against.
PLATFORM_TAGS = {
    "win-x86_64": "win_amd64",
    "linux-x86_64": "manylinux_2_28_x86_64",
}


#: The packaged bridge pins document of the tree being built.  Read
#: directly rather than through :mod:`gpuwm.bridge_assets`, because
#: setup.py must not import the package it is packaging.
_PINS_PATH = _HERE / "gpuwm" / "data" / "bridges" / "bridge-pins.json"

#: Explicit opt-in for building a wheel from an UNPINNED tree.  Dev and
#: test fixtures set it (their wheels never leave a temp directory); a
#: release must not, and a wheel built under it is unpublishable by
#: construction.
_UNPINNED_OK_ENV = "GPUWM_ALLOW_UNPINNED_WHEEL"


def _refuse_unpinned_wheel() -> None:
    """No wheel is built while the packaged bridge pins pin nothing.

    The breakage this prevents happened, on the 2.5.0 candidate:
    ``tools/build_bridge_bundle.py pin`` was skipped before the wheel
    build, the wheel shipped ``release: null, platforms: {}``, and on a
    clean home ``pip install gpuwm && gpuwm setup`` reported FAILED
    bridges -- no GRIB decoder, no NetCDF decoder, renderer
    ``rw_wrfbatch`` not built -- while every check on the builder's own
    box passed, because its bridges were staged long ago.  The pin step
    existed; nothing made skipping it fail.  This does.

    Only the two facts the skip erases are checked -- a named release
    and at least one pinned platform.  Deep validation belongs to
    ``gpuwm.bridge_assets.parse_pins``, which the pin tool already runs
    on every document it writes.
    """

    if os.environ.get(_UNPINNED_OK_ENV) == "1":
        print(f"setup.py: WARNING: {_UNPINNED_OK_ENV}=1 -- building a "
              "wheel from an unpinned tree.  It must never be published: "
              "on a clean home `gpuwm setup` cannot stage the bridges "
              "from it.", file=sys.stderr)
        return
    try:
        payload = json.loads(_PINS_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"setup.py: REFUSED: the packaged bridge pins document "
            f"{_PINS_PATH.relative_to(_HERE)} is unreadable ({error}); a "
            "wheel built without it installs on a clean home as FAILED "
            "bridges.  Restore gpuwm/data/bridges/bridge-pins.json, run "
            "tools/build_bridge_bundle.py pin, then build.")
    release = payload.get("release") if isinstance(payload, dict) else None
    platforms = payload.get("platforms") if isinstance(payload, dict) else None
    if release and isinstance(platforms, dict) and platforms:
        return
    raise SystemExit(
        "setup.py: REFUSED: gpuwm/data/bridges/bridge-pins.json declares "
        "no release and no platforms, and a wheel built from an unpinned "
        "tree is the 2.5.0 release blocker verbatim: on a clean home, "
        "`pip install gpuwm && gpuwm setup` reports FAILED bridges -- no "
        "GRIB decoder, no NetCDF decoder, renderer rw_wrfbatch not built "
        "-- while every check on the builder's own box passes because its "
        "bridges were staged long ago.  Run the pin step first:\n"
        "  python tools/build_bridge_bundle.py pin --release <tag> "
        "--source-rev <commit> --bundle <each bundle from `pack`>\n"
        "(RELEASE_CHECKLIST.md, 'Per-cut: the assets a wheel's pins point "
        f"at').  For a local dev/test wheel that will never be published, "
        f"set {_UNPINNED_OK_ENV}=1.")


def _refuse_staged_sdist() -> None:
    """No source distribution is built while a platform's bytes are staged.

    The breakage this prevents was measured on the 2.5.0 candidate, on
    Linux: ``tools/stage_wheel_bridges.py --platform linux-x86_64``
    followed by a bare ``python -m build`` swept every staged native ELF
    artifact into the SOURCE distribution -- the 18 the declaration
    carried at that commit, 20 extra members under
    ``gpuwm-2.5.0/gpuwm/libexec/bridges/``, the tarball up from
    95,236,252 B to 114,906,421 B, and Linux executables shipping as
    source.  The published shape escaped only because
    ``.github/workflows/publish.yml`` runs ``--clean`` immediately
    before ``python -m build``; nothing made the wrong build fail.

    ``MANIFEST.in`` prunes the directory, so the tarball's shape is
    right however it is built.  That alone would be a quieter defect,
    not a fix: ``python -m build`` builds its wheel FROM the sdist it
    just wrote, so a pruned sdist yields a tree with nothing staged and
    a ``py3-none-any`` wheel -- while the operator was staging for a
    platform one, and the wheel says nothing about it.  So the sdist
    step refuses instead, and names both ways out, because which is
    right depends on which distribution is wanted.

    There is deliberately no override.  Neither published shape needs
    one: the pure pair cleans first, and the platform wheel is built
    with ``--wheel``.
    """

    platform = _staged_platform()
    if platform is None:
        return
    raise SystemExit(
        "setup.py: REFUSED: gpuwm/libexec/bridges is staged with the "
        f"{platform} artifacts, and an sdist built now carries them: a "
        "SOURCE distribution shipping one platform's native binaries "
        "(measured on the 2.5.0 candidate at +19,670,169 B, 20 extra "
        "members, ELF executables inside gpuwm-<version>.tar.gz).\n"
        "Pick the distribution you actually want:\n"
        "  the pure pair, which is what publish.yml ships:\n"
        "    python tools/stage_wheel_bridges.py --clean\n"
        "    python -m build\n"
        "  the platform wheel, which needs the staged bytes and no "
        "sdist:\n"
        "    python -m build --wheel\n"
        "(`python -m build` builds its wheel from the sdist, so it can "
        "never produce a platform wheel; that is why this is a refusal "
        "and not a silent prune.)")


def _staged_platform() -> str | None:
    """The platform the staged artifacts are for, or None if none staged."""

    if not _MANIFEST.is_file():
        return None
    try:
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"{_MANIFEST} is unreadable ({error}); a staged bridge directory "
            "with no manifest cannot be tagged, because nothing says which "
            "platform its bytes are for.  Re-run tools/stage_wheel_bridges.py "
            "or pass --clean to build the pure wheel.") from None
    platform = manifest.get("platform")
    if platform not in PLATFORM_TAGS:
        raise SystemExit(
            f"{_MANIFEST} names platform {platform!r}, which has no wheel tag "
            f"in setup.py PLATFORM_TAGS; known: {sorted(PLATFORM_TAGS)}")
    return platform


class _BinaryDistribution(Distribution):
    """Declares the distribution impure when it carries staged artifacts."""

    def has_ext_modules(self) -> bool:  # noqa: D102 - setuptools protocol
        return _staged_platform() is not None

    def is_pure(self) -> bool:  # noqa: D102 - setuptools protocol
        return _staged_platform() is None


#: Unix mode the staged executables and shared libraries must carry
#: inside the wheel.
_EXECUTABLE_MODE = 0o755
_DEFAULT_MODE = 0o644

#: Archive prefix of the staged artifacts, as it appears in the wheel.
_STAGED_PREFIX = "gpuwm/libexec/bridges/"


def _force_executable_bits(wheel_path: Path) -> int:
    """Stamp 0755 on every staged artifact inside a built wheel.

    A wheel records each member's unix mode in the zip's
    ``external_attr``, and ``bdist_wheel`` copies it from the file on
    disk.  On Windows there is no executable bit to copy, so a
    manylinux wheel cross-built there ships its binaries 0644 -- present,
    correct, and unrunnable.  Measured, not theorised: installing that
    wheel into a clean venv on a real Linux userland put all eleven
    artifacts in place and then died with ``PermissionError: [Errno 13]
    Permission denied`` on the first one executed.

    Forcing it here rather than trusting the build host is deliberate.
    The mode is a property of what the file IS -- a program -- not of
    the machine that packed it, and a release must not depend on having
    been assembled on a platform that happens to preserve it.

    Rewrites the archive, which RECORD tolerates: RECORD hashes member
    CONTENT, and no content changes.
    """

    import shutil
    import tempfile
    import zipfile

    stamped = 0
    with tempfile.TemporaryDirectory() as work:
        rebuilt = Path(work) / wheel_path.name
        with zipfile.ZipFile(wheel_path) as source:
            with zipfile.ZipFile(rebuilt, "w", zipfile.ZIP_DEFLATED) as target:
                for info in source.infolist():
                    payload = source.read(info.filename)
                    updated = zipfile.ZipInfo(info.filename, info.date_time)
                    updated.compress_type = info.compress_type
                    updated.external_attr = info.external_attr
                    if (info.filename.startswith(_STAGED_PREFIX)
                            and not info.filename.endswith("/")
                            and not info.filename.endswith("BUNDLE.json")):
                        updated.external_attr = (_EXECUTABLE_MODE << 16)
                        stamped += 1
                    elif not info.external_attr >> 16:
                        updated.external_attr = (_DEFAULT_MODE << 16)
                    target.writestr(updated, payload)
        shutil.move(str(rebuilt), str(wheel_path))
    return stamped


class _PlatformWheel(_BdistWheel):
    """Tags the wheel ``py3-none-<platform>`` when artifacts are staged."""

    def finalize_options(self) -> None:
        # Before anything is copied or tagged: a wheel from an unpinned
        # tree must not exist at all.  Every build route -- python -m
        # build, pip wheel, pip install <tree>, setup.py bdist_wheel --
        # passes through this command's finalize_options.
        _refuse_unpinned_wheel()
        platform = _staged_platform()
        if platform is not None:
            self.plat_name = PLATFORM_TAGS[platform]
            self.plat_name_supplied = True
        super().finalize_options()
        self.root_is_pure = platform is None

    def get_tag(self) -> tuple[str, str, str]:
        python_tag, abi_tag, plat_tag = super().get_tag()
        if _staged_platform() is None:
            return python_tag, abi_tag, plat_tag
        # Keep py3/none: these are subprocess binaries and ctypes cdylibs,
        # not CPython extension modules.  See the module docstring.
        return "py3", "none", plat_tag

    def run(self) -> None:
        super().run()
        if _staged_platform() is None:
            return
        for _command, _version, path in self.distribution.dist_files:
            wheel_path = Path(path)
            if wheel_path.suffix != ".whl":
                continue
            stamped = _force_executable_bits(wheel_path)
            self.announce(
                f"stamped 0{_EXECUTABLE_MODE:o} on {stamped} staged "
                f"artifact(s) in {wheel_path.name}", level=2)


class _SourceDistribution(_Sdist):
    """Refuses to pack a source distribution out of a staged tree."""

    def finalize_options(self) -> None:
        # Before a file list is built: every route that makes an sdist
        # -- python -m build, pip install <tree> (via build_sdist),
        # python setup.py sdist -- passes through this command's
        # finalize_options, the same place the wheel's pin gate fires.
        _refuse_staged_sdist()
        super().finalize_options()


setup(distclass=_BinaryDistribution,
      cmdclass={"bdist_wheel": _PlatformWheel, "sdist": _SourceDistribution})

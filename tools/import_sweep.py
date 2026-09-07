"""Import every module the wheel ships, and refuse an undeclared absence.

    python -m tools.import_sweep

THE BREAKAGE THIS PREVENTS
--------------------------
xc-06-01 of the 2026-09 audit measured what this repository's automated gates
actually execute.  `.github/workflows/` held one file, `publish.yml`, whose
`test` job names 17 test files by hand -- 452 of the tree's 14,711 test
functions, 3.1%.  Resolving every `gpuwm.*` import in those 17 files reaches
51 of 525 package modules directly; the module-scope transitive closure
reaches 221 modules, 209,243 lines, 48.3% of the package.  The other half --
`gpuwm/da/`, most of `gpuwm/core/`, the whole verify tree -- was not IMPORTED
by anything CI ran.  A SyntaxError, a bad module-scope constant or a broken
internal import there produced a green check on the pull request.

This sweep is the cheapest complete answer to that half of the finding.  It
imports every module of the packages the wheel ships, in seconds, with no GPU,
no fixtures and no network, and it fails on anything that is not a DECLARED
optional dependency.  It checks module initialization; it does not execute every function body.

WHAT COUNTS AS A PASS
---------------------
A module that imports.  A module that raises `ModuleNotFoundError` naming one
of `OPTIONAL_MODULES` is tolerated and REPORTED BY NAME, because those are the
extras a CPU runner deliberately does not install.

Everything else fails, including `ModuleNotFoundError` for a `gpuwm.*` or
`tilestream.*` name: that is a broken internal import, not a missing extra,
and the distinction is the reason this tool does not simply swallow
`ImportError` the way an opportunistic walk would.  Measured at a70ade37 on a
CPU box with the base dependencies installed: 524 `gpuwm` modules and 131
`tilestream` modules walked, 0 failures, and the only absent names were
`cupy` (29 modules once the five programs below are set aside) and `netCDF4`
-- and netCDF4 is a BASE dependency, so on a runner that ran
`pip install -e .` cupy is the only legitimate absence.

A module that calls `sys.exit()` at import scope is a FAILURE, named like any
other.  It used to be the end of the sweep: the walk caught `Exception`, and
`SystemExit` is not one, so the whole gate exited with that module's status
and printed no report at all.  On a box with a visible CUDA device that is
what `python -m tools.import_sweep` did -- exit 2, no verdict -- because
`tilestream/attack_real12gb.py` refuses at import scope when the card is not
idle.  A gate a module can terminate says nothing about the modules after it.

WHY NOT `tools`, AND WHY NOT THE FIVE PROGRAMS UNDER `tilestream`
-----------------------------------------------------------------
`tools/` ships, but several of its top-level modules print a report at import
(`tools/perf_survey.py` and neighbours write their tables to stdout) and some
open files beside them.  Importing them is a side effect, not a check, so the
sweep covers the two packages whose modules are libraries.

`tilestream/` carries five modules of the same shape -- the four `attack_*`
one-off structural attacks and `digest_dump`, whose module bodies ARE the
program: they read the device, spawn `tilestream.vram_probe` subprocesses,
print their tables and, in `digest_dump`'s case, open `sys.argv[1]`.  They are
named in `SCRIPT_MODULES`, walked, reported, and not imported.
"""

from __future__ import annotations

import argparse
import importlib
import pkgutil
import os
import sys
import traceback

#: Third-party names a shipped module may fail to import without failing the
#: sweep, each with the extra that provides it.  An absence NOT on this list
#: is a failure -- that is the whole point, and it is why this is a table and
#: not an `except ImportError: continue`.
OPTIONAL_MODULES: dict[str, str] = {
    # The CUDA stack.  One wheel per CUDA major and no environment marker can
    # choose between them, so a CPU runner has neither.
    "cupy": "gpuwm[gpu-cu12] / gpuwm[gpu-cu13]",
    "cupyx": "gpuwm[gpu-cu12] / gpuwm[gpu-cu13]",
    # Rendering, terrain and the agent surface: extras by declaration.
    "rasterio": "gpuwm[geog]",
    "pyproj": "gpuwm[geog]",
    "affine": "gpuwm[geog]",
    "mcp": "gpuwm[mcp]",
    "huggingface_hub": "gpuwm[publish]",
    "h5py": "gpuwm[publish]",
    # The Rust extension modules, which are built rather than pip-installed.
    "rw_odim": "cargo build --release (tools/rustwx)",
    "rw_wps": "cargo build --release (tools/rw_wps)",
}

#: Modules the sweep ENUMERATES BUT DOES NOT IMPORT, each with what its
#: module body does instead of defining names.  These are programs: importing
#: `tilestream.attack_real12gb` spawns four `tilestream.vram_probe` trials on
#: the device, `tilestream.attack_timing` runs a two-arm benchmark, and
#: `tilestream.digest_dump` opens `sys.argv[1]` after a full physics reference
#: run.  That is the objection the WHY NOT `tools` paragraph above makes,
#: inside a package the sweep does cover, so it is answered the same way it is
#: there and the way `OPTIONAL_MODULES` answers a missing extra: by name, with
#: the reason, in a table -- never by widening an `except`.
#:
#: The exclusion costs no transitive coverage, because nothing else the two
#: packages ship imports any of them; that is gated by
#: tests/test_import_sweep.py::test_no_shipped_module_imports_a_declared_script.
#: The `syntax` lane's `compileall` still parses every one of them.
SCRIPT_MODULES: dict[str, str] = {
    "tilestream.attack_instrument": "runs the overlap-instrument attack",
    "tilestream.attack_intersect": "walks the carrier/arena intersection",
    "tilestream.attack_real12gb": "spawns vram_probe trials on the device",
    "tilestream.attack_timing": "runs the two-arm timing benchmark",
    "tilestream.digest_dump": "writes sys.argv[1] after a reference run",
}

#: The packages the wheel ships that are libraries rather than scripts.
PACKAGES = ("gpuwm", "tilestream")


class Failure:
    """One module that did not import, and why."""

    def __init__(self, module: str, error: BaseException) -> None:
        self.module = module
        self.error = error
        self.detail = "".join(traceback.format_exception_only(
            type(error), error)).strip()

    def __str__(self) -> str:
        return f"{self.module}: {self.detail}"


def tolerated(error: BaseException) -> str | None:
    """The extra that would supply this absence, or ``None`` if it is a fault.

    Only a `ModuleNotFoundError` naming a declared optional TOP-LEVEL module
    is tolerated.  `ImportError` is not: a module that imports and then fails
    to find a name inside it is a broken import, and a `ModuleNotFoundError`
    for a submodule of this project is a broken import wearing the same
    exception class.
    """

    if not isinstance(error, ModuleNotFoundError) or not error.name:
        return None
    return OPTIONAL_MODULES.get(error.name)


#: What one module's import is allowed to do to the sweep.  `SystemExit` is
#: here because a module that calls `sys.exit()` at import scope is a module
#: that does not import, and the sweep must be able to SAY so -- before this,
#: it inherited the exit status and printed nothing.  `KeyboardInterrupt` is
#: deliberately NOT here: Ctrl-C must still stop the walk rather than be
#: recorded as one more broken module, which is why this is a named pair and
#: not `BaseException`.
CAUGHT = (Exception, SystemExit)


def _module_infos(paths, *, prefix):
    paths = list(paths)
    for info in pkgutil.iter_modules(paths, prefix=prefix):
        yield info
        if info.ispkg:
            leaf = info.name.rsplit(".", 1)[-1]
            yield from _module_infos([os.path.join(path, leaf) for path in paths],
                                     prefix=info.name + ".")


def sweep(packages=PACKAGES):
    """``(walked, absences, failures)`` over every module in ``packages``.

    ``absences`` is ``{optional module: [importer, ...]}`` and is reported,
    never hidden.  ``failures`` is a list of :class:`Failure`.  A module named
    in :data:`SCRIPT_MODULES` is walked and left unimported; :func:`main`
    names it in the report.
    """

    walked: list[str] = []
    absences: dict[str, list[str]] = {}
    failures: list[Failure] = []
    for name in packages:
        try:
            package = importlib.import_module(name)
        except CAUGHT as error:                      # the package itself
            failures.append(Failure(name, error))
            continue
        walked.append(name)
        # Enumerate without importing package __init__ files: walk_packages
        # executes them outside the protected import below, so SystemExit
        # from a nested package could otherwise terminate the whole report.
        for info in _module_infos(package.__path__, prefix=f"{name}."):
            walked.append(info.name)
            if info.name in SCRIPT_MODULES:
                continue
            try:
                importlib.import_module(info.name)
            except CAUGHT as error:
                extra = tolerated(error)
                if extra is None:
                    failures.append(Failure(info.name, error))
                else:
                    absences.setdefault(error.name.split(".")[0],
                                        []).append(info.name)
    return walked, absences, failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("packages", nargs="*", default=None,
                        help=f"packages to sweep (default: {' '.join(PACKAGES)})")
    args = parser.parse_args(argv)
    packages = tuple(args.packages) if args.packages else PACKAGES

    walked, absences, failures = sweep(packages)
    print(f"import sweep: {len(walked)} modules across {', '.join(packages)}")
    for name in sorted(absences):
        print(f"  absent, declared optional: {name} "
              f"({len(absences[name])} modules; {OPTIONAL_MODULES[name]})")
    for name in sorted(set(walked) & set(SCRIPT_MODULES)):
        print(f"  not imported, the module body is the program: {name} "
              f"({SCRIPT_MODULES[name]})")
    if not failures:
        print("import sweep: OK")
        return 0
    print()
    print(f"IMPORT SWEEP FAILED -- {len(failures)} module(s) the wheel ships "
          "do not import:")
    for failure in failures:
        print(f"  {failure}")
    print("An absence that belongs to an extra goes in "
          "tools/import_sweep.py::OPTIONAL_MODULES with the extra's name.  "
          "Anything else is a module this distribution ships broken.")
    return 1


if __name__ == "__main__":            # pragma: no cover
    sys.exit(main())

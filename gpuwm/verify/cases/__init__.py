"""Verification-case registry: a case joins the CLI as data, not as a diff.

Every other dispatch table in this codebase is already registry-driven --
physics through :mod:`gpuwm.physics_registry` over
``physics_registry_v2.json``, forcing sources through the
``gpuwm.source_adapters`` adapter table, nest gates through
``gpuwm.verify.nest_gates.NEST_GATES``.  Cases were the exception: the
driver named each one in a literal dict, so adding a case meant editing
generic source, and the case name leaked into a generic module.  This
module closes that gap.

Discovery
---------
A case is a module in this package (or in a directory named by the
``GPUWM_CASE_PATH`` environment variable) whose module name does not start
with ``_``.  Its *capabilities* -- which entry points the driver may use --
are read from the module's top-level bindings:

``verify``
    ``GATES`` plus ``run(outdir) -> metrics``.  Driven by ``gpuwm verify
    NAME``; ``GATES`` maps metric -> ``(lo, hi)`` and the returned metrics
    dict carries ``nan`` plus every gate key.
``config``
    ``write_static(cfg, output)``, ``write_ingest(cfg, output)`` and
    ``run_config(cfg, outdir, restart=)``.  Driven by ``gpuwm
    static|ingest|run CONFIG`` on a legacy :class:`~gpuwm.config.RunConfig`
    TOML whose ``[run].case`` names the module.
``script``
    ``main(argv) -> int``, i.e. ``python -m gpuwm.verify.cases.NAME``.
    Recorded so a front end can enumerate these runners; the ``gpuwm``
    driver does not dispatch them.

A module declaring none of the three (a shared helper such as
``nest_ideal_common``) is not a case and is skipped -- no naming convention
and no opt-in marker is required.

The capability scan is a *static* AST read of the module's top-level
bindings, so enumerating the cases neither imports nor executes any of
them.  That matters twice over: building the CLI's ``--help`` must not pull
a multi-thousand-line real-data case (and its netCDF/cupy dependencies)
into the process, and a GUI wants the same list without running model code.
:func:`manifest` returns exactly that list in JSON-serializable form.  A
module is imported only when it is actually selected, by :func:`load`.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import os
import sys
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

#: ``os.pathsep``-separated directories searched for case modules in
#: addition to this package.  Lets a deployment (or a test) add a case
#: without writing into an installed package.
CASE_PATH_ENV = "GPUWM_CASE_PATH"

#: Capability ids.  Kept as module constants so no caller spells the
#: strings, and so a front end can enumerate them.
VERIFY = "verify"
CONFIG = "config"
SCRIPT = "script"

#: capability -> the top-level names a module must bind to declare it.
CAPABILITY_CONTRACTS: dict[str, tuple[str, ...]] = {
    VERIFY: ("GATES", "run"),
    CONFIG: ("write_static", "write_ingest", "run_config"),
    SCRIPT: ("main",),
}

_PACKAGE_DIR = Path(__file__).resolve().parent

# path -> (stat signature, top-level names).  Parsing 15 modules is
# milliseconds, but discovery runs per CLI invocation and per test, so the
# parse is memoized on (size, mtime_ns) and re-read whenever a file changes.
_BINDINGS_CACHE: dict[Path, tuple[tuple[int, int], frozenset[str]]] = {}


def _top_level_names(path: Path) -> frozenset[str]:
    """Names bound at module scope, without importing the module."""
    try:
        stat = path.stat()
    except OSError:
        return frozenset()
    signature = (stat.st_size, stat.st_mtime_ns)
    cached = _BINDINGS_CACHE.get(path)
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return frozenset()
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                found.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                found.add(alias.asname or alias.name.split(".")[0])
    frozen = frozenset(found)
    _BINDINGS_CACHE[path] = (signature, frozen)
    return frozen


def capabilities_of(path: Path) -> frozenset[str]:
    """Capabilities a module file declares, by its top-level bindings."""
    bound = _top_level_names(path)
    return frozenset(
        capability
        for capability, required in CAPABILITY_CONTRACTS.items()
        if all(name in bound for name in required))


@dataclass(frozen=True)
class CaseEntry:
    """One discovered case: what it is called and what it can do."""

    name: str
    path: Path
    capabilities: frozenset[str]
    #: True for a module shipped inside this package (imported by dotted
    #: name); False for one found through :data:`CASE_PATH_ENV`.
    packaged: bool

    @property
    def module_name(self) -> str:
        if self.packaged:
            return f"{__name__}.{self.name}"
        return f"{__name__}._external.{self.name}"

    def as_dict(self) -> dict:
        """JSON-serializable record for a front end / manifest."""
        return {
            "name": self.name,
            "module": self.module_name,
            "path": str(self.path),
            "packaged": self.packaged,
            "capabilities": sorted(self.capabilities),
        }


def search_paths() -> tuple[Path, ...]:
    """Directories searched for cases: this package, then CASE_PATH_ENV."""
    roots = [_PACKAGE_DIR]
    for chunk in os.environ.get(CASE_PATH_ENV, "").split(os.pathsep):
        chunk = chunk.strip()
        if not chunk:
            continue
        candidate = Path(chunk).expanduser()
        if candidate not in roots:
            roots.append(candidate)
    return tuple(roots)


def discover() -> dict[str, CaseEntry]:
    """Discovered cases, name -> :class:`CaseEntry`, sorted by name.

    The packaged directory is searched first and wins a name collision, so
    an entry on ``GPUWM_CASE_PATH`` can never shadow a shipped case.
    """
    found: dict[str, CaseEntry] = {}
    for index, root in enumerate(search_paths()):
        try:
            candidates = sorted(root.glob("*.py"))
        except OSError:
            continue
        for path in candidates:
            name = path.stem
            if name.startswith("_") or name in found:
                continue
            capabilities = capabilities_of(path)
            if not capabilities:
                continue
            found[name] = CaseEntry(name=name, path=path,
                                    capabilities=capabilities,
                                    packaged=index == 0)
    return dict(sorted(found.items()))


def names(capability: str | None = None) -> tuple[str, ...]:
    """Discovered case names, optionally filtered to one capability."""
    return tuple(
        name for name, record in discover().items()
        if capability is None or capability in record.capabilities)


def entry(name: str) -> CaseEntry:
    """One discovered case, or a KeyError naming what is available."""
    found = discover()
    try:
        return found[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a discovered verification case; found "
            f"{sorted(found)} under {[str(p) for p in search_paths()]}"
        ) from None


def manifest() -> list[dict]:
    """Every discovered case as a JSON-serializable record."""
    return [record.as_dict() for record in discover().values()]


def load(name: str) -> ModuleType:
    """Import and return one case module (cached by ``sys.modules``)."""
    record = entry(name)
    if record.packaged:
        return importlib.import_module(record.module_name)
    module_name = record.module_name
    existing = sys.modules.get(module_name)
    if existing is not None and getattr(
            existing, "__file__", None) == str(record.path):
        return existing
    spec = importlib.util.spec_from_file_location(module_name, record.path)
    if spec is None or spec.loader is None:
        raise ImportError(f"case module {record.path} is not importable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def require(name: str, capability: str) -> ModuleType:
    """Load ``name`` after checking it declares ``capability``."""
    record = entry(name)
    if capability not in record.capabilities:
        raise ValueError(
            f"case {name!r} does not provide the {capability!r} entry "
            f"points {CAPABILITY_CONTRACTS[capability]}; it declares "
            f"{sorted(record.capabilities)}")
    return load(name)


def gates(name: str) -> dict:
    """The case's own ``GATES`` table -- the single source for its bounds.

    Resolved through the case module, never through a driver-side copy, so
    a stubbed driver runner cannot silently relax a gate.
    """
    return require(name, VERIFY).GATES


class CaseMapping(MutableMapping):
    """``name -> case module`` for one capability, resolved on demand.

    Behaves like the literal dict it replaces (iteration, membership,
    ``[]``, and item assignment for tests that stub a runner) while the
    membership itself comes from discovery and each module is imported only
    when it is indexed.
    """

    __slots__ = ("_capability", "_overrides")

    def __init__(self, capability: str) -> None:
        self._capability = capability
        self._overrides: dict[str, ModuleType] = {}

    def _discovered(self) -> tuple[str, ...]:
        return names(self._capability)

    def __iter__(self):
        ordered = list(self._discovered())
        ordered.extend(n for n in self._overrides if n not in ordered)
        return iter(ordered)

    def __len__(self) -> int:
        return sum(1 for _ in iter(self))

    def __contains__(self, name: object) -> bool:
        # Overridden so a membership test stays a discovery lookup; the
        # Mapping default answers it by indexing, which would import the
        # module just to find out whether it exists.
        return name in self._overrides or name in self._discovered()

    def __getitem__(self, name: str) -> ModuleType:
        override = self._overrides.get(name)
        if override is not None:
            return override
        if name not in self._discovered():
            raise KeyError(name)
        return load(name)

    def __setitem__(self, name: str, module: ModuleType) -> None:
        self._overrides[name] = module

    def __delitem__(self, name: str) -> None:
        del self._overrides[name]

    def __repr__(self) -> str:
        return (f"CaseMapping({self._capability!r}: "
                f"{sorted(self._discovered())})")


class GateMapping(Mapping):
    """``name -> GATES`` for every discovered verify case."""

    __slots__ = ()

    def __iter__(self):
        return iter(names(VERIFY))

    def __len__(self) -> int:
        return len(names(VERIFY))

    def __contains__(self, name: object) -> bool:
        return name in names(VERIFY)

    def __getitem__(self, name: str) -> dict:
        if name not in names(VERIFY):
            raise KeyError(name)
        return gates(name)

    def __repr__(self) -> str:
        return f"GateMapping({sorted(names(VERIFY))})"


__all__ = [
    "CAPABILITY_CONTRACTS", "CASE_PATH_ENV", "CONFIG", "CaseEntry",
    "CaseMapping", "GateMapping", "SCRIPT", "VERIFY", "capabilities_of",
    "discover", "entry", "gates", "load", "manifest", "names", "require",
    "search_paths",
]

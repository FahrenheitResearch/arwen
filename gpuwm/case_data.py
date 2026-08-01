"""Case-data configuration: the ``[case_data]`` TOML table (Phase 5, Task 2).

An experiment TOML describes the *simulation* (grids, clocks, physics --
:mod:`gpuwm.experiment`); this module describes the *data* a real case
consumes and how its outputs are identified.  Every path, schema id, and
policy that used to be a hard-coded constant of the runtime path is a
declared key here: forcing GRIB paths/globs, the Vtable schema, the WPS
namelist used for grid construction (until config-driven Lambert grids
land in Task 5), the GEOG root, an optional declared source-orography
artifact (or the forcing catalog's ``era5_z_invariant`` provider), the
``sfcp_to_sfcp`` surface-pressure policy, the trace-gas choice, and the
output identity (domain id, title).

All validation is fail-loud: unknown keys, missing keys, wrong types,
missing files, and empty globs are hard errors at load.  Paths resolve
relative to the TOML file's directory; entries containing glob
characters expand (sorted) so a case may declare per-time forcing files
with one pattern.

Portable roots
--------------
Every declared path may reference an environment variable as
``${NAME}``, expanded before resolution.  Large case bundles (reference
wrfouts, WPS_GEOG, multi-gigabyte GRIB) live outside the repository and
their location differs per machine, so without this the only way to
declare them is an absolute developer path -- which is both unportable
and a machine path shipped in a source distribution.

One variable, :data:`CASE_DATA_ROOT_ENV`, carries a default
(:func:`default_case_data_root`) so the committed configs resolve
unchanged on a machine that sets nothing; every other name must be set
or the load fails naming the variable and the key.  Point the variable
somewhere else and the same config reads a bundle copied anywhere::

    GPUWM_CASE_DATA_ROOT=/data/bundles gpuwm check configs/<case>.toml

The Task-3 preflight (``gpuwm check``) later builds a hashed
:class:`InputCatalog` on top of these declarations; this module's
:class:`ResolvedInput` records are the minimal provenance surface the
Task-2 runtime prints and the catalog consumes.
"""

from __future__ import annotations

import glob as _glob
import math
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from gpuwm.experiment import ExperimentConfig, build_experiment
from gpuwm.explain import warn

_GLOB_CHARS = frozenset("*?[")

#: Environment variable naming the directory that holds case bundles.
#: This is the one path variable with a default, so a committed config
#: that references it resolves on a machine that sets nothing.
CASE_DATA_ROOT_ENV = "GPUWM_CASE_DATA_ROOT"

_PATH_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def default_case_data_root() -> Path:
    """Where case bundles live when CASE_DATA_ROOT_ENV is unset.

    The download directory -- the same fallback the case modules already
    use for their own reference-data environment variables, so nothing
    about resolution changes for a developer who sets no variable.
    """
    return Path.home() / "Downloads"


#: Path variables that resolve without being set.  Everything else must be
#: exported: a silent default for an unknown name would reintroduce
#: exactly the "works on one machine" failure this mechanism removes.
_PATH_VARIABLE_DEFAULTS = {CASE_DATA_ROOT_ENV: default_case_data_root}


def case_data_root() -> Path:
    """The effective case-bundle root: the variable, else the default.

    Callers that need to locate a bundle outside the ``[case_data]``
    loader (tooling, test fixtures) should ask here rather than spell a
    machine path, so everything relocates with the one variable.
    """
    raw = os.environ.get(CASE_DATA_ROOT_ENV)
    if raw is not None and raw.strip():
        return Path(raw.strip())
    return default_case_data_root()


def expand_path_variables(value: str, key: str, source: str) -> str:
    """Expand ``${NAME}`` references in a declared path (fail-loud)."""
    def replace(match: re.Match) -> str:
        name = match.group(1)
        raw = os.environ.get(name)
        if raw is not None and raw.strip():
            return str(Path(raw.strip()))
        fallback = _PATH_VARIABLE_DEFAULTS.get(name)
        # Unset with no registered default is a configuration error, not a
        # path that quietly becomes "${NAME}" on disk.
        if fallback is None:
            raise ValueError(
                f"{key} in [case_data] of {source} references ${{{name}}}, "
                f"which is not set in the environment.  Export {name} to "
                "the directory holding this case's data (only "
                f"{sorted(_PATH_VARIABLE_DEFAULTS)} resolve without being "
                "set).")
        return str(fallback())
    return _PATH_VARIABLE.sub(replace, value)

_REQUIRED_KEYS = (
    "forcing", "vtable", "wps_namelist", "geog_root",
    "sfcp_to_sfcp", "output_title",
)
_OPTIONAL_KEYS = (
    "forcing_interval_s", "output_domain", "source_orography",
    "source_orography_variable", "co2_vmr",
)
_KNOWN_KEYS = frozenset(_REQUIRED_KEYS) | frozenset(_OPTIONAL_KEYS)
_DOMAIN_SOURCE_KEY = re.compile(r"d([0-9]{2})")


@dataclass(frozen=True)
class SourceOrography:
    """Declared source-orography artifact: a NetCDF file + variable name.

    When absent, Task 4's ``era5_z_invariant`` provider resolves source
    orography from the forcing catalog's invariant geopotential.  The two
    sources are mutually exclusive and a conflict is a hard error.
    """

    path: Path
    variable: str


@dataclass(frozen=True)
class PerDomainSourceOrography:
    """Grid-specific source-orography artifacts keyed by WRF domain id.

    ``path`` and ``variable`` retain the historical d01-facing surface for
    root-only consumers.  Child initialization must call :meth:`for_domain`
    (normally through :meth:`CaseDataConfig.source_orography_for_domain`) so
    a missing grid-specific declaration cannot silently reuse d01.
    """

    by_domain: tuple[tuple[int, SourceOrography], ...]

    def for_domain(self, domain_id: int) -> SourceOrography:
        """Resolve one declared artifact, failing loudly when it is absent."""
        _validate_domain_id(domain_id)
        for declared_id, artifact in self.by_domain:
            if declared_id == domain_id:
                return artifact
        raise _missing_source_orography(domain_id)

    @property
    def path(self) -> Path:
        """The d01 path, preserving root-only catalog/preflight behavior."""
        return self.for_domain(1).path

    @property
    def variable(self) -> str:
        """The common variable name, preserving the historical surface."""
        return self.for_domain(1).variable


SourceOrographyDeclaration = SourceOrography | PerDomainSourceOrography


def _validate_domain_id(domain_id: int) -> None:
    if (isinstance(domain_id, bool) or not isinstance(domain_id, int)
            or domain_id < 1):
        raise ValueError(
            "source-orography domain id must be a positive integer, got "
            f"{domain_id!r}")


def _missing_source_orography(domain_id: int) -> ValueError:
    label = f"d{domain_id:02d}"
    return ValueError(
        "grid-specific source_orography declaration has no entry for "
        f"{label}; declare source_orography.{label} or use the "
        "grid-independent era5_z_invariant provider")


def resolve_source_orography(
        declaration: SourceOrographyDeclaration | None,
        domain_id: int) -> SourceOrography | None:
    """Resolve a declaration for one grid, or select the invariant provider.

    ``None`` means the forcing catalog's grid-independent
    ``era5_z_invariant`` provider and is therefore valid for every domain.  A
    legacy single artifact belongs to d01 only; declared artifacts are
    already remapped to a particular WRF grid and must never be reused by a
    child.
    """
    _validate_domain_id(domain_id)
    if declaration is None:
        return None
    if isinstance(declaration, PerDomainSourceOrography):
        return declaration.for_domain(domain_id)
    if domain_id != 1:
        raise _missing_source_orography(domain_id)
    return declaration


@dataclass(frozen=True)
class ResolvedInput:
    """One resolved input artifact for provenance reporting.

    ``role`` names the slot the artifact fills (``forcing``, ``vtable``,
    ``wps_namelist``, ``geog_root``, ``source_orography``); ``path`` is
    the resolved location; ``detail`` carries slot-specific context (the
    originating glob pattern, the declared variable name).
    """

    role: str
    path: Path
    detail: str = ""


@dataclass(frozen=True, kw_only=True)
class CaseDataConfig:
    """Validated ``[case_data]``: every input and policy, declared.

    ``forcing`` is the resolved (existing, sorted-within-pattern) GRIB
    file tuple.  ``forcing_interval_s`` is the declared forcing-interval
    policy: a positive interval validated against the decoded snapshot
    times, or ``None`` to accept whatever uniform-or-not schedule
    discovery finds.  ``co2_vmr`` is an optional declared trace-gas override
    (mole fraction), applied through the radiation scheme's trace-gas policy
    hook; omission engages Task 4's date-indexed selection.
    ``output_domain``/``output_title`` are the output identity: files
    are named ``wrfout_d{output_domain:02d}_<date>`` with ``TITLE`` set
    to ``output_title``.
    """

    forcing: tuple[Path, ...]
    vtable: Path
    wps_namelist: Path
    geog_root: Path
    source_orography: SourceOrographyDeclaration | None
    sfcp_to_sfcp: bool
    co2_vmr: float | None
    output_title: str
    output_domain: int = 1
    forcing_interval_s: float | None = None

    def source_orography_for_domain(
            self, domain_id: int) -> SourceOrography | None:
        """Resolve the artifact for ``domain_id`` or select the provider."""
        return resolve_source_orography(self.source_orography, domain_id)

    def resolved_inputs(self) -> tuple[ResolvedInput, ...]:
        """Every declared input artifact, for provenance reporting."""
        records = [ResolvedInput(role="forcing", path=path)
                   for path in self.forcing]
        records.append(ResolvedInput(role="vtable", path=self.vtable))
        records.append(ResolvedInput(role="wps_namelist",
                                     path=self.wps_namelist))
        records.append(ResolvedInput(role="geog_root", path=self.geog_root))
        if isinstance(self.source_orography, PerDomainSourceOrography):
            for domain_id, artifact in self.source_orography.by_domain:
                records.append(ResolvedInput(
                    role="source_orography", path=artifact.path,
                    detail=(f"variable={artifact.variable};"
                            f"domain=d{domain_id:02d}")))
        elif self.source_orography is not None:
            records.append(ResolvedInput(
                role="source_orography", path=self.source_orography.path,
                detail=f"variable={self.source_orography.variable}"))
        return tuple(records)


def _resolve_path(base_dir: Path, value, key: str, source: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{key} in [case_data] of {source} must be a non-empty path "
            f"string, got {value!r}.")
    path = Path(expand_path_variables(value, key, source))
    if not path.is_absolute():
        path = base_dir / path
    return path


def _resolve_forcing(base_dir: Path, value, source: str) -> tuple[Path, ...]:
    entries = value if isinstance(value, list) else [value]
    if not entries:
        raise ValueError(
            f"forcing in [case_data] of {source} must declare at least one "
            "GRIB path or glob pattern.")
    resolved: list[Path] = []
    for entry in entries:
        path = _resolve_path(base_dir, entry, "forcing", source)
        if _GLOB_CHARS & set(str(entry)):
            matches = sorted(Path(hit) for hit in _glob.glob(str(path)))
            if not matches:
                raise ValueError(
                    f"forcing glob {entry!r} in [case_data] of {source} "
                    f"matched no files (resolved pattern: {path}).")
            resolved.extend(matches)
        else:
            resolved.append(path)
    seen: set[Path] = set()
    unique: list[Path] = []
    duplicated: list[Path] = []
    for path in resolved:
        if path in seen:
            duplicated.append(path)
            continue
        seen.add(path)
        unique.append(path)
    if duplicated:
        warn(f"forcing in [case_data] of {source} lists "
             f"{len(duplicated)} path(s) more than once; each is read "
             "once (first occurrence kept)")
    return tuple(unique)


def build_case_data(raw: dict, *, source: str, base_dir: Path
                    ) -> CaseDataConfig:
    """Validate a parsed ``[case_data]`` table dict (fail-loud)."""
    unknown = sorted(set(raw) - _KNOWN_KEYS)
    if unknown:
        warn(f"ignoring unknown key(s) {unknown} in [case_data] of "
             f"{source}",
             why=f"Known [case_data] keys: {sorted(_KNOWN_KEYS)}.")
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(
            f"[case_data] of {source} is missing required key(s) "
            f"{missing}: every input path and policy is declared, never "
            "implicit.")

    forcing = _resolve_forcing(base_dir, raw["forcing"], source)
    vtable = _resolve_path(base_dir, raw["vtable"], "vtable", source)
    wps_namelist = _resolve_path(base_dir, raw["wps_namelist"],
                                 "wps_namelist", source)
    geog_root = _resolve_path(base_dir, raw["geog_root"], "geog_root",
                              source)
    has_orog_path = "source_orography" in raw
    has_orog_var = "source_orography_variable" in raw
    if has_orog_path != has_orog_var:
        raise ValueError(
            f"source_orography and source_orography_variable in [case_data] "
            f"of {source} must be declared together or both omitted for "
            "the era5_z_invariant forcing-catalog provider.")
    orog_path = None
    orog_by_domain = None
    orog_var = None
    if has_orog_path:
        orog_var = raw["source_orography_variable"]
        if not isinstance(orog_var, str) or not orog_var:
            raise ValueError(
                f"source_orography_variable in [case_data] of {source} must "
                f"be a non-empty variable name, got {orog_var!r}.")
        raw_orography = raw["source_orography"]
        if isinstance(raw_orography, dict):
            if not raw_orography:
                raise ValueError(
                    f"source_orography in [case_data] of {source} must "
                    "declare at least one per-domain artifact.")
            parsed: list[tuple[int, SourceOrography]] = []
            for key, value in raw_orography.items():
                match = (_DOMAIN_SOURCE_KEY.fullmatch(key)
                         if isinstance(key, str) else None)
                if match is None or int(match.group(1)) < 1:
                    raise ValueError(
                        f"per-domain source_orography key {key!r} in "
                        f"[case_data] of {source} must have WRF form d01, "
                        "d02, ...")
                domain_id = int(match.group(1))
                path = _resolve_path(
                    base_dir, value, f"source_orography.{key}", source)
                parsed.append((
                    domain_id, SourceOrography(path=path, variable=orog_var)))
            orog_by_domain = tuple(sorted(parsed))
        else:
            orog_path = _resolve_path(base_dir, raw_orography,
                                      "source_orography", source)

    file_inputs = [*(("forcing", path) for path in forcing),
                   ("vtable", vtable), ("wps_namelist", wps_namelist)]
    if orog_path is not None:
        file_inputs.append(("source_orography", orog_path))
    elif orog_by_domain is not None:
        file_inputs.extend(
            (f"source_orography.d{domain_id:02d}", artifact.path)
            for domain_id, artifact in orog_by_domain)
    for role, path in file_inputs:
        if not path.is_file():
            raise ValueError(
                f"{role} file {path} declared in [case_data] of {source} "
                "does not exist.")
    if not geog_root.is_dir():
        raise ValueError(
            f"geog_root directory {geog_root} declared in [case_data] of "
            f"{source} does not exist.")

    sfcp = raw["sfcp_to_sfcp"]
    if not isinstance(sfcp, bool):
        raise ValueError(
            f"sfcp_to_sfcp in [case_data] of {source} must be a boolean, "
            f"got {sfcp!r}.")
    if not sfcp:
        raise ValueError(
            f"sfcp_to_sfcp=false branch is not implemented in {source}: "
            "WRF reconstructs surface pressure from PMSL and pressure/GHT "
            "profiles through sfcprs3; copying PSFC is not equivalent.")

    co2 = raw.get("co2_vmr")
    if co2 is not None:
        if isinstance(co2, bool) or not isinstance(co2, (int, float)):
            raise ValueError(
                f"co2_vmr in [case_data] of {source} must be a number (mole "
                f"fraction), got {co2!r}.")
        co2 = float(co2)
        if not math.isfinite(co2) or co2 <= 0.0:
            raise ValueError(
                f"co2_vmr = {co2!r} in [case_data] of {source} must be a "
                "finite positive mole fraction -- e.g. 330 ppm is "
                "330.0e-6.")
        if co2 >= 1.0e-2:
            warn(f"co2_vmr = {co2!r} in [case_data] of {source} exceeds "
                 "the 1e-2 sanity envelope (330 ppm is 330.0e-6; check "
                 "the unit); continuing with it as written")

    title = raw["output_title"]
    if not isinstance(title, str) or not title:
        warn(f"output_title in [case_data] of {source} is empty or not "
             "a string; using 'gpuwm' as the wrfout TITLE attribute")
        title = "gpuwm"

    domain = raw.get("output_domain", 1)
    if isinstance(domain, bool) or not isinstance(domain, int) or domain < 1:
        raise ValueError(
            f"output_domain in [case_data] of {source} must be a positive "
            f"integer grid id, got {domain!r}.")

    interval = raw.get("forcing_interval_s")
    if interval is not None:
        if isinstance(interval, bool) or not isinstance(interval,
                                                        (int, float)):
            raise ValueError(
                f"forcing_interval_s in [case_data] of {source} must be a "
                f"number of seconds, got {interval!r}.")
        interval = float(interval)
        if not math.isfinite(interval) or interval <= 0.0:
            raise ValueError(
                f"forcing_interval_s = {interval!r} in [case_data] of "
                f"{source} must be a finite positive interval in seconds "
                "(omit the key to discover the schedule from the files).")

    return CaseDataConfig(
        forcing=forcing, vtable=vtable, wps_namelist=wps_namelist,
        geog_root=geog_root,
        source_orography=(
            SourceOrography(path=orog_path, variable=orog_var)
            if orog_path is not None else
            PerDomainSourceOrography(orog_by_domain)
            if orog_by_domain is not None else None),
        sfcp_to_sfcp=sfcp, co2_vmr=co2, output_title=title,
        output_domain=domain, forcing_interval_s=interval)


def load_case_data(path: str | Path) -> CaseDataConfig:
    """Load only the ``[case_data]`` table of an experiment TOML."""
    path = Path(path)
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    table = raw.get("case_data")
    if table is None:
        raise ValueError(
            f"experiment config {path} carries no [case_data] table; the "
            "experiment runtime requires declared inputs (forcing, vtable, "
            "wps_namelist, geog_root, and policies).")
    return build_case_data(table, source=str(path), base_dir=path.parent)


def load_experiment_case(path: str | Path
                         ) -> tuple[ExperimentConfig, CaseDataConfig]:
    """Load one TOML into its (experiment, case-data) config pair.

    The ``[case_data]`` table is split off before the experiment loader
    runs, so :mod:`gpuwm.experiment` (which rejects unknown tables)
    stays byte-untouched while one file describes a complete case.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    table = raw.pop("case_data", None)
    # An advisory [fetch] hints table (emitted by `gpuwm domain`) splits
    # off the same way; its schema is owned by gpuwm.fetch and validated,
    # never silently dropped.
    fetch_table = raw.pop("fetch", None)
    if fetch_table is not None:
        from gpuwm.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source=str(path))
    # Validate the experiment FIRST so an invalid experiment surfaces its
    # own error even when [case_data] is also missing.
    experiment = build_experiment(raw, source=str(path))
    if table is None:
        raise ValueError(
            f"experiment config {path} carries no [case_data] table; the "
            "experiment runtime requires declared inputs (forcing, vtable, "
            "wps_namelist, geog_root, and policies).")
    data = build_case_data(table, source=str(path), base_dir=path.parent)
    return experiment, data


__all__ = [
    "CaseDataConfig", "PerDomainSourceOrography", "ResolvedInput",
    "SourceOrography", "SourceOrographyDeclaration", "build_case_data",
    "load_case_data", "load_experiment_case", "resolve_source_orography",
]

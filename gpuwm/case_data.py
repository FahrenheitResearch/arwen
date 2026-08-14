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

Validation is fail-loud: unknown keys, missing keys, wrong types,
missing files, and empty globs are hard errors at load.  One residual
advisory is deliberate, and it is not a key-schema decision: a
``co2_vmr`` above the sanity envelope is loaded exactly as written with
one warning line -- the value the run uses is the one written.  A blank
``output_title`` used to fall back to ``gpuwm``, which substituted a
value into the wrfout ``TITLE`` attribute under a key the reader did
write; it refuses now, like every other wrong type here.

Paths resolve relative to the TOML file's directory; entries containing
glob characters expand (sorted) so a case may declare per-time forcing
files with one pattern.

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
import io
import math
import os
import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

from gpuwm.experiment import (ExperimentConfig, build_experiment,
                              did_you_mean)
from gpuwm.explain import layered, warn

_GLOB_CHARS = frozenset("*?[")

#: Environment variable naming the directory that holds case bundles.
#: This is the one path variable with a default, so a committed config
#: that references it resolves on a machine that sets nothing.
CASE_DATA_ROOT_ENV = "GPUWM_CASE_DATA_ROOT"

_PATH_VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _is_windows() -> bool:
    """Platform seam for :func:`default_case_data_root` (tests patch it)."""
    return os.name == "nt"


def default_case_data_root() -> Path:
    """Where case bundles live when CASE_DATA_ROOT_ENV is unset.

    Platform-aware.  On Windows it stays the download directory -- the
    same fallback the case modules have always used for their
    reference-data environment variables, so nothing about resolution
    changes for a developer who sets no variable.  Everywhere else it is
    the XDG data directory (``$XDG_DATA_HOME/gpuwm``, falling back to
    ``~/.local/share/gpuwm`` per the basedir spec, which also says a
    relative ``XDG_DATA_HOME`` must be ignored): ``~/Downloads`` is a
    Windows convention that a first ``pip install`` on a Linux box used
    to reproduce under the root account's own home directory -- a place
    no XDG-following tool would ever put data.
    """
    if _is_windows():
        return Path.home() / "Downloads"
    raw = os.environ.get("XDG_DATA_HOME", "").strip()
    if raw and Path(raw).is_absolute():
        return Path(raw) / "gpuwm"
    return Path.home() / ".local" / "share" / "gpuwm"


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
    "source_orography_variable", "co2_vmr", "water_temperature_overlay",
    "water_temperature_policy",
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
    #: Optional high-resolution water-temperature analysis (task #71):
    #: replaces SST/SKINTEMP over water SOURCE cells before horizontal
    #: interpolation.  Absent (None) is the identity path.  This key
    #: lives in [case_data], never on ExperimentConfig, so the
    #: experiment fingerprint / restart identity payload is untouched
    #: when it is not declared; declared, it joins the InputCatalog and
    #: the run identity moves exactly when the data does.
    water_temperature_overlay: Path | None = None
    #: Which provider decides the water-surface temperature.  Silence is
    #: ``era5_class_coherent``: one provider per surface class and per
    #: connected body of water, which is what makes a bare run's lakes
    #: coherent.  ``wrf_compat`` restores the historical per-cell choice
    #: between mapped SST and mapped SKINTEMP for stock-WRF certification,
    #: and a declared overlay keeps its own precedence.  See
    #: :mod:`gpuwm.ingest.water_temperature`.
    water_temperature_policy: str | None = None
    #: Optional validated ``[static.highres]`` block
    #: (:class:`gpuwm.static.highres_production.HighresStaticConfig`).
    #: Absence (None) is the identity path: the 30-arc-second baseline
    #: static build runs byte-unchanged and nothing is fetched.
    static_highres: object | None = None
    authority_backed: bool = field(default=False, repr=False, compare=False)
    authority_identity_forcing: tuple[Path, ...] = field(
        default=(), repr=False, compare=False)
    authority_identity_vtable: Path | None = field(
        default=None, repr=False, compare=False)
    authority_identity_wps_namelist: Path | None = field(
        default=None, repr=False, compare=False)
    authority_identity_source_orography: (
        SourceOrographyDeclaration | None) = field(
            default=None, repr=False, compare=False)
    authority_identity_water_temperature_overlay: Path | None = field(
        default=None, repr=False, compare=False)

    def forcing_identity(self) -> tuple[Path, ...]:
        """Original declared forcing paths, never transient CAS paths."""

        return (self.authority_identity_forcing if self.authority_backed
                else self.forcing)

    def vtable_identity(self) -> Path:
        return (self.authority_identity_vtable if self.authority_backed
                else self.vtable)

    def wps_namelist_identity(self) -> Path:
        return (self.authority_identity_wps_namelist
                if self.authority_backed else self.wps_namelist)

    def source_orography_identity(self) -> SourceOrographyDeclaration | None:
        return (self.authority_identity_source_orography
                if self.authority_backed else self.source_orography)

    def water_temperature_overlay_identity(self) -> Path | None:
        return (self.authority_identity_water_temperature_overlay
                if self.authority_backed else self.water_temperature_overlay)

    def source_orography_for_domain(
            self, domain_id: int) -> SourceOrography | None:
        """Resolve the artifact for ``domain_id`` or select the provider."""
        return resolve_source_orography(self.source_orography, domain_id)

    def resolved_inputs(self) -> tuple[ResolvedInput, ...]:
        """Every declared input artifact, for provenance reporting."""
        records = [ResolvedInput(role="forcing", path=path)
                   for path in self.forcing_identity()]
        records.append(ResolvedInput(
            role="vtable", path=self.vtable_identity()))
        records.append(ResolvedInput(role="wps_namelist",
                                     path=self.wps_namelist_identity()))
        records.append(ResolvedInput(role="geog_root", path=self.geog_root))
        source_orography = self.source_orography_identity()
        if isinstance(source_orography, PerDomainSourceOrography):
            for domain_id, artifact in source_orography.by_domain:
                records.append(ResolvedInput(
                    role="source_orography", path=artifact.path,
                    detail=(f"variable={artifact.variable};"
                            f"domain=d{domain_id:02d}")))
        elif source_orography is not None:
            records.append(ResolvedInput(
                role="source_orography", path=source_orography.path,
                detail=f"variable={source_orography.variable}"))
        overlay = self.water_temperature_overlay_identity()
        if overlay is not None:
            records.append(ResolvedInput(
                role="water_temperature_overlay", path=overlay))
        return tuple(records)


def remap_case_data_files(
        data: CaseDataConfig, replacements: Mapping[Path, Path],
        ) -> CaseDataConfig:
    """Return ``data`` with every declared file redirected atomically.

    ``geog_root`` is deliberately excluded: it is a directory authority,
    while this mapping is for the supervisor's content-addressed snapshots of
    forcing, Vtable, WPS namelist, and declared source-orography files.
    """

    normalized = {
        Path(source).resolve(): Path(target).resolve()
        for source, target in replacements.items()
    }

    def mapped(path: Path) -> Path:
        source = Path(path).resolve()
        try:
            return normalized[source]
        except KeyError as error:
            raise ValueError(
                f"no supervisor file authority was supplied for {source}") \
                from error

    identity_forcing = tuple(data.forcing_identity())
    identity_vtable = data.vtable_identity()
    identity_wps = data.wps_namelist_identity()
    identity_orography = data.source_orography_identity()
    identity_overlay = data.water_temperature_overlay_identity()
    declaration = data.source_orography
    if isinstance(declaration, PerDomainSourceOrography):
        declaration = replace(declaration, by_domain=tuple(
            (domain_id, replace(artifact, path=mapped(artifact.path)))
            for domain_id, artifact in declaration.by_domain))
    elif declaration is not None:
        declaration = replace(declaration, path=mapped(declaration.path))
    return replace(
        data,
        forcing=tuple(mapped(path) for path in data.forcing),
        vtable=mapped(data.vtable),
        wps_namelist=mapped(data.wps_namelist),
        source_orography=declaration,
        water_temperature_overlay=(
            None if data.water_temperature_overlay is None
            else mapped(data.water_temperature_overlay)),
        authority_backed=True,
        authority_identity_forcing=identity_forcing,
        authority_identity_vtable=identity_vtable,
        authority_identity_wps_namelist=identity_wps,
        authority_identity_source_orography=identity_orography,
        authority_identity_water_temperature_overlay=identity_overlay)


def _resolve_path(base_dir: Path, value, key: str, source: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{key} in [case_data] of {source} must be a non-empty path "
            f"string, got {value!r}.")
    path = Path(expand_path_variables(value, key, source))
    if not path.is_absolute():
        path = base_dir / path
    return path


def _resolve_forcing(base_dir: Path, value, source: str, *,
                     require_match: bool = True) -> tuple[Path, ...]:
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
                # A caller that never opens the forcing (gpuwm static)
                # must not be refused because the cycle it does not read
                # has not been downloaded.  The pattern still had to be
                # declared and still had to parse; it simply expands to
                # nothing here, and the empty tuple says so honestly
                # rather than inventing a path that does not exist.
                if not require_match:
                    continue
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


def build_case_data(raw: dict, *, source: str, base_dir: Path,
                    require_inputs: bool = True,
                    require_met_inputs: bool = True) -> CaseDataConfig:
    """Validate a parsed ``[case_data]`` table dict (fail-loud).

    ``require_inputs`` is on by default and every run keeps it on: a
    forecast that starts without its forcing is a forecast that fails
    an hour in instead of at the front door.

    It exists for the one caller that is honestly asking a different
    question -- what WOULD this run be, before its data is fetched.
    A plan can be resolved and its VRAM estimated from the geometry and
    physics alone, and a front end showing that estimate must not have
    to download a cycle first.  Turning it off skips ONLY the existence
    check; every schema, type and policy rule still runs, and the paths
    are still resolved, so the caller can report exactly which declared
    inputs are not there yet.

    ``require_met_inputs`` is the narrower form of the same honesty, for
    a caller that reads SOME inputs but provably not the meteorological
    ones.  ``gpuwm static`` is that caller: it builds geography from
    ``geog_root`` and the WPS namelist, and never opens the forcing GRIB
    or the Vtable.  With this False those two are resolved and validated
    but not required on disk, while ``geog_root`` and ``wps_namelist``
    -- which the static build DOES read -- stay required as strictly as
    ever.  Every declaration is still mandatory; this is about which
    bytes must already be fetched, not about which keys must be written.
    """
    unknown = sorted(set(raw) - _KNOWN_KEYS)
    if unknown:
        # This used to warn and drop.  The missing-key check below cannot
        # cover for it, and the reason is structural: the OPTIONAL keys
        # are optional by construction, so a misspelling of any of them
        # passes every check that follows and the case loads with the
        # built-in default under the name of the user's value.
        #
        # `forcing_interval_s` is the one that reaches the numbers.  It
        # is not only a policy assertion checked against the decoded
        # schedule (runtime.forcing_schedule); it also SELECTS the
        # forcing window the catalog consumes -- ingest/preflight's
        # _select_contiguous_times keeps the unique longest contiguous
        # run at the declared interval and excludes the rest, which is
        # how a retrieval returning a date x time cross-product has its
        # non-contiguous extras removed.  Dropped, the whole decoded set
        # reaches the run: a different forcing window, a longer run
        # ceiling, `time.forcing_interval_s = discover` in the resolved
        # config, and a forecast that completes and reports success.
        # One stderr line is not a defence against that.  The measured
        # before/after is in tests/test_case_data_unknown_keys_refuse.py.
        named = ", ".join(
            f"{key!r}{did_you_mean(key, _KNOWN_KEYS)}" for key in unknown)
        raise ValueError(layered(
            f"[case_data] of {source} does not have a key {named}; no key "
            "is ignored, because a dropped key runs a default under the "
            "name of your value.",
            f"Known [case_data] keys: {sorted(_KNOWN_KEYS)}."))
    missing = [key for key in _REQUIRED_KEYS if key not in raw]
    if missing:
        raise ValueError(
            f"[case_data] of {source} is missing required key(s) "
            f"{missing}: every input path and policy is declared, never "
            "implicit.")

    forcing = _resolve_forcing(base_dir, raw["forcing"], source,
                               require_match=require_met_inputs)
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

    water_policy = None
    if "water_temperature_policy" in raw:
        from gpuwm.ingest.water_temperature import (
            validate_water_temperature_policy)
        raw_policy = raw["water_temperature_policy"]
        if not isinstance(raw_policy, str):
            raise ValueError(
                "water_temperature_policy in [case_data] of "
                f"{source} must be a policy name")
        water_policy = validate_water_temperature_policy(raw_policy)

    overlay_path = None
    if "water_temperature_overlay" in raw:
        overlay_path = _resolve_path(
            base_dir, raw["water_temperature_overlay"],
            "water_temperature_overlay", source)

    # `forcing` and `vtable` are the MET inputs.  A caller that only
    # builds static geography (`gpuwm static`) never reads either one --
    # runtime.write_static touches geog_root, the WPS namelist and the
    # projection, and nothing else -- so requiring them to be on disk
    # made "build 30 m terrain for this domain" refuse until a whole
    # forecast cycle had been downloaded.  That is a gate on an input the
    # command does not consume, and it sat directly across the
    # documented high-resolution terrain path.
    #
    # The DECLARATIONS stay required in every case: the schema is the
    # config's identity and a static build still describes the case it
    # belongs to.  Only the existence check is scoped to the callers that
    # actually read the bytes.
    met_roles = {"forcing", "vtable"}
    file_inputs = [*(("forcing", path) for path in forcing),
                   ("vtable", vtable), ("wps_namelist", wps_namelist)]
    if overlay_path is not None:
        file_inputs.append(("water_temperature_overlay", overlay_path))
    if orog_path is not None:
        file_inputs.append(("source_orography", orog_path))
    elif orog_by_domain is not None:
        file_inputs.extend(
            (f"source_orography.d{domain_id:02d}", artifact.path)
            for domain_id, artifact in orog_by_domain)
    if require_inputs:
        for role, path in file_inputs:
            if not require_met_inputs and role.split(".")[0] in met_roles:
                continue
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
        # co2_vmr above, output_domain below and forcing_interval_s below
        # that all RAISE on a wrong type; this one alone warned and
        # substituted, in a module whose docstring says "Validation is
        # fail-loud: unknown keys, missing keys, wrong types".  The
        # substituted value is stamped into the wrfout TITLE attribute,
        # i.e. into the provenance of every frame, under a key the user
        # did write.
        raise ValueError(
            f"output_title in [case_data] of {source} must be a "
            f"non-empty string, got {title!r}; it is written to the "
            f"wrfout TITLE attribute, so a substituted default would "
            f"describe your output as something you did not name.")

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
        output_domain=domain, forcing_interval_s=interval,
        water_temperature_overlay=overlay_path,
        water_temperature_policy=water_policy)


def load_case_data(path: str | Path) -> CaseDataConfig:
    """Load only the ``[case_data]`` table of an experiment TOML."""
    from gpuwm.config_authority import read_config_authority

    authority = read_config_authority(path)
    path = authority.source
    raw = tomllib.load(io.BytesIO(authority.payload))
    table = raw.get("case_data")
    if table is None:
        raise ValueError(
            f"experiment config {path} carries no [case_data] table; the "
            "experiment runtime requires declared inputs (forcing, vtable, "
            "wps_namelist, geog_root, and policies).")
    data = build_case_data(table, source=str(path), base_dir=path.parent)
    from gpuwm.static.highres_production import parse_static_table
    highres = parse_static_table(
        raw.get("static"), source=str(path), base_dir=path.parent)
    if highres is not None:
        data = replace(data, static_highres=highres)
    return data


def load_experiment_case_bytes(
        payload: bytes, *, source: str, base_dir: str | Path,
        require_inputs: bool = True, require_met_inputs: bool = True
        ) -> tuple[ExperimentConfig, CaseDataConfig]:
    """Load captured TOML bytes while retaining their original path base.

    The ``[case_data]`` table is split off before the experiment loader
    runs, so :mod:`gpuwm.experiment` (which rejects unknown tables)
    stays byte-untouched while one file describes a complete case.
    """
    if not isinstance(payload, bytes):
        raise TypeError("experiment config payload must be bytes")
    base = Path(base_dir)
    raw = tomllib.load(io.BytesIO(payload))
    table = raw.pop("case_data", None)
    # An advisory [fetch] hints table (emitted by `gpuwm domain`) splits
    # off the same way; its schema is owned by gpuwm.fetch and validated,
    # never silently dropped.
    fetch_table = raw.pop("fetch", None)
    if fetch_table is not None:
        from gpuwm.fetch import validate_fetch_hints
        validate_fetch_hints(fetch_table, source=source)
    # The optional [static] table (high-resolution static geography)
    # splits off identically; its schema is owned by
    # gpuwm.static.highres_production and validated, never dropped.
    static_table = raw.pop("static", None)
    # Validate the experiment FIRST so an invalid experiment surfaces its
    # own error even when [case_data] is also missing.
    experiment = build_experiment(raw, source=source)
    if table is None:
        raise ValueError(
            f"experiment config {source} carries no [case_data] table; the "
            "experiment runtime requires declared inputs (forcing, vtable, "
            "wps_namelist, geog_root, and policies).")
    data = build_case_data(table, source=source, base_dir=base,
                           require_inputs=require_inputs,
                           require_met_inputs=require_met_inputs)
    if static_table is not None:
        from gpuwm.static.highres_production import parse_static_table
        highres = parse_static_table(
            static_table, source=source, base_dir=base)
        if highres is not None:
            data = replace(data, static_highres=highres)
    return experiment, data


def load_experiment_case(path: str | Path, *,
                         require_met_inputs: bool = True
                         ) -> tuple[ExperimentConfig, CaseDataConfig]:
    """Load one TOML into its (experiment, case-data) config pair.

    ``require_met_inputs=False`` is for `gpuwm static`, which builds
    geography and provably never opens the forcing GRIB or the Vtable.
    See :func:`build_case_data`.
    """

    from gpuwm.config_authority import read_config_authority

    authority = read_config_authority(path)
    path = authority.source
    return load_experiment_case_bytes(
        authority.payload, source=str(path), base_dir=path.parent,
        require_met_inputs=require_met_inputs)


__all__ = [
    "CaseDataConfig", "PerDomainSourceOrography", "ResolvedInput",
    "SourceOrography", "SourceOrographyDeclaration", "build_case_data",
    "load_case_data", "load_experiment_case", "load_experiment_case_bytes",
    "remap_case_data_files", "resolve_source_orography",
]

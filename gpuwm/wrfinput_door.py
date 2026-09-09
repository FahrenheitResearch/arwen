"""Resolve a WRF real.exe directory using the shared namelist translator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from gpuwm.ingest.wrfinput import (
    RELOCATION_REQUIREMENT, WrfinputMetadata, check_boundary_coverage,
    check_grid_agreement, format_scheme_matrix, read_wrfinput_metadata,
    wrfbdy_coverage,
)

#: The projection ids WPS and WRF share, spelled the way
#: ``namelist.wps`` spells them.  ``MAP_PROJ`` is an integer in the
#: wrfinput and a word in the WPS namelist, and this is the only place
#: the two spellings meet.
_MAP_PROJ_NAMES = {
    1: "lambert", 2: "polar", 3: "mercator", 6: "lat-lon",
}

#: Global attributes the synthesized WPS namelist is built from.  Named
#: so a wrfinput missing one is refused by ATTRIBUTE rather than by a
#: KeyError six frames down.
_REQUIRED_LAYOUT_ATTRIBUTES = (
    "MAP_PROJ", "TRUELAT1", "TRUELAT2", "STAND_LON", "CEN_LAT", "CEN_LON",
    "DX", "DY", "GRID_ID", "PARENT_ID", "I_PARENT_START", "J_PARENT_START",
    "PARENT_GRID_RATIO",
)


@dataclass(frozen=True)
class WrfinputRun:
    """One ``real.exe`` run directory, resolved but not yet restored."""

    directory: Path
    namelist_input: Path
    wrfinput_paths: Mapping[int, Path]
    wrfbdy_path: Path
    metadata: Mapping[int, WrfinputMetadata]
    coverage: object
    experiment: object
    toml_text: str
    substitution_report: object
    wps_namelist_text: str

    @property
    def grid_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.wrfinput_paths))


def discover_run_directory(directory: str | Path
                           ) -> tuple[Path, Path, dict[int, Path], Path]:
    """Locate ``namelist.input``, every ``wrfinput_d0N`` and ``wrfbdy_d01``.

    Named refusals, one per missing piece, because a reader who pointed
    the door at the wrong directory is owed the list of what a right one
    contains.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(
            f"--wrfinput {directory} is not a directory; point it at the "
            "WRF run directory holding wrfinput_d0N, wrfbdy_d01 and "
            "namelist.input")
    namelist = directory / "namelist.input"
    inputs = {}
    for path in sorted(directory.glob("wrfinput_d??")):
        suffix = path.name.removeprefix("wrfinput_d")
        if suffix.isdigit():
            inputs[int(suffix)] = path
    bdy = directory / "wrfbdy_d01"
    absent = []
    if not namelist.is_file():
        absent.append(
            "namelist.input (the run settings and the physics; ArWen reads "
            "the same file wrf.exe would)")
    if not inputs:
        absent.append(
            "wrfinput_d01 (real.exe's initial condition; wrfinput_d02.. "
            "join automatically when present)")
    if not bdy.is_file():
        absent.append(
            "wrfbdy_d01 (real.exe's lateral boundary values and tendencies; "
            "it is what caps the run length)")
    if absent:
        raise ValueError(
            f"{directory} is not a complete real.exe handoff.  Missing:\n"
            + "\n".join(f"  - {item}" for item in absent))
    if 1 not in inputs:
        raise ValueError(
            f"{directory} has {sorted(inputs)} but no wrfinput_d01; the "
            "root domain is the one wrfbdy_d01 forces and cannot be "
            "omitted")
    return directory, namelist, inputs, bdy


def synthesize_wps_namelist(metadata: Mapping[int, WrfinputMetadata]) -> str:
    """Write the ``&share``/``&geogrid`` facts out of the wrfinput files.

    This is the whole difference between this door and
    ``gpuwm import-namelist``.  A ``real.exe`` handoff has no
    ``namelist.wps`` -- WPS ran days ago on someone else's machine -- but
    every fact that namelist carried was stamped into the wrfinput's
    global attributes by ``real.exe`` itself.  Reconstituting it here
    means ONE translator serves both doors, and it means the projection
    the run uses is the projection the FILES were written on, not a
    second declaration that could disagree with them.
    """
    ordered = [metadata[grid_id] for grid_id in sorted(metadata)]
    root = ordered[0]
    for item in ordered:
        missing = sorted(
            name for name in _REQUIRED_LAYOUT_ATTRIBUTES
            if name not in item.global_attributes)
        if missing:
            raise ValueError(
                f"{item.path} is missing the global attribute(s) {missing} "
                "that carry the domain layout; real.exe stamps every one "
                "of them, so this file was not written by real.exe or was "
                "rewritten by a tool that dropped them")
    map_proj = int(float(root.global_attributes["MAP_PROJ"]))
    if map_proj not in _MAP_PROJ_NAMES:
        raise ValueError(
            f"{root.path} declares MAP_PROJ={map_proj}, which is not one "
            f"of WPS's {sorted(_MAP_PROJ_NAMES)} "
            f"({', '.join(_MAP_PROJ_NAMES[k] for k in sorted(_MAP_PROJ_NAMES))})")

    def column(attribute):
        # The Rust NetCDF bridge promotes every numeric attribute to f64,
        # so PARENT_ID arrives as 0.0; the namelist needs the integer.
        return ", ".join(
            str(int(float(item.global_attributes[attribute])))
            for item in ordered)

    def parent_id_column():
        """WPS spells the root's parent as ITSELF; WRF spells it 0.

        ``&geogrid parent_id`` is 1 for domain 1 (WPS's own convention,
        which gpuwm's WPS reader enforces), while ``real.exe`` stamps
        ``PARENT_ID = 0`` into the root wrfinput.  Same fact, two
        spellings; this is the one place they meet.  Children carry their
        real parent id through unchanged.
        """
        values = []
        for item in ordered:
            value = int(float(item.global_attributes["PARENT_ID"]))
            values.append(1 if value == 0 else value)
        return ", ".join(str(value) for value in values)

    start = root.start_date or ""
    lines = [
        "&share",
        " wrf_core = 'ARW',",
        f" max_dom = {len(ordered)},",
        " start_date = " + ", ".join(f"'{start}'" for _ in ordered) + ",",
        "/",
        "",
        "&geogrid",
        f" parent_id         = {parent_id_column()},",
        f" parent_grid_ratio = {column('PARENT_GRID_RATIO')},",
        f" i_parent_start    = {column('I_PARENT_START')},",
        f" j_parent_start    = {column('J_PARENT_START')},",
        " e_we              = " + ", ".join(
            str(item.nx + 1) for item in ordered) + ",",
        " e_sn              = " + ", ".join(
            str(item.ny + 1) for item in ordered) + ",",
        f" dx = {float(root.global_attributes['DX']):.10g},",
        f" dy = {float(root.global_attributes['DY']):.10g},",
        f" map_proj = '{_MAP_PROJ_NAMES[map_proj]}',",
        f" ref_lat   = {float(root.global_attributes['CEN_LAT']):.10g},",
        f" ref_lon   = {float(root.global_attributes['CEN_LON']):.10g},",
        f" truelat1  = {float(root.global_attributes['TRUELAT1']):.10g},",
        f" truelat2  = {float(root.global_attributes['TRUELAT2']):.10g},",
        f" stand_lon = {float(root.global_attributes['STAND_LON']):.10g},",
        "/",
        "",
    ]
    return "\n".join(lines)


def check_projection_agreement(experiment, metadata: Mapping[
        int, WrfinputMetadata], *, tolerance_deg: float = 5.0e-3) -> dict:
    """Prove the reconstituted projection lands on the file's own grid.

    The layout above is reconstituted from ``CEN_LAT``/``CEN_LON``, which
    is WPS's ``ref_lat``/``ref_lon`` only when the reference point sat at
    the domain centre -- the default, but not a law.  Rather than trust
    it, this evaluates ArWen's projection on every mass point and
    compares it against the ``XLAT``/``XLONG`` arrays ``real.exe`` wrote
    into the same file.  An absent check here would be scored as
    agreement; this is the measurement.

    Returns the per-domain maximum absolute latitude/longitude
    disagreement in degrees.
    """
    from gpuwm import netcdf_bridge
    from gpuwm.static.projection import grids_from_projection_config

    grids = grids_from_projection_config(experiment)
    worst = {}
    problems = []
    for domain, grid in zip(experiment.domains, grids):
        item = metadata.get(domain.grid_id)
        if item is None:
            continue
        with netcdf_bridge.open_dataset(item.path) as dataset:
            missing = [name for name in ("XLAT", "XLONG") if name not in dataset.variables]
            if missing:
                raise ValueError(f"{item.path}: missing coordinate evidence {missing}")
            xlat = np.asarray(dataset.variables["XLAT"][...],
                              dtype=np.float64)
            xlon = np.asarray(dataset.variables["XLONG"][...],
                              dtype=np.float64)
        xlat = xlat.reshape(xlat.shape[-2:])
        xlon = xlon.reshape(xlon.shape[-2:])
        lat, lon = grid.latlon_mass()
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.shape != xlat.shape:
            problems.append(
                f"  d{domain.grid_id:02d}: ArWen's grid is {lat.shape}, "
                f"{item.path.name} XLAT is {xlat.shape}")
            continue
        dlat = float(np.abs(lat - xlat).max())
        dlon = float(np.abs(
            ((lon - xlon + 180.0) % 360.0) - 180.0).max())
        worst[domain.grid_id] = (dlat, dlon)
        if not np.isfinite([dlat, dlon]).all() or max(dlat, dlon) > tolerance_deg:
            problems.append(
                f"  d{domain.grid_id:02d}: max |dlat| = {dlat:.6g} deg, "
                f"max |dlon| = {dlon:.6g} deg")
    if problems:
        raise ValueError(
            "the projection reconstituted from the wrfinput global "
            "attributes does not reproduce the grid real.exe wrote into "
            "the same file:\n" + "\n".join(problems)
            + f"\n(tolerance {tolerance_deg} deg).  The layout is taken "
            "from CEN_LAT/CEN_LON, which is WPS's ref_lat/ref_lon only "
            "when the reference point sat at the domain centre; this "
            "handoff placed it elsewhere.  Use `gpuwm import-namelist` "
            "with the original namelist.wps instead.")
    return worst


def require_preserved_wrf_selectors(report):
    """A reachable WRF door must not replace an explicit physics package.

    Only ``use_theta_m = 1`` to ``use_theta_m = 0`` is an admitted declared
    divergence, and it must carry the reason announced at the terminal by
    ``announce_wrf_substitutions``. A reason explains a change; it does not
    authorize replacing any other requested physics selector.
    """
    for item in report.substitutions:
        if (item.key == item.gpuwm_key == "use_theta_m"
                and item.wrf_value == 1 and item.gpuwm_value == 0
                and item.reason):
            continue
        if item.wrf_value != item.gpuwm_value:
            raise ValueError(
                f"the requested {item.wrf_name} ({item.key}={item.wrf_value}) has no native implementation; "
                f"translation would replace it with {item.gpuwm_name} ({item.gpuwm_key}={item.gpuwm_value}). "
                "Select a supported physics package explicitly in the producing namelist")


def resolve_wrfinput_run(directory: str | Path, *, name: str | None = None,
                         acknowledgements: Sequence[str] = (),
                         rrtmg_variant: str | None = None,
                         check_projection: bool = True) -> WrfinputRun:
    """Read, translate and CHECK one real.exe handoff.  No field is decoded.

    Every refusal this door owns fires here, on CPU, before any state
    exists: unsupported physics packages by name, a grid the namelist and
    the files disagree about, a projection that does not reproduce the
    file's own XLAT/XLONG, and a run longer than wrfbdy's coverage.
    """
    import tomllib

    from gpuwm.experiment import build_experiment
    from gpuwm.ingest.wrfinput_identity import read_wrfinput_identity, check_wrfinput_identity
    from gpuwm.config import soil_layer_count

    directory, namelist, inputs, bdy = discover_run_directory(directory)
    # Scheme support is established HERE, from each file's own
    # MP_PHYSICS/SF_SURFACE_PHYSICS, before the namelist is even parsed:
    # an unported package is the user's shortest path to an answer and
    # must not be buried under a namelist translation refusal.
    metadata = {grid_id: read_wrfinput_metadata(path)
                for grid_id, path in sorted(inputs.items())}
    for grid_id, item in metadata.items():
        if item.grid_id != grid_id:
            raise ValueError(
                f"{item.path} declares GRID_ID={item.grid_id} but is named "
                f"for domain {grid_id}; the file and its name disagree "
                "about which domain it initializes")
    identities = {grid_id: read_wrfinput_identity(path) for grid_id, path in inputs.items()}
    landuse_identity = None
    for item in metadata.values():
        identity = {key: item.global_attributes.get(key) for key in ('MMINLU', 'NUM_LAND_CAT')}
        if landuse_identity is not None and identity != landuse_identity:
            raise ValueError('WRF input files disagree on the producing run land-use identity')
        landuse_identity = identity
    wps_text = synthesize_wps_namelist(metadata)
    toml_text, report = _import_with_synthesized_wps(
        wps_text, namelist, inherited_eta=identities[min(identities)].eta_levels, name=name, rrtmg_variant=rrtmg_variant,
        acknowledgements=tuple(acknowledgements), landuse_identity=landuse_identity,
        wrf_boundary_use_theta_m=int(metadata[min(metadata)].global_attributes['USE_THETA_M']))
    require_preserved_wrf_selectors(report)
    # ``import_namelists`` has already validated this text through
    # ``build_experiment``; building it again is how the door gets the
    # ExperimentConfig object rather than a second validation.
    experiment = build_experiment(
        tomllib.loads(toml_text), source=f"{directory}:wrfinput")
    if len(experiment.domains) != len(metadata):
        raise ValueError(
            f"{namelist} declares {len(experiment.domains)} domain(s) but "
            f"{directory} carries wrfinput files for {sorted(metadata)}; "
            "real.exe writes one wrfinput per domain, so one of the two "
            "is from a different experiment")
    for domain in experiment.domains:
        item = metadata.get(domain.grid_id)
        if item is None:
            raise ValueError(
                f"{namelist} declares domain {domain.grid_id} but "
                f"{directory} has no wrfinput_d{domain.grid_id:02d}")
        check_grid_agreement(item, domain.run, source=str(namelist))
        check_wrfinput_identity(identities[domain.grid_id], domain=domain,
                               vertical=experiment.vertical, start_time=domain.start_time or experiment.start_time,
                               soil_layers=soil_layer_count(domain.run))
    if check_projection:
        check_projection_agreement(experiment, metadata)
    coverage = wrfbdy_coverage(bdy)
    check_boundary_coverage(coverage, experiment.run_seconds,
                            source=str(namelist))
    _check_start_time(experiment, metadata, coverage, source=str(namelist))
    return WrfinputRun(
        directory=directory, namelist_input=namelist,
        wrfinput_paths=MappingProxyType(dict(sorted(inputs.items()))),
        wrfbdy_path=bdy, metadata=MappingProxyType(metadata),
        coverage=coverage, experiment=experiment, toml_text=toml_text,
        substitution_report=report, wps_namelist_text=wps_text)


def _import_with_synthesized_wps(wps_text: str, namelist: Path, *, inherited_eta, **kwargs):
    """Run the shared translator against the reconstituted WPS namelist.

    ``import_namelists`` takes PATHS and it is the one translator both
    doors use, so the reconstituted text becomes a file for the length of
    the call.  It goes in a TEMPORARY directory, never beside the run: a
    WRF run directory is the user's evidence of what they ran, and a door
    whose job is to READ one must not leave a file in it.  The refusals
    that quote the layout quote ``run.wps_namelist_text``, which is
    carried on the result, so nothing is lost by not writing it there.
    """
    import tempfile

    from gpuwm.namelist_import import import_namelists, parse_namelist

    with tempfile.TemporaryDirectory(prefix="gpuwm-wrfinput-") as scratch:
        wps_path = Path(scratch) / "namelist.wps"
        wps_path.write_text(wps_text, encoding="utf-8")
        effective_input = namelist
        if not parse_namelist(namelist).get("domains", {}).get("eta_levels"):
            # real.exe has already generated this coordinate. Bind exactly
            # those values when the producing namelist used its generator.
            # Explicit user eta values are never replaced and are compared.
            import re
            text = namelist.read_text(encoding="utf-8")
            eta = ", ".join(repr(float(value)) for value in inherited_eta)
            text, count = re.subn(r"(?im)^([ \t]*&domains)\b", lambda m: m[0] + "\n eta_levels = " + eta + ",", text, count=1)
            if count != 1:
                raise ValueError(f"{namelist}: cannot locate &domains to bind file eta levels")
            effective_input = Path(scratch) / "namelist.input"
            effective_input.write_text(text, encoding="utf-8")
        return import_namelists(wps_path, effective_input, **kwargs)


def _check_start_time(experiment, metadata, coverage, *, source: str) -> None:
    """Refuse a namelist whose start instant is not the files' own."""
    problems = []
    for grid_id in sorted(metadata):
        declared = metadata[grid_id].start_date
        if not declared:
            continue
        try:
            stamped = datetime.strptime(declared, "%Y-%m-%d_%H:%M:%S")
        except ValueError:
            problems.append(
                f"  wrfinput_d{grid_id:02d} START_DATE {declared!r} is not "
                "a WRF timestamp")
            continue
        if stamped != experiment.start_time:
            problems.append(
                f"  wrfinput_d{grid_id:02d} was written for {stamped}, "
                f"{source} starts at {experiment.start_time}")
    if coverage.times and coverage.start != experiment.start_time:
        problems.append(
            f"  wrfbdy_d01's first boundary record is {coverage.start}, "
            f"{source} starts at {experiment.start_time}")
    if problems:
        raise ValueError(
            "the initial instant does not agree across the handoff:\n"
            + "\n".join(problems)
            + "\nreal.exe stamped these files for one instant; running "
            "them from another silently shifts the whole forecast.")


def format_run_report(run: WrfinputRun) -> str:
    """The human summary of everything the door established, on CPU."""
    lines = [
        f"wrfinput door: {run.directory}",
        f"  namelist.input   {run.namelist_input}",
        f"  experiment       {run.experiment.name}",
        f"  start            {run.experiment.start_time}",
        f"  run length       {run.experiment.run_seconds:.0f} s",
    ]
    for grid_id in run.grid_ids:
        item = run.metadata[grid_id]
        lines.append(
            f"  d{grid_id:02d}  {item.nx} x {item.ny} x {item.nz}, "
            f"{item.soil_layers} soil layers, "
            f"mp_physics={item.mp_physics}, "
            f"sf_surface_physics={item.sf_surface_physics}")
    coverage = run.coverage
    lines.append(
        f"  wrfbdy_d01       {len(coverage.times)} record(s), "
        f"{coverage.forcing_interval_seconds} s cadence, "
        f"bdy_width {coverage.spec_bdy_width}, "
        f"covers {coverage.coverage_seconds:.0f} s "
        f"({coverage.start:%Y-%m-%d_%H:%M:%S} -> "
        f"{coverage.end:%Y-%m-%d_%H:%M:%S})")
    lines.append("  LIMITS OF THIS ENTRY POINT")
    lines.append(
        "    vertical grid and domain layout are fixed by real.exe; "
        "ArWen's eta-ladder, per-domain vertical grids and the "
        "nz=129-256 tier are unavailable for this run")
    lines.append(
        f"    run length is capped at {coverage.coverage_seconds:.0f} s by "
        "wrfbdy_d01 coverage")
    lines.append("    " + RELOCATION_REQUIREMENT.replace("\n", " "))
    return "\n".join(lines)


def wrfinput_help_epilog() -> str:
    """The ``--help`` text for ``--wrfinput``: what it is and what it costs."""
    return (
        "--wrfinput DIR enters ArWen where WRF's real.exe left off.  DIR is "
        "a WRF run directory holding wrfinput_d0N, wrfbdy_d01 and "
        "namelist.input; the namelist supplies the run settings and the "
        "physics exactly as it does for wrf.exe, and the domain layout and "
        "projection are read out of the wrfinput global attributes (so no "
        "namelist.wps is needed).  The translation is `gpuwm "
        "import-namelist`'s, unchanged.\n"
        "\n"
        "WHAT THIS ENTRY POINT COSTS -- real.exe already made these "
        "choices, so ArWen cannot revisit them:\n"
        "  * the VERTICAL GRID is fixed: eta levels, the hybrid "
        "coefficients and p_top are read from the file, so ArWen's "
        "eta-ladder construction, its per-domain vertical grids and its "
        "nz=129-256 tier are unavailable for such a run;\n"
        "  * the DOMAIN LAYOUT is fixed: nx, ny, dx, the projection and "
        "the nest placement are whatever real.exe wrote;\n"
        "  * RUN LENGTH IS CAPPED by wrfbdy_d01 coverage (its records plus "
        "one forcing interval), and a longer run is refused at this door, "
        "naming both numbers, before any integration;\n"
        "  * the SCHEME SET IS CLOSED -- a wrfinput carries one scheme's "
        "hydrometeor inventory and one land-surface scheme's soil "
        "geometry:\n"
        + format_scheme_matrix() + "\n"
        "    anything else is refused by name with a remedy.\n"
        "\n"
        "WHAT SURVIVES: " + RELOCATION_REQUIREMENT
    )

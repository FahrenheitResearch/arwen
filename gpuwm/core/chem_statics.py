"""Grid- and geography-derived chem process arrays, written at preparation.

Some chem processes read fields no dynamical or physics input carries: the
dust schemes' erodibility and soil texture (WPS geography datasets) and the
sulfur chemistry's mass-point latitude and longitude.  They are process
arrays (``chemdiag_``, restart class serialize), declared here and spliced
into the owning processes' ``ALLOCATES``, so they are written ONCE, by the
preparation that builds the initial state from a grid and a geography tree,
and every prepared cache and checkpoint carries them from there.

Every real-case preparation route calls :func:`attach_chem_statics` once its
initial state exists and before that state is published or sealed.  A
chem-off state costs one attribute read.  A route with no geography at hand
passes ``geog_root=None``: the latitude and longitude are still written, the
dust statics are not, and the dust process then refuses at init by name
rather than run with no erodible cell.

Self-contained on purpose: the preparation routes are staged into the RW-WPS
preprocessing wheel, so this module reaches only the chem table, the chem
state and allocation interfaces and the static-field builder -- never a
process module or a kernel.
"""

from __future__ import annotations

import numpy as np

from gpuwm.core.chem_context import ChemAllocation

#: The process keys whose rows need these arrays (the table's process
#: column; gpuwm.core.chem_dust.KEY and gpuwm.core.chem_sulfur.KEY).
DUST_KEY = "emission.dust"
SULFUR_KEY = "chem.sulfur"

#: The WPS static inputs the dust schemes read (Registry.EM_COMMON:146 EROD
#: in three classes; AFWA's CLAYFRAC and SANDFRAC), by the variable names of
#: the chem table's static source rows that provide them.
STATIC_VARIABLES = ("EROD", "CLAYFRAC", "SANDFRAC")

#: The dust process's static arrays (spliced into chem_dust.ALLOCATES).
DUST_STATIC_ALLOCATIONS = (
    ChemAllocation("dust_erod_1", "2d", units="1",
                   description="EROD class 1 (sand), from the static source"),
    ChemAllocation("dust_erod_2", "2d", units="1",
                   description="EROD class 2 (silt), from the static source"),
    ChemAllocation("dust_erod_3", "2d", units="1",
                   description="EROD class 3 (clay), from the static source"),
    ChemAllocation("dust_clayfrac", "2d", units="1",
                   description="CLAYFRAC, from the static source (AFWA)"),
    ChemAllocation("dust_sandfrac", "2d", units="1",
                   description="SANDFRAC, from the static source (AFWA)"),
    ChemAllocation("dust_statics_ready", "2d", units="1",
                   description="1 where the dust statics were sampled onto "
                               "this grid; a dust run refuses a domain that "
                               "carries zeros instead"),
)

#: The sulfur process's grid arrays (spliced into chem_sulfur.ALLOCATES).
SOLAR_ALLOCATIONS = (
    ChemAllocation("sulfur_xlat", "2d", units="degree_north",
                   description="mass-point latitude for szangle, from the "
                               "grid at preparation"),
    ChemAllocation("sulfur_xlong", "2d", units="degree_east",
                   description="mass-point longitude for szangle"),
)


def put_plane(state, attr: str, plane) -> None:
    """Write a host plane into a state array on whichever backend holds it
    (the preparation may build its state on the CPU or on the card)."""
    target = getattr(state, attr)
    value = np.asarray(plane, dtype=np.float64).astype(np.float32)
    if isinstance(target, np.ndarray):
        target[...] = value
    else:
        import cupy as cp

        target[...] = cp.asarray(value)


def static_specs(table_catalog, geog_selection) -> list[dict]:
    """The Rust extra-field rows for the dust statics, from the static source
    rows that provide :data:`STATIC_VARIABLES`."""
    specs = []
    found = set()
    for source in table_catalog.sources.values():
        if source.kind != "static" or source.format != "wps_geog":
            continue
        for variable, spec in source.variables.items():
            if variable not in STATIC_VARIABLES:
                continue
            sel = spec["selector"]
            mask = sel.get("water_mask")
            row = {"output_name": variable,
                   "dataset_path": source.grid["dataset"],
                   "interp_options": list(sel["interp_options"]),
                   "fill_missing": float(sel["fill_missing"]),
                   "planes": int(sel["planes"])}
            if sel.get("z_dim_name"):
                row["z_dim_name"] = sel["z_dim_name"]
            if mask is not None:
                if mask != "landuse":
                    raise ValueError(f"static source {source.name!r}: water_mask "
                                     f"{mask!r} is not a native dataset role")
                row["water_mask"] = {"dataset_path": geog_selection.landuse,
                                     "fill_missing": float(sel["fill_missing"])}
            specs.append(row)
            found.add(variable)
    missing = [v for v in STATIC_VARIABLES if v not in found]
    if missing:
        raise ValueError(f"no static chem source provides {missing}")
    return specs


def attach_dust_statics(state, grid, geog_root, geog_selection) -> None:
    """Sample EROD/CLAYFRAC/SANDFRAC onto ``grid`` into the dust process's
    arrays.  Nothing happens on a chem-off state or one with no dust row."""
    chem = getattr(state, "chem", None)
    if chem is None or not chem.table.rows_for(DUST_KEY):
        return
    from gpuwm.chem_table import catalog
    from gpuwm.core.chem_state import process_attr
    from gpuwm.static.extra_fields import build_extra_fields

    if geog_selection is None:
        from gpuwm.static.build import GeogSelection
        geog_selection = GeogSelection.fallback(geog_root)
    fields = build_extra_fields(grid, geog_root,
                                static_specs(catalog(), geog_selection))
    attach_dust_static_fields(state, fields)


def attach_dust_static_fields(state, fields) -> None:
    """Copy native-sampled dust statics, including a sealed corridor crop."""
    chem = getattr(state, "chem", None)
    if chem is None or not chem.table.rows_for(DUST_KEY):
        return
    from gpuwm.core.chem_state import process_attr

    missing = [name for name in STATIC_VARIABLES if name not in fields]
    if missing:
        raise ValueError(f"dust chemistry statics are missing {missing}; "
                         "prepare EROD, CLAYFRAC and SANDFRAC together")
    by_name = {a.name: a for a in DUST_STATIC_ALLOCATIONS}

    def put(name, plane):
        put_plane(state, process_attr(by_name[name]), plane)

    erod = np.asarray(fields["EROD"], dtype=np.float64)
    for k in range(3):
        put(f"dust_erod_{k + 1}", erod[k])
    put("dust_clayfrac", fields["CLAYFRAC"])
    put("dust_sandfrac", fields["SANDFRAC"])
    put("dust_statics_ready", np.ones(np.shape(fields["CLAYFRAC"])))


def attach_solar_geometry(state, grid) -> None:
    """The grid's mass-point latitude and longitude into the sulfur arrays.

    WRF hands gocart_chem_driver XLAT/XLONG (module_gocart_chem.F:77-79).
    Nothing happens on a chem-off state or one with no sulfur row.
    """
    chem = getattr(state, "chem", None)
    if chem is None or not chem.table.rows_for(SULFUR_KEY):
        return
    from gpuwm.core.chem_state import process_attr

    lat, lon = grid.latlon_mass()
    by_name = {a.name: a for a in SOLAR_ALLOCATIONS}
    put_plane(state, process_attr(by_name["sulfur_xlat"]), lat)
    put_plane(state, process_attr(by_name["sulfur_xlong"]), lon)


def attach_chem_statics(state, grid, geog_root, geog_selection=None, *,
                        static_fields=None) -> None:
    """Write every chem process's preparation-time arrays onto ``state``."""
    if getattr(state, "chem", None) is None:
        return
    if static_fields is not None and any(
            name in static_fields for name in STATIC_VARIABLES):
        attach_dust_static_fields(state, static_fields)
    elif geog_root is not None:
        attach_dust_statics(state, grid, geog_root, geog_selection)
    attach_solar_geometry(state, grid)

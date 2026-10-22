"""WRF-Chem ``&chem`` at the namelist door: what WOOF's chemistry carries.

Owner ruling (2026-10-12): a WRF-Chem
namelist maps onto what WOOF has -- fire smoke, GOCART dust, sea salt and
simple sulfur, deposition -- and is refused by name for the rest.  It never
imports and runs with no chemistry.  Until then every ``chem_opt != 0``
imported as a declared divergence ("WOOF runs no WRF-Chem") and the forecast
ran with none.

How a namelist maps, all of it table data:

* ``chem_opt`` selects a WOOF species set through the set files' own
  ``wrf_chem_opt`` (gpuwm/data/chem/sets/*.json): 300 -> ``gocart_simple``,
  401 -> ``dust``.  The operational WRF 3.9 fork's smoke package
  (``chem_opt = 18`` beside the fork's own keys, NOAA-EMC/HRRR v4.1.21
  ``parm/conus/hrrr_wrf.nl``) maps onto ``smoke``, WOOF's HRRR-Smoke
  descendant, by :data:`FORK_CHEM_OPT_SETS`.  Every other package is refused
  with the menu of what is carried.
* every ``&chem`` key WOOF's RunConfig carries (:data:`CARRIED_KEYS`) is
  translated into ``[shared]`` as written, and
  :func:`gpuwm.config.validate_chem_config` -- the chem law every config
  door loads through -- refuses a value it cannot honour, naming the
  breakage (aerosol-radiation feedback is not wired, aqueous scavenging
  needs a mechanism, ...);
* a key WOOF does not carry is consumed when it provably changes nothing:
  at its WRF Registry default (the same run as omitting it), a process
  switch that is off, a control read only by a process the namelist turned
  off, or an output/file-naming control; otherwise it is refused by name;
* a key WRF v4.7.1's Registry does not declare is the fork's
  (:data:`FORK_KEYS`) or unknown, and is refused unless classified.

The Registry vocabulary is ``gpuwm/data/wrf_namelist/registry_chem_v4.7.1
.json`` (tools/wrf_namelist_registry.py --chem, WRF v4.7.1 Registry.EM_CHEM).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from gpuwm.namelist_contract import _registry_value

REGISTRY_TABLE = (Path(__file__).resolve().parent / "data" / "wrf_namelist"
                  / "registry_chem_v4.7.1.json")

#: The operational WRF 3.9 fork's chem_opt values with a WOOF set, accepted
#: only when the namelist carries the fork's own keys
#: (gpuwm.namelist_import.operational_fork_signature).  Stock WRF v4.7.1
#: declares no chem_opt = 18 package.
FORK_CHEM_OPT_SETS = {18: "smoke"}

#: ``&chem`` keys translated 1:1 onto RunConfig fields of the same name
#: (gpuwm/config.py, the chem block), with the Python type the field holds.
CARRIED_KEYS: dict[str, type] = {
    "chemdt": float, "kemit": int, "biomass_burn_opt": int,
    "plumerisefire_frq": int, "dust_opt": int, "seas_opt": int,
    "dmsemis_opt": int, "wetscav_onoff": int, "chem_conv_tr": int,
    "vertmix_onoff": int, "aer_ra_feedback": int, "aer_op_opt": int,
    "dust_alpha": float, "dust_gamma": float, "dust_smtune": float,
    "dust_ustune": float, "mynn_chem_vertmx": bool,
}

#: ``&dynamics`` switches of the chem array, carried under the same names.
CARRIED_DYNAMICS_KEYS = ("chem_adv_opt", "chem_mix2_off", "chem_mix6_off")

#: Every RunConfig field a WRF namelist can now spell through this door.
CHEM_DOOR_RUN_FIELDS = frozenset(CARRIED_KEYS) | frozenset(CARRIED_DYNAMICS_KEYS)

#: Output and file-naming controls: they decide what WRF-Chem writes or what
#: its input files are called, never what the chemistry computes.  WOOF reads
#: its own sources and writes its own chem history.
OUTPUT_KEYS = frozenset({
    "chemdiag", "diagnostic_chem", "diagnostic_dep", "opt_pars_out",
    "aero_diag_opt", "aero_cw_diag_opt", "aero_srf_area_diag", "kfcup_diag",
    "mozart_ph_diag", "io_style_emissions", "emi_inname", "emi_outname",
    "fireemi_inname", "fireemi_outname", "input_chem_inname",
    "input_chem_outname", "exo_coldens_inname", "wes_seasonal_inname",
    "trop_lev_inname", "fixed_ubc_inname", "track_chem_num",
    "track_chem_name", "track_rad_num", "track_tuv_num", "track_tuv_lev",
})

#: Process switches WOOF's chemistry has no counterpart for.  0 is "off",
#: which is exact; any other value asks for a process WOOF does not run.
#: Each value is what the switch turns on.
UNCARRIED_PROCESS_SWITCHES = {
    "emiss_opt": "anthropogenic emission inventory input",
    "emiss_opt_vol": "volcanic emissions",
    "aircraft_emiss_opt": "aircraft emissions",
    "bio_emiss_opt": "biogenic emissions",
    "phot_opt": "photolysis",
    "gas_bc_opt": "gas-phase lateral boundary profiles",
    "gas_ic_opt": "gas-phase initial profiles",
    "aer_bc_opt": "aerosol lateral boundary profiles",
    "aer_ic_opt": "aerosol initial profiles",
    "gas_drydep_opt": "gas dry deposition",
    "aer_drydep_opt": "aerosol dry deposition",
    "cldchem_onoff": "cloud chemistry",
    "conv_tr_wetscav": "wet scavenging in convective transport",
    "conv_tr_aqchem": "aqueous chemistry in convective transport",
    "dustwd_onoff": "dust wet deposition",
    "lnox_opt": "lightning NOx",
    "irr_opt": "integrated reaction rates",
    "n2o5_hetchem": "N2O5 heterogeneous chemistry",
    "nh3emis_opt": "NH3 emissions",
    "biomass_emiss_opt": "biomass-burning emission input",
}

#: Controls read only by one process; when that process is off in the
#: namelist the control is inert whatever its value.  key -> (switch, off).
READ_ONLY_UNDER = {
    "bioemdt": ("bio_emiss_opt", 0), "ne_area": ("bio_emiss_opt", 0),
    "nmegan": ("bio_emiss_opt", 0), "photdt": ("phot_opt", 0),
    "kemit_aircraft": ("aircraft_emiss_opt", 0),
    "emiss_inpt_opt": ("emiss_opt", 0), "emiss_ash_hgt": ("emiss_opt_vol", 0),
    "conv_tr_wetscav": ("chem_conv_tr", 0),
    "conv_tr_aqchem": ("chem_conv_tr", 0),
}

#: Keys of the operational WRF 3.9 fork's &chem that v4.7.1 does not
#: declare.  value -> what the key does; the decision is in
#: :func:`_fork_key`.
FORK_KEYS = {
    "simple_dir_fdb": "direct radiative feedback of the smoke",
    "simple_ind_fdb": "indirect (cloud) feedback of the smoke",
    "plumerise_flag": "plume-rise fire properties (2: fire radiative power)",
    "bb_dcycle": "diurnal cycle of the fire emissions",
    "flam_part": "flaming fraction of the fire emissions",
    "enh_vermix": "enhanced vertical mixing of the smoke",
    "debug_chem": "chemistry debug printing",
}


#: The operational fork's keys carried 1:1 onto RunConfig fields of the same
#: name (gpuwm/config.py), with the Python type the field holds:
#: simple_dir_fdb (smoke AOD in the aer_opt = 3 shortwave), enh_vermix
#: (mixing over burning columns, gpuwm/core/chem_vertmx.py) and flam_part
#: (the plume's flaming fraction, gpuwm/core/chem_plumerise.py).  The
#: fork's aer_drydep_opt = 111 (smoke deposition velocity) is carried the
#: same way; every other aer_drydep_opt keeps the stock switch handling.
FORK_CARRIED_KEYS = {"simple_dir_fdb": int, "enh_vermix": bool,
                     "flam_part": float}


#: The other way through every refusal of a chemistry companion key.
CHEM_OFF_WAY_OUT = ("  (Or set chem_opt = 0 to import the meteorology "
                    "without chemistry.)")


class ChemImport:
    """What the door translated: ``shared`` lines and the set chosen."""

    def __init__(self, chem_set: str, shared: dict[str, object],
                 sources: tuple[str, ...] = ()):
        self.chem_set = chem_set
        self.shared = shared
        self.sources = sources

    def toml_lines(self) -> list[str]:
        from gpuwm.namelist_import import _fmt

        lines = ["# WRF-Chem &chem, mapped onto WOOF's chemistry "
                 "(gpuwm/namelist_chem.py).",
                 f'chem_sets = "{self.chem_set}"']
        if self.sources:
            lines.append(f'chem_sources = "{",".join(self.sources)}"')
        for key, value in self.shared.items():
            lines.append(f"{key} = {_fmt(value)}")
        return lines


@lru_cache(maxsize=1)
def registry_keys() -> dict[str, dict]:
    table = json.loads(REGISTRY_TABLE.read_text(encoding="utf-8"))
    return {row["key"]: row for row in table["keys"]}


def _equal(value, default) -> bool:
    if isinstance(default, bool) or isinstance(value, bool):
        return value is default if isinstance(value, bool) else False
    if isinstance(default, (int, float)) and isinstance(value, (int, float)):
        return float(value) == float(default)
    return str(value).strip() == str(default).strip()


def carried_menu() -> str:
    from gpuwm.chem_table import catalog

    rows = [f"chem_opt = {row.wrf_chem_opt} ({row.name})"
            for row in catalog().sets.values()
            if row.wrf_chem_opt is not None and not row.test_only]
    rows += [f"the operational fork's chem_opt = {value} ({name})"
             for value, name in FORK_CHEM_OPT_SETS.items()]
    return ", ".join(sorted(rows))


def chem_set_for(chem_opt: int, *, fork: bool) -> str | None:
    from gpuwm.chem_table import catalog

    for row in catalog().sets.values():
        if row.wrf_chem_opt == chem_opt and not row.test_only:
            return row.name
    if fork:
        return FORK_CHEM_OPT_SETS.get(chem_opt)
    return None


def import_chem_section(entries: dict, *, max_dom: int, fork: bool,
                        drop, fix, error) -> ChemImport:
    """Map an active ``&chem`` group (``chem_opt != 0`` on some domain).

    ``error(section, key, value, why)`` builds the keyed refusal; every
    refusal is raised, the importer's all-refusals loop lists them all.
    """
    def column(key):
        values = entries[key]
        values = values if isinstance(values, list) else [values]
        return values, values[:max_dom]

    values, active = column("chem_opt")
    if len(set(active)) != 1:
        raise error("chem", "chem_opt", values,
                    "WOOF's chemistry is one species set for the whole "
                    "domain tree, and this namelist asks for "
                    f"{sorted(set(active))} on its {max_dom} domains: a "
                    "nest would carry chemistry its namelist turned off, or "
                    "the reverse.  Use one chem_opt on every domain.")
    chem_opt = active[0]
    if isinstance(chem_opt, bool) or not isinstance(chem_opt, int):
        raise error("chem", "chem_opt", values, "must be an integer")
    chem_set = chem_set_for(chem_opt, fork=fork)
    if chem_set is None:
        raise error(
            "chem", "chem_opt", values,
            f"WRF-Chem package chem_opt = {chem_opt} is not carried: WOOF's "
            "chemistry maps " + carried_menu() + ".  Imported anyway, the "
            "forecast would run with none of the requested species "
            "(emission, transport, deposition and any feedback all absent). "
            "Next: pick one of those packages, or set chem_opt = 0 to run "
            "the meteorology without chemistry.")
    from gpuwm.chem_table import catalog

    fix("chem", "chem_opt", values, chem_set,
        f"WRF-Chem package chem_opt = {chem_opt} runs as WOOF's "
        f"'{chem_set}' species set (gpuwm/data/chem/sets/{chem_set}.json: "
        f"{catalog().sets[chem_set].description}).  Its emissions, start "
        "and boundary chemistry come from that set's own source rows "
        "(chem_sources), not from WRF-Chem input files")

    registry = registry_keys()
    shared: dict[str, object] = {}
    switches = {key: column(key)[0][0] for key in entries}
    for key in sorted(entries):
        if key == "chem_opt":
            continue
        values, active = column(key)
        if key in CARRIED_KEYS:
            if len(set(map(repr, active))) != 1:
                raise error("chem", key, values,
                            "WOOF carries one value for the whole tree; "
                            f"this namelist gives {active} on its domains")
            value = active[0]
            kind = CARRIED_KEYS[key]
            if kind is bool:
                if not isinstance(value, bool):
                    raise error("chem", key, values,
                                "must be a Fortran logical")
            elif kind is int:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise error("chem", key, values, "must be an integer")
            else:
                if isinstance(value, bool) or not isinstance(
                        value, (int, float)):
                    raise error("chem", key, values, "must be a number")
                value = float(value)
            shared[key] = value
            continue
        if key in SOURCE_KEYS and any(value not in (0, False)
                                      for value in active):
            sourced = _set_rows_with_sources(chem_set)
            if sourced:
                fix("chem", key, values, "source rows",
                    f"{SOURCE_KEYS[key]} from the set's own source rows ("
                    + ", ".join(sourced) + "), which WOOF reads at the "
                    "start and at every boundary update")
                continue
            raise error("chem", key, values,
                        f"{SOURCE_KEYS[key]} requested, and no row of "
                        f"'{chem_set}' carries a source to take it from: the "
                        "chemistry would start from zero and flow in at its "
                        "default inflow.  Next: set it to 0/.false.")
        if key in OUTPUT_KEYS:
            drop("chem", key, values,
                 "WRF-Chem output or file-naming control; WOOF reads its own "
                 "chem sources and writes its own chem history, so the "
                 "chemistry computed is the same")
            continue
        if key in READ_ONLY_UNDER:
            switch, off = READ_ONLY_UNDER[key]
            if switches.get(switch, _registry_default(switch)) == off:
                drop("chem", key, values,
                     f"read only when {switch} != {off}, and {switch} is "
                     f"{off} here: inert")
                continue
        if (key == "aer_drydep_opt" and fork
                and all(v == 111 for v in active)):
            # The fork's smoke deposition velocity (smoke/dry_dep_driver.F
            # :244-256): RunConfig.aer_drydep_opt = 111, gpuwm/core/
            # chem_vertmx.py.
            shared[key] = 111
            continue
        if key in UNCARRIED_PROCESS_SWITCHES:
            # Decided on the switch, never on the Registry default: several
            # defaults are 1 (process ON), so "at the default" is a request.
            what = UNCARRIED_PROCESS_SWITCHES[key]
            process = PROCESS_SWITCH_CARRIERS.get(key)
            runs = process is not None and any(
                process in row.processes for row in _set_rows(chem_set))
            off = all(value == 0 for value in active)
            if off and not runs:
                drop("chem", key, values,
                     f"{what} is off (0): exact, '{chem_set}' runs no such "
                     "process")
                continue
            if runs and all(value == 1 for value in active):
                fix("chem", key, values, 1,
                    f"{what}: '{chem_set}' runs it ({process}, "
                    f"gpuwm/{CHEM_PROCESS_PATH.get(process, 'core')})")
                continue
            raise error(
                "chem", key, values,
                (f"{what} is always on in '{chem_set}' ({process}); WOOF "
                 "cannot turn it off, so the chemistry would deposit what "
                 "the request keeps.  Next: set it to 1."
                 if off else
                 f"asks for {what}, which '{chem_set}' does not run"
                 + (f" in this variant (only 1, {process}, is carried)"
                    if runs else "")
                 + ": the value would not be honoured.  Next: set it to "
                 + ("1." if runs else "0 to run without it.")
                 + CHEM_OFF_WAY_OUT))
        carried_fork = FORK_CARRIED_KEYS.get(key)
        # At the off value (0 / .false.) the fork key keeps its exact drop
        # below, so an import that turns it off writes nothing new.
        if fork and (carried_fork is not None
                     and not all(v is False or (v == 0 and key != "flam_part")
                                 for v in active)):
            if len(set(map(repr, active))) != 1:
                raise error("chem", key, values,
                            "WOOF carries one value for the whole tree; "
                            f"this namelist gives {active} on its domains")
            value = active[0]
            kind = carried_fork
            if kind is bool:
                if not isinstance(value, bool):
                    raise error("chem", key, values, "must be a Fortran logical")
            elif kind is int:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise error("chem", key, values, "must be an integer")
            else:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise error("chem", key, values, "must be a number")
                value = float(value)
            # Carried as RunConfig fields of the same name, each read by the
            # forecast and checked by validate_chem_config.
            shared[key] = value
            continue
        if key in FORK_KEYS and key not in registry:
            if not fork:
                raise error("chem", key, values,
                            "not a WRF v4.7.1 &chem key (it is the "
                            "operational WRF 3.9 fork's, read only beside "
                            "that fork's own keys)")
            _fork_key(key, values, active, drop=drop, fix=fix, error=error)
            continue
        row = registry.get(key)
        if row is None:
            raise error("chem", key, values,
                        "not a WRF v4.7.1 &chem key (Registry.EM_CHEM), and "
                        "WOOF's chemistry has no counterpart for it")
        default = _registry_value(row)
        if all(_equal(value, default) for value in active):
            drop("chem", key, values,
                 f"at its WRF Registry default ({row['default']}), the same "
                 "run as omitting it")
            continue
        raise error("chem", key, values,
                    "WOOF's chemistry does not carry this control (WRF "
                    f"Registry default {row['default']}): the value would "
                    "not be honoured.  Next: set it to the default or "
                    "remove it.")
    return ChemImport(chem_set, shared, _sources_for(chem_set, switches))


#: Process switches WOOF does carry for some sets: the WOOF process that
#: answers the switch's standard value (1), when the mapped set's rows run it.
PROCESS_SWITCH_CARRIERS = {"gas_drydep_opt": "drydep.wesely",
                           "aer_drydep_opt": "drydep.gocart"}
CHEM_PROCESS_PATH = {"drydep.wesely": "core/chem_drydep_gas.py",
                     "drydep.gocart": "core/chem_drydep_aerosol.py"}


def _set_rows(chem_set: str):
    from gpuwm.chem_table import catalog

    table = catalog()
    return [table.species[name] for name in table.set_members(chem_set)]


#: Controls that take chemistry from input files; WOOF takes it from the
#: species rows' declared sources (gpuwm/data/chem/sources).
SOURCE_KEYS = {
    "chem_in_opt": "initial chemistry from the input",
    "have_bcs_chem": "lateral boundary chemistry from the input",
}


def _set_rows_with_sources(chem_set: str) -> list[str]:
    """Source names the set's rows take start and boundary values from,
    or [] when any row has none (then the request cannot be met)."""
    rows = _set_rows(chem_set)
    if not rows or any(not row.boundary for row in rows):
        return []
    return sorted({entry["source"] for row in rows for entry in row.boundary})


#: WOOF chem processes fed by an emission source row, and the WRF switch
#: that asks for that emission (biomass burning; the anthropogenic
#: inventory).  A row's emission sources are enabled when the namelist's
#: switch is on; with it off WRF emits nothing, and the chem law refuses a
#: row that would then have no source.
EMISSION_SWITCHES = {"emission.fire": "biomass_burn_opt",
                     "emission.inventory": "emiss_opt"}


def _sources_for(chem_set: str, switches: dict) -> tuple[str, ...]:
    """``chem_sources`` for the mapped set, from the rows' own declarations:
    the emission sources of every process the namelist switched on, the
    start and boundary sources when it asked for input chemistry, and the
    oxidant sources the sulfur mechanism reads (WRF-Chem's GOCART reads its
    background oxidants from its own input; WOOF from these rows)."""
    from gpuwm.chem_table import catalog

    table = catalog()
    rows = _set_rows(chem_set)
    chosen: list[str] = []

    def add(name):
        if name not in chosen:
            chosen.append(name)

    for row in rows:
        for process, switch in EMISSION_SWITCHES.items():
            if process in row.processes and switches.get(switch) not in (
                    None, 0):
                for entry in row.emissions:
                    add(entry["source"])
    if any(switches.get(key) not in (None, 0, False) for key in SOURCE_KEYS):
        for row in rows:
            for entry in row.boundary:
                add(entry["source"])
    if any("chem.sulfur" in row.processes for row in rows):
        for name, source in sorted(table.sources.items()):
            if source.kind == "oxidant":
                add(name)
    return tuple(chosen)


def _registry_default(key: str):
    row = registry_keys().get(key)
    return None if row is None else _registry_value(row)


def _fork_key(key, values, active, *, drop, fix, error) -> None:
    if key == "debug_chem":
        drop("chem", key, values, "chemistry debug printing; WOOF writes its "
                                  "own run log")
        return
    if key in ("simple_dir_fdb", "simple_ind_fdb"):
        if all(v == 0 for v in active):
            drop("chem", key, values, f"{FORK_KEYS[key]} is off: exact")
            return
        raise error("chem", key, values,
                    f"{FORK_KEYS[key]} requested; the fork's empirical "
                    "simple feedback is not the prognostic chemistry optics "
                    "route. Radiation feedback is aer_ra_feedback=1 with "
                    "optics.gocart rows; indirect cloud feedback uses its "
                    "separate aerosol_mp_coupling route. Next: set "
                    f"{key} = 0." + CHEM_OFF_WAY_OUT)
    if key == "plumerise_flag":
        if all(v == 2 for v in active):
            fix("chem", key, values, "frp",
                "fire-radiative-power plume rise: WOOF's "
                "plume_fire_properties = 'frp', HRRR-Smoke's method")
            return
        raise error("chem", key, values,
                    "only 2 (fire radiative power) is carried; other fire "
                    "properties would need per-column flaming fractions and "
                    "fire sizes no source supplies")
    if key == "bb_dcycle":
        if all(v is True for v in active):
            fix("chem", key, values, "trailing_24h_dcycle",
                "fire emissions with a diurnal cycle: WOOF's default "
                "fire_emission_mode = 'trailing_24h_dcycle', HRRR-Smoke's "
                "method")
            return
        raise error("chem", key, values,
                    "fire emissions without a diurnal cycle are not carried; "
                    "WOOF's fire emission modes all apply one")
    if key == "enh_vermix":
        if all(v is False for v in active):
            drop("chem", key, values, f"{FORK_KEYS[key]} is off: exact")
            return
        raise error("chem", key, values,
                    f"{FORK_KEYS[key]} requested; WOOF's smoke mixes with "
                    "WRF-Chem's vertmx (mixing.vertmx) and carries no such "
                    "enhancement, so the requested mixing would not be "
                    "honoured.  Next: set enh_vermix = .false."
                    + CHEM_OFF_WAY_OUT)
    if key == "flam_part":
        raise error("chem", key, values,
                    "a flaming fraction for the fire emissions; WOOF's FRP "
                    "plume rise has no such control, so the value would not "
                    "be honoured.  Next: remove flam_part."
                    + CHEM_OFF_WAY_OUT)

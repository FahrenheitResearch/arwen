"""Phase-5 Task 1 byte-identity freeze pins (architecture section A).

``gpuwm/config.py`` carries the Phase-5 fields plus WRF-default
``top_lid = False`` and attribution-only ``moist_cq = True``;
``load_config()`` and the legacy
``[grid]``/``[dynamics]``/``[run]`` tables are untouched.  These pins
prove the guarantee: every existing legacy TOML under ``configs/`` and
every frozen-case constructed RunConfig re-resolves IDENTICALLY --
``dataclasses.asdict`` compared field-for-field against the golden
snapshot captured from the pre-change code plus explicitly reviewed new
configuration fields (``tests/data/config_freeze_golden.json``).

Freeze discipline: a legacy TOML added to ``configs/`` without a golden
entry FAILS here -- new files are pinned consciously, never implicitly.
Experiment-shaped TOMLs ([experiment]/[[domain]]) are exercised by
tests/test_experiment.py instead.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from gpuwm.config import RunConfig, load_config, _KNOWN_TABLES
from gpuwm.experiment import is_experiment_toml

REPO = Path(__file__).resolve().parents[1]
GOLDEN = json.loads(
    (Path(__file__).parent / "data" / "config_freeze_golden.json")
    .read_text())


def _frozen_constructors():
    from gpuwm.verify.cases import (hill2d, igw, moist_bubble, real74_d01,
                                    straka, wk82)
    return {
        "straka.default_config": straka.default_config,
        "igw.default_config": igw.default_config,
        "hill2d.default_config": hill2d.default_config,
        "moist_bubble.default_config": moist_bubble.default_config,
        "wk82.default_config": wk82.default_config,
        "real74_d01.config": real74_d01.config,
        "real74_d01.phase3_config": real74_d01.phase3_config,
    }


def test_new_fields_are_reviewed_defaults_appended_last():
    """New fields remain appended, preserving positional construction."""
    names = [f.name for f in dataclasses.fields(RunConfig)]
    # RE-BASELINED (lane/wif-climatology + lane/wif-default): 95 -> 97.
    # TWO fields were appended, ``wif_climatology_path`` and
    # ``mp28_aerosol_source``, and the second of them is last, which is the
    # property this test exists to hold.  Reconstructing the old assertion
    # from the new: names[-97:-2] is exactly the list this window held
    # before.
    assert names[-1] == "mp28_aerosol_source"
    assert names[-97:] == [
        "nested", "grid_id", "top_lid", "moist_cq", "morr_rimed_ice",
        "wsm6_hail_opt", "ra_lw_physics", "ra_sw_physics", "icloud",
        "swrad_scat", "wrf_rrtmg_compatibility", "num_soil_layers",
        "ra_rrtmg_variant",
        "nest_microphysics_transition", "isftcflx", "iz0tlnd",
        "usemonalb", "rdlai2d", "opt_thcnd",
        "bl_mynn_closure", "bl_mynn_cloudpdf", "bl_mynn_mixlength",
        "bl_mynn_edmf", "bl_mynn_edmf_mom", "bl_mynn_edmf_tke",
        "bl_mynn_mixscalars", "bl_mynn_cloudmix", "bl_mynn_mixqt",
        "bl_mynn_output", "bl_mynn_tkeadvect", "icloud_bl",
        "dveg", "opt_crs", "opt_btr", "opt_run", "opt_sfc", "opt_frz",
        "opt_inf", "opt_rad", "opt_alb", "opt_snf", "opt_tbot", "opt_stc",
        "opt_gla", "opt_rsf", "opt_soil", "opt_pedo", "opt_crop", "opt_irr",
        "opt_irrm", "opt_infdv", "opt_tdrn", "soiltstep", "noahmp_output",
        "noahmp_acc_dt",
        "mosaic_lu", "mosaic_soil", "flag_sm_adj", "spp_lsm",
        "nwp_diagnostics", "isfflx", "o3input", "use_mp_re",
        "seaice_albedo_default", "rdmaxalb",
        # km_opt=3 LES closure knobs (change record: appended with the 3-D
        # Smagorinsky port; WRF Registry defaults, frozen paths unchanged).
        "mix_isotropic", "mix_upper_bound", "tke_heat_flux",
        "tke_drag_coefficient", "c_k",
        # km_opt=2 lateral-boundary/bound_tke arm and the report-only
        # per-step TKE budget toggle (change record: appended with the
        # prognostic-TKE restart + boundary work; WRF Registry default
        # tke_upper_bound = 1000., budget off).
        "tke_upper_bound", "tke_budget",
        # Aerosol-aware Thompson (mp_physics=28) aerosol-source selectors
        # (change record: appended with the mp=28 merge onto this line).
        "aer_init_opt", "wif_input_opt",
        # The SASE closure's three knobs, appended last exactly as this
        # discipline requires.  Each is fail-closed on its NON-default
        # value only, so every configuration that predates them resolves
        # unchanged -- which is what the golden below re-checks.
        "sase_flux_diag", "sase_moist_n2", "sase_stable_dissipation",
        "sase_additive_dissipation",
        # The horizontal-mixing pair, appended on the same terms.  Both
        # default to the value the tree already ran: hmix_k_diag = False
        # publishes nothing, and an EMPTY acknowledgement is the state
        # every existing configuration is in, so the km_opt gate refuses
        # exactly what it refused before.
        "hmix_k_diag", "km_opt_zero_acknowledgement",
        # The LES-nest inflow seeding keys (change record: appended with
        # the P3 generator, INFLOW-GENERATOR-ACCEPTANCE-V2).  All four
        # default to the mechanism-off state every existing
        # configuration is in -- inflow_perturbation = False executes
        # nothing -- so every pre-change TOML and constructor resolves
        # unchanged, which the goldens below re-check with the four
        # reviewed defaults added.
        "inflow_perturbation", "inflow_perturbation_seed",
        "inflow_perturbation_amplitude_scale", "inflow_perturbation_faces",
        # WRF's own &dynamics moist_mix6_off, appended on the same terms.
        # Its default is WRF's Registry default (.false.,
        # Registry.EM_COMMON:2889), it is read only where diff_6th_opt > 0,
        # and at False the diff6 row set is the one the frozen trajectories
        # ran -- so it cannot move any of them.  It is divergence-ledger
        # entry L4 (gpuwm/physics_mode.py).
        "moist_mix6_off",
        # The Grell-family keys, appended last: read only where
        # cu_physics = 3, which no frozen configuration selects, and
        # both defaults are WRF's own (Registry.EM_COMMON:2544,2546).
        "clos_choice", "ishallow",
        # The NSSL variant selectors: one WRF scheme, four
        # flags on top of it (Registry.EM_COMMON:2420-2425).
        "nssl_2moment_on", "nssl_hail_on", "nssl_ccn_on",
        "nssl_density_on", "nssl_3moment",
        # The WDM6 pair, appended after them at the 1.9 assembly.  Both defaults are WRF's own
        # (hail_opt = 0, Registry.EM_COMMON:2665; ccn_conc = 1.0e8,
        # :2664) and both are read only where mp_physics = 16, which no
        # frozen configuration selects.  They are additionally dropped
        # from the restart identity of every run that is not WDM6
        # (gpuwm.core.model.restart_identity_payload), so appending them
        # moves neither a trajectory nor a fingerprint.
        "wdm6_hail_opt", "wdm6_ccn_conc",
        # The surface-radiation carrier policy, appended last.  A LABEL on
        # a guard rather than a physics knob: the default "required" is the
        # behaviour every frozen trajectory already had once the contract
        # is in place -- carriers those runs consume all have producers --
        # and the only other value is the declared escape, which a frozen
        # case does not select.  It moves no trajectory and it is not a WRF
        # namelist key, because WRF has no carrier provenance to declare.
        "surface_radiation_policy",
        # The mp=28 WIF climatology dataset path (change record: appended
        # with the lane/wif-climatology commit onto this line).  The empty
        # default runs not one new instruction, and
        # validate_aerosol_source_options refuses a set path whose
        # selectors (aer_init_opt=1 + wif_input_opt=1) do not consume it,
        # so no frozen trajectory can move.
        "wif_climatology_path",
        # Which mp=28 aerosol initial state the run wants (change record:
        # appended with the lane/wif-default commit onto this line).  At
        # "auto" a real-data mp=28 run resolves WRF's monthly WIF
        # climatology and announces the synthetic fallback by name when it
        # cannot; every frozen case here is idealized or has no dataset, so
        # each resolves to the same synthetic profile it always did.
        "mp28_aerosol_source",
    ]
    # Aerosol-aware Thompson (mp_physics=28) aerosol-source selectors,
    # appended last.  Both defaults are WRF's own Registry defaults
    # (Registry/Registry.EM_COMMON:2656 and
    # Registry/registry.new3d_wif:17), and both are the ONLY value
    # gpuwm.config.validate_aerosol_source_options admits -- so they cannot
    # move any trajectory, frozen or otherwise: there is no second value to
    # move it to.  They exist so that a request for WRF's climo/first-guess
    # aerosol IC/BC or its WIF metgrid stream is refused BY NAME rather than
    # being unrepresentable and therefore silently ignored.  Matching WRF's
    # defaults is load-bearing for a second reason: the prepared-forecast
    # runner compares physics_compat._SINGLE_DOMAIN_RUNTIME_SWITCHES rows
    # for exact equality, and a nonzero default would change every shipped
    # profile.
    assert RunConfig.__dataclass_fields__["aer_init_opt"].default == 0
    assert RunConfig.__dataclass_fields__["wif_input_opt"].default == 0

    # WIF climatology dataset path: the empty default is "resolve it, and
    # say so if you cannot", which is what makes the field inert for every
    # frozen case here -- none of them ships a dataset.
    assert RunConfig.__dataclass_fields__[
        "wif_climatology_path"].default == ""
    # And the selector's default, asserted for the same reason the pair
    # above is: "auto" is the value that reproduces the pre-lane behaviour
    # whenever no dataset resolves, so a silent change to it would move
    # every mp=28 real-data trajectory while this file stayed green.
    assert RunConfig.__dataclass_fields__[
        "mp28_aerosol_source"].default == "auto"

    assert RunConfig.__dataclass_fields__["hmix_k_diag"].default is False
    # WRF v4.6.1 Registry.EM_COMMON:2889 declares moist_mix6_off .false.,
    # and matching WRF's default is what makes the field inert: the moist
    # array keeps its 6th-order filter exactly as every frozen trajectory
    # ran it.
    assert RunConfig.__dataclass_fields__["moist_mix6_off"].default is False
    assert RunConfig.__dataclass_fields__[
        "km_opt_zero_acknowledgement"].default == ""
    # Grell-family keys, WRF v4.6.1 Registry defaults
    # (Registry.EM_COMMON:2544,2546).  Both are read only where
    # cu_physics=3, which no frozen configuration selects, so appending
    # them cannot move a frozen trajectory; validate_run_config refuses
    # nonzero values without the scheme.
    assert RunConfig.__dataclass_fields__["clos_choice"].default == 0
    assert RunConfig.__dataclass_fields__["ishallow"].default == 0
    # The carrier policy defaults to the strict value, not the escape:
    # a run that says nothing gets the guard, and a run that wants
    # pre-1.9 behaviour has to type the word.
    assert RunConfig.__dataclass_fields__[
        "surface_radiation_policy"].default == "required"
    # Noah option selectors. Each default reproduces the value the launcher
    # previously pinned, so exposing them cannot move a frozen trajectory.
    assert RunConfig.__dataclass_fields__["usemonalb"].default is False
    assert RunConfig.__dataclass_fields__["rdlai2d"].default is False
    assert RunConfig.__dataclass_fields__["opt_thcnd"].default == 1
    assert RunConfig.__dataclass_fields__["nested"].default is False
    assert RunConfig.__dataclass_fields__["grid_id"].default == 1
    # Stability defaults (2026-07-18): rigid lid + no cq until the open-top
    # and cq production defects have stability + falsification receipts
    # (see config.py field comments and the iso-* probe records).
    assert RunConfig.__dataclass_fields__["top_lid"].default is True
    assert RunConfig.__dataclass_fields__["moist_cq"].default is False
    # Morrison's scalar Registry default is hail; explicit 0 retains the
    # graupel branch (Registry.EM_COMMON:2663-2666).
    assert RunConfig.__dataclass_fields__["morr_rimed_ice"].default == 1
    # WSM6 uses a distinct switch and the opposite WRF default: graupel.
    assert RunConfig.__dataclass_fields__["wsm6_hail_opt"].default == 0
    assert RunConfig.__dataclass_fields__[
        "nest_microphysics_transition"].default == "same-scheme-only"
    assert RunConfig.__dataclass_fields__["ra_lw_physics"].default == -1
    assert RunConfig.__dataclass_fields__["ra_sw_physics"].default == -1
    assert RunConfig.__dataclass_fields__["icloud"].default == 1
    assert RunConfig.__dataclass_fields__["swrad_scat"].default == 1.0
    assert RunConfig.__dataclass_fields__["wrf_rrtmg_compatibility"].default \
        == "none"
    assert RunConfig.__dataclass_fields__["num_soil_layers"].default == 4
    # 4/4 stays on the modern RTE+RRTMGP adapter unless the exact
    # legacy-RRTMG port is selected explicitly (which fails closed until
    # its compute kernels land).
    assert RunConfig.__dataclass_fields__["ra_rrtmg_variant"].default \
        == "rte-rrtmgp"
    # nwp_diagnostics: WRF's own Registry default is 0 (off), and at 0 the
    # dycore epilogue never launches the UP_HELI_MAX kernel, so every frozen
    # trajectory is unchanged.  At 1 the diagnostic is trajectory-inert by
    # test (tests/test_uh_lifecycle.py).
    assert RunConfig.__dataclass_fields__["nwp_diagnostics"].default == 0
    assert RunConfig.__dataclass_fields__["isfflx"].default == 1
    # km_opt=3 knobs: WRF Registry defaults (mix_isotropic 0, upper bound
    # 0.1, prescribed fluxes 0), consumed only under km_opt=3 or the
    # PBL-off isfflx=0/2 paths, so frozen trajectories cannot move.
    assert RunConfig.__dataclass_fields__["mix_isotropic"].default == 0
    assert RunConfig.__dataclass_fields__["mix_upper_bound"].default == 0.1
    assert RunConfig.__dataclass_fields__["tke_heat_flux"].default == 0.0
    assert RunConfig.__dataclass_fields__[
        "tke_drag_coefficient"].default == 0.0
    assert RunConfig.__dataclass_fields__["c_k"].default == 0.15
    assert RunConfig.__dataclass_fields__["o3input"].default == 2
    assert RunConfig.__dataclass_fields__["use_mp_re"].default == 1
    assert RunConfig.__dataclass_fields__[
        "seaice_albedo_default"].default == 0.65
    assert RunConfig.__dataclass_fields__["rdmaxalb"].default is True
    # MM5 surface-layer options.  Both WRF Registry defaults are 0, and both
    # are inert at 0: the surface-layer kernel's isftcflx/iz0tlnd branches
    # are skipped entirely, so every frozen trajectory is unchanged.
    assert RunConfig.__dataclass_fields__["isftcflx"].default == 0
    assert RunConfig.__dataclass_fields__["iz0tlnd"].default == 0
    # MYNN PBL option identity.  Every default is the single value the ported
    # solver was validated at, and none of them is read unless
    # bl_pbl_physics=5, which no frozen configuration selects -- so exposing
    # them cannot move a frozen trajectory.  validate_run_config refuses any
    # other value, so these are pins rather than choices.
    from gpuwm.config import MYNN_PBL_OPTION_IDENTITY
    for name, admitted in MYNN_PBL_OPTION_IDENTITY.items():
        default = RunConfig.__dataclass_fields__[name].default
        assert default == admitted and type(default) is type(admitted), name
    # Noah-MP option identity, on exactly the same terms: every default is
    # the single admitted value, validate_run_config refuses any other, and
    # none is read unless sf_surface_physics=4, which no frozen configuration
    # selects -- real74_d01 is Noah (2).
    from gpuwm.config import NOAHMP_OPTION_IDENTITY
    for name, admitted in NOAHMP_OPTION_IDENTITY.items():
        default = RunConfig.__dataclass_fields__[name].default
        assert default == admitted and type(default) is type(admitted), name
    # RUC option identity, on exactly the same terms again.  All four defaults
    # are 0, validate_run_config refuses any other value, and none is read
    # unless sf_surface_physics=3 -- which no frozen configuration selects, so
    # appending them cannot move a frozen trajectory.  Unlike the MYNN and
    # Noah-MP blocks these are also WRF's own Registry defaults
    # (Registry.EM_COMMON:2535-2537 and Registry/registry.stoch:241), so a
    # namelist import of a stock ARW run lands on them without adjustment.
    from gpuwm.config import RUC_OPTION_IDENTITY
    for name, admitted in RUC_OPTION_IDENTITY.items():
        default = RunConfig.__dataclass_fields__[name].default
        assert default == admitted and type(default) is type(admitted), name
    for entry in GOLDEN.values():
        assert entry["sf_surface_physics"] not in (3, 4)
    # the golden snapshot carries exactly the current field set
    for key, entry in GOLDEN.items():
        assert set(entry) == set(names), key



def _legacy_run_config_tomls() -> list[Path]:
    """Every TOML in configs/ that IS a legacy RunConfig, and only those.

    "Not an experiment TOML" is not the same statement as "a legacy
    RunConfig TOML", and configs/ now holds a third kind: the
    high-resolution static demo carries only [demo] and [static.highres]
    and is read by tools/run_highres_static_demo.py.  ``load_config``
    refuses it outright -- unknown tables -- so it can never have a
    golden freeze entry, and demanding one was asking for a pin that
    cannot exist.

    The positive test is the loader's own table list rather than a
    hand-kept exclusion list, so a fourth kind of config in this
    directory does not silently join the frozen surface either.
    """

    return [path for path in sorted((REPO / "configs").glob("*.toml"))
            if not is_experiment_toml(path)
            and _declares_a_run_config_table(path)]


def _declares_a_run_config_table(path: Path) -> bool:
    import io
    import tomllib

    from gpuwm.config_authority import read_config_authority

    raw = tomllib.load(io.BytesIO(read_config_authority(path).payload))
    return any(table in raw for table in _KNOWN_TABLES)


def test_every_existing_legacy_toml_resolves_identically():
    """Re-resolve every legacy TOML in configs/ and compare dataclasses
    (as field dicts) against the pre-change golden snapshot."""
    tomls = sorted((REPO / "configs").glob("*.toml"))
    assert tomls, "configs/ directory is empty?"
    legacy = _legacy_run_config_tomls()
    assert legacy, "no legacy TOMLs found in configs/"
    for path in legacy:
        key = f"configs/{path.name}"
        assert key in GOLDEN, (
            f"{key} has no golden freeze entry: pin new legacy TOMLs in "
            "tests/data/config_freeze_golden.json consciously.")
        cfg = load_config(path)
        assert dataclasses.asdict(cfg) == GOLDEN[key], key
        assert cfg.nested is False and cfg.grid_id == 1, key


def test_frozen_case_constructed_configs_resolve_identically():
    """Every frozen verify-case RunConfig constructor re-resolves to the
    pre-change golden values (frozen cases never migrate to the
    experiment path -- they construct RunConfig directly)."""
    for key, ctor in _frozen_constructors().items():
        cfg = ctor()
        assert key in GOLDEN, key
        assert dataclasses.asdict(cfg) == GOLDEN[key], key
        assert cfg.nested is False and cfg.grid_id == 1, key


def test_golden_inventory_matches_known_frozen_surface():
    """The golden file covers exactly the frozen surface: every known
    constructor plus every legacy TOML (no stale or orphaned pins)."""
    expected = set(_frozen_constructors())
    expected |= {f"configs/{p.name}"
                 for p in _legacy_run_config_tomls()}
    assert set(GOLDEN) == expected

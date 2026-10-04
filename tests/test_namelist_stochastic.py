"""WRF &stoch controls: the importer translates them, and the import refuses them.

Two promises, tested apart:

* an active random-physics selector is refused at import with the
  calibration reason (its amplitude has not been measured against
  observations), so no imported config can carry one into a run;
* the translation itself still stands for the day calibration lands: the
  importer's ``&stoch`` reader turns representable controls into the typed
  provider's configuration, and names every control it cannot represent.
"""
from dataclasses import replace
import tomllib

import pytest

from gpuwm.ensemble.calibration_admission import UNCALIBRATED_SPREAD_REASON
from gpuwm.ensemble.stochastic_model import StochasticModelProvider
from gpuwm.experiment import build_experiment
from gpuwm.namelist_import import import_namelists
from test_namelist_import import INPUT_TEXT, _pair

MYNN = (("sf_sfclay_physics = 91, 91", "sf_sfclay_physics = 5, 5"),
        ("bl_pbl_physics = 11, 11", "bl_pbl_physics = 5, 5"))
SPPT = " sppt = 1, 1,\n nens = 19,\n iseed_sppt = -117,\n gridpt_stddev_sppt = .25, .25,\n timescale_sppt = 10800., 10800.,"
SKEBS = (" stoch_force_opt = 1, 1,\n tot_backscat_psi = 2.e-5, 2.e-5,\n ztau_t = 9000.,\n"
         " kminforc = 2,\n lminforc = 2,\n kmaxforc = 12,\n lmaxforc = 12,")
SKEBS_CLIP = " skebs = 1, 1,\n sppt = 0, 0,\n gridpt_stddev_sppt = .25, .25,\n stddev_cutoff_sppt = 3., 3.,"
SPP_PBL = " spp_pbl = 1, 1,\n gridpt_stddev_spp_pbl = .075, .075,"


def imported(tmp_path, controls, replacements=()):
    inp = INPUT_TEXT
    for before, after in replacements:
        assert before in inp
        inp = inp.replace(before, after)
    inp += "\n&stoch\n" + controls + "\n/\n"
    return import_namelists(*_pair(tmp_path, inp=inp))


def translated(controls, max_dom=2):
    """The importer's own &stoch reader, on the text a namelist would carry.

    Returns the provider controls, the per-domain SPP flags and the keys
    the reader dropped as inert.
    """
    from gpuwm.fortran_namelist import parse_namelist_text
    from gpuwm.namelist_import import _Section, _err
    from gpuwm.namelist_stochastic import import_stochastic_section
    entries = parse_namelist_text("&stoch\n" + controls + "\n/\n")["stoch"]
    section = _Section("stoch", entries, "stochastic namelist test")
    dropped = []
    controls, flags = import_stochastic_section(section, max_dom=max_dom,
        fix=lambda *args: None, drop=lambda *args: dropped.append(args[1]), error=_err)
    assert not section.entries
    return controls, flags, dropped


def mynn_experiment(tmp_path, flags):
    """The baseline import on MYNN physics, with the reader's SPP flags selected."""
    inp = INPUT_TEXT
    for before, after in MYNN:
        inp = inp.replace(before, after)
    text, _ = import_namelists(*_pair(tmp_path, inp=inp))
    exp = build_experiment(tomllib.loads(text), source="stochastic namelist test")
    return replace(exp, domains=tuple(
        replace(domain, run=replace(domain.run, **{name: column[index] for name, column in flags.items()}))
        for index, domain in enumerate(exp.domains)))


# ---- the import refuses every active selector ------------------------------

@pytest.mark.parametrize("controls, replacements", [
    (SPPT, ()), (SKEBS, ()), (SKEBS_CLIP, ()), (SPP_PBL, MYNN),
    (" spp_pbl = 0, 1,\n gridpt_stddev_spp_pbl = 0.9, .075,", MYNN),
    (" spp_pbl = 1, 0,\n gridpt_stddev_spp_pbl = .075, 0.9,", MYNN),
], ids=["sppt", "skebs-legacy-selector", "skebs", "spp-pbl", "spp-pbl-child", "spp-pbl-root"])
def test_an_active_stochastic_namelist_is_refused_with_the_calibration_reason(
        tmp_path, controls, replacements):
    with pytest.raises(ValueError) as caught:
        imported(tmp_path, controls, replacements)
    assert UNCALIBRATED_SPREAD_REASON in str(caught.value)


@pytest.mark.parametrize("selector", ["rand_perturb", "multi_perturb", "perturb_bdy", "pert_mynn", "pert_thom"])
def test_the_unsupported_selectors_give_the_same_reason_and_name_their_key(tmp_path, selector):
    """They used to read "consumer is not implemented": no breakage, no remedy."""
    value = ".true." if selector.startswith("pert_") else "1, 1"
    with pytest.raises(ValueError) as caught:
        imported(tmp_path, f" {selector} = {value},")
    message = str(caught.value)
    assert UNCALIBRATED_SPREAD_REASON in message
    assert message.startswith(f"&stoch {selector} = ")
    assert f"Next: turn {selector} off in &stoch" in message


def test_the_config_loader_refuses_an_authored_spp_switch(tmp_path):
    """The builder every config door loads through, not a per-door check."""
    text, _ = import_namelists(*_pair(tmp_path, inp=INPUT_TEXT.replace(*MYNN[0]).replace(*MYNN[1])))
    raw = tomllib.loads(text)
    build_experiment(tomllib.loads(text), source="unperturbed")
    raw["shared"]["spp_pbl"] = 1
    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="authored spp_pbl")
    assert str(caught.value) == UNCALIBRATED_SPREAD_REASON


# ---- the translation ---------------------------------------------------------

def test_sppt_controls_and_original_seed_labels_reach_the_provider():
    controls, flags, _ = translated(SPPT)
    provider = StochasticModelProvider.from_mapping(controls)
    assert provider.sppt.stddev == .25 and provider.sppt.timescale_s == 10800.
    assert provider.wrf_seed_labels["nens"] == 19 and provider.wrf_seed_labels["iseed_sppt"] == -117
    assert not any(any(column) for column in flags.values())


def test_skebs_legacy_selector_and_representable_limits():
    controls, _, _ = translated(SKEBS)
    provider = StochasticModelProvider.from_mapping(controls)
    assert provider.skebs_psi.backscatter == 2.e-5
    assert provider.skebs_psi.min_wavenumber == 2 and provider.skebs_psi.max_wavenumber == 12
    assert provider.skebs_theta.timescale_s == 9000.


def test_sppt_named_clip_coefficients_are_active_for_skebs_without_sppt():
    controls, _, dropped = translated(SKEBS_CLIP)
    provider = StochasticModelProvider.from_mapping(controls)
    assert provider.sppt is None
    assert provider.skebs_psi.stddev == provider.skebs_theta.stddev == .25
    assert provider.skebs_psi.cutoff_sigma == provider.skebs_theta.cutoff_sigma == 3.
    assert not {"gridpt_stddev_sppt", "stddev_cutoff_sppt"} & set(dropped)


def test_spp_mynn_flag_is_present_before_original_driver_selection(tmp_path):
    controls, flags, _ = translated(SPP_PBL)
    assert flags["spp_pbl"] == [1, 1]
    exp = mynn_experiment(tmp_path, flags)
    assert all(domain.run.spp_pbl == 1 and domain.run.bl_pbl_physics == 5 for domain in exp.domains)
    provider = StochasticModelProvider.from_mapping(controls)
    assert provider.configure_experiment(exp) is exp
    assert provider.spp_configs["pbl"].stddev == .075


@pytest.mark.parametrize("controls, name", [
    ("sppt = 1, 0,", "per-domain"),
    ("sppt = 1, 1,\n timescale_sppt = 9000.,", "timescale_sppt"),
    ("skebs = 1, 1,\n kminforc = 2,\n lminforc = 1,", "k/l"),
    ("skebs = 1, 1,\n skebs_vertstruc = 1,", "vertical phase"),
    ("sppt = 1, 1,\n sppt_vertstruc = 99,", "vertical phase"),
    ("sppt = 1, 1,\n hrrr_cycling = .false.,", "spectral history"),
    ("sppt = 1, 1,\n kmaxforct = 100,", "WRF rejects"),
    ("spp_lsm = 1, 1,", "no consumer"),
    ("sppt = 2, 2,", "integer 0 or 1"),
    ("sppt = .true., .true.,", "integer 0 or 1"),
    ("sppt = 1, 1,\n unknown_noise = 0,", "unmapped key"),
])
def test_unrepresentable_controls_name_the_breakage(tmp_path, controls, name):
    with pytest.raises(ValueError, match=name):
        imported(tmp_path, controls)


def test_disabled_defaults_keep_original_toml_bytes(tmp_path):
    from gpuwm.wrf_namelist_registry import wrf_namelist_keys
    baseline, _ = import_namelists(*_pair(tmp_path))
    entries = "\n".join(f" {key} = {row['default']}," for (group, key), row in wrf_namelist_keys().items()
                        if group == "stoch")
    disabled, _ = imported(tmp_path, entries)
    assert disabled == baseline and "ensemble" not in tomllib.loads(disabled)


def test_main_spp_selector_sets_the_complete_consumer_arrays():
    from gpuwm.namelist_import import _Section, _err
    from gpuwm.namelist_stochastic import import_stochastic_section
    section = _Section("stoch", {"spp": [1, 0], "spp_pbl": [0, 0]}, "test")
    controls, flags = import_stochastic_section(section, max_dom=2,
        fix=lambda *args: None, drop=lambda *args: None, error=_err)
    assert flags == {"spp_conv": [1, 1], "spp_pbl": [1, 1], "spp_lsm": [1, 1]}
    assert controls["spp"] is True and set(controls["spp_configs"]) == {"conv", "pbl", "lsm"}


@pytest.mark.parametrize("flags, amplitudes", [("0, 1", "0.9, .075"), ("1, 0", ".075, 0.9")])
def test_spp_domains_keep_original_flags_and_inactive_parameters(tmp_path, flags, amplitudes):
    controls, selected, _ = translated(f" spp_pbl = {flags},\n gridpt_stddev_spp_pbl = {amplitudes},")
    expected = [int(value) for value in flags.split(",")]
    assert selected["spp_pbl"] == expected
    exp = mynn_experiment(tmp_path, selected)
    assert [domain.run.spp_pbl for domain in exp.domains] == expected
    provider = StochasticModelProvider.from_mapping(controls)
    assert provider.configure_experiment(exp) is exp and provider.spp_configs["pbl"].stddev == .075


def test_stochastic_value_rules_match_actual_importer_reach(monkeypatch):
    import gpuwm.namelist_contract as contract
    from gpuwm.namelist_stochastic import SUPPORTED_SELECTORS
    monkeypatch.setattr(contract, "BASELINES", contract.BASELINES[:1])
    rules = {("stoch", key): {"values": [0, 1], "scalar": False} for key in SUPPORTED_SELECTORS}
    contract._measure_rules(rules)
    assert all(rule["reach"] == "max_dom" and rule["kind"] == "int" for rule in rules.values())


@pytest.mark.parametrize("active", ["sppt", "skebs", "spp_pbl", "spp", "all"])
def test_every_registered_active_parameter_is_consumed_or_classified_inert(active):
    from gpuwm.namelist_import import _Section, _err
    from gpuwm.namelist_stochastic import import_stochastic_section
    from gpuwm.namelist_contract import _registry_column
    from gpuwm.wrf_namelist_registry import wrf_namelist_keys
    entries = {key: _registry_column(row, 2) for (group, key), row in wrf_namelist_keys().items()
               if group == "stoch"}
    for name in (("sppt", "skebs", "spp") if active == "all" else (active,)):
        entries[name] = [1, 1]
    section = _Section("stoch", entries, "complete active registry")
    dropped = []
    controls, flags = import_stochastic_section(section, max_dom=2,
        fix=lambda *args: None, drop=lambda *args: dropped.append((args[1], args[3])), error=_err)
    assert not section.entries
    assert len(dropped) == len({key for key, _ in dropped})
    assert all("unselected" not in reason for _, reason in dropped)
    if active == "all":
        assert all(key in ("zsigma2_eps", "zsigma2_eta", "spdt", "num_pert_3d", "rand_pert_vertstruc")
                   or key.startswith("pert_") or key.endswith("rand_pert") for key, _ in dropped)

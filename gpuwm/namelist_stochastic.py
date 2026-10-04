"""Representable WRF v4.6.1 stochastic controls for the ensemble run door."""
from __future__ import annotations

from gpuwm.ensemble.stochastic import StochasticConfig
from gpuwm.ensemble.stochastic_seeds import WRF_SEED_DEFAULTS
from gpuwm.wrf_namelist_registry import wrf_namelist_keys

SUPPORTED_SELECTORS = frozenset({"sppt", "skebs", "stoch_force_opt", "spp",
                               "spp_conv", "spp_pbl", "spp_lsm"})
UNSUPPORTED_SELECTORS = frozenset({"rand_perturb", "multi_perturb", "perturb_bdy",
    "perturb_chem_bdy", "pert_cld3", "pert_deng", "pert_farms", "pert_mynn",
    "pert_noah", "pert_thom"})
SEED_NOTICE = ("Stochastic physics uses engine keyed Philox and cuFFT, not WRF's "
    "date-dependent RANDOM_NUMBER realization. Authored nens and iseed_* labels "
    "are retained and bound to versioned process keys and complete checkpoints. "
    "The imported ensemble has one stochastic forecast; --members N changes its "
    "member count. nens is a seed label and never a member count.")


def import_stochastic_section(section, *, max_dom, fix, drop, error):
    """Consume only supported controls, retaining disabled import bytes."""
    registry = {key: row for (group, key), row in wrf_namelist_keys().items()
                if group == "stoch"}

    def default(key):
        row = registry[key]
        raw = row["default"].lower()
        return (raw == ".true." if row["type"] == "logical" else
                int(raw) if row["type"] == "integer" else float(raw.replace("d", "e")))

    for key in sorted(UNSUPPORTED_SELECTORS & section.entries.keys()):
        values = section.take(key)
        if any(bool(value) for value in values):
            # The same reason every run door gives for these selectors
            # (gpuwm.ensemble_admission.RANDOM_SELECTORS), said here with
            # the key that set it.  It used to read "consumer is not
            # implemented", which named neither what would break nor what
            # to do.
            from gpuwm.ensemble_admission import UNCALIBRATED_SPREAD_REASON
            raise error("stoch", key, values,
                f"{UNCALIBRATED_SPREAD_REASON} Next: turn {key} off in &stoch to "
                "import this run without it.")
        fix("stoch", key, values, 0, "selected stochastic consumer is disabled")
    selectors = {}
    for key in sorted(SUPPORTED_SELECTORS):
        values = section.take(key)
        raw = [] if values is None else values[:max_dom]
        if any(type(value) is not int or value not in (0, 1) for value in raw):
            raise error("stoch", key, raw, "must be integer 0 or 1")
        column = raw[:max_dom] + [0] * max(0, max_dom - len(raw))
        selectors[key] = column
        if values is not None and not any(column):
            fix("stoch", key, values, 0, "selected stochastic consumer is disabled")
    selectors["skebs"] = [int(a or b) for a, b in
        zip(selectors["skebs"], selectors["stoch_force_opt"])]
    if any(selectors["spp"]):
        # module_check_a_mundo.F sets the complete arrays, not just domain i.
        for name in ("spp_conv", "spp_pbl", "spp_lsm"):
            selectors[name] = [1] * max_dom
    active = any(any(column) for column in selectors.values())
    if not active:
        for key in sorted(registry.keys() & section.entries.keys()):
            drop("stoch", key, section.take(key),
                "stochastic parameters and bookkeeping are inert with every &stoch selector off")
        section.finish()
        return None, {}

    def uniform(key, column):
        if len(set(column)) != 1:
            raise error("stoch", key, column,
                "the provider shares this stochastic control across domains; "
                "unequal per-domain forcing would apply a different pattern or enable an off domain")
        return column[0]

    enabled = {key: uniform(key, selectors[key]) for key in ("sppt", "skebs")}

    def value(key, selected=None):
        values = section.take(key)
        base = default(key)
        if values is None:
            return base
        if registry[key]["nentries"] == "max_domains":
            column = values[:max_dom] + [base] * max(0, max_dom - len(values))
            result = uniform(key, column if selected is None else
                             [item for item, flag in zip(column, selected) if flag])
        else:
            if len(values) != 1:
                raise error("stoch", key, values, "WRF declares one scalar value")
            result = values[0]
        if registry[key]["type"] == "integer" and type(result) is not int:
            raise error("stoch", key, values, "must be a WRF integer")
        if registry[key]["type"] == "real" and (type(result) not in (int, float)):
            raise error("stoch", key, values, "must be a finite numeric WRF coefficient")
        return result

    def pin(key, expected, reason):
        actual = value(key)
        if type(actual) is not type(expected) and not (
                type(actual) in (int, float) and type(expected) in (int, float)):
            raise error("stoch", key, actual, reason)
        if actual != expected:
            raise error("stoch", key, actual, reason)
        fix("stoch", key, [actual], expected, reason)

    def parameters(kind, suffix, vertical, selected=None):
        overrides = {field: value(prefix + suffix, selected) for field, prefix in (
            ("stddev", "gridpt_stddev_"), ("cutoff_sigma", "stddev_cutoff_"),
            ("lengthscale_m", "lengthscale_"), ("timescale_s", "timescale_"))}
        pin(vertical, 0, "vertical phase rotation is not implemented; a uniform pattern cannot replace it")
        try:
            StochasticConfig(kind=kind, **overrides)
        except ValueError as problem:
            raise error("stoch", "gridpt_stddev_" + suffix, overrides, str(problem)) from problem
        return overrides

    controls = {"sppt": False, "skebs": False, "spp": False}
    if enabled["sppt"]:
        controls["sppt"] = parameters("sppt", "sppt", "sppt_vertstruc")
    if enabled["skebs"]:
        pin("skebs_vertstruc", 0, "vertical phase rotation is not implemented; a uniform SKEBS field cannot replace it")
        pin("stoch_vertstruc_opt", 0, "the obsolete vertical-phase selector enables an unimplemented SKEBS structure")
        stddev = controls["sppt"]["stddev"] if enabled["sppt"] else value("gridpt_stddev_sppt")
        cutoff = controls["sppt"]["cutoff_sigma"] if enabled["sppt"] else value("stddev_cutoff_sppt")
        skebs = {}
        for kind, suffix, temporal in (("skebs_psi", "", "psi"), ("skebs_theta", "t", "t")):
            bounds = {}
            for field, prefix in (("min_wavenumber", "min"), ("max_wavenumber", "max")):
                k, l = value("k" + prefix + "forc" + suffix), value("l" + prefix + "forc" + suffix)
                if k != l:
                    raise error("stoch", "k" + prefix + "forc" + suffix, [k, l],
                        "k/l SKEBS wavenumber limits differ; the current spectral operator shares both limits")
                bounds[field] = k
            config = dict(stddev=stddev, cutoff_sigma=cutoff, **bounds,
                timescale_s=value("ztau_" + temporal), backscatter=value("tot_backscat_" + temporal),
                spectral_exponent=value("rexponent_" + temporal))
            try:
                StochasticConfig(kind=kind, **config)
            except ValueError as problem:
                raise error("stoch", "tot_backscat_" + temporal, config, str(problem)) from problem
            skebs["psi" if not suffix else "theta"] = config
        controls["skebs"] = skebs
    selected = {name: selectors["spp_" + name] for name in ("conv", "pbl", "lsm")}
    if any(any(column) for column in selected.values()):
        controls["spp"] = True
        controls["spp_configs"] = {name: parameters("spp_" + name, "spp_" + name,
            "vertstruc_spp_" + name, column) for name, column in selected.items() if any(column)}
    if enabled["sppt"] or any(any(column) for column in selected.values()):
        for key in ("kminforct", "lminforct", "kmaxforct", "lmaxforct"):
            if key in section.entries:
                pin(key, default(key), "WRF rejects changed temperature wavenumber limits when SPPT/SPP is enabled")
            elif controls["skebs"] and controls["skebs"]["theta"][
                    "min_wavenumber" if "min" in key else "max_wavenumber"] != default(key):
                raise error("stoch", key, controls["skebs"]["theta"],
                    "WRF rejects changed temperature wavenumber limits when SPPT/SPP is enabled")
    labels = {key: value(key) for key in WRF_SEED_DEFAULTS}
    controls["wrf_seed_labels"] = labels
    from gpuwm.ensemble.stochastic_seeds import normalize_wrf_seed_labels
    try:
        normalize_wrf_seed_labels(labels)
    except ValueError as problem:
        raise error("stoch", "nens", labels, str(problem)) from problem
    if enabled["sppt"] or enabled["skebs"]:
        pin("hrrr_cycling", True, "a restart that reseeds a stochastic process cannot preserve its complete spectral history")
    for key in sorted(registry.keys() & section.entries.keys()):
        if key in ("zsigma2_eps", "zsigma2_eta"):
            reason = "WRF computes the SKEBS noise variance from accepted dt; this registered coefficient has no consumer"
        elif key == "hrrr_cycling":
            reason = "WRF consults this restart reseeding switch for SPPT/SKEBS/random fields, not the selected SPP consumers"
        elif key in ("spdt", "num_pert_3d") or key.startswith("pert_"):
            reason = "multi_perturb and its field consumers are disabled; this perturbation-table coefficient is inert"
        elif "rand_pert" in key:
            reason = "rand_perturb is disabled; its random-field coefficient is inert"
        elif "spp_" in key:
            reason = "the corresponding spp_conv/pbl/lsm consumer is disabled; its parameter-pattern coefficient is inert"
        elif "sppt" in key:
            reason = "SPPT is disabled; this correlation/vertical parameter does not affect the selected SKEBS/SPP processes"
        elif key in {"kminforc", "kmaxforc", "lminforc", "lmaxforc", "rexponent_psi",
                     "rexponent_t", "skebs_vertstruc", "stoch_vertstruc_opt", "tot_backscat_psi",
                     "tot_backscat_t", "ztau_psi", "ztau_t"}:
            reason = "SKEBS is disabled; SETUP_RAND_PERTURB uses these controls only for its W/T backscatter branches"
        else:
            raise error("stoch", key, section.entries[key],
                "this registered parameter has no classified stochastic consumer; refusing to drop an active control")
        drop("stoch", key, section.take(key), reason)
    section.finish()
    return controls, {"spp_" + name: column for name, column in selected.items()}

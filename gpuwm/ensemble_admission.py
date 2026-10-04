"""The named reasons an ensemble request is refused, shared by every door.

Standard library only, and no import of the ensemble package.  The config
loaders (:func:`gpuwm.experiment.build_experiment`,
:func:`gpuwm.config.load_config`) call into this module on every load, and
they are staged into the preparation-only wheel, which carries no ensemble
runtime.

Each refusal here names the breakage it prevents:

* random spread whose amplitude has no observation calibration
  (:data:`UNCALIBRATED_SPREAD_REASON`): the spread and the probabilities
  drawn from it would mean nothing;
* N > 1 members that would all run one input (:func:`copies_breakage`):
  the run is N copies of one forecast, published with zero spread and
  probabilities of 0 or 1;
* listed member sources that no door binds
  (:func:`unbound_sources_refusal`): every member would run the one
  prepared input under the listed names;
* an ``[ensemble]`` table on a door that opens no ensemble session
  (:func:`table_not_honoured`): the table would be dropped and one
  forecast would run under the name of the ensemble it describes;
* the 2.8.4 overlay table or a perturbation provider name at a door that
  reads neither (:func:`overlay_table_refusal`,
  :func:`provider_name_refusal`): the run would not be the ensemble the
  file describes.
"""
from __future__ import annotations


UNCALIBRATED_SPREAD_REASON = (
    "Random ensemble perturbations are unavailable: spread amplitudes have not "
    "been calibrated against observations, so ensemble spread and probabilities "
    "would be meaningless.")
RANDOM_SELECTORS = frozenset({
    "sppt", "skebs", "stoch_force_opt", "spp", "spp_conv", "spp_pbl", "spp_lsm",
    "rand_perturb", "multi_perturb", "perturb_bdy", "perturb_chem_bdy",
    "pert_cld3", "pert_deng", "pert_farms", "pert_mynn", "pert_noah", "pert_thom",
})

#: The two keys every 2.8.4 overlay file carries (schema
#: ``gpuwm-ensemble-config.v1``, read by ``python -m tools.ensemble_forecast``).
#: Neither is a key of the experiment config's ``[ensemble]`` table, so
#: either one marks the table as that overlay.
OVERLAY_MARKERS = frozenset({"base_config", "n_members"})
OVERLAY_COMMAND = "python -m tools.ensemble_forecast run --ensemble-config FILE"
TABLE_KEYS = "members, keep_member_files, thresholds, recipe and trajectories"


def refuse_configured_random(raw):
    """Refuse an authored random-physics switch, wherever the config sets it.

    Called by the config loaders themselves, so every door that reads a
    config inherits the refusal before it downloads, spawns or allocates.
    The ``[ensemble]`` table is skipped here: its own reader checks its
    ``perturbation`` and ``stochastic`` entries.
    """
    if isinstance(raw, dict):
        for name, value in raw.items():
            if name == "ensemble":
                continue
            if name in RANDOM_SELECTORS:
                enabled = any(bool(item) for item in value) if isinstance(value, list) else bool(value)
                if enabled:
                    raise ValueError(UNCALIBRATED_SPREAD_REASON)
            elif isinstance(value, (dict, list)):
                refuse_configured_random(value)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, (dict, list)):
                refuse_configured_random(item)


def copies_breakage(members) -> str:
    """What N members drawn from one input are, in one sentence."""
    return (f"{members} members from one input are {members} copies of one forecast: "
            "spread is zero and probabilities are 0 or 1.")


def member_source_remedy(members) -> str:
    """The two front doors that give each member its own source trajectory."""
    return (f"Next: gpuwm ensemble CONFIG --members {members} --recipe time-lagged "
            "(earlier cycles of the config's own source), or gpuwm ensemble CONFIG "
            "--trajectories FILE (one source and cycle per member).")


def one_input_refusal(members, holds) -> str:
    """The refusal of a door that holds one input and was asked for N members.

    ``holds`` says, as a clause, what the one input is: the door's own
    words for the prepared bundle, checkpoint or input directory it runs.
    """
    return f"{copies_breakage(members)} {holds} {member_source_remedy(members)}"


def unbound_sources_refusal() -> str:
    """The refusal of ``[ensemble] sources`` where no door binds the listed sources."""
    return ("[ensemble] sources names a source for each member, and this door binds "
            "none of them: every member would run the one prepared input under those "
            "names, so the ensemble would have zero spread and probabilities of 0 or 1. "
            "Next: put the same list in a file and run gpuwm ensemble CONFIG "
            "--trajectories FILE (one source and cycle per member).")


def table_not_honoured(source, door, *, prepared=False) -> str:
    """The refusal of an ``[ensemble]`` table on a door with no ensemble session.

    ``prepared`` says the door runs a prepared bundle.  A bundle binds the
    config it was prepared from, so the way to one forecast there is to
    prepare the config again without the table, not to edit the bound copy.
    """
    one_forecast = ("take the [ensemble] table out of the config and prepare it again "
                    "to run one forecast" if prepared else
                    "remove the [ensemble] table to run this one forecast")
    return (f"{source} carries an [ensemble] table, and {door} runs one forecast "
            "and opens no ensemble session: the table would be dropped and one "
            "forecast would run under the name of the ensemble it describes. "
            "Next: gpuwm ensemble CONFIG on the config gpuwm domain wrote "
            f"(it fetches and prepares each member), or {one_forecast}.")


def overlay_table_refusal(keys) -> str:
    """The refusal of the 2.8.4 overlay table at an experiment-config door."""
    return (f"this [ensemble] table carries {', '.join(sorted(keys))}: it is the overlay "
            "file of the tools.ensemble_forecast command. This door reads none of it, "
            "so the run would not be the ensemble the file describes. "
            f"Next: {OVERLAY_COMMAND}. An experiment config's [ensemble] table "
            f"takes {TABLE_KEYS}.")


def provider_name_refusal(name) -> str:
    """The refusal of a 2.8.4 perturbation provider name at an experiment-config door."""
    return (f'[ensemble] perturbation = "{name}" is a provider name, and only the '
            "tools.ensemble_forecast command binds perturbation providers: this door "
            "would run every member unperturbed under that name. "
            f"Next: {OVERLAY_COMMAND}, with an overlay file that names the base "
            "config and this provider. An experiment config's [ensemble] table "
            f"takes {TABLE_KEYS}.")

"""Did the treatment actually run?

Five A/B screens in this project's history were declared, executed, scored
and reported before anyone noticed that the flag under test had never been
set.  The arms were byte-identical; the deltas were exactly 0.0; the
conclusions were void.  The failure was never in the science, and it was
never visible in the run's own output -- a run with a stream switched on
and a run with that stream silently contributing nothing produced the same
receipt, so nothing could tell them apart after the fact.

This module makes that indistinguishability impossible.  It has two jobs.

**Count what was assimilated, per observation type, per cycle.**  The
count is taken from the batches the filter actually solved, at the one
point where every type is present in one list, and it is keyed by the
adapters' own ``kind`` discriminators rather than by which flags were on.
"Reflectivity was enabled" and "reflectivity contributed 14,203
observations" are different claims and only the second is evidence.

**Refuse to call the run full-stack when an enabled stream contributed
nothing.**  Not a warning, not a downgrade to a smaller label: a refusal.
A stream that assimilated zero observations across the opening cycles is
either misconfigured or being fed files that do not cover the domain, and
in both cases the honest outcome is a stopped run and a message naming the
stream -- not six hours of card time producing a result whose headline
would be a lie.  Two cycles rather than one because a single cycle can
legitimately miss a satellite granule or a radar volume; a stream that is
genuinely live shows up in one of the first two.

The refusal is deliberately not symmetric with the enable flags.  Turning
a stream ON is a claim that it will contribute, and this module holds the
claim to account.  Leaving one off asserts nothing and is never checked.
"""

from __future__ import annotations

#: The observation types this project can assimilate, spelled as the
#: adapters' own ``kind`` discriminators.  Not an enum the adapters import
#: -- they each name their own kind, and this is the reader's list of what
#: those names are.  A new stream appears here when its adapter starts
#: emitting a kind, which is the same moment it becomes countable.
OBS_KINDS = (
    "radial_velocity",
    "reflectivity",
    "clear_air_reflectivity",
    "surface",
    "cloud_water_path",
)

#: What a caller says it turned on, mapped to the kind that proves it.
#: ``clear_air_reflectivity`` is intentionally absent: clear-air zeroes are
#: legitimately empty over a domain with no measured clear air, so a zero
#: there is a fact about the weather rather than about the configuration.
PROVABLE_KINDS = (
    "radial_velocity",
    "reflectivity",
    "surface",
    "cloud_water_path",
)


class TreatmentNotApplied(RuntimeError):
    """An enabled observation stream assimilated nothing.

    Raised rather than logged.  The whole point of this module is that a
    run which cannot prove its own treatment must not finish and publish a
    number, because the number would be attributed to a stream that never
    touched the state.
    """


def batch_kinds(*provenances) -> dict[str, str]:
    """Map batch name -> observation kind, from the adapters' own records.

    Each adapter publishes a ``batches`` list whose entries carry both the
    batch ``name`` the filter knows and the ``kind`` the adapter calls it.
    This joins them so counts taken off the solved batches -- which know
    only names -- can be attributed to types.

    Unknown or missing provenance contributes nothing rather than raising:
    a batch whose kind cannot be established is counted under
    ``"unattributed"`` by :func:`assimilated_counts`, which is a louder
    outcome than guessing.
    """

    kinds: dict[str, str] = {}
    for provenance in provenances:
        if not provenance:
            continue
        for entry in provenance.get("batches") or ():
            name = entry.get("name")
            kind = entry.get("kind")
            if name is not None and kind is not None:
                kinds[str(name)] = str(kind)
    return kinds


def assimilated_counts(innovations, kinds: dict[str, str]) -> dict[str, int]:
    """Sum assimilated observations per kind for one analysis.

    ``innovations`` is :func:`gpuwm.da.radar_assimilation.innovation_summary`
    output -- one entry per batch, each carrying ``observations``, which is
    ``count_nonzero(mask)``: the number of observations the filter actually
    had, after thinning, after QC, after every refusal.  That is the number
    this project has been unable to point at.

    Every kind in :data:`OBS_KINDS` is present in the result, at zero when
    absent.  A key that only appears when non-zero cannot be tested for
    zero, which is the whole failure mode.
    """

    counts = {kind: 0 for kind in OBS_KINDS}
    for entry in innovations or ():
        name = str(entry.get("name", ""))
        kind = kinds.get(name, "unattributed")
        counts[kind] = counts.get(kind, 0) + int(entry.get("observations", 0))
    return counts


def dealias_recovered(adapter_provenance) -> int | None:
    """Gates the unfolder recovered, or None if it never ran.

    ``None`` and ``0`` are different findings and are kept different.  None
    says the observation file was built without dealiasing; 0 says the
    unfolder ran over this volume and resolved no folded gate.  A run
    claiming dealiasing as a treatment must show a number here, and
    conflating the two would let "the flag was never wired" read as "there
    was nothing to unfold" -- which is precisely the substitution that
    voided the earlier screens.

    The account is a LIST, one entry per contributing radar, because
    :func:`gpuwm.obs.superob.merge_contributions` keeps each radar's
    unfolding separate -- a fold is a property of one radar's Nyquist and
    one radar's sweep, and pooling them before counting would lose which
    radar recovered what.  Summing here is the right reduction for the
    question this answers ("did the treatment do anything this cycle"),
    and the per-radar detail stays in the observation file.  A bare dict
    is accepted too, for a single-radar caller that never went through the
    merge.
    """

    if not adapter_provenance:
        return None
    account = adapter_provenance.get("dealias")
    if not account:
        return None
    entries = account if isinstance(account, list) else [account]
    total = None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        totals = entry.get("totals") or {}
        if "gates_unfolded" in totals:
            total = int(totals["gates_unfolded"]) + (total or 0)
    return total


def cycle_record(innovations, *, adapter_provenance=None,
                 extra_obs_provenance=None,
                 cwp_provenance=None) -> dict:
    """The per-cycle assimilation account that goes into the run record.

    One call per analysis, from the driver, with the provenance blocks the
    analysis just returned.  Everything here is measured off the solved
    batches; nothing is copied from a command-line flag.
    """

    kinds = batch_kinds(adapter_provenance, extra_obs_provenance,
                        cwp_provenance)
    counts = assimilated_counts(innovations, kinds)
    return {
        "counts": counts,
        "total": sum(counts.values()),
        "dealias_recovered_gates": dealias_recovered(adapter_provenance),
        "note": ("counts are count_nonzero(mask) on the batches the filter "
                 "solved -- after thinning and QC -- not a reading of the "
                 "files or of which flags were set"),
    }


def verify_treatment(enabled, cycles, *, require_cycles: int = 2,
                     label: str = "full-stack") -> dict:
    """Refuse the run's label unless every enabled stream contributed.

    Parameters
    ----------
    enabled
        The kinds the caller turned on, e.g.
        ``("radial_velocity", "cloud_water_path")``.  Kinds outside
        :data:`PROVABLE_KINDS` are recorded and not enforced.
    cycles
        The :func:`cycle_record` results so far, in cycle order.
    require_cycles
        How many opening cycles must be available before a verdict is
        possible, and over how many the evidence is pooled.  Fewer than
        this returns a ``"pending"`` verdict and raises nothing: a run that
        has not yet had the chance to show its streams has not failed.

    Returns the verdict block to store in the run record.  Raises
    :class:`TreatmentNotApplied` naming every silent stream.
    """

    enabled = tuple(dict.fromkeys(str(kind) for kind in enabled))
    unknown = [kind for kind in enabled if kind not in OBS_KINDS]
    if unknown:
        raise TreatmentNotApplied(
            f"treatment proof asked about unknown observation kind(s) "
            f"{unknown}; known kinds are {list(OBS_KINDS)}. A stream whose "
            "adapter publishes no kind cannot be proved to have run")

    window = list(cycles)[:require_cycles]
    totals = {kind: 0 for kind in enabled}
    for record in window:
        for kind, count in (record.get("counts") or {}).items():
            if kind in totals:
                totals[kind] += int(count)

    if len(window) < require_cycles:
        return {
            "label": label,
            "verdict": "pending",
            "enabled": list(enabled),
            "cycles_seen": len(window),
            "cycles_required": require_cycles,
            "assimilated_over_window": totals,
        }

    enforced = [kind for kind in enabled if kind in PROVABLE_KINDS]
    silent = sorted(kind for kind in enforced if totals[kind] == 0)
    verdict = {
        "label": label,
        "verdict": "silent_stream" if silent else "proved",
        "enabled": list(enabled),
        "enforced": enforced,
        "not_enforced": [k for k in enabled if k not in PROVABLE_KINDS],
        "cycles_seen": len(window),
        "cycles_required": require_cycles,
        "assimilated_over_window": totals,
        "silent_streams": silent,
    }
    if silent:
        detail = ", ".join(f"{kind}=0" for kind in silent)
        raise TreatmentNotApplied(
            f"{label} was claimed but {len(silent)} enabled observation "
            f"stream(s) assimilated nothing over the first {require_cycles} "
            f"cycle(s): {detail}. Every other enabled stream: "
            + ", ".join(f"{k}={totals[k]}" for k in enforced
                        if k not in silent)
            + ". This run is stopped rather than relabelled: a stream that "
            "contributes no observation cannot be credited with any part of "
            "the result, and five earlier screens in this project were "
            "voided by exactly this going unnoticed. Check that the stream's "
            "input files cover the domain and the analysis times, and that "
            "the flag enabling it reached the driver.")
    return verdict

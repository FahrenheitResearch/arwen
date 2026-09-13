"""WRF metgrid number moments and their exact active Registry targets."""

# The metgrid names this ingest reads, and the Registry name each one
# targets.  Adding a source is a ROW here, never a branch anywhere else:
# metgrid_number_targets below resolves every row through the SAME
# active_moisture_map/nest_field_kinds membership, so a package that does
# not transport a row's target discards it exactly like an inactive P_QN*.
# QNWFA/QNIFA are the aerosol-aware Thompson (mp=28) water-friendly and
# ice-friendly aerosol numbers; gpuwm/ingest/wrfinput.py MOISTURE_MAP
# already carries QNWFA->nwfa and QNIFA->nifa, so they need no new map.
_WRF_NAMES = {
    "QNI": "QNICE", "QNC": "QNCLOUD", "QNR": "QNRAIN", "QNS": "QNSNOW",
    "QNG": "QNGRAUPEL", "QNH": "QNHAIL",
    "QNWFA": "QNWFA", "QNIFA": "QNIFA",
}
METGRID_NUMBER_FIELDS = tuple(_WRF_NAMES)


def metgrid_number_targets(cfg):
    """Resolve P_QN* membership using the shared selected WRF package.

    module_initialize_real.F:1999-2120 interpolates each flagged source
    only when the corresponding P_QN* belongs to num_3d_s. In particular,
    QNC targets P_QNC, never NSSL's distinct P_QNDROP scalar.
    """
    from gpuwm.ingest.wrfinput import active_moisture_map
    from gpuwm.core.nest_fields import nest_field_kinds
    active = active_moisture_map(cfg)
    transported = set(nest_field_kinds(cfg))
    return {name: active[wrf] for name, wrf in _WRF_NAMES.items()
            if wrf in active and active[wrf] in transported}


#: The two rows of the table above that are AEROSOL numbers rather than
#: hydrometeor numbers.  A row pair, not a branch: every site that has to
#: ask "did this analysis bring its own aerosol" reads this tuple, so
#: adding an aerosol species is one entry here and not a hunt through the
#: ingest, the run-door floor and the receipt.
AEROSOL_NUMBER_FIELDS = ("QNWFA", "QNIFA")


def analyzed_aerosol_fields(decoded_numbers, number_targets):
    """Which aerosol rows this analysis carries AND this package transports.

    ONE SPELLING of that question.  The run-door floor in
    :func:`gpuwm.ingest.real.initialize_real` asks it before it decides
    whether a monthly climatology is a precondition at all, and the
    aerosol-source resolver asks it again when it chooses the source; the
    two answering differently is precisely how a run gets refused for a
    missing dataset it would never have opened.
    """

    return tuple(name for name in AEROSOL_NUMBER_FIELDS
                 if name in decoded_numbers and name in number_targets)


#: THE NAMELIST SPELLING of "initialize the aerosol from WRF's monthly WIF
#: climatology": ``&physics use_aero_icbc = .true.`` with ``&domains
#: wif_input_opt = 1``, which :mod:`gpuwm.namelist_import` admits and emits
#: into the TOML as this selector pair, and which
#: :func:`gpuwm.config.validate_aerosol_source_options` keeps admissible on
#: purpose.  A ROW, not a branch: every message that has to tell an
#: operator WHAT asked for the climatology prints this one pair, because a
#: sentence that names ``mp28_aerosol_source`` to an operator who never set
#: it is a way out that changes nothing.
WIF_CLIMATOLOGY_NAMELIST_PAIR = (("aer_init_opt", 1), ("wif_input_opt", 1))

#: The pair as it is spelled back in a refusal or a warning.
WIF_CLIMATOLOGY_NAMELIST_PHRASE = (
    "(" + ", ".join(name for name, _ in WIF_CLIMATOLOGY_NAMELIST_PAIR)
    + ") = (" + ", ".join(str(value)
                          for _, value in WIF_CLIMATOLOGY_NAMELIST_PAIR) + ")")


def wif_climatology_named_by_namelist(cfg) -> bool:
    """Is the NAMELIST PAIR, not ``mp28_aerosol_source``, what asked for it?

    ONE SPELLING of that question, for the same reason
    :func:`analyzed_aerosol_fields` is one spelling of its own: the
    aerosol-source resolver rewrites ``mp28_aerosol_source='auto'`` into
    ``'climatology'`` whenever this pair is set, so every site that tells
    an operator how to stop reading the climatology has to know whether
    the request came from the field or from the namelist.  Telling the
    second operator to "leave mp28_aerosol_source at 'auto'" names a
    setting that is ALREADY 'auto': a way out that is a no-op.
    """

    choice = str(getattr(cfg, "mp28_aerosol_source", "auto") or "auto")
    if choice != "auto":
        # The operator's own word wins and is what the message must name;
        # the pair is not what resolved this run.
        return False
    return all(int(getattr(cfg, name, 0) or 0) == value
               for name, value in WIF_CLIMATOLOGY_NAMELIST_PAIR)


def analyzed_aerosol_way_out(carried, *, named_by_namelist: bool = False) -> str:
    """The sentence naming the analyzed route as a way out of the floor.

    The mp=28 lateral-forcing precondition predates the analyzed route and
    names three ways out, all of them a dataset or a fallback.  A fourth
    one exists now, it is the one a met_em operator can actually take, and
    a refusal that does not name it sends that operator to fetch 225 MB
    they do not need.  Built from the analysis in hand, so it says which
    half is missing rather than reciting the pair.

    ``named_by_namelist`` (from :func:`wif_climatology_named_by_namelist`)
    is what keeps the way out from being a no-op.  With the selector pair
    set, ``mp28_aerosol_source`` is already 'auto' and leaving it there
    changes nothing: the pair is the thing that has to be cleared, so the
    pair is what the sentence names.
    """

    carried = tuple(name for name in AEROSOL_NUMBER_FIELDS if name in carried)
    missing = tuple(name for name in AEROSOL_NUMBER_FIELDS
                    if name not in carried)
    both = " and ".join(AEROSOL_NUMBER_FIELDS)
    flags = " and ".join(f"FLAG_{name}=1" for name in AEROSOL_NUMBER_FIELDS)
    if not missing:
        # The pair IS here and a source was asked for anyway; the way out
        # is to stop asking, in whichever of the two spellings did it.
        if named_by_namelist:
            return ("This analysis carries both " + both + ", which "
                    "initializes nwfa/nifa with no climatology, and "
                    "mp28_aerosol_source is already 'auto': "
                    + WIF_CLIMATOLOGY_NAMELIST_PHRASE + " is what asks for "
                    "the climatology here. Set either selector to 0 to take "
                    "the analyzed pair.")
        return ("This analysis carries both " + both + ", which initializes "
                "nwfa/nifa with no climatology: leave mp28_aerosol_source at "
                "'auto' to take it.")
    if carried:
        held = (f"This analysis carries {'/'.join(carried)} but not "
                f"{'/'.join(missing)}, and an aerosol-aware run needs both.")
    else:
        held = ("This analysis carries neither "
                f"{' nor '.join(AEROSOL_NUMBER_FIELDS)}.")
    if named_by_namelist:
        stop = ("set either selector of " + WIF_CLIMATOLOGY_NAMELIST_PHRASE
                + " to 0, which is what asks for the climatology here "
                  "(mp28_aerosol_source is already 'auto')")
    else:
        stop = "leave mp28_aerosol_source at 'auto'"
    return (held + " An analysis carrying both initializes nwfa/nifa from "
            "its own fields and opens no climatology: regenerate met_em "
            "with " + flags + " and " + stop + ".")

"""The mp=28 aerosol source, carried from the ingest into a run report.

WHAT WAS MISSING.  :func:`gpuwm.ingest.real.initialize_real` already
decides -- and already writes down -- which of two initial conditions an
aerosol-aware Thompson run received: WRF's global monthly
water/ice-friendly aerosol climatology
(``QNWFA_QNIFA_SIGMA_MONTHLY.dat``), or ``thompson_init``'s synthetic
CCN/IN profile (``phys/module_mp_thompson.F:493-551``).  That decision is
the run's ``RealInitResult.aerosol_initialization`` receipt and it is
announced on stderr by
``gpuwm.ingest.real._warn_synthetic_aerosol_fallback``.  Console
scrollback is not a record: two forecasts that differ in which of those
two the model started from are different experiments, and a reader
holding only ``report.json`` could not tell which one they had.  This
module is the one place that turns that receipt into a report key, so
every writer says it the same way.

THE TWO ABSENCES ARE DIFFERENT FACTS, and blurring them is the concrete
breakage this module exists to prevent.

  1. ABSENT BECAUSE INAPPLICABLE.  WSM6, Morrison, classic Thompson,
     Kessler and every other scheme have no water-friendly/ice-friendly
     aerosol number fields at all -- only ``thompsonaero`` carries
     qnwfa / qnifa / qnwfa2d / qnifa2d (``Registry.EM_COMMON:3036``).
     There is no dataset such a run could have read and none it failed
     to read.  Such a run gets NO KEY: :func:`aerosol_source_report_entry`
     returns an empty mapping and the report is the report it was before
     this module existed, key for key.  A key reading "no aerosol dataset
     was used" on a WSM6 run would be true of the bytes and false of the
     meaning -- it would put a WSM6 report and a climatology-less mp=28
     report in one bucket, which is exactly the pair a reader needs told
     apart.

  2. ABSENT BECAUSE THE FALLBACK WAS TAKEN.  An mp_physics=28 run that
     searched for the dataset and did not find it ran on an analytic
     profile.  That IS "no dataset was used", it is a material fact about
     the forecast, and it is reported loudly: ``dataset_used`` is
     ``False`` and ``aerosol_source`` names
     ``thompson_init-synthetic-profile``.

  3. And a third, so the second is never guessed at: an mp=28 run whose
     writer holds no ingest receipt at all -- a route initialized from an
     archived parent frame, a prepared cache written before the receipt
     was recorded.  ``recorded`` is ``False`` and ``dataset_used`` is
     ABSENT rather than ``False``: "not recorded" and "no dataset" are
     not the same sentence.  Each caller supplies its own
     ``when_unrecorded`` naming what its route actually knows.

WHY A MODULE RATHER THAN A HELPER IN ``real.py``.  The writers that need
this are report assemblers -- ``gpuwm.prepared_single_domain_forecast``,
the offline-child runners, the native benchmark and proof harnesses --
and several of them never import the real ingest at all.  This module
imports nothing but the standard library, so adding the key costs a
report writer nothing.
"""

from __future__ import annotations

from collections.abc import Mapping


#: The report.json key.  One spelling, used by every writer, so a reader
#: greps once and a bundle tool has one name to look for.
AEROSOL_SOURCE_KEY = "aerosol_initialization"

#: The multi-domain key, for a receipt that covers a whole nest tree.
#: A DIFFERENT NAME rather than the same one holding a different shape:
#: a consumer reading ``report["aerosol_initialization"]["dataset_used"]``
#: must not start throwing the moment it meets a tree receipt.  Both names
#: contain ``aerosol_initialization``, so one grep still finds every run.
AEROSOL_SOURCE_BY_DOMAIN_KEY = "aerosol_initialization_by_domain"

#: Versioned because the block is a receipt others will read
#: mechanically.  It is the BLOCK's version, not the report's: the
#: enclosing documents keep their own schema strings, and this key is
#: added the way ``report["tiles"]`` and ``report["stream_init"]`` were
#: added before it -- present only when it has something to say, so no
#: report that could not carry it changes by one byte and no enclosing
#: schema string has to move.
AEROSOL_SOURCE_SCHEMA = "gpuwm-run-aerosol-source-v1"

#: The schemes that HAVE water-friendly/ice-friendly aerosol number
#: fields.  Aerosol-aware Thompson and nothing else: ``thompsonaero`` is
#: the only Registry package declaring qnwfa/qnifa/qnwfa2d/qnifa2d
#: (``Registry.EM_COMMON:3036``, ``Registry/registry.new3d_wif``).  A
#: frozenset rather than ``== 28`` at six call sites, so that a second
#: aerosol-aware scheme is one edit here and not a hunt.
AEROSOL_AWARE_MP_PHYSICS = frozenset({28})


def scheme_has_aerosol_number_fields(mp_physics) -> bool:
    """Does this microphysics option have aerosol fields to initialize?

    ``False`` is the answer that means "the question does not apply to
    this run", and it is why such a run gets no report key at all.
    """

    try:
        return int(mp_physics) in AEROSOL_AWARE_MP_PHYSICS
    except (TypeError, ValueError):
        return False


def _plain(value):
    """Recursively unwrap read-only containers into plain JSON types.

    The ingest receipt reaches some writers through a prepared cache,
    which hands its metadata back as ``MappingProxyType`` so a restored
    document cannot be mutated.  ``json.dumps`` serializes ``dict`` and
    not ``Mapping``, so a proxy anywhere in the tree turns "write the
    report" into a ``TypeError`` naming nothing.  Normalizing here rather
    than at each writer keeps one rule and changes no output.
    """

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def aerosol_source_report_entry(receipt, *, mp_physics, when_unrecorded):
    """The report.json fragment naming this run's aerosol source.

    Returns a mapping the caller merges into its report -- either empty
    (nothing to say, and nothing added) or exactly one key,
    :data:`AEROSOL_SOURCE_KEY`.  Merging rather than returning the block
    itself is what lets a writer stay byte-identical on every run the
    question does not apply to::

        report.update(aerosol_source_report_entry(
            result.aerosol_initialization,
            mp_physics=cfg.mp_physics,
            when_unrecorded="..."))

    ``receipt`` is ``RealInitResult.aerosol_initialization`` -- possibly
    empty, possibly restored from a prepared cache, possibly ``None``.

    ``when_unrecorded`` is REQUIRED and has no default.  It is the
    sentence this route can honestly give for an aerosol-aware run it
    holds no ingest receipt for: what filled the fields, and why this
    writer cannot name a dataset.  There is no generic true answer -- an
    offline child inherits its parent's aerosol, a prepared forecast is
    reading a cache someone else wrote -- so each caller states its own
    rather than this module inventing one, and a route that cannot state
    one has found a gap rather than a formatting problem.
    """

    if not scheme_has_aerosol_number_fields(mp_physics):
        # ABSENCE (1): INAPPLICABLE.  No key.  See the module docstring:
        # a scheme with no aerosol fields did not "fail to use a
        # dataset", and saying so would make it indistinguishable from a
        # scheme that did.
        return {}
    entry = {
        "schema": AEROSOL_SOURCE_SCHEMA,
        "mp_physics": int(mp_physics),
        # Always True wherever this key exists at all -- and stated
        # anyway, because the fact a reader must not infer from an ABSENT
        # key is exactly this one, and a reader who never sees the key
        # has no place to learn the rule.
        "scheme_has_aerosol_number_fields": True,
    }
    if not receipt:
        # ABSENCE (3): APPLICABLE, NOT RECORDED.  Deliberately no
        # ``dataset_used``: writing ``False`` here would state that no
        # dataset was read, which this writer does not know.
        entry["recorded"] = False
        entry["not_recorded_because"] = str(when_unrecorded)
        return {AEROSOL_SOURCE_KEY: entry}
    plain = _plain(receipt)
    source = plain.get("aerosol_source")
    entry["recorded"] = True
    entry["aerosol_source"] = source
    # ABSENCE (2), when it is the answer: the run searched and found
    # nothing, ran on the analytic profile, and says so in one boolean a
    # reader can filter on without parsing prose.
    entry["dataset_used"] = source == "wif-climatology"
    entry["synthetic_fallback_in_use"] = bool(
        plain.get("synthetic_fallback_in_use", False))
    entry["synthetic_fallback_requested"] = bool(
        plain.get("synthetic_fallback_requested", False))
    entry["mp28_aerosol_source"] = plain.get("mp28_aerosol_source")
    entry["aerosol_source_statement"] = plain.get("aerosol_source_statement")
    # THE PINNED IDENTITY.  ``describe_wif_source`` content-addresses the
    # copy that was actually opened -- path, byte count, computed SHA-256
    # and whether it matches the WRF 4.7.1 reference -- so two runs that
    # read two releases' copies stay distinguishable afterwards, which a
    # repeated constant could never show.  On the fallback branch the
    # same key carries the search that came up empty, so "no dataset" is
    # actionable rather than merely stated.
    entry["dataset"] = plain.get("dataset")
    entry["initialization_receipt"] = plain
    return {AEROSOL_SOURCE_KEY: entry}


def aerosol_source_report_entries(domains, *, when_unrecorded):
    """The multi-domain form, for a receipt covering a whole nest tree.

    ``domains`` is an iterable of ``(label, mp_physics, receipt)``.  Each
    domain gets its own answer because each domain has its own
    initialization: a root fed from the WIF climatology and a child whose
    prepared cache predates the receipt are two different facts about one
    run, and averaging them into a single verdict would lose both.

    A SEPARATE KEY from :data:`AEROSOL_SOURCE_KEY`, deliberately.  Same
    key, two shapes -- a block on one receipt and a map of blocks on
    another -- is how a consumer that reads ``entry["dataset_used"]``
    starts throwing on half the receipts in a bundle.  The name says
    which shape it is, and both names contain "aerosol_initialization"
    so one grep still finds every run.

    Returns ``{}`` when no domain has anything to say, so a tree of
    non-aerosol-aware domains writes the receipt it wrote before.
    """

    blocks = {}
    for label, mp_physics, receipt in domains:
        entry = aerosol_source_report_entry(
            receipt, mp_physics=mp_physics, when_unrecorded=when_unrecorded)
        if entry:
            blocks[str(label)] = entry[AEROSOL_SOURCE_KEY]
    if not blocks:
        return {}
    return {AEROSOL_SOURCE_BY_DOMAIN_KEY: blocks}


__all__ = [
    "AEROSOL_AWARE_MP_PHYSICS",
    "AEROSOL_SOURCE_BY_DOMAIN_KEY",
    "AEROSOL_SOURCE_KEY",
    "AEROSOL_SOURCE_SCHEMA",
    "aerosol_source_report_entries",
    "aerosol_source_report_entry",
    "scheme_has_aerosol_number_fields",
]

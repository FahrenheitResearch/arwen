"""The moisture floors an initialization applied, carried into its proof.

WHAT WAS MISSING.  :func:`gpuwm.ingest.real.initialize_real` floors water
vapour in more than one place, and it already writes down what it did:
``RealInitResult.surface_moisture_floor`` records the cells of the
published surface mixing ratio that WPS's overshooting ``sixteen_pt``
operator drove below WRF's ``qv_min_value`` on the FLAG_SH lane, and any
further ``*_moisture_floor`` receipt the ingest grows joins it on the same
result.  Those receipts reached memory and stderr and stopped there.  A
prepared bundle carries a ``proof.json``, the run that consumes it is a
different process, and nothing in the document said whether a floor had
fired -- so a forecast whose initial vapour field was modified on the way
in was indistinguishable, afterwards, from one that was not.

THE ABSENCE AND THE ZERO ARE DIFFERENT FACTS, and keeping them apart is
the whole job of this module.

  1. A FLOOR THAT DID NOT FIRE is the usual case and it is STATED:
     ``fired`` is ``False`` and the floor's name is in the document.  An
     absent key would have been read as "this preparation predates the
     receipt", which is a different claim and one no reader could check.

  2. A FLOOR THAT FIRED carries the ingest's own receipt verbatim beside
     ``fired: True`` -- how many cells, how far below the floor they
     were -- because "some vapour was modified" without the magnitude is
     a fact nobody can act on.

  3. And a third, so the second is never guessed at: a writer holding an
     initialization result that declares no floor field at all -- a route
     whose state came back from an archived frame, a stand-in with no
     ingest behind it.  ``recorded`` is ``False`` and there is no
     ``fired``: "not recorded" and "nothing was floored" are not the same
     sentence.  Each caller supplies its own ``when_unrecorded`` naming
     what its route actually knows, and a caller that cannot state one has
     found a gap rather than a formatting problem, which is why the
     argument is required and has no default.

WHICH FLOORS ARE REPORTED IS NOT A LIST HERE.  It is read off the result
with :func:`dataclasses.fields`: every field whose name ends in
``_moisture_floor``.  A second floor added to ``RealInitResult`` -- the
prognostic-column floor the use_sh_qv lane needs, for one -- appears in
every proof this module writes with no edit here and no edit at the six
proof writers, which is the only arrangement under which the next floor
cannot be dropped the way the first one was.

WHY A MODULE RATHER THAN A HELPER IN ``real.py``.  The writers are proof
assemblers -- :mod:`gpuwm.era5_direct`, :mod:`gpuwm.gfs_direct`,
:mod:`gpuwm.mapped_direct` -- and a proof writer should not have to
import the GPU ingest to name what the ingest recorded.  This module
imports nothing but the standard library.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields as dataclass_fields, is_dataclass

#: The proof.json key for one domain's floors.  One spelling, so a reader
#: greps once and a bundle tool has one name to look for.
MOISTURE_FLOOR_KEY = "moisture_floors"

#: The multi-domain key, for a receipt covering a whole nest tree.  A
#: DIFFERENT NAME rather than the same one holding a different shape, the
#: rule :mod:`gpuwm.aerosol_source_receipt` set: a consumer reading
#: ``proof["moisture_floors"]["fired"]`` must not start throwing the
#: moment it meets a tree document.  Both names contain "moisture_floors",
#: so one grep still finds every preparation.
MOISTURE_FLOOR_BY_DOMAIN_KEY = "moisture_floors_by_domain"

#: Versioned because the block is a receipt others read mechanically.  It
#: is the BLOCK's version, not the document's: each proof keeps its own
#: schema string, and this key is added the way the aerosol receipt was
#: added before it.
MOISTURE_FLOOR_SCHEMA = "gpuwm-initialization-moisture-floors-v1"

#: The suffix that makes a ``RealInitResult`` field a moisture-floor
#: receipt.  Naming the convention once, here, is what lets the writers
#: stay ignorant of how many floors exist.
MOISTURE_FLOOR_SUFFIX = "_moisture_floor"


def _plain(value):
    """Recursively unwrap read-only containers into plain JSON types.

    A receipt can reach a writer through a prepared cache, which hands its
    metadata back as ``MappingProxyType`` so a restored document cannot be
    mutated.  ``json.dumps`` serializes ``dict`` and not ``Mapping``, so a
    proxy anywhere in the tree turns "write the proof" into a ``TypeError``
    naming nothing.
    """

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (str, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def moisture_floor_field_names(result) -> tuple[str, ...]:
    """Every ``*_moisture_floor`` field ``result`` declares.

    Read from :func:`dataclasses.fields` for the real result type, which
    is what makes a floor added later reported without an edit here.  A
    caller holding something else -- a restored view, a stand-in -- is
    answered from what the object actually exposes, and an object that
    exposes no floor is an empty tuple rather than an exception: "this
    result records no floor" is a state this module states, not one it
    crashes on.

    THE NON-DATACLASS FALLBACK IS ``dir``, NOT ``__dict__``.  A type using
    ``__slots__`` has no instance ``__dict__``, so reading that attribute
    reported ``recorded: false`` for an object that was holding floor
    receipts the whole time -- which is precisely the "not recorded" claim
    this module tells its callers never to guess at.  ``dir`` sees slots,
    class attributes and instance attributes alike, and each candidate is
    confirmed with :func:`hasattr` so a property that refuses to evaluate
    cannot turn "name the floors" into a traceback.

    Sorted, so two proofs written by two releases of the same floors
    compare key for key.
    """

    if is_dataclass(result) and not isinstance(result, type):
        names = [field.name for field in dataclass_fields(result)]
    else:
        try:
            names = list(dir(result))
        except Exception:
            names = list(getattr(result, "__dict__", ()) or ())
    return tuple(sorted(
        name for name in names
        if name.endswith(MOISTURE_FLOOR_SUFFIX) and hasattr(result, name)))


def moisture_floor_block(result, *, when_unrecorded) -> dict[str, object]:
    """The receipt block for one domain's initialization result.

    ``when_unrecorded`` is REQUIRED and has no default: it is the sentence
    this route can accurately give for a result it holds no floor receipt
    on.  There is no generic true answer, so each caller states its own.
    """

    names = moisture_floor_field_names(result)
    if not names:
        # ABSENCE (3): NOT RECORDED.  Deliberately no ``fired``: writing
        # ``False`` here would state that nothing was floored, which this
        # writer does not know.
        return {
            "schema": MOISTURE_FLOOR_SCHEMA,
            "recorded": False,
            "not_recorded_because": str(when_unrecorded),
        }
    floors: dict[str, object] = {}
    for name in names:
        receipt = _plain(getattr(result, name) or {})
        entry: dict[str, object] = {"fired": bool(receipt)}
        if receipt:
            entry["receipt"] = receipt
        floors[name] = entry
    return {
        "schema": MOISTURE_FLOOR_SCHEMA,
        "recorded": True,
        # The one boolean a reader filters a bundle on, stated so nobody
        # has to know the floor names to ask "was this run's vapour
        # modified on the way in".
        "fired": any(entry["fired"] for entry in floors.values()),
        "floors": floors,
    }


def moisture_floor_proof_entry(result, *, when_unrecorded) -> dict[str, object]:
    """``{MOISTURE_FLOOR_KEY: block}``, for a writer to merge in."""

    return {MOISTURE_FLOOR_KEY: moisture_floor_block(
        result, when_unrecorded=when_unrecorded)}


def moisture_floor_proof_entries(domains, *, when_unrecorded
                                 ) -> dict[str, object]:
    """``{MOISTURE_FLOOR_BY_DOMAIN_KEY: {label: block}}`` for a nest tree.

    ``domains`` is an iterable of ``(label, result)``.  Each domain gets
    its own answer because each domain has its own ``initialize_real``
    call: a root whose analyzed surface needed no floor and a child whose
    blended terrain produced one are two different facts about one
    forecast, and a tree-wide verdict would lose both.
    """

    blocks = {str(label): moisture_floor_block(
                  result, when_unrecorded=when_unrecorded)
              for label, result in domains}
    if not blocks:
        raise ValueError(
            "a hierarchy moisture-floor receipt needs at least the root "
            "domain's initialization result")
    return {MOISTURE_FLOOR_BY_DOMAIN_KEY: blocks}


__all__ = [
    "MOISTURE_FLOOR_BY_DOMAIN_KEY",
    "MOISTURE_FLOOR_KEY",
    "MOISTURE_FLOOR_SCHEMA",
    "MOISTURE_FLOOR_SUFFIX",
    "moisture_floor_block",
    "moisture_floor_field_names",
    "moisture_floor_proof_entries",
    "moisture_floor_proof_entry",
]

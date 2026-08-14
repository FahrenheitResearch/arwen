"""The full-state descriptor gate, armed over a streamed domain's HOST store.

Everything here runs with no card.  That is the point: the gate the resident
road runs is one CUDA launch over a domain-shaped device state, and the claim
this module has to support is that the STORE-DIRECT road -- which never builds
one -- can still be held to the same per-field bounds.  If that claim were only
checkable on a 16 GB card at 1024 x 1024 it would be a claim nobody could
falsify until it had already passed a forecast.

The instrument is tested in BOTH directions, per the house rule.  A healthy
store must pass and say how much it looked at; a store with one bad value in
one field must fail and NAME that field and that index.  A gate only ever
tested on healthy data is a gate whose failure path is unmeasured.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.health import (collect_state_fields, validate_fields_cpu,
                               validate_store_fields)


NZ, NY, NX, ROWS = 3, 8, 4, 2


class _Driver:
    def __init__(self, fields):
        self.fields = fields
        self.pbl_tendencies = None
        self.radiation_tendencies = None
        self.cumulus_tendencies = None
        self.microphysics = None


def _plane(shape, value=0.0):
    return np.full(shape, value, dtype=np.float32)


def _state(rows):
    """A state-shaped double at ``rows`` rows, and its carrier inventory.

    Shapes follow the model's staggering so the gate's domain-extent check is
    exercised by real ``ny + 1`` / ``nx + 1`` faces rather than by squares.
    Returns the state, the ``{carrier key: array}`` mapping a run's inventory
    rule would produce over it, and the two base-state auxiliaries.
    """

    class _State:
        pass

    state = _State()
    state.u = _plane((NZ, rows, NX + 1), 12.0)
    state.v = _plane((NZ, rows + 1, NX), -8.0)
    state.w = _plane((NZ + 1, rows, NX), 0.5)
    state.thp = _plane((NZ, rows, NX), 4.0)
    state.php = _plane((NZ + 1, rows, NX), 1000.0)
    state.mup = _plane((rows, NX), 200.0)
    state.p = _plane((NZ, rows, NX), 5.0e4)
    state.al = _plane((NZ, rows, NX), 0.1)
    state.alt = _plane((NZ, rows, NX), 0.9)
    state.qv = _plane((NZ, rows, NX), 0.01)
    state.qc = _plane((NZ, rows, NX), 0.0)
    # The two auxiliaries: total theta is thp + thb, coupled mass is mup + mub.
    state.thb = _plane((NZ, rows, NX), 296.0)
    state.mub2d = _plane((rows, NX), 9.0e4)
    state.physics = _Driver({"ust": _plane((rows, NX), 0.3),
                             "tsk": _plane((rows, NX), 290.0)})
    state._scratch = {"mp_rainnc": _plane((rows, NX), 1.5)}
    state._lateral_boundary_device = None
    carriers = {
        "state/u": state.u, "state/v": state.v, "state/w": state.w,
        "state/thp": state.thp, "state/php": state.php, "state/mup": state.mup,
        "state/p": state.p, "state/al": state.al, "state/alt": state.alt,
        "state/qv": state.qv, "state/qc": state.qc,
        "fields/ust": state.physics.fields["ust"],
        "fields/tsk": state.physics.fields["tsk"],
        "scratch/mp_rainnc": state._scratch["mp_rainnc"],
    }
    return state, carriers


def _store(**overrides):
    """A DOMAIN-shaped store carrying every carrier the template names."""
    template, carriers = _state(ROWS)
    store = {}
    for key, array in carriers.items():
        shape = array.shape[:-2] + (
            NY + (array.shape[-2] - ROWS), array.shape[-1])
        store[key] = np.full(shape, float(array.flat[0]), dtype=array.dtype)
    store.update(overrides)
    return store


def _auxiliaries():
    return {"thp": _plane((NZ, NY, NX), 296.0),
            "mup": _plane((NY, NX), 9.0e4)}


def _validate(store=None, template=None, carriers=None, auxiliaries=None,
              domain_shape=(NY, NX)):
    if template is None:
        template, carriers = _state(ROWS)
    return validate_store_fields(
        template, carriers, _store() if store is None else store,
        domain_shape=domain_shape,
        auxiliaries=_auxiliaries() if auxiliaries is None else auxiliaries,
        phase="initialized.d01")


# ---------------------------------------------------------------------------
# the gate passes, and says what it looked at
# ---------------------------------------------------------------------------

def test_a_healthy_store_passes_and_reports_its_coverage():
    report, coverage = _validate()
    assert report.ok is True
    assert report.status_bits == 0
    assert report.phase == "initialized.d01"
    # Every carrier the template names is a gated field here, so the gate
    # covered all of them and left nothing out.
    assert coverage["fields_checked"] == 14
    assert coverage["not_in_store"] == ()
    assert coverage["bytes_checked"] > 0


def test_the_gate_reads_the_store_and_not_the_slab_it_was_handed():
    """The count of ELEMENTS is the domain's, not the template's.

    The failure this whole seam exists to prevent is a verdict taken over one
    slab and read as a verdict over the analysis.  It is caught here by
    arithmetic rather than by trust: the elements the gate scanned must be the
    domain's count, which at ny=8 over 2-row slabs is four times the template's.
    """
    _, coverage = _validate()
    template, carriers = _state(ROWS)
    slab_elements = sum(int(a.size) for a in carriers.values())
    assert coverage["elements_checked"] > 3 * slab_elements


# ---------------------------------------------------------------------------
# ... and fails, naming the field, the index and the class
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,name,value,status_class", [
    ("state/w", "w", np.nan, "wind"),
    ("state/w", "w", 500.0, "wind"),
    ("state/qv", "qv", 1.5, "moisture"),
    ("state/qv", "qv", -1e-6, "moisture"),
    ("state/alt", "alt", 0.0, "specific_volume"),
    ("state/php", "php", 1.0e9, "geopotential"),
    ("fields/tsk", "surface.tsk", 40.0, "soil_temperature"),
])
def test_one_bad_value_anywhere_in_the_domain_fails_by_name(
        key, name, value, status_class):
    """Each rule class, tripped in the store, over the whole domain.

    The bad value is planted in the LAST row so a gate that quietly validated
    only the first slab's worth of rows would report a pass here.
    """
    from gpuwm.core.health import STATUS_BITS

    store = _store()
    store[key] = store[key].copy()
    store[key][(-1,) * store[key].ndim] = value
    report, _ = _validate(store=store)
    assert report.ok is False
    assert report.first_bad_field == name
    assert report.status_bits & STATUS_BITS[status_class]
    assert report.first_bad_index is not None
    assert report.reason


def test_the_coupled_mass_gate_reads_the_domains_base_state_not_the_slabs():
    """``mup`` is checked as ``mup + mub``, and mub is the DOMAIN's.

    Planting the failure in the auxiliary rather than in the field proves the
    substituted base state is really the one being added: a gate still using
    the template's slab-height ``mub2d`` could not see this at all.
    """
    auxiliaries = _auxiliaries()
    auxiliaries["mup"] = auxiliaries["mup"].copy()
    auxiliaries["mup"][NY - 1, NX - 1] = -9.0e4
    report, _ = _validate(auxiliaries=auxiliaries)
    assert report.ok is False
    assert report.first_bad_field == "mup"
    assert report.first_bad_index == (NY - 1, NX - 1)


# ---------------------------------------------------------------------------
# what it refuses rather than guesses
# ---------------------------------------------------------------------------

def test_a_slab_height_store_is_refused_rather_than_reported_as_the_domain():
    store = _store()
    store["state/thp"] = _plane((NZ, ROWS, NX), 4.0)
    with pytest.raises(ValueError, match="not this domain's"):
        _validate(store=store)


def test_a_missing_domain_auxiliary_is_refused_by_name():
    with pytest.raises(ValueError, match="domain-shaped substitute"):
        _validate(auxiliaries={"mup": _plane((NY, NX), 9.0e4)})


def test_a_gated_field_the_store_does_not_carry_is_named_not_skipped():
    """Absence is recorded, never inferred from a pass."""
    store = _store()
    del store["state/qc"]
    report, coverage = _validate(store=store)
    assert report.ok is True
    assert coverage["not_in_store"] == ("qc",)
    assert coverage["fields_checked"] == 13


def test_a_store_carrier_of_the_wrong_dtype_is_refused():
    store = _store()
    store["state/qv"] = store["state/qv"].astype(np.float64)
    with pytest.raises(ValueError, match="against a gated field"):
        _validate(store=store)


# ---------------------------------------------------------------------------
# the instrument, checked against the production mirror on the same numbers
# ---------------------------------------------------------------------------

def test_the_store_gate_agrees_with_the_state_gate_on_the_same_domain():
    """A known-answer check in both directions, against a DIFFERENT walker.

    ``validate_fields_cpu(collect_state_fields(domain_state))`` is the route a
    resident domain takes.  Built at full height with the same numbers, it must
    reach the same verdict as the store gate does over the store -- healthy on
    healthy data, and the same field on the same bad value.  Two roads, one
    rule set; that is the whole claim, and this is where it is falsifiable.
    """
    domain, domain_carriers = _state(NY)
    resident_ok = validate_fields_cpu(
        collect_state_fields(domain, backend="gpu"), phase="x")
    store_ok, _ = _validate()
    assert (resident_ok.ok, resident_ok.status_bits) == (True, 0)
    assert (store_ok.ok, store_ok.status_bits) == (True, 0)

    domain.qv[NZ - 1, NY - 1, NX - 1] = 2.0
    resident_bad = validate_fields_cpu(
        collect_state_fields(domain, backend="gpu"), phase="x")
    store = _store()
    store["state/qv"] = store["state/qv"].copy()
    store["state/qv"][NZ - 1, NY - 1, NX - 1] = 2.0
    store_bad, _ = _validate(store=store)
    assert resident_bad.ok is False and store_bad.ok is False
    assert resident_bad.first_bad_field == store_bad.first_bad_field == "qv"
    assert resident_bad.first_bad_index == store_bad.first_bad_index
    assert resident_bad.status_bits == store_bad.status_bits
    assert resident_bad.first_bad_value == store_bad.first_bad_value


def test_an_undeclared_integer_field_is_refused_exactly_as_the_kernel_does():
    """``gpu_integer_policy`` is the resident gate's classifier, and it is this
    gate's too: an integer field nobody has classified must stop the run rather
    than be admitted to a float32 gate on one road and refused on the other."""
    template, carriers = _state(ROWS)
    template.physics.fields["kpbl"] = np.zeros((ROWS, NX), dtype=np.int64)
    carriers["fields/kpbl"] = template.physics.fields["kpbl"]
    store = _store()
    store["fields/kpbl"] = np.zeros((NY, NX), dtype=np.int64)
    with pytest.raises(TypeError, match="unsupported dtype"):
        _validate(store=store, template=template, carriers=carriers)


def test_the_documented_integer_exclusions_are_excluded_here_too():
    template, carriers = _state(ROWS)
    template.physics.fields["ivgtyp"] = np.full((ROWS, NX), 7, dtype=np.int32)
    carriers["fields/ivgtyp"] = template.physics.fields["ivgtyp"]
    store = _store()
    store["fields/ivgtyp"] = np.full((NY, NX), 7, dtype=np.int32)
    report, coverage = _validate(store=store, template=template,
                                 carriers=carriers)
    assert report.ok is True
    assert "surface.ivgtyp" in coverage["excluded_integer_fields"]
    assert "surface.ivgtyp" not in coverage["not_in_store"]

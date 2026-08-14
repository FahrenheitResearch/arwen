"""The row-slab geometry of :mod:`gpuwm.ingest.prepared_store`, on the CPU.

Run it::

    python -m pytest tilestream/test_prepared_store.py

``store_from_prepared_cache`` reads a prepared cache one ROW SLAB at a time
so that no full-domain ``DomainState`` is ever allocated on the card.  Five
private helpers carry the whole of that geometry, and all five are pure
Python over NumPy shapes:

``_plan_slabs``
    which rows each slab owns.
``_row_window``
    which rows of a given array belong to a slab -- ``rows`` for a mass
    field, ``rows + 1`` for a y-staggered one, because the closing face at a
    slab's last row IS the next slab's first.
``_window_mapping``
    the same rule over a static/surface/met dict.
``_slab_base``
    the same rule over the ``BaseState``'s array fields, which under
    terrain are domain-shaped and were the one input this file used to
    watch a slab be handed WHOLE.
``_scatter_rows``
    the window written back into its place in the full-domain store, with
    the row span read off the source's own second-to-last extent.

WHY THIS FILE EXISTS SEPARATELY FROM THE PARITY GATE.  The loader's headline
claim -- that the slabbed road and the resident road produce bit-identical
frames -- needs a GPU, a cache on disk and a domain small enough for both.
The claim underneath it does not: **window-then-scatter must be the
identity**, for every field shape the store holds and every slab height the
plan produces, including the duplicated overlap row on staggered fields.
If that is false, no amount of parity testing on one domain size will say
which slab height broke it, and every failure mode here is SMOOTH -- a slab
written one row off, a closing face nobody owns, a y/x transposition -- so
the comparison is ``np.array_equal`` and the destination is poisoned with
NaN first.  ``tilestream.test_realdata``'s module docstring argues that at
length; this file is its CPU-only, GPU-free half.

WHAT IS NOT COVERED HERE, deliberately: everything that needs a device.
``_slab_state``, ``_boundaries_from_cache`` and ``store_from_prepared_cache``
itself build CuPy arrays and call into physics.  The only device contact this
file has is ``_scatter_rows``' ``import cupy`` (it asks CuPy two questions:
is this a device array, and if so hand me the host copy), which is answered
by a deliberately narrow stub -- see :class:`_StubCupy`.
"""

from __future__ import annotations

import dataclasses
import sys

import numpy as np
import pytest

from gpuwm.core.grid import BaseState
from gpuwm.ingest.prepared_store import (PreparedStoreError, _plan_slabs,
                                         _row_window, _scatter_rows,
                                         _slab_base, _window_mapping)


# --------------------------------------------------------------------------
# the synthetic domain, and why it has these numbers
# --------------------------------------------------------------------------

#: NY != NX so a y/x transposition cannot pass unnoticed, and NZ > 8 so a
#: 7-layer field cannot be read as a vertical one.  Both are
#: ``hoststore.manifest_from_arrays``' own refusals, and the loader derives
#: its manifest through that function, so a geometry this file tests at must
#: be one the loader could actually run at.  All three are prime, which also
#: means no slab height in ``SLAB_HEIGHTS`` divides NY except 1 and NY.
NZ, NY, NX = 11, 37, 23

#: Slab heights every round trip runs at.  ``1`` is 37 slabs -- the most
#: seams there are, every field written 37 times.  ``8`` does not divide 37,
#: so the last slab is short (5 rows), which is the case
#: ``build_store_by_slabs`` refuses outright and this loader must handle.
#: ``NY`` is one slab: the monolithic shape reached through the slab code
#: path, the control that says the slab machinery is not itself changing the
#: answer.  ``NY + 3`` is a rows_per_slab larger than the domain.
SLAB_HEIGHTS = (1, 4, 8, 13, NY, NY + 3)

#: One field per shape the store models, at DOMAIN height, keyed by the
#: staggering it stands for.  The names are the contract's own
#: (``hoststore.FieldSpec.stagger_code``) so a failure names a real field
#: class rather than a tuple.
CARRIER_SHAPES = {
    "2d mass": ((NY, NX), np.float32),
    "3d mass": ((NZ, NY, NX), np.float32),
    "u (x-staggered)": ((NZ, NY, NX + 1), np.float32),
    "v (y-staggered)": ((NZ, NY + 1, NX), np.float32),
    "w (z-staggered)": ((NZ + 1, NY, NX), np.float32),
    "L7 (snow-soil layers)": ((7, NY, NX), np.float64),
}

#: Rank-4 shapes.  The store manifest refuses these (``manifest_from_arrays``
#: models 2-D and 3-D fields only), but ``_row_window`` is applied to the
#: static/surface/met mappings as well as to carriers, and its rule is "the
#: last two axes are horizontal" at any rank.  A 4-D field windowed on the
#: wrong axis is exactly the silent shifted-copy failure the helper's
#: docstring is written against, so both a mass and a staggered rank-4 case
#: run the full round trip.
RANK4_SHAPES = {
    "4d mass": ((4, NZ, NY, NX), np.float64),
    "4d y-staggered": ((2, NZ, NY + 1, NX), np.float64),
}


def _ramp(shape, dtype=np.float64) -> np.ndarray:
    """Every element distinct, and exactly representable in ``dtype``.

    Distinctness is the point: a slab written one row off, or a window taken
    on the wrong axis, changes values everywhere rather than in a place that
    a coarser pattern might happen to agree on.  ``np.arange`` stays exact in
    float32 to 2**24 and these arrays are far smaller than that.
    """
    count = int(np.prod(shape))
    return np.arange(count, dtype=np.float64).astype(dtype).reshape(shape)


def _poisoned_like(array: np.ndarray) -> np.ndarray:
    """A destination in which an unwritten row is a NaN, never a zero.

    ``tilestream.test_realdata``'s idiom.  Zero is a plausible value for
    most of these fields, so a hole left by a mis-planned slab would pass a
    comparison against a zeroed reference; a NaN cannot, because
    ``np.array_equal`` is False for NaN against anything including itself.
    """
    return np.full(array.shape, np.nan, dtype=array.dtype)


# --------------------------------------------------------------------------
# the narrowest possible CuPy
# --------------------------------------------------------------------------

class _DeviceArray:
    """Stands in for ``cupy.ndarray``, and for nothing else.

    ``_scatter_rows`` accepts either a device array or a host one and is the
    only helper here that must tell them apart.  A NumPy array cannot play
    the device half of that test (it would take the ``np.asarray`` branch),
    so the device half needs a type that is not ``np.ndarray``.
    """

    def __init__(self, host) -> None:
        self.host = np.asarray(host)


class _StubCupy:
    """A CuPy that cannot be mistaken for the real one.

    ``_scatter_rows`` asks the module exactly two things: ``ndarray`` (for
    the isinstance) and ``asnumpy`` (for the copy).  Anything else it reaches
    for is a change in what it needs from the device, and a change in what a
    host-side helper needs from the device is the kind of thing this lane
    wants to hear about immediately -- so unknown attributes raise here
    rather than being answered with a MagicMock's silence.
    """

    ndarray = _DeviceArray

    @staticmethod
    def asnumpy(array):
        assert isinstance(array, _DeviceArray), (
            f"cupy.asnumpy called on {type(array).__name__}, which is not a "
            "device array; the isinstance guard let a host array through")
        return array.host.copy()

    def __getattr__(self, name):
        # The import machinery itself asks a sys.modules entry for __spec__
        # and friends before the importing code sees it, so the dunders have
        # to answer the ordinary "no such attribute" rather than the loud
        # one -- otherwise the stub fails the import instead of the test.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        raise AssertionError(                       # pragma: no cover
            f"_scatter_rows reached for cupy.{name}, which this stub does "
            "not model; it is documented as needing only ndarray + asnumpy")


@pytest.fixture()
def stub_cupy(monkeypatch):
    """CuPy, for the two questions ``_scatter_rows`` asks it.

    The loader runs on a card; this file does not, and CuPy is not installed
    where it runs.  The substitution is by ``sys.modules`` because the import
    is INSIDE the function (``prepared_store.py``, ``_scatter_rows``), which
    no attribute patch can reach -- the same reason
    ``tests/test_lateral_bc.py`` does it this way.
    """
    monkeypatch.setitem(sys.modules, "cupy", _StubCupy())


# --------------------------------------------------------------------------
# _plan_slabs: a partition, not an overlay
# --------------------------------------------------------------------------

@pytest.mark.parametrize("ny", [1, 2, 7, 37, 64, 200])
@pytest.mark.parametrize("rows_per_slab", [1, 3, 8, 37, 64, 205])
def test_slab_plans_partition_the_domain_exactly(ny, rows_per_slab):
    """Every mass row owned by exactly one slab, over 36 (ny, rows) pairs.

    Written as the concatenation of the owned ranges compared against
    ``arange(ny)``, not as a count: a plan that owns the right NUMBER of rows
    in the wrong ORDER, or that owns row 5 twice and row 6 never, has the
    right count.
    """
    slabs = _plan_slabs(ny, rows_per_slab)

    owned = np.concatenate([np.arange(j0, j0 + rows) for j0, rows in slabs])
    assert np.array_equal(owned, np.arange(ny)), (
        f"ny={ny} rows_per_slab={rows_per_slab}: owned rows {owned.tolist()} "
        f"are not a partition of [0, {ny})")
    assert all(1 <= rows <= rows_per_slab for _, rows in slabs)
    assert slabs[0][0] == 0
    assert len(slabs) == -(-ny // rows_per_slab)      # ceil, exactly
    # Only the LAST slab may be short; a short slab in the middle would still
    # partition the domain and would still be wrong.
    assert all(rows == min(rows_per_slab, ny) for _, rows in slabs[:-1])


def test_a_slab_height_that_does_not_divide_ny_gives_a_short_last_slab():
    """Stated as the literal plan, because this is the case bigdomain refuses.

    ``build_store_by_slabs`` raises unless ``slab_rows`` divides ``ny`` --
    it also sizes an analytic noise field by the slab height, so it can
    afford to.  This loader takes its rows off disk and cannot, so the
    short last slab is the normal case rather than an edge one: 37 rows at
    8 per slab is four full slabs and a five-row remainder.
    """
    assert _plan_slabs(37, 8) == ((0, 8), (8, 8), (16, 8), (24, 8), (32, 5))
    assert _plan_slabs(40, 8) == ((0, 8), (8, 8), (16, 8), (24, 8), (32, 8))
    assert _plan_slabs(1, 8) == ((0, 1),)


def test_a_slab_taller_than_the_domain_is_one_slab_of_the_domain():
    """rows_per_slab > ny must not produce a slab that runs off the end."""
    assert _plan_slabs(NY, NY + 3) == ((0, NY),)
    assert _plan_slabs(NY, 10_000) == ((0, NY),)


@pytest.mark.parametrize("rows_per_slab", [0, -1, -64])
def test_a_nonsensical_slab_height_becomes_one_row_rather_than_a_hang(
        rows_per_slab):
    """``max(1, rows_per_slab)`` is a termination guarantee, not a nicety.

    The loop advances ``j0`` by ``rows_per_slab``; at zero or negative it
    would never reach ``ny``.  Held so nobody removes the clamp as dead code.
    """
    slabs = _plan_slabs(5, rows_per_slab)
    assert slabs == ((0, 1), (1, 1), (2, 1), (3, 1), (4, 1))


@pytest.mark.parametrize("ny", [0, -1])
def test_a_domain_with_no_rows_is_refused_naming_the_count(ny):
    with pytest.raises(PreparedStoreError) as excinfo:
        _plan_slabs(ny, 8)
    assert f"{ny} rows" in str(excinfo.value)


# --------------------------------------------------------------------------
# _row_window: mass rows, and one more face
# --------------------------------------------------------------------------

def test_a_mass_field_gives_up_exactly_its_rows():
    mass = _ramp((NZ, NY, NX))
    window = _row_window(mass, 8, 5, NY)
    assert window.shape == (NZ, 5, NX)
    assert np.array_equal(window, mass[:, 8:13, :])


def test_a_y_staggered_field_gives_up_one_more_row_than_it_is_asked_for():
    """The closing face: slab k's last row IS slab k+1's first.

    Not an off-by-one to be tidied away.  A field of height ``ny + 1`` has a
    face BETWEEN every pair of mass rows plus one at each end, so a slab that
    took only ``rows`` of them would leave the face above its top row to
    whoever comes next -- and the last slab would leave the domain's closing
    face to nobody at all.
    """
    stag = _ramp((NZ, NY + 1, NX))
    window = _row_window(stag, 8, 5, NY)
    assert window.shape == (NZ, 6, NX)
    assert np.array_equal(window, stag[:, 8:14, :])

    # The last slab's window reaches the closing face and stops there.
    j0, rows = _plan_slabs(NY, 8)[-1]
    last = _row_window(stag, j0, rows, NY)
    assert last.shape == (NZ, rows + 1, NX)
    assert np.array_equal(last[:, -1, :], stag[:, NY, :])


def test_x_staggering_is_not_the_helper_s_business():
    """The window is the second-to-last axis only; x passes through whole."""
    xstag = _ramp((NZ, NY, NX + 1))
    window = _row_window(xstag, 4, 6, NY)
    assert window.shape == (NZ, 6, NX + 1)
    assert np.array_equal(window, xstag[:, 4:10, :])


def test_a_field_with_no_horizontal_extent_is_returned_untouched():
    """The eta table and its friends are the same at every slab height.

    ``ndim < 2`` is the loader's test for "does not vary by row", and those
    arrays must come back as they went in -- windowing ``znu`` by row would
    hand a slab a truncated vertical coordinate.
    """
    eta = _ramp((NZ,))
    assert _row_window(eta, 8, 5, NY) is eta
    zero_d = np.array(3.5)
    assert _row_window(zero_d, 8, 5, NY) is zero_d


def test_a_missing_field_stays_missing():
    assert _row_window(None, 0, 4, NY) is None


@pytest.mark.parametrize("value", ["MODIFIED_IGBP_MODIS_NOAH", b"raw",
                                   True, 17, 3.5])
def test_a_scalar_beside_the_fields_comes_back_as_itself(value):
    """A static bundle is not all arrays, and the difference is load-bearing.

    ``MMINLU`` is a string naming the land-use table; a slab whose static
    mapping held ``np.asarray("MODIS...")`` instead would still be a mapping
    of the right keys, and the 0-d array would still compare equal to the
    string in most places -- which is exactly the kind of near-miss that
    reaches the LANDUSE.TBL lookup before anyone notices.  So identity, not
    equality, is the assertion.
    """
    assert _row_window(value, 8, 5, NY) is value


@pytest.mark.parametrize("height", [NY - 1, NY + 2, 2 * NY, 1])
def test_a_height_that_is_neither_mass_nor_face_is_refused_loudly(height):
    """Silence here is the shifted-copy failure, so the message names both.

    The ``height == rows`` case in particular is a real mistake with a
    plausible shape: windowing an ALREADY-windowed array a second time.  It
    is refused rather than producing a slab-of-a-slab.
    """
    bad = _ramp((NZ, height, NX))
    with pytest.raises(PreparedStoreError) as excinfo:
        _row_window(bad, 0, 1, NY)
    message = str(excinfo.value)
    assert f"height {height}" in message, message
    assert f"{NY} mass rows" in message, message


# --------------------------------------------------------------------------
# _window_mapping: the same rule, over a dict
# --------------------------------------------------------------------------

def test_a_mapping_is_windowed_field_by_field_on_its_own_staggering():
    """A static mapping holds all three kinds at once, and MAPFAC_V is why.

    ``MAPFAC_M``/``F``/``E`` are mass, ``MAPFAC_U`` is x-staggered (mass
    height), ``MAPFAC_V`` is y-staggered -- one dict, three answers.
    """
    static = {
        "MAPFAC_M": _ramp((NY, NX)),
        "MAPFAC_U": _ramp((NY, NX + 1)),
        "MAPFAC_V": _ramp((NY + 1, NX)),
        "GREENFRAC": _ramp((12, NY, NX)),
        "ZNU": _ramp((NZ,)),
        "SNOWH": None,
    }
    got = _window_mapping(static, 8, 5, NY)

    assert sorted(got) == sorted(static)
    assert got["MAPFAC_M"].shape == (5, NX)
    assert got["MAPFAC_U"].shape == (5, NX + 1)
    assert got["MAPFAC_V"].shape == (6, NX)
    assert got["GREENFRAC"].shape == (12, 5, NX)
    assert got["ZNU"] is static["ZNU"]
    assert got["SNOWH"] is None
    assert np.array_equal(got["MAPFAC_V"], static["MAPFAC_V"][8:14, :])


def test_an_absent_mapping_is_not_an_empty_one():
    """``surface_fields`` is optional in the cache metadata; None must survive.

    An empty dict and a missing mapping mean different things downstream --
    one is "this cache carries no surface fields", the other is "nobody
    passed the surface fields" -- so the helper must not conflate them.
    """
    assert _window_mapping(None, 0, 4, NY) is None
    assert _window_mapping({}, 0, 4, NY) == {}


# --------------------------------------------------------------------------
# _slab_base: THE BASE STATE IS DOMAIN-SHAPED UNDER TERRAIN
# --------------------------------------------------------------------------
#
# WHY THIS SECTION EXISTS.  ``_slab_state`` used to hand each slab the
# DOMAIN's ``BaseState``, on a docstring claim that the base was
# "domain-invariant".  That claim is true only for FLAT terrain, where
# ``mub`` is a scalar and the profiles are 1-D columns that broadcast at any
# height.  With terrain ``BaseState`` is ``mub (ny, nx)`` and full 3-D
# profiles, and ``DomainState.load_base`` assigns each into a slab-shaped
# device array -- so the real-data gate died with "operands could not be
# broadcast together with shapes (49, 384, 384) (49, 64, 384)" three runs
# running while every test in this file stayed green.
#
# The tests below are the ones that would not have.  They use ny = NY = 37
# with slab heights that do NOT contain it, because the whole reason a flat
# 1-D base and a one-slab domain hid this is that both make the broadcast
# succeed; and the last of them reproduces load_base's assignment on the
# CPU with plain NumPy, which raises the same ValueError for the same
# reason and needs no card to say so.

def _terrain_base() -> BaseState:
    """A base state as a REAL case carries it: 2-D mass, 3-D profiles.

    The shapes are ``BaseState``'s own documented terrain shapes, and every
    element is distinct (``_ramp``) so a window taken at the wrong offset
    cannot compare equal to the right one.
    """
    return BaseState(
        mub=_ramp((NY, NX)),
        p_top=5000.0,
        pb=_ramp((NZ, NY, NX)),
        alb=_ramp((NZ, NY, NX)),
        thb=_ramp((NZ, NY, NX)),
        phb=_ramp((NZ + 1, NY, NX)),
        terrain_z=_ramp((NY, NX)))


def _flat_base() -> BaseState:
    """Phase 1's base state: a scalar dry mass and 1-D columns.

    What every existing store test has run against, and the reason the
    terrain case went unnoticed.
    """
    return BaseState(
        mub=98000.0, p_top=5000.0,
        pb=_ramp((NZ,)), alb=_ramp((NZ,)), thb=_ramp((NZ,)),
        phb=_ramp((NZ + 1,)), terrain_z=None)


@pytest.mark.parametrize("rows_per_slab", [1, 8, 13])
def test_a_terrain_base_state_is_windowed_field_by_field_for_every_slab(
        rows_per_slab):
    """Each slab gets ITS OWN rows of mub, the profiles and the terrain.

    Checked slab by slab against the source's own slice rather than against
    a shape alone: a base windowed at the wrong offset has exactly the right
    shape, and would give every slab the same plausible-looking surface
    pressure field from the wrong part of the domain.
    """
    base = _terrain_base()

    for j0, rows in _plan_slabs(NY, rows_per_slab):
        slab = _slab_base(base, j0, rows, NY)

        assert slab.mub.shape == (rows, NX)
        assert np.array_equal(slab.mub, base.mub[j0:j0 + rows, :])
        assert slab.terrain_z.shape == (rows, NX)
        assert np.array_equal(slab.terrain_z,
                              base.terrain_z[j0:j0 + rows, :])
        for name in ("pb", "alb", "thb"):
            got = getattr(slab, name)
            assert got.shape == (NZ, rows, NX), name
            assert np.array_equal(got, getattr(base, name)[
                :, j0:j0 + rows, :]), name
        assert slab.phb.shape == (NZ + 1, rows, NX)
        assert np.array_equal(slab.phb, base.phb[:, j0:j0 + rows, :])
        # p_top has no horizontal extent and must come through as itself,
        # not as a 0-d array: DomainState.load_base does DTYPE(base.p_top).
        assert slab.p_top is base.p_top


@pytest.mark.parametrize("rows_per_slab", [1, 8, 13])
def test_no_array_field_of_a_windowed_base_still_has_the_domains_height(
        rows_per_slab):
    """Derived from ``dataclasses.fields``, so a NEW field cannot be missed.

    The assertion is over every field the dataclass declares rather than
    over the six this file happens to know about: the failure mode being
    guarded is a field added to ``BaseState`` that nobody remembers to
    window, which is the original defect again in a new array.  Any array
    field left at domain height would be handed to ``load_base`` whole.
    """
    base = _terrain_base()

    for j0, rows in _plan_slabs(NY, rows_per_slab):
        slab = _slab_base(base, j0, rows, NY)
        checked = 0
        for spec in dataclasses.fields(base):
            value = getattr(slab, spec.name)
            if not isinstance(value, np.ndarray) or value.ndim < 2:
                continue
            checked += 1
            assert value.shape[-2] in (rows, rows + 1), (
                f"{spec.name} came out {value.shape[-2]} rows tall for a "
                f"{rows}-row slab; it was not windowed")
        assert checked >= 6, (
            "the terrain base state should present at least six array "
            f"fields with horizontal extent, presented {checked}")


def test_windowing_a_base_state_does_not_disturb_the_domains_own():
    """The bundle publishes the WHOLE-domain base, so it must survive.

    ``store_from_prepared_cache`` returns ``base=base`` -- the un-windowed
    object -- after building every slab from copies of it.  A helper that
    mutated the original in place would leave the bundle carrying a base
    state one slab tall, which is the same defect wearing the other hat.
    """
    base = _terrain_base()
    before = {spec.name: getattr(base, spec.name)
              for spec in dataclasses.fields(base)}

    _slab_base(base, 8, 13, NY)

    for name, value in before.items():
        now = getattr(base, name)
        assert now is value, f"{name} was replaced on the domain's base"
        if isinstance(value, np.ndarray):
            assert value.shape[-2] in (NY, NY + 1) or value.ndim < 2


def test_a_flat_base_state_comes_through_exactly_as_it_went_in():
    """The Phase 1 base is all scalars and columns; none of it is windowed.

    This is what makes the fix safe for the flat road that the parity gate
    was green on before terrain existed: ``_row_window`` returns scalars and
    ``ndim < 2`` arrays as THEMSELVES, so identity holds field by field.
    """
    base = _flat_base()
    slab = _slab_base(base, 8, 13, NY)

    for spec in dataclasses.fields(base):
        assert getattr(slab, spec.name) is getattr(base, spec.name), spec.name


def test_the_unwindowed_base_is_what_load_base_refused_to_broadcast():
    """The gate's own failure, reproduced on the CPU by the same operation.

    ``DomainState.load_base`` copies each profile with ``dev[...] =``, so
    the question it asks NumPy (and CuPy) is whether a domain-height array
    broadcasts into a slab-height one.  It does not, and the message names
    both shapes -- which is exactly what the real-data gate reported at
    384 x 384 with 64-row slabs.  Held here as the discriminating control:
    without it a passing window test proves only that the window is
    self-consistent, not that the un-windowed base was ever a problem.
    """
    base = _terrain_base()
    rows = 8
    device_side = np.zeros((NZ, rows, NX), dtype=np.float32)

    with pytest.raises(ValueError) as excinfo:
        device_side[...] = base.pb                     # the defect
    assert f"{NY}" in str(excinfo.value), str(excinfo.value)

    device_side[...] = _slab_base(base, 0, rows, NY).pb     # the fix
    assert np.array_equal(device_side,
                          base.pb[:, :rows, :].astype(np.float32))


def test_a_base_state_that_is_not_a_dataclass_is_refused_by_name():
    """The derivation is only as safe as the enumeration it rests on.

    ``_slab_base`` reads its field list off ``dataclasses.fields``.  Handed
    something else it must say so rather than return the object unwindowed,
    which would put the original defect back silently.
    """
    class _NotADataclass:
        mub = np.zeros((NY, NX))

    with pytest.raises(PreparedStoreError) as excinfo:
        _slab_base(_NotADataclass(), 0, 8, NY)
    assert "_NotADataclass" in str(excinfo.value)


def test_a_base_state_of_the_wrong_height_is_refused_by_the_shared_rule():
    """One rule, one refusal: the base goes through ``_row_window`` too.

    A base built for a different domain is neither mass- nor y-staggered
    against this one, and windowing it silently would hand every slab a
    shifted copy of somebody else's atmosphere.
    """
    base = dataclasses.replace(_terrain_base(), pb=_ramp((NZ, NY + 5, NX)))
    with pytest.raises(PreparedStoreError) as excinfo:
        _slab_base(base, 0, 8, NY)
    assert f"height {NY + 5}" in str(excinfo.value)


# --------------------------------------------------------------------------
# the store is harvested by the rule the tile buffers are built by
# --------------------------------------------------------------------------

class _StubStreamedState:
    """The two questions the streamed inventory rule asks a state.

    ``streaming_inventory`` short-circuits on an object exposing ``arrays``
    (that is the store/mapping branch, and it is what lets this run without
    a ``DomainState`` or a card), and ``refl_inventory`` asks
    ``existing_scratch`` for the REBUILT reflectivity slot.  Nothing else is
    modelled, deliberately -- a rule that started needing more of a state
    than this is a change worth failing on.
    """

    def __init__(self, carriers, refl):
        self.arrays = dict(carriers)
        self._refl = refl

    def existing_scratch(self, name):
        return self._refl if name == "refl_10cm" else None


def test_the_streamed_rule_carries_the_refl_slot_the_plain_one_omits():
    """The store/buffer disagreement that stopped the run, in two calls.

    ``scratch/refl_10cm`` is REBUILT scratch: microphysics computes it when
    a frame is due and nothing carries it between steps, so
    ``carrier_inventory`` excludes it BY CONSTRUCTION.  A store harvested
    with that rule and buffers built with the run's own
    ``streamed_store_inventory`` therefore differ by exactly one key, and
    ``TiledRun`` refuses the pair: "tile state inventory [143] != store
    inventory [142] ... in TILE not in STORE: ['scratch/refl_10cm']".  Both
    rules are asked here about the SAME object so that the difference
    between them is the only thing measured.
    """
    from gpuwm.core.streaming import REFL_STORE_KEY, streamed_store_inventory
    from tilestream import physics_inventory as physinv

    state = _StubStreamedState(
        {"state/u": _ramp((NZ, NY, NX + 1), np.float32),
         "state/t": _ramp((NZ, NY, NX), np.float32)},
        refl=_ramp((NZ, NY, NX), np.float32))

    plain = physinv.carrier_inventory(state)
    streamed = streamed_store_inventory()(state, None)

    assert REFL_STORE_KEY not in plain
    assert REFL_STORE_KEY in streamed
    assert set(streamed) - set(plain) == {REFL_STORE_KEY}, (
        "the two rules must differ in the REFL slot and nothing else")
    # A state that never primed the slot keeps it absent under both rules,
    # so the refusal in StreamedDomain.__call__ can still fire.
    bare = _StubStreamedState(dict(state.arrays), refl=None)
    assert REFL_STORE_KEY not in streamed_store_inventory()(bare, None)


# --------------------------------------------------------------------------
# _scatter_rows: the span comes from the source
# --------------------------------------------------------------------------

def test_the_scatter_reads_its_row_span_off_the_source(stub_cupy):
    """Six rows written at j0 means rows 8..13, and nothing else, changed."""
    dst = _poisoned_like(_ramp((NZ, NY, NX)))
    src = _ramp((NZ, 6, NX), dtype=dst.dtype)
    _scatter_rows(dst, src, 8)

    assert np.array_equal(dst[:, 8:14, :], src)
    untouched = np.concatenate([dst[:, :8, :], dst[:, 14:, :]], axis=1)
    assert np.isnan(untouched).all(), (
        "the scatter wrote outside the span its source declared")


def test_the_scatter_takes_a_device_array_through_asnumpy(stub_cupy):
    """The branch that exists because physics hands back device arrays.

    ``build_store_by_slabs`` converts at the call site
    (``_scatter_rows(store[name], cp.asnumpy(arr), j0)``); this loader
    converts inside the helper instead, so the helper -- not its caller --
    is what has to know a device array when it sees one.
    """
    dst = _poisoned_like(_ramp((NZ, NY, NX)))
    rows = _ramp((NZ, 6, NX), dtype=dst.dtype)
    _scatter_rows(dst, _DeviceArray(rows), 8)
    assert np.array_equal(dst[:, 8:14, :], rows)


def test_the_scatter_refuses_to_run_off_the_end_of_the_destination(stub_cupy):
    """A window that does not fit raises; it does not write a truncated one.

    NumPy's slicing is forgiving -- ``dst[..., 34:44, :]`` is silently six
    rows -- but the ASSIGNMENT is not, and that is what stands between a
    mis-planned slab and a store holding a quietly truncated field.
    """
    dst = _poisoned_like(_ramp((NZ, NY, NX)))
    src = _ramp((NZ, 10, NX), dtype=dst.dtype)
    with pytest.raises(ValueError):
        _scatter_rows(dst, src, NY - 3)


# --------------------------------------------------------------------------
# THE PROPERTY THAT MATTERS: window-then-scatter is the identity
# --------------------------------------------------------------------------

def _round_trip(source: np.ndarray, rows_per_slab: int, *,
                mass_rule: bool = False) -> np.ndarray:
    """Window every slab of ``source`` and scatter the windows back.

    ``mass_rule=True`` is the negative control: the staggering ignored, so
    every field is windowed as if it were mass-height.  That is
    ``test_realdata``'s ``force_kind='mass'``, and on a y-staggered field it
    leaves the closing face at row ``ny`` written by nobody.
    """
    dst = _poisoned_like(source)
    for j0, rows in _plan_slabs(NY, rows_per_slab):
        if mass_rule:
            window = source[..., j0:j0 + rows, :]
        else:
            window = _row_window(source, j0, rows, NY)
        _scatter_rows(dst, window, j0)
    return dst


@pytest.mark.parametrize("rows_per_slab", SLAB_HEIGHTS)
@pytest.mark.parametrize("label", sorted(CARRIER_SHAPES))
def test_window_then_scatter_reproduces_a_carrier_exactly(
        label, rows_per_slab, stub_cupy):
    """Bit-exact, for every store field shape at every slab height.

    ``np.array_equal`` and not ``allclose``: every failure mode in this lane
    is smooth.  A slab written one row off, a window taken on the wrong axis
    and a closing face nobody owns all produce a finite, correctly-shaped
    field that differs from the truth by a fraction of a percent, and
    ``allclose`` passes all three.
    """
    shape, dtype = CARRIER_SHAPES[label]
    source = _ramp(shape, dtype)
    got = _round_trip(source, rows_per_slab)

    assert got.shape == source.shape
    assert np.array_equal(got, source), (
        f"{label} at rows_per_slab={rows_per_slab}: "
        f"{int((got != source).sum())} of {source.size} elements differ, "
        f"{int(np.isnan(got).sum())} never written")


@pytest.mark.parametrize("rows_per_slab", SLAB_HEIGHTS)
@pytest.mark.parametrize("label", sorted(RANK4_SHAPES))
def test_window_then_scatter_reproduces_a_rank_4_field_exactly(
        label, rows_per_slab, stub_cupy):
    """The rule is "the last two axes are horizontal", at any rank."""
    shape, dtype = RANK4_SHAPES[label]
    source = _ramp(shape, dtype)
    got = _round_trip(source, rows_per_slab)

    assert got.shape == source.shape
    assert np.array_equal(got, source), (
        f"{label} at rows_per_slab={rows_per_slab}: "
        f"{int(np.isnan(got).sum())} elements never written")


def test_the_overlap_row_on_a_staggered_field_is_a_duplicate_not_a_conflict(
        stub_cupy):
    """Both slabs write the shared face, and both write the same number.

    This is the claim ``_row_window``'s docstring makes, and it is the one
    that makes the ``rows + 1`` window safe: slab k writes rows
    ``[j0, j0 + rows]`` of a y-staggered field and slab k+1 starts at
    ``j0 + rows``, so the seam row is written twice.  Checked by watching the
    writes rather than only the result -- a result that happens to be right
    would not distinguish "written twice with the same value" from "written
    once".
    """
    source = _ramp((NZ, NY + 1, NX))
    dst = _poisoned_like(source)
    writes = np.zeros(NY + 1, dtype=int)
    slabs = _plan_slabs(NY, 8)
    for j0, rows in slabs:
        window = _row_window(source, j0, rows, NY)
        _scatter_rows(dst, window, j0)
        writes[j0:j0 + rows + 1] += 1

    seams = [j0 for j0, _ in slabs[1:]]
    assert seams and all(writes[j] == 2 for j in seams), writes.tolist()
    assert writes[0] == 1 and writes[NY] == 1
    assert np.array_equal(dst, source)


@pytest.mark.parametrize("rows_per_slab", [1, 8, 13])
def test_control_the_mass_rule_on_a_staggered_field_orphans_the_closing_face(
        rows_per_slab, stub_cupy):
    """The negative control: without it the round trip above proves nothing.

    Ignore the staggering and the last slab stops one row short of the
    domain's closing face, which no other slab reaches either.  The whole
    row stays at the poison NaN -- and note that a ZEROED destination would
    have hidden it, which is why the destination is poisoned.
    """
    source = _ramp((NZ, NY + 1, NX))
    got = _round_trip(source, rows_per_slab, mass_rule=True)

    assert not np.array_equal(got, source)
    assert np.isnan(got[:, NY, :]).all(), (
        "the closing face was written after all; the control is vacuous")
    assert np.array_equal(got[:, :NY, :], source[:, :NY, :]), (
        "only the closing face should be missing")


def test_control_the_comparison_would_have_caught_a_single_wrong_element(
        stub_cupy):
    """One perturbed element flips the check, so the check discriminates."""
    source = _ramp((NZ, NY, NX))
    got = _round_trip(source, 8)
    assert np.array_equal(got, source)

    got[NZ // 2, 19, NX // 2] += 1.0
    assert not np.array_equal(got, source)


# --------------------------------------------------------------------------
# the manifest the loader sizes its store from, and the windows that fill it
# --------------------------------------------------------------------------

def _at_slab_height(shape, rows: int) -> tuple[int, ...]:
    """``shape`` with its height taken from a slab, not from the domain."""
    height = shape[-2]
    assert height in (NY, NY + 1), height
    return shape[:-2] + (rows + (height - NY), shape[-1])


def test_the_manifest_a_slab_reveals_re_expands_to_the_domain_it_came_from(
        stub_cupy):
    """The loader's two halves, checked against each other on the CPU.

    ``store_from_prepared_cache`` sizes the whole store from the FIRST
    slab's inventory -- ``manifest_from_arrays(inv, nz, rows, nx)`` -- and
    then fills it with windows of the domain.  Those are two independent
    pieces of shape arithmetic and nothing else compares them: if the
    manifest re-expands a y-staggered field to ``ny`` instead of ``ny + 1``,
    the store is allocated one row short and the failure surfaces as a
    broadcast error deep inside the fill, on a machine that has already
    pinned the store.

    The geography half uses a different expression again
    (``ny + (arr.shape[-2] - rows)``, written inline in the loader), so it is
    transcribed and checked on the same shapes.
    """
    from tilestream import hoststore

    rows_per_slab = 8
    first_rows = _plan_slabs(NY, rows_per_slab)[0][1]
    inventory = {
        label: np.zeros(_at_slab_height(shape, first_rows), dtype)
        for label, (shape, dtype) in CARRIER_SHAPES.items()}

    manifest = hoststore.manifest_from_arrays(inventory, NZ, first_rows, NX)
    assert len(manifest) == len(CARRIER_SHAPES)

    for spec in manifest:
        want, dtype = CARRIER_SHAPES[spec.name]
        assert spec.shape(NZ, NY, NX) == want, spec
        assert spec.dtype == np.dtype(dtype)
        # the geography sizing expression, on the same array
        slab_shape = inventory[spec.name].shape
        assert NY + (slab_shape[-2] - first_rows) == want[-2]

    # And the store those specs describe is exactly the store the windows
    # fill: allocate from the manifest, fill from the domain, compare.
    for spec in manifest:
        source = _ramp(spec.shape(NZ, NY, NX), spec.dtype)
        assert np.array_equal(_round_trip(source, rows_per_slab), source), (
            f"{spec.name} ({spec.stagger_code}) did not survive the round "
            f"trip into the store its own manifest sized")

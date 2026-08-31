"""The prepared cache's SOLVED soil, from the writer to physics setup.

A prepared cache stores the canonical Noah surface it solved, and the
writer therefore drops the native ``SOILT``/``SOILW`` pair from the met
contract stored beside it -- the surface IS the soil statement of a
portable cache.  Every consumer of a restore has to consume that
surface; one that re-derives instead is asking for ingredients the
writer deliberately did not keep.

MEASURED 2026-08-18, Linux shakeout, the benchmark's ``--prepared-cache``
restore arm::

    File "gpuwm/ingest/soil.py", line 197, in _require_same_shape
      raise KeyError(f"missing soil input field(s): {missing}")
    KeyError: "missing soil input field(s): ['ST000007', 'ST007028',
     'ST028100', 'ST100289', 'SM000007', 'SM007028', 'SM028100',
     'SM100289']"

Those ERA5-named layers were never in a native HRRR met snapshot at all;
they are what the soil router asks for once it finds no native pair and
no mapped or GFS markers.  The solved soil was in ``surface/`` the whole
time.

These tests are writer-to-reader: the cache is written by the shipped
writer and read back by the shipped reader, so the "which fields
survive" contract cannot drift into a fixture that agrees with itself.
"""

from __future__ import annotations

import json
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.hrrr_physics import (
    _CANONICAL_SURFACE_FIELDS, resolve_prepared_noah_surface)
from gpuwm.ingest.prepared_cache import (
    PreparedCacheReader, write_prepared_cache)
from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil
from gpuwm.native_wrf_contract import canonical_noah_surface

from test_prepared_cache import _fixture


SHAPE = (2, 2)
#: What the native route hands the soil derivation, physically plausible
#: so the certified derivation's own range checks pass.
SOIL_TYPE = np.full(SHAPE, 6.0)
DEEP_SOIL = np.full(SHAPE, 281.0)


def _native_met():
    """A freshly decoded native met snapshot: the soil pair, no surface."""

    _initial, met, _boundaries = _fixture()
    fields = dict(met.fields)
    nodes = fields["SOILT"].shape[0]
    fields["SOILT"] = np.stack(
        [np.full(SHAPE, 290.0 - 2.0 * k, dtype=np.float32)
         for k in range(nodes)])
    fields["SOILW"] = np.stack(
        [np.full(SHAPE, 0.10 + 0.02 * k, dtype=np.float32)
         for k in range(nodes)])
    fields.setdefault("XICE", np.zeros(SHAPE, dtype=np.float32))
    fields.setdefault("SNOW", np.zeros(SHAPE, dtype=np.float32))
    return SimpleNamespace(fields=MappingProxyType(fields))


def _cfg(sf_surface_physics: int = 2):
    # soil_layer_count(cfg) reads the requested count and checks it against
    # the scheme's defined geometries; Noah defines exactly four.  The
    # parametric-soil landing (54a18cf4e) made the count an explicit config
    # read and this fake never carried it.
    return SimpleNamespace(sf_surface_physics=sf_surface_physics,
                           num_soil_layers=4)


def _static():
    return {"SCT_DOM": SOIL_TYPE, "SOILTEMP": DEEP_SOIL}


def _solved_surface():
    """The surface the preparation solves and the cache persists."""

    return canonical_noah_surface(preprocess_land_surface_soil(
        _native_met().fields, sf_surface_physics=2,
        soil_type=SOIL_TYPE, deep_soil_temperature=DEEP_SOIL))


# ---------------------------------------------------------------------
# the two roads into physics setup
# ---------------------------------------------------------------------

def test_a_fresh_native_snapshot_derives_the_surface_from_its_soil_pair():
    """The decode road: no surface yet, the native pair is the source."""

    resolved = resolve_prepared_noah_surface(
        _native_met(), _cfg(), _static())
    assert set(resolved.fields) == _CANONICAL_SURFACE_FIELDS
    expected = _solved_surface()
    for name in sorted(_CANONICAL_SURFACE_FIELDS):
        assert np.array_equal(
            np.asarray(resolved.fields[name]), np.asarray(expected[name]))


def test_a_restored_state_uses_the_surface_it_was_handed_not_the_met():
    """The restore road: the answer is passed in, and nothing re-derives.

    The met here is a restored one -- no ``SOILT``/``SOILW``, because
    the writer dropped them.  Re-deriving would raise the shakeout's
    ``missing soil input field(s)`` KeyError; consuming the surface
    cannot, and this asserts the object identity so a future "helpful"
    re-derivation cannot pass by producing an equal answer.
    """

    surface = SimpleNamespace(fields=MappingProxyType(_solved_surface()))
    restored_met = SimpleNamespace(fields=MappingProxyType({
        name: value for name, value in _native_met().fields.items()
        if name not in {"SOILT", "SOILW"}}))
    assert {"SOILT", "SOILW"}.isdisjoint(restored_met.fields)

    resolved = resolve_prepared_noah_surface(
        restored_met, _cfg(), _static(), surface=surface)
    assert resolved is surface


def test_re_deriving_from_a_restored_met_dies_in_the_soil_router():
    """The measured failure itself, pinned as the reason this seam exists.

    Nothing here is a defect in the router: ``ST000007`` and its
    siblings are the names of its ERA5 arm, and a restored native met is
    not an ERA5 snapshot -- it is a snapshot whose soil was already
    solved and stored elsewhere.  This asserts the crash so that putting
    a re-derivation back on the restore road fails a test rather than a
    forecast.
    """

    restored_fields = {
        name: value for name, value in _native_met().fields.items()
        if name not in {"SOILT", "SOILW"}}
    with pytest.raises(KeyError, match="ST000007"):
        preprocess_land_surface_soil(
            restored_fields, sf_surface_physics=2,
            soil_type=SOIL_TYPE, deep_soil_temperature=DEEP_SOIL)


def test_a_state_with_neither_surface_nor_soil_is_refused_by_name():
    """An incomplete cache is an owned refusal, not a KeyError four frames down.

    The sentence has to name the breakage -- no soil state for the land
    model -- and the remedy, because the reader who meets it has a tree
    on disk and needs to know whether to re-prepare it.
    """

    stripped = SimpleNamespace(fields=MappingProxyType({
        name: value for name, value in _native_met().fields.items()
        if name not in {"SOILT", "SOILW"}}))
    with pytest.raises(ValueError) as refusal:
        resolve_prepared_noah_surface(stripped, _cfg(), _static())
    sentence = str(refusal.value)
    assert "SOILT" in sentence and "SOILW" in sentence
    assert "no solved Noah surface" in sentence
    assert "surface/TSLB" in sentence
    assert "prepared again" in sentence
    # The old failure named ERA5 soil layers that a native snapshot never
    # carried; that spelling must not come back as the reader's answer.
    assert "ST000007" not in sentence


# ---------------------------------------------------------------------
# the real write -> restore cycle the refusal is measured against
# ---------------------------------------------------------------------

def test_the_real_writer_stores_the_surface_and_drops_the_native_pair(
        tmp_path):
    """Written by the shipped writer, read back by the shipped reader.

    This is the contract the restore arm broke on, asserted end to end:
    the persisted met has no native soil pair, the persisted surface has
    the full canonical inventory, and the pair that comes back out of
    the reader resolves through physics setup's own entry point without
    touching the soil router.
    """

    initial, _met, boundaries = _fixture()
    met = _native_met()
    surface = _solved_surface()
    identity = {"source": "hrrr-native"}

    receipt = write_prepared_cache(
        tmp_path / "cache", identity=identity, initial_result=initial,
        met=met, boundaries=boundaries, surface=surface)
    assert receipt["status"] == "BUILT"

    reader = PreparedCacheReader(tmp_path / "cache",
                                 expected_identity=identity)
    assert reader.verify_all()["status"] == "PASS"
    metadata = json.loads(
        (tmp_path / "cache" / "header.json").read_text(
            encoding="utf-8"))["metadata"]
    assert set(metadata["met_fields"]).isdisjoint({"SOILT", "SOILW"})
    assert set(metadata["surface_fields"]) == _CANONICAL_SURFACE_FIELDS

    # Exactly what restore_prepared_cache reconstructs from this header,
    # by the same array names, without the CUDA state rebuild.
    restored_met = SimpleNamespace(fields=MappingProxyType({
        name: reader.read_array(f"met/{name}")
        for name in metadata["met_fields"]}))
    restored_surface = SimpleNamespace(fields=MappingProxyType({
        name: reader.read_array(f"surface/{name}")
        for name in metadata["surface_fields"]}))

    resolved = resolve_prepared_noah_surface(
        restored_met, _cfg(), _static(), surface=restored_surface)
    assert set(resolved.fields) == _CANONICAL_SURFACE_FIELDS
    for name in sorted(_CANONICAL_SURFACE_FIELDS):
        assert np.array_equal(
            np.asarray(resolved.fields[name]), np.asarray(surface[name]))


def test_the_benchmark_forwards_the_restored_surface_into_physics_setup():
    """The call site, pinned: the restore arm passes what it restored.

    ``da_cycle_prepared`` and ``hrrr_hierarchy_direct`` both consumed
    ``restored.surface`` already; the single-domain benchmark was the one
    consumer that dropped it on the floor, which is why the defect showed
    up only on ``--prepared-cache``.
    """

    import inspect
    import sys
    from pathlib import Path

    tools = Path(__file__).resolve().parents[1] / "tools"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    import hrrr_single_domain_benchmark as benchmark

    body = inspect.getsource(benchmark.run)
    assert "root_surface = restored.surface" in body
    assert "surface=root_surface" in body

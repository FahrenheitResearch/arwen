"""Prep and the model cannot disagree about an adapted coordinate.

The prepared artifacts carry the hybrid coefficient arrays themselves --
``ingest/prepared_cache`` writes ``coord_scalars``/``coord_arrays`` and
restores ``VerticalCoord(**...)`` from them, the way WRF's ``wrfinput``
carries ``C3H``/``C4H`` -- so the model integrates the coordinate
preparation built.  What could still have differed is everything the
RUNNER derives from the configuration beside the bundle: the prepared
identity comparison, a tile buffer's rebuilt coordinate, a nest spawned
mid-run.  These tests walk that seam end to end on an adapted run.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.grid import compute_hybrid_coeffs, make_vertical_coord
from gpuwm.experiment import load_experiment
from gpuwm.ingest.prepared_cache import (_coord_metadata,
                                         prepared_domain_config_identity)
from gpuwm.runtime import vertical_coord_for
from gpuwm.vertical_adaptation import (TerrainField, adapt_experiment_vertical,
                                       adopt_prepared_vertical,
                                       prepared_domain_coordinate_refusal,
                                       vertical_coordinate_receipt)

import gpuwm.core.constants as c

#: The committed two-domain real-data configuration, at the shipped
#: etac = 0.2 / p_top = 10000 Pa pair.
CONFIG = Path(__file__).parents[1] / "configs" / "gfs_wrf_hierarchy_proof.toml"

#: Ground the shipped etac = 0.2 / p_top = 10000 Pa pair cannot order.
UNORDERABLE_M = 6800.0

_COEFFICIENTS = ("c1f", "c2f", "c3f", "c4f", "c1h", "c2h", "c3h", "c4h")


@pytest.fixture
def adapted():
    exp = load_experiment(CONFIG)
    terrain = np.full((6, 7), 10.0)
    terrain[2, 3] = UNORDERABLE_M
    fields = [TerrainField(f"d{int(dc.grid_id):02d} static terrain", terrain,
                           float(dc.run.base_temp)) for dc in exp.domains]
    adapted_exp, adaptation = adapt_experiment_vertical(exp, fields)
    assert adaptation.adapted, "the fixture must actually adapt"
    return exp, adapted_exp, adaptation


def _prep_coordinate(exp):
    """What preparation builds: the source doors' own call."""
    return make_vertical_coord(
        exp.root.run.nz, hybrid_opt=exp.root.run.hybrid_opt,
        etac=exp.root.run.etac, eta_levels=exp.vertical.eta_levels)


def test_the_adaptation_reaches_every_derived_copy(adapted):
    _, adapted_exp, adaptation = adapted
    assert adapted_exp.vertical.etac == adaptation.etac
    assert all(dc.run.etac == adaptation.etac for dc in adapted_exp.domains)
    # The eta ladder and the model top are untouched.
    assert adapted_exp.vertical.eta_levels == adapted_exp.vertical.eta_levels
    assert adapted_exp.vertical.p_top == adapted_exp.vertical.p_top


def test_prep_and_model_coefficients_are_identical_for_an_adapted_run(adapted):
    """THE SEAM, both sides of it, element for element."""
    _, adapted_exp, _ = adapted
    prep = _prep_coordinate(adapted_exp)
    # What the model would build for itself from the same experiment.
    model = vertical_coord_for(adapted_exp.vertical, adapted_exp.root.run.nz)
    for name in _COEFFICIENTS + ("znw", "znu", "dnw", "rdnw", "dn", "rdn",
                                 "fnp", "fnm"):
        assert np.array_equal(getattr(prep, name), getattr(model, name)), name
    assert prep.etac == model.etac == adapted_exp.vertical.etac
    assert prep.hybrid_opt == model.hybrid_opt


def test_the_cache_round_trip_returns_the_same_coefficients(adapted):
    """The prepared cache stores the coordinate, not a recipe for one."""
    from gpuwm.core.grid import VerticalCoord, finalize_vertical_coord

    _, adapted_exp, _ = adapted
    prep = _prep_coordinate(adapted_exp)
    finalize_vertical_coord(prep, adapted_exp.vertical.p_top)
    scalars = _coord_metadata(prep)
    arrays = {name: value for name, value in vars(prep).items()
              if isinstance(value, np.ndarray)}
    restored = VerticalCoord(**scalars, **arrays)
    for name in _COEFFICIENTS:
        assert np.array_equal(getattr(prep, name), getattr(restored, name))
    assert scalars["etac"] == adapted_exp.vertical.etac
    assert scalars["hybrid_opt"] == adapted_exp.vertical.hybrid_opt


def test_the_runner_adopts_the_prepared_coordinate(adapted):
    configured_exp, adapted_exp, adaptation = adapted
    receipt = {"vertical_coordinate": vertical_coordinate_receipt(
        adapted_exp, adaptation)}
    said = []
    adopted_exp, record = adopt_prepared_vertical(
        configured_exp, receipt, announce=said.append)
    assert record["etac"] == adaptation.etac
    assert adopted_exp.vertical == adapted_exp.vertical
    assert all(dc.run.etac == adaptation.etac for dc in adopted_exp.domains)
    assert len(said) == 1
    assert f"etac {adaptation.etac:g}" in said[0]
    assert "d01 static terrain" in said[0]
    # And what it adopted rebuilds prep's coefficients exactly.
    for name in _COEFFICIENTS:
        assert np.array_equal(
            getattr(_prep_coordinate(adapted_exp), name),
            getattr(_prep_coordinate(adopted_exp), name))


def test_the_prepared_identity_matches_only_after_adoption(adapted):
    """Without adoption the cache identity itself would refuse the run."""
    configured_exp, adapted_exp, adaptation = adapted
    prepared = prepared_domain_config_identity(adapted_exp.root)
    assert prepared["run"]["etac"] == adaptation.etac
    assert (prepared_domain_config_identity(configured_exp.root)["run"]["etac"]
            != prepared["run"]["etac"])
    receipt = {"vertical_coordinate": vertical_coordinate_receipt(
        adapted_exp, adaptation)}
    adopted_exp, _ = adopt_prepared_vertical(configured_exp, receipt)
    assert prepared_domain_config_identity(adopted_exp.root) == prepared


def test_an_unadapted_bundle_leaves_the_experiment_exactly_alone():
    exp = load_experiment(CONFIG)
    fields = [TerrainField("d01 static terrain", np.full((4, 4), 500.0),
                           float(exp.root.run.base_temp))]
    same, adaptation = adapt_experiment_vertical(exp, fields)
    assert same is exp and not adaptation.adapted
    receipt = {"vertical_coordinate": vertical_coordinate_receipt(
        exp, adaptation)}
    adopted, record = adopt_prepared_vertical(exp, receipt)
    assert adopted is exp
    assert record["etac"] == exp.vertical.etac


def test_a_bundle_with_no_coordinate_record_is_left_alone():
    """A bundle written before this existed ran the configured value."""
    exp = load_experiment(CONFIG)
    adopted, record = adopt_prepared_vertical(exp, {"schema": "older"})
    assert adopted is exp and record is None


def test_a_prepared_domain_is_held_to_the_adopted_coordinate(adapted):
    _, adapted_exp, adaptation = adapted
    vertical = adapted_exp.vertical
    ordered = np.full((3, 3), 90000.0) - vertical.p_top
    assert prepared_domain_coordinate_refusal(
        label="d01", vertical=vertical,
        coord_scalars={"etac": adaptation.etac,
                       "hybrid_opt": vertical.hybrid_opt},
        base_arrays={"mub": ordered}) is None
    # A sibling stored on a DIFFERENT coordinate is the disagreement this
    # whole seam exists to make impossible.
    message = prepared_domain_coordinate_refusal(
        label="d02", vertical=vertical,
        coord_scalars={"etac": 0.2, "hybrid_opt": vertical.hybrid_opt},
        base_arrays={"mub": ordered})
    assert message is not None
    assert "cannot sit on a different vertical coordinate" in message


def test_a_prepared_cache_with_no_coordinate_scalars_is_refused(adapted):
    _, adapted_exp, _ = adapted
    message = prepared_domain_coordinate_refusal(
        label="d01", vertical=adapted_exp.vertical, coord_scalars={},
        base_arrays={})
    assert message is not None
    assert "records no hybrid coordinate" in message


def test_a_mismatched_model_top_is_refused_rather_than_adopted(adapted):
    _, adapted_exp, adaptation = adapted
    record = vertical_coordinate_receipt(adapted_exp, adaptation)
    record["p_top_pa"] = float(adapted_exp.vertical.p_top) + 500.0
    with pytest.raises(ValueError, match="different vertical grid"):
        adopt_prepared_vertical(adapted_exp, {"vertical_coordinate": record})


def test_an_adopted_etac_above_the_configured_one_is_refused(adapted):
    configured_exp, adapted_exp, adaptation = adapted
    record = vertical_coordinate_receipt(adapted_exp, adaptation)
    record["etac"] = float(configured_exp.vertical.etac) + 0.05
    with pytest.raises(ValueError, match="may only be at or below"):
        adopt_prepared_vertical(
            configured_exp, {"vertical_coordinate": record})


def test_the_adapted_coefficients_are_wrfs_own_for_that_etac(adapted):
    """No new numerics: only the etac fed to WRF's transcribed formulas."""
    _, adapted_exp, adaptation = adapted
    prep = _prep_coordinate(adapted_exp)
    reference = compute_hybrid_coeffs(
        np.asarray(adapted_exp.vertical.eta_levels, dtype=np.float64),
        2, adaptation.etac, c.P0, float(adapted_exp.vertical.p_top))
    from gpuwm.core.grid import finalize_vertical_coord

    finalize_vertical_coord(prep, adapted_exp.vertical.p_top)
    for name in _COEFFICIENTS:
        assert np.array_equal(getattr(prep, name), reference[name]), name


# --------------------------------------------------------------------------
# The case-data route's ORDER.  ``gpuwm.core.model.build_experiment`` takes
# its startup tree from the experiment by value and stores each of those
# DomainConfigs on a node, where core/streaming.domain_vertical_coord
# rebuilds a streamed tile buffer's coordinate from ``cfg.etac``.  Derive
# after that copy and the run holds two coordinates.
# --------------------------------------------------------------------------


class _StopAfterTheTree(Exception):
    """Ends the builder at the first line these tests care about."""


def _case_route_stubs(monkeypatch, adapted_exp):
    """Everything ``build_experiment`` reads before the startup tree.

    None of it is vertical: a catalog handle, the forcing snapshots and
    their cadence.  Stubbing them lets the ordering be tested without a
    decode, a device or a case on disk.
    """

    import gpuwm.core.model as core_model
    import gpuwm.ingest.preflight as preflight
    import gpuwm.runtime as runtime_module

    monkeypatch.setattr(preflight, "build_input_catalog",
                        lambda data: object())
    monkeypatch.setattr(core_model, "_adapt_experiment_vertical_for_case",
                        lambda e, data, catalog: adapted_exp)
    monkeypatch.setattr(runtime_module, "forcing_snapshots",
                        lambda *a, **k: {})
    monkeypatch.setattr(runtime_module, "forcing_schedule", lambda *a, **k: ())
    monkeypatch.setattr(core_model, "_forcing_cadence_seconds",
                        lambda catalog: 10800.0)


def test_the_case_data_route_derives_before_it_takes_the_startup_tree(
        monkeypatch, adapted):
    """Every nest's own RunConfig carries the derived value, not the configured one."""
    import gpuwm.core.model as core_model
    import gpuwm.experiment as experiment_module

    configured_exp, adapted_exp, adaptation = adapted
    assert len(configured_exp.domains) > 1, "the fixture must carry a nest"
    assert adaptation.etac < configured_exp.vertical.etac
    _case_route_stubs(monkeypatch, adapted_exp)

    seen = {}

    def record(handed):
        seen["vertical"] = float(handed.vertical.etac)
        seen["domains"] = tuple(float(dc.run.etac) for dc in handed.domains)
        raise _StopAfterTheTree

    monkeypatch.setattr(experiment_module, "pre_spawn_experiment", record)
    with pytest.raises(_StopAfterTheTree):
        core_model.build_experiment(configured_exp, object())

    assert seen["vertical"] == adaptation.etac
    assert set(seen["domains"]) == {adaptation.etac}, (
        "a nest's RunConfig kept the configured coordinate: the derivation "
        "ran after the startup tree was taken")


def test_a_tree_taken_before_the_derivation_is_refused_by_name(
        monkeypatch, adapted):
    """The order cannot drift back silently: the builder asks."""
    import gpuwm.core.model as core_model
    import gpuwm.experiment as experiment_module

    configured_exp, adapted_exp, _ = adapted
    _case_route_stubs(monkeypatch, adapted_exp)
    # The defect exactly: the tree is the one that was taken BEFORE the
    # derivation, while the experiment carries the derived coordinate.
    monkeypatch.setattr(experiment_module, "pre_spawn_experiment",
                        lambda handed: configured_exp)
    with pytest.raises(RuntimeError, match="left a domain off this run"):
        core_model.build_experiment(configured_exp, object())


def test_the_refusal_passes_a_tree_already_on_the_coordinate(adapted):
    from gpuwm.vertical_adaptation import (
        refuse_tree_off_the_experiment_coordinate)

    configured_exp, adapted_exp, _ = adapted
    refuse_tree_off_the_experiment_coordinate(
        adapted_exp, adapted_exp.domains, route="experiment-tree")
    refuse_tree_off_the_experiment_coordinate(
        configured_exp, configured_exp.domains, route="experiment-tree")


# --------------------------------------------------------------------------
# What the case-data route HANDS the survey.  ``gpuwm.core.model`` is a
# clock module under an AST audit that bans reflection outright
# (tests/test_clock.py::test_no_float_elapsed_accumulation_audit), so the
# derivation reads the case data's declared ``static_highres`` field as a
# field and asks ``vertical_adaptation.static_catalog_for_survey`` for the
# catalog.  These hold each caller shape to the objects the reflective
# reads (``getattr(case_data, "static_highres", None)`` and
# ``getattr(catalog, "static_catalog", catalog)``) selected before, by
# identity, so the survey's inputs did not move when the reads did.
# --------------------------------------------------------------------------


def _survey_route_stubs(monkeypatch, adapted_exp, terrain, seen):
    """Everything the derivation touches besides the survey call itself.

    The root's statics, the projection grids and the GEOG selection are
    not what these tests are about; the survey is replaced by a recorder
    that returns the fixture's adapted experiment.
    """
    import gpuwm.runtime as runtime_module
    import gpuwm.vertical_adaptation as adaptation_module
    from gpuwm.static import build as static_build
    from gpuwm.static import lambert

    monkeypatch.setattr(lambert, "grids_from_projection_config",
                        lambda exp: [object() for _ in exp.domains])
    monkeypatch.setattr(static_build.GeogSelection, "from_case_data",
                        staticmethod(lambda data, domain_id=1: object()))

    def case_static_fields(grid, geog_root, **kwargs):
        seen["static_fields_kwargs"] = kwargs
        return {"HGT_M": terrain}

    def survey(exp, grids, **kwargs):
        seen["exp"] = exp
        seen["survey_kwargs"] = kwargs
        return adapted_exp, None

    monkeypatch.setattr(runtime_module, "case_static_fields",
                        case_static_fields)
    monkeypatch.setattr(adaptation_module, "adapt_experiment_for_statics",
                        survey)


def _validated_case_data(tmp_path):
    """The production shape: the validated case data every route hands in."""
    from test_case_data import make_case_toml

    from gpuwm.case_data import CaseDataConfig, load_experiment_case

    _, case_data = load_experiment_case(make_case_toml(tmp_path))
    assert isinstance(case_data, CaseDataConfig)
    return case_data


def test_the_two_type_facts_the_direct_reads_rest_on():
    from gpuwm.case_data import CaseDataConfig
    from gpuwm.ingest.preflight import InputCatalog

    # The case data declares the field, default None (the identity path),
    # so reading it as a field is what reading it with a None default was.
    field = CaseDataConfig.__dataclass_fields__["static_highres"]
    assert field.default is None
    # The source catalog carries no ``static_catalog``: it IS the static
    # catalog, so selecting it yields the catalog itself.
    assert "static_catalog" not in InputCatalog.__dataclass_fields__
    assert not hasattr(InputCatalog, "static_catalog")


@pytest.mark.parametrize("highres", [None, object()],
                         ids=["identity-path", "overlay-declared"])
def test_a_nested_run_hands_the_survey_its_catalog_and_its_overlay(
        monkeypatch, adapted, tmp_path, highres):
    from dataclasses import replace

    import gpuwm.core.model as core_model

    configured_exp, adapted_exp, _ = adapted
    assert len(configured_exp.domains) > 1, "the fixture must carry a nest"
    case_data = replace(_validated_case_data(tmp_path), static_highres=highres)
    # The source catalog's shape: GEOG roles among its files, no wrapper.
    catalog = SimpleNamespace(files=())
    terrain = np.full((6, 7), 10.0)
    seen = {}
    _survey_route_stubs(monkeypatch, adapted_exp, terrain, seen)

    handed = core_model._adapt_experiment_vertical_for_case(
        configured_exp, case_data, catalog)

    assert handed is adapted_exp
    assert seen["exp"] is configured_exp
    assert seen["survey_kwargs"]["root_terrain"] is terrain
    # The catalog: what selecting ``static_catalog`` with the catalog as
    # the default selected, which for the source catalog is itself.
    assert seen["survey_kwargs"]["static_catalog"] is getattr(
        catalog, "static_catalog", catalog)
    assert seen["survey_kwargs"]["static_catalog"] is catalog
    # The overlay: what reading ``static_highres`` with a None default
    # read, at the survey and at the root's statics alike.
    assert seen["survey_kwargs"]["static_highres"] is getattr(
        case_data, "static_highres", None)
    assert seen["survey_kwargs"]["static_highres"] is highres
    assert seen["static_fields_kwargs"]["static_highres"] is highres


def test_a_single_domain_run_hands_the_survey_no_catalog(
        monkeypatch, adapted, tmp_path):
    from dataclasses import replace

    import gpuwm.core.model as core_model

    configured_exp, adapted_exp, _ = adapted
    single = replace(configured_exp, domains=configured_exp.domains[:1])
    case_data = _validated_case_data(tmp_path)
    catalog = SimpleNamespace(files=())
    seen = {}
    _survey_route_stubs(monkeypatch, adapted_exp, np.full((6, 7), 10.0), seen)

    core_model._adapt_experiment_vertical_for_case(single, case_data, catalog)

    assert seen["survey_kwargs"]["static_catalog"] is None
    assert seen["survey_kwargs"]["static_highres"] is None


def test_the_survey_selects_the_catalog_the_way_a_child_is_initialised():
    """One selection rule, in the module that owns the derivation."""
    from gpuwm.ingest.nest_init import _static_catalog
    from gpuwm.vertical_adaptation import static_catalog_for_survey

    inner = object()
    wrapper = SimpleNamespace(static_catalog=inner)
    source = SimpleNamespace(files=())
    assert static_catalog_for_survey(wrapper) is inner
    assert static_catalog_for_survey(source) is source
    for catalog in (wrapper, source):
        assert static_catalog_for_survey(catalog) is _static_catalog(catalog)

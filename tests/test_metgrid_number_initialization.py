"""Flagged metgrid number analyses reach the exact selected scalar package."""
from dataclasses import replace

import numpy as np
import pytest

from gpuwm.ingest.analyzed_numbers import METGRID_NUMBER_FIELDS, metgrid_number_targets
from gpuwm.ingest.real import initialize_real


@pytest.mark.parametrize("mp,expected", [
    (0, {}), (1, {}), (6, {}), (8, {"QNI":"ni", "QNR":"nr"}),
    # EXPLICIT, not dict(zip(METGRID_NUMBER_FIELDS, ...)): the tuple grew
    # by QNWFA/QNIFA and a zip would have silently re-aligned every pair
    # rather than failing.
    (9, {"QNI":"ni", "QNC":"nc", "QNR":"nr", "QNS":"ns", "QNG":"ng", "QNH":"nh"}),
    (10, {"QNI":"ni", "QNR":"nr", "QNS":"ns", "QNG":"ng"}),
    (16, {"QNC":"nc", "QNR":"nr"}),
    (18, {"QNI":"qni", "QNR":"qnr", "QNS":"qns", "QNG":"qng", "QNH":"qnh"}),
    (28, {"QNI":"ni", "QNC":"nc", "QNR":"nr", "QNWFA":"nwfa", "QNIFA":"nifa"}),
    (50, {"QNI":"ni", "QNR":"nr"}),
])
def test_registry_membership_distinguishes_qnc_and_qndrop(mp, expected):
    cfg = _case(mp)[1]
    assert metgrid_number_targets(cfg) == expected


def _case(mp):
    from test_metem_differential import _synthetic
    snapshot, cfg, coord, terrain, orography = _synthetic(2500., 2500.)
    cfg = replace(cfg, mp_physics=mp, mp28_aerosol_source="synthetic")
    # Different vertical and horizontal values expose accidental broadcast,
    # field aliasing and use of an unrelated donor column.
    shape = snapshot.fields["TT"].shape
    profile = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1) * 64
    fields = dict(snapshot.fields)
    for index, name in enumerate(METGRID_NUMBER_FIELDS):
        fields[name] = profile * np.float32(2 ** index)
        fields[name+"_SFC"] = (np.arange(cfg.ny*cfg.nx, dtype=np.float32).reshape(cfg.ny,cfg.nx)+3) * np.float32(2 ** index)
    return replace(snapshot, fields=fields), cfg, coord, terrain, orography


def numbers_in_effect(cfg, receipt):
    """The number targets a run ACTUALLY filled, and the rows it binned.

    ``metgrid_number_targets(cfg)`` answers one question only: which rows
    of the table the selected scalar package transports.  It cannot know
    that the mp=28 aerosol pair was dropped before it reached the state,
    and on this fixture it always is: :func:`_case` names
    ``mp28_aerosol_source='synthetic'``, so
    ``gpuwm/ingest/real.py:3579`` pops QNWFA/QNIFA out of the moments and
    ``nwfa``/``nifa`` stay at exact zero.  Every consumer that iterates
    the targets and then demands a nonzero field needs that second half of
    the answer, and the CUDA twin of this module needs the SAME one, so
    it imports this function rather than repeating the subtraction.
    """

    by_request = frozenset(receipt.get(
        "discarded_by_requested_aerosol_source", {}).get("fields", ()))
    return ({name: target for name, target
             in metgrid_number_targets(cfg).items()
             if name not in by_request}, by_request)


def _require_cpu_bridge():
    from gpuwm.ingest.cpu_backend import CpuPreprocessBackend
    try:
        CpuPreprocessBackend()
    except (FileNotFoundError, OSError) as error:
        pytest.skip(f"native CPU bridge is not available: {error}")


def test_requested_aerosol_source_empties_the_rows_it_discards():
    """The mp=28 arm of this fixture fills no nwfa/nifa, and the receipt says so.

    The table grew QNWFA/QNIFA rows, so ``metgrid_number_targets`` now
    answers the shared mp=28 fixture with two targets whose state stays at
    exact zero, because that fixture names an explicit aerosol source.
    This pins the fact both twins stand on, so that a consumer written
    against the old three-row answer cannot iterate the new one and assert
    a nonzero field, a finite reference match and a matching zero mask
    against an all-zero array.
    """
    _require_cpu_bridge()
    snapshot, cfg, coord, terrain, orography = _case(28)
    result = initialize_real(snapshot, cfg, coord, terrain,
        source_orography=orography, p_top=5000., preprocess_backend="cpu",
        state_backend="preprocess", analyzed_species=(),
        analyzed_number_fields=METGRID_NUMBER_FIELDS)
    announced = metgrid_number_targets(cfg)
    targets, by_request = numbers_in_effect(
        cfg, result.hydrometeor_initialization["number_moments"])
    assert by_request == {"QNWFA", "QNIFA"}
    assert set(announced) - set(targets) == set(by_request)
    for source in by_request:
        assert np.count_nonzero(getattr(result.state, announced[source])) == 0
    assert set(targets) == {"QNI", "QNC", "QNR"}
    for target in targets.values():
        assert np.count_nonzero(getattr(result.state, target)) > 0


def test_analyzed_aerosol_pair_is_a_table_row_resolved_by_the_active_package():
    """QNWFA/QNIFA reach nwfa/nifa through the SAME membership as QNI/QNC.

    They used to be a refusal in ``check_analyzed_scalar_capability``
    rather than rows of this table, which meant the one generic route that
    already carries every other metgrid number field could not carry them.
    A package that does not transport the pair still discards it, exactly
    like an inactive P_QN*, so the row costs that package nothing.
    """
    assert "QNWFA" in METGRID_NUMBER_FIELDS and "QNIFA" in METGRID_NUMBER_FIELDS
    targets28 = metgrid_number_targets(_case(28)[1])
    assert targets28["QNWFA"] == "nwfa" and targets28["QNIFA"] == "nifa"
    targets8 = metgrid_number_targets(_case(8)[1])
    assert "QNWFA" not in targets8 and "QNIFA" not in targets8


def test_analyzed_aerosol_beats_the_climatology_on_auto_and_says_so(monkeypatch):
    """The analysis beats the climatology on ``auto``, and says so.

    A monthly climatological mean stands in for exactly the field the
    analysis is carrying, so an analyzed QNWFA/QNIFA pair wins when nobody
    asked for a particular source.  ``nwfa2d`` comes from WRF's own
    :4530-4547 surface-emission formula applied to the lowest analyzed
    level, through the same ``wif_surface_emission`` the climatology uses;
    ``nifa2d`` stays zero, as it does in WRF.

    This fixture's domain has no external lateral boundaries, so it does
    not cross the mp=28 lateral-forcing floor and says nothing about a
    met_em root domain, which always does.  That is
    ``test_analyzed_aerosol_runs_a_specified_met_em_root_with_no_wif_dataset``.
    """
    _require_cpu_bridge()
    _no_wif_dataset(monkeypatch)
    snapshot, cfg, coord, terrain, orography = _case(28)
    cfg = replace(cfg, mp28_aerosol_source="auto")
    assert cfg.specified is False
    result = initialize_real(snapshot, cfg, coord, terrain,
        source_orography=orography, p_top=5000., preprocess_backend="cpu",
        state_backend="preprocess", analyzed_species=(),
        analyzed_number_fields=METGRID_NUMBER_FIELDS)
    for name in ("nwfa", "nifa"):
        value = getattr(result.state, name)
        assert np.isfinite(value).all() and np.count_nonzero(value) > 0
    assert np.count_nonzero(result.state.nwfa2d) > 0
    assert np.all(np.isfinite(result.state.nwfa2d))
    assert np.count_nonzero(result.state.nifa2d) == 0
    retained = result.hydrometeor_initialization["number_moments"]["retained_correspondence"]
    assert retained["QNWFA"] == "nwfa" and retained["QNIFA"] == "nifa"
    aerosol = result.aerosol_initialization
    assert aerosol["aerosol_source"] == "metgrid-analyzed"
    assert aerosol["awaiting_profile_fill"] is False
    assert aerosol["analyzed_aerosol_fields"] == ["QNWFA", "QNIFA"]
    assert "synthetic_fallback_in_use" not in aerosol
    assert "dataset" not in aerosol


def test_named_aerosol_source_keeps_its_request_and_warns_about_the_discard(capsys):
    """Rule: warn, never refuse. The operator named a source; it is used."""
    _require_cpu_bridge()
    snapshot, cfg, coord, terrain, orography = _case(28)
    result = initialize_real(snapshot, cfg, coord, terrain,
        source_orography=orography, p_top=5000., preprocess_backend="cpu",
        state_backend="preprocess", analyzed_species=(),
        analyzed_number_fields=METGRID_NUMBER_FIELDS)
    assert result.aerosol_initialization["aerosol_source"] == (
        "thompson_init-synthetic-profile")
    assert np.count_nonzero(result.state.nwfa) == 0
    assert np.count_nonzero(result.state.nwfa2d) == 0
    receipt = result.hydrometeor_initialization["number_moments"]
    assert "QNWFA" not in receipt["retained_correspondence"]
    assert receipt["discarded_by_requested_aerosol_source"]["fields"] == [
        "QNWFA", "QNIFA"]
    assert "QNWFA/QNIFA" in capsys.readouterr().err


@pytest.mark.parametrize("mp", [8, 9, 10, 16, 18, 28, 50])
def test_native_cpu_initialization_retains_numbers_without_changing_thermodynamics(mp):
    snapshot, cfg, coord, terrain, orography = _case(mp)
    kw = dict(source_orography=orography, p_top=5000., preprocess_backend="cpu",
              state_backend="preprocess", analyzed_species=())
    absent = initialize_real(snapshot, cfg, coord, terrain, **kw)
    supplied = initialize_real(snapshot, cfg, coord, terrain,
                               analyzed_number_fields=METGRID_NUMBER_FIELDS, **kw)
    receipt = supplied.hydrometeor_initialization["number_moments"]
    # "the package does not carry this field" and "the operator named a
    # different aerosol source" are two different facts about a discarded
    # field, so they live under two keys.  This fixture names 'synthetic',
    # which discards the mp=28 analyzed QNWFA/QNIFA pair by request.
    targets, by_request = numbers_in_effect(cfg, receipt)
    inactive = set(receipt["discarded_inactive_package_fields"])
    assert receipt["retained_correspondence"] == targets
    assert not inactive & by_request
    assert inactive | by_request == set(METGRID_NUMBER_FIELDS)-set(targets)
    for name in ("mup", "thp", "php", "qv", "u", "v", "qc", "qr", "pb", "alb"):
        np.testing.assert_array_equal(getattr(supplied.state, name), getattr(absent.state, name))
    for source, target in targets.items():
        actual = getattr(supplied.state, target)
        assert actual.dtype == np.float32 and np.isfinite(actual).all()
        assert np.count_nonzero(actual) > 0
        assert np.all(getattr(absent.state, target) == 0)
        assert np.ptp(actual) > 0
    # Every field uses the same existing Q operator. Exact powers of two
    # distinguish number-field routing while keeping a bitwise scale oracle.
    first_source = next(iter(targets))
    base = getattr(supplied.state, targets[first_source])
    first_index = METGRID_NUMBER_FIELDS.index(first_source)
    for source, target in targets.items():
        factor = np.float32(2 ** (METGRID_NUMBER_FIELDS.index(source)-first_index))
        np.testing.assert_array_equal(getattr(supplied.state, target), base * factor)
    if mp == 18:
        np.testing.assert_array_equal(supplied.state.qndrop, absent.state.qndrop)


@pytest.mark.parametrize("bad", [-1., np.inf, np.nan])
def test_bad_number_analysis_refuses_even_when_package_does_not_consume_it(bad):
    snapshot, cfg, coord, terrain, orography = _case(6)
    snapshot.fields["QNI"][1, 1, 1] = bad
    with pytest.raises(ValueError, match="analyzed number field QNI"):
        initialize_real(snapshot, cfg, coord, terrain, source_orography=orography,
            preprocess_backend="cpu", state_backend="preprocess", analyzed_species=(),
            analyzed_number_fields=("QNI",))


@pytest.mark.parametrize("declared", [("QNI","QNI"), ("unknown",), "QNI"])
def test_number_inventory_must_be_explicit_and_unique(declared):
    snapshot, cfg, coord, terrain, orography = _case(8)
    with pytest.raises(ValueError, match="analyzed_number_fields"):
        initialize_real(snapshot, cfg, coord, terrain, source_orography=orography,
            preprocess_backend="cpu", state_backend="preprocess", analyzed_number_fields=declared)


def test_missing_declared_number_field_refuses_before_interpolation():
    snapshot, cfg, coord, terrain, orography = _case(8)
    snapshot = replace(snapshot, fields={k:v for k,v in snapshot.fields.items() if k != "QNI"})
    with pytest.raises(KeyError, match="QNI"):
        initialize_real(snapshot, cfg, coord, terrain, source_orography=orography,
            preprocess_backend="cpu", state_backend="preprocess", analyzed_number_fields=("QNI",))


def test_surface_pseudo_level_can_be_the_only_number_source():
    snapshot, cfg, coord, terrain, orography = _case(8)
    snapshot.fields["QNI"][...] = 0
    kw = dict(source_orography=orography, preprocess_backend="cpu", state_backend="preprocess",
              analyzed_species=(), analyzed_number_fields=("QNI",))
    supplied = initialize_real(snapshot, cfg, coord, terrain, **kw)
    assert np.count_nonzero(supplied.state.ni) > 0
    snapshot.fields["QNI_SFC"][...] = 0
    absent = initialize_real(snapshot, cfg, coord, terrain, **kw)
    assert np.count_nonzero(absent.state.ni) == 0


def _mass_case(mp=18):
    snapshot, cfg, coord, terrain, orography = _case(mp)
    fields = dict(snapshot.fields)
    for name in ("QC", "QH"):
        fields[name] = np.zeros_like(fields["TT"])
        fields[name+"_SFC"] = np.full((cfg.ny,cfg.nx), .003, np.float32)
    return replace(snapshot, fields=fields), cfg, coord, terrain, orography


def test_mass_surface_analysis_reaches_state_before_moist_pressure_recurrence():
    snapshot, cfg, coord, terrain, orography = _mass_case()
    kw = dict(source_orography=orography, preprocess_backend="cpu", state_backend="preprocess",
              analyzed_species=("QC", "QH"), analyzed_surface_fields=("QC", "QH"))
    supplied = initialize_real(snapshot, cfg, coord, terrain, **kw)
    for name in ("qc", "qh"):
        assert np.count_nonzero(getattr(supplied.state, name)) > 0
    for name in ("QC", "QH"):
        snapshot.fields[name+"_SFC"][...] = 0
    empty = initialize_real(snapshot, cfg, coord, terrain, **kw)
    assert np.count_nonzero(empty.state.qc) == np.count_nonzero(empty.state.qh) == 0
    assert np.any(supplied.state.php != empty.state.php)
    receipt = supplied.hydrometeor_initialization
    assert receipt["schema"] == "gpuwm-metgrid-hydrometeor-initialization-v1"
    assert "vertical_disposition" not in receipt
    assert set(receipt["surface_pseudo_levels"]) == {"QC", "QH"}
    assert receipt["retained_correspondence"] == {"QC":"qc", "QH":"qh"}


def test_preparation_prices_original_mass_number_levels_and_surfaces():
    from types import SimpleNamespace
    from pathlib import Path
    from gpuwm.metem_door import metgrid_analysis_shapes
    names = ("QC", "QH", *METGRID_NUMBER_FIELDS)
    shapes = {name:(1,7,11,13) for name in ("TT","GHT","RH",*names)}
    shapes.update(UU=(1,7,11,14), VV=(1,7,12,13), PSFC=(1,11,13), SOILHGT=(1,11,13))
    metadata = SimpleNamespace(path=Path("met_em.test"), nx=13, ny=11,
        global_attributes={"FLAG_"+name:1 for name in names}, variables=shapes)
    actual = metgrid_analysis_shapes(metadata)
    for name in names:
        assert actual[name] == (6,11,13)
        assert actual[name+"_SFC"] == (11,13)


def test_explicit_metgrid_mass_inventory_honors_passive_vapor_package():
    snapshot, cfg, coord, terrain, orography = _mass_case(0)
    kw = dict(source_orography=orography, preprocess_backend="cpu", state_backend="preprocess")
    supplied = initialize_real(snapshot, cfg, coord, terrain, analyzed_species=("QC","QH"),
        analyzed_surface_fields=("QC","QH"), **kw)
    absent = initialize_real(snapshot, cfg, coord, terrain, analyzed_species=(), **kw)
    receipt = supplied.hydrometeor_initialization
    assert receipt["retained_correspondence"] == {}
    assert set(receipt["discarded_source_species"]) == {"QC","QH"}
    for name in ("mup", "php", "thp", "qv", "qc", "qr"):
        np.testing.assert_array_equal(getattr(supplied.state,name), getattr(absent.state,name))


def _no_wif_dataset(monkeypatch):
    """This machine has no QNWFA_QNIFA_SIGMA_MONTHLY.dat, deterministically.

    The 225 MB climatology is not staged in CI, but "not staged" must be a
    fact the test states rather than one it inherits from the machine it
    happens to run on.
    """
    from gpuwm.ingest import wif_climatology
    unresolved = wif_climatology.WifSourceResolution(
        path=None, origin="no copy staged on this machine",
        candidates=("$GPUWM_WIF_CLIMATOLOGY",
                    "~/.gpuwm/wif/QNWFA_QNIFA_SIGMA_MONTHLY.dat"),
        fallback_reason="no copy of the dataset is staged on this machine")
    monkeypatch.setattr(wif_climatology, "resolve_wif_climatology",
                        lambda *arguments, **keywords: unresolved)
    return unresolved


def _metem_root_case(mp=28, **overrides):
    """The mp=28 case as the met_em door actually builds the ROOT domain.

    ``import_namelists`` writes WRF's own default
    ``specified = [True] + [False] * (max_dom - 1)``
    (gpuwm/namelist_import.py) into the emitted TOML, so every met_em root
    domain carries external lateral boundaries.  A fixture that inherits
    ``specified = False`` never crosses the mp=28 lateral-forcing floor and
    so proves nothing about the route a met_em operator takes.
    """
    snapshot, cfg, coord, terrain, orography = _case(mp)
    settings = {"mp28_aerosol_source": "auto", "specified": True, **overrides}
    return replace(snapshot), replace(cfg, **settings), coord, terrain, orography


def test_analyzed_aerosol_runs_a_specified_met_em_root_with_no_wif_dataset(
        monkeypatch):
    """C-040 on the domain the met_em door builds, not on a fixture default.

    ``initialize_real`` asks the machine-dependent mp=28 precondition
    before it resolves the aerosol source, and that precondition refuses a
    ``specified`` domain whose machine holds no monthly WIF climatology.
    A run whose analysis carries the whole QNWFA/QNIFA pair opens no
    climatology at all, so the dataset is not a precondition of it and the
    floor is not asked.
    """
    _require_cpu_bridge()
    _no_wif_dataset(monkeypatch)
    snapshot, cfg, coord, terrain, orography = _metem_root_case()
    assert cfg.specified is True
    result = initialize_real(snapshot, cfg, coord, terrain,
        source_orography=orography, p_top=5000., preprocess_backend="cpu",
        state_backend="preprocess", analyzed_species=(),
        analyzed_number_fields=METGRID_NUMBER_FIELDS)
    aerosol = result.aerosol_initialization
    assert aerosol["aerosol_source"] == "metgrid-analyzed"
    assert aerosol["awaiting_profile_fill"] is False
    for name in ("nwfa", "nifa"):
        assert np.count_nonzero(getattr(result.state, name)) > 0
    assert result.state._external_scalar_boundary_fields == (
        "qv", "nwfa", "nifa")


def test_the_wif_floor_still_fires_and_names_the_route_that_now_exists(
        monkeypatch):
    """A surviving refusal names a way out that exists, or it is stale.

    The precondition sentence lives in the out-of-bounds ``gpuwm/config.py``
    and predates the analyzed route, so it names a dataset, an environment
    variable, a fetch command and the synthetic fallback, and not the one
    way out a met_em operator can actually take.  The call site says which
    half of the pair the analysis in hand is missing.
    """
    _no_wif_dataset(monkeypatch)
    snapshot, cfg, coord, terrain, orography = _metem_root_case()
    call = dict(source_orography=orography, p_top=5000.,
                preprocess_backend="cpu", state_backend="preprocess",
                analyzed_species=())
    with pytest.raises(ValueError) as neither:
        initialize_real(snapshot, cfg, coord, terrain,
                        analyzed_number_fields=("QNI",), **call)
    said = str(neither.value)
    assert "QNWFA_QNIFA_SIGMA_MONTHLY.dat" in said
    assert "carries neither QNWFA nor QNIFA" in said
    assert "FLAG_QNWFA=1 and FLAG_QNIFA=1" in said
    assert "leave mp28_aerosol_source at 'auto'" in said
    with pytest.raises(ValueError) as half:
        initialize_real(snapshot, cfg, coord, terrain,
                        analyzed_number_fields=("QNWFA",), **call)
    assert "carries QNWFA but not QNIFA" in str(half.value)


def test_half_an_analyzed_aerosol_pair_is_warned_about_and_never_claimed(
        monkeypatch, capsys):
    """FLAG_QNWFA and FLAG_QNIFA are independent; one of them is not a source.

    Treating either half as "the analysis supplies the aerosol" left the
    absent half at exact zero while the receipt said thompson_init's
    synthetic profile was not used -- untrue for that half, whose MAXVAL
    presence test reads an all-zero field and installs the synthetic
    profile -- and declared both as carried-from-input lateral boundary
    scalars, so the absent one drained aerosol-free air in with nothing
    saying so.
    """
    _require_cpu_bridge()
    _no_wif_dataset(monkeypatch)
    # specified=False so the lateral-forcing floor stays out of the way and
    # this test grades the aerosol resolver and nothing else.
    snapshot, cfg, coord, terrain, orography = _metem_root_case(specified=False)
    result = initialize_real(snapshot, cfg, coord, terrain,
        source_orography=orography, p_top=5000., preprocess_backend="cpu",
        state_backend="preprocess", analyzed_species=(),
        analyzed_number_fields=("QNWFA",))
    aerosol = result.aerosol_initialization
    assert aerosol["aerosol_source"] == "thompson_init-synthetic-profile"
    assert aerosol["awaiting_profile_fill"] is True
    assert "analyzed_aerosol_fields" not in aerosol
    for name in ("nwfa", "nifa", "nwfa2d", "nifa2d"):
        assert np.count_nonzero(getattr(result.state, name)) == 0
    assert result.state._external_scalar_boundary_fields == ("qv",)
    receipt = result.hydrometeor_initialization["number_moments"]
    assert receipt["discarded_by_incomplete_analyzed_aerosol_pair"] == {
        "fields": ["QNWFA"], "missing": ["QNIFA"]}
    assert "QNWFA" not in receipt["retained_correspondence"]
    said = capsys.readouterr().err
    assert "carries QNWFA" in said and "but not QNIFA" in said
    assert "FLAG_QNWFA=1 and FLAG_QNIFA=1" in said


def _namelist_climatology_case(**overrides):
    """The met_em root whose NAMELIST asked for the WIF climatology.

    ``gpuwm/namelist_import.py`` admits ``&physics use_aero_icbc = .true.``
    with ``&domains wif_input_opt = 1`` and emits the pair into the TOML,
    and ``validate_aerosol_source_options`` keeps it admissible, so this is
    a supported met_em import and not a hand-built configuration.  On it
    ``mp28_aerosol_source`` stays at its default 'auto' while the resolver
    reads the pair as 'climatology'.
    """
    return _metem_root_case(aer_init_opt=1, wif_input_opt=1, **overrides)


def test_the_wif_floor_names_the_selector_that_asked_and_not_a_no_op(
        monkeypatch):
    """A way out the operator has already taken is not a way out.

    ``(aer_init_opt, wif_input_opt) = (1, 1)`` resolves 'auto' to
    'climatology', so on this configuration "leave mp28_aerosol_source at
    'auto'" instructs the operator to leave a field exactly where it
    already is while the pair keeps selecting the dataset the refusal is
    about.  The sentence names the pair instead.
    """
    _no_wif_dataset(monkeypatch)
    snapshot, cfg, coord, terrain, orography = _namelist_climatology_case()
    assert cfg.mp28_aerosol_source == "auto" and cfg.specified is True
    call = dict(source_orography=orography, p_top=5000.,
                preprocess_backend="cpu", state_backend="preprocess",
                analyzed_species=())
    with pytest.raises(ValueError) as whole_pair:
        initialize_real(snapshot, cfg, coord, terrain,
                        analyzed_number_fields=METGRID_NUMBER_FIELDS, **call)
    said = str(whole_pair.value)
    assert "QNWFA_QNIFA_SIGMA_MONTHLY.dat" in said
    assert ("(aer_init_opt, wif_input_opt) = (1, 1) is what asks for the "
            "climatology here") in said
    assert "mp28_aerosol_source is already 'auto'" in said
    assert "Set either selector to 0 to take the analyzed pair." in said
    assert "leave mp28_aerosol_source at 'auto' to take it" not in said
    with pytest.raises(ValueError) as half:
        initialize_real(snapshot, cfg, coord, terrain,
                        analyzed_number_fields=("QNWFA",), **call)
    incomplete = str(half.value)
    assert "carries QNWFA but not QNIFA" in incomplete
    assert "FLAG_QNWFA=1 and FLAG_QNIFA=1" in incomplete
    assert ("set either selector of (aer_init_opt, wif_input_opt) = (1, 1) "
            "to 0") in incomplete
    assert "and leave mp28_aerosol_source at 'auto'." not in incomplete


def test_the_analyzed_discard_warning_names_the_namelist_pair_that_asked(
        monkeypatch, capsys):
    """The discard names WHO asked, not only WHAT was resolved.

    With the pair set, the resolved source is 'climatology' and
    ``mp28_aerosol_source`` was never touched, so a warning saying it "was
    requested explicitly" reports a field the operator never set and
    offers a change already made.
    """
    _require_cpu_bridge()
    _no_wif_dataset(monkeypatch)
    # specified=False keeps the lateral-forcing floor out of the way so
    # this test grades the discard warning and nothing else.
    snapshot, cfg, coord, terrain, orography = _namelist_climatology_case(
        specified=False)
    result = initialize_real(snapshot, cfg, coord, terrain,
        source_orography=orography, p_top=5000., preprocess_backend="cpu",
        state_backend="preprocess", analyzed_species=(),
        analyzed_number_fields=METGRID_NUMBER_FIELDS)
    said = capsys.readouterr().err
    assert "this analysis carries QNWFA/QNIFA" in said
    assert "(aer_init_opt, wif_input_opt) = (1, 1) asks for the WIF" in said
    assert "Set either selector to 0" in said
    assert "was requested explicitly" not in said
    discarded = result.hydrometeor_initialization["number_moments"][
        "discarded_by_requested_aerosol_source"]
    assert discarded["fields"] == ["QNWFA", "QNIFA"]
    assert discarded["mp28_aerosol_source"] == "climatology"
    assert discarded["requested_by"] == "namelist"
    assert discarded["aer_init_opt"] == 1 and discarded["wif_input_opt"] == 1


def test_an_explicitly_named_source_is_still_attributed_to_the_field(capsys):
    """The other spelling keeps its own attribution, pair or no pair."""
    _require_cpu_bridge()
    snapshot, cfg, coord, terrain, orography = _case(28)
    cfg = replace(cfg, specified=False, mp28_aerosol_source="synthetic")
    result = initialize_real(snapshot, cfg, coord, terrain,
        source_orography=orography, p_top=5000., preprocess_backend="cpu",
        state_backend="preprocess", analyzed_species=(),
        analyzed_number_fields=METGRID_NUMBER_FIELDS)
    said = capsys.readouterr().err
    assert "mp28_aerosol_source='synthetic' was requested explicitly" in said
    assert "aer_init_opt" not in said
    discarded = result.hydrometeor_initialization["number_moments"][
        "discarded_by_requested_aerosol_source"]
    assert discarded["requested_by"] == "mp28_aerosol_source"
    assert "aer_init_opt" not in discarded

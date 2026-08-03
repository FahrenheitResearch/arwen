"""Stock-WRF package inventories and arbitrary vertical-grid gates."""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.vertical_contract import (
    expected_coordinate_shapes,
    validate_explicit_eta_grid,
)
from gpuwm.wrf_physics_inventory import (
    WRFINPUT_2D_DIMS,
    WRFINPUT_3D_DIMS,
    stock_wrf_physics_inventory,
    supported_stock_wrf_mp_physics,
)


@pytest.mark.parametrize(
    ("mp_physics", "expected"),
    [
        (6, ("QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")),
        (
            8,
            (
                "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                "QNICE", "QNRAIN",
            ),
        ),
        (
            10,
            (
                "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                "QNICE", "QNSNOW", "QNRAIN", "QNGRAUPEL",
            ),
        ),
        (
            18,
            (
                "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
                "QHAIL", "QNDROP", "QNRAIN", "QNICE", "QNSNOW",
                "QNGRAUPEL", "QNHAIL", "QNCCN", "QVGRAUPEL", "QVHAIL",
            ),
        ),
    ],
)
def test_wrf_v461_registry_package_inventory(mp_physics, expected):
    inventory = stock_wrf_physics_inventory(mp_physics)
    assert tuple(field.netcdf_name for field in inventory.wrfinput_fields) == expected
    assert all(field.dtype == "float32" for field in inventory.wrfinput_fields)
    assert all(field.dimensions == WRFINPUT_3D_DIMS
               for field in inventory.wrfinput_fields)
    assert inventory.wrfinput_fields[0].initialization == "source_specific_humidity"
    assert all(
        field.initialization == "zero_if_source_absent"
        for field in inventory.wrfinput_fields[1:]
    )


def test_runtime_only_registry_state_is_not_claimed_as_wrfinput():
    for value in supported_stock_wrf_mp_physics():
        inventory = stock_wrf_physics_inventory(value)
        input_names = {field.netcdf_name for field in inventory.wrfinput_fields}
        runtime_names = {
            field.netcdf_name for field in inventory.runtime_state_not_wrfinput
        }
        assert input_names.isdisjoint(runtime_names)
    assert {
        field.netcdf_name
        for field in stock_wrf_physics_inventory(10).runtime_state_not_wrfinput
    } == {"RQRCUTEN", "RQSCUTEN", "RQICUTEN"}
    assert {
        field.netcdf_name
        for field in stock_wrf_physics_inventory(18).runtime_state_not_wrfinput
    } == {"RE_CLOUD", "RE_ICE", "RE_SNOW"}


def test_nssl2_default_inventory_pins_exact_registry_packages_and_units():
    inventory = stock_wrf_physics_inventory(18)

    assert inventory.scheme == "NSSL-2"
    assert inventory.registry_package == (
        "nssl_2mom+nssl2mconc+nssl_hail+nssl_ccn_opt+nssl_hailvol"
    )
    assert {
        field.netcdf_name: field.units for field in inventory.wrfinput_fields
    } == {
        "QVAPOR": "kg kg-1",
        "QCLOUD": "kg kg-1",
        "QRAIN": "kg kg-1",
        "QICE": "kg kg-1",
        "QSNOW": "kg kg-1",
        "QGRAUP": "kg kg-1",
        "QHAIL": "kg kg-1",
        "QNDROP": "# kg-1",
        "QNRAIN": "# kg(-1)",
        "QNICE": "# kg-1",
        "QNSNOW": "# kg(-1)",
        "QNGRAUPEL": "# kg(-1)",
        "QNHAIL": "# kg(-1)",
        "QNCCN": "# kg(-1)",
        "QVGRAUPEL": "m(3) kg(-1)",
        "QVHAIL": "m(3) kg(-1)",
    }


def test_unknown_package_fails_closed_with_actionable_message():
    # 38 is thompsongh (Registry.EM_COMMON:3037), a real WRF v4.6.1 scheme
    # with a real package that this module deliberately does not inventory.
    # It replaces 28, which now HAS a row -- the refusal is still the
    # subject, only the uninventoried example moved.
    with pytest.raises(ValueError, match=r"mp_physics=38.*Registry package"):
        stock_wrf_physics_inventory(38)
    with pytest.raises(TypeError, match="WRF integer"):
        stock_wrf_physics_inventory(True)
    message = str(
        pytest.raises(ValueError, stock_wrf_physics_inventory, 38).value)
    assert "28" in message, (
        "the refusal must list mp_physics=28 among the evidenced schemes")


@pytest.mark.parametrize("mass_levels", [35, 49, 80])
def test_explicit_vertical_grid_is_structural_not_49_level_allowlist(mass_levels):
    eta = np.linspace(1.0, 0.0, mass_levels + 1, dtype=np.float64)
    checked = validate_explicit_eta_grid(
        eta,
        nz=mass_levels,
        p_top=5000.0,
        source_top_pressure_pa=5000.0,
        context=f"{mass_levels}-mass-level RW-WPS fixture",
    )
    assert np.array_equal(checked, eta)
    shapes = expected_coordinate_shapes(mass_levels)
    assert shapes["znu"] == (mass_levels,)
    assert shapes["znw"] == (mass_levels + 1,)


def test_vertical_source_coverage_fails_closed():
    eta = np.linspace(1.0, 0.0, 81, dtype=np.float64)
    with pytest.raises(ValueError, match=r"source atmosphere stops at 10000 Pa"):
        validate_explicit_eta_grid(
            eta,
            nz=80,
            p_top=5000.0,
            source_top_pressure_pa=10000.0,
            context="80-level unsupported source top",
        )


# ---------------------------------------------------------------------------
# mp_physics=28, Thompson aerosol-aware (Registry.EM_COMMON:3036).
#
# Every claim below is bound to WRF v4.6.1 commit
# d66e442fccc04111067e29274c9f9eaccc3cef28 -- either to a Registry line or,
# where the Registry alone is ambiguous, to the code the Registry GENERATES
# in the stock build tree (inc/scalar_indices.inc), which is the only place
# that settles what WRF actually allocates and writes.
# ---------------------------------------------------------------------------

MP28_WRFINPUT_FIELDS = (
    # moist:qv,qc,qr,qi,qs,qg -- character for character mp=8's list (:3024)
    "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP",
    # scalar:qni,qnr,qnc,qnwfa,qnifa,qnbca -- SIX, not five
    "QNICE", "QNRAIN", "QNCLOUD", "QNWFA", "QNIFA", "QNBCA",
    # state:qnwfa2d,qnifa2d -- 2-D, and wrfinput because their I/O string
    # begins with i0 (Registry.EM_COMMON:492-493)
    "QNWFA2D", "QNIFA2D",
)


def test_mp28_inventory_is_the_thompsonaero_package():
    inventory = stock_wrf_physics_inventory(28)

    assert inventory.mp_physics == 28
    assert inventory.scheme == "Thompson aerosol-aware"
    assert inventory.registry_package == "thompsonaero"
    assert inventory.registry_authority == (
        "WRF-v4.6.1 Registry/Registry.EM_COMMON")
    assert tuple(
        field.netcdf_name for field in inventory.wrfinput_fields
    ) == MP28_WRFINPUT_FIELDS
    assert all(field.dtype == "float32" for field in inventory.wrfinput_fields)
    assert inventory.wrfinput_fields[0].initialization == (
        "source_specific_humidity")
    assert all(
        field.initialization == "zero_if_source_absent"
        for field in inventory.wrfinput_fields[1:]
    )


def test_mp28_moist_list_is_identical_to_classic_thompsons():
    """The one identity that makes the ingest lane's mass path shareable.

    Registry.EM_COMMON:3024 (``thompson``) and :3036 (``thompsonaero``) both
    declare ``moist:qv,qc,qr,qi,qs,qg``.  Everything mp=28 adds is a
    ``scalar:`` or ``state:`` member.  ``gpuwm/ingest/real.py`` relies on this
    to feed 28 the same decoded HRRR analyzed-hydrometeor inventory it feeds
    8, so it is pinned here rather than assumed there.
    """
    moist = {
        mp: tuple(
            field.netcdf_name
            for field in stock_wrf_physics_inventory(mp).wrfinput_fields
            if field.collection == "moist"
        )
        for mp in (8, 28)
    }
    assert moist[28] == moist[8]
    assert moist[28] == (
        "QVAPOR", "QCLOUD", "QRAIN", "QICE", "QSNOW", "QGRAUP")


def test_mp28_carries_qnbca_because_the_package_declares_it():
    """QNBCA is a thompsonaero member for EVERY mp=28 run, not only wif=2.

    ``Registry/Registry.EM_COMMON:3036`` lists ``qnbca`` in the package's own
    ``scalar:`` list, and WRF's generated registration proves the
    consequence: the ``mp_physics(idomain)==28`` block of the stock build
    tree's ``inc/scalar_indices.inc`` (:2449-2618) sets
    ``scalar_dname_table( idomain, P_qnbca ) = 'QNBCA'`` (:2612) and
    ``F_qnbca = .TRUE.`` (:2617) with no wif_input_opt guard; the
    ``wif_input_opt==2`` block at :18973-18988 registers it a second,
    idempotent time.  This module answers what an unchanged WRF v4.6.1
    executable expects in wrfinput, so QNBCA is listed even though gpuwm
    implements no nbca species anywhere.
    """
    names = {
        field.netcdf_name for field in stock_wrf_physics_inventory(
            28).wrfinput_fields
    }
    assert "QNBCA" in names

    # ... and that is a statement about WRF, not about gpuwm: gpuwm refuses
    # the only selector under which it would have to carry the species.
    from gpuwm.config import MP28_AEROSOL_SOURCE_OPTIONS

    only, citation, why = MP28_AEROSOL_SOURCE_OPTIONS["wif_input_opt"]
    assert only == 0
    assert "qnbca" in why
    assert citation == "Registry/registry.new3d_wif:17"


def test_mp28_surface_emissions_are_two_dimensional_wrfinput_members():
    """``state:`` with ``i0`` is an input field, and it is 2-D.

    ``qnwfa2d``/``qnifa2d`` (Registry.EM_COMMON:492-493) carry the I/O string
    ``i01{17}rhdu``.  The ``i`` list begins with stream 0 -- the same ``i0``
    prefix as unambiguous wrfinput members HGT (:1407) and TSK (:1417) -- so
    unlike ``re_cloud``/``re_ice``/``re_snow`` (:497-499, bare ``r``) they are
    initialization-file variables.  ``real.exe`` writes them at
    ``dyn_em/module_initialize_real.F:4496-4653``, zeroing both under
    aer_init_opt=0 (:4501-4510).
    """
    inventory = stock_wrf_physics_inventory(28)
    by_name = {field.netcdf_name: field for field in inventory.wrfinput_fields}

    for name in ("QNWFA2D", "QNIFA2D"):
        field = by_name[name]
        assert field.dimensions == WRFINPUT_2D_DIMS
        assert "bottom_top" not in field.dimensions
        assert field.collection == "state"
        assert field.units == "kg-1 s-1"
        assert field.initialization == "zero_if_source_absent"

    for name in MP28_WRFINPUT_FIELDS:
        if name in ("QNWFA2D", "QNIFA2D"):
            continue
        assert by_name[name].dimensions == WRFINPUT_3D_DIMS


def test_mp28_runtime_state_is_the_radii_plus_the_optical_depths():
    """``state:`` members with ``r`` and no ``i`` are runtime/restart only.

    The thompsonaero package's ``state:`` list is
    ``re_cloud,re_ice,re_snow,qnwfa2d,qnifa2d,taod5503d,taod5502d``.  Five of
    the seven have no input stream: re_cloud/re_ice/re_snow are bare ``r``
    (Registry.EM_COMMON:497-499), taod5503d is ``r`` and taod5502d is ``rh``
    (:1738-1739).  taod5502d is 2-D (``ij``).
    """
    inventory = stock_wrf_physics_inventory(28)
    runtime = {
        field.netcdf_name: field
        for field in inventory.runtime_state_not_wrfinput
    }

    assert set(runtime) == {
        "RE_CLOUD", "RE_ICE", "RE_SNOW", "TAOD5503D", "TAOD5502D"}
    assert runtime["TAOD5502D"].dimensions == WRFINPUT_2D_DIMS
    assert runtime["TAOD5503D"].dimensions == WRFINPUT_3D_DIMS
    assert all(field.initialization == "wrf_runtime"
               for field in runtime.values())
    input_names = {
        field.netcdf_name for field in inventory.wrfinput_fields}
    assert input_names.isdisjoint(runtime)


def test_mp28_units_are_wrfs_resolved_values_not_the_registry_line():
    """WRF blanks every ``#`` inside a quoted Registry string.

    ``tools/reg_parse.c:203-208`` walks the raw line before tokenizing:
    ``else if ( *p == '#' && inquote ) *p = ' ' ;`` (:206).  So the Registry
    text ``"# kg(-1)"`` becomes ``'  kg(-1)'``, which is what the generated
    table carries -- inside the mp_physics==28 block of the stock tree's
    ``inc/scalar_indices.inc``, ``P_qni`` is ``'  kg-1'`` (:2544) and
    ``P_qnr`` (:2558), ``P_qnc`` (:2572), ``P_qnwfa`` (:2586), ``P_qnifa``
    (:2600) and ``P_qnbca`` (:2614) are ``'  kg(-1)'`` -- and what WRF's own
    writer emits as the NetCDF ``units`` attribute
    (``inc/wrf_bdyout.inc:1449``).

    gpuwm's independently verified wrfout metadata agrees, and is checked
    here so the two cannot drift: a units string that appears in no WRF file
    would be a fabricated attribute.
    """
    from gpuwm.io.wrfout import _VAR_META

    units = {
        field.netcdf_name: field.units
        for field in stock_wrf_physics_inventory(28).wrfinput_fields
    }

    assert units["QNICE"] == "  kg-1"
    for name in ("QNRAIN", "QNCLOUD", "QNWFA", "QNIFA", "QNBCA"):
        assert units[name] == "  kg(-1)", name
    for name, value in units.items():
        assert "#" not in value, (
            f"{name} units {value!r} carries a '#' that WRF's own Registry "
            "parser blanks (tools/reg_parse.c:206)")

    for name in ("QNCLOUD", "QNRAIN", "QNICE"):
        assert units[name] == _VAR_META[name][1], (
            f"{name} units disagree with gpuwm's wrfout metadata, which was "
            "verified against a stock v4.6.1 wrfout")


def test_mp28_is_reachable_through_every_inventory_front_door():
    """The row exists so the mp=28 export/report paths stop failing closed.

    ``gpuwm/namelist_compat.py`` and ``gpuwm/prepared_single_domain_forecast``
    both look the scheme up before they can say anything about an mp=28
    configuration; until this row landed they raised and reported
    "unsupported microphysics" for a runtime that implements the scheme.
    """
    assert 28 in supported_stock_wrf_mp_physics()
    assert supported_stock_wrf_mp_physics() == (6, 8, 10, 18, 28)

    report = stock_wrf_physics_inventory(28).as_report()
    assert report["schema"] == "rw-wps.stock-wrf-physics-inventory.v1"
    assert report["target"] == "stock_wrf_v4.6.1"
    assert report["mp_physics"] == 28
    assert report["scheme"] == "Thompson aerosol-aware"
    assert report["registry_package"] == "thompsonaero"
    assert [
        field["netcdf_name"] for field in report["wrfinput_fields"]
    ] == list(MP28_WRFINPUT_FIELDS)
    assert [
        field["netcdf_name"] for field in report["runtime_state_not_wrfinput"]
    ] == ["RE_CLOUD", "RE_ICE", "RE_SNOW", "TAOD5503D", "TAOD5502D"]

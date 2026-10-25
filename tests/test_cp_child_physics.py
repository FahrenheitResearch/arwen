"""Children finer than 4 km run the HRRR-shape suite unless a suite is named.

The defect (WOOF-INIT-FROM-ANY-RUN-2026-10-24/FOLLOW-UPS item 1): a 1 km
`gpuwm downscale` of a 3 km Morrison parent inherited Morrison (ghost-rain
streaks in 9.6 % of columns, the parent's graupel falling out as a block in
the relaxation zone); a 3 km hex-global child took the same Morrison default.
"""

import numpy as np

from gpuwm.downscale import child_physics_config
from gpuwm.hrrr_route_inputs import ROUTE_DEFAULT_PHYSICS_PROFILE
from gpuwm.offline_child import (noah_soil_onto_ruc_levels,
                                 parent_soil_onto_child_levels)
from gpuwm.physics_menu import default_profile_for

_MORRISON_PARENT = {"mp_physics": 10, "bl_pbl_physics": 1,
                    "sf_surface_physics": 2, "num_soil_layers": 4,
                    "cu_physics": 1, "epssm": 0.2, "dt": 18.0}


def test_a_child_finer_than_4km_takes_the_hrrr_shape_suite_and_a_named_one_wins():
    cfg, profile, _ = child_physics_config(_MORRISON_PARENT, child_dx_m=1000.0,
                                           requested=None)
    assert profile == ROUTE_DEFAULT_PHYSICS_PROFILE
    assert (cfg["mp_physics"], cfg["bl_pbl_physics"], cfg["sf_surface_physics"],
            cfg["num_soil_layers"], cfg["cu_physics"]) == (28, 5, 3, 9, 0)
    assert cfg["epssm"] == 0.2
    for dx, requested in ((4000.0, None), (1000.0, "parent")):
        cfg, profile, _ = child_physics_config(_MORRISON_PARENT, child_dx_m=dx,
                                               requested=requested)
        assert profile is None and cfg["mp_physics"] == 10


def test_a_hex_global_child_finer_than_4km_defaults_to_the_same_suite():
    assert default_profile_for("hex-global", 3000.0, 1) == ROUTE_DEFAULT_PHYSICS_PROFILE
    assert default_profile_for("hex-global", 9000.0, 1) != ROUTE_DEFAULT_PHYSICS_PROFILE
    assert default_profile_for("gfs", 3000.0, 1) != ROUTE_DEFAULT_PHYSICS_PROFILE


def test_a_noah_parent_soil_column_is_rebuilt_on_ruc_levels():
    assert parent_soil_onto_child_levels(4, 9)
    assert not parent_soil_onto_child_levels(9, 4)
    shape = (3, 2)
    fields = {"TSLB": np.stack([np.full(shape, t, np.float32) for t in (290, 288, 286, 284)]),
              "SMOIS": np.stack([np.full(shape, m, np.float32) for m in (0.30, 0.28, 0.26, 0.25)]),
              "SH2O": np.zeros((4,) + shape, np.float32),
              "TSK": np.full(shape, 295.0, np.float32), "TMN": np.full(shape, 285.0, np.float32),
              "LANDMASK": np.ones(shape, np.float32)}
    out = noah_soil_onto_ruc_levels(fields, parent_zs_m=[0.05, 0.25, 0.7, 1.5],
                                    num_soil_layers=9)
    assert out["TSLB"].shape == (9,) + shape and out["SMOIS"].shape == (9,) + shape
    assert "SH2O" not in out
    assert np.allclose(out["TSLB"][0], 295.0)      # the 0 m node is the skin
    assert np.allclose(out["TSLB"][-1], 285.0)     # the 3 m node is the deep soil


def test_a_morrison_to_thompson_edge_carries_rain_and_ice_numbers():
    """Re-diagnosed rain numbers drew a 15-28 dBZ blanket over 9,704 first-frame columns."""
    from types import SimpleNamespace
    from gpuwm.core.microphysics_transition import resolve_microphysics_transition

    def cfg(mp, policy="same-scheme-v1"):
        return SimpleNamespace(mp_physics=mp, moist=True, moist_cq=True, morr_rimed_ice=1,
                               wsm6_hail_opt=0, wdm6_hail_opt=0, wdm6_ccn_conc=1.0e8,
                               nest_microphysics_transition=policy)

    contract = resolve_microphysics_transition(cfg(10), cfg(28, "mp-edge-mass-diagnosed-v1"))
    assert contract.number_source("nr") == "nr" and contract.number_source("ni") == "ni"
    assert contract.number_source("nc") is None
    mapped = {row["target_field"] for row in contract.species_actions()
              if row["reason"] == "shared_exponential_number_moment"}
    assert mapped == {"nr", "ni"}


def test_a_carried_number_outside_thompsons_size_bounds_is_set_to_the_bound():
    """Morrison's 7e-8 kg/kg rain beside 1e-6 per kg drew 50-66 dBZ first-frame points."""
    from gpuwm.ingest import handover_numbers as hn

    masses = {"qr": np.array([7.0e-8, 1.0e-4], np.float32), "qi": np.zeros(2, np.float32)}
    numbers = {"nr": np.array([1.0e-6, 3.0e3], np.float32), "ni": np.zeros(2, np.float32)}
    before = numbers["nr"].copy()
    receipt = {}
    hn._bound_thompson_numbers(masses, numbers, rho=np.ones(2, np.float32),
                               at=lambda value, idx, shape: value[idx], receipt=receipt)
    assert numbers["nr"][0] > 1.0e2 * before[0]          # the trace rain gets the 2.5 mm bound
    assert numbers["nr"][1] == before[1]                  # an in-bounds number is untouched
    assert receipt["rain_number_bounded"]["cells"] == 1


def test_exact_mode_turns_the_number_carry_and_the_thompson_bounds_off():
    """HN1 off (exact mode) keeps 2.9.6's re-diagnosed edge numbers and leaves numbers unbounded.

    The review fix 5c436be6c3 made every handover number closure correct-mode key HN1; the
    carry and the Thompson entry bounds this branch adds are number closures too, so an
    exact-mode Morrison -> Thompson child stays byte-identical to 2.9.6.
    """
    from types import SimpleNamespace
    from gpuwm.core.microphysics_transition import resolve_microphysics_transition
    from gpuwm.offline_child import _offline_transition_contract, close_child_numbers

    def cfg(mp, policy="same-scheme-v1", **extra):
        return SimpleNamespace(mp_physics=mp, moist=True, moist_cq=True, morr_rimed_ice=1,
                               wsm6_hail_opt=0, wdm6_hail_opt=0, wdm6_ccn_conc=1.0e8,
                               nest_microphysics_transition=policy, **extra)

    exact = resolve_microphysics_transition(
        cfg(10), cfg(28, "mp-edge-mass-diagnosed-v1", hydrometeor_handover_numbers=False))
    assert exact.carry_numbers is False
    assert exact.number_source("nr") is None and exact.number_source("ni") is None
    assert not any(row["reason"] == "shared_exponential_number_moment"
                   for row in exact.species_actions())
    offline = _offline_transition_contract(
        10, 28, morr_rimed_ice=1, hail_opt=None,
        child_cfg=cfg(28, hydrometeor_handover_numbers=False))
    assert offline.number_source("nr") is None
    correct = _offline_transition_contract(
        10, 28, morr_rimed_ice=1, hail_opt=None,
        child_cfg=cfg(28, hydrometeor_handover_numbers=True))
    assert correct.number_source("nr") == "nr"
    mixing = {"qr": np.full((1, 1, 2), 7.0e-8, np.float32),
              "nr": np.full((1, 1, 2), 1.0e-6, np.float32)}
    fields = {"T": np.zeros((1, 1, 2), np.float32), "P": np.zeros((1, 1, 2), np.float32),
              "PB": np.full((1, 1, 2), 9.0e4, np.float32)}
    before = mixing["nr"].copy()
    assert close_child_numbers(28, mixing, fields,
                               cfg={"hydrometeor_handover_numbers": False}) is None
    assert np.array_equal(mixing["nr"], before)

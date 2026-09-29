"""A 3 km HRRR root takes the widest soil donor search wherever it sits.

The breakage this guards: every HRRR target carried a fixed 8-cell
(24 km) soil donor search, so a fitted 3 km root over the Gulf coast
failed its preparation with "no valid surface-matched HRRR donor within
8 cells" while its own refusal measured that 14 cells reach one.  The
search then shrank back toward 8 near an edge of HRRR's grid, where a
wider search box was refused; the box now stops at HRRR's own edge, so
the widest search is given everywhere.
"""

from __future__ import annotations

import json

from gpuwm.hrrr_route_inputs import (SURFACE_DONOR_RADIUS_CELLS,
                                     target_coverage_refusal)
from gpuwm.ingest.hrrr_target import (HRRR_SOURCE_NX, HRRR_SOURCE_NY,
                                      HrrrTargetDomain,
                                      load_hrrr_target_domain,
                                      required_hrrr_source_window)


def _document(nx, ny, lat, lon):
    return {"name": "room", "map_proj": "lambert", "nx": nx, "ny": ny, "nz": 49,
            "dx_m": 3000.0, "dy_m": 3000.0, "ref_lat": lat, "ref_lon": lon,
            "truelat1": 30.0, "truelat2": 60.0, "stand_lon": lon,
            "time_step_seconds": 15, "spec_bdy_width": 5, "spec_zone": 1,
            "relax_zone": 4}


def _widest(document):
    return HrrrTargetDomain(**{**document, "surface_fallback_radius_cells":
                               SURFACE_DONOR_RADIUS_CELLS})


def test_a_root_with_room_gets_the_widest_radius_and_passes_the_coverage_test(tmp_path):
    assert SURFACE_DONOR_RADIUS_CELLS == 24
    document = _document(363, 335, 32.4, -89.8)
    assert target_coverage_refusal(_widest(document)) is None
    spec = tmp_path / "d01.json"
    spec.write_text(json.dumps({"schema": "gpuwm-hrrr-target-domain-v1", **document,
                                "surface_fallback_radius_cells": SURFACE_DONOR_RADIUS_CELLS}))
    assert (load_hrrr_target_domain(spec).surface_fallback_radius_cells
            == SURFACE_DONOR_RADIUS_CELLS)


def test_a_root_against_the_edge_gets_the_widest_radius_too():
    # The field root that sat on HRRR's top edge: it used to keep 8 because
    # a wider search box was refused there.  The box stops at the edge.
    document = _document(1234, 986, 39.0, -98.0)
    document.update(truelat1=29.0, truelat2=49.0)
    target = _widest(document)
    assert target_coverage_refusal(target) is None
    window = required_hrrr_source_window(target)
    assert window.j_end == HRRR_SOURCE_NY - 1
    assert window.surface_fallback_radius_cells == SURFACE_DONOR_RADIUS_CELLS


def test_the_donor_radius_never_decides_coverage():
    # Inside HRRR, at its edge, and past its top (the last one's own
    # interpolation needs rows HRRR does not have): the widest search
    # gets exactly the verdict no search at all gets.
    verdicts = {}
    for nx, ny, lat, lon in ((900, 700, 38.0, -97.0), (1100, 800, 36.0, -95.0),
                             (500, 300, 26.0, -81.0), (400, 400, 47.5, -122.0)):
        document = _document(nx, ny, lat, lon)
        verdict = target_coverage_refusal(_widest(document))
        assert verdict == target_coverage_refusal(HrrrTargetDomain(
            **{**document, "surface_fallback_radius_cells": 0})), (nx, ny, lat, lon)
        verdicts[(nx, ny, lat, lon)] = verdict
        if verdict is None:
            window = required_hrrr_source_window(_widest(document))
            assert 0 <= window.i_start and window.i_end <= HRRR_SOURCE_NX - 1
            assert 0 <= window.j_start and window.j_end <= HRRR_SOURCE_NY - 1
    assert verdicts[(900, 700, 38.0, -97.0)] is None
    assert "leaves HRRR coverage" in verdicts[(400, 400, 47.5, -122.0)]

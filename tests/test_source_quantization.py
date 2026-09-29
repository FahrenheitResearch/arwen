"""The Python half of the bound-kissing class the GFS bridge closed.

An audit reproduced the reported `1.0000000019` refusal on the HRRR and
mapped routes as well; these pin the fix at every seam it reached.
"""

import numpy as np
import pytest

from gpuwm.ingest.quantization import (
    admit_bounded, bound_tolerance, clamp_bound_kissing)

#: The value a field user's GFS preparation died on, and the value the
#: audit reproduced on the HRRR and mapped routes.
REPORTED = 1.0000000019


def test_a_cell_at_a_bound_is_moved_onto_it_and_counted():
    values, report = clamp_bound_kissing(
        np.array([0.0, 0.25, REPORTED, 1.0]), minimum=0.0, maximum=1.0)
    assert report.clamps == 1
    assert report.max_excursion == pytest.approx(REPORTED - 1.0, rel=1e-9)
    assert values.tolist() == [0.0, 0.25, 1.0, 1.0]


def test_a_cell_past_the_bound_is_left_for_the_callers_refusal():
    """Not clamped, and not silently passed either: left exactly as it
    was, so the gate that already existed still sees it and still says
    no."""

    values, report = clamp_bound_kissing(
        np.array([0.5, 1.05]), minimum=0.0, maximum=1.0)
    assert report.clamps == 0
    assert values.tolist() == [0.5, 1.05]
    assert np.any(values > 1.0)


def test_a_dry_cell_below_zero_clamps_on_the_span_not_the_bound():
    """A zero bound has no magnitude of its own to be a fraction of, so
    the head-room comes from the range instead -- otherwise no encoder's
    tiny negative zero could ever be admitted."""

    assert bound_tolerance(0.0, 1.0) > 0.0
    values, report = clamp_bound_kissing(
        np.array([-1.9e-9, 0.5]), minimum=0.0, maximum=1.0)
    assert report.clamps == 1
    assert values[0] == 0.0


def test_admission_refuses_past_the_head_room_and_shows_its_arithmetic():
    with pytest.raises(ValueError) as refused:
        admit_bounded(np.array([0.5, 1.05]), name="SOILW",
                      minimum=0.0, maximum=1.0, subject="source")
    message = str(refused.value)
    assert "SOILW" in message
    assert "1.05" in message
    assert "quantization" in message and "clamped" in message


def test_admission_passes_the_reported_value_through():
    values, report = admit_bounded(
        np.array([0.2, REPORTED]), name="SOILW", minimum=0.0, maximum=1.0)
    assert report.clamps == 1
    assert values.max() == 1.0


def test_the_input_array_is_never_modified_in_place():
    original = np.array([REPORTED, 0.5])
    kept = original.copy()
    clamp_bound_kissing(original, minimum=0.0, maximum=1.0)
    assert original.tolist() == kept.tolist()


def test_the_hrrr_source_gate_admits_the_reported_value_and_refuses_a_bad_one():
    from gpuwm.ingest.hrrr import _require_source_physical_ranges

    def source(**overrides):
        base = dict(
            LANDSEA=np.array([1.0]), SOILW=np.array([1.0]),
            SOILT=np.array([273.0]), SPFH=np.array([0.01]),
            Q2=np.array([0.01]))
        base.update({k: np.array([v]) for k, v in overrides.items()})
        return base

    for name in ("LANDSEA", "SOILW"):
        _require_source_physical_ranges(source(**{name: REPORTED}))
        with pytest.raises(ValueError, match=name):
            _require_source_physical_ranges(source(**{name: 1.05}))
    # The plausibility windows are untouched: nothing real approaches
    # them, so nothing about packing applies.  SOILT's window holds on
    # source OPEN WATER; on source land the soil initializer rebuilds an
    # out-of-band column TSK-to-TMN as real.exe does, so there it is
    # admitted for that rebuild, as it is on an ice-covered water cell,
    # where HRRR runs its land-ice column.
    with pytest.raises(ValueError, match="SOILT"):
        _require_source_physical_ranges(
            source(SOILT=400.0000001, LANDSEA=0.0))
    for soilt in (400.0000001, 64.0):
        with pytest.raises(ValueError, match="SOILT.*water"):
            _require_source_physical_ranges(source(SOILT=soilt, LANDSEA=0.0))
        _require_source_physical_ranges(source(SOILT=soilt, LANDSEA=1.0))
    # One source window holding both: the land cell is admitted, the
    # water cell still refuses the whole window.
    window = dict(
        LANDSEA=np.array([1.0, 0.0]), SOILW=np.full((2, 2), 0.3),
        SOILT=np.array([[64.0, 280.0], [150.0, 280.0]]),
        SPFH=np.full((2, 2), 0.01), Q2=np.full(2, 0.01))
    _require_source_physical_ranges(window)
    window["SOILT"][1, 1] = 64.0
    with pytest.raises(ValueError, match="SOILT.*on 1 source open-water cell"):
        _require_source_physical_ranges(window)
    # The measured 2017-01-19 00Z shape: a one-cell frozen lake (LANDSEA
    # 0, XICE 1) carrying the 158 K snowpack column of the land around it.
    window["XICE"] = np.array([0.0, 1.0])
    window["SOILT"][:, 1] = 157.6
    _require_source_physical_ranges(window)
    window["XICE"] = np.array([0.0, 0.0])
    with pytest.raises(ValueError, match="SOILT.*on 1 source open-water cell"):
        _require_source_physical_ranges(window)

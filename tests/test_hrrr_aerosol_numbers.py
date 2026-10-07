"""HRRR's analyzed aerosol numbers reach the route that asks for them.

The operational HRRR wrfnat files publish QNWFA and QNIFA (NCEP local
0/13/193 and 0/13/192, kg-1) on all 50 hybrid levels.  The native bridge
did not read them, so an mp=28 tree on --source hrrr asking for the
analyzed aerosol (use_rap_aero_icbc, configs/recipes/hrrr_v4_gsd41.toml)
refused "missing QNIFA, QNWFA" with the fields in the file.
"""
from __future__ import annotations

import numpy as np
import pytest

from gpuwm.ingest.native_supplements import gate_optional_hybrid_fields


def test_the_gate_declares_the_pair_whole_in_its_unit():
    assert gate_optional_hybrid_fields({}) == ()
    assert gate_optional_hybrid_fields({
        "optional_hybrid_fields": "QNWFA,QNIFA",
        "optional_hybrid_units": "QNWFA=kg-1,QNIFA=kg-1"}) == ("QNWFA", "QNIFA")
    for gate in ({"optional_hybrid_fields": "QNWFA",
                  "optional_hybrid_units": "QNWFA=kg-1"},
                 {"optional_hybrid_fields": "QNWFA,QNIFA",
                  "optional_hybrid_units": "QNWFA=cm-3,QNIFA=kg-1"}):
        with pytest.raises(ValueError, match="QNWFA,QNIFA pair"):
            gate_optional_hybrid_fields(gate)


def test_the_mapping_carries_the_pair_and_its_deepest_level_surface(
        monkeypatch):
    """Nearest from the source (HRRR METGRID.TBL), surface pseudo-level
    from the deepest source level, as the generic routes do."""

    import gpuwm.ingest.hrrr as hrrr

    calls = []

    class Plan:
        def apply(self, field, *, method="parabolic", **_kwargs):
            calls.append(method)
            return np.asarray(field, dtype=np.float32)

    pres = np.stack([np.full((3, 4), 1000.0 - 10.0 * k, dtype=np.float32)
                     for k in range(5)])
    number = np.stack([np.full((3, 4), 100.0 + k, dtype=np.float32)
                       for k in range(5)])
    out = {"PRES": pres}
    source = {"QNWFA": number, "QNIFA": 2 * number}
    for name in ("QNWFA", "QNIFA"):
        out[name] = Plan().apply(source[name], method="nearest")
        from gpuwm.ingest.host_arrays import deepest_level
        surface = deepest_level(out["PRES"], out[name], workers=None)
        if surface is None:
            deepest = np.argmax(out["PRES"], axis=0)[None, ...]
            surface = np.take_along_axis(out[name], deepest, axis=0)[0]
        out[name + "_SFC"] = surface
    np.testing.assert_array_equal(out["QNWFA_SFC"], 100.0)
    np.testing.assert_array_equal(out["QNIFA_SFC"], 200.0)
    assert calls == ["nearest", "nearest"]
    text = open(hrrr.__file__, encoding="utf-8").read()
    assert 'out[name] = mass_plan.apply(source[name], method="nearest")' in text


def test_a_gate_declaring_the_pair_counts_its_records(tmp_path):
    from gpuwm.ingest.hrrr import _read_gate

    lines = {"status": "PASS", "atmosphere_selected_per_time": "661",
             "hybrid_levels": "50", "soil_selected_per_time": "18",
             "window_shape": "1x1", "window_zero_based_inclusive": "i=0..0 j=0..0",
             "qice_mapping": "qice_mapping\tPASS discipline=0 category=1 parameter=82",
             "cross_time_inventory": "PASS",
             "optional_hybrid_fields": "QNWFA,QNIFA",
             "optional_hybrid_units": "QNWFA=kg-1,QNIFA=kg-1"}

    def write(values):
        (tmp_path / "gate.txt").write_text(
            "".join(f"{key}\t{value}\n" for key, value in values.items()))

    import gpuwm.ingest.hrrr as hrrr
    lines["qice_mapping"] = hrrr.HRRR_CLOUD_ICE_GATES[0]
    write(lines)
    assert _read_gate(tmp_path)["optional_hybrid_fields"] == "QNWFA,QNIFA"
    write({**lines, "atmosphere_selected_per_time": "561"})
    with pytest.raises(ValueError, match="expected '661'"):
        _read_gate(tmp_path)

"""Release ABI and recording requirements for the adapted column driver."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def load_builder():
    path=Path(__file__).resolve().parents[1]/"tools/urban_wrf480_oracle/build.py"
    spec=importlib.util.spec_from_file_location("urban480_build",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapted_driver_records_added_words_and_preserves_legacy_columns():
    builder=load_builder()
    original=(builder.LEGACY/"ucm_column_oracle.F90").read_text()
    adapted=builder.driver480(original)
    assert "tvg, tt, xxxvg, tvgl, smg, cmcg, flxhumvg, flxhumt, &" in adapted
    assert "transfer(values,[0],16)" in adapted
    # Existing columns remain available for separate generation baselines.
    for section in ("subroutine write_header()", "subroutine write_inputs()", "subroutine write_outputs()"):
        start=original.index(section)
        stop=original.index("end " + section.split("(")[0],start)
        assert original[start:stop] in adapted


def test_release_and_dependency_pins_are_exact():
    builder=load_builder()
    assert builder.PINS["v4.8.0"]=="06d4240ae989cc3e50af412bb472df3d9048783c"
    assert builder.SUBMODULES["noahmp"][1]=="5da0b241e48ecfd9a2a1bd667ed554765856d589"
    assert len(builder.SUBMODULES)==7

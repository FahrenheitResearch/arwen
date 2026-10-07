"""CPU checks for the namelist metadata used by the SFIRE CUDA routines."""
import pytest

from gpuwm.core.sfire_phys import FuelTable, FUEL_CODES


def test_fuel_models_and_nonburnable_row():
    table=FuelTable()
    assert len(FUEL_CODES)==54
    assert FUEL_CODES[14]==101
    assert FUEL_CODES[-1]==204
    assert table.scalars["nfuelcats"]==54
    assert table.categories["fuel_name"][53].endswith("[SB4 (204)]")
    assert table.categories["fgi"][0]==0.166
    assert table.categories["fgi"][53]==3.1384


def test_namelist_index_sections_keep_unspecified_defaults():
    table=FuelTable.from_namelist_text("""&fuel_categories
      fgi(1)=0.42, fueldepthm(15:17)=3*0.7,
      fgi(54)=0.8 /
      &fuel_moisture drying_lag(2)=12, fmc_live=0.5 /
      &fuel_scalars fuelmc_g=0.2 /""")
    assert table.categories["fgi"][0]==0.42
    assert table.categories["fgi"][1]==0.896
    assert table.categories["fgi"][53]==0.8
    assert table.categories["fueldepthm"][14:17]==[0.7]*3
    assert table.moisture["drying_lag"]==[1.,12.,100.,1000.,1.e9]
    assert table.moisture["fmc_live"]==0.5
    assert table.scalars["fuelmc_g"]==0.2


@pytest.mark.parametrize("text,match",[
    ("&fuel_moisture drying_lag(2)=0 /","positive"),
    ("&fuel_categories fgi_1h(3)=-1 /","nonnegative"),
    ("&fuel_moisture wetting_model(1)=2 /","only model"),
    ("&fuel_categories typo=1 /","unknown"),
    ("&fuel_scalars fuelmc_g=-0.1 /","nonnegative"),
    ("&fuel_scalars nfuelcats=13.5 /","integer"),
])
def test_invalid_inputs_name_the_breakage(text,match):
    with pytest.raises(ValueError,match=match):
        FuelTable.from_namelist_text(text)

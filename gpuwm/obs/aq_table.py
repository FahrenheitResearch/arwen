"""AirNow mirror of rw-obs/src/table.rs. Provisional verification policy.

The optional arwen_global.obs_table named by the original Rust header is
absent from this export. Its integration must add these same vocabulary rows.
"""
VARIABLE_TABLE = {
    "pm25_mass_concentration": ("ug m-3", 0.0, 5000.0, 5.0),
    "pm10_mass_concentration": ("ug m-3", 0.0, 10000.0, 10.0),
    "ozone_mole_fraction": ("ppb", 0.0, 1000.0, 5.0),
    "no2_mole_fraction": ("ppb", 0.0, 2000.0, 5.0),
    "co_mole_fraction": ("ppm", 0.0, 100.0, 0.2),
    "so2_mole_fraction": ("ppb", 0.0, 2000.0, 5.0),
}
MEASUREMENT_TABLE = {"airnow_hourly_concentration": "Hourly ambient concentration"}

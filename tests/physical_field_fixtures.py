"""Explicit physical meanings for the analytical ensemble fixtures only."""
from gpuwm.ensemble.physical_store import FIELD_SCHEMA, canonical_grid_sha256


def analytic_field_contract(grid):
    def row(units, dimensions, basis="scalar"):
        return {"units": units, "dimensions": dimensions, "basis": basis,
                "source_fields": ["analytical test fixture"],
                "operation": "explicit analytical fixture values; no source acquisition"}
    arrays = {"levels_hpa": row("hPa", ["level"])}
    for name, units in {"PRES": "Pa", "TT": "K", "SPFH": "kg kg-1", "GHT": "m", "RH": "%"}.items():
        arrays["field__"+name] = row(units, ["level", "y", "x"])
    for name, units in {"PSFC": "Pa", "T2": "K", "Q2": "kg kg-1", "SOURCE_OROGRAPHY": "m",
                        "ST000010": "K", "RH2": "%"}.items():
        arrays["field__"+name] = row(units, ["y", "x"])
    for name, dims, basis in (
            ("UU", ["level", "y", "x_stag"], "grid_x"),
            ("VV", ["level", "y_stag", "x"], "grid_y"),
            ("U10", ["y", "x_stag"], "grid_x"), ("V10", ["y_stag", "x"], "grid_y")):
        arrays["field__"+name] = row("m s-1", dims, basis)
    arrays["meta__soil_no_source_land"] = row("1", ["y", "x"])
    return {"schema": FIELD_SCHEMA, "grid_sha256": canonical_grid_sha256(grid),
            "vertical": {"kind": "pressure_levels", "units": "hPa", "values": "levels_hpa",
                         "pressure_field": "field__PRES"},
            "arrays": arrays, "evidence": {"analytical_fixture_definition": "a"*64}}

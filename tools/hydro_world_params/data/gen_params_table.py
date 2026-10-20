"""Writes data/params.v1.json (schema arwen.hydro-world.params.v1).

Orchestration only: the numbers below are copied from the cited sources (the
STAS rows from ngen be11a180 data/gauge_01073000/NOAH/parameters/SOILPARM.TBL,
the van Genuchten rows from LGAR-C 9166bcfa data/vG_default_params.dat, both
Apache-2.0) and the cited publications. Run: python3 -I gen_params_table.py OUT.json
"""
import json
import sys

STAS_SRC = ("Noah/Noah-MP SOILPARM.TBL section STAS (the NWM soil class table), as shipped in NOAA-OWP/ngen "
            "be11a180565a78ff069d855b0b5dbfc23845e7d8 data/gauge_01073000/NOAH/parameters/SOILPARM.TBL (Apache-2.0); "
            "class values after Cosby et al. (1984), Water Resour. Res. 20(6):682-690")
# id, name, BB, MAXSMC, REFSMC, SATPSI, SATDK, WLTSMC  (columns of the STAS section)
STAS = [
    (1, "SAND", 2.79, 0.339, 0.192, 0.069, 4.66e-5, 0.010),
    (2, "LOAMY SAND", 4.26, 0.421, 0.283, 0.036, 1.41e-5, 0.028),
    (3, "SANDY LOAM", 4.74, 0.434, 0.312, 0.141, 5.23e-6, 0.047),
    (4, "SILT LOAM", 5.33, 0.476, 0.360, 0.759, 2.81e-6, 0.084),
    (5, "SILT", 3.86, 0.484, 0.347, 0.955, 2.18e-6, 0.061),
    (6, "LOAM", 5.25, 0.439, 0.329, 0.355, 3.38e-6, 0.066),
    (7, "SANDY CLAY LOAM", 6.77, 0.404, 0.315, 0.135, 4.45e-6, 0.069),
    (8, "SILTY CLAY LOAM", 8.72, 0.464, 0.387, 0.617, 2.03e-6, 0.120),
    (9, "CLAY LOAM", 8.17, 0.465, 0.382, 0.263, 2.45e-6, 0.103),
    (10, "SANDY CLAY", 10.73, 0.406, 0.338, 0.098, 7.22e-6, 0.100),
    (11, "SILTY CLAY", 10.39, 0.468, 0.404, 0.324, 1.34e-6, 0.126),
    (12, "CLAY", 11.55, 0.468, 0.412, 0.468, 9.74e-7, 0.138),
    (13, "ORGANIC MATERIAL", 5.25, 0.439, 0.329, 0.355, 3.38e-6, 0.066),
    (14, "WATER", 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    (15, "BEDROCK", 2.79, 0.20, 0.17, 0.069, 1.41e-4, 0.006),
    (16, "OTHER(land-ice)", 4.26, 0.421, 0.283, 0.036, 1.41e-5, 0.028),
    (17, "PLAYA", 11.55, 0.468, 0.454, 0.468, 9.74e-7, 0.030),
    (18, "LAVA", 2.79, 0.200, 0.17, 0.069, 1.41e-4, 0.006),
    (19, "WHITE SAND", 2.79, 0.339, 0.192, 0.069, 4.66e-5, 0.01),
]
USDA_TO_STAS = {"sand": 1, "loamy sand": 2, "sandy loam": 3, "silt loam": 4, "silt": 5, "loam": 6,
                "sandy clay loam": 7, "silty clay loam": 8, "clay loam": 9, "sandy clay": 10,
                "silty clay": 11, "clay": 12}

LGAR_SRC = ("NOAA-OWP/LGAR-C 9166bcfa1d8c37af26a4f6329872400f07a33ff6 data/vG_default_params.dat (Apache-2.0): "
            "the LGAR default van Genuchten parameters per USDA texture class")
# texture, theta_r, theta_e, alpha (1/cm), n, Ks (cm/h), rows 1..12 in file order
VG = [
    ("clay", 0.1, 0.46, 1.00e-02, 1.25, 0.612),
    ("clay loam", 0.08, 0.44, 2.00e-02, 1.42, 0.3348),
    ("loam", 0.06, 0.4, 1.00e-02, 1.47, 0.504),
    ("loamy sand", 0.05, 0.39, 3.00e-02, 1.75, 4.32),
    ("sand", 0.05, 0.38, 4.00e-02, 3.18, 26.64),
    ("sandy clay", 0.12, 0.39, 3.00e-02, 1.21, 0.468),
    ("sandy clay loam", 0.06, 0.38, 2.00e-02, 1.33, 0.54),
    ("sandy loam", 0.04, 0.39, 3.00e-02, 1.45, 1.584),
    ("silt", 0.05, 0.49, 1.00e-02, 1.68, 1.836),
    ("silty clay", 0.11, 0.48, 2.00e-02, 1.32, 0.432),
    ("silty clay loam", 0.09, 0.48, 1.00e-02, 1.52, 0.468),
    ("silt loam", 0.07, 0.44, 1.00e-02, 1.66, 0.756),
]

NEH = "USDA-NRCS (2010) National Engineering Handbook Part 630 Hydrology, ch. 15, Table 15-3 (V = k S^0.5, V ft/s, S ft/ft)"
# WorldCover v200 classes (Zanaga et al. 2022, product user manual v2.0 legend)
WC = [
    (10, "Tree cover", {"USGS": 15, "MODIFIED_IGBP_MODIS_NOAH": 5}, {"USGS": 13, "MODIFIED_IGBP_MODIS_NOAH": 2}, 0.0, 2.5, "Forest with heavy ground litter; hay meadow", False, False),
    (20, "Shrubland", {"USGS": 8, "MODIFIED_IGBP_MODIS_NOAH": 7}, {}, 0.0, 5.0, "Trash fallow or minimum tillage cultivation; contour or strip cropped; woodland", False, False),
    (30, "Grassland", {"USGS": 7, "MODIFIED_IGBP_MODIS_NOAH": 10}, {}, 0.0, 7.0, "Short grass pasture", False, False),
    (40, "Cropland", {"USGS": 2, "MODIFIED_IGBP_MODIS_NOAH": 12}, {}, 0.0, 9.0, "Cultivated straight row", False, False),
    (50, "Built-up", {"USGS": 1, "MODIFIED_IGBP_MODIS_NOAH": 13}, {}, 1.0, 20.3282, "Paved area", False, False),
    (60, "Bare / sparse vegetation", {"USGS": 19, "MODIFIED_IGBP_MODIS_NOAH": 16}, {}, 0.0, 10.0, "Nearly bare and untilled; alluvial fans in western mountain regions", False, False),
    (70, "Snow and ice", {"USGS": 24, "MODIFIED_IGBP_MODIS_NOAH": 15}, {}, 0.0, 10.0, "Nearly bare and untilled; alluvial fans in western mountain regions", False, True),
    (80, "Permanent water bodies", {"USGS": 16, "MODIFIED_IGBP_MODIS_NOAH": 17}, {}, 0.0, 15.0, "Grassed waterway", True, False),
    (90, "Herbaceous wetland", {"USGS": 17, "MODIFIED_IGBP_MODIS_NOAH": 11}, {}, 0.0, 2.5, "Forest with heavy ground litter; hay meadow", False, False),
    (95, "Mangroves", {"USGS": 18, "MODIFIED_IGBP_MODIS_NOAH": 2}, {}, 0.0, 2.5, "Forest with heavy ground litter; hay meadow", False, False),
    (100, "Moss and lichen", {"USGS": 20, "MODIFIED_IGBP_MODIS_NOAH": 19}, {}, 0.0, 7.0, "Short grass pasture", False, False),
]

NGEN_CFE = ("NOAA-OWP/ngen be11a180565a78ff069d855b0b5dbfc23845e7d8 data/gauge_01073000/CFE/cat-*_bmi_config_cfe_pass.txt "
            "(Apache-2.0), the NextGen example CFE packets the CONUS lanes replay")


def main(out):
    t = {
        "schema": "arwen.hydro-world.params.v1",
        "dem_sources": {
            "copernicus-glo90": {
                "dataset": "Copernicus DEM GLO-90 (3 arc-second COG tiles)",
                "url_pattern": "https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_{ns}{lat2}_00_{ew}{lon3}_00_DEM/Copernicus_DSM_COG_30_{ns}{lat2}_00_{ew}{lon3}_00_DEM.tif",
                "tile_pattern": "Copernicus_DSM_COG_30_{ns}{lat2}_00_{ew}{lon3}_00_DEM.tif",
                "vertical_datum": "EGM2008",
                "licence": {
                    "name": "Copernicus WorldDEM-90 licence (free, worldwide, with attribution)",
                    "attribution": "produced using Copernicus WorldDEM-90 (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-90",
                    "verdict_ref": "docs/public/HYDRO-WORLD-LICENCES.md (lane hydro-world-licences 5c1bbd89eb4): ALLOWED with attribution",
                },
            }
        },
        "soil_sources": {
            "soilgrids-2.0-wcs-4326": {
                "dataset": "SoilGrids 2.0 mean predictions (Poggio et al. 2021, SOIL 7:217-240), WCS 2.0.1 GetCoverage in EPSG:4326 at 0.0025 deg",
                "url_pattern": "https://maps.isric.org/mapserv?map=/map/{prop}.map&SERVICE=WCS&VERSION=2.0.1&REQUEST=GetCoverage&COVERAGEID={prop}_{depth}_mean&FORMAT=GEOTIFF_INT16&SUBSET=long(W,E)&SUBSET=lat(S,N)&SUBSETTINGCRS=http://www.opengis.net/def/crs/EPSG/0/4326&OUTPUTCRS=http://www.opengis.net/def/crs/EPSG/0/4326&SCALESIZE=long(NX),lat(NY)",
                "file_pattern": "{prefix}_{prop}_{depth}_mean.tif",
                "depth_labels": ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"],
                "depths_cm": [[0, 5], [5, 15], [15, 30], [30, 60], [60, 100], [100, 200]],
                "props": {
                    "sand": {"scale": 0.1, "unit": "% (stored g/kg)"},
                    "silt": {"scale": 0.1, "unit": "% (stored g/kg)"},
                    "clay": {"scale": 0.1, "unit": "% (stored g/kg)"},
                    "soc": {"scale": 0.1, "unit": "g/kg (stored dg/kg)"},
                    "bdod": {"scale": 0.01, "unit": "g/cm3 (stored cg/cm3)"},
                    "cfvo": {"scale": 0.1, "unit": "% volume (stored cm3/dm3)"},
                },
                "nodata": -32768,
                "sand": "sand", "silt": "silt", "clay": "clay", "organic_carbon": "soc",
                "licence": {
                    "name": "CC BY 4.0",
                    "attribution": "Contains SoilGrids 2.0 data, ISRIC World Soil Information, CC BY 4.0 (Poggio et al. 2021, SOIL 7, 217-240)",
                    "verdict_ref": "docs/public/HYDRO-WORLD-LICENCES.md (lane hydro-world-licences 5c1bbd89eb4): ALLOWED with attribution",
                },
            }
        },
        "land_cover_sources": {
            "esa-worldcover-v200-2021": {
                "dataset": "ESA WorldCover 10 m 2021 v200 (Zanaga et al. 2022, doi:10.5281/zenodo.7254221), 3x3 degree COG tiles",
                "url_pattern": "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{ns}{lat2}{ew}{lon3}_Map.tif",
                "tile_pattern": "ESA_WorldCover_10m_2021_v200_{ns}{lat2}{ew}{lon3}_Map.tif",
                "tile_degrees": 3,
                "classes": [
                    {"code": c, "name": n, "veg": v, "veg_tropical": vt, "impervious": imp, "neh_k_ft_s": k, "neh_row": row, "water": w, "ice": ice}
                    for (c, n, v, vt, imp, k, row, w, ice) in WC
                ],
                "licence": {
                    "name": "CC BY 4.0",
                    "attribution": "(c) ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium",
                    "verdict_ref": "docs/public/HYDRO-WORLD-LICENCES.md (lane hydro-world-licences 5c1bbd89eb4): ALLOWED with attribution",
                },
            }
        },
        "soil_recipes": {
            "nwm-stas-lookup": {
                "method": "class_lookup",
                "soil_class_table": "STAS",
                "source": "NWM / NextGen recipe: soil texture class -> STAS SOILPARM.TBL row; " + STAS_SRC,
                "texture_depth_cm": [0, 30],
                "organic_carbon_min_g_kg": 120.0,
                "organic_class": 13, "water_class": 14, "ice_class": 16, "gap_fill_cells": 40,
                "usda_to_class": USDA_TO_STAS,
                "classes": [
                    {"id": i, "name": n, "bb": bb, "maxsmc": mx, "refsmc": rf, "satpsi": ps, "satdk": dk, "wltsmc": wl, "hydraulic": i not in (14, 16)}
                    for (i, n, bb, mx, rf, ps, dk, wl) in STAS
                ],
                "aggregation": {
                    "soil_class": "area mode over all classes, water and ice included (Noah-OWP isltyp)",
                    "soil_b": "b of the area-modal hydraulic class (NextGen hydrofabric divide attribute mode.bexp)",
                    "soil_satdk_m_s": "area-weighted geometric mean over hydraulic classes (geom_mean.dksat)",
                    "soil_satpsi_m": "area-weighted geometric mean over hydraulic classes (geom_mean.psisat)",
                    "soil_smcmax": "area-weighted arithmetic mean over hydraulic classes (mean.smcmax)",
                    "soil_wltsmc": "area-weighted arithmetic mean over hydraulic classes (mean.smcwlt)",
                },
                "cosby": None,
            },
            "cosby1984-continuous": {
                "method": "cosby1984",
                "soil_class_table": "STAS",
                "source": "Cosby, Hornberger, Clapp and Ginn (1984), Water Resour. Res. 20(6):682-690, Table 4 multivariate regressions on sand/silt/clay %; wilting point from the Campbell (1974, Soil Sci. 117:311-314) retention curve at 150 m (1.5 MPa)",
                "texture_depth_cm": [0, 30],
                "organic_carbon_min_g_kg": 120.0,
                "organic_class": 13, "water_class": 14, "ice_class": 16, "gap_fill_cells": 40,
                "usda_to_class": USDA_TO_STAS,
                "classes": [
                    {"id": i, "name": n, "bb": bb, "maxsmc": mx, "refsmc": rf, "satpsi": ps, "satdk": dk, "wltsmc": wl, "hydraulic": i not in (14, 16)}
                    for (i, n, bb, mx, rf, ps, dk, wl) in STAS
                ],
                "aggregation": {
                    "soil_b": "area-weighted arithmetic mean of the per-cell regression value",
                    "soil_satdk_m_s": "area-weighted geometric mean", "soil_satpsi_m": "area-weighted geometric mean",
                    "soil_smcmax": "area-weighted arithmetic mean", "soil_wltsmc": "area-weighted arithmetic mean",
                },
                "cosby": {"b0": 3.10, "b_clay": 0.157, "b_sand": -0.003, "psi0": 1.54, "psi_sand": -0.0095, "psi_silt": 0.0063,
                          "ks0": -0.60, "ks_sand": 0.0126, "ks_clay": -0.0064, "ts0": 50.5, "ts_sand": -0.142, "ts_clay": -0.037,
                          "psi_wilt_m": 150.0},
            },
        },
        "lgar_recipes": {
            "lgar-c-default-vg": {"source": LGAR_SRC, "rows": [
                {"usda": u, "theta_r": tr, "theta_e": te, "alpha_per_cm": a, "n": n, "ks_cm_h": ks} for (u, tr, te, a, n, ks) in VG]}
        },
        "constants": {
            "soil_depth_m": {"value": 2.0, "unit": "m", "source": "CFE soil_params.depth in every NextGen example packet (" + NGEN_CFE + "); the 2 m Noah-MP four-layer column the NWM runs"},
            "refkdt": {"value": 3.0, "unit": "-", "source": "Schaake et al. (1996), J. Geophys. Res. 101(D3):7461-7475 surface-runoff parameter; the uncalibrated Noah/NWM default REFKDT_DATA = 3.0 (GENPARM.TBL in ngen be11a180 data/gauge_01073000/NOAH/parameters); NWM values are calibrated per basin"},
            "alpha_fc": {"value": 0.33, "unit": "-", "source": "CFE field-capacity fraction alpha_fc in every NextGen example packet (" + NGEN_CFE + ")"},
            "k_lf": {"value": 0.01, "unit": "-", "source": "CFE lateral-flow coefficient K_lf in every NextGen example packet (" + NGEN_CFE + ")"},
            "k_nash": {"value": 0.03, "unit": "-", "source": "CFE Nash-cascade coefficient K_nash in every NextGen example packet (" + NGEN_CFE + "); two Nash reservoirs (nash_storage=0.0,0.0)"},
            "max_gw_storage_m": {"value": 0.012863189697, "unit": "m", "source": "PLACEHOLDER replaced by gen step 2"},
            "gw_coeff_m3_s": {"value": 0.005, "unit": "m3 s-1", "source": "PLACEHOLDER replaced by gen step 2"},
            "gw_expon": {"value": 7.0, "unit": "-", "source": "PLACEHOLDER replaced by gen step 2"},
            "soil_color": {"value": 4, "unit": "class", "source": "Noah-OWP soilcolor in every NextGen example namelist (ngen be11a180 data/gauge_01073000/NOAH/cat-*.input soilcolor = 4)"},
        },
        "slop": {
            "method": "constant", "coarse_m": 1000.0, "min": 0.1, "max": 0.1,
            "source": "Noah/NWM uncalibrated drainage slope coefficient SLOPE_DATA category 1 = 0.1 (GENPARM.TBL in ngen be11a180 data/gauge_01073000/NOAH/parameters). CFE soil_params.slop is this drainage coefficient: the NWM v3.1.6 soilproperties 'slope' field it is taken from carries calibrated values clustered around 0.1 (0.057, 0.1, 0.24 over the Russian River), not terrain slope. Alternative method terrain_tan_coarse (mean tan(slope) of the DEM at coarse_m) stays selectable; it matched NWM worse (median ratio 0.69, 51 % within 2x)"},
        "giuh": {
            "dt_s": 3600.0, "slope_floor": 0.005, "channel_init_area_km2": 0.5,
            "channel_neh_k_ft_s": 16.1345, "channel_neh_row": "Unpaved area; small upland gullies", "max_ordinates": 48,
            "source": "time-area GIUH (Clark 1945; Maidment et al. 1996, Hydrol. Process. 10:831-844) on Priority-Flood paths (Barnes et al. 2014) to the catchment outlet; V = k S^0.5 with k from " + NEH + "; slope floored at 0.005 m/m; channel initiation 0.5 km2, inside the 0.01-0.5 km2 range of field channel heads (Montgomery and Dietrich 1988, Nature 336:232-234); ordinates per CFE step of 3600 s"},
        "cfe_recipes": {
            "example-packet-priors": {
                "method": "constants",
                "source": "K_lf 0.01, K_nash 0.03, alpha_fc 0.33 from constants: the NextGen example CFE packets (" + NGEN_CFE + "); the priors the 2026-10-18 Rhine proof ran with (peaks about 4 days late)"},
            "hillslope-kinematic-v1": {
                "method": "hillslope_kinematic",
                "source": ("Regionalised CFE subsurface parameters from catchment attributes. K_lf (m/h) = 3600 * 2 F Ks sin(b) D / L: "
                           "Sloan and Moore (1984), Modeling subsurface stormflow on steeply sloping forested watersheds, Water Resour. Res. "
                           "20(12):1815-1822, kinematic storage model (linear drainage constant 2 Ks sin(b) / (theta_d L)) applied to the CFE "
                           "lateral outlet K_lf (S - S_fc)/(S_max - S_fc), S_max - S_fc = D theta_d. K_nash (1/h) = N 3600 F Ks sin(b) / (theta_d L): "
                           "Beven (1981), Kinematic subsurface stormflow, Water Resour. Res. 17(5):1419-1424 (celerity F Ks sin(b)/theta_d), "
                           "hillslope travel time L/c shared over a Nash (1957, IASH Publ. 45(3):114-121) cascade of N reservoirs. L = mean "
                           "Priority-Flood flow distance from hillslope cells to the first channel cell on Copernicus GLO-90 (Horton 1945 "
                           "overland-flow length, Geol. Soc. Am. Bull. 56(3):275-370, measured on the paths), sin(b) = mean hillslope Horn slope; "
                           "Ks = catchment satdk (SoilGrids texture -> STAS); theta_d = smcmax - theta_fc with theta_fc on the Clapp and "
                           "Hornberger (1978, Water Resour. Res. 14(4):601-604) curve at the field-capacity head alpha_fc * 10.33 m. "
                           "This is the starting point for an uncalibrated basin anywhere; gauged basins are calibrated by the gauge-calibration "
                           "door (tools/hydro_world_calibration) and ungauged ones take the Gower (1971) nearest calibrated donor."),
                "lateral_ksat_factor": 1000.0,
                "lateral_ksat_factor_source": ("F, the lateral to vertical saturated conductivity ratio of the hillslope (macropores, "
                                               "layering): LKSATFAC = 1000, the default lateral saturated conductivity multiplier of the "
                                               "NCAR WRF-Hydro subsurface routing (Gochis et al. 2020, The WRF-Hydro Modeling System Technical "
                                               "Description, Version 5.1.1, NCAR Technical Note, cited as a publication; no UCAR code used)"),
                "n_nash": 2.0,
                "atmosphere_head_m": 10.33,
                "alpha_fc_by_soil_class": {"1": 0.1, "2": 0.1},
                "alpha_fc_default": 0.33,
                "alpha_fc_source": ("Field capacity as the water held at 1/3 bar (Richards and Weaver 1944, Soil Sci. 58:207-218; the CFE "
                                    "alpha_fc = 0.33 convention), and at 1/10 bar for coarse-textured soils (STAS 1 SAND, 2 LOAMY SAND), "
                                    "as USDA-NRCS National Soil Survey Handbook Part 618 takes the upper end of available water capacity"),
                "min_hillslope_length_m": 45.0,
                "min_sin_slope": 0.005,
                "min_drainable_porosity": 0.02,
                "k_lf_range": [0.0001, 0.5],
                "k_nash_range": [0.005, 0.9]},
        },
        "cfe_recipe_default": "hillslope-kinematic-v1",
        "bounds": {
            "elevation_mean_m": [-450.0, 8850.0], "slope_mean_deg": [0.0, 90.0], "aspect_mean_deg": [0.0, 360.0],
            "centroid_lat_deg": [-90.0, 90.0], "centroid_lon_deg": [-180.0, 180.0],
            "soil_class": [1, 19], "veg_class": [1, 24], "soil_color": [1, 20], "impervious_fraction": [0.0, 1.0],
            "soil_depth_m": [0.1, 10.0], "soil_b": [1.0, 15.0], "soil_satdk_m_s": [1e-8, 1e-3], "soil_satpsi_m": [0.01, 2.0],
            "soil_slop": [0.0, 1.0], "soil_smcmax": [0.15, 0.7], "soil_wltsmc": [0.0, 0.35], "refkdt": [0.1, 10.0],
            "alpha_fc": [0.0, 1.0], "max_gw_storage_m": [0.001, 10.0], "cgw_m_h": [1e-9, 10.0], "gw_expon": [1.0, 10.0],
            "k_lf": [0.0, 1.0], "k_nash": [0.0, 1.0],
        },
    }
    if len(sys.argv) > 2:
        gw = json.load(open(sys.argv[2]))
        for k in ("max_gw_storage_m", "gw_coeff_m3_s", "gw_expon"):
            t["constants"][k] = gw[k]
    json.dump(t, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])

/// Product definitions for weather plotting.
///
/// Two product systems coexist:
/// 1. `GribProduct` - Simple GRIB2 variable definitions for operational model downloads
/// 2. `Product` - Full plot product definitions with colormaps, contours, overlays
///    (ported from wrf-solar's Solarpower07 framework)

// ============================================================
// GRIB2 Product Definitions (for operational model downloads)
// ============================================================

/// A GRIB2 product definition for downloading and basic plotting
#[derive(Debug, Clone)]
pub struct GribProduct {
    /// Display name
    pub name: &'static str,
    /// GRIB2 variable patterns to download (VAR:level format)
    pub grib_vars: &'static [&'static str],
    /// Colormap name
    pub colormap: &'static str,
    /// Value range (min, max) in display units
    pub range: (f64, f64),
    /// Display units
    pub units: &'static str,
    /// Description
    pub description: &'static str,
}

/// All available GRIB2 products
pub static GRIB_PRODUCTS: &[GribProduct] = &[
    GribProduct {
        name: "Surface Temp",
        grib_vars: &["TMP:2 m above ground"],
        colormap: "temperature",
        range: (230.0, 320.0),
        units: "K",
        description: "2-meter temperature",
    },
    GribProduct {
        name: "Surface Dewpoint",
        grib_vars: &["DPT:2 m above ground"],
        colormap: "temperature",
        range: (230.0, 310.0),
        units: "K",
        description: "2-meter dewpoint temperature",
    },
    GribProduct {
        name: "CAPE",
        grib_vars: &["CAPE:surface"],
        colormap: "cape",
        range: (0.0, 5000.0),
        units: "J/kg",
        description: "Surface-based Convective Available Potential Energy",
    },
    GribProduct {
        name: "Composite Reflectivity",
        grib_vars: &["REFC:entire atmosphere"],
        colormap: "reflectivity",
        range: (-10.0, 75.0),
        units: "dBZ",
        description: "Composite radar reflectivity",
    },
    GribProduct {
        name: "10m Wind",
        grib_vars: &["UGRD:10 m above ground", "VGRD:10 m above ground"],
        colormap: "wind",
        range: (0.0, 40.0),
        units: "m/s",
        description: "10-meter wind speed (computed from U/V components)",
    },
    GribProduct {
        name: "Wind Gust",
        grib_vars: &["GUST:surface"],
        colormap: "wind",
        range: (0.0, 50.0),
        units: "m/s",
        description: "Surface wind gust",
    },
    GribProduct {
        name: "MSLP",
        grib_vars: &["MSLMA:mean sea level"],
        colormap: "temperature",
        range: (980.0, 1040.0),
        units: "hPa",
        description: "Mean sea level pressure",
    },
    GribProduct {
        name: "Visibility",
        grib_vars: &["VIS:surface"],
        colormap: "jet",
        range: (0.0, 16000.0),
        units: "m",
        description: "Surface visibility",
    },
    GribProduct {
        name: "500mb Height",
        grib_vars: &["HGT:500 mb"],
        colormap: "jet",
        range: (4800.0, 6000.0),
        units: "m",
        description: "500mb geopotential height",
    },
    GribProduct {
        name: "500mb Wind",
        grib_vars: &["UGRD:500 mb", "VGRD:500 mb"],
        colormap: "wind",
        range: (0.0, 80.0),
        units: "m/s",
        description: "500mb wind speed",
    },
    GribProduct {
        name: "250mb Wind",
        grib_vars: &["UGRD:250 mb", "VGRD:250 mb"],
        colormap: "wind",
        range: (0.0, 100.0),
        units: "m/s",
        description: "250mb jet stream wind speed",
    },
    GribProduct {
        name: "Total Cloud Cover",
        grib_vars: &["TCDC:entire atmosphere"],
        colormap: "jet",
        range: (0.0, 100.0),
        units: "%",
        description: "Total cloud cover",
    },
    GribProduct {
        name: "PBL Height",
        grib_vars: &["HPBL:surface"],
        colormap: "jet",
        range: (0.0, 4000.0),
        units: "m",
        description: "Planetary boundary layer height",
    },
    GribProduct {
        name: "Precipitable Water",
        grib_vars: &["PWAT:entire atmosphere"],
        colormap: "dewpoint_f",
        range: (0.0, 70.0),
        units: "mm",
        description: "Total precipitable water",
    },
    GribProduct {
        name: "2m RH",
        grib_vars: &["RH:2 m above ground"],
        colormap: "rh",
        range: (0.0, 100.0),
        units: "%",
        description: "2-meter relative humidity",
    },
    GribProduct {
        name: "Total Precip",
        grib_vars: &["APCP:surface"],
        colormap: "precip_in",
        range: (0.0, 2.0),
        units: "in",
        description: "Total accumulated precipitation",
    },
    GribProduct {
        name: "Max Updraft Helicity",
        grib_vars: &["MXUPHL:5000-2000 m above ground"],
        colormap: "uh",
        range: (0.0, 200.0),
        units: "m^2/s^2",
        description: "Maximum updraft helicity (2-5 km)",
    },
    GribProduct {
        name: "850mb Temp",
        grib_vars: &["TMP:850 mb"],
        colormap: "temperature",
        range: (230.0, 310.0),
        units: "K",
        description: "850mb temperature",
    },
    GribProduct {
        name: "700mb RH",
        grib_vars: &["RH:700 mb"],
        colormap: "rh",
        range: (0.0, 100.0),
        units: "%",
        description: "700mb relative humidity",
    },
    GribProduct {
        name: "Surface Pressure",
        grib_vars: &["PRES:surface"],
        colormap: "temperature",
        range: (50000.0, 105000.0),
        units: "Pa",
        description: "Surface pressure",
    },
    GribProduct {
        name: "Lifted Index",
        grib_vars: &["LFTX:500-1000 mb"],
        colormap: "cape",
        range: (-10.0, 10.0),
        units: "K",
        description: "Surface lifted index",
    },
    GribProduct {
        name: "CIN",
        grib_vars: &["CIN:surface"],
        colormap: "cape",
        range: (-300.0, 0.0),
        units: "J/kg",
        description: "Convective inhibition",
    },
    GribProduct {
        name: "Storm Motion",
        grib_vars: &["USTM:0-6000 m above ground", "VSTM:0-6000 m above ground"],
        colormap: "wind",
        range: (0.0, 40.0),
        units: "m/s",
        description: "Storm motion (Bunkers)",
    },
];

/// Find a GRIB product by name (case-insensitive partial match)
pub fn find_grib_product(name: &str) -> Option<&'static GribProduct> {
    let name_lower = name.to_lowercase();
    GRIB_PRODUCTS.iter().find(|p| p.name.to_lowercase() == name_lower)
        .or_else(|| GRIB_PRODUCTS.iter().find(|p| p.name.to_lowercase().contains(&name_lower)))
}

/// List all available GRIB product names
pub fn list_grib_products() {
    println!("{:<25} {:<12} {:<8} {}", "Product", "Colormap", "Units", "Description");
    println!("{}", "-".repeat(80));
    for p in GRIB_PRODUCTS {
        println!("{:<25} {:<12} {:<8} {}", p.name, p.colormap, p.units, p.description);
    }
}

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivedRecipeInventoryEntry {
    pub slug: &'static str,
    pub title: &'static str,
    pub experimental: bool,
    pub heavy: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockedDerivedRecipeInventoryEntry {
    pub slug: &'static str,
    pub title: &'static str,
    pub reason: &'static str,
}

const SUPPORTED_DERIVED_RECIPE_INVENTORY: &[DerivedRecipeInventoryEntry] = &[
    DerivedRecipeInventoryEntry {
        slug: "sbcape",
        title: "SBCAPE",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "sbcin",
        title: "SBCIN",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "sblcl",
        title: "SBLCL",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "mlcape",
        title: "MLCAPE",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "mlcin",
        title: "MLCIN",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "mucape",
        title: "MUCAPE",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "mucin",
        title: "MUCIN",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "dcape",
        title: "DCAPE",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "sbecape",
        title: "SBECAPE",
        experimental: false,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "mlecape",
        title: "MLECAPE",
        experimental: false,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "muecape",
        title: "MUECAPE",
        experimental: false,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "sb_ecape_derived_cape_ratio",
        title: "SB ECAPE / Derived CAPE Ratio (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "ml_ecape_derived_cape_ratio",
        title: "ML ECAPE / Derived CAPE Ratio (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "mu_ecape_derived_cape_ratio",
        title: "MU ECAPE / Derived CAPE Ratio (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "sb_ecape_native_cape_ratio",
        title: "SB ECAPE / Native CAPE Ratio (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "ml_ecape_native_cape_ratio",
        title: "ML ECAPE / Native CAPE Ratio (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "mu_ecape_native_cape_ratio",
        title: "MU ECAPE / Native CAPE Ratio (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "sbncape",
        title: "SBNCAPE",
        experimental: false,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "sbecin",
        title: "SBECIN",
        experimental: false,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "mlecin",
        title: "MLECIN",
        experimental: false,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "ecape_scp",
        title: "ECAPE SCP (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "ecape_ehi_0_1km",
        title: "ECAPE EHI 0-1 km (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "ecape_ehi_0_3km",
        title: "ECAPE EHI 0-3 km (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "ecape_stp",
        title: "ECAPE STP (EXP)",
        experimental: true,
        heavy: true,
    },
    DerivedRecipeInventoryEntry {
        slug: "theta_e_2m_10m_winds",
        title: "2 m Theta-e, 10 m Wind Barbs",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "vpd_2m",
        title: "2 m Vapor Pressure Deficit",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "dewpoint_depression_2m",
        title: "2 m Dewpoint Depression",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "wetbulb_2m",
        title: "2 m Wet-Bulb Temperature",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "fire_weather_composite",
        title: "Fire Weather Composite",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "apparent_temperature_2m",
        title: "2 m Apparent Temperature",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "heat_index_2m",
        title: "2 m Heat Index",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "wind_chill_2m",
        title: "2 m Wind Chill",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "lifted_index",
        title: "Surface-Based Lifted Index",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "lapse_rate_700_500",
        title: "700-500 mb Virtual Temperature Lapse Rate",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "lapse_rate_0_3km",
        title: "0-3 km Lapse Rate",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "bulk_shear_0_1km",
        title: "0-1 km Bulk Shear",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "bulk_shear_0_6km",
        title: "0-6 km Bulk Shear",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "srh_0_1km",
        title: "0-1 km SRH",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "srh_0_3km",
        title: "0-3 km SRH",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "ehi_0_1km",
        title: "EHI 0-1 km",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "ehi_0_3km",
        title: "EHI 0-3 km",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "stp_fixed",
        title: "STP (FIXED)",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "scp_mu_0_3km_0_6km_proxy",
        title: "SCP (MU / 0-3 km / 0-6 km PROXY)",
        experimental: true,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "temperature_advection_700mb",
        title: "700 mb Temperature Advection",
        experimental: false,
        heavy: false,
    },
    DerivedRecipeInventoryEntry {
        slug: "temperature_advection_850mb",
        title: "850 mb Temperature Advection",
        experimental: false,
        heavy: false,
    },
];

const BLOCKED_DERIVED_RECIPE_INVENTORY: &[BlockedDerivedRecipeInventoryEntry] = &[
    BlockedDerivedRecipeInventoryEntry {
        slug: "stp_effective",
        title: "STP (EFFECTIVE)",
        reason: "requires mixed-layer CAPE/CIN/LCL plus effective SRH and effective bulk wind difference; rustwx-products does not yet derive effective SRH or EBWD from HRRR profiles",
    },
    BlockedDerivedRecipeInventoryEntry {
        slug: "scp",
        title: "SCP",
        reason: "requires effective SRH and effective bulk wind difference; rustwx-products does not yet derive those effective-layer kinematics from HRRR profiles",
    },
    BlockedDerivedRecipeInventoryEntry {
        slug: "scp_effective",
        title: "SCP (EFFECTIVE)",
        reason: "requires effective SRH and effective bulk wind difference; rustwx-products does not yet derive those effective-layer kinematics from HRRR profiles",
    },
];

pub fn supported_derived_recipe_inventory() -> &'static [DerivedRecipeInventoryEntry] {
    SUPPORTED_DERIVED_RECIPE_INVENTORY
}

pub fn blocked_derived_recipe_inventory() -> &'static [BlockedDerivedRecipeInventoryEntry] {
    BLOCKED_DERIVED_RECIPE_INVENTORY
}

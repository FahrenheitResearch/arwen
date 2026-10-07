//! Snow-cover fallback (decision D26) when a history has no SNOWC.
//!
//! The Noah land model's snow-cover fraction (J. Geophys. Res. 108 (2003)
//! 8851 [R11]): with r = SWE / SNUP,
//!
//! ```text
//! f = 1 - (exp(-a r) - r exp(-a))   for SWE < SNUP,   f = 1 otherwise
//! ```
//!
//! where SNUP is the vegetation class's snow-depletion threshold (metres of
//! water equivalent, the public WRF `VEGPARM.TBL`) and a is `SALP_DATA` of
//! the public WRF `GENPARM.TBL`.  Water, lake and ice classes get 0 (the
//! specification's rule).  Classes of tables not listed here give missing
//! cover, never zero.

/// GENPARM.TBL SALP_DATA.
pub const NOAH_SALP: f32 = 2.6;

/// SNUP (m) of the MODIFIED_IGBP_MODIS_NOAH classes 1 to 21 of VEGPARM.TBL.
/// 0 marks the water, permanent ice and lake classes.
const MODIS_SNUP: [f32; 21] = [
    0.08,  // 1 evergreen needleleaf forest
    0.08,  // 2 evergreen broadleaf forest
    0.08,  // 3 deciduous needleleaf forest
    0.08,  // 4 deciduous broadleaf forest
    0.08,  // 5 mixed forests
    0.03,  // 6 closed shrublands
    0.035, // 7 open shrublands
    0.03,  // 8 woody savannas
    0.04,  // 9 savannas
    0.04,  // 10 grasslands
    0.015, // 11 permanent wetlands
    0.04,  // 12 croplands
    0.04,  // 13 urban and built-up
    0.04,  // 14 cropland/natural vegetation mosaic
    0.0,   // 15 snow and ice
    0.02,  // 16 barren or sparsely vegetated
    0.0,   // 17 water
    0.02,  // 18 wooded tundra
    0.025, // 19 mixed tundra
    0.025, // 20 barren tundra
    0.0,   // 21 lakes
];

/// The per-cell SNUP plane for a land-use table and the dominant class
/// plane (IVGTYP), or `None` when the table is not known here.
pub fn snup_plane(land_use_table: &str, ivgtyp: &[f32]) -> Option<Vec<f32>> {
    let table: &[f32] = match land_use_table.trim() {
        "MODIFIED_IGBP_MODIS_NOAH" | "MODIS" => &MODIS_SNUP,
        _ => return None,
    };
    Some(
        ivgtyp
            .iter()
            .map(|&c| {
                let k = c as i64;
                if c.is_finite() && k >= 1 && (k as usize) <= table.len() && k as f32 == c {
                    table[k as usize - 1]
                } else {
                    f32::NAN
                }
            })
            .collect(),
    )
}

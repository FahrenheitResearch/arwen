//! The unit boundary between a stored derived plane and its named product.
//!
//! A named derived product draws its computed slot on a colour bar built
//! in that slot's units: bulk shear in knots, CAPE in J/kg, and so on.  A
//! plane read back from a store carries the units its WRITER declared,
//! and the writers do not all agree: the GRIB ingest stores each slot in
//! the product's own units, while a raw-wrfout import stores what the WRF
//! diagnostic returns, which for bulk shear is metres per second.
//!
//! WHAT BREAKAGE THIS PREVENTS: a 3 km forecast whose deep-layer shear
//! peaked at 22.7 m/s was drawn as 22.7 kt, because the stored numbers
//! were placed straight into the knot slot.  Every consumer of a stored
//! derived plane (the named picture and the `var:` picture alike) now
//! asks this module how the stored units reach the colour bar's units.

use super::KNOTS_PER_MS;
use super::query::computed_recipe_units;
use super::recipes::DerivedRecipe;

/// How a stored plane's values become the values its named product draws.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreUnitConversion {
    /// The plane is already in the product's units, however they are
    /// spelled ("m2/s2" and "m^2/s^2" are one unit).
    Same,
    /// Metres per second to knots, with the same factor the compute lane
    /// multiplies its own shear by, so a store-read plane draws exactly
    /// what the compute lane would have drawn.
    MetresPerSecondToKnots,
}

impl StoreUnitConversion {
    /// One stored value in the product's units.
    pub fn apply(self, value: f64) -> f64 {
        match self {
            Self::Same => value,
            Self::MetresPerSecondToKnots => value * KNOTS_PER_MS,
        }
    }
}

/// The units the named product `slug` draws its colour bar in, for the
/// derived recipes a store holds as computed-slot planes.  `None` for a
/// slug that is not such a recipe.
pub fn derived_product_units(slug: &str) -> Option<&'static str> {
    let recipe = DerivedRecipe::parse(slug).ok()?;
    if recipe.is_heavy() {
        return None;
    }
    Some(computed_recipe_units(recipe))
}

/// How a plane stored as `stored_units` under the derived slug `slug`
/// reaches the units of that product's colour bar.
///
/// A slug that is not a computed-slot recipe draws its plane as stored
/// (`Same`).  A recipe plane whose units are neither the product's nor a
/// unit with a known conversion is refused: its numbers would be printed
/// on a colour bar that states a different unit.
pub fn store_plane_conversion(
    slug: &str,
    stored_units: &str,
) -> Result<StoreUnitConversion, String> {
    let Some(product_units) = derived_product_units(slug) else {
        return Ok(StoreUnitConversion::Same);
    };
    let stored = unit_token(stored_units);
    let product = unit_token(product_units);
    if stored == product {
        return Ok(StoreUnitConversion::Same);
    }
    if stored == "m/s" && product == "kt" {
        return Ok(StoreUnitConversion::MetresPerSecondToKnots);
    }
    Err(format!(
        "the stored plane is in {stored_units:?}, the {slug} colour bar is in \
         {product_units:?}, and there is no conversion between the two"
    ))
}

/// One spelling per unit, so two writers that spell a unit differently
/// still agree.  Only spellings of the SAME unit are merged here; a
/// conversion between two units is [`store_plane_conversion`]'s decision.
fn unit_token(units: &str) -> String {
    let compact: String = units
        .trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|ch| !ch.is_whitespace() && !matches!(ch, '^' | '*' | '{' | '}'))
        .collect();
    let same = match compact.as_str() {
        "m/s" | "ms-1" | "m/sec" | "mps" | "meterspersecond" | "metrespersecond" => "m/s",
        "kt" | "kts" | "knot" | "knots" => "kt",
        "j/kg" | "jkg-1" => "j/kg",
        "m2/s2" | "m2s-2" => "m2/s2",
        "m" | "meters" | "metres" => "m",
        "k" | "kelvin" => "k",
        "degc" | "°c" | "celsius" => "degc",
        "hpa" | "mb" | "mbar" => "hpa",
        // A lapse rate and an advection rate are temperature DIFFERENCES
        // per distance or time, where one kelvin is one degree Celsius.
        "degc/km" | "k/km" | "°c/km" | "ckm-1" | "kkm-1" => "degc/km",
        "degc/hr" | "degc/h" | "k/hr" | "k/h" | "°c/hr" | "°c/h" | "ch-1" | "kh-1" => "degc/hr",
        "" | "1" | "-" | "dimensionless" | "unitless" | "nondimensional" | "none" | "index"
        | "ratio" => "1",
        other => return other.to_string(),
    };
    same.to_string()
}

#[cfg(test)]
mod tests {
    use super::super::store::store_derived_recipe_slugs;
    use super::*;

    #[test]
    fn every_store_recipe_names_the_units_its_product_draws() {
        for slug in store_derived_recipe_slugs() {
            let units = derived_product_units(slug)
                .unwrap_or_else(|| panic!("store recipe '{slug}' has no product units"));
            assert_eq!(
                store_plane_conversion(slug, units),
                Ok(StoreUnitConversion::Same),
                "a '{slug}' plane stored in its own product's units draws as stored"
            );
        }
    }

    #[test]
    fn shear_stored_in_metres_per_second_is_drawn_in_knots() {
        for slug in ["bulk_shear_0_1km", "bulk_shear_0_6km"] {
            for units in ["m/s", "m s-1", "m s^-1", "M/S"] {
                let conversion = store_plane_conversion(slug, units).unwrap();
                assert_eq!(
                    conversion,
                    StoreUnitConversion::MetresPerSecondToKnots,
                    "{units}"
                );
                let knots = conversion.apply(22.689_922_332_763_672);
                assert!((knots - 44.105_680_7).abs() < 1.0e-6, "{knots}");
            }
            for units in ["kt", "kts", "knots"] {
                assert_eq!(
                    store_plane_conversion(slug, units),
                    Ok(StoreUnitConversion::Same)
                );
            }
        }
    }

    #[test]
    fn spellings_of_one_unit_draw_as_stored() {
        assert_eq!(
            store_plane_conversion("srh_0_1km", "m2/s2"),
            Ok(StoreUnitConversion::Same)
        );
        assert_eq!(
            store_plane_conversion("srh_0_3km", "m2 s-2"),
            Ok(StoreUnitConversion::Same)
        );
        assert_eq!(
            store_plane_conversion("sbcape", "J kg-1"),
            Ok(StoreUnitConversion::Same)
        );
        assert_eq!(
            store_plane_conversion("stp_fixed", "dimensionless"),
            Ok(StoreUnitConversion::Same)
        );
        assert_eq!(
            store_plane_conversion("lapse_rate_0_3km", "K/km"),
            Ok(StoreUnitConversion::Same)
        );
    }

    #[test]
    fn a_plane_in_a_foreign_unit_is_refused_with_both_units_named() {
        let reason = store_plane_conversion("bulk_shear_0_6km", "K").unwrap_err();
        assert!(reason.contains("\"K\""), "{reason}");
        assert!(reason.contains("\"kt\""), "{reason}");
        // An absolute temperature in kelvin is not silently shifted onto
        // a Celsius bar: whether 273.15 applies depends on the quantity.
        assert!(store_plane_conversion("wetbulb_2m", "K").is_err());
        assert!(store_plane_conversion("sbcape", "m/s").is_err());
    }

    #[test]
    fn a_slug_that_is_not_a_store_recipe_draws_as_stored() {
        assert_eq!(derived_product_units("not_a_recipe"), None);
        assert_eq!(
            store_plane_conversion("not_a_recipe", "widgets"),
            Ok(StoreUnitConversion::Same)
        );
    }

    #[test]
    fn the_knot_factor_is_the_compute_lanes_own() {
        assert_eq!(
            StoreUnitConversion::MetresPerSecondToKnots.apply(1.0),
            KNOTS_PER_MS
        );
    }
}

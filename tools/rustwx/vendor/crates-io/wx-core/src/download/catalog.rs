//! Pre-defined variable groups for common weather analysis use cases.
//!
//! Each group bundles related .idx search patterns so users can request
//! `"severe"` or `"winter"` instead of listing individual variables.
//!
//! # Usage
//!
//! ```
//! use wx_core::download::catalog;
//!
//! // List all available groups
//! for group in catalog::variable_groups() {
//!     println!("{}: {}", group.name, group.description);
//! }
//!
//! // Expand a group name to its patterns
//! if let Some(patterns) = catalog::expand_var_group("severe") {
//!     for pat in &patterns {
//!         println!("  {}", pat);
//!     }
//! }
//! ```

/// A named collection of .idx search patterns for a common weather analysis task.
#[derive(Debug, Clone)]
pub struct VariableGroup {
    /// Short identifier (e.g., "severe", "surface_basic").
    pub name: &'static str,
    /// Human-readable description of what this group contains.
    pub description: &'static str,
    /// .idx search patterns in `"VAR:level"` format.
    pub patterns: &'static [&'static str],
}

/// Standard pressure levels used for sounding analysis (mb).
const SOUNDING_LEVELS: &[u32] = &[
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 675, 650,
    625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 375, 350, 325, 300, 275,
    250, 225, 200, 175, 150, 125, 100,
];

/// Return the full list of pre-defined variable groups.
pub fn variable_groups() -> &'static [VariableGroup] {
    &[
        VariableGroup {
            name: "surface_basic",
            description: "Basic surface variables (T2m, Td2m, wind, pressure)",
            patterns: &[
                "TMP:2 m",
                "DPT:2 m",
                "UGRD:10 m",
                "VGRD:10 m",
                "PRES:surface",
                "MSLMA",
            ],
        },
        VariableGroup {
            name: "surface_precip",
            description: "Precipitation and moisture",
            patterns: &[
                "APCP:surface",
                "CRAIN:surface",
                "CFRZR:surface",
                "CICEP:surface",
                "CSNOW:surface",
                "PRATE:surface",
            ],
        },
        VariableGroup {
            name: "severe",
            description: "Severe weather parameters",
            patterns: &[
                "CAPE:surface",
                "CIN:surface",
                "REFC:entire",
                "MXUPHL",
                "USTM:6000",
                "VSTM:6000",
                "HLCY:3000",
            ],
        },
        VariableGroup {
            name: "upper_air",
            description: "Standard pressure level analysis",
            patterns: &[
                "HGT:500 mb",
                "HGT:250 mb",
                "TMP:850 mb",
                "TMP:700 mb",
                "TMP:500 mb",
                "UGRD:250 mb",
                "VGRD:250 mb",
                "VVEL:700 mb",
                "RH:700 mb",
            ],
        },
        VariableGroup {
            name: "winter",
            description: "Winter weather variables",
            patterns: &[
                "SNOD:surface",
                "WEASD:surface",
                "CSNOW:surface",
                "CFRZR:surface",
                "CICEP:surface",
                "TMP:2 m",
                "TMP:surface",
                "TMP:850 mb",
            ],
        },
        VariableGroup {
            name: "fire_weather",
            description: "Fire weather variables",
            patterns: &[
                "TMP:2 m",
                "RH:2 m",
                "UGRD:10 m",
                "VGRD:10 m",
                "GUST:surface",
                "VIS:surface",
                "HINDEX:surface",
            ],
        },
        VariableGroup {
            name: "aviation",
            description: "Aviation weather",
            patterns: &[
                "VIS:surface",
                "CEIL:cloud ceiling",
                "LCDC",
                "MCDC",
                "HCDC",
                "GUST:surface",
                "UGRD:80 m",
                "VGRD:80 m",
                "ICIP",
                "ICSEV",
            ],
        },
        VariableGroup {
            name: "marine",
            description: "Marine weather",
            patterns: &[
                "UGRD:10 m",
                "VGRD:10 m",
                "GUST:surface",
                "PRMSL",
                "HTSGW",
                "WVHGT",
                "WVPER",
                "WVDIR",
            ],
        },
        VariableGroup {
            name: "radiation",
            description: "Radiation and energy budget",
            patterns: &[
                "DSWRF:surface",
                "DLWRF:surface",
                "USWRF:surface",
                "ULWRF:surface",
                "USWRF:top of atmosphere",
                "ULWRF:top of atmosphere",
            ],
        },
        VariableGroup {
            name: "turbulence",
            description: "Boundary layer and turbulence",
            patterns: &[
                "HPBL:surface",
                "FRICV:surface",
                "GUST:surface",
                "VUCSH:0-1000",
                "VVCSH:0-1000",
                "TKE",
            ],
        },
        VariableGroup {
            name: "moisture",
            description: "Moisture and instability",
            patterns: &[
                "PWAT:entire",
                "RH:2 m",
                "RH:700 mb",
                "RH:850 mb",
                "CAPE:surface",
                "CIN:surface",
                "CAPE:255-0 mb",
                "CIN:255-0 mb",
                "LFTX:500-1000",
                "4LFTX:180-0 mb",
            ],
        },
    ]
}

/// Expand a variable group name to its constituent search patterns.
///
/// Returns `None` if the group name is not recognized.
///
/// For the special group `"full_sounding"`, dynamically generates patterns
/// for TMP, RH, UGRD, VGRD, and HGT at all 37 standard pressure levels.
pub fn expand_var_group(name: &str) -> Option<Vec<&'static str>> {
    // Check the special "full_sounding" group first
    if name == "full_sounding" {
        return Some(full_sounding_patterns());
    }

    variable_groups()
        .iter()
        .find(|g| g.name == name)
        .map(|g| g.patterns.to_vec())
}

/// List all available variable group names.
pub fn group_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = variable_groups()
        .iter()
        .map(|g| g.name)
        .collect();
    names.push("full_sounding");
    names
}

/// Get a variable group by name, returning the full VariableGroup struct.
pub fn get_group(name: &str) -> Option<&'static VariableGroup> {
    variable_groups().iter().find(|g| g.name == name)
}

/// Generate search patterns for a full sounding at all standard pressure levels.
///
/// Produces patterns for TMP, RH, UGRD, VGRD, and HGT at each of the 37
/// standard pressure levels from 1000 mb to 100 mb.
fn full_sounding_patterns() -> Vec<&'static str> {
    // We use leaked static strings so they have 'static lifetime.
    // This is acceptable because it's a fixed, bounded set called rarely.
    use std::sync::OnceLock;
    static PATTERNS: OnceLock<Vec<&'static str>> = OnceLock::new();

    PATTERNS
        .get_or_init(|| {
            let vars = &["TMP", "RH", "UGRD", "VGRD", "HGT"];
            let mut pats = Vec::with_capacity(vars.len() * SOUNDING_LEVELS.len());
            for &var in vars {
                for &level in SOUNDING_LEVELS {
                    let s = format!("{}:{} mb", var, level);
                    pats.push(&*Box::leak(s.into_boxed_str()));
                }
            }
            pats
        })
        .clone()
}

/// Expand a vars specification which may contain group names, returning all
/// individual patterns. Non-group strings are passed through unchanged.
///
/// # Example
///
/// ```
/// use wx_core::download::catalog::expand_vars;
///
/// let patterns = expand_vars(&["severe", "TMP:2 m", "winter"]);
/// // Returns all severe patterns + "TMP:2 m" + all winter patterns
/// ```
pub fn expand_vars(vars: &[&str]) -> Vec<String> {
    let mut result = Vec::new();
    for &v in vars {
        if let Some(patterns) = expand_var_group(v) {
            for p in patterns {
                let s = p.to_string();
                if !result.contains(&s) {
                    result.push(s);
                }
            }
        } else {
            let s = v.to_string();
            if !result.contains(&s) {
                result.push(s);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_groups_not_empty() {
        let groups = variable_groups();
        assert!(!groups.is_empty());
        for group in groups {
            assert!(!group.name.is_empty());
            assert!(!group.description.is_empty());
            assert!(!group.patterns.is_empty());
        }
    }

    #[test]
    fn test_expand_known_group() {
        let patterns = expand_var_group("severe").unwrap();
        assert!(patterns.contains(&"CAPE:surface"));
        assert!(patterns.contains(&"REFC:entire"));
    }

    #[test]
    fn test_expand_unknown_group() {
        assert!(expand_var_group("nonexistent_group").is_none());
    }

    #[test]
    fn test_full_sounding() {
        let patterns = expand_var_group("full_sounding").unwrap();
        // 5 variables * 37 levels = 185 patterns
        assert_eq!(patterns.len(), 5 * SOUNDING_LEVELS.len());
        assert!(patterns.contains(&"TMP:500 mb"));
        assert!(patterns.contains(&"HGT:250 mb"));
        assert!(patterns.contains(&"RH:850 mb"));
    }

    #[test]
    fn test_group_names() {
        let names = group_names();
        assert!(names.contains(&"severe"));
        assert!(names.contains(&"surface_basic"));
        assert!(names.contains(&"full_sounding"));
    }

    #[test]
    fn test_expand_vars_mixed() {
        let expanded = expand_vars(&["severe", "TMP:2 m", "CUSTOM:level"]);
        assert!(expanded.contains(&"CAPE:surface".to_string()));
        assert!(expanded.contains(&"TMP:2 m".to_string()));
        assert!(expanded.contains(&"CUSTOM:level".to_string()));
    }

    #[test]
    fn test_expand_vars_dedup() {
        // "surface_basic" and "fire_weather" both have "TMP:2 m"
        let expanded = expand_vars(&["surface_basic", "fire_weather"]);
        let tmp_count = expanded.iter().filter(|p| *p == "TMP:2 m").count();
        assert_eq!(tmp_count, 1, "duplicate patterns should be deduplicated");
    }
}

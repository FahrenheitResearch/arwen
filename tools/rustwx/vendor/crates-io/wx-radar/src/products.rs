//! Radar product definitions.
//!
//! Enumerates the standard base and dual-pol radar products produced by
//! NEXRAD (WSR-88D) and similar systems.

use std::fmt;

/// A radar product type (base data moment or dual-pol variable).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RadarProduct {
    /// Reflectivity (dBZ).
    Reflectivity,
    /// Radial velocity (m/s).
    Velocity,
    /// Spectrum width (m/s).
    SpectrumWidth,
    /// Differential reflectivity (dB).
    DifferentialReflectivity,
    /// Correlation coefficient (dimensionless, 0-1).
    CorrelationCoefficient,
    /// Specific differential phase (deg/km).
    SpecificDifferentialPhase,
    /// Differential phase (degrees).
    DifferentialPhase,
    /// Composite reflectivity (dBZ) -- max in column.
    CompositeReflectivity,
    /// Storm-relative velocity (m/s).
    StormRelativeVelocity,
    /// Vertically integrated liquid (kg/m^2).
    VIL,
    /// Echo tops (km).
    EchoTops,
    /// Hydrometeor classification.
    HydrometeorClassification,
    /// Unknown or unrecognized moment type.
    Unknown,
}

impl RadarProduct {
    /// Parse product from NEXRAD moment name (e.g. "REF", "VEL", "ZDR").
    pub fn from_name(name: &str) -> Self {
        match name.trim() {
            "REF" | "DREF" => Self::Reflectivity,
            "VEL" | "DVEL" => Self::Velocity,
            "SW" | "DSW" => Self::SpectrumWidth,
            "ZDR" => Self::DifferentialReflectivity,
            "RHO" | "CC" => Self::CorrelationCoefficient,
            "PHI" => Self::DifferentialPhase,
            "KDP" => Self::SpecificDifferentialPhase,
            "HHC" | "HCA" => Self::HydrometeorClassification,
            _ => Self::Unknown,
        }
    }

    /// Base product (for future super-res support).
    pub fn base_product(&self) -> Self {
        *self
    }

    /// Products available for display in UI.
    pub fn all_display() -> &'static [Self] {
        &[
            Self::Reflectivity,
            Self::Velocity,
            Self::SpectrumWidth,
            Self::DifferentialReflectivity,
            Self::CorrelationCoefficient,
            Self::SpecificDifferentialPhase,
        ]
    }

    /// Human-readable display name.
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Reflectivity => "Reflectivity",
            Self::Velocity => "Radial Velocity",
            Self::SpectrumWidth => "Spectrum Width",
            Self::DifferentialReflectivity => "Differential Reflectivity",
            Self::CorrelationCoefficient => "Correlation Coefficient",
            Self::SpecificDifferentialPhase => "Specific Differential Phase",
            Self::DifferentialPhase => "Differential Phase",
            Self::CompositeReflectivity => "Composite Reflectivity",
            Self::StormRelativeVelocity => "Storm-Relative Velocity",
            Self::VIL => "Vertically Integrated Liquid",
            Self::EchoTops => "Echo Tops",
            Self::HydrometeorClassification => "Hydrometeor Classification",
            Self::Unknown => "Unknown",
        }
    }

    /// Short abbreviation used in data files and legends.
    pub fn short_name(&self) -> &'static str {
        match self {
            Self::Reflectivity => "REF",
            Self::Velocity => "VEL",
            Self::SpectrumWidth => "SW",
            Self::DifferentialReflectivity => "ZDR",
            Self::CorrelationCoefficient => "RHO",
            Self::SpecificDifferentialPhase => "KDP",
            Self::DifferentialPhase => "PHI",
            Self::CompositeReflectivity => "CREF",
            Self::StormRelativeVelocity => "SRV",
            Self::VIL => "VIL",
            Self::EchoTops => "ET",
            Self::HydrometeorClassification => "HCA",
            Self::Unknown => "UNK",
        }
    }

    /// Physical unit string.
    pub fn unit(&self) -> &'static str {
        match self {
            Self::Reflectivity => "dBZ",
            Self::Velocity => "m/s",
            Self::SpectrumWidth => "m/s",
            Self::DifferentialReflectivity => "dB",
            Self::CorrelationCoefficient => "",
            Self::SpecificDifferentialPhase => "deg/km",
            Self::DifferentialPhase => "deg",
            Self::CompositeReflectivity => "dBZ",
            Self::StormRelativeVelocity => "m/s",
            Self::VIL => "kg/m^2",
            Self::EchoTops => "km",
            Self::HydrometeorClassification => "",
            Self::Unknown => "",
        }
    }
}

impl fmt::Display for RadarProduct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_product_display_name() {
        assert_eq!(RadarProduct::Reflectivity.display_name(), "Reflectivity");
        assert_eq!(RadarProduct::Velocity.display_name(), "Radial Velocity");
    }

    #[test]
    fn test_product_short_name() {
        assert_eq!(RadarProduct::Reflectivity.short_name(), "REF");
        assert_eq!(RadarProduct::CorrelationCoefficient.short_name(), "RHO");
        assert_eq!(RadarProduct::SpecificDifferentialPhase.short_name(), "KDP");
    }

    #[test]
    fn test_product_unit() {
        assert_eq!(RadarProduct::Reflectivity.unit(), "dBZ");
        assert_eq!(RadarProduct::Velocity.unit(), "m/s");
        assert_eq!(RadarProduct::VIL.unit(), "kg/m^2");
    }

    #[test]
    fn test_product_display_trait() {
        let product = RadarProduct::Reflectivity;
        assert_eq!(format!("{}", product), "REF");
    }

    #[test]
    fn test_all_variants_have_names() {
        let all = [
            RadarProduct::Reflectivity,
            RadarProduct::Velocity,
            RadarProduct::SpectrumWidth,
            RadarProduct::DifferentialReflectivity,
            RadarProduct::CorrelationCoefficient,
            RadarProduct::SpecificDifferentialPhase,
            RadarProduct::DifferentialPhase,
            RadarProduct::CompositeReflectivity,
            RadarProduct::StormRelativeVelocity,
            RadarProduct::VIL,
            RadarProduct::EchoTops,
            RadarProduct::HydrometeorClassification,
        ];
        for p in &all {
            assert!(!p.display_name().is_empty());
            assert!(!p.short_name().is_empty());
        }
    }
}

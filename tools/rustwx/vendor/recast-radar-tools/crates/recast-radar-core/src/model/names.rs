//! Dataset variable names, semantic classes and the Py-ART alias table
//! (`docs/design/fm301-model.md` section 8).

use std::borrow::Cow;
use std::fmt;

#[cfg(feature = "serde")]
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::volume::SourceFormat;

/// Dataset variable name.
///
/// Known names are variants, so comparisons are cheap and each has one
/// spelling. Any other name is `Other`, verbatim. `Other` never holds a known
/// spelling: construct names through [`FieldName::parse`] (or `From<&str>`).
/// Serialized names are the FM301 spellings (`"DBZH"`), not variant names.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum FieldName {
    // FM301-2022 Table 301-9
    /// `DBZH`: equivalent reflectivity factor, horizontal channel (dBZ).
    Dbzh,
    /// `DBZV`: equivalent reflectivity factor, vertical channel (dBZ).
    Dbzv,
    /// `ZH`: linear equivalent reflectivity factor, horizontal channel (mm⁶ m⁻³).
    Zh,
    /// `ZV`: linear equivalent reflectivity factor, vertical channel (mm⁶ m⁻³).
    Zv,
    /// `DBTH`: total power (uncorrected reflectivity), horizontal channel (dBZ).
    Dbth,
    /// `DBTV`: total power (uncorrected reflectivity), vertical channel (dBZ).
    Dbtv,
    /// `TH`: linear total power, horizontal channel (mm⁶ m⁻³).
    Th,
    /// `TV`: linear total power, vertical channel (mm⁶ m⁻³).
    Tv,
    /// `VRADH`: radial velocity of scatterers away from the radar, horizontal channel (m s⁻¹).
    Vradh,
    /// `VRADV`: radial velocity of scatterers away from the radar, vertical channel (m s⁻¹).
    Vradv,
    /// `WRADH`: Doppler spectrum width, horizontal channel (m s⁻¹).
    Wradh,
    /// `WRADV`: Doppler spectrum width, vertical channel (m s⁻¹).
    Wradv,
    /// `ZDR`: log differential reflectivity H/V (dB).
    Zdr,
    /// `LDR`: log linear depolarization ratio (dB).
    Ldr,
    /// `LDRH`: log linear depolarization ratio, horizontal transmit (dB).
    Ldrh,
    /// `LDRV`: log linear depolarization ratio, vertical transmit (dB).
    Ldrv,
    /// `PHIDP`: differential phase H/V (degrees).
    Phidp,
    /// `KDP`: specific differential phase (degrees per km).
    Kdp,
    /// `PHIHX`: cross-polar differential phase (degrees).
    Phihx,
    /// `RHOHV`: co-polar correlation coefficient H/V.
    Rhohv,
    /// `RHOHX`: co-to-cross-polar correlation coefficient, horizontal channel.
    Rhohx,
    /// `RHOVX`: co-to-cross-polar correlation coefficient, vertical channel.
    Rhovx,
    /// `DBM`: received signal power, raw total power (dBm).
    Dbm,
    /// `DBMHC`: received signal power, horizontal co-polar channel (dBm).
    Dbmhc,
    /// `DBMHX`: received signal power, horizontal cross-polar channel (dBm).
    Dbmhx,
    /// `DBMVC`: received signal power, vertical co-polar channel (dBm).
    Dbmvc,
    /// `DBMVX`: received signal power, vertical cross-polar channel (dBm).
    Dbmvx,
    /// `SNR`: signal-to-noise ratio (dB).
    Snr,
    /// `SNRHC`: signal-to-noise ratio, horizontal co-polar channel (dB).
    Snrhc,
    /// `SNRHX`: signal-to-noise ratio, horizontal cross-polar channel (dB).
    Snrhx,
    /// `SNRVC`: signal-to-noise ratio, vertical co-polar channel (dB).
    Snrvc,
    /// `SNRVX`: signal-to-noise ratio, vertical cross-polar channel (dB).
    Snrvx,
    /// `NCP`: normalized coherent power.
    Ncp,
    /// `NCPH`: normalized coherent power, horizontal channel.
    Ncph,
    /// `NCPV`: normalized coherent power, vertical channel.
    Ncpv,
    /// `RR`: radar-estimated precipitation rate (mm h⁻¹).
    Rr,
    /// `REC`: radar echo classification.
    Rec,
    // Outside Table 301-9, but emitted by xradar 0.12 for sources we decode, or
    // by our algorithms.
    /// `DBZ`: equivalent reflectivity factor, polarization not stated (dBZ).
    Dbz,
    /// `VRAD`: radial velocity of scatterers away from the radar, polarization not stated (m s⁻¹).
    Vrad,
    /// `WRAD`: Doppler spectrum width, polarization not stated (m s⁻¹).
    Wrad,
    /// `CCORH`: clutter correction, horizontal channel (dB).
    Ccorh,
    /// `CCORV`: clutter correction, vertical channel (dB).
    Ccorv,
    /// `SQIH`: signal quality index, horizontal channel.
    Sqih,
    /// `SQIV`: signal quality index, vertical channel.
    Sqiv,
    /// `SNRH`: signal-to-noise ratio, horizontal channel (dB).
    Snrh,
    /// `SNRV`: signal-to-noise ratio, vertical channel (dB).
    Snrv,
    /// `RATE`: rainfall rate (mm h⁻¹).
    Rate,
    /// `VRADDH`: dealiased radial velocity, horizontal channel (m s⁻¹).
    Vraddh,
    /// `UZDR`: differential reflectivity before corrections (dB).
    Uzdr,
    /// `UPHIDP`: differential phase before corrections (degrees).
    Uphidp,
    /// `URHOHV`: correlation coefficient before corrections.
    Urhohv,
    /// A verbatim source name (CfRadial `VEL`, DORADE `DBZHC_F`, ODIM `QIND`)
    /// or a derived-product id. Never a known spelling.
    Other(Box<str>),
}

/// Semantic class of a field, regardless of its spelling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Quantity {
    /// Equivalent reflectivity factor, in dBZ.
    Reflectivity,
    /// Linear equivalent reflectivity factor.
    LinearReflectivity,
    /// Total power (reflectivity before clutter and other corrections), in dBZ.
    TotalPower,
    /// Linear total power.
    LinearTotalPower,
    /// Radial velocity.
    RadialVelocity,
    /// Radial velocity after dealiasing.
    DealiasedRadialVelocity,
    /// Doppler spectrum width.
    SpectrumWidth,
    /// Differential reflectivity.
    DifferentialReflectivity,
    /// Linear depolarization ratio.
    LinearDepolarizationRatio,
    /// Differential phase.
    DifferentialPhase,
    /// Specific differential phase.
    SpecificDifferentialPhase,
    /// Co-polar correlation coefficient.
    CorrelationCoefficient,
    /// Cross-polar differential phase.
    CrossPolarDifferentialPhase,
    /// Co-to-cross-polar correlation coefficient.
    CrossPolarCorrelation,
    /// Received power.
    ReceivedPower,
    /// Signal-to-noise ratio.
    SignalToNoiseRatio,
    /// Normalized coherent power.
    NormalizedCoherentPower,
    /// Signal quality index.
    SignalQualityIndex,
    /// Clutter correction.
    ClutterCorrection,
    /// Precipitation rate.
    PrecipitationRate,
    /// Echo classification (hydrometeor or target type).
    EchoClassification,
    /// Anything else.
    Other,
}

/// Polarization channel of a field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum Polarization {
    /// Horizontal.
    H,
    /// Vertical.
    V,
    /// Both, or a quantity of the H/V pair (ZDR, PHIDP, RHOHV).
    Hv,
    /// Horizontal co-polar channel.
    CopolarH,
    /// Horizontal cross-polar channel.
    CrosspolarH,
    /// Vertical co-polar channel.
    CopolarV,
    /// Vertical cross-polar channel.
    CrosspolarV,
    /// Not stated by the name.
    Unspecified,
}

/// xradar 0.12 `sweep_vars_mapping` attributes, verbatim (typos included).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct XradarAttrs {
    /// xradar's `standard_name`.
    pub standard_name: &'static str,
    /// xradar's `long_name`.
    pub long_name: &'static str,
    /// xradar's `units`.
    pub units: &'static str,
}

/// Static metadata for a known name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NameInfo {
    /// The variable name.
    pub name: &'static str,
    /// Semantic class of the field.
    pub quantity: Quantity,
    /// Polarization channel of the field.
    pub polarization: Polarization,
    /// FM301 Table 301-9 standard name, else xradar's.
    pub standard_name: Option<&'static str>,
    /// FM301 Table 301-9 long name.
    pub long_name: &'static str,
    /// UDUNITS spelling for the WMO flavor (`"m s-1"`, `"dB"`, `"degree"`).
    pub units: &'static str,
    /// Units verbatim from xradar 0.12 `sweep_vars_mapping` (`"meters per
    /// seconds"`, `"unitless"`); the WMO units when xradar has no entry.
    pub units_xradar: &'static str,
    /// The xradar 0.12 mapping entry, when xradar has one. Without one xradar
    /// writes no `standard_name`, `long_name` or `units`.
    pub xradar: Option<XradarAttrs>,
    /// Default field name in `pyart.config` ([`PyartNames::Config`]).
    pub pyart: Option<&'static str>,
    /// Name Py-ART's ODIM reader (`aux_io.read_odim_h5`, `ODIM_H5_FIELD_NAMES`)
    /// gives this quantity; `None` when that reader does not map it.
    pub pyart_odim: Option<&'static str>,
}

/// How a Py-ART export names fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PyartNames {
    /// `pyart.config` defaults for every source: the names Py-ART's algorithms
    /// expect.
    Config,
    /// The names Py-ART's reader for the volume's source format produces:
    /// config names for NEXRAD (`read_nexrad_archive`), verbatim names for
    /// CfRadial (`read_cfradial`), `aux_io` names for ODIM.
    Reader,
}

macro_rules! field_names {
    ($($variant:ident => $text:literal,)*) => {
        impl FieldName {
            /// The variable name.
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $text,)*
                    Self::Other(name) => name,
                }
            }

            /// Exact, case-sensitive match against the known names; otherwise
            /// `Other(name)`.
            pub fn parse(name: &str) -> FieldName {
                match name {
                    $($text => Self::$variant,)*
                    other => Self::Other(other.into()),
                }
            }

            /// Every known name, in table order.
            pub const KNOWN: &'static [FieldName] = &[$(Self::$variant,)*];
        }
    };
}

field_names! {
    Dbzh => "DBZH",
    Dbzv => "DBZV",
    Zh => "ZH",
    Zv => "ZV",
    Dbth => "DBTH",
    Dbtv => "DBTV",
    Th => "TH",
    Tv => "TV",
    Vradh => "VRADH",
    Vradv => "VRADV",
    Wradh => "WRADH",
    Wradv => "WRADV",
    Zdr => "ZDR",
    Ldr => "LDR",
    Ldrh => "LDRH",
    Ldrv => "LDRV",
    Phidp => "PHIDP",
    Kdp => "KDP",
    Phihx => "PHIHX",
    Rhohv => "RHOHV",
    Rhohx => "RHOHX",
    Rhovx => "RHOVX",
    Dbm => "DBM",
    Dbmhc => "DBMHC",
    Dbmhx => "DBMHX",
    Dbmvc => "DBMVC",
    Dbmvx => "DBMVX",
    Snr => "SNR",
    Snrhc => "SNRHC",
    Snrhx => "SNRHX",
    Snrvc => "SNRVC",
    Snrvx => "SNRVX",
    Ncp => "NCP",
    Ncph => "NCPH",
    Ncpv => "NCPV",
    Rr => "RR",
    Rec => "REC",
    Dbz => "DBZ",
    Vrad => "VRAD",
    Wrad => "WRAD",
    Ccorh => "CCORH",
    Ccorv => "CCORV",
    Sqih => "SQIH",
    Sqiv => "SQIV",
    Snrh => "SNRH",
    Snrv => "SNRV",
    Rate => "RATE",
    Vraddh => "VRADDH",
    Uzdr => "UZDR",
    Uphidp => "UPHIDP",
    Urhohv => "URHOHV",
}

const fn xr(
    standard_name: &'static str,
    long_name: &'static str,
    units: &'static str,
) -> Option<XradarAttrs> {
    Some(XradarAttrs {
        standard_name,
        long_name,
        units,
    })
}

#[allow(clippy::too_many_arguments)]
const fn info(
    name: &'static str,
    quantity: Quantity,
    polarization: Polarization,
    standard_name: &'static str,
    long_name: &'static str,
    units: &'static str,
    xradar: Option<XradarAttrs>,
    pyart: Option<&'static str>,
    pyart_odim: Option<&'static str>,
) -> NameInfo {
    let units_xradar = match xradar {
        Some(attrs) => attrs.units,
        None => units,
    };
    NameInfo {
        name,
        quantity,
        polarization,
        standard_name: Some(standard_name),
        long_name,
        units,
        units_xradar,
        xradar,
        pyart,
        pyart_odim,
    }
}

use Polarization as P;
use Quantity as Q;

// Section 8.2 of the design note. xradar entries are copied verbatim from
// xradar 0.12.0 `model.sweep_vars_mapping`; Py-ART names from arm_pyart 2.2.5
// `default_config` and `aux_io.odim_h5.ODIM_H5_FIELD_NAMES`.
static DBZH: NameInfo = info(
    "DBZH",
    Q::Reflectivity,
    P::H,
    "radar_equivalent_reflectivity_factor_h",
    "Equivalent reflectivity factor H",
    "dBZ",
    xr(
        "radar_equivalent_reflectivity_factor_h",
        "Equivalent reflectivity factor H",
        "dBZ",
    ),
    Some("reflectivity"),
    Some("reflectivity_horizontal"),
);
static DBZV: NameInfo = info(
    "DBZV",
    Q::Reflectivity,
    P::V,
    "radar_equivalent_reflectivity_factor_v",
    "Equivalent reflectivity factor V",
    "dBZ",
    xr(
        "radar_equivalent_reflectivity_factor_v",
        "Equivalent reflectivity factor V",
        "dBZ",
    ),
    None,
    Some("reflectivity_vertical"),
);
static ZH: NameInfo = info(
    "ZH",
    Q::LinearReflectivity,
    P::H,
    "radar_linear_equivalent_reflectivity_factor_h",
    "Linear equivalent reflectivity factor H",
    "mm6 m-3",
    xr(
        "radar_linear_equivalent_reflectivity_factor_h",
        "Linear equivalent reflectivity factor H",
        "unitless",
    ),
    None,
    None,
);
static ZV: NameInfo = info(
    "ZV",
    Q::LinearReflectivity,
    P::V,
    "radar_linear_equivalent_reflectivity_factor_v",
    "Linear equivalent reflectivity factor V",
    "mm6 m-3",
    xr(
        "radar_equivalent_reflectivity_factor_v",
        "Linear equivalent reflectivity factor V",
        "unitless",
    ),
    None,
    None,
);
static DBTH: NameInfo = info(
    "DBTH",
    Q::TotalPower,
    P::H,
    "radar_equivalent_reflectivity_factor_h",
    "Total power H (uncorrected reflectivity)",
    "dBZ",
    xr(
        "radar_equivalent_reflectivity_factor_h",
        "Total power H (uncorrected reflectivity)",
        "dBZ",
    ),
    Some("total_power"),
    None,
);
static DBTV: NameInfo = info(
    "DBTV",
    Q::TotalPower,
    P::V,
    "radar_equivalent_reflectivity_factor_v",
    "Total power V (uncorrected reflectivity)",
    "dBZ",
    xr(
        "radar_equivalent_reflectivity_factor_v",
        "Total power V (uncorrected reflectivity)",
        "dBZ",
    ),
    None,
    None,
);
static TH: NameInfo = info(
    "TH",
    Q::LinearTotalPower,
    P::H,
    "radar_linear_equivalent_reflectivity_factor_h",
    "Linear total power H (uncorrected reflectivity)",
    "mm6 m-3",
    xr(
        "radar_linear_equivalent_reflectivity_factor_h",
        "Linear total power H (uncorrected reflectivity)",
        "unitless",
    ),
    Some("total_power"),
    Some("total_power_horizontal"),
);
static TV: NameInfo = info(
    "TV",
    Q::LinearTotalPower,
    P::V,
    "radar_linear_equivalent_reflectivity_factor_v",
    "Linear total power V (uncorrected reflectivity)",
    "mm6 m-3",
    xr(
        "radar_linear_equivalent_reflectivity_factor_v",
        "Linear total power V (uncorrected reflectivity)",
        "unitless",
    ),
    None,
    Some("total_power_vertical"),
);
static VRADH: NameInfo = info(
    "VRADH",
    Q::RadialVelocity,
    P::H,
    "radial_velocity_of_scatterers_away_from_instrument_h",
    "Radial velocity of scatterers away from instrument H",
    "m s-1",
    xr(
        "radial_velocity_of_scatterers_away_from_instrument_h",
        "Radial velocity of scatterers away from instrument H",
        "meters per seconds",
    ),
    Some("velocity"),
    Some("velocity_horizontal"),
);
static VRADV: NameInfo = info(
    "VRADV",
    Q::RadialVelocity,
    P::V,
    "radial_velocity_of_scatterers_away_from_instrument_v",
    "Radial velocity of scatterers away from instrument V",
    "m s-1",
    xr(
        "radial_velocity_of_scatterers_away_from_instrument_v",
        "Radial velocity of scatterers away from instrument V",
        "meters per second",
    ),
    None,
    Some("velocity_vertical"),
);
static WRADH: NameInfo = info(
    "WRADH",
    Q::SpectrumWidth,
    P::H,
    "radar_doppler_spectrum_width_h",
    "Doppler spectrum width H",
    "m s-1",
    xr(
        "radar_doppler_spectrum_width_h",
        "Doppler spectrum width H",
        "meters per seconds",
    ),
    Some("spectrum_width"),
    None,
);
static WRADV: NameInfo = info(
    "WRADV",
    Q::SpectrumWidth,
    P::V,
    "radar_doppler_spectrum_width_v",
    "Doppler spectrum width V",
    "m s-1",
    xr(
        "radar_doppler_spectrum_width_v",
        "Doppler spectrum width V",
        "meters per second",
    ),
    None,
    None,
);
static ZDR: NameInfo = info(
    "ZDR",
    Q::DifferentialReflectivity,
    P::Hv,
    "radar_differential_reflectivity_hv",
    "Log differential reflectivity H/V",
    "dB",
    xr(
        "radar_differential_reflectivity_hv",
        "Log differential reflectivity H/V",
        "dB",
    ),
    Some("differential_reflectivity"),
    Some("differential_reflectivity"),
);
static LDR: NameInfo = info(
    "LDR",
    Q::LinearDepolarizationRatio,
    P::Hv,
    "radar_linear_depolarization_ratio",
    "Log-linear depolarization ratio HV",
    "dB",
    xr(
        "radar_linear_depolarization_ratio",
        "Log-linear depolarization ratio HV",
        "dB",
    ),
    Some("linear_polarization_ratio"),
    Some("linear_polarization_ratio"),
);
static LDRH: NameInfo = info(
    "LDRH",
    Q::LinearDepolarizationRatio,
    P::H,
    "radar_linear_depolarization_ratio_h",
    "Log-linear depolarization ratio H",
    "dB",
    None,
    Some("linear_depolarization_ratio_h"),
    None,
);
static LDRV: NameInfo = info(
    "LDRV",
    Q::LinearDepolarizationRatio,
    P::V,
    "radar_linear_depolarization_ratio_v",
    "Log-linear depolarization ratio V",
    "dB",
    None,
    Some("linear_depolarization_ratio_v"),
    None,
);
static PHIDP: NameInfo = info(
    "PHIDP",
    Q::DifferentialPhase,
    P::Hv,
    "radar_differential_phase_hv",
    "Differential phase HV",
    "degree",
    xr(
        "radar_differential_phase_hv",
        "Differential phase HV",
        "degrees",
    ),
    Some("differential_phase"),
    Some("differential_phase"),
);
static KDP: NameInfo = info(
    "KDP",
    Q::SpecificDifferentialPhase,
    P::Hv,
    "radar_specific_differential_phase_hv",
    "Specific differential phase HV",
    "degree km-1",
    xr(
        "radar_specific_differential_phase_hv",
        "Specific differential phase HV",
        "degrees per kilometer",
    ),
    Some("specific_differential_phase"),
    Some("specific_differential_phase"),
);
static PHIHX: NameInfo = info(
    "PHIHX",
    Q::CrossPolarDifferentialPhase,
    P::Hv,
    "radar_differential_phase_copolar_h_crosspolar_v",
    "Cross-polar differential phase",
    "degree",
    None,
    None,
    None,
);
static RHOHV: NameInfo = info(
    "RHOHV",
    Q::CorrelationCoefficient,
    P::Hv,
    "radar_correlation_coefficient_hv",
    "Correlation coefficient HV",
    "1",
    xr(
        "radar_correlation_coefficient_hv",
        "Correlation coefficient HV",
        "unitless",
    ),
    Some("cross_correlation_ratio"),
    Some("cross_correlation_ratio"),
);
static RHOHX: NameInfo = info(
    "RHOHX",
    Q::CrossPolarCorrelation,
    P::CopolarH,
    "radar_correlation_coefficient_copolar_h_crosspolar_v",
    "Co-to-cross polar correlation coefficient H",
    "1",
    None,
    None,
    None,
);
static RHOVX: NameInfo = info(
    "RHOVX",
    Q::CrossPolarCorrelation,
    P::CopolarV,
    "radar_correlation_coefficient_copolar_v_crosspolar_h",
    "Co-to-cross polar correlation coefficient V",
    "1",
    None,
    None,
    None,
);
static DBM: NameInfo = info(
    "DBM",
    Q::ReceivedPower,
    P::Unspecified,
    "radar_received_signal_power",
    "Radar Received Signal Power (raw total power)",
    "dBm",
    xr(
        "radar_received_signal_power",
        "Radar Received Signal Power (raw total power)",
        "dBm",
    ),
    None,
    None,
);
static DBMHC: NameInfo = info(
    "DBMHC",
    Q::ReceivedPower,
    P::CopolarH,
    "radar_received_signal_power_copolar_h",
    "Received signal power copolar H",
    "dBm",
    None,
    None,
    None,
);
static DBMHX: NameInfo = info(
    "DBMHX",
    Q::ReceivedPower,
    P::CrosspolarH,
    "radar_received_signal_power_crosspolar_h",
    "Received signal power crosspolar H",
    "dBm",
    None,
    None,
    None,
);
static DBMVC: NameInfo = info(
    "DBMVC",
    Q::ReceivedPower,
    P::CopolarV,
    "radar_received_signal_power_copolar_v",
    "Received signal power copolar V",
    "dBm",
    None,
    None,
    None,
);
static DBMVX: NameInfo = info(
    "DBMVX",
    Q::ReceivedPower,
    P::CrosspolarV,
    "radar_received_signal_power_crosspolar_v",
    "Received signal power crosspolar V",
    "dBm",
    None,
    None,
    None,
);
static SNR: NameInfo = info(
    "SNR",
    Q::SignalToNoiseRatio,
    P::Unspecified,
    "radar_signal_to_noise_ratio",
    "Signal-to-noise ratio",
    "dB",
    None,
    Some("signal_to_noise_ratio"),
    Some("signal_to_noise_ratio"),
);
static SNRHC: NameInfo = info(
    "SNRHC",
    Q::SignalToNoiseRatio,
    P::CopolarH,
    "radar_signal_to_noise_ratio_copolar_h",
    "Signal-to-noise ratio copolar H",
    "dB",
    None,
    None,
    None,
);
static SNRHX: NameInfo = info(
    "SNRHX",
    Q::SignalToNoiseRatio,
    P::CrosspolarH,
    "radar_signal_to_noise_ratio_crosspolar_h",
    "Signal-to-noise ratio crosspolar H",
    "dB",
    None,
    None,
    None,
);
static SNRVC: NameInfo = info(
    "SNRVC",
    Q::SignalToNoiseRatio,
    P::CopolarV,
    "radar_signal_to_noise_ratio_copolar_v",
    "Signal-to-noise ratio copolar V",
    "dB",
    None,
    None,
    None,
);
static SNRVX: NameInfo = info(
    "SNRVX",
    Q::SignalToNoiseRatio,
    P::CrosspolarV,
    "radar_signal_to_noise_ratio_crosspolar_v",
    "Signal-to-noise ratio crosspolar V",
    "dB",
    None,
    None,
    None,
);
static NCP: NameInfo = info(
    "NCP",
    Q::NormalizedCoherentPower,
    P::Unspecified,
    "radar_normalized_coherent_power",
    "Normalized coherent power",
    "1",
    None,
    Some("normalized_coherent_power"),
    None,
);
static NCPH: NameInfo = info(
    "NCPH",
    Q::NormalizedCoherentPower,
    P::H,
    "radar_normalized_coherent_power_h",
    "Normalized coherent power H",
    "1",
    None,
    None,
    None,
);
static NCPV: NameInfo = info(
    "NCPV",
    Q::NormalizedCoherentPower,
    P::V,
    "radar_normalized_coherent_power_v",
    "Normalized coherent power V",
    "1",
    None,
    None,
    None,
);
static RR: NameInfo = info(
    "RR",
    Q::PrecipitationRate,
    P::Unspecified,
    "radar_estimated_precipitation_rate",
    "Radar estimated precipitation rate",
    "mm h-1",
    None,
    Some("radar_estimated_rain_rate"),
    None,
);
static REC: NameInfo = info(
    "REC",
    Q::EchoClassification,
    P::Unspecified,
    "radar_scatterer_classification",
    "Radar echo classification",
    "1",
    None,
    Some("radar_echo_classification"),
    None,
);
static DBZ: NameInfo = info(
    "DBZ",
    Q::Reflectivity,
    P::Unspecified,
    "radar_equivalent_reflectivity_factor",
    "Equivalent reflectivity factor",
    "dBZ",
    xr(
        "radar_equivalent_reflectivity_factor",
        "Equivalent reflectivity factor",
        "dBZ",
    ),
    Some("reflectivity"),
    None,
);
static VRAD: NameInfo = info(
    "VRAD",
    Q::RadialVelocity,
    P::Unspecified,
    "radial_velocity_of_scatterers_away_from_instrument",
    "Radial velocity of scatterers away from instrument",
    "m s-1",
    xr(
        "radial_velocity_of_scatterers_away_from_instrument",
        "Radial velocity of scatterers away from instrument",
        "meters per seconds",
    ),
    Some("velocity"),
    Some("velocity"),
);
static WRAD: NameInfo = info(
    "WRAD",
    Q::SpectrumWidth,
    P::Unspecified,
    "radar_doppler_spectrum_width",
    "Doppler spectrum width",
    "m s-1",
    xr(
        "radar_doppler_spectrum_width",
        "Doppler spectrum width",
        "meters per second",
    ),
    Some("spectrum_width"),
    Some("spectrum_width"),
);
static CCORH: NameInfo = info(
    "CCORH",
    Q::ClutterCorrection,
    P::H,
    "clutter_correction_h",
    "Clutter Correction H",
    "dB",
    xr("clutter_correction_h", "Clutter Correction H", "unitless"),
    Some("clutter_filter_power_removed"),
    None,
);
static CCORV: NameInfo = info(
    "CCORV",
    Q::ClutterCorrection,
    P::V,
    "clutter_correction_v",
    "Clutter Correction V",
    "dB",
    xr("clutter_correction_v", "Clutter Correction V", "unitless"),
    None,
    None,
);
static SQIH: NameInfo = info(
    "SQIH",
    Q::SignalQualityIndex,
    P::H,
    "signal_quality_index_h",
    "Signal Quality H",
    "1",
    xr("signal_quality_index_h", "Signal Quality H", "unitless"),
    Some("normalized_coherent_power"),
    None,
);
static SQIV: NameInfo = info(
    "SQIV",
    Q::SignalQualityIndex,
    P::V,
    "signal_quality_index_v",
    "Signal Quality V",
    "1",
    xr("signal_quality_index_v", "Signal Quality V", "unitless"),
    None,
    None,
);
static SNRH: NameInfo = info(
    "SNRH",
    Q::SignalToNoiseRatio,
    P::H,
    "signal_noise_ratio_h",
    "Signal Noise Ratio H",
    "dB",
    xr("signal_noise_ratio_h", "Signal Noise Ratio H", "unitless"),
    Some("signal_to_noise_ratio"),
    None,
);
static SNRV: NameInfo = info(
    "SNRV",
    Q::SignalToNoiseRatio,
    P::V,
    "signal_noise_ratio_v",
    "Signal Noise Ratio V",
    "dB",
    xr("signal_noise_ratio_v", "Signal Noise Ratio V", "unitless"),
    None,
    None,
);
static RATE: NameInfo = info(
    "RATE",
    Q::PrecipitationRate,
    P::Unspecified,
    "rainfall_rate",
    "rainfall_rate",
    "mm h-1",
    xr("rainfall_rate", "rainfall_rate", "mm h-1"),
    Some("radar_estimated_rain_rate"),
    None,
);
static VRADDH: NameInfo = info(
    "VRADDH",
    Q::DealiasedRadialVelocity,
    P::H,
    "radial_velocity_of_scatterers_away_from_instrument_h",
    "Radial velocity of scatterers away from instrument H",
    "m s-1",
    xr(
        "radial_velocity_of_scatterers_away_from_instrument_h",
        "Radial velocity of scatterers away from instrument H",
        "meters per seconds",
    ),
    Some("corrected_velocity"),
    None,
);
static UZDR: NameInfo = info(
    "UZDR",
    Q::DifferentialReflectivity,
    P::Hv,
    "radar_differential_reflectivity_hv",
    "Log differential reflectivity H/V",
    "dB",
    xr(
        "radar_differential_reflectivity_hv",
        "Log differential reflectivity H/V",
        "dB",
    ),
    None,
    None,
);
static UPHIDP: NameInfo = info(
    "UPHIDP",
    Q::DifferentialPhase,
    P::Hv,
    "radar_differential_phase_hv",
    "Differential phase HV",
    "degree",
    xr(
        "radar_differential_phase_hv",
        "Differential phase HV",
        "degrees",
    ),
    None,
    None,
);
static URHOHV: NameInfo = info(
    "URHOHV",
    Q::CorrelationCoefficient,
    P::Hv,
    "radar_correlation_coefficient_hv",
    "Correlation coefficient HV",
    "1",
    xr(
        "radar_correlation_coefficient_hv",
        "Correlation coefficient HV",
        "unitless",
    ),
    None,
    None,
);

/// Py-ART aliases of derived-product names (section 8.3) that are not known
/// [`FieldName`] variants.
const DERIVED_PYART: &[(&str, &str)] = &[
    ("PHIDP_CLEAN", "corrected_differential_phase"),
    ("AH", "specific_attenuation"),
    ("PIA", "path_integrated_attenuation"),
    ("ADP", "specific_differential_attenuation"),
    // pyart 2.2.5's default name, typo included, so Py-ART algorithms find it.
    ("PIDA", "path_integrateddifferential_attenuation"),
    ("DBZH_CORR", "corrected_reflectivity"),
    ("DBZ_CORR", "corrected_reflectivity"),
    ("ZDR_CORR", "corrected_differential_reflectivity"),
    ("CDR", "circular_depolarization_ratio"),
    ("RHOHV_LOG", "logarithmic_cross_correlation_ratio"),
    ("DBZH_TEX", "reflectivity_texture"),
    ("DBZ_TEX", "reflectivity_texture"),
    ("ZDR_TEX", "differential_reflectivity_texture"),
    ("RHOHV_TEX", "cross_correlation_ratio_texture"),
    ("PHIDP_TEX", "differential_phase_texture"),
];

/// Py-ART ODIM reader names for ODIM quantities that are not known variants.
const ODIM_READER_OTHER: &[(&str, &str)] = &[
    ("SQI", "normalized_coherent_power"),
    ("QIND", "quality_index"),
];

impl FieldName {
    /// Message 31 or Message 1 data block name (space- or NUL-padded) to the
    /// xradar name: REF→DBZH, VEL→VRADH, SW→WRADH, ZDR→ZDR, PHI→PHIDP,
    /// RHO→RHOHV, CFP→CCORH. Other names are kept verbatim (through
    /// [`FieldName::parse`]).
    pub fn from_nexrad_block(name: &[u8]) -> FieldName {
        let trimmed = trim_block_name(name);
        match trimmed {
            b"REF" => Self::Dbzh,
            b"VEL" => Self::Vradh,
            b"SW" => Self::Wradh,
            b"ZDR" => Self::Zdr,
            b"PHI" => Self::Phidp,
            b"RHO" => Self::Rhohv,
            b"CFP" => Self::Ccorh,
            other => Self::parse(&String::from_utf8_lossy(other)),
        }
    }

    /// Static metadata for a known name; `None` for `Other`.
    pub fn info(&self) -> Option<&'static NameInfo> {
        Some(match self {
            Self::Dbzh => &DBZH,
            Self::Dbzv => &DBZV,
            Self::Zh => &ZH,
            Self::Zv => &ZV,
            Self::Dbth => &DBTH,
            Self::Dbtv => &DBTV,
            Self::Th => &TH,
            Self::Tv => &TV,
            Self::Vradh => &VRADH,
            Self::Vradv => &VRADV,
            Self::Wradh => &WRADH,
            Self::Wradv => &WRADV,
            Self::Zdr => &ZDR,
            Self::Ldr => &LDR,
            Self::Ldrh => &LDRH,
            Self::Ldrv => &LDRV,
            Self::Phidp => &PHIDP,
            Self::Kdp => &KDP,
            Self::Phihx => &PHIHX,
            Self::Rhohv => &RHOHV,
            Self::Rhohx => &RHOHX,
            Self::Rhovx => &RHOVX,
            Self::Dbm => &DBM,
            Self::Dbmhc => &DBMHC,
            Self::Dbmhx => &DBMHX,
            Self::Dbmvc => &DBMVC,
            Self::Dbmvx => &DBMVX,
            Self::Snr => &SNR,
            Self::Snrhc => &SNRHC,
            Self::Snrhx => &SNRHX,
            Self::Snrvc => &SNRVC,
            Self::Snrvx => &SNRVX,
            Self::Ncp => &NCP,
            Self::Ncph => &NCPH,
            Self::Ncpv => &NCPV,
            Self::Rr => &RR,
            Self::Rec => &REC,
            Self::Dbz => &DBZ,
            Self::Vrad => &VRAD,
            Self::Wrad => &WRAD,
            Self::Ccorh => &CCORH,
            Self::Ccorv => &CCORV,
            Self::Sqih => &SQIH,
            Self::Sqiv => &SQIV,
            Self::Snrh => &SNRH,
            Self::Snrv => &SNRV,
            Self::Rate => &RATE,
            Self::Vraddh => &VRADDH,
            Self::Uzdr => &UZDR,
            Self::Uphidp => &UPHIDP,
            Self::Urhohv => &URHOHV,
            Self::Other(_) => return None,
        })
    }

    /// Py-ART field name for `mode` and the volume's source format, or the
    /// name itself when Py-ART has no alias.
    pub fn pyart_name(&self, mode: PyartNames, source: SourceFormat) -> Cow<'_, str> {
        let name = self.as_str();
        let info = self.info();
        let config = || {
            info.and_then(|info| info.pyart).or_else(|| {
                DERIVED_PYART
                    .iter()
                    .find(|(derived, _)| *derived == name)
                    .map(|(_, alias)| *alias)
            })
        };
        let alias = match (mode, source) {
            (PyartNames::Config, _) => config(),
            (PyartNames::Reader, SourceFormat::CfRadial1 | SourceFormat::CfRadial2) => None,
            (PyartNames::Reader, SourceFormat::OdimH5) => {
                info.and_then(|info| info.pyart_odim).or_else(|| {
                    ODIM_READER_OTHER
                        .iter()
                        .find(|(odim, _)| *odim == name)
                        .map(|(_, alias)| *alias)
                })
            }
            (PyartNames::Reader, _) => config(),
        };
        match alias {
            Some(alias) => Cow::Borrowed(alias),
            None => Cow::Borrowed(name),
        }
    }
}

fn trim_block_name(mut bytes: &[u8]) -> &[u8] {
    while let [0 | b' ' | b'\t' | b'\r' | b'\n', rest @ ..] = bytes {
        bytes = rest;
    }
    while let [rest @ .., 0 | b' ' | b'\t' | b'\r' | b'\n'] = bytes {
        bytes = rest;
    }
    bytes
}

impl fmt::Display for FieldName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for FieldName {
    fn from(name: &str) -> Self {
        Self::parse(name)
    }
}

#[cfg(feature = "serde")]
impl Serialize for FieldName {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for FieldName {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = Cow::<'de, str>::deserialize(deserializer)?;
        Ok(Self::parse(&name))
    }
}

/// Standard names that are not a known name's own standard name: CF / Py-ART
/// (`pyart.default_config`) spellings.
const CF_STANDARD_NAMES: &[(&str, Quantity, Polarization)] = &[
    (
        "equivalent_reflectivity_factor",
        Q::Reflectivity,
        P::Unspecified,
    ),
    (
        "corrected_equivalent_reflectivity_factor",
        Q::Reflectivity,
        P::Unspecified,
    ),
    (
        "radial_velocity_of_scatterers_away_from_instrument",
        Q::RadialVelocity,
        P::Unspecified,
    ),
    (
        "corrected_radial_velocity_of_scatterers_away_from_instrument",
        Q::DealiasedRadialVelocity,
        P::Unspecified,
    ),
    ("doppler_spectrum_width", Q::SpectrumWidth, P::Unspecified),
    (
        "log_differential_reflectivity_hv",
        Q::DifferentialReflectivity,
        P::Hv,
    ),
    (
        "corrected_log_differential_reflectivity_hv",
        Q::DifferentialReflectivity,
        P::Hv,
    ),
    (
        "cross_correlation_ratio_hv",
        Q::CorrelationCoefficient,
        P::Hv,
    ),
    ("differential_phase_hv", Q::DifferentialPhase, P::Hv),
    (
        "corrected_differential_phase_hv",
        Q::DifferentialPhase,
        P::Hv,
    ),
    (
        "specific_differential_phase_hv",
        Q::SpecificDifferentialPhase,
        P::Hv,
    ),
    (
        "corrected_specific_differential_phase_hv",
        Q::SpecificDifferentialPhase,
        P::Hv,
    ),
    (
        "normalized_coherent_power",
        Q::NormalizedCoherentPower,
        P::Unspecified,
    ),
    (
        "log_linear_depolarization_ratio_hv",
        Q::LinearDepolarizationRatio,
        P::Hv,
    ),
    (
        "log_linear_depolarization_ratio_h",
        Q::LinearDepolarizationRatio,
        P::H,
    ),
    (
        "log_linear_depolarization_ratio_v",
        Q::LinearDepolarizationRatio,
        P::V,
    ),
    (
        "signal_to_noise_ratio",
        Q::SignalToNoiseRatio,
        P::Unspecified,
    ),
    (
        "radar_estimated_rain_rate",
        Q::PrecipitationRate,
        P::Unspecified,
    ),
    ("rain_rate", Q::PrecipitationRate, P::Unspecified),
    (
        "radar_echo_classification",
        Q::EchoClassification,
        P::Unspecified,
    ),
    ("clutter_filter_power_removed", Q::ClutterCorrection, P::H),
];

/// Py-ART config and ODIM-reader field names.
const PYART_NAMES: &[(&str, Quantity, Polarization)] = &[
    ("reflectivity", Q::Reflectivity, P::Unspecified),
    ("reflectivity_horizontal", Q::Reflectivity, P::H),
    ("reflectivity_vertical", Q::Reflectivity, P::V),
    ("corrected_reflectivity", Q::Reflectivity, P::Unspecified),
    ("total_power", Q::TotalPower, P::Unspecified),
    ("total_power_horizontal", Q::TotalPower, P::H),
    ("total_power_vertical", Q::TotalPower, P::V),
    ("velocity", Q::RadialVelocity, P::Unspecified),
    ("velocity_horizontal", Q::RadialVelocity, P::H),
    ("velocity_vertical", Q::RadialVelocity, P::V),
    (
        "corrected_velocity",
        Q::DealiasedRadialVelocity,
        P::Unspecified,
    ),
    ("spectrum_width", Q::SpectrumWidth, P::Unspecified),
    (
        "differential_reflectivity",
        Q::DifferentialReflectivity,
        P::Hv,
    ),
    (
        "corrected_differential_reflectivity",
        Q::DifferentialReflectivity,
        P::Hv,
    ),
    ("cross_correlation_ratio", Q::CorrelationCoefficient, P::Hv),
    ("differential_phase", Q::DifferentialPhase, P::Hv),
    ("corrected_differential_phase", Q::DifferentialPhase, P::Hv),
    (
        "specific_differential_phase",
        Q::SpecificDifferentialPhase,
        P::Hv,
    ),
    (
        "linear_polarization_ratio",
        Q::LinearDepolarizationRatio,
        P::Hv,
    ),
    (
        "linear_depolarization_ratio_h",
        Q::LinearDepolarizationRatio,
        P::H,
    ),
    (
        "linear_depolarization_ratio_v",
        Q::LinearDepolarizationRatio,
        P::V,
    ),
    (
        "signal_to_noise_ratio",
        Q::SignalToNoiseRatio,
        P::Unspecified,
    ),
    (
        "normalized_coherent_power",
        Q::NormalizedCoherentPower,
        P::Unspecified,
    ),
    (
        "radar_estimated_rain_rate",
        Q::PrecipitationRate,
        P::Unspecified,
    ),
    (
        "radar_echo_classification",
        Q::EchoClassification,
        P::Unspecified,
    ),
    ("clutter_filter_power_removed", Q::ClutterCorrection, P::H),
];

impl Quantity {
    /// Semantic class of a verbatim source name. Tries, in order:
    ///
    /// 1. `standard_name`: a known name's own standard name when `name` is that
    ///    known name, then FM301 Table 301-9 / xradar and CF / Py-ART standard
    ///    names. An unrecognised standard name falls through (DOW8 writes the
    ///    variable name as its standard name).
    /// 2. The exact name: a known FM301 / xradar name, then Py-ART config and
    ///    ODIM-reader names (`reflectivity_horizontal` → Reflectivity / H).
    /// 3. Suffix stripping over FM301, ODIM, CfRadial, DORADE and Radx stems
    ///    (`DBZHC_F` → `DBZHC` → `DBZ` → Reflectivity).
    ///
    /// Returns `(Other, Unspecified)` when nothing matches.
    pub fn classify(name: &str, standard_name: Option<&str>) -> (Quantity, Polarization) {
        let name = name.trim();
        let known = FieldName::parse(name);
        if let Some(standard_name) = standard_name.map(str::trim).filter(|s| !s.is_empty()) {
            if let Some(info) = known.info()
                && info.standard_name == Some(standard_name)
            {
                return (info.quantity, info.polarization);
            }
            if let Some(found) = FieldName::KNOWN
                .iter()
                .filter_map(FieldName::info)
                .find(|info| info.standard_name == Some(standard_name))
            {
                return (found.quantity, found.polarization);
            }
            if let Some((_, quantity, polarization)) = CF_STANDARD_NAMES
                .iter()
                .find(|(candidate, _, _)| *candidate == standard_name)
            {
                return (*quantity, *polarization);
            }
        }
        if let Some(info) = known.info() {
            return (info.quantity, info.polarization);
        }
        if let Some((_, quantity, polarization)) = PYART_NAMES
            .iter()
            .find(|(candidate, _, _)| *candidate == name)
        {
            return (*quantity, *polarization);
        }
        classify_stem(name)
    }
}

fn classify_stem(name: &str) -> (Quantity, Polarization) {
    let normalized = name.to_ascii_uppercase();
    let mut stem = normalized.as_str();
    loop {
        if let Some(found) = match_stem(stem) {
            return found;
        }
        let Some(next) = ["_F", "_HC", "_VC", "HC", "_V", "_H"]
            .iter()
            .find_map(|suffix| stem.strip_suffix(suffix).filter(|rest| !rest.is_empty()))
        else {
            return (Q::Other, P::Unspecified);
        };
        stem = next;
    }
}

fn match_stem(stem: &str) -> Option<(Quantity, Polarization)> {
    if let Some(info) = FieldName::parse(stem).info() {
        return Some((info.quantity, info.polarization));
    }
    Some(match stem {
        "DZ" | "REF" | "CZ" | "UZ" => (Q::Reflectivity, P::Unspecified),
        "VR" | "VE" | "VEL" | "VU" | "VG" | "VT" => (Q::RadialVelocity, P::Unspecified),
        "SW" | "WIDTH" | "SPW" | "SPECTRUM_WIDTH" => (Q::SpectrumWidth, P::Unspecified),
        "ZD" => (Q::DifferentialReflectivity, P::Hv),
        "RHO" | "RH" | "ROHV" => (Q::CorrelationCoefficient, P::Hv),
        "PHI" | "PH" => (Q::DifferentialPhase, P::Hv),
        "KD" => (Q::SpecificDifferentialPhase, P::Hv),
        "SQI" => (Q::SignalQualityIndex, P::Unspecified),
        "CFP" => (Q::ClutterCorrection, P::H),
        _ => return None,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trips_every_known_name() {
        for name in FieldName::KNOWN {
            assert_eq!(&FieldName::parse(name.as_str()), name);
            assert_eq!(name.info().map(|info| info.name), Some(name.as_str()));
        }
        assert_eq!(FieldName::parse("VEL"), FieldName::Other("VEL".into()));
        assert_eq!(FieldName::parse("dbzh"), FieldName::Other("dbzh".into()));
    }

    #[test]
    fn nexrad_block_names_follow_xradar_mapping() {
        assert_eq!(FieldName::from_nexrad_block(b"REF"), FieldName::Dbzh);
        assert_eq!(FieldName::from_nexrad_block(b"SW "), FieldName::Wradh);
        assert_eq!(FieldName::from_nexrad_block(b"\0CFP"), FieldName::Ccorh);
        assert_eq!(FieldName::from_nexrad_block(b"KDP"), FieldName::Kdp);
        assert_eq!(
            FieldName::from_nexrad_block(b"XYZ"),
            FieldName::Other("XYZ".into())
        );
    }

    #[test]
    fn pyart_aliases_depend_on_mode_and_source() {
        let dbzh = FieldName::Dbzh;
        assert_eq!(
            dbzh.pyart_name(PyartNames::Config, SourceFormat::OdimH5),
            "reflectivity"
        );
        assert_eq!(
            dbzh.pyart_name(PyartNames::Reader, SourceFormat::OdimH5),
            "reflectivity_horizontal"
        );
        assert_eq!(
            dbzh.pyart_name(PyartNames::Reader, SourceFormat::CfRadial1),
            "DBZH"
        );
        assert_eq!(
            FieldName::parse("PIDA").pyart_name(PyartNames::Config, SourceFormat::NexradLevel2),
            "path_integrateddifferential_attenuation"
        );
        assert_eq!(
            FieldName::Ccorh.pyart_name(PyartNames::Reader, SourceFormat::NexradLevel2),
            "clutter_filter_power_removed"
        );
    }

    #[test]
    fn classify_uses_standard_name_then_name_then_stems() {
        assert_eq!(
            Quantity::classify("DBZHC_F", None),
            (Quantity::Reflectivity, Polarization::Unspecified)
        );
        assert_eq!(
            Quantity::classify("DBTH", Some("radar_equivalent_reflectivity_factor_h")),
            (Quantity::TotalPower, Polarization::H)
        );
        assert_eq!(
            Quantity::classify(
                "reflectivity_horizontal",
                Some("equivalent_reflectivity_factor")
            ),
            (Quantity::Reflectivity, Polarization::Unspecified)
        );
        // DOW8 writes the variable name as its standard name: falls through.
        assert_eq!(
            Quantity::classify("VEL", Some("VEL")),
            (Quantity::RadialVelocity, Polarization::Unspecified)
        );
        assert_eq!(
            Quantity::classify("QIND", None),
            (Quantity::Other, Polarization::Unspecified)
        );
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_uses_fm301_spelling() {
        let names = vec![FieldName::Dbzh, FieldName::parse("VEL")];
        let json = serde_json::to_string(&names).unwrap();
        assert_eq!(json, "[\"DBZH\",\"VEL\"]");
        let back: Vec<FieldName> = serde_json::from_str(&json).unwrap();
        assert_eq!(back, names);
    }
}

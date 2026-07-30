use crate::presentation::ProductVisualMode;
use crate::request::{Color, DiscreteColorScale, ExtendMode, ProductSemantics};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeatherPalette {
    Cape,
    Ecape,
    ThreeCape,
    Ncape,
    Cin,
    Scp,
    Ehi,
    TornadicEhi,
    TornadicTts,
    ViolentTornado,
    Srh,
    Stp,
    HeightAgl,
    EquilibriumLevel,
    LapseRate,
    Uh,
    EcapeRatio,
    MlMetric,
    Reflectivity,
    Winds,
    Temperature,
    Dewpoint,
    Rh,
    RelVort,
    Advection,
    SimIr,
    GeopotAnomaly,
    Precip,
    ShadedOverlay,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WeatherPreset {
    Cape,
    Ecape,
    ThreeCape,
    Ncape,
    Cin,
    Lcl,
    Lfc,
    El,
    Srh,
    Stp,
    Scp,
    Ehi,
    Tehi,
    Tts,
    Vtp,
    EcapeCapeRatio,
    Uh,
    LapseRate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DerivedScalePreset {
    LiftedIndex,
    TemperatureAdvection,
    BulkShear,
    SurfaceComfort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DerivedProductStyle {
    LiftedIndex,
    TemperatureAdvection700mb,
    TemperatureAdvection850mb,
    BulkShear01km,
    BulkShear06km,
    ApparentTemperature,
    HeatIndex,
    WindChill,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WeatherProduct {
    Sbcape,
    Mlcape,
    Mucape,
    Sbecape,
    Mlecape,
    Muecape,
    SbEcapeDerivedCapeRatio,
    MlEcapeDerivedCapeRatio,
    MuEcapeDerivedCapeRatio,
    SbEcapeNativeCapeRatio,
    MlEcapeNativeCapeRatio,
    MuEcapeNativeCapeRatio,
    Sbncape,
    Mlncape,
    Muncape,
    Sbcin,
    Mlcin,
    Mucin,
    Sbecin,
    Mlecin,
    Muecin,
    EcapeCape,
    EcapeCin,
    Lcl,
    Lfc,
    El,
    EcapeLfc,
    EcapeEl,
    Srh01km,
    Srh03km,
    Stp,
    StpFixed,
    StpEffective,
    Scp,
    Ehi,
    Tehi,
    Tts,
    VtpMod,
    Uh,
    EcapeScpExperimental,
    EcapeEhi01kmExperimental,
    EcapeEhi03kmExperimental,
    EcapeStpExperimental,
}

pub const SEVERE_CLASSIC_PANEL_PRODUCTS: [WeatherProduct; 8] = [
    WeatherProduct::Sbcape,
    WeatherProduct::Mlcape,
    WeatherProduct::Mucape,
    WeatherProduct::Mlcin,
    WeatherProduct::Srh01km,
    WeatherProduct::Srh03km,
    WeatherProduct::Stp,
    WeatherProduct::Scp,
];

pub const ECAPE_SEVERE_PANEL_PRODUCTS: [WeatherProduct; 16] = [
    WeatherProduct::Sbecape,
    WeatherProduct::Mlecape,
    WeatherProduct::Muecape,
    WeatherProduct::SbEcapeDerivedCapeRatio,
    WeatherProduct::MlEcapeDerivedCapeRatio,
    WeatherProduct::MuEcapeDerivedCapeRatio,
    WeatherProduct::SbEcapeNativeCapeRatio,
    WeatherProduct::MlEcapeNativeCapeRatio,
    WeatherProduct::MuEcapeNativeCapeRatio,
    WeatherProduct::Sbncape,
    WeatherProduct::Sbecin,
    WeatherProduct::Mlecin,
    WeatherProduct::EcapeScpExperimental,
    WeatherProduct::EcapeEhi01kmExperimental,
    WeatherProduct::EcapeEhi03kmExperimental,
    WeatherProduct::EcapeStpExperimental,
];

impl WeatherProduct {
    pub fn from_product_name(name: &str) -> Option<Self> {
        match normalize(name).as_str() {
            "sbcape" => Some(Self::Sbcape),
            "mlcape" => Some(Self::Mlcape),
            "mucape" => Some(Self::Mucape),
            "sbecape" => Some(Self::Sbecape),
            "mlecape" => Some(Self::Mlecape),
            "muecape" => Some(Self::Muecape),
            "sb_ecape_derived_cape_ratio" | "sbecape_derived_cape_ratio" => {
                Some(Self::SbEcapeDerivedCapeRatio)
            }
            "ml_ecape_derived_cape_ratio" | "mlecape_derived_cape_ratio" => {
                Some(Self::MlEcapeDerivedCapeRatio)
            }
            "mu_ecape_derived_cape_ratio" | "muecape_derived_cape_ratio" => {
                Some(Self::MuEcapeDerivedCapeRatio)
            }
            "sb_ecape_native_cape_ratio" | "sbecape_native_cape_ratio" => {
                Some(Self::SbEcapeNativeCapeRatio)
            }
            "ml_ecape_native_cape_ratio" | "mlecape_native_cape_ratio" => {
                Some(Self::MlEcapeNativeCapeRatio)
            }
            "mu_ecape_native_cape_ratio" | "muecape_native_cape_ratio" => {
                Some(Self::MuEcapeNativeCapeRatio)
            }
            "sbncape" => Some(Self::Sbncape),
            "mlncape" => Some(Self::Mlncape),
            "muncape" => Some(Self::Muncape),
            "sbcin" => Some(Self::Sbcin),
            "mlcin" => Some(Self::Mlcin),
            "mucin" => Some(Self::Mucin),
            "sbecin" => Some(Self::Sbecin),
            "mlecin" => Some(Self::Mlecin),
            "muecin" => Some(Self::Muecin),
            "ecape_cape" => Some(Self::EcapeCape),
            "ecape_cin" => Some(Self::EcapeCin),
            "lcl" => Some(Self::Lcl),
            "lfc" => Some(Self::Lfc),
            "el" => Some(Self::El),
            "ecape_lfc" => Some(Self::EcapeLfc),
            "ecape_el" => Some(Self::EcapeEl),
            "srh1" | "srh_0_1km" | "srh01km" => Some(Self::Srh01km),
            "srh3" | "srh_0_3km" | "srh03km" => Some(Self::Srh03km),
            "stp" => Some(Self::Stp),
            "stp_fixed" => Some(Self::StpFixed),
            "stp_effective" => Some(Self::StpEffective),
            "scp" => Some(Self::Scp),
            "ehi" | "ehi_0_1km" | "ehi01km" | "ehi_0_3km" | "ehi03km" => Some(Self::Ehi),
            "tehi" | "tornadic_ehi" | "tornadic_0_1km_ehi" => Some(Self::Tehi),
            "tts" | "tornadic_tilting_stretching" => Some(Self::Tts),
            "vtp_mod" | "modified_vtp" | "vtp" => Some(Self::VtpMod),
            "uhel" | "uh" => Some(Self::Uh),
            "ecape_scp" => Some(Self::EcapeScpExperimental),
            "ecape_ehi" | "ecape_ehi_0_1km" | "ecape_ehi_01km" => {
                Some(Self::EcapeEhi01kmExperimental)
            }
            "ecape_ehi_0_3km" | "ecape_ehi_03km" => Some(Self::EcapeEhi03kmExperimental),
            "ecape_stp" => Some(Self::EcapeStpExperimental),
            _ => None,
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            Self::Sbcape => "sbcape",
            Self::Mlcape => "mlcape",
            Self::Mucape => "mucape",
            Self::Sbecape => "sbecape",
            Self::Mlecape => "mlecape",
            Self::Muecape => "muecape",
            Self::SbEcapeDerivedCapeRatio => "sb_ecape_derived_cape_ratio",
            Self::MlEcapeDerivedCapeRatio => "ml_ecape_derived_cape_ratio",
            Self::MuEcapeDerivedCapeRatio => "mu_ecape_derived_cape_ratio",
            Self::SbEcapeNativeCapeRatio => "sb_ecape_native_cape_ratio",
            Self::MlEcapeNativeCapeRatio => "ml_ecape_native_cape_ratio",
            Self::MuEcapeNativeCapeRatio => "mu_ecape_native_cape_ratio",
            Self::Sbncape => "sbncape",
            Self::Mlncape => "mlncape",
            Self::Muncape => "muncape",
            Self::Sbcin => "sbcin",
            Self::Mlcin => "mlcin",
            Self::Mucin => "mucin",
            Self::Sbecin => "sbecin",
            Self::Mlecin => "mlecin",
            Self::Muecin => "muecin",
            Self::EcapeCape => "ecape_cape",
            Self::EcapeCin => "ecape_cin",
            Self::Lcl => "lcl",
            Self::Lfc => "lfc",
            Self::El => "el",
            Self::EcapeLfc => "ecape_lfc",
            Self::EcapeEl => "ecape_el",
            Self::Srh01km => "srh1",
            Self::Srh03km => "srh3",
            Self::Stp => "stp",
            Self::StpFixed => "stp_fixed",
            Self::StpEffective => "stp_effective",
            Self::Scp => "scp",
            Self::Ehi => "ehi",
            Self::Tehi => "tehi",
            Self::Tts => "tts",
            Self::VtpMod => "vtp_mod",
            Self::Uh => "uhel",
            Self::EcapeScpExperimental => "ecape_scp",
            Self::EcapeEhi01kmExperimental => "ecape_ehi_0_1km",
            Self::EcapeEhi03kmExperimental => "ecape_ehi_0_3km",
            Self::EcapeStpExperimental => "ecape_stp",
        }
    }

    pub fn display_title(self) -> &'static str {
        match self {
            Self::Sbcape => "SBCAPE",
            Self::Mlcape => "MLCAPE",
            Self::Mucape => "MUCAPE",
            Self::Sbecape => "SBECAPE",
            Self::Mlecape => "MLECAPE",
            Self::Muecape => "MUECAPE",
            Self::SbEcapeDerivedCapeRatio => "SB ECAPE/DERIVED CAPE RATIO",
            Self::MlEcapeDerivedCapeRatio => "ML ECAPE/DERIVED CAPE RATIO",
            Self::MuEcapeDerivedCapeRatio => "MU ECAPE/DERIVED CAPE RATIO",
            Self::SbEcapeNativeCapeRatio => "SB ECAPE/NATIVE CAPE RATIO",
            Self::MlEcapeNativeCapeRatio => "ML ECAPE/NATIVE CAPE RATIO",
            Self::MuEcapeNativeCapeRatio => "MU ECAPE/NATIVE CAPE RATIO",
            Self::Sbncape => "SBNCAPE",
            Self::Mlncape => "MLNCAPE",
            Self::Muncape => "MUNCAPE",
            Self::Sbcin => "SBCIN",
            Self::Mlcin => "MLCIN",
            Self::Mucin => "MUCIN",
            Self::Sbecin => "SBECIN",
            Self::Mlecin => "MLECIN",
            Self::Muecin => "MUECIN",
            Self::EcapeCape => "ECAPE CAPE",
            Self::EcapeCin => "ECAPE CIN",
            Self::Lcl => "LCL",
            Self::Lfc => "LFC",
            Self::El => "EL",
            Self::EcapeLfc => "ECAPE LFC",
            Self::EcapeEl => "ECAPE EL",
            Self::Srh01km => "0-1 KM SRH",
            Self::Srh03km => "0-3 KM SRH",
            Self::Stp => "STP",
            Self::StpFixed => "STP (FIXED)",
            Self::StpEffective => "STP (EFFECTIVE)",
            Self::Scp => "SCP",
            Self::Ehi => "EHI",
            Self::Tehi => "TEHI",
            Self::Tts => "TTS",
            Self::VtpMod => "VTP MOD",
            Self::Uh => "UH",
            Self::EcapeScpExperimental => "ECAPE SCP (EXP)",
            Self::EcapeEhi01kmExperimental => "ECAPE EHI 0-1 KM (EXP)",
            Self::EcapeEhi03kmExperimental => "ECAPE EHI 0-3 KM (EXP)",
            Self::EcapeStpExperimental => "ECAPE STP (EXP)",
        }
    }

    pub fn scale_preset(self) -> WeatherPreset {
        match self {
            Self::Sbcape | Self::Mlcape | Self::Mucape => WeatherPreset::Cape,
            Self::Sbecape | Self::Mlecape | Self::Muecape | Self::EcapeCape => WeatherPreset::Ecape,
            Self::Sbncape | Self::Mlncape | Self::Muncape => WeatherPreset::Ncape,
            Self::SbEcapeDerivedCapeRatio
            | Self::MlEcapeDerivedCapeRatio
            | Self::MuEcapeDerivedCapeRatio
            | Self::SbEcapeNativeCapeRatio
            | Self::MlEcapeNativeCapeRatio
            | Self::MuEcapeNativeCapeRatio => WeatherPreset::EcapeCapeRatio,
            Self::Sbcin
            | Self::Mlcin
            | Self::Mucin
            | Self::Sbecin
            | Self::Mlecin
            | Self::Muecin
            | Self::EcapeCin => WeatherPreset::Cin,
            Self::Lcl => WeatherPreset::Lcl,
            Self::Lfc | Self::EcapeLfc => WeatherPreset::Lfc,
            Self::El | Self::EcapeEl => WeatherPreset::El,
            Self::Srh01km | Self::Srh03km => WeatherPreset::Srh,
            Self::Stp | Self::StpFixed | Self::StpEffective | Self::EcapeStpExperimental => {
                WeatherPreset::Stp
            }
            Self::Scp | Self::EcapeScpExperimental => WeatherPreset::Scp,
            Self::Tehi => WeatherPreset::Tehi,
            Self::Tts => WeatherPreset::Tts,
            Self::VtpMod => WeatherPreset::Vtp,
            Self::Ehi | Self::EcapeEhi01kmExperimental | Self::EcapeEhi03kmExperimental => {
                WeatherPreset::Ehi
            }
            Self::Uh => WeatherPreset::Uh,
        }
    }

    pub fn default_tick_step(self) -> Option<f64> {
        match self.scale_preset() {
            WeatherPreset::Cape => Some(500.0),
            WeatherPreset::Ecape => Some(500.0),
            WeatherPreset::ThreeCape => Some(50.0),
            WeatherPreset::Ncape => Some(250.0),
            WeatherPreset::Cin => Some(50.0),
            WeatherPreset::Lcl => Some(500.0),
            WeatherPreset::Lfc => Some(500.0),
            WeatherPreset::El => Some(1000.0),
            WeatherPreset::Srh => Some(50.0),
            WeatherPreset::Stp => Some(1.0),
            WeatherPreset::Scp => Some(5.0),
            WeatherPreset::Ehi => Some(1.0),
            WeatherPreset::Tehi | WeatherPreset::Tts | WeatherPreset::Vtp => Some(1.0),
            WeatherPreset::EcapeCapeRatio => Some(0.25),
            WeatherPreset::Uh => Some(20.0),
            WeatherPreset::LapseRate => Some(1.0),
        }
    }

    pub fn legend_levels(self) -> Option<Vec<f64>> {
        self.scale_preset().legend_levels()
    }

    pub fn semantics(self) -> ProductSemantics {
        if self.is_experimental() {
            ProductSemantics::experimental()
        } else {
            ProductSemantics::operational()
        }
    }

    pub fn default_visual_mode(self) -> ProductVisualMode {
        ProductVisualMode::SevereDiagnostic
    }

    pub fn is_experimental(self) -> bool {
        matches!(
            self,
            Self::EcapeScpExperimental
                | Self::SbEcapeDerivedCapeRatio
                | Self::MlEcapeDerivedCapeRatio
                | Self::MuEcapeDerivedCapeRatio
                | Self::SbEcapeNativeCapeRatio
                | Self::MlEcapeNativeCapeRatio
                | Self::MuEcapeNativeCapeRatio
                | Self::EcapeEhi01kmExperimental
                | Self::EcapeEhi03kmExperimental
                | Self::EcapeStpExperimental
        )
    }
}

impl From<WeatherProduct> for WeatherPreset {
    fn from(value: WeatherProduct) -> Self {
        value.scale_preset()
    }
}

impl DerivedProductStyle {
    pub fn from_product_name(name: &str) -> Option<Self> {
        match normalize(name).as_str() {
            "lifted_index" | "li" | "surface_based_lifted_index" | "sbli" => {
                Some(Self::LiftedIndex)
            }
            "temperature_advection_700mb" | "temp_advection_700mb" | "tadv700" => {
                Some(Self::TemperatureAdvection700mb)
            }
            "temperature_advection_850mb" | "temp_advection_850mb" | "tadv850" => {
                Some(Self::TemperatureAdvection850mb)
            }
            "bulk_shear_0_1km" | "bulk_shear_01km" | "shear_01km" | "shear01km" => {
                Some(Self::BulkShear01km)
            }
            "bulk_shear_0_6km" | "bulk_shear_06km" | "shear_06km" | "shear06km" => {
                Some(Self::BulkShear06km)
            }
            "apparent_temperature" | "apparent_temp" => Some(Self::ApparentTemperature),
            "heat_index" => Some(Self::HeatIndex),
            "wind_chill" => Some(Self::WindChill),
            _ => None,
        }
    }

    pub fn display_title(self) -> &'static str {
        match self {
            Self::LiftedIndex => "LIFTED INDEX",
            Self::TemperatureAdvection700mb => "700 MB TEMPERATURE ADVECTION",
            Self::TemperatureAdvection850mb => "850 MB TEMPERATURE ADVECTION",
            Self::BulkShear01km => "0-1 KM BULK SHEAR",
            Self::BulkShear06km => "0-6 KM BULK SHEAR",
            Self::ApparentTemperature => "APPARENT TEMPERATURE",
            Self::HeatIndex => "HEAT INDEX",
            Self::WindChill => "WIND CHILL",
        }
    }

    pub fn scale_preset(self) -> DerivedScalePreset {
        match self {
            Self::LiftedIndex => DerivedScalePreset::LiftedIndex,
            Self::TemperatureAdvection700mb | Self::TemperatureAdvection850mb => {
                DerivedScalePreset::TemperatureAdvection
            }
            Self::BulkShear01km | Self::BulkShear06km => DerivedScalePreset::BulkShear,
            Self::ApparentTemperature | Self::HeatIndex | Self::WindChill => {
                DerivedScalePreset::SurfaceComfort
            }
        }
    }

    pub fn scale(self) -> DiscreteColorScale {
        self.scale_preset().scale()
    }

    pub fn default_tick_step(self) -> Option<f64> {
        self.scale_preset().default_tick_step()
    }

    pub fn legend_levels(self) -> Option<Vec<f64>> {
        self.scale_preset().legend_levels()
    }

    pub fn semantics(self) -> ProductSemantics {
        match self {
            Self::ApparentTemperature | Self::HeatIndex | Self::WindChill => {
                ProductSemantics::operational()
            }
            Self::LiftedIndex
            | Self::TemperatureAdvection700mb
            | Self::TemperatureAdvection850mb
            | Self::BulkShear01km
            | Self::BulkShear06km => ProductSemantics::operational(),
        }
    }

    pub fn default_visual_mode(self) -> ProductVisualMode {
        match self {
            Self::TemperatureAdvection700mb | Self::TemperatureAdvection850mb => {
                ProductVisualMode::UpperAirAnalysis
            }
            Self::ApparentTemperature | Self::HeatIndex | Self::WindChill => {
                ProductVisualMode::FilledMeteorology
            }
            Self::LiftedIndex | Self::BulkShear01km | Self::BulkShear06km => {
                ProductVisualMode::SevereDiagnostic
            }
        }
    }
}

impl From<DerivedProductStyle> for DerivedScalePreset {
    fn from(value: DerivedProductStyle) -> Self {
        value.scale_preset()
    }
}

impl WeatherPreset {
    pub fn from_product_name(name: &str) -> Option<Self> {
        if let Some(product) = WeatherProduct::from_product_name(name) {
            return Some(product.scale_preset());
        }

        match normalize(name).as_str() {
            "sbcape" | "mlcape" | "mucape" | "cape" | "effective_cape" => Some(Self::Cape),
            "ecape" | "sbecape" | "mlecape" | "muecape" | "ecape_cape" => Some(Self::Ecape),
            "sb_ecape_derived_cape_ratio"
            | "ml_ecape_derived_cape_ratio"
            | "mu_ecape_derived_cape_ratio"
            | "sb_ecape_native_cape_ratio"
            | "ml_ecape_native_cape_ratio"
            | "mu_ecape_native_cape_ratio"
            | "sbecape_derived_cape_ratio"
            | "mlecape_derived_cape_ratio"
            | "muecape_derived_cape_ratio"
            | "sbecape_native_cape_ratio"
            | "mlecape_native_cape_ratio"
            | "muecape_native_cape_ratio" => Some(Self::EcapeCapeRatio),
            "cape3d" | "three_cape" => Some(Self::ThreeCape),
            "sbncape" | "mlncape" | "muncape" | "ncape" | "normalized_cape" => Some(Self::Ncape),
            "sbcin" | "mlcin" | "mucin" | "cin" | "ecape_cin" | "sbecin" | "mlecin" | "muecin" => {
                Some(Self::Cin)
            }
            "lcl" => Some(Self::Lcl),
            "lfc" | "ecape_lfc" => Some(Self::Lfc),
            "el" | "ecape_el" => Some(Self::El),
            "srh" | "srh1" | "srh3" | "effective_srh" => Some(Self::Srh),
            "stp" | "stp_fixed" | "stp_effective" | "ecape_stp" => Some(Self::Stp),
            "scp" | "ecape_scp" => Some(Self::Scp),
            "ehi" | "ehi_0_1km" | "ehi01km" | "ehi_0_3km" | "ehi03km" | "ecape_ehi"
            | "ecape_ehi_0_1km" | "ecape_ehi_0_3km" => Some(Self::Ehi),
            "tehi" | "tornadic_ehi" | "tornadic_0_1km_ehi" => Some(Self::Tehi),
            "tts" | "tornadic_tilting_stretching" => Some(Self::Tts),
            "vtp_mod" | "modified_vtp" | "vtp" => Some(Self::Vtp),
            "uhel" | "uh" => Some(Self::Uh),
            "lapse_rate" | "lapse_rate_700_500" | "lapse_rate_0_3km" => Some(Self::LapseRate),
            _ => None,
        }
    }

    pub fn legend_levels(self) -> Option<Vec<f64>> {
        match self {
            Self::Cape => Some(vec![
                0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0, 8000.0,
            ]),
            Self::Ecape => Some(vec![
                0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0,
            ]),
            Self::ThreeCape => Some(vec![
                0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0,
            ]),
            Self::Ncape => Some(vec![
                0.0, 50.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0,
            ]),
            Self::Cin => Some(vec![-300.0, -250.0, -200.0, -150.0, -100.0, -50.0, 0.0]),
            Self::Lcl => Some(vec![
                0.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0,
            ]),
            Self::Lfc => Some(vec![
                0.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0, 8000.0,
            ]),
            Self::El => Some(vec![
                4000.0, 6000.0, 8000.0, 10000.0, 12000.0, 14000.0, 16000.0, 18000.0,
            ]),
            Self::Srh => Some(vec![
                0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0, 550.0,
                600.0, 700.0, 800.0, 900.0, 1000.0, 1250.0, 1500.0,
            ]),
            Self::Stp => Some(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0,
            ]),
            Self::Scp => Some(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0,
                70.0,
            ]),
            Self::Ehi => Some(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
                22.0, 24.0,
            ]),
            Self::Tehi | Self::Tts | Self::Vtp => Some(vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
            ]),
            Self::EcapeCapeRatio => Some(vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.1]),
            Self::Uh => Some(vec![
                25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0,
            ]),
            Self::LapseRate => Some(vec![4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0]),
        }
    }

    pub fn mask_below(self) -> Option<f64> {
        match self {
            Self::Cape | Self::Ecape | Self::ThreeCape | Self::Ncape => Some(1.0),
            Self::Srh
            | Self::Stp
            | Self::Scp
            | Self::Ehi
            | Self::Tehi
            | Self::Tts
            | Self::Vtp
            | Self::Uh => Some(0.01),
            _ => None,
        }
    }

    pub fn default_tick_step(self) -> Option<f64> {
        match self {
            Self::Cape | Self::Ecape => Some(500.0),
            Self::ThreeCape => Some(50.0),
            Self::Ncape => Some(250.0),
            Self::Cin => Some(50.0),
            Self::Lcl | Self::Lfc => Some(500.0),
            Self::El => Some(1000.0),
            Self::Srh => Some(50.0),
            Self::Stp | Self::Ehi | Self::Tehi | Self::Tts | Self::Vtp => Some(1.0),
            Self::Scp => Some(5.0),
            Self::EcapeCapeRatio => Some(0.25),
            Self::Uh => Some(20.0),
            Self::LapseRate => Some(1.0),
        }
    }

    pub fn scale(self) -> DiscreteColorScale {
        let mask_below = self.mask_below();
        match self {
            Self::Cape => DiscreteColorScale {
                levels: range_step(0.0, 8100.0, 100.0),
                colors: weather_palette(WeatherPalette::Cape),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Ecape => DiscreteColorScale {
                levels: range_step(0.0, 5000.1, 50.0),
                colors: weather_palette(WeatherPalette::Ecape),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::ThreeCape => DiscreteColorScale {
                levels: concat_ranges(&[(0.0, 300.0, 5.0), (300.0, 501.0, 20.0)]),
                colors: weather_palette(WeatherPalette::ThreeCape),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Ncape => DiscreteColorScale {
                levels: range_step(0.0, 2000.1, 50.0),
                colors: weather_palette(WeatherPalette::Ncape),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Cin => DiscreteColorScale {
                levels: range_step(-300.0, 1.0, 25.0),
                colors: weather_palette(WeatherPalette::Cin),
                extend: ExtendMode::Min,
                mask_below,
            },
            Self::Lcl => DiscreteColorScale {
                levels: range_step(0.0, 4000.1, 250.0),
                colors: weather_palette(WeatherPalette::HeightAgl),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Lfc => DiscreteColorScale {
                levels: range_step(0.0, 8000.1, 500.0),
                colors: weather_palette(WeatherPalette::HeightAgl),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::El => DiscreteColorScale {
                levels: range_step(4000.0, 18000.1, 1000.0),
                colors: weather_palette(WeatherPalette::EquilibriumLevel),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Srh => DiscreteColorScale {
                levels: srh_scale_levels(),
                colors: weather_palette(WeatherPalette::Srh),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Stp => DiscreteColorScale {
                levels: stp_scale_levels(),
                colors: weather_palette(WeatherPalette::Stp),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Scp => DiscreteColorScale {
                levels: range_step(0.0, 70.1, 1.0),
                colors: weather_palette(WeatherPalette::Scp),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Ehi => DiscreteColorScale {
                levels: concat_ranges(&[(0.0, 2.0, 0.1), (2.0, 24.2, 0.2)]),
                colors: weather_palette(WeatherPalette::Ehi),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Tehi => DiscreteColorScale {
                levels: range_step(0.0, 20.1, 0.2),
                colors: weather_palette(WeatherPalette::TornadicEhi),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Tts => DiscreteColorScale {
                levels: range_step(0.0, 20.1, 0.2),
                colors: weather_palette(WeatherPalette::TornadicTts),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Vtp => DiscreteColorScale {
                levels: range_step(0.0, 20.1, 0.2),
                colors: weather_palette(WeatherPalette::ViolentTornado),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::EcapeCapeRatio => DiscreteColorScale {
                levels: range_step(0.0, 1.15, 0.05),
                colors: weather_palette(WeatherPalette::EcapeRatio),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::Uh => DiscreteColorScale {
                levels: concat_ranges(&[(0.0, 200.0, 5.0), (200.0, 401.0, 10.0)]),
                colors: weather_palette(WeatherPalette::Uh),
                extend: ExtendMode::Max,
                mask_below,
            },
            Self::LapseRate => DiscreteColorScale {
                levels: range_step(2.0, 10.1, 0.1),
                colors: weather_palette(WeatherPalette::LapseRate),
                extend: ExtendMode::Both,
                mask_below,
            },
        }
    }
}

impl DerivedScalePreset {
    pub fn legend_levels(self) -> Option<Vec<f64>> {
        match self {
            Self::LiftedIndex => Some(vec![-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0]),
            Self::TemperatureAdvection => {
                Some(vec![-12.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 12.0])
            }
            Self::BulkShear => Some(vec![0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            Self::SurfaceComfort => {
                Some(vec![-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
            }
        }
    }

    pub fn scale(self) -> DiscreteColorScale {
        match self {
            Self::LiftedIndex => {
                let mut colors = weather_palette(WeatherPalette::Advection);
                colors.reverse();
                DiscreteColorScale {
                    levels: range_step(-12.0, 14.0, 2.0),
                    colors,
                    extend: ExtendMode::Both,
                    mask_below: None,
                }
            }
            Self::TemperatureAdvection => DiscreteColorScale {
                levels: range_step(-12.0, 14.0, 2.0),
                colors: weather_palette(WeatherPalette::Advection),
                extend: ExtendMode::Both,
                mask_below: None,
            },
            Self::BulkShear => DiscreteColorScale {
                levels: range_step(0.0, 65.0, 5.0),
                colors: weather_palette(WeatherPalette::Winds),
                extend: ExtendMode::Max,
                mask_below: None,
            },
            Self::SurfaceComfort => DiscreteColorScale {
                levels: range_step(-30.0, 50.0, 5.0),
                colors: weather_palette(WeatherPalette::Temperature),
                extend: ExtendMode::Both,
                mask_below: None,
            },
        }
    }

    pub fn default_tick_step(self) -> Option<f64> {
        match self {
            Self::LiftedIndex => Some(2.0),
            Self::TemperatureAdvection => Some(2.0),
            Self::BulkShear => Some(5.0),
            Self::SurfaceComfort => Some(5.0),
        }
    }
}

pub fn weather_palette(palette: WeatherPalette) -> Vec<Color> {
    use crate::colormaps;

    let colors = match palette {
        WeatherPalette::Cape => colormaps::cape(),
        WeatherPalette::Ecape => ecape_palette(),
        WeatherPalette::ThreeCape => colormaps::three_cape(),
        WeatherPalette::Ncape => ncape_palette(),
        WeatherPalette::Cin => cin_palette(),
        WeatherPalette::Scp => scp_palette(),
        WeatherPalette::Ehi => colormaps::ehi(),
        WeatherPalette::TornadicEhi => tornadic_ehi_palette(),
        WeatherPalette::TornadicTts => tornadic_tts_palette(),
        WeatherPalette::ViolentTornado => violent_tornado_palette(),
        WeatherPalette::Srh => colormaps::srh(),
        WeatherPalette::Stp => colormaps::stp(),
        WeatherPalette::HeightAgl => height_agl_palette(),
        WeatherPalette::EquilibriumLevel => equilibrium_level_palette(),
        WeatherPalette::LapseRate => colormaps::lapse_rate(),
        WeatherPalette::Uh => colormaps::uh(),
        WeatherPalette::EcapeRatio => ecape_ratio_palette(),
        WeatherPalette::MlMetric => colormaps::ml_metric(),
        WeatherPalette::Reflectivity => colormaps::reflectivity(),
        WeatherPalette::Winds => colormaps::winds(60),
        WeatherPalette::Temperature => colormaps::temperature(180),
        WeatherPalette::Dewpoint => colormaps::dewpoint(80, 50),
        WeatherPalette::Rh => colormaps::rh(),
        WeatherPalette::RelVort => colormaps::relvort(100),
        WeatherPalette::Advection => advection_palette(),
        WeatherPalette::SimIr => colormaps::sim_ir(),
        WeatherPalette::GeopotAnomaly => colormaps::geopot_anomaly(100),
        WeatherPalette::Precip => colormaps::precip_in(),
        WeatherPalette::ShadedOverlay => colormaps::shaded_overlay(),
    };

    colors.into_iter().map(Into::into).collect()
}

pub fn winds_palette_segments(n_segments: usize) -> Vec<Color> {
    crate::colormaps::winds(n_segments)
        .into_iter()
        .map(Into::into)
        .collect()
}

pub fn temperature_palette_cropped_f(crop_f: Option<(f64, f64)>, n_segments: usize) -> Vec<Color> {
    crate::colormaps::temperature_cropped(n_segments, crop_f)
        .into_iter()
        .map(Into::into)
        .collect()
}

pub fn dewpoint_palette_params(dry_points: usize, moist_points_total: usize) -> Vec<Color> {
    crate::colormaps::dewpoint(dry_points, moist_points_total)
        .into_iter()
        .map(Into::into)
        .collect()
}

pub fn dewpoint_palette_fahrenheit_for_levels(levels_f: &[f64]) -> Vec<Color> {
    dewpoint_palette_for_levels(levels_f, |value_f| value_f)
}

pub fn dewpoint_palette_celsius_for_levels(levels_c: &[f64]) -> Vec<Color> {
    dewpoint_palette_for_levels(levels_c, |value_c| value_c * 9.0 / 5.0 + 32.0)
}

fn dewpoint_palette_for_levels(levels: &[f64], to_fahrenheit: impl Fn(f64) -> f64) -> Vec<Color> {
    if levels.len() < 2 {
        return Vec::new();
    }

    const REFERENCE_MIN_F: f64 = -40.0;
    const REFERENCE_MAX_F: f64 = 90.0;

    let palette = dewpoint_palette_params(80, 50);
    if palette.is_empty() {
        return Vec::new();
    }

    let span = REFERENCE_MAX_F - REFERENCE_MIN_F;
    levels
        .windows(2)
        .map(|window| {
            let midpoint_f = to_fahrenheit((window[0] + window[1]) * 0.5);
            let scaled = ((midpoint_f - REFERENCE_MIN_F) / span).clamp(0.0, 1.0);
            let index = (scaled * palette.len() as f64).floor() as usize;
            palette[index.min(palette.len() - 1)]
        })
        .collect()
}

pub fn palette_scale(
    palette: WeatherPalette,
    levels: Vec<f64>,
    extend: ExtendMode,
    mask_below: Option<f64>,
) -> DiscreteColorScale {
    DiscreteColorScale {
        levels,
        colors: weather_palette(palette),
        extend,
        mask_below,
    }
}

pub fn stp_scale_levels() -> Vec<f64> {
    concat_ranges(&[
        (0.0, 1.1, 0.1),
        (1.0, 2.1, 0.1),
        (2.0, 3.1, 0.1),
        (3.0, 4.1, 0.1),
        (4.0, 5.1, 0.1),
        (5.0, 6.1, 0.1),
        (6.0, 8.2, 0.2),
        (8.0, 10.2, 0.2),
        (10.0, 15.5, 0.5),
        (15.0, 20.5, 0.5),
    ])
}

pub fn srh_scale_levels() -> Vec<f64> {
    concat_ranges(&[
        (0.0, 150.0, 10.0),
        (150.0, 300.0, 10.0),
        (300.0, 450.0, 10.0),
        (450.0, 600.0, 10.0),
        (600.0, 1000.0, 20.0),
        (1000.0, 1500.1, 50.0),
    ])
}

fn advection_palette() -> Vec<crate::color::Rgba> {
    const ADVECTION_HEX: [&str; 9] = [
        "#0b3c5d", "#328cc1", "#74b3ce", "#d9ecf2", "#f7f7f7", "#f3d9ca", "#e39b7b", "#c75d43",
        "#8f2d1f",
    ];

    ADVECTION_HEX
        .into_iter()
        .map(rgba_from_hex)
        .collect::<Vec<_>>()
}

fn ecape_ratio_palette() -> Vec<crate::color::Rgba> {
    const ECAPE_RATIO_HEX: [&str; 11] = [
        "#7f1d1d", "#b91c1c", "#dc2626", "#f97316", "#f59e0b", "#facc15", "#fde047", "#bef264",
        "#84cc16", "#22c55e", "#15803d",
    ];

    palette_from_hex(&ECAPE_RATIO_HEX)
}

fn ecape_palette() -> Vec<crate::color::Rgba> {
    palette_from_hex(&[
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#31a354", "#fdd049", "#fdae61",
        "#f46d43", "#d73027", "#7f0000", "#4d004b",
    ])
}

fn cin_palette() -> Vec<crate::color::Rgba> {
    palette_from_hex(&[
        "#35004f", "#5f006d", "#8b0f6f", "#b52e57", "#d95f35", "#f29f3d", "#f7d77a",
    ])
}

fn ncape_palette() -> Vec<crate::color::Rgba> {
    palette_from_hex(&[
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#31a354", "#fed976", "#fd8d3c",
        "#bd0026",
    ])
}

fn scp_palette() -> Vec<crate::color::Rgba> {
    palette_from_hex(&[
        "#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#31a354", "#ffd92f", "#fc8d59", "#d7301f",
        "#7f0000", "#54278f",
    ])
}

fn height_agl_palette() -> Vec<crate::color::Rgba> {
    palette_from_hex(&[
        "#ffffe5", "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04",
    ])
}

fn equilibrium_level_palette() -> Vec<crate::color::Rgba> {
    palette_from_hex(&[
        "#f7fcfd", "#e0ecf4", "#bfd3e6", "#9ebcda", "#8c96c6", "#8c6bb1", "#88419d", "#810f7c",
        "#4d004b",
    ])
}

fn tornadic_ehi_palette() -> Vec<crate::color::Rgba> {
    palette_from_hex(&[
        "#fff7bc", "#fee391", "#fec44f", "#fe9929", "#ec7014", "#cc4c02", "#8c2d04", "#4d004b",
    ])
}

fn tornadic_tts_palette() -> Vec<crate::color::Rgba> {
    palette_from_hex(&[
        "#f7fcf5", "#c7e9c0", "#74c476", "#31a354", "#006d2c", "#fdd049", "#f16913", "#a63603",
    ])
}

fn violent_tornado_palette() -> Vec<crate::color::Rgba> {
    palette_from_hex(&[
        "#edf8fb", "#b2e2e2", "#66c2a4", "#238b45", "#fdd049", "#fdae6b", "#e6550d", "#7f2704",
        "#4a1486",
    ])
}

fn palette_from_hex(values: &[&str]) -> Vec<crate::color::Rgba> {
    values.iter().map(|value| rgba_from_hex(value)).collect()
}

fn rgba_from_hex(value: &str) -> crate::color::Rgba {
    let trimmed = value.trim_start_matches('#');
    let red = u8::from_str_radix(&trimmed[0..2], 16).expect("valid red component");
    let green = u8::from_str_radix(&trimmed[2..4], 16).expect("valid green component");
    let blue = u8::from_str_radix(&trimmed[4..6], 16).expect("valid blue component");
    crate::color::Rgba {
        r: red,
        g: green,
        b: blue,
        a: u8::MAX,
    }
}

fn normalize(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn range_step(start: f64, stop: f64, step: f64) -> Vec<f64> {
    let mut values = Vec::new();
    let mut current = start;
    while current < stop - step * 1.0e-9 {
        values.push(current);
        current += step;
    }
    values
}

fn concat_ranges(parts: &[(f64, f64, f64)]) -> Vec<f64> {
    let mut values: Vec<f64> = Vec::new();
    for (start, stop, step) in parts {
        let part = range_step(*start, *stop, *step);
        if let (Some(last), Some(first)) = (values.last().copied(), part.first().copied()) {
            if (last - first).abs() < 1.0e-9 {
                values.extend(part.into_iter().skip(1));
                continue;
            }
        }
        values.extend(part);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{ProductMaturity, ProductSemanticFlag};

    #[test]
    fn explicit_ecape_panel_products_have_expected_titles_and_experimental_flags() {
        assert_eq!(WeatherProduct::Sbecape.display_title(), "SBECAPE");
        assert_eq!(WeatherProduct::Mlecin.display_title(), "MLECIN");
        assert!(WeatherProduct::EcapeScpExperimental.is_experimental());
        assert!(WeatherProduct::EcapeEhi01kmExperimental.is_experimental());
        assert!(WeatherProduct::EcapeEhi03kmExperimental.is_experimental());
        assert!(WeatherProduct::EcapeStpExperimental.is_experimental());
        assert!(WeatherProduct::SbEcapeDerivedCapeRatio.is_experimental());
        assert!(WeatherProduct::SbEcapeNativeCapeRatio.is_experimental());
        assert!(!WeatherProduct::Muecape.is_experimental());
        assert_eq!(
            WeatherProduct::EcapeScpExperimental.semantics().maturity,
            ProductMaturity::Experimental
        );
        assert_eq!(
            WeatherProduct::Sbcape.semantics().maturity,
            ProductMaturity::Operational
        );
    }

    #[test]
    fn ecape_panel_defaults_match_requested_operational_layout() {
        assert_eq!(
            ECAPE_SEVERE_PANEL_PRODUCTS,
            [
                WeatherProduct::Sbecape,
                WeatherProduct::Mlecape,
                WeatherProduct::Muecape,
                WeatherProduct::SbEcapeDerivedCapeRatio,
                WeatherProduct::MlEcapeDerivedCapeRatio,
                WeatherProduct::MuEcapeDerivedCapeRatio,
                WeatherProduct::SbEcapeNativeCapeRatio,
                WeatherProduct::MlEcapeNativeCapeRatio,
                WeatherProduct::MuEcapeNativeCapeRatio,
                WeatherProduct::Sbncape,
                WeatherProduct::Sbecin,
                WeatherProduct::Mlecin,
                WeatherProduct::EcapeScpExperimental,
                WeatherProduct::EcapeEhi01kmExperimental,
                WeatherProduct::EcapeEhi03kmExperimental,
                WeatherProduct::EcapeStpExperimental,
            ]
        );
    }

    #[test]
    fn severe_panel_defaults_cover_classic_severe_suite() {
        assert_eq!(
            SEVERE_CLASSIC_PANEL_PRODUCTS,
            [
                WeatherProduct::Sbcape,
                WeatherProduct::Mlcape,
                WeatherProduct::Mucape,
                WeatherProduct::Mlcin,
                WeatherProduct::Srh01km,
                WeatherProduct::Srh03km,
                WeatherProduct::Stp,
                WeatherProduct::Scp,
            ]
        );
    }

    #[test]
    fn product_name_resolution_covers_parcel_explicit_ecape_fields() {
        assert_eq!(
            WeatherProduct::from_product_name("mlecin"),
            Some(WeatherProduct::Mlecin)
        );
        assert_eq!(
            WeatherProduct::from_product_name("ecape_scp"),
            Some(WeatherProduct::EcapeScpExperimental)
        );
        assert_eq!(
            WeatherProduct::from_product_name("sb_ecape_derived_cape_ratio"),
            Some(WeatherProduct::SbEcapeDerivedCapeRatio)
        );
        assert_eq!(
            WeatherProduct::from_product_name("ml_ecape_native_cape_ratio"),
            Some(WeatherProduct::MlEcapeNativeCapeRatio)
        );
        assert_eq!(
            WeatherPreset::from_product_name("mu_ecape_native_cape_ratio"),
            Some(WeatherPreset::EcapeCapeRatio)
        );
        assert_eq!(
            WeatherPreset::from_product_name("ncape"),
            Some(WeatherPreset::Ncape)
        );
        assert_eq!(WeatherProduct::Sbncape.scale_preset(), WeatherPreset::Ncape);
        assert_eq!(
            WeatherProduct::from_product_name("ecape_ehi"),
            Some(WeatherProduct::EcapeEhi01kmExperimental)
        );
        assert_eq!(
            WeatherProduct::from_product_name("ecape_ehi_0_1km"),
            Some(WeatherProduct::EcapeEhi01kmExperimental)
        );
        assert_eq!(
            WeatherProduct::from_product_name("ecape_ehi_0_3km"),
            Some(WeatherProduct::EcapeEhi03kmExperimental)
        );
        assert_eq!(
            WeatherProduct::from_product_name("vtp_mod"),
            Some(WeatherProduct::VtpMod)
        );
        assert_eq!(
            WeatherPreset::from_product_name("ecape_ehi_0_3km"),
            Some(WeatherPreset::Ehi)
        );
        assert_eq!(
            WeatherPreset::from_product_name("vtp_mod"),
            Some(WeatherPreset::Vtp)
        );
    }

    #[test]
    fn palette_scale_wraps_palette_and_levels_into_discrete_scale() {
        let scale = palette_scale(
            WeatherPalette::Reflectivity,
            vec![5.0, 15.0, 25.0, 35.0],
            ExtendMode::Max,
            Some(5.0),
        );

        assert_eq!(scale.levels, vec![5.0, 15.0, 25.0, 35.0]);
        assert_eq!(scale.extend, ExtendMode::Max);
        assert_eq!(scale.mask_below, Some(5.0));
        assert!(!scale.colors.is_empty());
    }

    #[test]
    fn derived_product_styles_cover_new_helper_tranche() {
        assert_eq!(
            DerivedProductStyle::from_product_name("lifted_index"),
            Some(DerivedProductStyle::LiftedIndex)
        );
        assert_eq!(
            DerivedProductStyle::from_product_name("temperature_advection_850mb"),
            Some(DerivedProductStyle::TemperatureAdvection850mb)
        );
        assert_eq!(
            DerivedProductStyle::from_product_name("bulk_shear_0_6km"),
            Some(DerivedProductStyle::BulkShear06km)
        );
        assert_eq!(
            DerivedProductStyle::from_product_name("apparent_temperature"),
            Some(DerivedProductStyle::ApparentTemperature)
        );
    }

    #[test]
    fn lifted_index_and_advection_scales_use_diverging_advection_helper() {
        let li = DerivedScalePreset::LiftedIndex.scale();
        let advection = DerivedScalePreset::TemperatureAdvection.scale();

        assert_eq!(li.levels, range_step(-12.0, 14.0, 2.0));
        assert_eq!(advection.levels, range_step(-12.0, 14.0, 2.0));
        assert_eq!(li.extend, ExtendMode::Both);
        assert_eq!(advection.extend, ExtendMode::Both);
        assert_eq!(li.colors.first(), advection.colors.last());
        assert_eq!(li.colors.last(), advection.colors.first());
    }

    #[test]
    fn severe_reference_scales_match_upstream_wrf_runner_bins() {
        assert_eq!(
            WeatherPreset::Cape.scale().levels,
            range_step(0.0, 8100.0, 100.0)
        );
        assert_eq!(
            WeatherPreset::ThreeCape.scale().levels,
            concat_ranges(&[(0.0, 300.0, 5.0), (300.0, 501.0, 20.0)])
        );
        assert_eq!(
            WeatherPreset::Ncape.scale().levels,
            range_step(0.0, 2000.1, 50.0)
        );
        assert_eq!(
            WeatherPreset::Cin.scale().levels,
            range_step(-300.0, 1.0, 25.0)
        );
        assert_eq!(
            WeatherPreset::Lcl.scale().levels,
            range_step(0.0, 4000.1, 250.0)
        );
        assert_eq!(
            WeatherPreset::Lfc.scale().levels,
            range_step(0.0, 8000.1, 500.0)
        );
        assert_eq!(
            WeatherPreset::El.scale().levels,
            range_step(4000.0, 18000.1, 1000.0)
        );
        assert_eq!(WeatherPreset::Srh.scale().levels, srh_scale_levels());
        assert_eq!(WeatherPreset::Stp.scale().levels, stp_scale_levels());
        assert_eq!(
            WeatherPreset::Scp.scale().levels,
            range_step(0.0, 70.1, 1.0)
        );
        assert_eq!(
            WeatherPreset::Ehi.scale().levels,
            concat_ranges(&[(0.0, 2.0, 0.1), (2.0, 24.2, 0.2)])
        );
        for preset in [WeatherPreset::Tehi, WeatherPreset::Tts, WeatherPreset::Vtp] {
            assert_eq!(preset.scale().levels, range_step(0.0, 20.1, 0.2));
        }
        assert_eq!(
            WeatherPreset::EcapeCapeRatio.scale().levels,
            range_step(0.0, 1.15, 0.05)
        );
        assert_eq!(
            WeatherPreset::Uh.scale().levels,
            concat_ranges(&[(0.0, 200.0, 5.0), (200.0, 401.0, 10.0)])
        );
        assert_eq!(
            WeatherPreset::LapseRate.scale().levels,
            range_step(2.0, 10.1, 0.1)
        );
    }

    #[test]
    fn weather_presets_expose_operational_legend_thresholds_and_masks() {
        for preset in [
            WeatherPreset::Cape,
            WeatherPreset::Ecape,
            WeatherPreset::ThreeCape,
            WeatherPreset::Ncape,
            WeatherPreset::Cin,
            WeatherPreset::Lcl,
            WeatherPreset::Lfc,
            WeatherPreset::El,
            WeatherPreset::Srh,
            WeatherPreset::Stp,
            WeatherPreset::Scp,
            WeatherPreset::Ehi,
            WeatherPreset::Tehi,
            WeatherPreset::Tts,
            WeatherPreset::Vtp,
            WeatherPreset::EcapeCapeRatio,
            WeatherPreset::Uh,
            WeatherPreset::LapseRate,
        ] {
            let scale = preset.scale();
            let legend = preset
                .legend_levels()
                .unwrap_or_else(|| panic!("{preset:?} should expose operational legend levels"));
            assert!(
                legend.len() >= 2,
                "{preset:?} should expose at least one legend interval"
            );
            assert!(
                scale.levels.len() >= legend.len(),
                "{preset:?} should keep dense fill levels separate from sparse legend thresholds"
            );
            assert!(
                *legend.first().unwrap() + 1.0e-6 >= *scale.levels.first().unwrap(),
                "{preset:?} legend should start inside the fill scale"
            );
            assert!(
                *legend.last().unwrap() <= *scale.levels.last().unwrap() + 1.0e-6,
                "{preset:?} legend should end inside the fill scale"
            );
        }

        assert_eq!(WeatherPreset::Cape.scale().mask_below, Some(1.0));
        assert_eq!(WeatherPreset::Ecape.scale().mask_below, Some(1.0));
        assert_eq!(WeatherPreset::Ncape.scale().mask_below, Some(1.0));
        assert_eq!(WeatherPreset::Stp.scale().mask_below, Some(0.01));
        assert_eq!(WeatherPreset::Uh.scale().mask_below, Some(0.01));
        assert_eq!(WeatherPreset::Cin.scale().mask_below, None);
    }

    #[test]
    fn renderer_weather_presets_do_not_borrow_generic_severe_palettes() {
        for preset in [
            WeatherPreset::Ecape,
            WeatherPreset::Cin,
            WeatherPreset::Ncape,
            WeatherPreset::Lcl,
            WeatherPreset::Lfc,
            WeatherPreset::El,
            WeatherPreset::Scp,
            WeatherPreset::Tehi,
            WeatherPreset::Tts,
            WeatherPreset::Vtp,
        ] {
            let scale = preset.scale();
            assert_ne!(
                scale.colors,
                weather_palette(WeatherPalette::Cape),
                "{preset:?} should not borrow the CAPE palette"
            );
            assert_ne!(
                scale.colors,
                weather_palette(WeatherPalette::Stp),
                "{preset:?} should not borrow the STP palette"
            );
            assert!(
                scale.colors.len() >= 2,
                "{preset:?} should use a visible operational palette"
            );
        }

        assert_eq!(WeatherProduct::Tehi.scale_preset(), WeatherPreset::Tehi);
        assert_eq!(WeatherProduct::Tts.scale_preset(), WeatherPreset::Tts);
        assert_eq!(WeatherProduct::VtpMod.scale_preset(), WeatherPreset::Vtp);
        assert_eq!(WeatherProduct::Sbecape.scale_preset(), WeatherPreset::Ecape);
        assert_eq!(WeatherProduct::Sbncape.scale_preset(), WeatherPreset::Ncape);
        assert_eq!(WeatherProduct::Scp.default_tick_step(), Some(5.0));
    }

    #[test]
    fn srh_and_ehi_palettes_finish_with_blue_high_end() {
        let srh = weather_palette(WeatherPalette::Srh);
        let srh_top = srh.last().unwrap();
        assert!(srh_top.b > srh_top.r);
        assert!(srh_top.g > srh_top.r);

        let ehi = weather_palette(WeatherPalette::Ehi);
        let ehi_top = ehi.last().unwrap();
        assert!(ehi_top.b > ehi_top.r);
        assert!(ehi_top.g > ehi_top.r);
    }

    #[test]
    fn bulk_shear_and_surface_comfort_have_sane_tick_steps() {
        assert_eq!(DerivedScalePreset::BulkShear.default_tick_step(), Some(5.0));
        assert_eq!(
            DerivedProductStyle::ApparentTemperature.default_tick_step(),
            Some(5.0)
        );
        assert_eq!(
            DerivedProductStyle::TemperatureAdvection700mb.display_title(),
            "700 MB TEMPERATURE ADVECTION"
        );
    }

    #[test]
    fn semantic_flags_stay_narrow_in_render_presets() {
        let severe = WeatherProduct::Scp.semantics();
        assert_eq!(severe.maturity, ProductMaturity::Operational);
        assert!(!severe.has_flag(ProductSemanticFlag::Proxy));

        let ecape_01km = WeatherProduct::EcapeEhi01kmExperimental.semantics();
        assert_eq!(ecape_01km.maturity, ProductMaturity::Experimental);
        assert!(!ecape_01km.has_flag(ProductSemanticFlag::ProofOriented));

        let ecape_03km = WeatherProduct::EcapeEhi03kmExperimental.semantics();
        assert_eq!(ecape_03km.maturity, ProductMaturity::Experimental);
        assert!(!ecape_03km.has_flag(ProductSemanticFlag::ProofOriented));
    }
}

use crate::palette::CrossSectionPalette;
use crate::render::CrossSectionRenderRequest;

/// High-level product groupings aligned to the reference style guide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrossSectionProductGroup {
    TemperatureMoisture,
    WindDynamics,
    CloudsPrecip,
    HazardsComposites,
}

impl CrossSectionProductGroup {
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::TemperatureMoisture => "Temperature & Moisture",
            Self::WindDynamics => "Wind & Dynamics",
            Self::CloudsPrecip => "Clouds & Precip",
            Self::HazardsComposites => "Hazards & Composites",
        }
    }
}

/// Cross-section product catalog informed by `wxsection_ref`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrossSectionProduct {
    Temperature,
    WindSpeed,
    ThetaE,
    RelativeHumidity,
    SpecificHumidity,
    Omega,
    Vorticity,
    Shear,
    LapseRate,
    CloudWater,
    TotalCondensate,
    WetBulb,
    Icing,
    Frontogenesis,
    Smoke,
    VaporPressureDeficit,
    DewpointDepression,
    MoistureTransport,
    PotentialVorticity,
    FireWeather,
}

/// Full cross-section product catalog exported by the crate.
pub const ALL_CROSS_SECTION_PRODUCTS: [CrossSectionProduct; 20] = [
    CrossSectionProduct::Temperature,
    CrossSectionProduct::WindSpeed,
    CrossSectionProduct::ThetaE,
    CrossSectionProduct::RelativeHumidity,
    CrossSectionProduct::SpecificHumidity,
    CrossSectionProduct::Omega,
    CrossSectionProduct::Vorticity,
    CrossSectionProduct::Shear,
    CrossSectionProduct::LapseRate,
    CrossSectionProduct::CloudWater,
    CrossSectionProduct::TotalCondensate,
    CrossSectionProduct::WetBulb,
    CrossSectionProduct::Icing,
    CrossSectionProduct::Frontogenesis,
    CrossSectionProduct::Smoke,
    CrossSectionProduct::VaporPressureDeficit,
    CrossSectionProduct::DewpointDepression,
    CrossSectionProduct::MoistureTransport,
    CrossSectionProduct::PotentialVorticity,
    CrossSectionProduct::FireWeather,
];

impl CrossSectionProduct {
    pub fn from_name(name: &str) -> Option<Self> {
        match normalize(name).as_str() {
            "temperature" | "temp" => Some(Self::Temperature),
            "wind_speed" | "wind" => Some(Self::WindSpeed),
            "theta_e" => Some(Self::ThetaE),
            "relative_humidity" | "rh" => Some(Self::RelativeHumidity),
            "specific_humidity" | "q" => Some(Self::SpecificHumidity),
            "omega" | "vertical_velocity" => Some(Self::Omega),
            "vorticity" => Some(Self::Vorticity),
            "shear" | "wind_shear" => Some(Self::Shear),
            "lapse_rate" => Some(Self::LapseRate),
            "cloud" | "cloud_water" => Some(Self::CloudWater),
            "cloud_total" | "total_condensate" => Some(Self::TotalCondensate),
            "wetbulb" | "wet_bulb" => Some(Self::WetBulb),
            "icing" => Some(Self::Icing),
            "frontogenesis" => Some(Self::Frontogenesis),
            "smoke" => Some(Self::Smoke),
            "vpd" | "vapor_pressure_deficit" => Some(Self::VaporPressureDeficit),
            "dewpoint_dep" | "dewpoint_depression" => Some(Self::DewpointDepression),
            "moisture_transport" => Some(Self::MoistureTransport),
            "pv" | "potential_vorticity" => Some(Self::PotentialVorticity),
            "fire_wx" | "fire_weather" => Some(Self::FireWeather),
            _ => None,
        }
    }

    /// Public-facing product key.
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Temperature => "temperature",
            Self::WindSpeed => "wind_speed",
            Self::ThetaE => "theta_e",
            Self::RelativeHumidity => "rh",
            Self::SpecificHumidity => "q",
            Self::Omega => "omega",
            Self::Vorticity => "vorticity",
            Self::Shear => "shear",
            Self::LapseRate => "lapse_rate",
            Self::CloudWater => "cloud",
            Self::TotalCondensate => "cloud_total",
            Self::WetBulb => "wetbulb",
            Self::Icing => "icing",
            Self::Frontogenesis => "frontogenesis",
            Self::Smoke => "smoke",
            Self::VaporPressureDeficit => "vpd",
            Self::DewpointDepression => "dewpoint_dep",
            Self::MoistureTransport => "moisture_transport",
            Self::PotentialVorticity => "pv",
            Self::FireWeather => "fire_wx",
        }
    }

    /// Internal reference style key.
    pub const fn style_key(self) -> &'static str {
        match self {
            Self::Temperature => "temp",
            Self::WindSpeed => "wind_speed",
            Self::ThetaE => "theta_e",
            Self::RelativeHumidity => "rh",
            Self::SpecificHumidity => "q",
            Self::Omega => "omega",
            Self::Vorticity => "vorticity",
            Self::Shear => "shear",
            Self::LapseRate => "lapse_rate",
            Self::CloudWater => "cloud",
            Self::TotalCondensate => "cloud_total",
            Self::WetBulb => "wetbulb",
            Self::Icing => "icing",
            Self::Frontogenesis => "frontogenesis",
            Self::Smoke => "smoke",
            Self::VaporPressureDeficit => "vpd",
            Self::DewpointDepression => "dewpoint_dep",
            Self::MoistureTransport => "moisture_transport",
            Self::PotentialVorticity => "pv",
            Self::FireWeather => "fire_wx",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Temperature => "Temperature",
            Self::WindSpeed => "Wind Speed",
            Self::ThetaE => "Equivalent Potential Temperature",
            Self::RelativeHumidity => "Relative Humidity",
            Self::SpecificHumidity => "Specific Humidity",
            Self::Omega => "Vertical Velocity",
            Self::Vorticity => "Absolute Vorticity",
            Self::Shear => "Wind Shear",
            Self::LapseRate => "Lapse Rate",
            Self::CloudWater => "Cloud Water",
            Self::TotalCondensate => "Total Condensate",
            Self::WetBulb => "Wet-Bulb Temperature",
            Self::Icing => "Icing Potential",
            Self::Frontogenesis => "Frontogenesis",
            Self::Smoke => "PM2.5 Smoke",
            Self::VaporPressureDeficit => "Vapor Pressure Deficit",
            Self::DewpointDepression => "Dewpoint Depression",
            Self::MoistureTransport => "Moisture Transport",
            Self::PotentialVorticity => "Potential Vorticity",
            Self::FireWeather => "Fire Weather",
        }
    }

    pub const fn units(self) -> &'static str {
        match self {
            Self::Temperature | Self::WetBulb | Self::DewpointDepression => "C",
            Self::WindSpeed => "kt",
            Self::ThetaE => "K",
            Self::RelativeHumidity => "%",
            Self::SpecificHumidity | Self::CloudWater | Self::TotalCondensate | Self::Icing => {
                "g/kg"
            }
            Self::Omega => "hPa/hr",
            Self::Vorticity => "1e-5 s^-1",
            Self::Shear => "1e-3 s^-1",
            Self::LapseRate => "C/km",
            Self::Frontogenesis => "K/100km/3hr",
            Self::Smoke => "ug/m^3",
            Self::VaporPressureDeficit => "hPa",
            Self::MoistureTransport => "g*m/kg/s",
            Self::PotentialVorticity => "PVU",
            Self::FireWeather => "RH% + wind",
        }
    }

    pub const fn group(self) -> CrossSectionProductGroup {
        match self {
            Self::Temperature
            | Self::ThetaE
            | Self::RelativeHumidity
            | Self::SpecificHumidity
            | Self::WetBulb
            | Self::VaporPressureDeficit
            | Self::DewpointDepression => CrossSectionProductGroup::TemperatureMoisture,
            Self::WindSpeed
            | Self::Omega
            | Self::Vorticity
            | Self::Shear
            | Self::MoistureTransport
            | Self::PotentialVorticity => CrossSectionProductGroup::WindDynamics,
            Self::CloudWater
            | Self::TotalCondensate
            | Self::Icing
            | Self::LapseRate
            | Self::Frontogenesis => CrossSectionProductGroup::CloudsPrecip,
            Self::Smoke | Self::FireWeather => CrossSectionProductGroup::HazardsComposites,
        }
    }

    pub const fn supports_anomaly(self) -> bool {
        matches!(
            self,
            Self::Temperature
                | Self::WindSpeed
                | Self::ThetaE
                | Self::RelativeHumidity
                | Self::SpecificHumidity
                | Self::Omega
                | Self::Vorticity
                | Self::Shear
                | Self::LapseRate
                | Self::WetBulb
                | Self::VaporPressureDeficit
                | Self::DewpointDepression
                | Self::MoistureTransport
        )
    }

    pub const fn default_palette(self) -> CrossSectionPalette {
        match self {
            Self::Temperature => CrossSectionPalette::TemperatureStandard,
            Self::WindSpeed => CrossSectionPalette::WindSpeed,
            Self::ThetaE => CrossSectionPalette::ThetaE,
            Self::RelativeHumidity => CrossSectionPalette::RelativeHumidity,
            Self::SpecificHumidity => CrossSectionPalette::SpecificHumidity,
            Self::Omega => CrossSectionPalette::Omega,
            Self::Vorticity => CrossSectionPalette::Vorticity,
            Self::Shear => CrossSectionPalette::Shear,
            Self::LapseRate => CrossSectionPalette::LapseRate,
            Self::CloudWater => CrossSectionPalette::CloudWater,
            Self::TotalCondensate => CrossSectionPalette::TotalCondensate,
            Self::WetBulb => CrossSectionPalette::WetBulb,
            Self::Icing => CrossSectionPalette::Icing,
            Self::Frontogenesis => CrossSectionPalette::Frontogenesis,
            Self::Smoke => CrossSectionPalette::Smoke,
            Self::VaporPressureDeficit => CrossSectionPalette::VaporPressureDeficit,
            Self::DewpointDepression => CrossSectionPalette::DewpointDepression,
            Self::MoistureTransport => CrossSectionPalette::MoistureTransport,
            Self::PotentialVorticity => CrossSectionPalette::PotentialVorticity,
            Self::FireWeather => CrossSectionPalette::FireWeather,
        }
    }

    pub const fn default_value_range(self) -> Option<(f32, f32)> {
        match self {
            Self::Temperature => Some((-66.0, 54.0)),
            Self::WindSpeed => Some((0.0, 100.0)),
            Self::ThetaE => Some((280.0, 360.0)),
            Self::RelativeHumidity => Some((0.0, 100.0)),
            Self::SpecificHumidity => Some((0.0, 20.0)),
            Self::Omega => Some((-20.0, 20.0)),
            Self::Vorticity => Some((-30.0, 30.0)),
            Self::Shear => Some((0.0, 10.0)),
            Self::LapseRate => Some((0.0, 12.0)),
            Self::CloudWater => Some((0.0, 0.5)),
            Self::TotalCondensate => Some((0.0, 1.0)),
            Self::WetBulb => Some((-40.0, 30.0)),
            Self::Icing => Some((0.0, 0.3)),
            Self::Frontogenesis => Some((-2.0, 2.0)),
            Self::Smoke => None,
            Self::VaporPressureDeficit => Some((0.0, 10.0)),
            Self::DewpointDepression => Some((0.0, 40.0)),
            Self::MoistureTransport => Some((0.0, 200.0)),
            Self::PotentialVorticity => Some((-2.0, 10.0)),
            Self::FireWeather => Some((0.0, 100.0)),
        }
    }

    pub fn default_value_ticks(self) -> &'static [f32] {
        match self {
            Self::Temperature => &TEMPERATURE_TICKS,
            Self::WindSpeed => &WIND_SPEED_TICKS,
            Self::ThetaE => &THETA_E_TICKS,
            Self::RelativeHumidity => &HUMIDITY_TICKS,
            Self::SpecificHumidity => &SPECIFIC_HUMIDITY_TICKS,
            Self::Omega => &OMEGA_TICKS,
            Self::Vorticity => &VORTICITY_TICKS,
            Self::Shear => &SHEAR_TICKS,
            Self::LapseRate => &LAPSE_RATE_TICKS,
            Self::CloudWater => &CLOUD_WATER_TICKS,
            Self::TotalCondensate => &TOTAL_CONDENSATE_TICKS,
            Self::WetBulb => &WET_BULB_TICKS,
            Self::Icing => &ICING_TICKS,
            Self::Frontogenesis => &FRONTOGENESIS_TICKS,
            Self::Smoke => &SMOKE_TICKS,
            Self::VaporPressureDeficit => &VPD_TICKS,
            Self::DewpointDepression => &DEWPOINT_DEPRESSION_TICKS,
            Self::MoistureTransport => &MOISTURE_TRANSPORT_TICKS,
            Self::PotentialVorticity => &POTENTIAL_VORTICITY_TICKS,
            Self::FireWeather => &HUMIDITY_TICKS,
        }
    }

    pub const fn default_colorbar_label(self) -> &'static str {
        match self {
            Self::Temperature => "Temperature (C)",
            Self::WindSpeed => "Wind Speed (kt)",
            Self::ThetaE => "Theta-E (K)",
            Self::RelativeHumidity => "Relative Humidity (%)",
            Self::SpecificHumidity => "Specific Humidity (g/kg)",
            Self::Omega => "Omega (hPa/hr)",
            Self::Vorticity => "Vorticity (1e-5 s^-1)",
            Self::Shear => "Shear (1e-3 s^-1)",
            Self::LapseRate => "Lapse Rate (C/km)",
            Self::CloudWater => "Cloud Water (g/kg)",
            Self::TotalCondensate => "Total Condensate (g/kg)",
            Self::WetBulb => "Wet-Bulb Temp (C)",
            Self::Icing => "Icing (SLW g/kg)",
            Self::Frontogenesis => "Frontogenesis (K/100km/3hr)",
            Self::Smoke => "PM2.5 (ug/m^3)",
            Self::VaporPressureDeficit => "VPD (hPa)",
            Self::DewpointDepression => "Dewpoint Depression (C)",
            Self::MoistureTransport => "Moisture Transport (g*m/kg/s)",
            Self::PotentialVorticity => "Potential Vorticity (PVU)",
            Self::FireWeather => "Relative Humidity (%)",
        }
    }

    pub fn default_isotherms_c(self) -> &'static [f32] {
        match self {
            Self::Temperature => &TEMPERATURE_ISOTHERMS,
            Self::WetBulb => &WET_BULB_ISOTHERMS,
            _ => &NO_ISOTHERMS,
        }
    }

    pub const fn default_highlight_isotherm_c(self) -> Option<f32> {
        match self {
            Self::Temperature | Self::WetBulb => Some(0.0),
            _ => None,
        }
    }

    pub fn default_style(self) -> CrossSectionStyle {
        CrossSectionStyle::new(self)
    }
}

/// Small, composable render preset for a cross-section product.
#[derive(Debug, Clone, PartialEq)]
pub struct CrossSectionStyle {
    product: CrossSectionProduct,
    palette: CrossSectionPalette,
    value_range: Option<(f32, f32)>,
    value_ticks: Vec<f32>,
    colorbar_label: Option<String>,
    isotherms_c: Vec<f32>,
    highlight_isotherm_c: Option<f32>,
}

impl CrossSectionStyle {
    pub fn new(product: CrossSectionProduct) -> Self {
        Self {
            product,
            palette: product.default_palette(),
            value_range: product.default_value_range(),
            value_ticks: product.default_value_ticks().to_vec(),
            colorbar_label: Some(product.default_colorbar_label().to_string()),
            isotherms_c: product.default_isotherms_c().to_vec(),
            highlight_isotherm_c: product.default_highlight_isotherm_c(),
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Some(Self::new(CrossSectionProduct::from_name(name)?))
    }

    pub const fn product(&self) -> CrossSectionProduct {
        self.product
    }

    pub const fn palette(&self) -> CrossSectionPalette {
        self.palette
    }

    pub const fn value_range(&self) -> Option<(f32, f32)> {
        self.value_range
    }

    pub fn value_ticks(&self) -> &[f32] {
        &self.value_ticks
    }

    pub fn colorbar_label(&self) -> Option<&str> {
        self.colorbar_label.as_deref()
    }

    pub fn isotherms_c(&self) -> &[f32] {
        &self.isotherms_c
    }

    pub const fn highlight_isotherm_c(&self) -> Option<f32> {
        self.highlight_isotherm_c
    }

    pub fn with_palette(mut self, palette: CrossSectionPalette) -> Self {
        self.palette = palette;
        self
    }

    pub fn with_value_range(mut self, min_value: f32, max_value: f32) -> Self {
        self.value_range = Some((min_value, max_value));
        self
    }

    pub fn without_value_range(mut self) -> Self {
        self.value_range = None;
        self
    }

    pub fn with_value_ticks(mut self, ticks: Vec<f32>) -> Self {
        self.value_ticks = ticks;
        self
    }

    pub fn with_colorbar_label(mut self, label: impl Into<String>) -> Self {
        self.colorbar_label = Some(label.into());
        self
    }

    pub fn without_colorbar_label(mut self) -> Self {
        self.colorbar_label = None;
        self
    }

    pub fn with_isotherms(mut self, levels_c: Vec<f32>, highlight_c: Option<f32>) -> Self {
        self.isotherms_c = levels_c;
        self.highlight_isotherm_c = highlight_c;
        self
    }

    pub fn without_isotherms(mut self) -> Self {
        self.isotherms_c.clear();
        self.highlight_isotherm_c = None;
        self
    }

    pub fn apply_to_request(&self, request: &mut CrossSectionRenderRequest) {
        request.palette = self.palette.build();
        request.value_range = self.value_range;
        request.value_ticks = self.value_ticks.clone();
        request.colorbar_label = self.colorbar_label.clone();
        request.isotherms_c = self.isotherms_c.clone();
        request.highlight_isotherm_c = self.highlight_isotherm_c;
    }

    pub fn to_render_request(&self) -> CrossSectionRenderRequest {
        let mut request = CrossSectionRenderRequest::default();
        self.apply_to_request(&mut request);
        request
    }
}

impl Default for CrossSectionStyle {
    fn default() -> Self {
        Self::new(CrossSectionProduct::Temperature)
    }
}

impl CrossSectionRenderRequest {
    pub fn with_style(mut self, style: CrossSectionStyle) -> Self {
        style.apply_to_request(&mut self);
        self
    }
}

const NO_ISOTHERMS: [f32; 0] = [];
const TEMPERATURE_ISOTHERMS: [f32; 5] = [-30.0, -20.0, -10.0, 0.0, 10.0];
const WET_BULB_ISOTHERMS: [f32; 3] = [-20.0, -10.0, 0.0];

const TEMPERATURE_TICKS: [f32; 6] = [-60.0, -40.0, -20.0, 0.0, 20.0, 40.0];
const WIND_SPEED_TICKS: [f32; 8] = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 90.0];
const THETA_E_TICKS: [f32; 5] = [280.0, 300.0, 320.0, 340.0, 360.0];
const HUMIDITY_TICKS: [f32; 6] = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0];
const SPECIFIC_HUMIDITY_TICKS: [f32; 5] = [0.0, 5.0, 10.0, 15.0, 20.0];
const OMEGA_TICKS: [f32; 9] = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0];
const VORTICITY_TICKS: [f32; 7] = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0];
const SHEAR_TICKS: [f32; 6] = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
const LAPSE_RATE_TICKS: [f32; 7] = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0];
const CLOUD_WATER_TICKS: [f32; 6] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
const TOTAL_CONDENSATE_TICKS: [f32; 6] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
const WET_BULB_TICKS: [f32; 8] = [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0];
const ICING_TICKS: [f32; 4] = [0.0, 0.1, 0.2, 0.3];
const FRONTOGENESIS_TICKS: [f32; 5] = [-2.0, -1.0, 0.0, 1.0, 2.0];
const SMOKE_TICKS: [f32; 0] = [];
const VPD_TICKS: [f32; 6] = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0];
const DEWPOINT_DEPRESSION_TICKS: [f32; 6] = [0.0, 8.0, 16.0, 24.0, 32.0, 40.0];
const MOISTURE_TRANSPORT_TICKS: [f32; 5] = [0.0, 50.0, 100.0, 150.0, 200.0];
const POTENTIAL_VORTICITY_TICKS: [f32; 7] = [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0];

fn normalize(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::Color;

    #[test]
    fn product_resolution_accepts_reference_style_and_product_aliases() {
        assert_eq!(
            CrossSectionProduct::from_name("temperature"),
            Some(CrossSectionProduct::Temperature)
        );
        assert_eq!(
            CrossSectionProduct::from_name("temp"),
            Some(CrossSectionProduct::Temperature)
        );
        assert_eq!(
            CrossSectionProduct::from_name("relative_humidity"),
            Some(CrossSectionProduct::RelativeHumidity)
        );
        assert_eq!(
            CrossSectionProduct::from_name("fire-weather"),
            Some(CrossSectionProduct::FireWeather)
        );
    }

    #[test]
    fn non_temperature_style_clears_default_temperature_isotherms() {
        let request = CrossSectionRenderRequest::default().with_style(CrossSectionStyle::new(
            CrossSectionProduct::RelativeHumidity,
        ));

        assert_eq!(
            request.colorbar_label.as_deref(),
            Some("Relative Humidity (%)")
        );
        assert_eq!(request.value_range, Some((0.0, 100.0)));
        assert!(request.isotherms_c.is_empty());
        assert_eq!(request.highlight_isotherm_c, None);
    }

    #[test]
    fn temperature_style_can_swap_named_palette_and_keep_freezing_highlight() {
        let request = CrossSectionStyle::new(CrossSectionProduct::Temperature)
            .with_palette(CrossSectionPalette::TemperatureNwsNdfd)
            .to_render_request();

        assert_eq!(request.highlight_isotherm_c, Some(0.0));
        assert_eq!(request.isotherms_c, vec![-30.0, -20.0, -10.0, 0.0, 10.0]);
        assert_eq!(request.palette.len(), 33);
        assert_eq!(request.palette.first(), Some(&Color::rgb(75, 0, 130)));
    }

    #[test]
    fn product_catalog_exposes_group_and_anomaly_metadata() {
        assert_eq!(
            CrossSectionProduct::MoistureTransport.group(),
            CrossSectionProductGroup::WindDynamics
        );
        assert!(CrossSectionProduct::WetBulb.supports_anomaly());
        assert!(!CrossSectionProduct::Smoke.supports_anomaly());
    }
}

use crate::render::{Color, CrossSectionRenderRequest};

/// Named palettes for cross-section scalar rendering.
///
/// The temperature variants are ported from the `wxsection_ref` temperature
/// colormap options. The remaining palettes are lightweight Rust-side presets
/// aligned to the broader reference style catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CrossSectionPalette {
    TemperatureStandard,
    TemperatureWhiteZero,
    TemperatureNwsNdfd,
    TemperatureGreenPurple,
    WindSpeed,
    RelativeHumidity,
    ThetaE,
    SpecificHumidity,
    Omega,
    Shear,
    CloudWater,
    /// The tree's cloud-ICE ladder: a cyan-to-violet ramp that separates
    /// from the blue cloud-water ladder at a glance, so a section carrying
    /// both does not read as one blue field (gpuwm addition, VENDOR.md).
    CloudIce,
    TotalCondensate,
    LapseRate,
    WetBulb,
    Icing,
    Vorticity,
    Smoke,
    Frontogenesis,
    VaporPressureDeficit,
    DewpointDepression,
    MoistureTransport,
    PotentialVorticity,
    FireWeather,
}

/// Full named palette catalog exported by the cross-section crate.
pub const ALL_CROSS_SECTION_PALETTES: [CrossSectionPalette; 24] = [
    CrossSectionPalette::TemperatureStandard,
    CrossSectionPalette::TemperatureWhiteZero,
    CrossSectionPalette::TemperatureNwsNdfd,
    CrossSectionPalette::TemperatureGreenPurple,
    CrossSectionPalette::WindSpeed,
    CrossSectionPalette::RelativeHumidity,
    CrossSectionPalette::ThetaE,
    CrossSectionPalette::SpecificHumidity,
    CrossSectionPalette::Omega,
    CrossSectionPalette::Shear,
    CrossSectionPalette::CloudWater,
    CrossSectionPalette::CloudIce,
    CrossSectionPalette::TotalCondensate,
    CrossSectionPalette::LapseRate,
    CrossSectionPalette::WetBulb,
    CrossSectionPalette::Icing,
    CrossSectionPalette::Vorticity,
    CrossSectionPalette::Smoke,
    CrossSectionPalette::Frontogenesis,
    CrossSectionPalette::VaporPressureDeficit,
    CrossSectionPalette::DewpointDepression,
    CrossSectionPalette::MoistureTransport,
    CrossSectionPalette::PotentialVorticity,
    CrossSectionPalette::FireWeather,
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PaletteStop {
    pub position: f32,
    pub color: Color,
}

impl PaletteStop {
    pub const fn new(position: f32, color: Color) -> Self {
        Self { position, color }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FahrenheitAnchor {
    degrees_f: f32,
    color: Color,
}

impl FahrenheitAnchor {
    const fn new(degrees_f: f32, color: Color) -> Self {
        Self { degrees_f, color }
    }
}

impl Default for CrossSectionPalette {
    fn default() -> Self {
        Self::TemperatureStandard
    }
}

impl CrossSectionPalette {
    pub fn from_name(name: &str) -> Option<Self> {
        match normalize(name).as_str() {
            "standard" | "temperature" | "temp" | "temperature_standard" | "temp_standard" => {
                Some(Self::TemperatureStandard)
            }
            "white_zero" | "temperature_white_zero" | "temp_white_zero" => {
                Some(Self::TemperatureWhiteZero)
            }
            "nws_ndfd" | "temperature_nws_ndfd" | "temp_nws_ndfd" => Some(Self::TemperatureNwsNdfd),
            "green_purple" | "temperature_green_purple" | "temp_green_purple" => {
                Some(Self::TemperatureGreenPurple)
            }
            "wind_speed" | "wind" => Some(Self::WindSpeed),
            "relative_humidity" | "rh" => Some(Self::RelativeHumidity),
            "theta_e" => Some(Self::ThetaE),
            "specific_humidity" | "q" => Some(Self::SpecificHumidity),
            "omega" | "vertical_velocity" => Some(Self::Omega),
            "shear" | "wind_shear" => Some(Self::Shear),
            "cloud" | "cloud_water" => Some(Self::CloudWater),
            "cloud_ice" | "ice" => Some(Self::CloudIce),
            "cloud_total" | "total_condensate" => Some(Self::TotalCondensate),
            "lapse_rate" => Some(Self::LapseRate),
            "wetbulb" | "wet_bulb" => Some(Self::WetBulb),
            "icing" => Some(Self::Icing),
            "vorticity" => Some(Self::Vorticity),
            "smoke" => Some(Self::Smoke),
            "frontogenesis" => Some(Self::Frontogenesis),
            "vpd" | "vapor_pressure_deficit" => Some(Self::VaporPressureDeficit),
            "dewpoint_dep" | "dewpoint_depression" => Some(Self::DewpointDepression),
            "moisture_transport" => Some(Self::MoistureTransport),
            "pv" | "potential_vorticity" => Some(Self::PotentialVorticity),
            "fire_wx" | "fire_weather" => Some(Self::FireWeather),
            _ => None,
        }
    }

    pub const fn slug(self) -> &'static str {
        match self {
            Self::TemperatureStandard => "temperature_standard",
            Self::TemperatureWhiteZero => "temperature_white_zero",
            Self::TemperatureNwsNdfd => "temperature_nws_ndfd",
            Self::TemperatureGreenPurple => "temperature_green_purple",
            Self::WindSpeed => "wind_speed",
            Self::RelativeHumidity => "relative_humidity",
            Self::ThetaE => "theta_e",
            Self::SpecificHumidity => "specific_humidity",
            Self::Omega => "omega",
            Self::Shear => "shear",
            Self::CloudWater => "cloud_water",
            Self::CloudIce => "cloud_ice",
            Self::TotalCondensate => "total_condensate",
            Self::LapseRate => "lapse_rate",
            Self::WetBulb => "wetbulb",
            Self::Icing => "icing",
            Self::Vorticity => "vorticity",
            Self::Smoke => "smoke",
            Self::Frontogenesis => "frontogenesis",
            Self::VaporPressureDeficit => "vpd",
            Self::DewpointDepression => "dewpoint_depression",
            Self::MoistureTransport => "moisture_transport",
            Self::PotentialVorticity => "potential_vorticity",
            Self::FireWeather => "fire_weather",
        }
    }

    pub const fn display_name(self) -> &'static str {
        match self {
            Self::TemperatureStandard => "Temperature Standard",
            Self::TemperatureWhiteZero => "Temperature White Zero",
            Self::TemperatureNwsNdfd => "Temperature NWS NDFD",
            Self::TemperatureGreenPurple => "Temperature Green Purple",
            Self::WindSpeed => "Wind Speed",
            Self::RelativeHumidity => "Relative Humidity",
            Self::ThetaE => "Theta-E",
            Self::SpecificHumidity => "Specific Humidity",
            Self::Omega => "Omega",
            Self::Shear => "Wind Shear",
            Self::CloudWater => "Cloud Water",
            Self::CloudIce => "Cloud Ice",
            Self::TotalCondensate => "Total Condensate",
            Self::LapseRate => "Lapse Rate",
            Self::WetBulb => "Wet-Bulb Temperature",
            Self::Icing => "Icing Potential",
            Self::Vorticity => "Vorticity",
            Self::Smoke => "PM2.5 Smoke",
            Self::Frontogenesis => "Frontogenesis",
            Self::VaporPressureDeficit => "Vapor Pressure Deficit",
            Self::DewpointDepression => "Dewpoint Depression",
            Self::MoistureTransport => "Moisture Transport",
            Self::PotentialVorticity => "Potential Vorticity",
            Self::FireWeather => "Fire Weather",
        }
    }

    pub fn sampled_colors(self, sample_count: usize) -> Vec<Color> {
        let sample_count = sample_count.max(2);
        match self {
            Self::TemperatureStandard => {
                sample_fahrenheit_palette(&TEMPERATURE_STANDARD_ANCHORS, sample_count)
            }
            Self::TemperatureWhiteZero => {
                sample_fahrenheit_palette(&TEMPERATURE_WHITE_ZERO_ANCHORS, sample_count)
            }
            Self::TemperatureNwsNdfd => {
                sample_fahrenheit_palette(&TEMPERATURE_NWS_NDFD_ANCHORS, sample_count)
            }
            Self::TemperatureGreenPurple => {
                sample_fahrenheit_palette(&TEMPERATURE_GREEN_PURPLE_ANCHORS, sample_count)
            }
            Self::WindSpeed => sample_palette(&WIND_SPEED_STOPS, sample_count),
            Self::RelativeHumidity => sample_palette(&RELATIVE_HUMIDITY_STOPS, sample_count),
            Self::ThetaE => sample_palette(&THETA_E_STOPS, sample_count),
            Self::SpecificHumidity => sample_palette(&SPECIFIC_HUMIDITY_STOPS, sample_count),
            Self::Omega => sample_palette(&OMEGA_STOPS, sample_count),
            Self::Shear => sample_palette(&SHEAR_STOPS, sample_count),
            Self::CloudWater => sample_palette(&CLOUD_WATER_STOPS, sample_count),
            Self::CloudIce => sample_palette(&CLOUD_ICE_STOPS, sample_count),
            Self::TotalCondensate => sample_palette(&TOTAL_CONDENSATE_STOPS, sample_count),
            Self::LapseRate => sample_palette(&LAPSE_RATE_STOPS, sample_count),
            Self::WetBulb => sample_palette(&WET_BULB_STOPS, sample_count),
            Self::Icing => sample_palette(&ICING_STOPS, sample_count),
            Self::Vorticity => sample_palette(&VORTICITY_STOPS, sample_count),
            Self::Smoke => sample_palette(&SMOKE_STOPS, sample_count),
            Self::Frontogenesis => sample_palette(&FRONTOGENESIS_STOPS, sample_count),
            Self::VaporPressureDeficit => sample_palette(&VPD_STOPS, sample_count),
            Self::DewpointDepression => sample_palette(&DEWPOINT_DEPRESSION_STOPS, sample_count),
            Self::MoistureTransport => sample_palette(&MOISTURE_TRANSPORT_STOPS, sample_count),
            Self::PotentialVorticity => sample_palette(&POTENTIAL_VORTICITY_STOPS, sample_count),
            Self::FireWeather => sample_palette(&FIRE_WEATHER_STOPS, sample_count),
        }
    }

    pub fn build(self) -> Vec<Color> {
        self.sampled_colors(self.recommended_samples())
    }

    const fn recommended_samples(self) -> usize {
        match self {
            Self::TemperatureStandard
            | Self::TemperatureWhiteZero
            | Self::TemperatureNwsNdfd
            | Self::TemperatureGreenPurple => 33,
            Self::WindSpeed | Self::RelativeHumidity | Self::WetBulb => 25,
            _ => 21,
        }
    }
}

impl CrossSectionRenderRequest {
    pub fn with_named_palette(mut self, palette: CrossSectionPalette) -> Self {
        self.palette = palette.build();
        self
    }
}

const TEMPERATURE_STANDARD_ANCHORS: [FahrenheitAnchor; 19] = [
    FahrenheitAnchor::new(-80.0, Color::rgb(15, 0, 60)),
    FahrenheitAnchor::new(-65.0, Color::rgb(30, 5, 110)),
    FahrenheitAnchor::new(-50.0, Color::rgb(45, 20, 170)),
    FahrenheitAnchor::new(-35.0, Color::rgb(20, 55, 210)),
    FahrenheitAnchor::new(-20.0, Color::rgb(15, 95, 235)),
    FahrenheitAnchor::new(-5.0, Color::rgb(25, 150, 250)),
    FahrenheitAnchor::new(12.0, Color::rgb(40, 190, 235)),
    FahrenheitAnchor::new(24.0, Color::rgb(60, 210, 210)),
    FahrenheitAnchor::new(32.0, Color::rgb(80, 220, 190)),
    FahrenheitAnchor::new(42.0, Color::rgb(110, 210, 140)),
    FahrenheitAnchor::new(52.0, Color::rgb(170, 215, 80)),
    FahrenheitAnchor::new(62.0, Color::rgb(230, 210, 40)),
    FahrenheitAnchor::new(72.0, Color::rgb(255, 175, 20)),
    FahrenheitAnchor::new(82.0, Color::rgb(255, 130, 10)),
    FahrenheitAnchor::new(92.0, Color::rgb(240, 75, 10)),
    FahrenheitAnchor::new(100.0, Color::rgb(215, 30, 15)),
    FahrenheitAnchor::new(108.0, Color::rgb(170, 10, 25)),
    FahrenheitAnchor::new(118.0, Color::rgb(115, 5, 35)),
    FahrenheitAnchor::new(125.0, Color::rgb(70, 0, 40)),
];

const TEMPERATURE_WHITE_ZERO_ANCHORS: [FahrenheitAnchor; 16] = [
    FahrenheitAnchor::new(-80.0, Color::rgb(100, 50, 150)),
    FahrenheitAnchor::new(-60.0, Color::rgb(120, 60, 180)),
    FahrenheitAnchor::new(-40.0, Color::rgb(140, 80, 200)),
    FahrenheitAnchor::new(-20.0, Color::rgb(160, 110, 220)),
    FahrenheitAnchor::new(0.0, Color::rgb(190, 150, 230)),
    FahrenheitAnchor::new(15.0, Color::rgb(220, 200, 240)),
    FahrenheitAnchor::new(32.0, Color::rgb(255, 255, 255)),
    FahrenheitAnchor::new(45.0, Color::rgb(255, 240, 200)),
    FahrenheitAnchor::new(55.0, Color::rgb(255, 220, 150)),
    FahrenheitAnchor::new(65.0, Color::rgb(255, 200, 100)),
    FahrenheitAnchor::new(75.0, Color::rgb(255, 170, 70)),
    FahrenheitAnchor::new(85.0, Color::rgb(255, 130, 50)),
    FahrenheitAnchor::new(95.0, Color::rgb(240, 80, 30)),
    FahrenheitAnchor::new(105.0, Color::rgb(210, 40, 30)),
    FahrenheitAnchor::new(115.0, Color::rgb(160, 20, 60)),
    FahrenheitAnchor::new(125.0, Color::rgb(90, 10, 140)),
];

const TEMPERATURE_NWS_NDFD_ANCHORS: [FahrenheitAnchor; 13] = [
    FahrenheitAnchor::new(-80.0, Color::rgb(75, 0, 130)),
    FahrenheitAnchor::new(-60.0, Color::rgb(106, 0, 205)),
    FahrenheitAnchor::new(-40.0, Color::rgb(0, 0, 205)),
    FahrenheitAnchor::new(-20.0, Color::rgb(0, 0, 255)),
    FahrenheitAnchor::new(0.0, Color::rgb(0, 191, 255)),
    FahrenheitAnchor::new(15.0, Color::rgb(0, 255, 255)),
    FahrenheitAnchor::new(32.0, Color::rgb(255, 255, 0)),
    FahrenheitAnchor::new(50.0, Color::rgb(255, 215, 0)),
    FahrenheitAnchor::new(65.0, Color::rgb(255, 165, 0)),
    FahrenheitAnchor::new(80.0, Color::rgb(255, 69, 0)),
    FahrenheitAnchor::new(100.0, Color::rgb(255, 0, 0)),
    FahrenheitAnchor::new(115.0, Color::rgb(180, 0, 60)),
    FahrenheitAnchor::new(125.0, Color::rgb(90, 10, 140)),
];

const TEMPERATURE_GREEN_PURPLE_ANCHORS: [FahrenheitAnchor; 20] = [
    FahrenheitAnchor::new(-80.0, Color::rgb(220, 220, 255)),
    FahrenheitAnchor::new(-60.0, Color::rgb(180, 180, 255)),
    FahrenheitAnchor::new(-40.0, Color::rgb(140, 160, 240)),
    FahrenheitAnchor::new(-20.0, Color::rgb(100, 140, 220)),
    FahrenheitAnchor::new(0.0, Color::rgb(60, 140, 160)),
    FahrenheitAnchor::new(10.0, Color::rgb(60, 160, 130)),
    FahrenheitAnchor::new(20.0, Color::rgb(70, 180, 110)),
    FahrenheitAnchor::new(32.0, Color::rgb(80, 160, 80)),
    FahrenheitAnchor::new(40.0, Color::rgb(140, 210, 140)),
    FahrenheitAnchor::new(50.0, Color::rgb(255, 225, 140)),
    FahrenheitAnchor::new(60.0, Color::rgb(255, 200, 100)),
    FahrenheitAnchor::new(70.0, Color::rgb(255, 170, 80)),
    FahrenheitAnchor::new(80.0, Color::rgb(255, 140, 60)),
    FahrenheitAnchor::new(90.0, Color::rgb(255, 100, 40)),
    FahrenheitAnchor::new(100.0, Color::rgb(230, 60, 40)),
    FahrenheitAnchor::new(105.0, Color::rgb(200, 30, 30)),
    FahrenheitAnchor::new(110.0, Color::rgb(170, 20, 40)),
    FahrenheitAnchor::new(115.0, Color::rgb(140, 20, 70)),
    FahrenheitAnchor::new(120.0, Color::rgb(110, 20, 110)),
    FahrenheitAnchor::new(125.0, Color::rgb(90, 10, 140)),
];

const WIND_SPEED_STOPS: [PaletteStop; 11] = [
    PaletteStop::new(0.00, Color::rgb(255, 255, 255)),
    PaletteStop::new(0.10, Color::rgb(227, 242, 253)),
    PaletteStop::new(0.20, Color::rgb(144, 202, 249)),
    PaletteStop::new(0.30, Color::rgb(66, 165, 245)),
    PaletteStop::new(0.40, Color::rgb(30, 136, 229)),
    PaletteStop::new(0.50, Color::rgb(123, 31, 162)),
    PaletteStop::new(0.60, Color::rgb(233, 30, 99)),
    PaletteStop::new(0.70, Color::rgb(255, 235, 59)),
    PaletteStop::new(0.80, Color::rgb(255, 193, 7)),
    PaletteStop::new(0.90, Color::rgb(255, 152, 0)),
    PaletteStop::new(1.00, Color::rgb(244, 67, 54)),
];

const RELATIVE_HUMIDITY_STOPS: [PaletteStop; 8] = [
    PaletteStop::new(0.00, Color::rgb(153, 102, 51)),
    PaletteStop::new(0.14, Color::rgb(179, 128, 77)),
    PaletteStop::new(0.29, Color::rgb(217, 191, 128)),
    PaletteStop::new(0.43, Color::rgb(230, 230, 179)),
    PaletteStop::new(0.57, Color::rgb(179, 230, 179)),
    PaletteStop::new(0.71, Color::rgb(102, 204, 102)),
    PaletteStop::new(0.86, Color::rgb(51, 153, 77)),
    PaletteStop::new(1.00, Color::rgb(26, 102, 51)),
];

const THETA_E_STOPS: [PaletteStop; 11] = [
    PaletteStop::new(0.00, Color::rgb(94, 79, 162)),
    PaletteStop::new(0.10, Color::rgb(50, 136, 189)),
    PaletteStop::new(0.20, Color::rgb(102, 194, 165)),
    PaletteStop::new(0.30, Color::rgb(171, 221, 164)),
    PaletteStop::new(0.40, Color::rgb(230, 245, 152)),
    PaletteStop::new(0.50, Color::rgb(255, 255, 191)),
    PaletteStop::new(0.60, Color::rgb(254, 224, 139)),
    PaletteStop::new(0.70, Color::rgb(253, 174, 97)),
    PaletteStop::new(0.80, Color::rgb(244, 109, 67)),
    PaletteStop::new(0.90, Color::rgb(213, 62, 79)),
    PaletteStop::new(1.00, Color::rgb(158, 1, 66)),
];

const SPECIFIC_HUMIDITY_STOPS: [PaletteStop; 9] = [
    PaletteStop::new(0.00, Color::rgb(255, 255, 217)),
    PaletteStop::new(0.12, Color::rgb(237, 248, 177)),
    PaletteStop::new(0.25, Color::rgb(199, 233, 180)),
    PaletteStop::new(0.38, Color::rgb(127, 205, 187)),
    PaletteStop::new(0.50, Color::rgb(65, 182, 196)),
    PaletteStop::new(0.62, Color::rgb(29, 145, 192)),
    PaletteStop::new(0.75, Color::rgb(34, 94, 168)),
    PaletteStop::new(0.88, Color::rgb(37, 52, 148)),
    PaletteStop::new(1.00, Color::rgb(8, 29, 88)),
];

const OMEGA_STOPS: [PaletteStop; 11] = [
    PaletteStop::new(0.00, Color::rgb(5, 48, 97)),
    PaletteStop::new(0.10, Color::rgb(33, 102, 172)),
    PaletteStop::new(0.20, Color::rgb(67, 147, 195)),
    PaletteStop::new(0.30, Color::rgb(146, 197, 222)),
    PaletteStop::new(0.40, Color::rgb(209, 229, 240)),
    PaletteStop::new(0.50, Color::rgb(247, 247, 247)),
    PaletteStop::new(0.60, Color::rgb(253, 219, 199)),
    PaletteStop::new(0.70, Color::rgb(244, 165, 130)),
    PaletteStop::new(0.80, Color::rgb(214, 96, 77)),
    PaletteStop::new(0.90, Color::rgb(178, 24, 43)),
    PaletteStop::new(1.00, Color::rgb(103, 0, 31)),
];

const SHEAR_STOPS: [PaletteStop; 9] = [
    PaletteStop::new(0.00, Color::rgb(255, 247, 236)),
    PaletteStop::new(0.12, Color::rgb(254, 232, 200)),
    PaletteStop::new(0.25, Color::rgb(253, 212, 158)),
    PaletteStop::new(0.38, Color::rgb(253, 187, 132)),
    PaletteStop::new(0.50, Color::rgb(252, 141, 89)),
    PaletteStop::new(0.62, Color::rgb(239, 101, 72)),
    PaletteStop::new(0.75, Color::rgb(215, 48, 31)),
    PaletteStop::new(0.88, Color::rgb(179, 0, 0)),
    PaletteStop::new(1.00, Color::rgb(127, 0, 0)),
];

const CLOUD_WATER_STOPS: [PaletteStop; 9] = [
    PaletteStop::new(0.00, Color::rgb(247, 251, 255)),
    PaletteStop::new(0.12, Color::rgb(222, 235, 247)),
    PaletteStop::new(0.25, Color::rgb(198, 219, 239)),
    PaletteStop::new(0.38, Color::rgb(158, 202, 225)),
    PaletteStop::new(0.50, Color::rgb(107, 174, 214)),
    PaletteStop::new(0.62, Color::rgb(66, 146, 198)),
    PaletteStop::new(0.75, Color::rgb(33, 113, 181)),
    PaletteStop::new(0.88, Color::rgb(8, 81, 156)),
    PaletteStop::new(1.00, Color::rgb(8, 48, 107)),
];

/// Cloud ice: pale cyan into saturated teal and on into violet.  Deliberately
/// NOT another blue -- ice and supercooled water are drawn on the same sheet.
const CLOUD_ICE_STOPS: [PaletteStop; 8] = [
    PaletteStop::new(0.00, Color::rgb(240, 253, 255)),
    PaletteStop::new(0.14, Color::rgb(198, 244, 250)),
    PaletteStop::new(0.29, Color::rgb(129, 227, 238)),
    PaletteStop::new(0.43, Color::rgb(62, 196, 214)),
    PaletteStop::new(0.57, Color::rgb(30, 152, 186)),
    PaletteStop::new(0.71, Color::rgb(56, 100, 176)),
    PaletteStop::new(0.86, Color::rgb(88, 58, 158)),
    PaletteStop::new(1.00, Color::rgb(72, 24, 108)),
];

const TOTAL_CONDENSATE_STOPS: [PaletteStop; 8] = [
    PaletteStop::new(0.00, Color::rgb(255, 255, 255)),
    PaletteStop::new(0.14, Color::rgb(240, 240, 245)),
    PaletteStop::new(0.29, Color::rgb(216, 220, 232)),
    PaletteStop::new(0.43, Color::rgb(184, 196, 216)),
    PaletteStop::new(0.57, Color::rgb(152, 172, 200)),
    PaletteStop::new(0.71, Color::rgb(120, 148, 184)),
    PaletteStop::new(0.86, Color::rgb(88, 120, 168)),
    PaletteStop::new(1.00, Color::rgb(56, 88, 152)),
];

const LAPSE_RATE_STOPS: [PaletteStop; 11] = [
    PaletteStop::new(0.00, Color::rgb(49, 54, 149)),
    PaletteStop::new(0.10, Color::rgb(69, 117, 180)),
    PaletteStop::new(0.20, Color::rgb(116, 173, 209)),
    PaletteStop::new(0.30, Color::rgb(171, 217, 233)),
    PaletteStop::new(0.40, Color::rgb(224, 243, 248)),
    PaletteStop::new(0.50, Color::rgb(255, 255, 191)),
    PaletteStop::new(0.60, Color::rgb(254, 224, 144)),
    PaletteStop::new(0.70, Color::rgb(253, 174, 97)),
    PaletteStop::new(0.80, Color::rgb(244, 109, 67)),
    PaletteStop::new(0.90, Color::rgb(215, 48, 39)),
    PaletteStop::new(1.00, Color::rgb(165, 0, 38)),
];

const WET_BULB_STOPS: [PaletteStop; 8] = [
    PaletteStop::new(0.00, Color::rgb(59, 76, 192)),
    PaletteStop::new(0.14, Color::rgb(104, 138, 239)),
    PaletteStop::new(0.29, Color::rgb(152, 193, 255)),
    PaletteStop::new(0.43, Color::rgb(201, 215, 240)),
    PaletteStop::new(0.57, Color::rgb(237, 209, 194)),
    PaletteStop::new(0.71, Color::rgb(247, 168, 137)),
    PaletteStop::new(0.86, Color::rgb(226, 105, 82)),
    PaletteStop::new(1.00, Color::rgb(180, 4, 38)),
];

const ICING_STOPS: [PaletteStop; 7] = [
    PaletteStop::new(0.00, Color::rgb(255, 255, 255)),
    PaletteStop::new(0.17, Color::rgb(227, 242, 253)),
    PaletteStop::new(0.33, Color::rgb(187, 222, 251)),
    PaletteStop::new(0.50, Color::rgb(100, 181, 246)),
    PaletteStop::new(0.67, Color::rgb(33, 150, 243)),
    PaletteStop::new(0.83, Color::rgb(21, 101, 192)),
    PaletteStop::new(1.00, Color::rgb(13, 71, 161)),
];

const VORTICITY_STOPS: [PaletteStop; 11] = OMEGA_STOPS;

const SMOKE_STOPS: [PaletteStop; 8] = [
    PaletteStop::new(0.00, Color::rgb(230, 243, 255)),
    PaletteStop::new(0.14, Color::rgb(135, 206, 235)),
    PaletteStop::new(0.29, Color::rgb(144, 238, 144)),
    PaletteStop::new(0.43, Color::rgb(255, 255, 0)),
    PaletteStop::new(0.57, Color::rgb(255, 165, 0)),
    PaletteStop::new(0.71, Color::rgb(255, 69, 0)),
    PaletteStop::new(0.86, Color::rgb(255, 0, 0)),
    PaletteStop::new(1.00, Color::rgb(128, 0, 128)),
];

const FRONTOGENESIS_STOPS: [PaletteStop; 9] = [
    PaletteStop::new(0.00, Color::rgb(33, 102, 172)),
    PaletteStop::new(0.12, Color::rgb(67, 147, 195)),
    PaletteStop::new(0.25, Color::rgb(146, 197, 222)),
    PaletteStop::new(0.38, Color::rgb(209, 229, 240)),
    PaletteStop::new(0.50, Color::rgb(247, 247, 247)),
    PaletteStop::new(0.62, Color::rgb(253, 219, 199)),
    PaletteStop::new(0.75, Color::rgb(244, 165, 130)),
    PaletteStop::new(0.88, Color::rgb(214, 96, 77)),
    PaletteStop::new(1.00, Color::rgb(178, 24, 43)),
];

const VPD_STOPS: [PaletteStop; 9] = [
    PaletteStop::new(0.00, Color::rgb(26, 152, 80)),
    PaletteStop::new(0.12, Color::rgb(102, 189, 99)),
    PaletteStop::new(0.25, Color::rgb(166, 217, 106)),
    PaletteStop::new(0.38, Color::rgb(217, 239, 139)),
    PaletteStop::new(0.50, Color::rgb(254, 224, 139)),
    PaletteStop::new(0.62, Color::rgb(253, 174, 97)),
    PaletteStop::new(0.75, Color::rgb(244, 109, 67)),
    PaletteStop::new(0.88, Color::rgb(215, 48, 39)),
    PaletteStop::new(1.00, Color::rgb(165, 0, 38)),
];

const DEWPOINT_DEPRESSION_STOPS: [PaletteStop; 10] = [
    PaletteStop::new(0.00, Color::rgb(0, 104, 55)),
    PaletteStop::new(0.11, Color::rgb(26, 152, 80)),
    PaletteStop::new(0.22, Color::rgb(102, 189, 99)),
    PaletteStop::new(0.33, Color::rgb(166, 217, 106)),
    PaletteStop::new(0.44, Color::rgb(217, 239, 139)),
    PaletteStop::new(0.56, Color::rgb(254, 224, 139)),
    PaletteStop::new(0.67, Color::rgb(253, 174, 97)),
    PaletteStop::new(0.78, Color::rgb(244, 109, 67)),
    PaletteStop::new(0.89, Color::rgb(215, 48, 39)),
    PaletteStop::new(1.00, Color::rgb(165, 0, 38)),
];

const MOISTURE_TRANSPORT_STOPS: [PaletteStop; 9] = SPECIFIC_HUMIDITY_STOPS;

const POTENTIAL_VORTICITY_STOPS: [PaletteStop; 11] = [
    PaletteStop::new(0.00, Color::rgb(84, 48, 5)),
    PaletteStop::new(0.10, Color::rgb(140, 81, 10)),
    PaletteStop::new(0.20, Color::rgb(191, 129, 45)),
    PaletteStop::new(0.30, Color::rgb(223, 194, 125)),
    PaletteStop::new(0.40, Color::rgb(246, 232, 195)),
    PaletteStop::new(0.50, Color::rgb(245, 245, 245)),
    PaletteStop::new(0.60, Color::rgb(199, 234, 229)),
    PaletteStop::new(0.70, Color::rgb(128, 205, 193)),
    PaletteStop::new(0.80, Color::rgb(53, 151, 143)),
    PaletteStop::new(0.90, Color::rgb(1, 102, 94)),
    PaletteStop::new(1.00, Color::rgb(0, 60, 48)),
];

const FIRE_WEATHER_STOPS: [PaletteStop; 8] = [
    PaletteStop::new(0.00, Color::rgb(139, 0, 0)),
    PaletteStop::new(0.14, Color::rgb(204, 0, 0)),
    PaletteStop::new(0.29, Color::rgb(255, 69, 0)),
    PaletteStop::new(0.43, Color::rgb(255, 140, 0)),
    PaletteStop::new(0.57, Color::rgb(255, 215, 0)),
    PaletteStop::new(0.71, Color::rgb(173, 255, 47)),
    PaletteStop::new(0.86, Color::rgb(50, 205, 50)),
    PaletteStop::new(1.00, Color::rgb(34, 139, 34)),
];

fn normalize(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn sample_fahrenheit_palette(anchors: &[FahrenheitAnchor], sample_count: usize) -> Vec<Color> {
    if anchors.is_empty() {
        return vec![Color::BLACK; sample_count.max(2)];
    }
    if anchors.len() == 1 {
        return vec![anchors[0].color; sample_count.max(2)];
    }

    let min_f = anchors[0].degrees_f;
    let max_f = anchors[anchors.len() - 1].degrees_f;
    let span = (max_f - min_f).max(f32::EPSILON);
    let stops = anchors
        .iter()
        .map(|anchor| PaletteStop::new((anchor.degrees_f - min_f) / span, anchor.color))
        .collect::<Vec<_>>();
    sample_palette(&stops, sample_count)
}

fn sample_palette(stops: &[PaletteStop], sample_count: usize) -> Vec<Color> {
    if stops.is_empty() {
        return vec![Color::BLACK; sample_count.max(2)];
    }
    if stops.len() == 1 {
        return vec![stops[0].color; sample_count.max(2)];
    }

    let sample_count = sample_count.max(2);
    let mut sampled = Vec::with_capacity(sample_count);
    for index in 0..sample_count {
        let position = if sample_count == 1 {
            0.0
        } else {
            index as f32 / (sample_count - 1) as f32
        };
        sampled.push(color_at_position(stops, position));
    }
    sampled
}

fn color_at_position(stops: &[PaletteStop], position: f32) -> Color {
    if position <= stops[0].position {
        return stops[0].color;
    }
    if position >= stops[stops.len() - 1].position {
        return stops[stops.len() - 1].color;
    }

    for pair in stops.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if position <= right.position {
            let span = (right.position - left.position).max(f32::EPSILON);
            let fraction = ((position - left.position) / span).clamp(0.0, 1.0);
            return mix_color(left.color, right.color, fraction);
        }
    }

    stops[stops.len() - 1].color
}

fn mix_color(start: Color, end: Color, fraction: f32) -> Color {
    let fraction = fraction.clamp(0.0, 1.0);
    let mix = |left: u8, right: u8| -> u8 {
        let left = left as f32;
        let right = right as f32;
        (left + (right - left) * fraction).round() as u8
    };

    Color::rgba(
        mix(start.r, end.r),
        mix(start.g, end.g),
        mix(start.b, end.b),
        mix(start.a, end.a),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_palette_aliases_resolve_expected_variants() {
        assert_eq!(
            CrossSectionPalette::from_name("standard"),
            Some(CrossSectionPalette::TemperatureStandard)
        );
        assert_eq!(
            CrossSectionPalette::from_name("white_zero"),
            Some(CrossSectionPalette::TemperatureWhiteZero)
        );
        assert_eq!(
            CrossSectionPalette::from_name("nws_ndfd"),
            Some(CrossSectionPalette::TemperatureNwsNdfd)
        );
        assert_eq!(
            CrossSectionPalette::from_name("green-purple"),
            Some(CrossSectionPalette::TemperatureGreenPurple)
        );
    }

    #[test]
    fn palette_builder_generates_requested_sample_count() {
        let colors = CrossSectionPalette::Smoke.sampled_colors(9);
        assert_eq!(colors.len(), 9);
        assert_eq!(colors.first(), Some(&Color::rgb(230, 243, 255)));
        assert_eq!(colors.last(), Some(&Color::rgb(128, 0, 128)));
    }

    #[test]
    fn request_builder_swaps_palette_without_touching_other_request_fields() {
        let request = CrossSectionRenderRequest::default()
            .with_dimensions(800, 600)
            .with_named_palette(CrossSectionPalette::TemperatureWhiteZero);

        assert_eq!(request.width, 800);
        assert_eq!(request.height, 600);
        assert_eq!(request.palette.len(), 33);
        assert_ne!(
            request.palette,
            CrossSectionPalette::TemperatureStandard.build()
        );
    }
}

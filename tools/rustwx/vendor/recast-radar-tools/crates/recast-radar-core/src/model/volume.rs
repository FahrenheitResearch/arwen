//! The FM301 root group (`docs/design/fm301-model.md` sections 2 and 11).

use chrono::{DateTime, Duration, Timelike, Utc};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::sweep::{Sweep, SweepError};
use super::values::{AttrValue, ExtraVariable, VariableAttrs};

/// FM301 volume: the root group of an FM301 / CfRadial 2 file and the root node
/// of an xradar `DataTree`.
///
/// With the `serde` feature, deserializing a volume checks what
/// [`Volume::seal`] checks and fails when a check does.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(try_from = "super::serde_checked::VolumeRepr")
)]
pub struct Volume {
    /// Root attributes (Tables 301-1, 301-2, 301-3; WMO-CF-2).
    pub attrs: GlobalAttrs,
    /// `/volume_number`.
    pub volume_number: Option<i32>,
    /// Epoch for every "seconds since `<reftime>`" value in the volume
    /// (`Rays::time_s`, `RadarCalibration::time_s`). Always whole seconds: FM301
    /// Table 301-6b writes it as `YYYY-MM-DDThh:mm:ssZ`, and any fraction lives
    /// in `time_s`.
    pub time_reference: DateTime<Utc>,
    /// `/time_coverage_start`, `/time_coverage_end`: first and last ray.
    pub time_coverage: Option<TimeCoverage>,
    /// `/latitude`, `/longitude`, `/altitude`, `/altitude_agl`.
    pub location: Location,
    /// `/platform_type` (Table 301-15).
    pub platform_type: PlatformType,
    /// `/instrument_type` (Table 301-15).
    pub instrument_type: InstrumentType,
    /// `/primary_axis` (Table 301-15).
    pub primary_axis: Option<PrimaryAxis>,
    /// `/status_str`.
    pub status_str: Option<String>,
    /// `scan_name`, `scan_id`, and scan-table provenance.
    pub scan: ScanStrategy,
    /// `/radar_parameters` and the `frequency` coordinate.
    pub radar_parameters: RadarParameters,
    /// `/radar_calibration`, one entry per `calib` index.
    pub radar_calibration: Vec<RadarCalibration>,
    /// `/georeferencing_correction` (CfRadial 1/2; not part of FM301).
    pub georeferencing_correction: Option<Box<GeoreferencingCorrection>>,
    /// Root variables with no slot above, verbatim and in file order.
    pub extra_vars: Vec<ExtraVariable>,
    /// The source's own attributes of variables whose values a typed slot
    /// holds (CfRadial `azimuth:comment`, `range:meters_between_gates`), by
    /// group and variable, verbatim and in file order. The view writes them
    /// with `Passthrough::All` beside the attributes it derives.
    #[cfg_attr(feature = "serde", serde(default))]
    pub variable_attrs: Vec<VariableAttrs>,
    /// Source format, container version, decode statistics. Not exported as
    /// variables.
    pub provenance: Provenance,
    /// Model / forward-operator provenance for simulated volumes.
    pub simulation: Option<Box<SimulationProvenance>>,
    /// `sweep_0 ..` in acquisition order; `sweeps[i].sweep_number == i`.
    pub sweeps: Vec<Sweep>,
}

/// First and last ray times of a volume: `/time_coverage_start` and `/time_coverage_end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeCoverage {
    /// Time of the first ray.
    pub start: DateTime<Utc>,
    /// Time of the last ray.
    pub end: DateTime<Utc>,
}

/// Global attributes of the root group (FM301 Tables 301-1 to 301-3 and the
/// WMO-CF attributes).
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GlobalAttrs {
    /// `title`: a short description of what the file holds.
    pub title: Option<String>,
    /// `institution`: where the data were produced.
    pub institution: Option<String>,
    /// `references`: published or web references for the data or the methods.
    pub references: Option<String>,
    /// `source`: how the original data were produced.
    pub source: Option<String>,
    /// `history`: the processing the data have been through.
    pub history: Option<String>,
    /// `comment`: anything else about the data.
    pub comment: Option<String>,
    /// `instrument_name`: NEXRAD ICAO; ODIM `what/source` NOD, else RAD, else
    /// WMO; CfRadial attribute. Empty when the source has none.
    pub instrument_name: String,
    /// `site_name`: the name of the radar site.
    pub site_name: Option<String>,
    /// `platform_is_mobile`: `true` when the source says the platform moves
    /// (a CfRadial file's attribute; a DORADE airborne or shipborne radar,
    /// as LROSE Radx writes it), else `false`, the only value FM301 2022
    /// uses for the fixed platforms it describes.
    pub platform_is_mobile: bool,
    /// `ray_times_increase`: whether ray times increase through the file
    /// (CfRadial); `None` when the source does not say.
    pub ray_times_increase: Option<bool>,
    /// `simulated` (Table 301-3).
    pub simulated: bool,
    /// `wmo__*` attributes (WMO-CF.6.10).
    pub wmo: WmoAttrs,
    /// Source attributes with no slot above, verbatim, typed and in file order.
    pub other: Vec<(Box<str>, AttrValue)>,
}

/// The WMO-CF global attributes (`wmo__*`, WMO-CF.6.10).
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct WmoAttrs {
    /// `wmo__wsi`.
    pub wsi: Option<String>,
    /// `wmo__id` (ODIM "WMO:03962" -> "03962").
    pub id: Option<String>,
    /// `wmo__originating_centre` (Common Code Table C-11).
    pub originating_centre: Option<u16>,
    /// `wmo__originating_sub_centre` (Common Code Table C-12).
    pub originating_sub_centre: Option<u16>,
    /// `wmo__data_category` (C-13).
    pub data_category: Option<u8>,
    /// `wmo__data_policy`.
    pub data_policy: Option<WmoDataPolicy>,
    /// `wmo__update_sequence_number`: 0 for the original data, incremented with
    /// each correction.
    pub update_sequence_number: Option<u32>,
}

/// `wmo__data_policy`, the WMO Unified Data Policy category of the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum WmoDataPolicy {
    /// `core`: exchanged without charge or conditions.
    Core,
    /// `recommended`: exchanged, possibly with conditions.
    Recommended,
}

impl WmoDataPolicy {
    /// The attribute spelling: `core` or `recommended`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Core => "core",
            Self::Recommended => "recommended",
        }
    }
}

/// `/latitude`, `/longitude`, `/altitude` and `/altitude_agl`: where the radar is.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Location {
    /// WGS84 degrees north. `None` when the source has no location (Message 1
    /// archives).
    pub latitude_deg: Option<f64>,
    /// WGS84 degrees east. `None` when the source has no location.
    pub longitude_deg: Option<f64>,
    /// Metres above MSL at the antenna's centre of rotation.
    pub altitude_m: Option<f64>,
    /// Metres above the ground at the antenna's centre of rotation.
    pub altitude_agl_m: Option<f64>,
}

macro_rules! table_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident => $text:literal,)* }) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        #[non_exhaustive]
        pub enum $name {
            $(
                #[doc = concat!("The spelling `", $text, "`.")]
                $variant,
            )*
        }

        impl $name {
            /// The Table 301-15 spelling.
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $text,)*
                }
            }

            /// Table 301-15 spelling (trimmed) to its variant.
            pub fn parse(text: &str) -> Option<Self> {
                match text.trim() {
                    $($text => Some(Self::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

table_enum! {
    /// `platform_type` (Table 301-15).
    PlatformType {
        Fixed => "fixed",
        Vehicle => "vehicle",
        Ship => "ship",
        Aircraft => "aircraft",
        AircraftFore => "aircraft_fore",
        AircraftAft => "aircraft_aft",
        AircraftTail => "aircraft_tail",
        AircraftBelly => "aircraft_belly",
        AircraftRoof => "aircraft_roof",
        AircraftNose => "aircraft_nose",
        SatelliteOrbit => "satellite_orbit",
        SatelliteGeostat => "satellite_geostat",
    }
}

table_enum! {
    /// `instrument_type` (Table 301-15).
    InstrumentType {
        Radar => "radar",
        Lidar => "lidar",
    }
}

table_enum! {
    /// `primary_axis` (Table 301-15).
    PrimaryAxis {
        AxisZ => "axis_z",
        AxisY => "axis_y",
        AxisX => "axis_x",
        AxisZPrime => "axis_z_prime",
        AxisYPrime => "axis_y_prime",
        AxisXPrime => "axis_x_prime",
    }
}

/// The scan strategy: `scan_name`, `scan_id`, the NEXRAD volume coverage
/// pattern, and the scan table the volume follows.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScanStrategy {
    /// `scan_name`. NEXRAD: "VCP-212" (xradar's spelling).
    pub name: Option<String>,
    /// `scan_id` (FM301 int). NEXRAD: the VCP number.
    pub id: Option<i64>,
    /// NEXRAD VCP pattern.
    pub vcp_pattern: Option<u16>,
    /// Scan-table provenance.
    pub definition: Option<Box<ScanDefinition>>,
}

/// Provenance of a scan table: the document it was taken from, and one leg per
/// sweep.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScanDefinition {
    /// Document the table comes from (for example a WSR-88D ICD).
    pub source_document: Option<String>,
    /// Revision of that document.
    pub source_revision: Option<String>,
    /// RDA software build the table applies to.
    pub source_rda_build: Option<String>,
    /// Figure or table of the document that lists the scan.
    pub source_figure: Option<String>,
    /// Pulse length of the scan (`short`, `long`), as the document names it.
    pub pulse_length: Option<String>,
    /// Site adaptations applied to the table, as written.
    pub adaptations: Option<String>,
    /// `scan_id` text as the source wrote it, whenever it is not exactly
    /// `ScanStrategy::id.to_string()`: non-numeric ids, and numeric ids with
    /// other spellings ("00", "+5"). `None` when `id` reproduces the text.
    pub scan_id_text: Option<String>,
    /// One leg per sweep.
    pub legs: Vec<ScanLeg>,
}

/// Source-qualified provenance for one physical scan leg / sweep.
///
/// PRF values here are source-table *codes* and pulse counts, never
/// frequencies or pulse-repetition times.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ScanLeg {
    /// Zero-based row within the qualified source definition.
    pub source_row_index: Option<u16>,
    /// Elevation angle of the leg, in degrees.
    pub elevation_deg: Option<f32>,
    /// Antenna azimuth rate of the leg, in degrees per second.
    pub azimuth_rate_deg_per_second: Option<f32>,
    /// Duration of the leg in the source table, in seconds.
    pub source_period_seconds: Option<f32>,
    /// Source waveform abbreviation (for example `CS`, `CD/W`, or `SZCD`).
    pub waveform: Option<String>,
    /// `surveillance`, `doppler`, or `all` for catalog-backed synthetic cuts.
    pub moment_coverage: Option<String>,
    /// PRF code of the surveillance (long-PRT) scan of the leg.
    pub surveillance_prf_code: Option<u8>,
    /// Pulse count per radial of the surveillance scan.
    pub surveillance_pulse_count: Option<u16>,
    /// PRF code of the Doppler (short-PRT) scan of the leg.
    pub doppler_prf_code: Option<u8>,
    /// Pulse count per radial of the Doppler scan.
    pub doppler_pulse_count: Option<u16>,
}

/// `/radar_parameters`. Variable names differ by flavor (design note 12.4).
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RadarParameters {
    /// Operating frequencies in Hz (`frequency` dimension).
    pub frequency_hz: Vec<f64>,
    /// Antenna gain, horizontal polarization, in dB.
    pub antenna_gain_h_db: Option<f32>,
    /// Antenna gain, vertical polarization, in dB.
    pub antenna_gain_v_db: Option<f32>,
    /// Antenna 3 dB beam width, horizontal plane, in degrees.
    pub beam_width_h_deg: Option<f32>,
    /// Antenna 3 dB beam width, vertical plane, in degrees.
    pub beam_width_v_deg: Option<f32>,
    /// Receiver bandwidth, in Hz.
    pub receiver_bandwidth_hz: Option<f32>,
    /// Volume-constant values some sources declare instead of per-ray vectors.
    /// For sweeps whose `RayVariables` lack them, the view broadcasts these into
    /// `(time)` variables.
    pub pulse_width_s: Option<f32>,
    /// Pulse repetition time, in seconds, for the whole volume.
    pub prt_s: Option<f32>,
    /// Unambiguous range, in metres, for the whole volume.
    pub unambiguous_range_m: Option<f32>,
}

/// One `calib` entry of `/radar_calibration` (Table 301-14a names plus a unit
/// suffix).
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct RadarCalibration {
    /// `calib_index`: this entry's index, as the source numbers it.
    pub calib_index: Option<i32>,
    /// Seconds since `Volume::time_reference`.
    pub time_s: Option<f64>,
    /// `pulse_width`: the pulse width the entry applies to, in seconds.
    pub pulse_width_s: Option<f32>,
    /// `antenna_gain_h`: antenna gain, horizontal polarization, dB.
    pub antenna_gain_h_db: Option<f32>,
    /// `antenna_gain_v`: antenna gain, vertical polarization, dB.
    pub antenna_gain_v_db: Option<f32>,
    /// `xmit_power_h`: transmitted power, horizontal channel, dBm.
    pub xmit_power_h_dbm: Option<f32>,
    /// `xmit_power_v`: transmitted power, vertical channel, dBm.
    pub xmit_power_v_dbm: Option<f32>,
    /// `two_way_waveguide_loss_h`: two-way waveguide loss, horizontal channel, dB.
    pub two_way_waveguide_loss_h_db: Option<f32>,
    /// `two_way_waveguide_loss_v`: two-way waveguide loss, vertical channel, dB.
    pub two_way_waveguide_loss_v_db: Option<f32>,
    /// `two_way_radome_loss_h`: two-way radome loss, horizontal channel, dB.
    pub two_way_radome_loss_h_db: Option<f32>,
    /// `two_way_radome_loss_v`: two-way radome loss, vertical channel, dB.
    pub two_way_radome_loss_v_db: Option<f32>,
    /// `receiver_mismatch_loss`: receiver mismatch loss, dB.
    pub receiver_mismatch_loss_db: Option<f32>,
    /// `receiver_mismatch_loss_h`: receiver mismatch loss, horizontal channel, dB.
    pub receiver_mismatch_loss_h_db: Option<f32>,
    /// `receiver_mismatch_loss_v`: receiver mismatch loss, vertical channel, dB.
    pub receiver_mismatch_loss_v_db: Option<f32>,
    /// `radar_constant_h`: radar constant, horizontal channel, dB.
    pub radar_constant_h: Option<f32>,
    /// `radar_constant_v`: radar constant, vertical channel, dB.
    pub radar_constant_v: Option<f32>,
    /// `probert_jones_correction`: Probert-Jones beam-filling correction, dB.
    pub probert_jones_correction: Option<f32>,
    /// `dielectric_factor_used`: the dielectric factor |K|² the calibration assumes.
    pub dielectric_factor_used: Option<f32>,
    /// `noise_hc`: noise power, horizontal co-polar channel, dBm.
    pub noise_hc_dbm: Option<f32>,
    /// `noise_vc`: noise power, vertical co-polar channel, dBm.
    pub noise_vc_dbm: Option<f32>,
    /// `noise_hx`: noise power, horizontal cross-polar channel, dBm.
    pub noise_hx_dbm: Option<f32>,
    /// `noise_vx`: noise power, vertical cross-polar channel, dBm.
    pub noise_vx_dbm: Option<f32>,
    /// `receiver_gain_hc`: receiver gain, horizontal co-polar channel, dB.
    pub receiver_gain_hc_db: Option<f32>,
    /// `receiver_gain_vc`: receiver gain, vertical co-polar channel, dB.
    pub receiver_gain_vc_db: Option<f32>,
    /// `receiver_gain_hx`: receiver gain, horizontal cross-polar channel, dB.
    pub receiver_gain_hx_db: Option<f32>,
    /// `receiver_gain_vx`: receiver gain, vertical cross-polar channel, dB.
    pub receiver_gain_vx_db: Option<f32>,
    /// `base_1km_hc`: reflectivity of a noise-level signal at 1 km, horizontal
    /// co-polar channel, dBZ.
    pub base_1km_hc_dbz: Option<f32>,
    /// `base_1km_vc`: reflectivity of a noise-level signal at 1 km, vertical
    /// co-polar channel, dBZ.
    pub base_1km_vc_dbz: Option<f32>,
    /// `base_1km_hx`: reflectivity of a noise-level signal at 1 km, horizontal
    /// cross-polar channel, dBZ.
    pub base_1km_hx_dbz: Option<f32>,
    /// `base_1km_vx`: reflectivity of a noise-level signal at 1 km, vertical
    /// cross-polar channel, dBZ.
    pub base_1km_vx_dbz: Option<f32>,
    /// `sun_power_hc`: measured sun power, horizontal co-polar channel, dBm.
    pub sun_power_hc_dbm: Option<f32>,
    /// `sun_power_vc`: measured sun power, vertical co-polar channel, dBm.
    pub sun_power_vc_dbm: Option<f32>,
    /// `sun_power_hx`: measured sun power, horizontal cross-polar channel, dBm.
    pub sun_power_hx_dbm: Option<f32>,
    /// `sun_power_vx`: measured sun power, vertical cross-polar channel, dBm.
    pub sun_power_vx_dbm: Option<f32>,
    /// `noise_source_power_h`: calibration noise source power, horizontal channel, dBm.
    pub noise_source_power_h_dbm: Option<f32>,
    /// `noise_source_power_v`: calibration noise source power, vertical channel, dBm.
    pub noise_source_power_v_dbm: Option<f32>,
    /// `power_measure_loss_h`: loss in the power measurement path, horizontal channel, dB.
    pub power_measure_loss_h_db: Option<f32>,
    /// `power_measure_loss_v`: loss in the power measurement path, vertical channel, dB.
    pub power_measure_loss_v_db: Option<f32>,
    /// `coupler_forward_loss_h`: directional coupler forward loss, horizontal channel, dB.
    pub coupler_forward_loss_h_db: Option<f32>,
    /// `coupler_forward_loss_v`: directional coupler forward loss, vertical channel, dB.
    pub coupler_forward_loss_v_db: Option<f32>,
    /// `zdr_correction`: correction added to differential reflectivity, dB.
    pub zdr_correction_db: Option<f32>,
    /// `ldr_correction_h`: correction added to LDR, horizontal channel, dB.
    pub ldr_correction_h_db: Option<f32>,
    /// `ldr_correction_v`: correction added to LDR, vertical channel, dB.
    pub ldr_correction_v_db: Option<f32>,
    /// `system_phidp`: the system's differential phase, degrees.
    pub system_phidp_deg: Option<f32>,
    /// `test_power_h`: calibration test signal power, horizontal channel, dBm.
    pub test_power_h_dbm: Option<f32>,
    /// `test_power_v`: calibration test signal power, vertical channel, dBm.
    pub test_power_v_dbm: Option<f32>,
    /// `receiver_slope_hc`: slope of the receiver power response, horizontal co-polar channel.
    pub receiver_slope_hc: Option<f32>,
    /// `receiver_slope_vc`: slope of the receiver power response, vertical co-polar channel.
    pub receiver_slope_vc: Option<f32>,
    /// `receiver_slope_hx`: slope of the receiver power response, horizontal cross-polar channel.
    pub receiver_slope_hx: Option<f32>,
    /// `receiver_slope_vx`: slope of the receiver power response, vertical cross-polar channel.
    pub receiver_slope_vx: Option<f32>,
    /// Entries outside Table 301-14a, named as xradar names them (the CfRadial
    /// `r_calib_` prefix removed): `k_squared_water`, `i0_dbm_hc`, ...
    pub extra: Vec<(Box<str>, AttrValue)>,
}

impl RadarCalibration {
    /// `(Table 301-14a name, value)` for every float entry, in table order
    /// (`calib_index` and `time` excluded).
    pub fn float_entries(&self) -> [(&'static str, Option<f32>); 48] {
        [
            ("pulse_width", self.pulse_width_s),
            ("antenna_gain_h", self.antenna_gain_h_db),
            ("antenna_gain_v", self.antenna_gain_v_db),
            ("xmit_power_h", self.xmit_power_h_dbm),
            ("xmit_power_v", self.xmit_power_v_dbm),
            ("two_way_waveguide_loss_h", self.two_way_waveguide_loss_h_db),
            ("two_way_waveguide_loss_v", self.two_way_waveguide_loss_v_db),
            ("two_way_radome_loss_h", self.two_way_radome_loss_h_db),
            ("two_way_radome_loss_v", self.two_way_radome_loss_v_db),
            ("receiver_mismatch_loss", self.receiver_mismatch_loss_db),
            ("receiver_mismatch_loss_h", self.receiver_mismatch_loss_h_db),
            ("receiver_mismatch_loss_v", self.receiver_mismatch_loss_v_db),
            ("radar_constant_h", self.radar_constant_h),
            ("radar_constant_v", self.radar_constant_v),
            ("probert_jones_correction", self.probert_jones_correction),
            ("dielectric_factor_used", self.dielectric_factor_used),
            ("noise_hc", self.noise_hc_dbm),
            ("noise_vc", self.noise_vc_dbm),
            ("noise_hx", self.noise_hx_dbm),
            ("noise_vx", self.noise_vx_dbm),
            ("receiver_gain_hc", self.receiver_gain_hc_db),
            ("receiver_gain_vc", self.receiver_gain_vc_db),
            ("receiver_gain_hx", self.receiver_gain_hx_db),
            ("receiver_gain_vx", self.receiver_gain_vx_db),
            ("base_1km_hc", self.base_1km_hc_dbz),
            ("base_1km_vc", self.base_1km_vc_dbz),
            ("base_1km_hx", self.base_1km_hx_dbz),
            ("base_1km_vx", self.base_1km_vx_dbz),
            ("sun_power_hc", self.sun_power_hc_dbm),
            ("sun_power_vc", self.sun_power_vc_dbm),
            ("sun_power_hx", self.sun_power_hx_dbm),
            ("sun_power_vx", self.sun_power_vx_dbm),
            ("noise_source_power_h", self.noise_source_power_h_dbm),
            ("noise_source_power_v", self.noise_source_power_v_dbm),
            ("power_measure_loss_h", self.power_measure_loss_h_db),
            ("power_measure_loss_v", self.power_measure_loss_v_db),
            ("coupler_forward_loss_h", self.coupler_forward_loss_h_db),
            ("coupler_forward_loss_v", self.coupler_forward_loss_v_db),
            ("zdr_correction", self.zdr_correction_db),
            ("ldr_correction_h", self.ldr_correction_h_db),
            ("ldr_correction_v", self.ldr_correction_v_db),
            ("system_phidp", self.system_phidp_deg),
            ("test_power_h", self.test_power_h_dbm),
            ("test_power_v", self.test_power_v_dbm),
            ("receiver_slope_hc", self.receiver_slope_hc),
            ("receiver_slope_vc", self.receiver_slope_vc),
            ("receiver_slope_hx", self.receiver_slope_hx),
            ("receiver_slope_vx", self.receiver_slope_vx),
        ]
    }

    /// Set the float entry with the Table 301-14a `name` (one of the names
    /// [`Self::float_entries`] returns). Returns `false`, leaving `self`
    /// unchanged, when `name` is not one of them.
    pub fn set_float_entry(&mut self, name: &str, value: Option<f32>) -> bool {
        let slot: &mut Option<f32> = match name {
            "pulse_width" => &mut self.pulse_width_s,
            "antenna_gain_h" => &mut self.antenna_gain_h_db,
            "antenna_gain_v" => &mut self.antenna_gain_v_db,
            "xmit_power_h" => &mut self.xmit_power_h_dbm,
            "xmit_power_v" => &mut self.xmit_power_v_dbm,
            "two_way_waveguide_loss_h" => &mut self.two_way_waveguide_loss_h_db,
            "two_way_waveguide_loss_v" => &mut self.two_way_waveguide_loss_v_db,
            "two_way_radome_loss_h" => &mut self.two_way_radome_loss_h_db,
            "two_way_radome_loss_v" => &mut self.two_way_radome_loss_v_db,
            "receiver_mismatch_loss" => &mut self.receiver_mismatch_loss_db,
            "receiver_mismatch_loss_h" => &mut self.receiver_mismatch_loss_h_db,
            "receiver_mismatch_loss_v" => &mut self.receiver_mismatch_loss_v_db,
            "radar_constant_h" => &mut self.radar_constant_h,
            "radar_constant_v" => &mut self.radar_constant_v,
            "probert_jones_correction" => &mut self.probert_jones_correction,
            "dielectric_factor_used" => &mut self.dielectric_factor_used,
            "noise_hc" => &mut self.noise_hc_dbm,
            "noise_vc" => &mut self.noise_vc_dbm,
            "noise_hx" => &mut self.noise_hx_dbm,
            "noise_vx" => &mut self.noise_vx_dbm,
            "receiver_gain_hc" => &mut self.receiver_gain_hc_db,
            "receiver_gain_vc" => &mut self.receiver_gain_vc_db,
            "receiver_gain_hx" => &mut self.receiver_gain_hx_db,
            "receiver_gain_vx" => &mut self.receiver_gain_vx_db,
            "base_1km_hc" => &mut self.base_1km_hc_dbz,
            "base_1km_vc" => &mut self.base_1km_vc_dbz,
            "base_1km_hx" => &mut self.base_1km_hx_dbz,
            "base_1km_vx" => &mut self.base_1km_vx_dbz,
            "sun_power_hc" => &mut self.sun_power_hc_dbm,
            "sun_power_vc" => &mut self.sun_power_vc_dbm,
            "sun_power_hx" => &mut self.sun_power_hx_dbm,
            "sun_power_vx" => &mut self.sun_power_vx_dbm,
            "noise_source_power_h" => &mut self.noise_source_power_h_dbm,
            "noise_source_power_v" => &mut self.noise_source_power_v_dbm,
            "power_measure_loss_h" => &mut self.power_measure_loss_h_db,
            "power_measure_loss_v" => &mut self.power_measure_loss_v_db,
            "coupler_forward_loss_h" => &mut self.coupler_forward_loss_h_db,
            "coupler_forward_loss_v" => &mut self.coupler_forward_loss_v_db,
            "zdr_correction" => &mut self.zdr_correction_db,
            "ldr_correction_h" => &mut self.ldr_correction_h_db,
            "ldr_correction_v" => &mut self.ldr_correction_v_db,
            "system_phidp" => &mut self.system_phidp_deg,
            "test_power_h" => &mut self.test_power_h_dbm,
            "test_power_v" => &mut self.test_power_v_dbm,
            "receiver_slope_hc" => &mut self.receiver_slope_hc,
            "receiver_slope_vc" => &mut self.receiver_slope_vc,
            "receiver_slope_hx" => &mut self.receiver_slope_hx,
            "receiver_slope_vx" => &mut self.receiver_slope_vx,
            _ => return false,
        };
        *slot = value;
        true
    }
}

/// Where a volume came from: source format and file, container version and
/// conventions, compression and decode counts. The FM301 view does not export
/// these as variables.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Provenance {
    /// The format the volume was decoded from.
    pub source_format: SourceFormat,
    /// The file (or archive member) the volume was read from, when there was one.
    pub source_path: Option<String>,
    /// As written: "AR2V0006", "ARCHIVE2.036", "H5rad 2.3", "CF-Radial-1.3".
    pub source_version: Option<String>,
    /// The source's `Conventions` ("ODIM_H5/V2_2", "CF-1.6").
    pub source_conventions: Option<String>,
    /// The container's compression or encoding, as the decoder names it (for
    /// example `bzip2-whole-file`, `jma-grib2-tar`).
    pub compression: Option<String>,
    /// What the decoder read, decoded and skipped.
    pub decode: DecodeStats,
    /// Free text carried by BowEcho-written CfRadial files.
    pub polarization_note: Option<String>,
    /// Free text about the calibration, carried by BowEcho-written CfRadial files.
    pub calibration_note: Option<String>,
}

/// A format a volume can be decoded from.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[non_exhaustive]
pub enum SourceFormat {
    /// NEXRAD Archive II (Level II).
    NexradLevel2,
    /// NEXRAD or TDWR Level III product.
    NexradLevel3,
    /// ODIM_H5 (OPERA Data Information Model, HDF5).
    OdimH5,
    /// CfRadial 1.x (netCDF).
    CfRadial1,
    /// CfRadial 2 / WMO FM301 (netCDF-4).
    CfRadial2,
    /// DORADE sweepfile.
    Dorade,
    /// Japan Meteorological Agency polar radar GRIB2.
    JmaGrib2,
    /// Meteo-France polar radar BUFR (PAG, PAM).
    MeteoFranceBufr,
    /// A simulated volume (a forward operator's output).
    Simulated,
    /// Not known (the default of a volume built by hand).
    #[default]
    Unknown,
}

/// Decode counts. What a message is depends on the format: a Level II
/// message, a CfRadial sweep, a DORADE ray block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DecodeStats {
    /// Messages (or the format's unit of decoding) the decoder read.
    pub message_count: usize,
    /// Rays decoded into the volume.
    pub decoded_ray_count: usize,
    /// Messages the decoder skipped as malformed or out of scope instead of failing.
    pub skipped_message_count: usize,
}

/// Provenance of a simulated volume: the model and forward operator that produced it.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimulationProvenance {
    /// The forward operator that simulated the radar observations.
    pub forward_operator: Option<String>,
    /// The forward operator's configuration, as written.
    pub forward_operator_config: Option<String>,
    /// The numerical weather model the simulation started from.
    pub source_model: Option<String>,
    /// The model's microphysics scheme.
    pub microphysics_scheme: Option<String>,
    /// The scattering model the forward operator used.
    pub scattering_model: Option<String>,
}

/// `/georeferencing_correction`: one `Option<f32>` per name in xradar's
/// `georeferencing_correction_subgroup`.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct GeoreferencingCorrection {
    /// `azimuth_correction`: added to the antenna azimuth, degrees.
    pub azimuth_correction: Option<f32>,
    /// `elevation_correction`: added to the antenna elevation, degrees.
    pub elevation_correction: Option<f32>,
    /// `range_correction`: added to the gate ranges, metres.
    pub range_correction: Option<f32>,
    /// `longitude_correction`: added to the platform longitude, degrees.
    pub longitude_correction: Option<f32>,
    /// `latitude_correction`: added to the platform latitude, degrees.
    pub latitude_correction: Option<f32>,
    /// `pressure_altitude_correction`: added to the pressure altitude, metres.
    pub pressure_altitude_correction: Option<f32>,
    /// `radar_altitude_correction`: added to the radar altitude, metres.
    pub radar_altitude_correction: Option<f32>,
    /// `eastward_ground_speed_correction`: added to the platform's eastward ground speed, m/s.
    pub eastward_ground_speed_correction: Option<f32>,
    /// `northward_ground_speed_correction`: added to the platform's northward ground speed, m/s.
    pub northward_ground_speed_correction: Option<f32>,
    /// `vertical_velocity_correction`: added to the platform's vertical velocity, m/s.
    pub vertical_velocity_correction: Option<f32>,
    /// `heading_correction`: added to the platform heading, degrees.
    pub heading_correction: Option<f32>,
    /// `roll_correction`: added to the platform roll, degrees.
    pub roll_correction: Option<f32>,
    /// `pitch_correction`: added to the platform pitch, degrees.
    pub pitch_correction: Option<f32>,
    /// `drift_correction`: added to the platform drift angle, degrees.
    pub drift_correction: Option<f32>,
    /// `rotation_correction`: added to the antenna rotation angle, degrees.
    pub rotation_correction: Option<f32>,
    /// `tilt_correction`: added to the antenna tilt angle, degrees.
    pub tilt_correction: Option<f32>,
}

impl GeoreferencingCorrection {
    /// `(name, value)` in xradar's order.
    pub fn entries(&self) -> [(&'static str, Option<f32>); 16] {
        [
            ("azimuth_correction", self.azimuth_correction),
            ("elevation_correction", self.elevation_correction),
            ("range_correction", self.range_correction),
            ("longitude_correction", self.longitude_correction),
            ("latitude_correction", self.latitude_correction),
            (
                "pressure_altitude_correction",
                self.pressure_altitude_correction,
            ),
            ("radar_altitude_correction", self.radar_altitude_correction),
            (
                "eastward_ground_speed_correction",
                self.eastward_ground_speed_correction,
            ),
            (
                "northward_ground_speed_correction",
                self.northward_ground_speed_correction,
            ),
            (
                "vertical_velocity_correction",
                self.vertical_velocity_correction,
            ),
            ("heading_correction", self.heading_correction),
            ("roll_correction", self.roll_correction),
            ("pitch_correction", self.pitch_correction),
            ("drift_correction", self.drift_correction),
            ("rotation_correction", self.rotation_correction),
            ("tilt_correction", self.tilt_correction),
        ]
    }
}

/// `time` floored to the whole second (the form of `Volume::time_reference`).
pub fn floor_to_second(time: DateTime<Utc>) -> DateTime<Utc> {
    time.with_nanosecond(0).unwrap_or(time)
}

impl Volume {
    /// An empty fixed-platform radar volume. `time_reference` is floored to the
    /// whole second.
    pub fn new(instrument_name: impl Into<String>, time_reference: DateTime<Utc>) -> Self {
        Self {
            attrs: GlobalAttrs {
                instrument_name: instrument_name.into(),
                ..GlobalAttrs::default()
            },
            volume_number: None,
            time_reference: floor_to_second(time_reference),
            time_coverage: None,
            location: Location::default(),
            platform_type: PlatformType::Fixed,
            instrument_type: InstrumentType::Radar,
            primary_axis: None,
            status_str: None,
            scan: ScanStrategy::default(),
            radar_parameters: RadarParameters::default(),
            radar_calibration: Vec::new(),
            georeferencing_correction: None,
            extra_vars: Vec::new(),
            variable_attrs: Vec::new(),
            provenance: Provenance::default(),
            simulation: None,
            sweeps: Vec::new(),
        }
    }

    /// Absolute time of a `time_s` offset.
    pub fn instant(&self, time_s: f64) -> Option<DateTime<Utc>> {
        if !time_s.is_finite() {
            return None;
        }
        let nanos = (time_s * 1e9).round();
        if nanos.abs() > i64::MAX as f64 {
            return None;
        }
        self.time_reference
            .checked_add_signed(Duration::nanoseconds(nanos as i64))
    }

    /// Absolute time of ray `ray` of sweep `sweep`.
    pub fn ray_time(&self, sweep: usize, ray: usize) -> Option<DateTime<Utc>> {
        self.instant(*self.sweeps.get(sweep)?.rays.time_s.get(ray)?)
    }

    /// [`Sweep::tilt_elevation_deg`] of sweep `sweep` for this volume's
    /// source format; `None` when the sweep does not exist.
    pub fn tilt_elevation_deg(&self, sweep: usize) -> Option<f32> {
        Some(
            self.sweeps
                .get(sweep)?
                .tilt_elevation_deg(self.provenance.source_format),
        )
    }

    /// First and last ray time over all sweeps (by value, not storage order).
    pub fn ray_time_extent(&self) -> Option<TimeCoverage> {
        let mut extent: Option<(f64, f64)> = None;
        for time in self
            .sweeps
            .iter()
            .flat_map(|sweep| sweep.rays.time_s.iter().copied())
            .filter(|time| time.is_finite())
        {
            extent = Some(match extent {
                None => (time, time),
                Some((lo, hi)) => (lo.min(time), hi.max(time)),
            });
        }
        let (lo, hi) = extent?;
        Some(TimeCoverage {
            start: self.instant(lo)?,
            end: self.instant(hi)?,
        })
    }

    /// Seal every sweep and check `sweeps[i].sweep_number == i`.
    pub fn seal(&mut self) -> Result<(), SweepError> {
        for (index, sweep) in self.sweeps.iter_mut().enumerate() {
            if sweep.sweep_number as usize != index {
                return Err(SweepError::SweepNumber {
                    index,
                    sweep_number: sweep.sweep_number,
                });
            }
            sweep.seal()?;
        }
        Ok(())
    }
}

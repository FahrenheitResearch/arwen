//! Per-radial Level II values the FM301 coordinates have no place for,
//! carried into the model (`docs/design/fm301-model.md` section 9).
//!
//! The volume decoder reads these from every radial and keeps one column per
//! value and sweep; [`RayColumns::attach`] turns the columns into the sweep's
//! per-ray variables when the volume is finished. Where FM301 has a standard
//! name the value goes there, otherwise it is a `nexrad_*` variable or sweep
//! attribute named after the ICD field, in the ICD's units:
//!
//! | Source | Model |
//! |---|---|
//! | Message 31 byte 21, Message 1 halfword 7: radial status | `nexrad_radial_status(time)` |
//! | Message 31 bytes 10-11, Message 1 bytes 10-11: azimuth number | `nexrad_azimuth_number(time)` |
//! | Message 31 byte 23, Message 1 bytes 30-31: cut sector number | `nexrad_cut_sector_number(time)` |
//! | Message 31 byte 28, Message 1 bytes 66-67: spot blanking status | `nexrad_spot_blanking_status(time)` |
//! | Message header byte 2 (ICD 2620002 Table II halfword 2, high byte: the RDA redundant channel) | sweep attribute `nexrad_message_channels` |
//! | Message header halfwords 1 and 3-8: size, sequence number, generation date and time, segment count and number | `nexrad_message_size(time)`, `nexrad_message_sequence_number(time)`, `nexrad_message_date(time)`, `nexrad_message_milliseconds(time)`, `nexrad_message_segments(time)`, `nexrad_message_segment_number(time)`, as stored |
//! | Message 31 byte 20: azimuthal spacing | `Sweep::rays_angle_resolution_deg` |
//! | Message 31 byte 29: azimuth indexing angle | `Sweep::rays_are_indexed`, sweep attribute `nexrad_azimuth_indexing_angle_deg` |
//! | RAD bytes 8-15: noise levels | `nexrad_horizontal_noise_level(time)`, `nexrad_vertical_noise_level(time)` |
//! | RAD bytes 18-19: radial flags | `nexrad_radial_flags(time)` |
//! | RAD bytes 20-27: calibration constants | `nexrad_horizontal_calibration_constant(time)`, `nexrad_vertical_calibration_constant(time)` |
//! | VOL bytes 20-23, 32-39: calibration constant (dBZ0), system ZDR, initial system PhiDP | `radar_calibration` `base_1km_hc`, `zdr_correction`, `system_phidp` (as LROSE Radx maps them), one entry per distinct set, with per-ray `calib_index` |
//! | VOL bytes 24-31: SHV transmitter power | monitoring `radar_measured_transmit_power_h` / `_v` (dBm) |
//! | VOL bytes 8-15: latitude and longitude | the volume `latitude` and `longitude` from the first radial that has them, and sweep attributes `nexrad_latitude_deg`, `nexrad_longitude_deg` |
//! | VOL, other values | sweep attributes `nexrad_*` (see below) |
//! | ELV | sweep attributes `nexrad_atmospheric_attenuation_db_per_km`, `nexrad_elevation_calibration_constant_db` |
//! | Message 1 bytes 32-35: calibration constant | `nexrad_calibration_constant(time)` |
//! | Message 1 bytes 62-63: atmospheric attenuation | `nexrad_atmospheric_attenuation(time)` |
//! | Message 1 bytes 64-65: threshold parameter (TOVER) | `nexrad_tover(time)` |
//!
//! The message header channel byte, the Message 31 azimuthal spacing and
//! indexing angle, the VOL values
//! other than the calibration values and transmitter powers (the position
//! included) and the ELV values are sweep attributes when every radial of
//! the sweep has the same ones (as in every file of the corpus: a scan of
//! 342 real volumes found no sweep whose radials differ). When a radial's
//! differ, each becomes a per-ray variable of the same name instead, so no
//! radial's value is lost; `rays_angle_resolution` and `rays_are_indexed`
//! then follow the first radial.
//!
//! Structural values (compression indicator, radial length, data block count
//! and pointers, block sizes) are consumed by the decoder and not carried.
//! The message header is carried whole; its generation time is the
//! message's, not the collection time the ray carries.
//!
//! Every ray costs the volume's `DecodeBudget` [`RAY_TABLE_BYTES`] for its
//! coordinates and these columns, and a sweep whose radials differ in a
//! value kept once per sweep is charged for the per-ray copies too.

use std::collections::HashMap;

use recast_radar_core::bounded_read::DecodeBudget;
use recast_radar_core::model::{
    ArrayBuf, AttrValue, ExtraVariable, RadarCalibration, Scalar, Sweep,
};

/// Which message a radial came from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RadialMessage {
    /// Message 1, Digital Radar Data (ICD 2620002B Table III).
    Legacy,
    /// Message 31, Digital Radar Data Generic Format (Table XVII).
    Generic,
}

/// Radial Data Constant block values beyond the Nyquist velocity and
/// unambiguous range (Table XVII-H).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RadialBlockConstants {
    /// Bytes 8-11, dBm.
    pub horizontal_noise_level_dbm: f32,
    /// Bytes 12-15, dBm.
    pub vertical_noise_level_dbm: f32,
    /// Bytes 18-19.
    pub radial_flags: u16,
    /// Bytes 20-27 of the 28-byte layout (Build 14.0 on), dBZ.
    pub calibration_dbz: Option<(f32, f32)>,
}

/// Volume Data Constant block (Table XVII-E) as the fast decoder reads it.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct VolumeConstants {
    /// Byte 6.
    pub version_major: u8,
    /// Byte 7.
    pub version_minor: u8,
    /// Bytes 8-11, degrees.
    pub latitude_deg: f32,
    /// Bytes 12-15, degrees.
    pub longitude_deg: f32,
    /// Bytes 16-17, m.
    pub site_height_m: i16,
    /// Bytes 18-19, m.
    pub feedhorn_height_m: u16,
    /// Bytes 20-23, dB.
    pub calibration_constant_db: f32,
    /// Bytes 24-27, kW.
    pub horizontal_tx_power_kw: f32,
    /// Bytes 28-31, kW.
    pub vertical_tx_power_kw: f32,
    /// Bytes 32-35, dB.
    pub system_zdr_db: f32,
    /// Bytes 36-39, degrees.
    pub initial_system_phidp_deg: f32,
    /// Bytes 40-41.
    pub vcp_number: u16,
    /// Bytes 42-43.
    pub processing_status: u16,
    /// Bytes 44-45 of the 52-byte layout (Build 20.0 on), encoded like the
    /// ZDR moment.
    pub zdr_bias_estimate_raw: Option<u16>,
}

/// Elevation Data Constant block (Table XVII-F).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ElevationConstants {
    /// Bytes 6-7, 0.001 dB/km.
    pub atmospheric_attenuation_raw: i16,
    /// Bytes 8-11, dB.
    pub calibration_constant_db: f32,
}

/// Message 1 values with no Message 31 counterpart in the columns above
/// (ICD 2620002B Table III, offsets as MetPy and Py-ART read them).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LegacyConstants {
    /// Bytes 32-35: system gain calibration constant, dB.
    pub calibration_constant_db: f32,
    /// Bytes 62-63: atmospheric attenuation, 0.001 dB/km.
    pub atmospheric_attenuation_raw: i16,
    /// Bytes 64-65: threshold parameter (TOVER), 0.1 dB.
    pub tover_raw: i16,
}

/// The message header values of a radial's message (ICD 2620002 Table II)
/// beyond its type and channel byte, as stored.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct RadialMessageHeader {
    /// Halfword 1: the message size in halfwords (65535: the size is in
    /// halfwords 7-8 instead).
    pub size_halfwords: u16,
    /// Halfword 3: the message sequence number.
    pub sequence_number: u16,
    /// Halfword 4: the generation date, days with 1 January 1970 as day 1.
    pub date: u16,
    /// Halfwords 5-6: the generation time, milliseconds past midnight.
    pub milliseconds: u32,
    /// Halfword 7: the number of message segments.
    pub segments: u16,
    /// Halfword 8: the message segment number.
    pub segment_number: u16,
}

impl RadialMessageHeader {
    pub fn of(header: &crate::MessageHeader) -> Self {
        Self {
            size_halfwords: header.size_halfwords,
            sequence_number: header.sequence_id,
            date: header.date,
            milliseconds: header.milliseconds,
            segments: header.segments,
            segment_number: header.segment_number,
        }
    }
}

/// Everything one radial contributes beyond its coordinates and moments.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RadialExtras {
    pub message: RadialMessage,
    /// Message header byte 2: the RDA redundant channel (channel number and
    /// the Open RDA flag).
    pub message_channels: u8,
    /// The other message header values.
    pub message_header: RadialMessageHeader,
    pub status_code: u16,
    pub azimuth_number: u16,
    pub cut_sector_number: u16,
    pub spot_blanking: u16,
    /// Message 31 byte 20 (1 = 0.5 degree, 2 = 1.0 degree).
    pub azimuth_spacing_code: Option<u8>,
    /// Message 31 byte 29, 0.01 degree (0 = no indexing).
    pub azimuth_indexing_raw: Option<u8>,
    pub radial: Option<RadialBlockConstants>,
    pub volume: Option<VolumeConstants>,
    pub elevation: Option<ElevationConstants>,
    /// The VOL ZDR bias estimate converted with this radial's ZDR moment
    /// (Table XVII-E notes 20 and 33).
    pub zdr_bias_estimate_db: Option<f32>,
    pub legacy: Option<LegacyConstants>,
    /// Message 31 bytes 0-3: the radar identifier.
    pub radar_identifier: Option<[u8; 4]>,
}

impl RadialExtras {
    /// A Message 31 radial with nothing but its header values.
    pub fn generic(status_code: u8, azimuth_number: u16, cut_sector: u8, spot: u8) -> Self {
        Self {
            message: RadialMessage::Generic,
            message_channels: 0,
            message_header: RadialMessageHeader::default(),
            status_code: u16::from(status_code),
            azimuth_number,
            cut_sector_number: u16::from(cut_sector),
            spot_blanking: u16::from(spot),
            azimuth_spacing_code: None,
            azimuth_indexing_raw: None,
            radial: None,
            volume: None,
            elevation: None,
            zdr_bias_estimate_db: None,
            legacy: None,
            radar_identifier: None,
        }
    }
}

/// The per-radial values written as sweep attributes when every radial of
/// the sweep has the same ones.
#[derive(Clone, Copy, Debug, Default)]
struct Opening {
    message_channels: u8,
    azimuth_spacing_code: Option<u8>,
    azimuth_indexing_raw: Option<u8>,
    volume: Option<VolumeConstants>,
    elevation: Option<ElevationConstants>,
    zdr_bias_estimate_db: Option<f32>,
}

impl Opening {
    fn of(extras: &RadialExtras) -> Self {
        Self {
            message_channels: extras.message_channels,
            azimuth_spacing_code: extras.azimuth_spacing_code,
            azimuth_indexing_raw: extras.azimuth_indexing_raw,
            volume: extras.volume,
            elevation: extras.elevation,
            zdr_bias_estimate_db: extras.zdr_bias_estimate_db,
        }
    }

    /// Equal as stored: floats compare by bit pattern. The VOL calibration
    /// values and transmitter powers are left out; they are carried per ray.
    fn same(&self, other: &Self) -> bool {
        let bits = |value: Option<f32>| value.map(f32::to_bits);
        let volume = |v: &Option<VolumeConstants>| {
            v.map(|v| {
                (
                    v.version_major,
                    v.version_minor,
                    v.latitude_deg.to_bits(),
                    v.longitude_deg.to_bits(),
                    v.site_height_m,
                    v.feedhorn_height_m,
                    v.vcp_number,
                    v.processing_status,
                    v.zdr_bias_estimate_raw,
                )
            })
        };
        let elevation = |e: &Option<ElevationConstants>| {
            e.map(|e| {
                (
                    e.atmospheric_attenuation_raw,
                    e.calibration_constant_db.to_bits(),
                )
            })
        };
        self.message_channels == other.message_channels
            && self.azimuth_spacing_code == other.azimuth_spacing_code
            && self.azimuth_indexing_raw == other.azimuth_indexing_raw
            && volume(&self.volume) == volume(&other.volume)
            && elevation(&self.elevation) == elevation(&other.elevation)
            && bits(self.zdr_bias_estimate_db) == bits(other.zdr_bias_estimate_db)
    }
}

/// The VOL calibration constant (dBZ0), system differential reflectivity
/// and initial system differential phase of a radial, as bit patterns.
pub(crate) type CalibrationKey = [u32; 3];

fn calibration_key(volume: &VolumeConstants) -> CalibrationKey {
    [
        volume.calibration_constant_db.to_bits(),
        volume.system_zdr_db.to_bits(),
        volume.initial_system_phidp_deg.to_bits(),
    ]
}

/// The `radar_calibration` entries of a volume, one per distinct
/// [`CalibrationKey`], in first-seen order.
#[derive(Clone, Debug, Default)]
pub(crate) struct CalibrationTable {
    entries: Vec<RadarCalibration>,
    index: HashMap<CalibrationKey, i32>,
}

impl CalibrationTable {
    /// The `calib_index` of `key`, adding an entry (charged to `budget`)
    /// the first time it is seen.
    fn index_of(&mut self, key: CalibrationKey, budget: &mut DecodeBudget) -> Result<i32, String> {
        if let Some(index) = self.index.get(&key) {
            return Ok(*index);
        }
        budget.charge(
            1,
            size_of::<RadarCalibration>() + size_of::<(CalibrationKey, i32)>(),
            "Level II calibration entries",
        )?;
        let index = i32::try_from(self.entries.len())
            .map_err(|_| "too many Level II calibration entries".to_owned())?;
        let [base_1km, zdr, phidp] = key.map(f32::from_bits);
        self.entries.push(RadarCalibration {
            base_1km_hc_dbz: Some(base_1km),
            zdr_correction_db: Some(zdr),
            system_phidp_deg: Some(phidp),
            ..RadarCalibration::default()
        });
        self.index.insert(key, index);
        Ok(index)
    }

    /// The entries, for `Volume::radar_calibration`.
    pub(crate) fn into_entries(self) -> Vec<RadarCalibration> {
        self.entries
    }
}

/// A per-ray value stored once while every ray has the same one, and as a
/// column from the first ray that differs.
#[derive(Clone, Debug)]
struct Uniform<T> {
    first: Option<T>,
    all: Option<Vec<T>>,
    rays: usize,
}

impl<T> Default for Uniform<T> {
    fn default() -> Self {
        Self {
            first: None,
            all: None,
            rays: 0,
        }
    }
}

impl<T: Copy> Uniform<T> {
    /// Record one ray's value. Returns how many values the column stores
    /// anew: none while every ray agrees, every ray so far when this is the
    /// first to differ, and one per ray after that.
    fn push(&mut self, value: T, same: impl Fn(&T, &T) -> bool) -> usize {
        let stored = if let Some(all) = &mut self.all {
            all.push(value);
            1
        } else if let Some(first) = self.first {
            if same(&first, &value) {
                0
            } else {
                let mut all = Vec::with_capacity(self.rays + 1);
                all.resize(self.rays, first);
                all.push(value);
                self.all = Some(all);
                self.rays + 1
            }
        } else {
            self.first = Some(value);
            0
        };
        self.rays += 1;
        stored
    }

    /// The value of ray `index`.
    fn get(&self, index: usize) -> Option<T> {
        match &self.all {
            Some(all) => all.get(index).copied(),
            None => self.first.filter(|_| index < self.rays),
        }
    }
}

/// One sweep's per-ray columns, one entry per ray.
#[derive(Clone, Debug, Default)]
pub(crate) struct RayColumns {
    /// The sweep-attribute values of every ray.
    opening: Uniform<Opening>,
    /// The VOL calibration values of every ray (`None`: no VOL block).
    calibration: Uniform<Option<CalibrationKey>>,
    /// The Message 31 radar identifier of every ray.
    radar_identifier: Uniform<Option<[u8; 4]>>,
    /// Some ray came from Message 1: the header columns are 16-bit.
    legacy_rays: bool,
    status: Vec<u16>,
    azimuth_number: Vec<u16>,
    cut_sector_number: Vec<u16>,
    spot_blanking: Vec<u16>,
    message_size: Vec<u16>,
    message_sequence_number: Vec<u16>,
    message_date: Vec<u16>,
    message_milliseconds: Vec<u32>,
    message_segments: Vec<u16>,
    message_segment_number: Vec<u16>,
    has_radial: bool,
    has_radial_calibration: bool,
    noise_h_dbm: Vec<f32>,
    noise_v_dbm: Vec<f32>,
    radial_flags: Vec<u16>,
    calibration_h_dbz: Vec<f32>,
    calibration_v_dbz: Vec<f32>,
    has_tx_power: bool,
    tx_power_h_kw: Vec<f32>,
    tx_power_v_kw: Vec<f32>,
    has_legacy: bool,
    legacy_calibration_db: Vec<f32>,
    legacy_atmospheric_attenuation_raw: Vec<i16>,
    legacy_tover_raw: Vec<i16>,
}

/// Fill of an unsigned 16-bit column for a ray that lacks the value.
const U16_FILL: u16 = u16::MAX;
/// Fill of a signed 16-bit column.
const I16_FILL: i16 = i16::MIN;

/// Upper bound of the bytes one ray holds in the sweep's ray table and in
/// the columns every ray fills: time, azimuth and elevation (16), Nyquist
/// velocity and unambiguous range (8), the four Message 31 header columns
/// (8), the six message header columns (14), `calib_index` (4), the RAD
/// columns (18), the transmitter powers (8) and the Message 1 columns (8).
pub(crate) const RAY_TABLE_BYTES: usize = 96;
/// Bytes per ray of the per-ray variables a sweep gets instead of its
/// opening sweep attributes when its radials differ
/// ([`attach_opening_per_ray`]: four `u8`, four `u16`, one `i16` and six
/// `f32` values, 38 bytes, rounded up).
const OPENING_COLUMN_BYTES: usize = 40;
/// Bytes per ray of the per-ray radar identifier text.
const IDENTIFIER_COLUMN_BYTES: usize = size_of::<Box<str>>() + 4;

impl RayColumns {
    /// Reserve room for `rays` rays in the columns every radial fills.
    pub fn reserve(&mut self, rays: usize) {
        self.status.reserve(rays);
        self.azimuth_number.reserve(rays);
        self.cut_sector_number.reserve(rays);
        self.spot_blanking.reserve(rays);
        self.message_size.reserve(rays);
        self.message_sequence_number.reserve(rays);
        self.message_date.reserve(rays);
        self.message_milliseconds.reserve(rays);
        self.message_segments.reserve(rays);
        self.message_segment_number.reserve(rays);
    }

    /// Record one ray. Every column gets one entry, a fill value where the
    /// radial lacks the item, so the columns stay aligned with the rays.
    ///
    /// Returns the bytes the ray added beyond [`RAY_TABLE_BYTES`]: those of
    /// the values kept once per sweep while every radial agrees, which the
    /// sweep holds per ray from the first radial that differs.
    pub fn push(&mut self, extras: &RadialExtras) -> usize {
        let openings_stored = self.opening.push(Opening::of(extras), Opening::same);
        let keys_stored = self
            .calibration
            .push(extras.volume.as_ref().map(calibration_key), |a, b| a == b);
        let identifiers_stored = self
            .radar_identifier
            .push(extras.radar_identifier, |a, b| a == b);
        self.legacy_rays |= extras.message == RadialMessage::Legacy;
        self.status.push(extras.status_code);
        self.azimuth_number.push(extras.azimuth_number);
        self.cut_sector_number.push(extras.cut_sector_number);
        self.spot_blanking.push(extras.spot_blanking);
        let header = &extras.message_header;
        self.message_size.push(header.size_halfwords);
        self.message_sequence_number.push(header.sequence_number);
        self.message_date.push(header.date);
        self.message_milliseconds.push(header.milliseconds);
        self.message_segments.push(header.segments);
        self.message_segment_number.push(header.segment_number);

        let rays_before = self.status.len() - 1;
        let radial = extras.radial;
        if radial.is_some() && !self.has_radial {
            self.has_radial = true;
            fill(&mut self.noise_h_dbm, rays_before, f32::NAN);
            fill(&mut self.noise_v_dbm, rays_before, f32::NAN);
            fill(&mut self.radial_flags, rays_before, U16_FILL);
        }
        if self.has_radial {
            self.noise_h_dbm
                .push(radial.map_or(f32::NAN, |r| r.horizontal_noise_level_dbm));
            self.noise_v_dbm
                .push(radial.map_or(f32::NAN, |r| r.vertical_noise_level_dbm));
            self.radial_flags
                .push(radial.map_or(U16_FILL, |r| r.radial_flags));
        }
        let calibration = radial.and_then(|r| r.calibration_dbz);
        if calibration.is_some() && !self.has_radial_calibration {
            self.has_radial_calibration = true;
            fill(&mut self.calibration_h_dbz, rays_before, f32::NAN);
            fill(&mut self.calibration_v_dbz, rays_before, f32::NAN);
        }
        if self.has_radial_calibration {
            self.calibration_h_dbz
                .push(calibration.map_or(f32::NAN, |(h, _)| h));
            self.calibration_v_dbz
                .push(calibration.map_or(f32::NAN, |(_, v)| v));
        }

        let volume = extras.volume;
        if volume.is_some() && !self.has_tx_power {
            self.has_tx_power = true;
            fill(&mut self.tx_power_h_kw, rays_before, f32::NAN);
            fill(&mut self.tx_power_v_kw, rays_before, f32::NAN);
        }
        if self.has_tx_power {
            self.tx_power_h_kw
                .push(volume.map_or(f32::NAN, |v| v.horizontal_tx_power_kw));
            self.tx_power_v_kw
                .push(volume.map_or(f32::NAN, |v| v.vertical_tx_power_kw));
        }

        let legacy = extras.legacy;
        if legacy.is_some() && !self.has_legacy {
            self.has_legacy = true;
            fill(&mut self.legacy_calibration_db, rays_before, f32::NAN);
            fill(
                &mut self.legacy_atmospheric_attenuation_raw,
                rays_before,
                I16_FILL,
            );
            fill(&mut self.legacy_tover_raw, rays_before, I16_FILL);
        }
        if self.has_legacy {
            self.legacy_calibration_db
                .push(legacy.map_or(f32::NAN, |l| l.calibration_constant_db));
            self.legacy_atmospheric_attenuation_raw
                .push(legacy.map_or(I16_FILL, |l| l.atmospheric_attenuation_raw));
            self.legacy_tover_raw
                .push(legacy.map_or(I16_FILL, |l| l.tover_raw));
        }
        openings_stored * size_of::<Opening>()
            + keys_stored * size_of::<Option<CalibrationKey>>()
            + identifiers_stored * size_of::<Option<[u8; 4]>>()
    }

    /// Write the columns into `sweep` as per-ray variables, monitoring
    /// variables and sweep attributes, and its VOL calibration values into
    /// `calibration` with the per-ray `calib_index`. The columns are moved,
    /// not copied. Errors when a new calibration entry exceeds `budget`.
    pub fn attach(
        self,
        sweep: &mut Sweep,
        calibration: &mut CalibrationTable,
        budget: &mut DecodeBudget,
    ) -> Result<(), String> {
        let nrays = self.status.len();
        if nrays == 0 || nrays != sweep.nrays() {
            return Ok(());
        }
        let fallback_columns = usize::from(self.opening.all.is_some()) * OPENING_COLUMN_BYTES
            + usize::from(self.radar_identifier.all.is_some()) * IDENTIFIER_COLUMN_BYTES;
        if fallback_columns > 0 {
            budget.charge(nrays, fallback_columns, "Level II per-ray sweep values")?;
        }
        if let Some(first) = self.opening.first {
            match &self.opening.all {
                None => attach_opening(sweep, &first),
                Some(all) => attach_opening_per_ray(sweep, &first, all),
            }
        }
        if (0..nrays).any(|ray| self.calibration.get(ray).flatten().is_some()) {
            let mut indices = Vec::with_capacity(nrays);
            let mut uniform: Option<i32> = None;
            for ray in 0..nrays {
                let index = match self.calibration.get(ray).flatten() {
                    Some(key) if self.calibration.all.is_none() => match uniform {
                        Some(index) => index,
                        None => {
                            let index = calibration.index_of(key, budget)?;
                            uniform = Some(index);
                            index
                        }
                    },
                    Some(key) => calibration.index_of(key, budget)?,
                    None => CALIB_INDEX_FILL,
                };
                indices.push(index);
            }
            sweep.ray_vars.calib_index = Some(indices);
        }
        // The volume keeps the first radial's identifier; a sweep whose
        // radials do not all carry it has them per ray.
        if let Some(all) = &self.radar_identifier.all {
            sweep.extra_vars.push(per_ray(
                "nexrad_radar_identifier",
                ArrayBuf::Text(
                    all.iter()
                        .map(|identifier| {
                            identifier
                                .map(|bytes| crate::ascii_trim(&bytes))
                                .unwrap_or_default()
                                .into()
                        })
                        .collect(),
                ),
                nrays,
                "radar identifier",
                "ICD 2620002 Table XVII-A bytes 0-3 (Message 31), per radial: the radials of this sweep do not all agree",
            ));
        }
        let wide = self.legacy_rays;
        let status_comment = "ICD 2620002 Table XVII-A byte 21 (Message 31; bit 7 set marks bad data) or Table III halfword 7 (Message 1)";
        let mut status = header_column(
            "nexrad_radial_status",
            self.status,
            wide,
            "radial status",
            status_comment,
        );
        // CF flags with masks and values: the status in the low seven bits,
        // the bad-data flag in bit 7, so every stored byte is described.
        const STATUS_MASKS: [u8; 7] = [0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x80];
        const STATUS_VALUES: [u8; 7] = [0, 1, 2, 3, 4, 5, 0x80];
        let widen = |values: [u8; 7]| values.map(u16::from).to_vec();
        status.attrs.push((
            "flag_masks".into(),
            AttrValue::Array(if wide {
                ArrayBuf::U16(widen(STATUS_MASKS))
            } else {
                ArrayBuf::U8(STATUS_MASKS.to_vec())
            }),
        ));
        status.attrs.push((
            "flag_values".into(),
            AttrValue::Array(if wide {
                ArrayBuf::U16(widen(STATUS_VALUES))
            } else {
                ArrayBuf::U8(STATUS_VALUES.to_vec())
            }),
        ));
        status.attrs.push((
            "flag_meanings".into(),
            AttrValue::text(
                "start_of_elevation intermediate end_of_elevation start_of_volume end_of_volume start_of_elevation_last_cut bad_data",
            ),
        ));
        sweep.extra_vars.push(status);
        sweep.extra_vars.push(header_column(
            "nexrad_azimuth_number",
            self.azimuth_number,
            // The azimuth number is 16-bit in both messages.
            true,
            "radial number within the elevation scan",
            "ICD 2620002 Table XVII-A bytes 10-11 (Message 31) or Table III halfword 6 (Message 1)",
        ));
        sweep.extra_vars.push(header_column(
            "nexrad_cut_sector_number",
            self.cut_sector_number,
            wide,
            "sector number within the elevation cut",
            "ICD 2620002 Table XVII-A byte 23 (Message 31) or Table III halfword 16 (Message 1)",
        ));
        let mut spot = header_column(
            "nexrad_spot_blanking_status",
            self.spot_blanking,
            wide,
            "radial spot blanking status",
            "ICD 2620002 Table XVII-A byte 28 (Message 31) or Table III halfword 34 (Message 1)",
        );
        spot.attrs.push((
            "flag_masks".into(),
            AttrValue::Array(if wide {
                ArrayBuf::U16(vec![1, 2, 4])
            } else {
                ArrayBuf::U8(vec![1, 2, 4])
            }),
        ));
        spot.attrs.push((
            "flag_meanings".into(),
            AttrValue::text("radial_blanked elevation_has_blanked_radials volume_blanking_enabled"),
        ));
        sweep.extra_vars.push(spot);
        attach_message_header(
            sweep,
            MessageHeaderColumns {
                size: self.message_size,
                sequence_number: self.message_sequence_number,
                date: self.message_date,
                milliseconds: self.message_milliseconds,
                segments: self.message_segments,
                segment_number: self.message_segment_number,
            },
        );

        if self.has_radial {
            sweep.extra_vars.push(float_column(
                "nexrad_horizontal_noise_level",
                self.noise_h_dbm,
                "dBm",
                "horizontal channel noise level",
                "ICD 2620002 Table XVII-H bytes 8-11",
            ));
            sweep.extra_vars.push(float_column(
                "nexrad_vertical_noise_level",
                self.noise_v_dbm,
                "dBm",
                "vertical channel noise level",
                "ICD 2620002 Table XVII-H bytes 12-15",
            ));
            let mut flags = per_ray(
                "nexrad_radial_flags",
                ArrayBuf::U16(self.radial_flags),
                nrays,
                "radial flags for RPG processing",
                "ICD 2620002 Table XVII-H bytes 18-19",
            );
            flags.attrs.push((
                "_FillValue".into(),
                AttrValue::Scalar(Scalar::U16(U16_FILL)),
            ));
            sweep.extra_vars.push(flags);
        }
        if self.has_radial_calibration {
            sweep.extra_vars.push(float_column(
                "nexrad_horizontal_calibration_constant",
                self.calibration_h_dbz,
                "dBZ",
                "horizontal channel calibration constant (dBZ0)",
                "ICD 2620002 Table XVII-H bytes 20-23",
            ));
            sweep.extra_vars.push(float_column(
                "nexrad_vertical_calibration_constant",
                self.calibration_v_dbz,
                "dBZ",
                "vertical channel calibration constant (dBZ0)",
                "ICD 2620002 Table XVII-H bytes 24-27",
            ));
        }
        if self.has_tx_power {
            let monitoring = sweep.monitoring.get_or_insert_with(Box::default);
            monitoring.radar_measured_transmit_power_h_dbm =
                Some(self.tx_power_h_kw.into_iter().map(kw_to_dbm).collect());
            monitoring.radar_measured_transmit_power_v_dbm =
                Some(self.tx_power_v_kw.into_iter().map(kw_to_dbm).collect());
        }
        if self.has_legacy {
            sweep.extra_vars.push(float_column(
                "nexrad_calibration_constant",
                self.legacy_calibration_db,
                "dB",
                "system gain calibration constant",
                "ICD 2620002 Table III halfwords 17-18 (Message 1)",
            ));
            sweep.extra_vars.push(scaled_i16_column(
                "nexrad_atmospheric_attenuation",
                self.legacy_atmospheric_attenuation_raw,
                0.001,
                "dB/km",
                "atmospheric attenuation factor",
                "ICD 2620002 Table III halfword 32 (Message 1)",
            ));
            sweep.extra_vars.push(scaled_i16_column(
                "nexrad_tover",
                self.legacy_tover_raw,
                0.1,
                "dB",
                "threshold parameter (TOVER)",
                "ICD 2620002 Table III halfword 33 (Message 1)",
            ));
        }
        Ok(())
    }
}

/// `calib_index` of a ray without a VOL block in a sweep whose other rays
/// have one.
const CALIB_INDEX_FILL: i32 = -1;

/// The message header columns of a sweep, one entry per ray.
struct MessageHeaderColumns {
    size: Vec<u16>,
    sequence_number: Vec<u16>,
    date: Vec<u16>,
    milliseconds: Vec<u32>,
    segments: Vec<u16>,
    segment_number: Vec<u16>,
}

/// Each radial's message header values (ICD 2620002 Table II) as per-ray
/// variables, as stored: `nexrad_message_size`, `_sequence_number`,
/// `_date`, `_milliseconds`, `_segments` and `_segment_number`. The date
/// and time are when the message was generated, not the collection time the
/// ray's `time` holds.
fn attach_message_header(sweep: &mut Sweep, columns: MessageHeaderColumns) {
    let nrays = columns.size.len();
    let table = "ICD 2620002 Table II (message header) of each radial's Message 31 or Message 1";
    let comment = |location: &str| format!("{table}, {location}");
    let size = per_ray(
        "nexrad_message_size",
        ArrayBuf::U16(columns.size),
        nrays,
        "message size in halfwords",
        &comment("halfword 1; 65535 means halfwords 7-8 hold the message size in bytes"),
    );
    let sequence = per_ray(
        "nexrad_message_sequence_number",
        ArrayBuf::U16(columns.sequence_number),
        nrays,
        "message sequence number",
        &comment("halfword 3 (ID sequence, 0 to 0x7FFF, then wraps)"),
    );
    let mut date = per_ray(
        "nexrad_message_date",
        ArrayBuf::U16(columns.date),
        nrays,
        "message generation date",
        &comment("halfword 4 (modified Julian date, 1 January 1970 is day 1)"),
    );
    date.attrs.push((
        "units".into(),
        AttrValue::text("days since 1969-12-31T00:00:00Z"),
    ));
    let mut milliseconds = per_ray(
        "nexrad_message_milliseconds",
        ArrayBuf::U32(columns.milliseconds),
        nrays,
        "message generation time, milliseconds past midnight UTC of nexrad_message_date",
        &comment("halfwords 5-6"),
    );
    milliseconds
        .attrs
        .push(("units".into(), AttrValue::text("ms")));
    let segments = per_ray(
        "nexrad_message_segments",
        ArrayBuf::U16(columns.segments),
        nrays,
        "number of message segments",
        &comment("halfword 7 (with a size of 65535: the high halfword of the size in bytes)"),
    );
    let segment_number = per_ray(
        "nexrad_message_segment_number",
        ArrayBuf::U16(columns.segment_number),
        nrays,
        "message segment number",
        &comment("halfword 8 (with a size of 65535: the low halfword of the size in bytes)"),
    );
    sweep
        .extra_vars
        .extend([size, sequence, date, milliseconds, segments, segment_number]);
}

/// Transmitter power in kW as dBm (1 kW is 60 dBm; 0 kW is minus infinity,
/// so the source value is recoverable).
pub(crate) fn kw_to_dbm(kw: f32) -> f32 {
    10.0 * kw.log10() + 60.0
}

fn fill<T: Copy>(column: &mut Vec<T>, len: usize, value: T) {
    column.resize(len, value);
}

fn attach_opening(sweep: &mut Sweep, opening: &Opening) {
    sweep.other.push((
        "nexrad_message_channels".into(),
        AttrValue::Scalar(Scalar::U8(opening.message_channels)),
    ));
    if let Some(code) = opening.azimuth_spacing_code {
        sweep.rays_angle_resolution_deg = match code {
            1 => Some(0.5),
            2 => Some(1.0),
            _ => None,
        };
        sweep.other.push((
            "nexrad_azimuthal_spacing_code".into(),
            AttrValue::Scalar(Scalar::U8(code)),
        ));
    }
    if let Some(raw) = opening.azimuth_indexing_raw {
        sweep.rays_are_indexed = Some(raw != 0);
        sweep.other.push((
            "nexrad_azimuth_indexing_angle_deg".into(),
            AttrValue::Scalar(Scalar::F32(f32::from(raw) * 0.01)),
        ));
    }
    let attr = |name: &str, value: Scalar| (Box::<str>::from(name), AttrValue::Scalar(value));
    if let Some(volume) = &opening.volume {
        sweep.other.extend([
            attr(
                "nexrad_volume_block_version_major",
                Scalar::U8(volume.version_major),
            ),
            attr(
                "nexrad_volume_block_version_minor",
                Scalar::U8(volume.version_minor),
            ),
            attr("nexrad_latitude_deg", Scalar::F32(volume.latitude_deg)),
            attr("nexrad_longitude_deg", Scalar::F32(volume.longitude_deg)),
            attr("nexrad_site_height_m", Scalar::I16(volume.site_height_m)),
            attr(
                "nexrad_feedhorn_height_m",
                Scalar::U16(volume.feedhorn_height_m),
            ),
            attr(
                "nexrad_volume_coverage_pattern",
                Scalar::U16(volume.vcp_number),
            ),
            attr(
                "nexrad_processing_status",
                Scalar::U16(volume.processing_status),
            ),
        ]);
        if let Some(raw) = volume.zdr_bias_estimate_raw {
            sweep
                .other
                .push(attr("nexrad_zdr_bias_estimate_raw", Scalar::U16(raw)));
        }
    }
    if let Some(db) = opening.zdr_bias_estimate_db {
        sweep
            .other
            .push(attr("nexrad_zdr_bias_estimate_db", Scalar::F32(db)));
    }
    if let Some(elevation) = &opening.elevation {
        sweep.other.extend([
            attr(
                "nexrad_atmospheric_attenuation_db_per_km",
                Scalar::F32(f32::from(elevation.atmospheric_attenuation_raw) * 0.001),
            ),
            attr(
                "nexrad_elevation_calibration_constant_db",
                Scalar::F32(elevation.calibration_constant_db),
            ),
        ]);
    }
}

/// The sweep-attribute values of [`attach_opening`] as per-ray variables of
/// the same names, for a sweep whose radials do not all agree. The FM301
/// sweep slots follow the first radial. A ray without the item has the
/// fill value (`u8`/`u16` maximum, `i16` minimum, NaN).
fn attach_opening_per_ray(sweep: &mut Sweep, first: &Opening, all: &[Opening]) {
    let nrays = all.len();
    if let Some(code) = first.azimuth_spacing_code {
        sweep.rays_angle_resolution_deg = match code {
            1 => Some(0.5),
            2 => Some(1.0),
            _ => None,
        };
    }
    if let Some(raw) = first.azimuth_indexing_raw {
        sweep.rays_are_indexed = Some(raw != 0);
    }
    let comment = "per radial: the radials of this sweep do not all agree";
    let u8s = |get: &dyn Fn(&Opening) -> Option<u8>| {
        ArrayBuf::U8(all.iter().map(|o| get(o).unwrap_or(u8::MAX)).collect())
    };
    let u16s = |get: &dyn Fn(&Opening) -> Option<u16>| {
        ArrayBuf::U16(all.iter().map(|o| get(o).unwrap_or(u16::MAX)).collect())
    };
    let f32s = |get: &dyn Fn(&Opening) -> Option<f32>| {
        ArrayBuf::F32(all.iter().map(|o| get(o).unwrap_or(f32::NAN)).collect())
    };
    let columns: Vec<(&str, ArrayBuf)> = vec![
        (
            "nexrad_message_channels",
            u8s(&|o| Some(o.message_channels)),
        ),
        (
            "nexrad_azimuthal_spacing_code",
            u8s(&|o| o.azimuth_spacing_code),
        ),
        (
            "nexrad_azimuth_indexing_angle_deg",
            f32s(&|o| o.azimuth_indexing_raw.map(|raw| f32::from(raw) * 0.01)),
        ),
        (
            "nexrad_volume_block_version_major",
            u8s(&|o| o.volume.map(|v| v.version_major)),
        ),
        (
            "nexrad_volume_block_version_minor",
            u8s(&|o| o.volume.map(|v| v.version_minor)),
        ),
        (
            "nexrad_latitude_deg",
            f32s(&|o| o.volume.map(|v| v.latitude_deg)),
        ),
        (
            "nexrad_longitude_deg",
            f32s(&|o| o.volume.map(|v| v.longitude_deg)),
        ),
        (
            "nexrad_site_height_m",
            ArrayBuf::I16(
                all.iter()
                    .map(|o| o.volume.map_or(I16_FILL, |v| v.site_height_m))
                    .collect(),
            ),
        ),
        (
            "nexrad_feedhorn_height_m",
            u16s(&|o| o.volume.map(|v| v.feedhorn_height_m)),
        ),
        (
            "nexrad_volume_coverage_pattern",
            u16s(&|o| o.volume.map(|v| v.vcp_number)),
        ),
        (
            "nexrad_processing_status",
            u16s(&|o| o.volume.map(|v| v.processing_status)),
        ),
        (
            "nexrad_zdr_bias_estimate_raw",
            u16s(&|o| o.volume.and_then(|v| v.zdr_bias_estimate_raw)),
        ),
        (
            "nexrad_zdr_bias_estimate_db",
            f32s(&|o| o.zdr_bias_estimate_db),
        ),
        (
            "nexrad_atmospheric_attenuation_db_per_km",
            f32s(&|o| {
                o.elevation
                    .map(|e| f32::from(e.atmospheric_attenuation_raw) * 0.001)
            }),
        ),
        (
            "nexrad_elevation_calibration_constant_db",
            f32s(&|o| o.elevation.map(|e| e.calibration_constant_db)),
        ),
    ];
    for (name, values) in columns {
        sweep
            .extra_vars
            .push(per_ray(name, values, nrays, name, comment));
    }
}

fn per_ray(
    name: &str,
    values: ArrayBuf,
    nrays: usize,
    long_name: &str,
    comment: &str,
) -> ExtraVariable {
    ExtraVariable {
        name: name.into(),
        dims: vec!["time".into()],
        shape: vec![u32::try_from(nrays).unwrap_or(u32::MAX)],
        values,
        attrs: vec![
            ("long_name".into(), AttrValue::text(long_name)),
            ("comment".into(), AttrValue::text(comment)),
        ],
    }
}

/// A header value column: 8-bit when every ray is a Message 31 radial
/// (whose fields are bytes), 16-bit when `wide`.
fn header_column(
    name: &str,
    values: Vec<u16>,
    wide: bool,
    long_name: &str,
    comment: &str,
) -> ExtraVariable {
    let nrays = values.len();
    let buf = if wide {
        ArrayBuf::U16(values)
    } else {
        // Message 31 header fields are single bytes, so the narrowing is
        // exact.
        ArrayBuf::U8(values.into_iter().map(|v| v as u8).collect())
    };
    per_ray(name, buf, nrays, long_name, comment)
}

fn float_column(
    name: &str,
    values: Vec<f32>,
    units: &str,
    long_name: &str,
    comment: &str,
) -> ExtraVariable {
    let nrays = values.len();
    // NaN marks a ray without the value. No NaN `_FillValue` attribute: an
    // attribute that is not equal to itself would make equal volumes compare
    // unequal.
    let mut variable = per_ray(name, ArrayBuf::F32(values), nrays, long_name, comment);
    variable
        .attrs
        .push(("units".into(), AttrValue::text(units)));
    variable
}

fn scaled_i16_column(
    name: &str,
    values: Vec<i16>,
    scale: f32,
    units: &str,
    long_name: &str,
    comment: &str,
) -> ExtraVariable {
    let nrays = values.len();
    let mut variable = per_ray(name, ArrayBuf::I16(values), nrays, long_name, comment);
    variable.attrs.extend([
        ("units".into(), AttrValue::text(units)),
        ("scale_factor".into(), AttrValue::Scalar(Scalar::F32(scale))),
        (
            "_FillValue".into(),
            AttrValue::Scalar(Scalar::I16(I16_FILL)),
        ),
    ]);
    variable
}

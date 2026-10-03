//! Level II volumes together with their NEXRAD metadata messages.
//!
//! [`read_volume_with_metadata`] returns the same [`Volume`] as
//! [`crate::read_volume_from_bytes`], plus a [`NexradMetadata`] holding the
//! typed metadata messages ([`crate::messages`]) and per-sweep message 31
//! constant blocks, which the FM301 model has no place for (design note
//! `docs/design/fm301-model.md` section 2: format metadata sits beside the
//! volume).
//!
//! Where each field comes from:
//!
//! - The metadata messages (2, 3, 5 or 7, 8, 13, 15, 18 and 32) are read from
//!   the Archive II metadata record only ([`messages::metadata_record`]: the
//!   first LDM record, or the first 134 frames of raw-record files), never
//!   from the data records. For each field the first message of its type
//!   that decodes is kept. Files from before the metadata record existed
//!   (ARCHIVE2 headers, 1991-2003) have radials in those frames and at most
//!   a message 2.
//! - [`NexradMetadata::per_sweep_elevation_data`] is filled while the volume
//!   is decoded, from the first message 31 radial of each cut, so the radial
//!   data is decompressed once. Only the metadata record is decompressed a
//!   second time.

use chrono::{DateTime, Utc};
use recast_radar_core::model::Volume;

use crate::messages::adaptation::RdaAdaptationData;
use crate::messages::bypass_map::ClutterFilterBypassMap;
use crate::messages::clutter_censor::ClutterCensorZones;
use crate::messages::clutter_filter_map::ClutterFilterMap;
use crate::messages::msg31_blocks::{
    DigitalRadarDataGeneric, ElevationDataBlock, RadialDataBlock, SpotBlankingStatus,
    VolumeDataBlock,
};
use crate::messages::performance::PerformanceMaintenance;
use crate::messages::prf::RdaPrfData;
use crate::messages::rda_status::{RdaBuild, RdaStatus};
use crate::messages::vcp::VolumeCoveragePattern;
use crate::messages::{self, MessageBody, RawMessages};
use crate::{RadialObserver, Result};

/// A decoded Level II volume and its NEXRAD metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct NexradVolume {
    /// The volume, identical to what [`crate::read_volume_from_bytes`]
    /// returns for the same bytes.
    pub volume: Volume,
    /// Metadata messages and per-sweep message 31 constant blocks.
    pub metadata: NexradMetadata,
}

/// NEXRAD metadata of one Level II volume. Every field is `None` when the
/// file has no such message (or none that decoded); see the module
/// documentation for where each comes from.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NexradMetadata {
    /// The Archive II volume header time (Table I), or, when the header
    /// date is the epoch (ARCHIVE2 files without a date, GR2 `.msg31`
    /// exports) or the volume is Message 1, the first radial's collection
    /// time. `Volume::time_reference` is the first radial's time floored to
    /// the second; this is what the header says. Set by
    /// [`read_volume_with_metadata`] only.
    pub volume_header_time: Option<DateTime<Utc>>,
    /// Message 2, RDA Status Data (Table IV).
    pub rda_status: Option<RdaStatus>,
    /// Message 3, Performance/Maintenance Data (Table V).
    pub performance: Option<Box<PerformanceMaintenance>>,
    /// Message 5 (or 7), Volume Coverage Pattern (Table XI).
    pub vcp: Option<VolumeCoveragePattern>,
    /// Message 18, RDA Adaptation Data (Table XV).
    pub adaptation: Option<Box<RdaAdaptationData>>,
    /// Message 15, Clutter Filter Map (Table XIV).
    pub clutter_filter_map: Option<ClutterFilterMap>,
    /// Message 13, Clutter Filter Bypass Map (Table IX).
    pub bypass_map: Option<ClutterFilterBypassMap>,
    /// Message 8, Clutter Censor Zones (Table XII). The RPG sends it to the
    /// RDA, so no Archive II file in the corpus has one.
    pub clutter_censor_zones: Option<ClutterCensorZones>,
    /// Message 32, RDA PRF Data (Table XVIII), sent from Build 23.
    pub prf: Option<RdaPrfData>,
    /// Message 31 constant blocks of each sweep's first radial, in the order
    /// of [`Volume::sweeps`]: one entry per sweep, except sweeps opened by a
    /// message 1 radial or by a radial whose blocks do not decode (listed in
    /// [`Self::errors`]). `None` when no radial of the volume is a message 31
    /// (Message 1 volumes).
    pub per_sweep_elevation_data: Option<Vec<SweepElevationData>>,
    /// RDA build from the message 2 above ([`RdaStatus::rda_build`]); `None`
    /// for the legacy RDA layout, which has no build field.
    pub build: Option<RdaBuild>,
    /// Problems met while reading the metadata, as display text: messages in
    /// the metadata record that could not be framed or decoded (for example
    /// the stale orphan segments of 2008 metadata records, or the zero-filled
    /// Message 15 of KLIX 2005), and cut-opening radials whose constant
    /// blocks did not decode. None of them fails the decode.
    pub errors: Vec<String>,
}

/// Message 31 constant blocks of the first radial of one sweep, and what
/// every radial of the sweep carried beyond the volume's model.
///
/// The ELV block is constant within a sweep. VOL and RAD blocks are sent with
/// every radial and can change within a sweep (noise levels, and the Nyquist
/// velocity of Doppler sectors), so these are the values at the sweep's
/// start; [`Self::radials`] has each radial's own.
#[derive(Clone, Debug, PartialEq)]
pub struct SweepElevationData {
    /// Index of the sweep in [`Volume::sweeps`].
    pub sweep_index: usize,
    /// Data Header Block elevation number (1-based cut number in the VCP).
    pub elevation_number: u8,
    /// Data Header Block elevation angle of the first radial, degrees.
    pub elevation_angle_deg: f32,
    /// Elevation Data Constant block (Table XVII-F): atmospheric attenuation
    /// and calibration constant.
    pub elevation: Option<ElevationDataBlock>,
    /// Volume Data Constant block (Table XVII-E).
    pub volume: Option<VolumeDataBlock>,
    /// Radial Data Constant block (Table XVII-H).
    pub radial: Option<RadialDataBlock>,
    /// The VOL block's ZDR bias estimate in dB, converted with the same
    /// radial's ZDR moment block (Table XVII-E notes 20 and 33; see
    /// [`VolumeDataBlock::zdr_bias_estimate_db`]). `None` without a VOL
    /// block, in the 44-byte VOL layout (before Build 20), or when the RPG
    /// reports it as not available.
    pub zdr_bias_estimate_db: Option<f32>,
    /// Every radial of the sweep, in ray order: the Data Header Block items
    /// and constant blocks the volume has no place for. The RAD block's
    /// noise levels and calibration constants change from radial to radial,
    /// and the VOL block's transmitter power on the last radials of a cut.
    /// Empty when a radial of the sweep did not decode, or went to the sweep
    /// out of ray order (listed in [`NexradMetadata::errors`]).
    pub radials: Vec<RadialConstants>,
}

/// What one Message 31 radial carried besides its time, angles, status,
/// Nyquist velocity, unambiguous range and moments (which the volume holds).
#[derive(Clone, Debug, PartialEq)]
pub struct RadialConstants {
    /// Data Header Block bytes 0-3: the radar identifier as recorded, which
    /// can differ from the volume header's (early Build 10 files such as
    /// KVWX 2008 leave it blank).
    pub radar_identifier: [u8; 4],
    /// Data Header Block bytes 10-11: radial number within the cut.
    pub azimuth_number: u16,
    /// Byte 17: spare.
    pub spare: u8,
    /// Byte 20: azimuth resolution spacing code as recorded (1 for 0.5
    /// degree, 2 for 1 degree; Table XVII-A).
    pub azimuth_resolution_code: u8,
    /// Byte 21: radial status code (Table III-C) as recorded, bad-data bit
    /// and codes outside the table included (KVWX 2008 has 8).
    pub radial_status_code: u8,
    /// Byte 23: sector number within the cut.
    pub cut_sector_number: u8,
    /// Byte 28: spot blanking status.
    pub spot_blanking: SpotBlankingStatus,
    /// Byte 29: azimuth indexing angle in 0.01 degree steps; 0 means no
    /// indexing.
    pub azimuth_indexing_raw: u8,
    /// The radial's Elevation Data Constant block (Table XVII-F).
    pub elevation: Option<ElevationDataBlock>,
    /// The radial's Volume Data Constant block (Table XVII-E).
    pub volume: Option<VolumeDataBlock>,
    /// The radial's Radial Data Constant block (Table XVII-H).
    pub radial: Option<RadialDataBlock>,
}

impl RadialConstants {
    fn of(radial: &DigitalRadarDataGeneric<'_>) -> Self {
        Self {
            radar_identifier: radial.header.radar_identifier,
            azimuth_number: radial.header.azimuth_number,
            spare: radial.header.spare,
            azimuth_resolution_code: radial.header.azimuth_resolution.code(),
            radial_status_code: radial.header.radial_status_code,
            cut_sector_number: radial.header.cut_sector_number,
            spot_blanking: radial.header.spot_blanking,
            azimuth_indexing_raw: radial.header.azimuth_indexing_raw,
            elevation: radial.elevation,
            volume: radial.volume,
            radial: radial.radial,
        }
    }
}

/// Decode a Level II volume and its NEXRAD metadata.
///
/// Accepts the same inputs as [`crate::read_volume_from_bytes`] and returns
/// the same volume, with the same errors. Problems in the metadata alone do
/// not fail the call; they are listed in [`NexradMetadata::errors`].
pub fn read_volume_with_metadata(bytes: &[u8]) -> Result<NexradVolume> {
    let mut sweeps = SweepCollector::default();
    let builder = crate::builder_observed(bytes, &mut sweeps)?;
    let header_time = builder.header_time;
    let volume = builder.finish()?.0;
    let mut metadata = NexradMetadata::from_metadata_record(bytes);
    metadata.volume_header_time = Some(header_time);
    metadata.per_sweep_elevation_data = sweeps.saw_message_31.then_some(sweeps.sweeps);
    metadata.errors.extend(sweeps.errors);
    Ok(NexradVolume { volume, metadata })
}

impl NexradMetadata {
    /// The metadata messages of a Level II file or real-time start chunk,
    /// read from its metadata record, without decoding the volume.
    /// [`Self::per_sweep_elevation_data`] stays `None`.
    pub fn from_metadata_record(bytes: &[u8]) -> Self {
        let mut metadata = Self::default();
        let record = match messages::metadata_record(bytes) {
            Ok(record) => record,
            Err(error) => {
                metadata.errors.push(format!("metadata record: {error}"));
                return metadata;
            }
        };
        for item in RawMessages::new(&record) {
            // Only the message types kept below are decoded: an RDA log
            // (message 33) is not inflated just to be dropped.
            let decoded = item.and_then(|raw| {
                if matches!(
                    raw.header.message_type,
                    2 | 3 | 5 | 7 | 8 | 13 | 15 | 18 | 32
                ) {
                    raw.decode().map(Some)
                } else {
                    Ok(None)
                }
            });
            let body = match decoded {
                Ok(Some((_, body))) => body,
                Ok(None) => continue,
                Err(error) => {
                    metadata.errors.push(format!("metadata record: {error}"));
                    continue;
                }
            };
            match body {
                MessageBody::RdaStatus(status) => set_first(&mut metadata.rda_status, status),
                MessageBody::Performance(data) => set_first(&mut metadata.performance, data),
                MessageBody::Vcp(vcp) => set_first(&mut metadata.vcp, vcp),
                MessageBody::Adaptation(data) => set_first(&mut metadata.adaptation, data),
                MessageBody::ClutterFilterMap(map) => {
                    set_first(&mut metadata.clutter_filter_map, map);
                }
                MessageBody::BypassMap(map) => set_first(&mut metadata.bypass_map, map),
                MessageBody::ClutterCensorZones(zones) => {
                    set_first(&mut metadata.clutter_censor_zones, zones);
                }
                MessageBody::Prf(prf) => set_first(&mut metadata.prf, prf),
                _ => {}
            }
        }
        metadata.build = metadata.rda_status.as_ref().and_then(RdaStatus::rda_build);
        metadata
    }
}

fn set_first<T>(slot: &mut Option<T>, value: T) {
    if slot.is_none() {
        *slot = Some(value);
    }
}

/// Collects the constant blocks of each sweep's first message 31 radial,
/// and every radial's own, while the volume decoder runs.
///
/// The decoder creates a sweep only for the radial that opens it (ray 0).
/// Later radials may go to earlier sweeps (for example out-of-order
/// real-time chunks), but never open one; a sweep opened by a message 1
/// radial (which the observer does not see) gets no entry.
#[derive(Default)]
struct SweepCollector {
    sweeps: Vec<SweepElevationData>,
    /// Sweeps whose radial list was dropped: a radial did not decode, or
    /// did not arrive in ray order.
    incomplete: Vec<usize>,
    errors: Vec<String>,
    saw_message_31: bool,
}

impl RadialObserver for SweepCollector {
    fn message_31(&mut self, body: &[u8], _volume: &Volume, sweep: usize, ray: usize) {
        self.saw_message_31 = true;
        // The opening radial is decoded whole (its ZDR block converts the
        // VOL block's ZDR bias); the others need their constant blocks only.
        let decoded = if ray == 0 {
            DigitalRadarDataGeneric::decode(body)
        } else {
            DigitalRadarDataGeneric::decode_constant_blocks(body)
        };
        if ray == 0 {
            match &decoded {
                Ok(radial) => self.sweeps.push(SweepElevationData {
                    sweep_index: sweep,
                    elevation_number: radial.header.elevation_number,
                    elevation_angle_deg: radial.header.elevation_angle_deg,
                    elevation: radial.elevation,
                    volume: radial.volume,
                    radial: radial.radial,
                    zdr_bias_estimate_db: radial.zdr_bias_estimate_db(),
                    radials: Vec::new(),
                }),
                Err(error) => {
                    self.errors
                        .push(format!("sweep {sweep} first radial: {error}"));
                    return;
                }
            }
        }
        if self.incomplete.contains(&sweep) {
            return;
        }
        let Some(entry) = self
            .sweeps
            .iter_mut()
            .rev()
            .find(|entry| entry.sweep_index == sweep)
        else {
            return;
        };
        let problem = match decoded {
            Ok(radial) if entry.radials.len() == ray => {
                entry.radials.push(RadialConstants::of(&radial));
                return;
            }
            Ok(_) => format!(
                "sweep {sweep} radial {ray} arrived after radial {}",
                entry.radials.len()
            ),
            Err(error) => format!("sweep {sweep} radial {ray}: {error}"),
        };
        entry.radials = Vec::new();
        self.incomplete.push(sweep);
        self.errors.push(problem);
    }
}

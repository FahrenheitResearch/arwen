//! Clutter Censor Zones (message 8, ICD 2620002AA Table XII).
//!
//! Operator-defined clutter map override regions that the RPG sends to the
//! RDA; the RDA then recomputes the Clutter Filter Map (message 15, see
//! [`super::clutter_filter_map`]) and sends it back. Archive II files record
//! messages from the RDA, so no file in the test corpus holds a message 8
//! and this decoder follows the ICD without a verified real sample.
//!
//! Layout (halfwords, big-endian): 1 number of override regions (0 to 25);
//! then 6 halfwords per region `i` (0-based) at halfwords `2 + 6i` to
//! `7 + 6i`: start range (km), stop range (km), start azimuth (degrees),
//! stop azimuth (degrees), elevation segment number, operator select code.
//! The legacy RDA layout of ICD 2620002B (8 halfwords per region, scaled
//! azimuths, per-channel suppression levels) is not decoded.

use std::borrow::Cow;

use super::MessageBody;
pub use super::clutter_filter_map::OperatorSelectCode;
use crate::{NexradError, Result};

/// Largest number of override regions (Table XII halfword 1).
pub const MAX_OVERRIDE_REGIONS: u16 = 25;

/// Halfwords per override region.
pub const REGION_HALFWORDS: usize = 6;

/// One clutter map override region (Table XII R1 to R6).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct CensorZone {
    /// R1: start range in km (0 to 511).
    pub start_range_km: u16,
    /// R2: stop range in km (0 to 511).
    pub stop_range_km: u16,
    /// R3: start azimuth in degrees clockwise from true north (0 to 360).
    pub start_azimuth_deg: u16,
    /// R4: stop azimuth in degrees clockwise from true north (0 to 360).
    pub stop_azimuth_deg: u16,
    /// R5: elevation segment number (1 to 5; 1 is closest to the ground).
    pub elevation_segment: u16,
    /// R6: clutter filter operation forced in the region.
    pub operator_select: OperatorSelectCode,
}

/// Decoded clutter censor zones message (Table XII).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClutterCensorZones {
    /// Override regions in the order sent; halfword 1 gives their number.
    pub regions: Vec<CensorZone>,
}

impl ClutterCensorZones {
    /// Decode a message body (the bytes after the 16-byte message header).
    ///
    /// Rejects a region count above 25 and a body too short for the declared
    /// regions. Region values are kept as sent.
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, 2, "clutter censor zone count")?;
        let count = crate::be_u16(body, 0);
        if count > MAX_OVERRIDE_REGIONS {
            return Err(NexradError::InvalidMessage {
                offset: 0,
                reason: format!("clutter censor zone count {count} exceeds {MAX_OVERRIDE_REGIONS}"),
            });
        }
        let regions_len = usize::from(count) * REGION_HALFWORDS * 2;
        crate::require_len(body, 2, regions_len, "clutter censor zones")?;
        let regions = body[2..2 + regions_len]
            .chunks_exact(REGION_HALFWORDS * 2)
            .map(|region| CensorZone {
                start_range_km: crate::be_u16(region, 0),
                stop_range_km: crate::be_u16(region, 2),
                start_azimuth_deg: crate::be_u16(region, 4),
                stop_azimuth_deg: crate::be_u16(region, 6),
                elevation_segment: crate::be_u16(region, 8),
                operator_select: OperatorSelectCode::from_code(crate::be_u16(region, 10)),
            })
            .collect();
        Ok(Self { regions })
    }
}

/// Walker hook: the typed body for message 8.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    ClutterCensorZones::decode(&body).map(MessageBody::ClutterCensorZones)
}

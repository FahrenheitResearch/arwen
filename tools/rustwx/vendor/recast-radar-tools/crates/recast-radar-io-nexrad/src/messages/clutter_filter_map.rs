//! Clutter Filter Map (message 15, ICD 2620002AA Table XIV).
//!
//! Metadata the RDA sends upon wideband connection and whenever the map
//! changes: for each elevation segment and each of 360 one-degree azimuth
//! segments, a list of range zones, each an operator select code and an end
//! range. Archive II files carry it in the metadata record, split into
//! segments that [`RawMessages`](super::RawMessages) joins.
//!
//! Layout (halfwords, big-endian): 1 map generation date (days, 1 January
//! 1970 = 1), 2 generation time (minutes after midnight UTC), 3 number of
//! elevation segments (1 to 5); then for each elevation segment, for each
//! azimuth segment 0 to 359: the number of range zones (1 to 20) followed by
//! that many (op code, end range in km) pairs.
//!
//! `tests/messages_clutter.rs` checks it against MetPy 1.7.1
//! `Level2File.clutter_filter_map` on 20 metadata records, from KPAH 2008
//! (Build 10) to KIWA 2026 (Build 24.1). Legacy RDA files use the same
//! message number for the older "Clutter Filter Notchwidth Map" (ICD
//! 2620002B Table XIV). That layout is not decoded, because the corpus has
//! no populated sample: the KLIX 2005 and KVWX 2008 messages are all zeros.
//! The decoder rejects them because their elevation segment count is 0.

use std::borrow::Cow;

use chrono::{DateTime, Utc};

use super::MessageBody;
use crate::{NexradError, Result};

/// Azimuth segments per elevation segment (Table XIV note 3).
pub const AZIMUTH_SEGMENTS: usize = 360;

/// Largest number of elevation segments (Table XIV halfword 3).
pub const MAX_ELEVATION_SEGMENTS: u16 = 5;

/// Largest number of range zones per azimuth segment (Table XIV note 4).
pub const MAX_RANGE_ZONES: u16 = 20;

/// End range of the last zone of every azimuth segment, in km (note 4).
pub const LAST_ZONE_END_RANGE_KM: u16 = 511;

const HEADER_HALFWORDS: usize = 3;

/// Clutter filter operator select code, shared by message 15 range zones
/// (Table XIV "Op Code") and message 8 censor zones (Table XII "Operator
/// Select Code").
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum OperatorSelectCode {
    /// 0: bypass the clutter filter (no filtering).
    BypassFilter,
    /// 1: the clutter filter bypass map decides per range bin.
    BypassMapInControl,
    /// 2: force clutter filtering.
    ForceFilter,
    /// Any other code.
    Unknown(u16),
}

impl OperatorSelectCode {
    /// Map a Table XII / XIV code.
    pub fn from_code(code: u16) -> Self {
        match code {
            0 => Self::BypassFilter,
            1 => Self::BypassMapInControl,
            2 => Self::ForceFilter,
            other => Self::Unknown(other),
        }
    }

    /// The Table XII / XIV code.
    pub fn code(self) -> u16 {
        match self {
            Self::BypassFilter => 0,
            Self::BypassMapInControl => 1,
            Self::ForceFilter => 2,
            Self::Unknown(code) => code,
        }
    }
}

/// One range zone of an azimuth segment (Table XIV R1, R2).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RangeZone {
    /// R1: clutter filter operation in this zone.
    pub op_code: OperatorSelectCode,
    /// R2: stop range of the zone in km (0 to 511). Zones are listed in
    /// increasing range; the first starts at the radar and the last ends at
    /// 511 km.
    pub end_range_km: u16,
}

/// Range zones of one elevation segment, for azimuth segments 0 to 359.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClutterMapSegment {
    zones: Vec<RangeZone>,
    /// `zones[azimuth_start[a]..azimuth_start[a + 1]]` belong to azimuth
    /// segment `a`; 361 entries.
    azimuth_start: Vec<usize>,
}

impl ClutterMapSegment {
    /// Range zones of azimuth segment `azimuth` (0 to 359), which covers
    /// azimuths from `azimuth` up to `azimuth + 1` degrees clockwise from
    /// true north; `None` past 359.
    pub fn azimuth(&self, azimuth: usize) -> Option<&[RangeZone]> {
        let start = *self.azimuth_start.get(azimuth)?;
        let end = *self.azimuth_start.get(azimuth + 1)?;
        self.zones.get(start..end)
    }

    /// Range zones of each azimuth segment, 0 to 359 in order.
    pub fn azimuths(&self) -> impl ExactSizeIterator<Item = &[RangeZone]> + '_ {
        self.azimuth_start
            .windows(2)
            .map(|bounds| &self.zones[bounds[0]..bounds[1]])
    }

    /// Number of azimuth segments (360 in a decoded map).
    pub fn azimuth_count(&self) -> usize {
        self.azimuth_start.len().saturating_sub(1)
    }

    /// Range zones of all azimuth segments, concatenated in azimuth order.
    pub fn zones(&self) -> &[RangeZone] {
        &self.zones
    }
}

/// Decoded clutter filter map (Table XIV).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClutterFilterMap {
    /// Halfword 1: map generation date in days, 1 January 1970 = 1.
    pub generation_date: u16,
    /// Halfword 2: map generation time in minutes after midnight UTC
    /// (0 to 1440).
    pub generation_minutes: u16,
    /// Elevation segments in halfword 3's count; the first is closest to the
    /// ground.
    pub segments: Vec<ClutterMapSegment>,
    /// Body bytes after the last elevation segment. Zero for a conforming
    /// message; non-zero when the walker joined stale segments, as in the
    /// Build 10 KPAH 2008-04-15 metadata record.
    pub trailing_bytes: usize,
}

impl ClutterFilterMap {
    /// Decode a message body (the bytes after the 16-byte message header,
    /// joined across segments).
    ///
    /// Rejects an elevation segment count outside 1 to 5, a range zone count
    /// outside 1 to 20, and a body too short for the counts it declares.
    /// Op codes and end ranges are kept as sent.
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, HEADER_HALFWORDS * 2, "clutter filter map header")?;
        let segment_count = crate::be_u16(body, 4);
        if !(1..=MAX_ELEVATION_SEGMENTS).contains(&segment_count) {
            return Err(NexradError::InvalidMessage {
                offset: 4,
                reason: format!(
                    "clutter filter map elevation segment count {segment_count} outside 1..={MAX_ELEVATION_SEGMENTS}"
                ),
            });
        }
        let mut cursor = HEADER_HALFWORDS * 2;
        let mut segments = Vec::with_capacity(usize::from(segment_count));
        for _ in 0..segment_count {
            let mut zones = Vec::with_capacity(AZIMUTH_SEGMENTS);
            let mut azimuth_start = Vec::with_capacity(AZIMUTH_SEGMENTS + 1);
            azimuth_start.push(0);
            for _ in 0..AZIMUTH_SEGMENTS {
                crate::require_len(body, cursor, 2, "clutter filter map range zone count")?;
                let zone_count = crate::be_u16(body, cursor);
                if !(1..=MAX_RANGE_ZONES).contains(&zone_count) {
                    return Err(NexradError::InvalidMessage {
                        offset: cursor,
                        reason: format!(
                            "clutter filter map range zone count {zone_count} outside 1..={MAX_RANGE_ZONES}"
                        ),
                    });
                }
                cursor += 2;
                let zones_len = usize::from(zone_count) * 4;
                crate::require_len(body, cursor, zones_len, "clutter filter map range zones")?;
                zones.extend(
                    body[cursor..cursor + zones_len]
                        .chunks_exact(4)
                        .map(|pair| RangeZone {
                            op_code: OperatorSelectCode::from_code(crate::be_u16(pair, 0)),
                            end_range_km: crate::be_u16(pair, 2),
                        }),
                );
                cursor += zones_len;
                azimuth_start.push(zones.len());
            }
            segments.push(ClutterMapSegment {
                zones,
                azimuth_start,
            });
        }
        Ok(Self {
            generation_date: crate::be_u16(body, 0),
            generation_minutes: crate::be_u16(body, 2),
            segments,
            trailing_bytes: body.len() - cursor,
        })
    }

    /// Map generation time (UTC) from the date and minutes fields.
    pub fn generation_time(&self) -> DateTime<Utc> {
        crate::nexrad_date_ms_to_datetime(
            u32::from(self.generation_date),
            u32::from(self.generation_minutes) * 60_000,
        )
    }
}

/// Walker hook: the typed body for message 15.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    ClutterFilterMap::decode(&body).map(MessageBody::ClutterFilterMap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operator_select_codes_round_trip() {
        for code in 0..4 {
            assert_eq!(OperatorSelectCode::from_code(code).code(), code);
        }
        assert_eq!(
            OperatorSelectCode::from_code(1),
            OperatorSelectCode::BypassMapInControl
        );
        assert_eq!(
            OperatorSelectCode::from_code(3),
            OperatorSelectCode::Unknown(3)
        );
    }
}

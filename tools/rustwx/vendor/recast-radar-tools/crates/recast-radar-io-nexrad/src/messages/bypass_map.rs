//! Clutter Filter Bypass Map (message 13, ICD 2620002AA Table IX).
//!
//! For each elevation segment, one bit per 1 km range bin (512 bins) per
//! azimuth radial: 1 = bypass the clutter filters, 0 = perform clutter
//! filtering. Each radial is 32 halfwords, and the most significant bit of a
//! radial's first halfword is bin 0 (Table IX note 4). The RDA stopped
//! sending message 13 with Build 19. Archive II metadata records in the
//! corpus carry it up to Build 18.2.
//!
//! Real files use one of two layouts. Halfword 1 tells them apart:
//!
//! - [`BypassMapLayout::Current`], 2620002AA Table IX: halfword 1 is the
//!   generation date (days, 1 January 1970 = 1), halfword 2 the generation
//!   time (minutes after midnight UTC), halfword 3 the number of elevation
//!   segments (1 to 5). Each segment follows: its segment number, then 360
//!   radials, with radial `r` covering azimuths `r` to `r + 1` degrees. The
//!   corpus has it in KPAH and KDMX 2008 through KDVN 2020 (Build 18.2),
//!   always 5 elevation segments in 49 message segments.
//! - [`BypassMapLayout::Legacy`], 2620002B (Build 1.0, 2001) Table IX:
//!   halfword 1 is the number of elevation segments. Each segment follows:
//!   its segment number, then 256 radials, with radial `r` centred on
//!   `r * 360 / 256` degrees. There is no generation time. The corpus has
//!   it in KLIX 2005-08-29 (2 segments).
//!
//! A halfword 1 of 1 to 5 selects the legacy layout, as in MetPy. In the
//! current layout that halfword is the generation date, and no real map
//! dates from the first five days of 1970 (13 985 is the earliest in the
//! corpus).
//!
//! `tests/messages_clutter.rs` checks real files against MetPy 1.7.1
//! `Level2File.clutter_filter_bypass_map` for the generation time, the
//! segment count, the radial count and radial 0 of each segment. Only
//! radial 0 can be compared: MetPy repeats it for every radial of a segment,
//! and it lists each halfword's bits least significant first, so the test
//! compares whole halfwords. The test checks other radials against halfwords
//! read from the file bytes. It checks the note 4 bit order by how often
//! filtered bins continue across halfword boundaries.

use std::borrow::Cow;

use chrono::{DateTime, Utc};

use super::MessageBody;
use crate::{NexradError, Result};

/// Range bins per radial; each bin is 1 km, bin 0 covering 0 to 1 km.
pub const RANGE_BINS: usize = 512;

/// Halfwords per radial (16 range bins each).
pub const HALFWORDS_PER_RADIAL: usize = 32;

/// Largest number of elevation segments.
pub const MAX_ELEVATION_SEGMENTS: u16 = 5;

/// Range-bin bits of one radial, 32 halfwords; bin 0 is the most
/// significant bit of halfword 0.
pub type BypassRadial = [u16; HALFWORDS_PER_RADIAL];

/// Which revision of Table IX a bypass map follows.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum BypassMapLayout {
    /// 2620002AA Table IX: generation date and time, 360 radials of 1 degree.
    #[default]
    Current,
    /// 2620002B Table IX: no generation time, 256 radials of 1.40625 degrees.
    Legacy,
}

impl BypassMapLayout {
    /// Radials per elevation segment: 360 or 256.
    pub fn radial_count(self) -> usize {
        match self {
            Self::Current => 360,
            Self::Legacy => 256,
        }
    }

    /// Azimuthal width of one radial in degrees.
    pub fn radial_width_deg(self) -> f64 {
        360.0 / self.radial_count() as f64
    }

    /// Azimuth (degrees clockwise from true north, in `[0, 360)`) where
    /// `radial` starts. Current radials start on whole degrees; legacy radial
    /// 0 starts half a radial before north (`360 - 180/256`).
    pub fn radial_start_azimuth_deg(self, radial: usize) -> f64 {
        let start = match self {
            Self::Current => radial as f64,
            Self::Legacy => (radial as f64 - 0.5) * self.radial_width_deg(),
        };
        start.rem_euclid(360.0)
    }

    /// Radial holding `azimuth_deg` (any finite angle, wrapped into
    /// `[0, 360)`).
    pub fn radial_at_azimuth(self, azimuth_deg: f64) -> usize {
        let width = self.radial_width_deg();
        let offset = match self {
            Self::Current => 0.0,
            Self::Legacy => width / 2.0,
        };
        let radial = ((azimuth_deg + offset).rem_euclid(360.0) / width) as usize;
        radial.min(self.radial_count() - 1)
    }
}

/// One elevation segment of a bypass map.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BypassMapSegment {
    /// Segment number as sent (1 to 5; 1 is closest to the ground).
    pub segment_number: u16,
    /// Range-bin bits per radial, radial 0 first.
    pub radials: Vec<BypassRadial>,
}

impl BypassMapSegment {
    /// True when range bin `bin` (0 to 511, 1 km each) of `radial` bypasses
    /// the clutter filters, false when clutter filtering is performed; `None`
    /// outside the map.
    pub fn bypass(&self, radial: usize, bin: usize) -> Option<bool> {
        if bin >= RANGE_BINS {
            return None;
        }
        let halfword = self.radials.get(radial)?[bin / 16];
        Some(halfword & (0x8000 >> (bin % 16)) != 0)
    }

    /// Number of range bins in the segment that bypass the clutter filters.
    pub fn bypass_bin_count(&self) -> usize {
        self.radials
            .iter()
            .flatten()
            .map(|halfword| halfword.count_ones() as usize)
            .sum()
    }
}

/// Decoded clutter filter bypass map (Table IX).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClutterFilterBypassMap {
    /// Table IX revision the message follows.
    pub layout: BypassMapLayout,
    /// Halfword 1 (current layout): generation date in days,
    /// 1 January 1970 = 1.
    pub generation_date: Option<u16>,
    /// Halfword 2 (current layout): generation time in minutes after
    /// midnight UTC (0 to 1440).
    pub generation_minutes: Option<u16>,
    /// Elevation segments in the order sent.
    pub segments: Vec<BypassMapSegment>,
    /// Body bytes after the last segment; zero for a conforming message.
    pub trailing_bytes: usize,
}

impl ClutterFilterBypassMap {
    /// Decode a message body (the bytes after the 16-byte message header,
    /// joined across segments).
    ///
    /// Rejects an elevation segment count outside 1 to 5 and a body too short
    /// for the declared segments. Segment numbers are kept as sent.
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, 2, "clutter filter bypass map header")?;
        let first = crate::be_u16(body, 0);
        let legacy = (1..=MAX_ELEVATION_SEGMENTS).contains(&first);
        let (layout, generation_date, generation_minutes, count_offset) = if legacy {
            (BypassMapLayout::Legacy, None, None, 0)
        } else {
            crate::require_len(body, 0, 6, "clutter filter bypass map header")?;
            (
                BypassMapLayout::Current,
                Some(first),
                Some(crate::be_u16(body, 2)),
                4,
            )
        };
        let segment_count = crate::be_u16(body, count_offset);
        if !(1..=MAX_ELEVATION_SEGMENTS).contains(&segment_count) {
            return Err(NexradError::InvalidMessage {
                offset: count_offset,
                reason: format!(
                    "clutter filter bypass map elevation segment count {segment_count} outside 1..={MAX_ELEVATION_SEGMENTS}"
                ),
            });
        }
        let radial_count = layout.radial_count();
        let segment_len = 2 + radial_count * HALFWORDS_PER_RADIAL * 2;
        let mut cursor = count_offset + 2;
        crate::require_len(
            body,
            cursor,
            usize::from(segment_count) * segment_len,
            "clutter filter bypass map segments",
        )?;
        let segments = (0..segment_count)
            .map(|_| {
                let segment_number = crate::be_u16(body, cursor);
                let bits = &body[cursor + 2..cursor + segment_len];
                cursor += segment_len;
                BypassMapSegment {
                    segment_number,
                    radials: bits
                        .chunks_exact(HALFWORDS_PER_RADIAL * 2)
                        .map(|radial| std::array::from_fn(|index| crate::be_u16(radial, index * 2)))
                        .collect(),
                }
            })
            .collect();
        Ok(Self {
            layout,
            generation_date,
            generation_minutes,
            segments,
            trailing_bytes: body.len() - cursor,
        })
    }

    /// Map generation time (UTC); `None` for the legacy layout, which has no
    /// generation time.
    pub fn generation_time(&self) -> Option<DateTime<Utc>> {
        Some(crate::nexrad_date_ms_to_datetime(
            u32::from(self.generation_date?),
            u32::from(self.generation_minutes?) * 60_000,
        ))
    }
}

/// Walker hook: the typed body for message 13.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    ClutterFilterBypassMap::decode(&body).map(MessageBody::BypassMap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radial_azimuths_follow_table_ix_notes() {
        let current = BypassMapLayout::Current;
        assert_eq!(current.radial_start_azimuth_deg(0), 0.0);
        assert_eq!(current.radial_start_azimuth_deg(359), 359.0);
        assert_eq!(current.radial_at_azimuth(0.999), 0);
        assert_eq!(current.radial_at_azimuth(1.0), 1);
        assert_eq!(current.radial_at_azimuth(-0.5), 359);

        // 2620002B note 1: 360 - 180/256 <= R0 < 180/256, 180/256 <= R1 < 540/256.
        let legacy = BypassMapLayout::Legacy;
        assert_eq!(legacy.radial_width_deg(), 1.40625);
        assert_eq!(legacy.radial_start_azimuth_deg(0), 360.0 - 180.0 / 256.0);
        assert_eq!(legacy.radial_start_azimuth_deg(1), 180.0 / 256.0);
        assert_eq!(legacy.radial_at_azimuth(359.5), 0);
        assert_eq!(legacy.radial_at_azimuth(180.0 / 256.0 - 1e-9), 0);
        assert_eq!(legacy.radial_at_azimuth(180.0 / 256.0), 1);
        assert_eq!(legacy.radial_at_azimuth(540.0 / 256.0), 2);
        assert_eq!(legacy.radial_at_azimuth(359.0), 255);
    }
}

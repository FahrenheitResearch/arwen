//! Volume Coverage Pattern (messages 5 and 7, ICD 2620002AA Table XI).
//!
//! The RDA sends message 5 to the RPG on wideband connection and at the start
//! of every volume scan, so it is in the metadata record of Archive II files
//! from 2005 on. The RPG sends the same layout as message 7 to command a
//! pattern; Archive II files do not record message 7.
//!
//! Layout: an 11-halfword header, then 23 halfwords (E1 to E23) per elevation
//! cut. Halfwords are numbered from 1 at the first body byte, so halfword `n`
//! is at body byte `2 * (n - 1)`, and E1 of cut `c` (from 1) is halfword
//! `12 + 23 * (c - 1)` (Table XI note 18).
//!
//! Angle codes (Table III-A) and azimuth rates (Table XI-D) are decoded from
//! all 16 bits, as MetPy, Py-ART and xradar do. The ICD marks bits 0 to 2 as
//! not applicable; every real message in the corpus has them clear.
//!
//! Verified against MetPy's `Level2File.vcp_info` on 22 real metadata records
//! (Build 10.0 to 24.1, including TDWR); see `tests/messages_vcp.rs`.

use std::borrow::Cow;

use super::MessageBody;
use crate::{NexradError, Result};

/// Header length of Table XI, halfwords 1 to 11.
pub const VCP_HEADER_HALFWORDS: usize = 11;

/// Halfwords per elevation cut (E1 to E23) in Build 24.0 ("Number_of_E_Values",
/// Table XI note 18).
pub const VCP_CUT_HALFWORDS: usize = 23;

/// Degrees per unit of a Table III-A angle code (360 / 65536).
const ANGLE_DEG_PER_CODE: f32 = 360.0 / 65536.0;

/// Degrees per second per unit of a Table XI-D azimuth rate code (bit 14
/// weighs 22.5 deg/s, so one unit is 22.5 / 16384 = 90 / 65536).
const AZIMUTH_RATE_DEG_PER_S_PER_CODE: f32 = 90.0 / 65536.0;

/// dB per unit of a scaled SNR threshold (Table XI E6 to E11).
const SNR_THRESHOLD_DB_PER_CODE: f32 = 0.125;

/// Decoded Volume Coverage Pattern (Table XI). Halfwords 7, 8 and 11 are
/// reserved for RPG use and not kept.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeCoveragePattern {
    /// Halfword 1: message size in halfwords, counted from this halfword (the
    /// 16-byte message header is not included). ICD range 34 to 747; equal to
    /// 11 + 23 per cut in every real message.
    pub message_size: u16,
    /// Halfword 2.
    pub pattern_type: PatternType,
    /// Halfword 3: VCP number (Appendix C), for example 212.
    pub pattern_number: u16,
    /// Halfword 4: number of elevation cuts in one complete volume scan (ICD
    /// range 1 to 32). Equals `cuts.len()`.
    pub number_of_cuts: u16,
    /// Halfword 5, upper byte: VCP version (ICD range 1 to 99; 0 in files
    /// from Builds 10 to 16).
    pub version: u8,
    /// Halfword 5, lower byte: clutter map group (1; groups are not
    /// implemented).
    pub clutter_map_group: u8,
    /// Halfword 6, upper byte.
    pub doppler_velocity_resolution: DopplerVelocityResolution,
    /// Halfword 6, lower byte.
    pub pulse_width: PulseWidth,
    /// Halfword 9: VCP sequencing values (RPG use).
    pub sequencing: VcpSequencing,
    /// Halfword 10: VCP supplemental data (RPG use).
    pub supplemental: VcpSupplemental,
    /// The elevation cuts in scan order.
    pub cuts: Vec<VcpCut>,
}

impl VolumeCoveragePattern {
    /// Decode a message body (the bytes after the 16-byte message header).
    ///
    /// The same layout is embedded in RDA adaptation data (message 18), so
    /// this also decodes those blocks.
    ///
    /// Errors: a body shorter than the header or than `message_size`
    /// halfwords; a `message_size` of 0 (the zero-filled message 5 of the
    /// 2005 KLIX metadata record, which MetPy also skips) or 0 cuts; and a
    /// `message_size` that does not equal 11 plus a whole number of at least
    /// 23 halfwords per cut. When that number is above 23 (a later ICD adding
    /// E values), the extra halfwords of each cut are skipped.
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, VCP_HEADER_HALFWORDS * 2, "VCP header")?;
        let halfword = |number: usize| crate::be_u16(body, (number - 1) * 2);
        let message_size = halfword(1);
        let number_of_cuts = halfword(4);
        if message_size == 0 {
            return Err(invalid(
                "VCP message size is 0 (the message holds no pattern)",
            ));
        }
        if number_of_cuts == 0 {
            return Err(invalid("VCP has no elevation cuts"));
        }
        let size = usize::from(message_size);
        let cuts = usize::from(number_of_cuts);
        let cut_halfwords = size
            .checked_sub(VCP_HEADER_HALFWORDS)
            .filter(|cut_area| cut_area % cuts == 0)
            .map(|cut_area| cut_area / cuts)
            .filter(|per_cut| *per_cut >= VCP_CUT_HALFWORDS)
            .ok_or_else(|| {
                invalid(format!(
                    "VCP message size {size} halfwords does not hold {cuts} cuts of at least \
                     {VCP_CUT_HALFWORDS} halfwords after the {VCP_HEADER_HALFWORDS}-halfword header"
                ))
            })?;
        crate::require_len(body, 0, size * 2, "VCP elevation cuts")?;

        let [version, clutter_map_group] = halfword(5).to_be_bytes();
        let [resolution, pulse_width] = halfword(6).to_be_bytes();
        let cuts = (0..cuts)
            .map(|index| {
                let start = (VCP_HEADER_HALFWORDS + index * cut_halfwords) * 2;
                VcpCut::decode(&body[start..start + VCP_CUT_HALFWORDS * 2])
            })
            .collect();
        Ok(Self {
            message_size,
            pattern_type: PatternType::from_code(halfword(2)),
            pattern_number: halfword(3),
            number_of_cuts,
            version,
            clutter_map_group,
            doppler_velocity_resolution: DopplerVelocityResolution::from_code(resolution),
            pulse_width: PulseWidth::from_code(pulse_width),
            sequencing: VcpSequencing { code: halfword(9) },
            supplemental: VcpSupplemental { code: halfword(10) },
            cuts,
        })
    }
}

fn invalid(reason: impl Into<String>) -> NexradError {
    NexradError::InvalidMessage {
        offset: 0,
        reason: reason.into(),
    }
}

/// One elevation cut of a VCP (Table XI E1 to E23). E23 is reserved and not
/// kept.
#[derive(Clone, Debug, PartialEq)]
pub struct VcpCut {
    /// E1: elevation angle in degrees. Codes above 90 degrees are negative
    /// angles (Table III-A note), so the range is -270 to 90.
    pub elevation_angle_deg: f32,
    /// E2, upper byte.
    pub channel_configuration: ChannelConfiguration,
    /// E2, lower byte.
    pub waveform: WaveformType,
    /// E3, upper byte.
    pub super_resolution: SuperResolutionControl,
    /// E3, lower byte: surveillance PRF number (0 to 8; 0 when the cut has no
    /// surveillance part). Resolve it to Hz with
    /// [`RdaPrfData::surveillance_prf_hz`](super::prf::RdaPrfData::surveillance_prf_hz).
    pub surveillance_prf_number: u8,
    /// E4: surveillance pulse count per radial (0 to 999; 0 when not
    /// applicable).
    pub surveillance_pulse_count: u16,
    /// E5: azimuth rate in degrees per second (Table XI-D, two's complement;
    /// ICD range -44.989 to +44.989).
    pub azimuth_rate_deg_per_s: f32,
    /// E6 to E11: SNR thresholds.
    pub snr_threshold_db: SnrThresholds,
    /// E12 to E14, E16 to E18 and E20 to E22: the three Doppler azimuth
    /// sectors.
    pub doppler_sectors: [DopplerSector; 3],
    /// E15: supplemental data (RPG use).
    pub supplemental: CutSupplemental,
    /// E19: correction added to the elevation angle for this cut, in degrees.
    /// Codes above 90 degrees are negative, as for E1 (the 2023 KDGX volume
    /// has corrections of -0.088 and -0.132 degrees).
    pub ebc_angle_deg: f32,
}

impl VcpCut {
    /// Decode the 46 bytes of E1 to E23.
    fn decode(bytes: &[u8]) -> Self {
        let e = |number: usize| crate::be_u16(bytes, (number - 1) * 2);
        let signed = |number: usize| crate::be_i16(bytes, (number - 1) * 2);
        let [channel, waveform] = e(2).to_be_bytes();
        let [super_resolution, surveillance_prf_number] = e(3).to_be_bytes();
        let sector = |edge: usize| DopplerSector {
            edge_angle_deg: angle_deg(e(edge)),
            prf_number: e(edge + 1),
            pulse_count: e(edge + 2),
        };
        Self {
            elevation_angle_deg: elevation_deg(e(1)),
            channel_configuration: ChannelConfiguration::from_code(channel),
            waveform: WaveformType::from_code(u16::from(waveform)),
            super_resolution: SuperResolutionControl {
                code: super_resolution,
            },
            surveillance_prf_number,
            surveillance_pulse_count: e(4),
            azimuth_rate_deg_per_s: f32::from(signed(5)) * AZIMUTH_RATE_DEG_PER_S_PER_CODE,
            snr_threshold_db: SnrThresholds {
                reflectivity: snr_db(signed(6)),
                velocity: snr_db(signed(7)),
                spectrum_width: snr_db(signed(8)),
                differential_reflectivity: snr_db(signed(9)),
                differential_phase: snr_db(signed(10)),
                correlation_coefficient: snr_db(signed(11)),
            },
            doppler_sectors: [sector(12), sector(16), sector(20)],
            supplemental: CutSupplemental { code: e(15) },
            ebc_angle_deg: elevation_deg(e(19)),
        }
    }
}

/// A Table III-A angle code in degrees, 0 to 359.995.
fn angle_deg(code: u16) -> f32 {
    f32::from(code) * ANGLE_DEG_PER_CODE
}

/// A Table III-A elevation code in degrees: angles above 90 degrees are
/// negative (the angle minus 360).
fn elevation_deg(code: u16) -> f32 {
    let angle = angle_deg(code);
    if angle > 90.0 { angle - 360.0 } else { angle }
}

fn snr_db(code: i16) -> f32 {
    f32::from(code) * SNR_THRESHOLD_DB_PER_CODE
}

/// SNR thresholds of one cut, in dB (Table XI E6 to E11; ICD range -12 to
/// +20, precision 0.125).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnrThresholds {
    /// E6: reflectivity.
    pub reflectivity: f32,
    /// E7: velocity.
    pub velocity: f32,
    /// E8: spectrum width.
    pub spectrum_width: f32,
    /// E9: differential reflectivity.
    pub differential_reflectivity: f32,
    /// E10: differential phase.
    pub differential_phase: f32,
    /// E11: correlation coefficient.
    pub correlation_coefficient: f32,
}

/// One Doppler azimuth sector of a cut. All fields are 0 when the cut has no
/// Doppler part (Table XI note 5).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DopplerSector {
    /// Clockwise edge (start) azimuth of the sector in degrees, 0 to 359.995.
    pub edge_angle_deg: f32,
    /// Doppler PRF number (0 to 8). Resolve it to Hz with
    /// [`RdaPrfData::doppler_prf_hz`](super::prf::RdaPrfData::doppler_prf_hz).
    pub prf_number: u16,
    /// Doppler pulse count per radial (0 to 999).
    pub pulse_count: u16,
}

/// Pattern type (halfword 2).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PatternType {
    /// 2: constant elevation cut, used by every operational VCP.
    ConstantElevationCut,
    /// Any other code.
    Unknown(u16),
}

impl PatternType {
    /// Map a Table XI code.
    pub fn from_code(code: u16) -> Self {
        match code {
            2 => Self::ConstantElevationCut,
            other => Self::Unknown(other),
        }
    }

    /// The Table XI code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u16 {
        match self {
            Self::ConstantElevationCut => 2,
            Self::Unknown(code) => code,
        }
    }
}

/// Doppler velocity resolution (halfword 6, upper byte).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DopplerVelocityResolution {
    /// 2 (bit 1): 0.5 m/s.
    HalfMetrePerSecond,
    /// 4 (bit 2): 1.0 m/s.
    OneMetrePerSecond,
    /// Any other code.
    Unknown(u8),
}

impl DopplerVelocityResolution {
    /// Map a Table XI code.
    pub fn from_code(code: u8) -> Self {
        match code {
            2 => Self::HalfMetrePerSecond,
            4 => Self::OneMetrePerSecond,
            other => Self::Unknown(other),
        }
    }

    /// The Table XI code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u8 {
        match self {
            Self::HalfMetrePerSecond => 2,
            Self::OneMetrePerSecond => 4,
            Self::Unknown(code) => code,
        }
    }

    /// The resolution in m/s, or `None` for an unknown code.
    pub fn metres_per_second(self) -> Option<f32> {
        match self {
            Self::HalfMetrePerSecond => Some(0.5),
            Self::OneMetrePerSecond => Some(1.0),
            Self::Unknown(_) => None,
        }
    }
}

/// Pulse width (halfword 6, lower byte).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PulseWidth {
    /// 2 (bit 1).
    Short,
    /// 4 (bit 2); the long-pulse clear-air VCPs (31, 34).
    Long,
    /// Any other code.
    Unknown(u8),
}

impl PulseWidth {
    /// Map a Table XI code.
    pub fn from_code(code: u8) -> Self {
        match code {
            2 => Self::Short,
            4 => Self::Long,
            other => Self::Unknown(other),
        }
    }

    /// The Table XI code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u8 {
        match self {
            Self::Short => 2,
            Self::Long => 4,
            Self::Unknown(code) => code,
        }
    }
}

/// Channel configuration of a cut (E2, upper byte).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ChannelConfiguration {
    /// 0.
    ConstantPhase,
    /// 1.
    RandomPhase,
    /// 2: SZ-2 phase coding.
    Sz2Phase,
    /// Any other code.
    Unknown(u8),
}

impl ChannelConfiguration {
    /// Map a Table XI code.
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => Self::ConstantPhase,
            1 => Self::RandomPhase,
            2 => Self::Sz2Phase,
            other => Self::Unknown(other),
        }
    }

    /// The Table XI code, the inverse of [`Self::from_code`].
    pub fn code(self) -> u8 {
        match self {
            Self::ConstantPhase => 0,
            Self::RandomPhase => 1,
            Self::Sz2Phase => 2,
            Self::Unknown(code) => code,
        }
    }
}

/// Waveform type (Table XI E2 lower byte, and Table XVIII P1 of message 32).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum WaveformType {
    /// 1: contiguous surveillance (CS).
    ContiguousSurveillance,
    /// 2: contiguous Doppler with ambiguity resolution (CD/W).
    ContiguousDopplerWithAmbiguityResolution,
    /// 3: contiguous Doppler without ambiguity resolution (CD/WO).
    ContiguousDopplerWithoutAmbiguityResolution,
    /// 4: batch (B).
    Batch,
    /// 5: staggered pulse pair (SPP).
    StaggeredPulsePair,
    /// Any other code.
    Unknown(u16),
}

impl WaveformType {
    /// Map a Table XI or Table XVIII code.
    pub fn from_code(code: u16) -> Self {
        match code {
            1 => Self::ContiguousSurveillance,
            2 => Self::ContiguousDopplerWithAmbiguityResolution,
            3 => Self::ContiguousDopplerWithoutAmbiguityResolution,
            4 => Self::Batch,
            5 => Self::StaggeredPulsePair,
            other => Self::Unknown(other),
        }
    }

    /// The ICD code.
    pub fn code(self) -> u16 {
        match self {
            Self::ContiguousSurveillance => 1,
            Self::ContiguousDopplerWithAmbiguityResolution => 2,
            Self::ContiguousDopplerWithoutAmbiguityResolution => 3,
            Self::Batch => 4,
            Self::StaggeredPulsePair => 5,
            Self::Unknown(code) => code,
        }
    }
}

/// Super resolution control bits of a cut (E3, upper byte). The bits are
/// independent (Table XI note 13).
///
/// MetPy 1.7.1 names bit 0 "0.5 azimuth and 0.25km range res.", bit 1
/// "Doppler to 300km" and bit 2 "Dual Polarization Control"; the names here
/// follow the Build 24.0 table. Under it, the split-cut surveillance cuts of
/// real dual-polarization volumes carry 11 (0.5 degree, 1/4 km, dual
/// polarization to 300 km) and their Doppler partners 7 (0.5 degree, 1/4 km,
/// Doppler to 300 km).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SuperResolutionControl {
    /// The raw byte.
    pub code: u8,
}

impl SuperResolutionControl {
    /// Bit 0: 0.5 degree azimuth spacing.
    pub fn half_degree_azimuth(self) -> bool {
        self.code & 0x01 != 0
    }

    /// Bit 1: 1/4 km reflectivity gates.
    pub fn quarter_km_reflectivity(self) -> bool {
        self.code & 0x02 != 0
    }

    /// Bit 2: Doppler moments to 300 km.
    pub fn doppler_to_300_km(self) -> bool {
        self.code & 0x04 != 0
    }

    /// Bit 3: dual polarization moments to 300 km.
    pub fn dual_polarization_to_300_km(self) -> bool {
        self.code & 0x08 != 0
    }
}

/// VCP sequencing values (halfword 9, Table XI note 15).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VcpSequencing {
    /// The raw halfword.
    pub code: u16,
}

impl VcpSequencing {
    /// Bits 0 to 4: number of elevation cuts in the truncated VCP.
    pub fn number_of_elevations(self) -> u8 {
        (self.code & 0x1F) as u8
    }

    /// Bits 5 and 6: maximum SAILS cuts in the truncated VCP (up to 3).
    pub fn max_sails_cuts(self) -> u8 {
        ((self.code >> 5) & 0x03) as u8
    }

    /// Bit 13: the VCP is part of an active VCP sequence.
    pub fn sequence_active(self) -> bool {
        self.code & (1 << 13) != 0
    }

    /// Bit 14: the VCP is part of an active sequence and truncated.
    pub fn truncated(self) -> bool {
        self.code & (1 << 14) != 0
    }
}

/// VCP supplemental data (halfword 10, Table XI note 16).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VcpSupplemental {
    /// The raw halfword.
    pub code: u16,
}

impl VcpSupplemental {
    /// Bit 0: the VCP contains SAILS cuts.
    pub fn sails(self) -> bool {
        self.code & 0x0001 != 0
    }

    /// Bits 1 to 3: number of SAILS cuts (up to 3).
    pub fn sails_cuts(self) -> u8 {
        ((self.code >> 1) & 0x07) as u8
    }

    /// Bit 4: the VCP contains MRLE (mid-volume rescan of low-level
    /// elevations) cuts.
    pub fn mrle(self) -> bool {
        self.code & 0x0010 != 0
    }

    /// Bits 5 to 7: number of MRLE cuts (up to 4).
    pub fn mrle_cuts(self) -> u8 {
        ((self.code >> 5) & 0x07) as u8
    }

    /// Bit 10: MPDA cuts added. Listed in the Table XI body; note 16 calls
    /// bits 8 to 10 spare.
    pub fn mpda_cuts_added(self) -> bool {
        self.code & (1 << 10) != 0
    }

    /// Bit 11: multi-PRF dealiasing algorithm (MPDA) VCP.
    pub fn mpda(self) -> bool {
        self.code & (1 << 11) != 0
    }

    /// Bit 12: the VCP contains at least one base tilt.
    pub fn base_tilt(self) -> bool {
        self.code & (1 << 12) != 0
    }

    /// Bits 13 to 15: number of base tilts.
    pub fn base_tilt_cuts(self) -> u8 {
        (self.code >> 13) as u8
    }
}

/// Supplemental data of one cut (E15, Table XI note 17).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CutSupplemental {
    /// The raw halfword.
    pub code: u16,
}

impl CutSupplemental {
    /// Bit 0: SAILS cut.
    pub fn sails_cut(self) -> bool {
        self.code & 0x0001 != 0
    }

    /// Bits 1 to 3: SAILS sequence number.
    pub fn sails_sequence_number(self) -> u8 {
        ((self.code >> 1) & 0x07) as u8
    }

    /// Bit 4: MRLE cut.
    pub fn mrle_cut(self) -> bool {
        self.code & 0x0010 != 0
    }

    /// Bits 5 to 7: MRLE sequence number (the RPG elevation index of the
    /// cut).
    pub fn mrle_sequence_number(self) -> u8 {
        ((self.code >> 5) & 0x07) as u8
    }

    /// Bit 9: MPDA cut.
    pub fn mpda_cut(self) -> bool {
        self.code & (1 << 9) != 0
    }

    /// Bit 10: base tilt cut.
    pub fn base_tilt_cut(self) -> bool {
        self.code & (1 << 10) != 0
    }
}

/// Body length of a message in a fixed 2432-byte frame, given the body
/// length its header declares and the body bytes the frame holds
/// (`frame_body`, from the first body byte to the end of the frame or of the
/// input).
///
/// Every message but 5 and 7 has the declared length (as far as the frame
/// holds it). A Message 5 or 7 runs to its own length when that is longer:
/// Table XI halfword 1 counts the halfwords of the VCP without the message
/// header, and some converted files write that same number as the
/// message header's size, leaving out the header's 8 halfwords, so the last
/// 16 bytes of their cut table lie past the declared size but inside the
/// frame. MetPy and Py-ART read the cut table the VCP declares; so do the
/// walker and the volume decoder. NEXRAD files declare the same length both
/// ways.
pub(crate) fn fixed_frame_body_len(message_type: u8, declared: usize, frame_body: &[u8]) -> usize {
    let declared = declared.min(frame_body.len());
    if !matches!(message_type, 5 | 7) {
        return declared;
    }
    match frame_body.first_chunk::<2>() {
        Some(size) => {
            let own = usize::from(u16::from_be_bytes(*size)) * 2;
            if own > declared && own <= frame_body.len() {
                own
            } else {
                declared
            }
        }
        None => declared,
    }
}

/// Walker hook: the typed body for messages 5 and 7.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    VolumeCoveragePattern::decode(&body).map(MessageBody::Vcp)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_codes_decode_exactly_and_negative_above_90_degrees() {
        assert_eq!(f64::from(elevation_deg(88)), 0.483_398_437_5);
        assert_eq!(elevation_deg(0), 0.0);
        // 359.8681640625 degrees (KDGX 2023 EBC angle code 65512).
        assert_eq!(f64::from(elevation_deg(65512)), -0.131_835_937_5);
        assert_eq!(f64::from(angle_deg(60984)), 334.995_117_187_5);
    }

    #[test]
    fn supplemental_bits_follow_table_xi_notes() {
        // KDGX 2023: SAILS x2, base tilt VCP with 2 base tilts.
        let vcp = VcpSupplemental { code: 0x5005 };
        assert!(vcp.sails() && vcp.base_tilt() && !vcp.mrle() && !vcp.mpda());
        assert_eq!((vcp.sails_cuts(), vcp.base_tilt_cuts()), (2, 2));
        // KILX 2026 cut 16: MRLE cut with sequence number 3.
        let cut = CutSupplemental { code: 0x70 };
        assert!(cut.mrle_cut() && !cut.sails_cut());
        assert_eq!(cut.mrle_sequence_number(), 3);
        // KDVN 2020 sequencing: 7 elevations, up to 2 SAILS cuts, inactive.
        let sequencing = VcpSequencing { code: 0x47 };
        assert_eq!(sequencing.number_of_elevations(), 7);
        assert_eq!(sequencing.max_sails_cuts(), 2);
        assert!(!sequencing.sequence_active() && !sequencing.truncated());
    }
}

//! RDA PRF Data (message 32, ICD 2620002AA Table XVIII).
//!
//! The PRF values the RDA uses for each waveform type, in millihertz. PRF
//! numbers in the Volume Coverage Pattern (message 5) index these tables:
//! the surveillance PRF number of a cut selects from the contiguous
//! surveillance table and its Doppler PRF numbers from the Doppler table of
//! its waveform (Table XVIII note 1 and section 3.2.3.13).
//!
//! Layout: halfword 1 is the number of waveforms, halfword 2 is spare, then
//! one variable-length section per waveform: P1 waveform type, P2 PRF count
//! `N`, then `N` PRFs as 32-bit integers (P3-P4, P5-P6, ...). Real messages
//! (Builds 23.1 and 24.1) hold 3 sections of 8 PRFs each, for waveforms 1, 2
//! and 5.
//!
//! MetPy, Py-ART and xradar do not decode this message. `tests/messages_vcp.rs`
//! checks exact values read from the bytes of real files, and checks that the
//! PRFs selected through message 5 match the unambiguous range of every sweep
//! in the same volumes.

use std::borrow::Cow;

use super::MessageBody;
use super::vcp::{VcpCut, WaveformType};
use crate::{NexradError, Result};

/// Decoded RDA PRF data (Table XVIII).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RdaPrfData {
    /// Halfword 1: number of waveforms with PRF values (ICD range 1 to 5).
    /// Equals `waveforms.len()`.
    pub number_of_waveforms: u16,
    /// One section per waveform, in message order.
    pub waveforms: Vec<WaveformPrfs>,
}

/// PRF values of one waveform type (Table XVIII P1 to P'X').
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct WaveformPrfs {
    /// P1: waveform type (Table XI codes; the ICD lists 1, 2 and 5).
    pub waveform: WaveformType,
    /// PRF 1 to PRF N in millihertz (ICD range 0 to 1,500,000, precision
    /// 0.001 Hz). Index 0 holds PRF number 1. The P2 count is `prfs_mhz.len()`.
    pub prfs_mhz: Vec<u32>,
}

impl WaveformPrfs {
    /// The PRF with 1-based `prf_number`, in Hz; `None` for 0 or a number
    /// past the count.
    pub fn prf_hz(&self, prf_number: u16) -> Option<f64> {
        let index = usize::from(prf_number).checked_sub(1)?;
        self.prfs_mhz
            .get(index)
            .map(|millihertz| f64::from(*millihertz) / 1000.0)
    }
}

impl RdaPrfData {
    /// Decode a message body (the bytes after the 16-byte message header).
    ///
    /// Bytes after the last section are ignored. Errors: 0 waveforms (no
    /// PRF data), or a body too short for the sections it declares.
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, 4, "PRF data header")?;
        let number_of_waveforms = crate::be_u16(body, 0);
        if number_of_waveforms == 0 {
            return Err(NexradError::InvalidMessage {
                offset: 0,
                reason: "PRF data has no waveforms".to_owned(),
            });
        }
        let mut offset = 4;
        let mut waveforms = Vec::new();
        for _ in 0..number_of_waveforms {
            crate::require_len(body, offset, 4, "PRF data waveform section")?;
            let waveform = WaveformType::from_code(crate::be_u16(body, offset));
            let count = usize::from(crate::be_u16(body, offset + 2));
            offset += 4;
            crate::require_len(body, offset, count * 4, "PRF data values")?;
            let prfs_mhz = (0..count)
                .map(|index| crate::be_u32(body, offset + index * 4))
                .collect();
            offset += count * 4;
            waveforms.push(WaveformPrfs { waveform, prfs_mhz });
        }
        Ok(Self {
            number_of_waveforms,
            waveforms,
        })
    }

    /// The first section for `waveform`.
    pub fn waveform(&self, waveform: WaveformType) -> Option<&WaveformPrfs> {
        self.waveforms
            .iter()
            .find(|section| section.waveform == waveform)
    }

    /// The surveillance PRF of a VCP cut in Hz: its surveillance PRF number
    /// in the contiguous surveillance (waveform 1) table, which also serves
    /// the surveillance part of batch cuts (Table XVIII note 1). `None` when
    /// the number is 0 (no surveillance part) or not in the table.
    pub fn surveillance_prf_hz(&self, cut: &VcpCut) -> Option<f64> {
        self.waveform(WaveformType::ContiguousSurveillance)?
            .prf_hz(u16::from(cut.surveillance_prf_number))
    }

    /// The Doppler PRF of Doppler sector `sector` (0 to 2) of a VCP cut in
    /// Hz. The table is the cut waveform's own, except that contiguous
    /// Doppler without ambiguity resolution (3) and batch (4) cuts use the
    /// waveform 2 table (Table XVIII note 1). `None` for a contiguous
    /// surveillance cut, a sector past 2, a PRF number of 0, or a number not
    /// in the table.
    pub fn doppler_prf_hz(&self, cut: &VcpCut, sector: usize) -> Option<f64> {
        let table = match cut.waveform {
            WaveformType::ContiguousSurveillance => return None,
            WaveformType::ContiguousDopplerWithoutAmbiguityResolution | WaveformType::Batch => {
                WaveformType::ContiguousDopplerWithAmbiguityResolution
            }
            other => other,
        };
        let prf_number = cut.doppler_sectors.get(sector)?.prf_number;
        self.waveform(table)?.prf_hz(prf_number)
    }
}

/// Walker hook: the typed body for message 32.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    RdaPrfData::decode(&body).map(MessageBody::Prf)
}

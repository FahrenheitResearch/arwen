//! Request for Data (message 9, ICD 2620002AA Table XIII).
//!
//! Sent from the RPG to the RDA, so Archive II files do not record it; no file
//! in the test corpus holds one and this decoder follows the ICD without a
//! verified real sample.

use std::borrow::Cow;

use super::MessageBody;
use crate::Result;

/// Decoded request for data (Table XIII).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestForData {
    /// Halfword 1: which message the RPG asks the RDA to send.
    pub request: DataRequestType,
}

impl RequestForData {
    /// Decode a message body (the bytes after the 16-byte message header).
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, 2, "request for data")?;
        Ok(Self {
            request: DataRequestType::from_code(crate::be_u16(body, 0)),
        })
    }
}

/// Data request type codes (Table XIII, halfword 1): one bit from 0 to 5 plus
/// bit 7.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DataRequestType {
    /// 129: summary RDA status (message 2).
    RdaStatus,
    /// 130: RDA performance/maintenance data (message 3).
    PerformanceMaintenance,
    /// 132: clutter filter bypass map (message 13).
    ClutterFilterBypassMap,
    /// 136: clutter filter map (message 15).
    ClutterFilterMap,
    /// 144: RDA adaptation data (message 18).
    RdaAdaptationData,
    /// 160: volume coverage pattern data (message 5).
    VolumeCoveragePattern,
    /// Any other code.
    Unknown(u16),
}

impl DataRequestType {
    /// Map a Table XIII code.
    pub fn from_code(code: u16) -> Self {
        match code {
            129 => Self::RdaStatus,
            130 => Self::PerformanceMaintenance,
            132 => Self::ClutterFilterBypassMap,
            136 => Self::ClutterFilterMap,
            144 => Self::RdaAdaptationData,
            160 => Self::VolumeCoveragePattern,
            other => Self::Unknown(other),
        }
    }

    /// The Table XIII code.
    pub fn code(self) -> u16 {
        match self {
            Self::RdaStatus => 129,
            Self::PerformanceMaintenance => 130,
            Self::ClutterFilterBypassMap => 132,
            Self::ClutterFilterMap => 136,
            Self::RdaAdaptationData => 144,
            Self::VolumeCoveragePattern => 160,
            Self::Unknown(code) => code,
        }
    }

    /// Message type the RDA answers with, per Table I.
    pub fn response_message_type(self) -> Option<u8> {
        match self {
            Self::RdaStatus => Some(2),
            Self::PerformanceMaintenance => Some(3),
            Self::ClutterFilterBypassMap => Some(13),
            Self::ClutterFilterMap => Some(15),
            Self::RdaAdaptationData => Some(18),
            Self::VolumeCoveragePattern => Some(5),
            Self::Unknown(_) => None,
        }
    }
}

/// Walker hook: the typed body for message 9.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    RequestForData::decode(&body).map(MessageBody::RequestForData)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_codes_round_trip_and_set_bit_seven() {
        for code in [129, 130, 132, 136, 144, 160] {
            let request = DataRequestType::from_code(code);
            assert_ne!(request, DataRequestType::Unknown(code));
            assert_eq!(request.code(), code);
            assert_eq!(code & 0x80, 0x80);
        }
        assert_eq!(
            DataRequestType::from_code(128),
            DataRequestType::Unknown(128)
        );
    }
}

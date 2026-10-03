//! Loop Back Test (messages 11 and 12, ICD 2620002AA Table VIII).
//!
//! The RDA sends message 11 and the RPG message 12 on wideband connection;
//! the receiver echoes it unchanged. Archive II files do not record them; no
//! file in the test corpus holds one and this decoder follows the ICD without
//! a verified real sample.

use std::borrow::Cow;

use super::MessageBody;
use crate::{NexradError, Result};

/// Largest loopback message size Table VIII allows, in halfwords.
pub const MAX_LOOPBACK_HALFWORDS: u16 = 1200;

/// Decoded loop back test message (Table VIII).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LoopbackTest {
    /// Halfword 1: message size in halfwords, excluding the message header
    /// and including this size halfword (2 to 1200 per the ICD).
    pub message_size: u16,
    /// Halfwords 2 to `message_size`: the test bit pattern.
    pub bit_pattern: Vec<u8>,
}

impl LoopbackTest {
    /// Decode a message body (the bytes after the 16-byte message header).
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, 2, "loopback message size")?;
        let message_size = crate::be_u16(body, 0);
        if message_size > MAX_LOOPBACK_HALFWORDS {
            return Err(NexradError::InvalidMessage {
                offset: 0,
                reason: format!(
                    "loopback message size {message_size} exceeds {MAX_LOOPBACK_HALFWORDS} halfwords"
                ),
            });
        }
        let end = usize::from(message_size.max(1)) * 2;
        crate::require_len(body, 0, end, "loopback bit pattern")?;
        Ok(Self {
            message_size,
            bit_pattern: body[2..end].to_vec(),
        })
    }
}

/// Walker hook: the typed body for messages 11 and 12.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    LoopbackTest::decode(&body).map(MessageBody::Loopback)
}

//! Console Message (messages 4 and 10, ICD 2620002AA Table VI).
//!
//! Message 4 travels from the RDA to the RPG, message 10 from the RPG to the
//! RDA; both carry operator text. No Archive II file in the test corpus holds
//! either, so this decoder follows the ICD without a verified real sample.

use std::borrow::Cow;

use super::MessageBody;
use crate::{NexradError, Result};

/// Largest text length Table VI allows (halfwords 2 to 203).
pub const MAX_CONSOLE_TEXT_BYTES: usize = 404;

/// Decoded console message (Table VI).
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConsoleMessage {
    /// Halfword 1: number of bytes of text (2 to 404 per the ICD).
    pub message_size: u16,
    /// Halfwords 2 to 203: the text bytes, `message_size` of them, including
    /// embedded carriage returns and line feeds. Since Build 13 the RDA sends
    /// NUL-terminated strings; the terminator and anything after it are kept
    /// here and dropped by [`ConsoleMessage::text`].
    pub bytes: Vec<u8>,
}

impl ConsoleMessage {
    /// Decode a message body (the bytes after the 16-byte message header).
    pub fn decode(body: &[u8]) -> Result<Self> {
        crate::require_len(body, 0, 2, "console message size")?;
        let message_size = crate::be_u16(body, 0);
        let text_len = usize::from(message_size);
        if text_len > MAX_CONSOLE_TEXT_BYTES {
            return Err(NexradError::InvalidMessage {
                offset: 0,
                reason: format!(
                    "console message size {text_len} exceeds {MAX_CONSOLE_TEXT_BYTES} bytes"
                ),
            });
        }
        crate::require_len(body, 2, text_len, "console message text")?;
        Ok(Self {
            message_size,
            bytes: body[2..2 + text_len].to_vec(),
        })
    }

    /// The text up to the first NUL, with invalid UTF-8 replaced.
    pub fn text(&self) -> Cow<'_, str> {
        let end = self
            .bytes
            .iter()
            .position(|byte| *byte == 0)
            .unwrap_or(self.bytes.len());
        String::from_utf8_lossy(&self.bytes[..end])
    }
}

/// Walker hook: the typed body for messages 4 and 10.
pub(crate) fn message_body(body: Cow<'_, [u8]>) -> Result<MessageBody<'_>> {
    ConsoleMessage::decode(&body).map(MessageBody::Console)
}

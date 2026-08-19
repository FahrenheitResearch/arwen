//! HDF5 Shared Message (Phase 2 stub).
//!
//! A shared message indicates that the actual message content is stored
//! elsewhere — either in another object header or in the shared message
//! heap (SOHM).

use crate::error::{Error, Result};
use crate::io::Cursor;

/// Where the shared message is stored.
#[derive(Debug, Clone)]
pub enum SharedMessage {
    /// Shared in another object header at the given address.
    SharedInOhdr { address: u64 },
    /// Shared in the SOHM fractal heap, identified by a heap ID.
    SharedInSohm { heap_id: Vec<u8> },
}

/// Parse a shared message wrapper.
///
/// The `type_field` indicates how the message is shared:
/// - 0 or 2: shared in another object header (version-dependent)
/// - 1 or 3: shared in the SOHM
pub fn parse(
    cursor: &mut Cursor<'_>,
    offset_size: u8,
    _length_size: u8,
    msg_size: usize,
) -> Result<SharedMessage> {
    let start = cursor.position();
    let version = cursor.read_u8()?;

    let msg = match version {
        1 => {
            // Version 1: type byte + reserved(6) + address
            let _type_field = cursor.read_u8()?;
            let _reserved = cursor.read_bytes(6)?;
            let address = cursor.read_offset(offset_size)?;
            SharedMessage::SharedInOhdr { address }
        }
        2 | 3 => {
            let type_field = cursor.read_u8()?;
            match type_field {
                0 | 2 => {
                    // Shared in another object header
                    let address = cursor.read_offset(offset_size)?;
                    SharedMessage::SharedInOhdr { address }
                }
                1 | 3 => {
                    // Shared in SOHM heap — heap ID is 8 bytes
                    let heap_id = cursor.read_bytes(8)?.to_vec();
                    SharedMessage::SharedInSohm { heap_id }
                }
                t => {
                    return Err(Error::InvalidData(format!(
                        "unknown shared message type: {}",
                        t
                    )));
                }
            }
        }
        v => {
            return Err(Error::InvalidData(format!(
                "unsupported shared message version: {}",
                v
            )));
        }
    };

    let consumed = (cursor.position() - start) as usize;
    if consumed < msg_size {
        cursor.skip(msg_size - consumed)?;
    }

    Ok(msg)
}

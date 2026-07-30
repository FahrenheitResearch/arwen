//! Streaming download + decode for GRIB2 data.
//!
//! Downloads byte ranges from a GRIB2 URL and feeds them into a
//! [`StreamingParser`] so that messages can be decoded and delivered
//! via a callback as soon as they are complete — before the full
//! download finishes.

use std::io::Read;

use super::client::DownloadClient;
use crate::grib2::streaming::StreamingParser;
use crate::grib2::{Grib2Message, unpack_message};

/// Size of each read chunk when streaming from the HTTP response body.
const STREAM_CHUNK_SIZE: usize = 256 * 1024; // 256 KB

/// Download GRIB2 data from `url` using the given byte `ranges` and decode
/// messages on-the-fly, calling `on_message` for each decoded message along
/// with its unpacked data values.
///
/// This is useful when you want to start processing fields (e.g., rendering
/// a map tile) before the full GRIB2 download has completed.
///
/// Each range is downloaded sequentially (to keep ordering deterministic),
/// and within each range the bytes are streamed in chunks through a
/// [`StreamingParser`].
///
/// # Arguments
///
/// * `client` - The HTTP client to use for downloads.
/// * `url` - The GRIB2 file URL.
/// * `ranges` - Byte ranges to download (start, end inclusive).
/// * `on_message` - Callback invoked with each fully-parsed message and its
///   unpacked `f64` values. If unpacking fails, the values vector will be empty.
///
/// # Errors
///
/// Returns `Err` if any HTTP request fails. Parse errors for individual
/// messages are logged to stderr but do not abort the download.
pub fn fetch_streaming<F>(
    client: &DownloadClient,
    url: &str,
    ranges: &[(u64, u64)],
    mut on_message: F,
) -> Result<(), String>
where
    F: FnMut(Grib2Message, Vec<f64>),
{
    let mut parser = StreamingParser::new();

    for &(start, end) in ranges {
        let range_header = if end == u64::MAX {
            format!("bytes={}-", start)
        } else {
            format!("bytes={}-{}", start, end)
        };

        // Use the client's agent to make a streaming request
        let mut response = client
            .agent()
            .get(url)
            .header("Range", &range_header)
            .call()
            .map_err(|e| format!("HTTP request failed for {}: {}", url, e))?;

        // Read the response body in chunks via as_reader()
        let mut reader = response.body_mut().as_reader();
        let mut buf = vec![0u8; STREAM_CHUNK_SIZE];

        loop {
            let n = reader
                .read(&mut buf)
                .map_err(|e| format!("Read error streaming {}: {}", url, e))?;
            if n == 0 {
                break;
            }

            parser.feed(&buf[..n]);

            // Deliver any complete messages immediately
            if parser.has_messages() {
                for msg in parser.take_messages() {
                    let values = unpack_message(&msg).unwrap_or_default();
                    on_message(msg, values);
                }
            }
        }
    }

    // Drain any remaining messages
    match parser.finish() {
        Ok(remaining) => {
            for msg in remaining {
                let values = unpack_message(&msg).unwrap_or_default();
                on_message(msg, values);
            }
        }
        Err(e) => {
            eprintln!("StreamingParser finish warning: {}", e);
        }
    }

    Ok(())
}

/// Download a full URL (no byte ranges) and stream-decode GRIB2 messages,
/// calling `on_message` for each.
///
/// This is the simpler variant for when you want to download an entire
/// GRIB2 file while decoding messages incrementally.
pub fn fetch_streaming_full<F>(
    client: &DownloadClient,
    url: &str,
    mut on_message: F,
) -> Result<(), String>
where
    F: FnMut(Grib2Message, Vec<f64>),
{
    let mut response = client
        .agent()
        .get(url)
        .call()
        .map_err(|e| format!("HTTP request failed for {}: {}", url, e))?;

    let mut parser = StreamingParser::new();
    let mut reader = response.body_mut().as_reader();
    let mut buf = vec![0u8; STREAM_CHUNK_SIZE];

    loop {
        let n = reader
            .read(&mut buf)
            .map_err(|e| format!("Read error streaming {}: {}", url, e))?;
        if n == 0 {
            break;
        }

        parser.feed(&buf[..n]);

        if parser.has_messages() {
            for msg in parser.take_messages() {
                let values = unpack_message(&msg).unwrap_or_default();
                on_message(msg, values);
            }
        }
    }

    match parser.finish() {
        Ok(remaining) => {
            for msg in remaining {
                let values = unpack_message(&msg).unwrap_or_default();
                on_message(msg, values);
            }
        }
        Err(e) => {
            eprintln!("StreamingParser finish warning: {}", e);
        }
    }

    Ok(())
}

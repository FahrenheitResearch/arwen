#![allow(dead_code)]
#![allow(mutable_transmutes)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
#[macro_use]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std as alloc;

#[cfg(feature = "file-io")]
extern crate libc;

#[cfg(not(feature = "std"))]
use alloc::string::String;

mod c_api_types;
mod consts;
mod types;

#[macro_use]
mod event;

#[cfg(feature = "file-io")]
#[macro_use]
mod fprintf;

// Public OpenJpeg interface.
pub mod image;
pub mod openjpeg;
pub mod stream;

// Export safe API
pub use c_api_types::*;
pub use codec::Codec;
pub use image::{opj_image, opj_image_comptparm};
pub use types::{ICCProfile, ICCProfileMut, ICCProfileRef, Stream, TileInfo};

/// Magic bytes for JP2 RFC3745.
pub const JP2_RFC3745_MAGIC: &'static [u8] = &[
  0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a,
];
pub const JP2_MAGIC: &'static [u8] = &[0x0d, 0x0a, 0x87, 0x0a];
/// Magic bytes for J2K Codestream.
pub const J2K_CODESTREAM_MAGIC: &'static [u8] = &[0xff, 0x4f, 0xff, 0x51];

/// Supported Jpeg 2000 formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum J2KFormat {
  JP2,
  J2K,
  JPT,
}

/// Detect Jpeg 2000 format from magic bytes.
pub fn detect_format(buf: &[u8]) -> Result<J2KFormat, String> {
  if buf.starts_with(JP2_RFC3745_MAGIC) {
    Ok(J2KFormat::JP2)
  } else if buf.starts_with(JP2_MAGIC) {
    Ok(J2KFormat::JP2)
  } else if buf.starts_with(J2K_CODESTREAM_MAGIC) {
    Ok(J2KFormat::J2K)
  } else {
    Err("Can't detect image format from bytes".into())
  }
}

/// Detect Jpeg 2000 format from file extension.
pub fn detect_format_from_extension(ext: Option<&str>) -> Result<J2KFormat, String> {
  let lower_ext = ext.map(|e| e.to_ascii_lowercase());
  match lower_ext.as_deref() {
    Some("jp2") => Ok(J2KFormat::JP2),
    Some("jpt") => Ok(J2KFormat::JPT),
    Some("j2k") | Some("j2c") | Some("jpc") => Ok(J2KFormat::J2K),
    // HTJ2K with JP2 boxes
    Some("jph") => Ok(J2KFormat::JP2),
    // HTJ2K codestream
    Some("jhc") => Ok(J2KFormat::J2K),
    Some(ext) => Err(format!("Unknown file extension: {}", ext)),
    None => Err("No file extension".into()),
  }
}

#[cfg(feature = "std")]
pub fn detect_format_from_extension_os_str(
  ext: Option<&std::ffi::OsStr>,
) -> Result<J2KFormat, String> {
  let ext = ext.map(|ext| ext.to_string_lossy());
  detect_format_from_extension(ext.as_deref())
}

#[cfg(feature = "file-io")]
fn detect_from_file_magic<P: AsRef<std::path::Path>>(path: P) -> Result<J2KFormat, String> {
  use alloc::io::Read;
  // Read magic bytes from file
  let mut buf = [0; 12];
  let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
  file.read_exact(&mut buf).map_err(|e| e.to_string())?;

  // Detect format from magic bytes
  detect_format(&buf)
}

#[cfg(feature = "file-io")]
pub fn detect_format_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<J2KFormat, String> {
  let path = path.as_ref();
  let ext = path.extension().map(|ext| ext.to_string_lossy());
  let ext_format = detect_format_from_extension(ext.as_deref())?;
  if ext_format == J2KFormat::JPT {
    return Ok(J2KFormat::JPT);
  }

  // Detect format from magic bytes
  let Ok(magic_format) = detect_from_file_magic(path) else {
    return Ok(ext_format);
  };

  // Log warning if magic bytes and file extension don't match
  if ext_format != magic_format {
    eprintln!(
      "Warning: File extension format ({:?}) doesn't match magic bytes format ({:?})",
      ext_format, magic_format
    );
  }

  Ok(magic_format)
}

mod bio;
mod cio;
mod codec;
mod dwt;
mod function_list;
mod ht_dec;
mod invert;
mod j2k;
mod jp2;
mod malloc;
mod math;
mod mct;
mod mqc;
mod pi;
mod sparse_array;
mod t1;
mod t1_ht_luts;
mod t1_luts;
mod t2;
mod tcd;
mod tgt;

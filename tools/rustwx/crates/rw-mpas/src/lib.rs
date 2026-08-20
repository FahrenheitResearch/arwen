//! MPAS mesh handling and the history-to-wrfout render bridge, in Rust.
//!
//! This is the model-output-to-pixels half of the one-stop preprocessing
//! engine: it takes an MPAS history frame on an unstructured mesh, gathers
//! every field with a WRF product equivalent onto a structured render window,
//! and writes a wrfout-shaped classic netCDF frame the pinned renderer opens
//! directly. Reading is netcrust; writing is `rw_store::netcdf_classic`. No
//! HDF5 write binding, no Python, no SciPy.

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

pub mod convert;
pub mod error;
pub mod fieldmap;
pub mod history;
pub mod init;
pub mod weights;
pub mod window;

pub use error::{MpasError, MpasResult};

/// SHA-256 of a file, streamed. Every digest this crate publishes -- and
/// every digest it refuses against -- comes from here.
pub fn sha256_file(path: &Path) -> MpasResult<String> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 16 * 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(weights::hex(&hasher.finalize()))
}

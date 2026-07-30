//! Unified pure-Rust WRF file reader.
//!
//! Dispatches between classic NetCDF/CDF files and NetCDF4/HDF5 files by magic
//! bytes. Both paths are Rust-only.

use std::io::Read;
use std::path::Path;

use crate::classic_netcdf_reader::ClassicNetcdfFile;
use crate::error::{WrfError, WrfResult};
use crate::hdf5_reader::PureRustFile as Hdf5PureRustFile;

const CDF_MAGIC_PREFIX: [u8; 3] = *b"CDF";
const HDF5_SIGNATURE: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0D, 0x0A, 0x1A, 0x0A];

pub enum PureRustFile {
    Classic(ClassicNetcdfFile),
    Hdf5(Hdf5PureRustFile),
}

impl PureRustFile {
    pub fn open<P: AsRef<Path>>(path: P) -> WrfResult<Self> {
        let path = path.as_ref();
        let mut file =
            std::fs::File::open(path).map_err(|error| WrfError::NetCdf(error.to_string()))?;
        let mut magic = [0u8; 8];
        let bytes_read = file
            .read(&mut magic)
            .map_err(|error| WrfError::NetCdf(error.to_string()))?;

        if bytes_read >= 4 && magic[..3] == CDF_MAGIC_PREFIX {
            return Ok(Self::Classic(ClassicNetcdfFile::open(path)?));
        }
        if bytes_read >= 8 && magic == HDF5_SIGNATURE {
            return Ok(Self::Hdf5(Hdf5PureRustFile::open(path)?));
        }

        Err(WrfError::NetCdf(
            "unsupported WRF file format for pure-Rust reader".to_string(),
        ))
    }

    pub fn has_dataset(&self, name: &str) -> bool {
        match self {
            Self::Classic(reader) => reader.has_dataset(name),
            Self::Hdf5(reader) => reader.has_dataset(name),
        }
    }

    pub fn dataset_shape(&self, name: &str) -> WrfResult<Vec<usize>> {
        match self {
            Self::Classic(reader) => reader.dataset_shape(name),
            Self::Hdf5(reader) => reader.dataset_shape(name),
        }
    }

    pub fn read_f64_slice(&self, name: &str, time_index: usize) -> WrfResult<Vec<f64>> {
        match self {
            Self::Classic(reader) => reader.read_f64_slice(name, time_index),
            Self::Hdf5(reader) => reader.read_f64_slice(name, time_index),
        }
    }

    pub fn read_u8(&self, name: &str) -> WrfResult<Vec<u8>> {
        match self {
            Self::Classic(reader) => reader.read_u8(name),
            Self::Hdf5(reader) => reader.read_u8(name),
        }
    }

    pub fn global_attr_f64(&self, name: &str) -> WrfResult<f64> {
        match self {
            Self::Classic(reader) => reader.global_attr_f64(name),
            Self::Hdf5(reader) => reader.global_attr_f64(name),
        }
    }

    pub fn global_attr_i32(&self, name: &str) -> WrfResult<i32> {
        match self {
            Self::Classic(reader) => reader.global_attr_i32(name),
            Self::Hdf5(reader) => reader.global_attr_i32(name),
        }
    }

    pub fn global_attr_string(&self, name: &str) -> WrfResult<String> {
        match self {
            Self::Classic(reader) => reader.global_attr_string(name),
            Self::Hdf5(reader) => reader.global_attr_string(name),
        }
    }
}

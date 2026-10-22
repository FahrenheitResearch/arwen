//! This crate is for building `c-blosc` and linking to the static build.

#![allow(clippy::redundant_static_lifetimes)]
#![allow(non_camel_case_types)]

include!("bindgen.rs");

#[cfg(feature = "zlib")]
extern crate libz_sys;

#[cfg(feature = "zstd")]
extern crate zstd_sys;

#[cfg(feature = "lz4")]
extern crate lz4_sys;

#[cfg(feature = "snappy")]
extern crate snappy_src;

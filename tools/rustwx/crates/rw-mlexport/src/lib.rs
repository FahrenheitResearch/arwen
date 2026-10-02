//! The machine-learning dataset exporter: wrfout-shaped history files in,
//! one Zarr dataset per domain out, on pressure levels with ERA5's names,
//! units and below-ground rules.
//!
//! `gpuwm ml-export` resolves its tables into a request
//! ([`request::Request`], schema `ml-export.request/v1`) and runs
//! `rw_mlexport --request FILE`; every array operation happens here.

pub mod accumulation;
pub mod blosc;
pub mod error;
pub mod export;
pub mod finalize;
pub mod frame;
pub mod grid;
pub mod inputs;
pub mod ops;
pub mod regrid;
pub mod request;
pub mod state;
pub mod times;
pub mod vertical;
pub mod zarr;
pub mod zipin;
pub mod zipout;

/// The contract line `rw_mlexport --abi` prints, and the literal
/// `gpuwm.bridges.BRIDGE_ABI_MARKERS` looks for in the built binary: the
/// request schema, the modes and the progress grammar, which is what
/// changes when the contract does.
pub const ABI: &str =
    "rw_mlexport --request REQUEST.json schema=ml-export.request/v1 modes=run,append,finalize progress=jsonl";

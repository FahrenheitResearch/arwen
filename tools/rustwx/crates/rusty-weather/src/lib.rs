//! Reusable Rusty Weather engine surfaces.
//!
//! The command-line binaries remain thin hosts.  GUI applications can use
//! the same store-backed production renderer through [`batch_render`]
//! instead of copying a renderer into an app shell.

pub mod batch_render;
/// The host-memory reader, shared with the MPAS static builder
/// (`rw-host-memory`).
pub use rw_host_memory as host_memory;
pub mod render_all;

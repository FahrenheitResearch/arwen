//! Reusable Rusty Weather engine surfaces.
//!
//! The command-line binaries remain thin hosts.  GUI applications can use
//! the same store-backed production renderer through [`batch_render`]
//! instead of copying a renderer into an app shell.

pub mod batch_render;
mod host_memory;
pub mod render_all;

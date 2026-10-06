# netcdf-reader 0.3.0

Source: [roteiro-gis/netcdf-rust](https://github.com/roteiro-gis/netcdf-rust),
the netcdf-reader 0.3.0 registry snapshot already mirrored in this tree.
License: MIT OR Apache-2.0, with both license files retained.

The Rust sources, tests and benchmark are unchanged from that mirror.
The normalized Cargo.toml restores the original sibling HDF5 dependency
path, `../hdf5-reader`, recorded in Cargo.toml.orig. NetCDF metadata and
values therefore use the same HDF5 decoder when the facade is built
outside a workspace that declares registry patches.

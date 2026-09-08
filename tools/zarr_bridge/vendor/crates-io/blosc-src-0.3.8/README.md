# blosc-src

This is an FFI crate for using the Blosc compressor as implemented in [`c-blosc`](https://github.com/Blosc/c-blosc). The blosc compressor is a library of lossless compressors, which enables the developer to more easily use different compression algorithms depending on the data.


## Non-rust dependencies
The crate builds `c-blosc` from source using the `cc` crate. As such it is required to have a C compiler installed.


## Features
`c-blosc` can transparently use different compressors, but some of these are only available when included though `cargo` features. These include
* `zlib`
* `zstd`
* `lz4`
* `snappy`

When these are requested they will be built from source and available for use by `blosc`.

## Usage
As this crate only provides FFI, the [examples from `c-blosc`](https://github.com/Blosc/c-blosc/tree/main/examples) also functions as examples for this crate. A simple roundtrip is included in [`test.rs`](tests/test.rs).

The developer must take special care in dealing with memory and in multi-threaded environments. It is recommended to create and use a safe interface instead of this crate directly.

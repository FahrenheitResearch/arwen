# Optional coordinate setup decoder

Complete notices for the `zarr-coordinates` feature accompany this file.
The actual decoder source notices were read before introducing the dependency.

| Source | Pinned version | Selected notice |
| --- | --- | --- |
| zstd | 0.13.3 | MIT, “The MIT License (MIT)” |
| zstd-safe | 7.2.4 | MIT option, “The MIT License (MIT)” |
| zstd-sys | 2.0.16+zstd.1.5.7 | MIT wrapper, BSD-3-Clause generated bindings |
| bundled Zstandard | 1.5.7 | BSD option, “Redistribution and use in source and binary forms ... are permitted” |
| cc | 1.8.0 | MIT option |
| find-msvc-tools | 0.1.14 | MIT option |
| jobserver | 0.1.35 | MIT option |
| pkg-config | 0.3.34 | MIT option |
| shlex | 2.0.1 | MIT option |
| getrandom | 0.4.3 | MIT option; jobserver build dependency |
| r-efi | 6.0.0 | MIT option in AUTHORS; conditional UEFI dependency, not built on Linux |

Exact crates.io source archive checksums are in `Cargo.lock`. Copied notices
retain their copyright. The bundled Zstandard header explicitly offers BSD
instead of GPLv2; r-efi's AUTHORS explicitly offers MIT instead of Apache/LGPL.
No GPL or LGPL option is selected. Build dependencies add no host numerical
work to resident stepping.

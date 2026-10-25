# Hydro projection dependency

The three hydro bridges link PROJ 9.7.1 and SQLite 3.46.1 statically on
Linux and Windows. PROJ's database and resource files are embedded in the
library. External resource lookup, TIFF grids and network support are disabled.
Runtime installation of PROJ, SQLite or `proj.db` is unnecessary.
Windows builds also use the static Microsoft C/C++ runtime; they add no
Visual C++ redistributable installation step.

The build needs the normal C/C++ compiler and CMake 3.22 or newer. Cargo's
jobserver controls the CMake build parallelism. Windows build paths should be
short (for example `CARGO_TARGET_DIR=C:\hps-build`) to avoid CMake's Windows
path-length limit.

Sources are checked against SHA-256 before unpacking:

| Archive | Upstream | SHA-256 |
| --- | --- | --- |
| `vendor/proj-9.7.1.tar.gz` | https://download.osgeo.org/proj/proj-9.7.1.tar.gz | `6c097dc803c561929cdfcc46e4bf9945ea977611fb31493ad14e88edaeae260f` |
| `vendor/sqlite-3.46.1.tar.gz` | https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip | `4b1a04df3638bcf51960eeebca6305c60cbeab9846306b0b9c9a296e9a1b5b5e` |

The SQLite archive repacks the contents of the upstream amalgamation directory,
including its original source and headers, without source changes. SQLite is in
the public domain; its source headers carry that notice. PROJ's upstream notices
are preserved in the archive and in `PROJ-COPYING`.
The wheel carries the same notice through `licenses/HYDRO-PROJ-COPYING.txt`.

SQLite enables `SQLITE_SECURE_DELETE` to match the Linux baseline's GeoPackage
layout. Changing SQLite versions or this flag can change file bytes even when
all SQL rows agree. The projection and SQLite versions are deliberately pinned
to the measured dlopen baseline; upgrade them with a fresh byte comparison.

`rw_hydro_world --proj-probe`, `rw_hydro_init --proj-probe` and
`rw_hydrofabric --proj-probe` load the embedded database and transform one point.
The hydrofabric shared library exports `arwen_hydrofabric_proj_probe` for doctor.

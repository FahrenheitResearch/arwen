# WPS orographic arithmetic fixtures

The input modules are byte-unmodified WPS v4.6.0 geogrid sources from
https://github.com/wrf-model/WPS/tree/v4.6.0/geogrid/src.
`SOURCE.sha256` pins every module and the C logging support file.

Run `bash build.sh WPS_SOURCE_DIRECTORY OUTPUT_DIRECTORY` with gfortran 15.
The wrapper writes 137,048 float32 forward coordinates, 222 inverse
coordinates, 32 interpolation outputs, 12 half-integer NINT results, and
the sums, counts and nine means of a positive grid-cell aggregation.
Copy the six binaries into
`tools/rustwx/crates/static-fields/golden/orographic/` and run
`cargo test --release -p static-fields orographic -- --nocapture`.

Lambert, Mercator and polar coordinates and both interpolators match the
source Fortran bits on the qualified Linux build. The polar transform
retains WPS's REAL(HIGH) intermediate precision.
The interpolation fixture covers ordinary fractional coordinates, exact
source rows and columns, half-cell tile edges, and post-interpolation
scaling of integer source words, one missing corner, and all corners missing.
Projection fixtures include tangent Lambert and southern-hemisphere grids.
The aggregation fixture maps a 21 by 21 regular latitude-longitude source
at 0.2 degrees onto a 3 by 3 Lambert grid at 100 km. The Fortran wrapper
uses the unmodified WPS inverse transforms and NINT, then accumulates
source words and counts in default REAL before averaging and scaling.

The retained full geogrid comparison has different latitude and longitude
bits from the strict source build at some cells. Its few integer-boundary
stencil differences are not accepted as an arithmetic oracle.

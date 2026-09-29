Portable WPS float32 authority

The previous WPS twin goldens used platform NumPy/libm rounding. Those bits
are unsuitable for statics prepared on one operating system and rebuilt on
another: at 2 km over 30 arc-second terrain, the first move changed 2160
terrain values and was refused. The fixed implementations use vendored
libm 0.2.16 with force-soft-floats. The existing sin/cos/exp/log kernels and
the float32 operation order stay unchanged.

This directory pins the new float32 twin states, forward/inverse coordinates,
latitude sampling surfaces and compiler-band masks. It does not replace the
separate float64 NumPy authorities in ../lane1 or a native qualification
folder. Those fields retain their original assertions. Every payload here
has a SHA256 in manifest.json. Both Windows and Linux must read these same
portable bytes, regardless of GPUWM_STATIC_LANE1_GOLDENS.

To re-pin into a new evidence directory, explicitly set
GPUWM_STATIC_WPS32_OUTPUT and run:
  cargo run --release --offline -p static-fields --example repin_portable

The initial portable set was written by the fresh Windows build and checked
on Linux. The permanent fine-grid regression separately rebuilds 30
arc-second GEOG statics at 2 km, 1 km and 500 m and compares the shared
footprint against Windows-prepared hashes, including terrain and TMN.
The 2 km GPU run checks the actual overlap gate and donor alignment.

The sampler float64 longitudes and corner coordinates are pinned here too.
Every portable array is independently bounded against the original NumPy
array by numpy-bounds.json. See static_qualification.json for the declared
authority and docs/dev/static-platform-qualification.md for its limits.

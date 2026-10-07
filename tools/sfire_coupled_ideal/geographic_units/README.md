The original WRF geographic ignition units now enter the default coupled GPU
driver. WRF's `reradius` is the reciprocal Earth radius, so its original
`pi2 / (360 * reradius)` expression is correct. The previous constructor used
double constants and cosine before rounding to REAL. That changed the
latitude unit by two ULP and 59 of the 64 unit words in this control.

The source-pinned extraction copies the two original driver expressions,
the original REAL constant declarations and the complete original `nearest`
routine without changing their scientific bytes. The wrapper calls them at
32 latitudes, including signs, signed zero, near-pole values and both poles,
then evaluates point and line distances and times. Build with
`bash tools/sfire_coupled_ideal/geographic_units/build.sh` under a bounded
CPU allocation. Native streams are packaged without numerical transformations.

Both RTX 4090 and RTX 5090 controls pass 12 checks. The separately graded
64 unit and 128 point/line words are bit-identical, zero ULP. Ten constructor
checks per card also run the actual ignition kernel with the produced units.
The device helper retains the native REAL expression order and the existing
glibc scalar cosine transcription; its compile options are explicit in the
receipts. The pole cosine is nearly zero, so the former constructor's large
ULP distance there represents a small absolute metre-per-degree difference.

The metric full-hour ideal runs and observed-perimeter real runs do not use
this geographic point/line conversion. Their immutable numerical/source
receipts remain unchanged. This source correction is separately qualified;
the earlier claim that all SFIRE source bytes still match those older
executed manifests does not apply to the new coupled-driver composition.

# Davies target alignment

The specified-boundary U and V tables count inward from the outermost
staggered point. For an east U row with 151 points, zero-based points
147, 148, 149 and 150 read width slots 3, 2, 1 and 0. This is WRF's
`b_dist = ide - i`, followed by its one-based `b_dist + 1` table index.
The staggered extent belongs to the field, not the mass grid.

The first solve uses the post-increment boundary clock. A 12-second step
therefore evaluates the target at 12 seconds, even while the state's
elapsed time is still zero. The native-input forecast runner binds its
root clock before stepping. If a nonlinear boundary time law evaluates
the tables before the state kernel, a zero kernel offset refers to that
already-evaluated table; it does not imply a time-zero target.

A first-step capture at the first east U relaxation point found the
time-zero table and coupled current both equal to 26097.34375. The
received boundary offset was 12 seconds and the target was
26056.89453125. The residual was -40.44921875 in WRF's coupled units,
equivalent to approximately -0.000437831 m/s after mass and map-factor
normalization. It is the expected rounded 12-second boundary change.
The held relaxation tendencies at the three inward points were
-0.10554090887308121, -0.24488498270511627 and -0.367449551820755,
with the same stored values in compiled WRF. The specified outer point
has zero held relaxation tendency.

This capture rejects a shifted target slot and a one-step target lag at
the nominated point. It does not identify the operator responsible for
the remaining boundary wind difference. No target-index or clock
formula was changed on the basis of this test.

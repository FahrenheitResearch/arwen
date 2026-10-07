# Native moisture driver control

The decision block is extracted byte for byte from the pinned WRF v4.7.1
fire driver. A minimal clock/configuration record supplies the fields that
the block reads. Original compiled WRF `advance_moisture` and `fuel_moisture`
perform the model evolution and refined-grid interpolation. The original
source and two explicitly corrected controls are recorded separately.

The source clock uses `itimestep*dt`, one atmosphere interval ahead of the
interval being advanced. The corrected clock uses
`max(itimestep-1,0)*dt`. The source initializes moisture only when an actual
model call occurs on atmosphere step one. With a positive frequency above
one, the first due call uses uninitialized old surface fields and does not
initialize the class moisture. The corrected caller initializes the first
actual scheduled model call, preserving its native elapsed duration.

The initialization pass sets `LASTTIME` and `NEXTTIME`, but cannot advance
the model because its call is inside fire pass three. Positive-frequency
scheduling never updates `NEXTTIME`; it remains the initial value. Time
interval scheduling updates it using binary32 addition. Moisture-only
runs advance the coarse classes and leave fine fuel parameters untouched.

Twelve controls contain ten atmosphere steps each: time intervals of
600, 275.5 and 0.1 seconds; integer frequencies 1, 3 and 5; interpolation
enabled or disabled; moisture-only; interpolation-only; and three active
classes inside five allocated classes. Realistic changing temperature,
pressure, vapor and accumulated rain drive the original physics. The
actual GPU driver and production interpolation pass 80,880 graded words
on each card, with zero differences and 0 ULP.

Reproduce with `build.sh WRF_SOURCE_ROOT NATIVE_ORACLE_BUILD OUTPUT`, pack
with `tools.sfire_wrf471_oracle.pack_fixtures`, and run
`tests/test_sfire_moisture_driver_wrf471_parity.py`. The extraction hash,
support-object hashes, per-card grades and logs are in `receipts/`.

The additional `fixtures_nfmc7` corpus allocates seven moisture classes
while advancing five, or three in the reduced-class control. The same
compiled moisture objects and scheduling control supply all 120 steps.
The actual driver grades 98,160 words on the RTX 5090 with zero differences
and 0 ULP. Unused allocated classes retain their supplied values. Generate
this corpus with `build.sh WRF_SOURCE_ROOT NATIVE_ORACLE_BUILD OUTPUT 7`;
`allocated_classes.py` changes only the harness allocation declaration.

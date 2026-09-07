# Supplied RUC and MYNN physics fields

The wrfinput reader now accepts the following selected physics input fields,
validates their declared geometry and values, and restores them through the
existing physics initialization assignment loop:

| WRF field | Geometry | Existing consumer |
| --- | --- | --- |
| ACRUNOFF | Mass surface | RUC `fields['acrunoff']` to LSMRUC `acrunoff` |
| RHOSNF | Mass surface | RUC `fields['rhosnf']` to LSMRUC `rhosnf` |
| SNOWFALLAC | Mass surface | RUC `fields['snowfallac']` to LSMRUC `snowfallac` |
| SOILT1 | Mass surface | RUC `fields['soilt1']` to LSMRUC `soilt1` |
| qke / QKE | Mass volume | MYNN `fields['qke']` to column `qke` |

The RUC fields require `sf_surface_physics=3`; the MYNN fields require
`bl_pbl_physics=5`. Conflicting qke/QKE aliases are refused. Missing optional
fields leave the existing cold-start initialization intact.

Stock WRF v4.6.1 commit `d66e442fccc04111067e29274c9f9eaccc3cef28`
declares these fields in `Registry/Registry.EM_COMMON:866,1003-1004,1119,1975`.
The RUC package is at line 3147. The MYNN package at line 3168 allocates
`qke_adv` even when TKE advection is disabled. Its three MYNN accesses are
conditional on `bl_mynn_tkeadvect` in
`phys/module_bl_mynn.F:837-841,861-864,1442-1445`.

Consequently, the reader retains and validates `qke_adv` as inactive input
only with an explicit `bl_mynn_tkeadvect=False` selection. It is not an alias
for qke and is not reported as a mapped live field. An active or unresolved
selection names the unimplemented qke_adv transport/feedback operation.

`tests/test_wrfinput_physics_fields.py` covers both qke spellings, exact
FP32 restoration, finite/missing-value/staggering controls for every added
field, selected-consumer checks, conflicting aliases, the inactive/active
qke_adv distinction, and absent-field preservation. GPU tests read actual
NetCDF fixtures through the Rust decoder, initialize the actual physics
drivers, and intercept the existing RUC/MYNN column entries to verify their
arguments exactly. The MYNN test also verifies that stock cold-start
`initflag=1` is retained: forwarding input is not a claim that the column
solver preserves every input through its own first-step initialization.

Measured result: 27 CPU controls and 3 GPU controls pass. These establish
the named reader-to-consumer connections, not full-file or full-forecast
parity for a modified real.exe product carrying additional private fields.

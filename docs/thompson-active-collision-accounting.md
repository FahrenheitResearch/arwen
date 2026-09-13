# Classic Thompson active collision accounting

The warm snow/cloud and graupel/cloud collection terms are absent from the
fusion-heating sum in both the classic source and the pinned reference at
`d66e442fccc04111067e29274c9f9eaccc3cef28`. They are not missing CUDA terms.
The reference warm expression is at `phys/module_mp_thompson.F:3175`; the
cold expression immediately above it includes the cold collection terms.

Two active columns start at 274.15 K and 80000 Pa with cloud, rain, snow and
graupel mixing ratios of 0.001, 0.0003, 0.0002 and 0.0002 kg/kg, and rain
number of 30000/kg. Saturated and 80-percent liquid-saturation inputs activate
both cloud-collection terms and both rain/frozen collision families after
their limiters. Canonical assets and all 87 sampled coefficient values agree
exactly. The source-rate comparison uses complete compiled reference calls,
not only a temperature expression evaluated with CUDA's own rates.

For source-stage bookkeeping, let C be the sum of snow/cloud and graupel/cloud
collection, V the signed frozen vapor-source sum, and F minus the melt and
signed rain-collision rain tendencies. The active warm groups satisfy

```
dqv/dt = -V/rho
d(qs+qg)/dt = (V+C+F)/rho
dT/dt = inverse_cp * (334000*F + 2834000*V)/rho
```

Total source water closes within floating-point accumulation error. Holding
the coefficients fixed and treating all snow and graupel as solid, the ledger
`dT/inverse_cp + 2500000*dqv - 334000*d(qs+qg)` has the residual
`-334000*C*dt/rho`. The two ten-second source cases give about -9.13 and
-10.27 J/kg. This conditional residual is present in the reference too.
The classic wet-particle treatment has no prognostic coating-liquid fraction
that supplies a complete alternative enthalpy ledger. Adding the two terms
would select a new phase model, not restore the transcribed expression.

Cloud adjustment supplies vaporization heating, ordinary rain evaporation
supplies its cooling, and final cloud-ice cleanup supplies its phase heating.
Fallout moves particle mass and number; its temperature arrays are read-only.
The adapter then converts temperature to theta and stores the same increment
once in the actual heating owner. Output-due reflectivity does not alter that
trajectory. A precipitation-only column sum is not a complete enthalpy proof:
cloud settling through the bottom and the reference's changing density
carriers must be included separately. No general closed-column enthalpy or
forecast-skill claim follows from these source checks.

The complete columns exposed two distinct composition errors that are fixed:

- The reference forms rain mass and number concentrations before cloud
  adjustment (`module_mp_thompson.F:3236`). Those concentrations persist
  unless ordinary rain evaporation executes and refreshes them at its own
  incoming density (`:3568`). Recapturing density on an inactive evaporation
  call changed subsequent fallout after cloud evaporation.
- Positive cloud condensation suppresses same-call rain evaporation
  (`:3502`). A small saturation residual cannot replace that held decision.

The adapter carries both decisions into the existing rain helper. It reuses
the full-theta scratch after the pre-physics theta has been saved; no extra
state-sized allocation is required. Isolated calls without history retain
their existing behavior. Warm and cold collision arithmetic and canonical
table identities are unchanged. The continuation algorithm advances to v4
because corrected fallout can change later states; an old v3 continuation
must retain its original algorithm or begin a new run with the corrected one.

`tools/thompson_wrf461_oracle/active_collision_fixture.py` regenerates the
three four-level column fixtures through the complete pinned classic driver.
The third input has 110-percent liquid saturation and activates condensation.
`tests/test_thompson_active_collision.py` checks all mass/number fields,
surface precipitation, actual theta/heating ownership and output-due parity.
The fixture does not replace the existing oracle records or their tolerances.

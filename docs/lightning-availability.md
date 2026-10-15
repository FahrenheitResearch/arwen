# Lightning producers and named gaps

The selector contract is `gpuwm/lightning_contract.py`. Lightning is off by
default. These are producer requirements, not observation-based skill claims.

| Selector | Required producer | Named refusal |
|---|---|---|
| 1, PR92 maximum vertical velocity | Moist microphysics and `do_radar_ref=1` | Missing reflectivity prevents convective-cell identification. |
| 2, PR92 resolved cloud top | Moist microphysics and `do_radar_ref=1` | Missing reflectivity prevents convective-cell identification. |
| 3, lightning potential index | Separate graupel mass `qg` | The selected microphysics does not publish the graupel mass used in the ice mixture. |
| 11, PR92 parameterized cloud top | `ltng_ktop_deep` from Grell-Freitas (`cu_physics=3`) or Grell 3D (`5`) | Other cumulus adapters do not publish the deep-cloud top consumed by this diagnostic. |

2.9.0 admits every accepted microphysics selector whose state allocates and
transports `qg`: 2, 6, 8, 9, 10, 16, 18, 24, 28, 30, 106 and 900 (TEMPO).
In particular, NSSL selector 18 has graupel and must not be described as
lacking it. Selectors without `qg` (1, 3, 4, 5, 50) are refused by number. The Grell 3D adapter copies
its actual `ktop_deep` output into the lightning scratch field, as the
Grell-Freitas adapter already does.

Selectors outside 0, 1, 2, 3 and 11 have no implemented flash or LPI producer.
Dry states and microphysics selector 0 have no moist species for the diagnostic.
Invalid IC:CG partition methods, convective-cell counting methods, non-finite
clock or scaling values, the undefined ratio 0/0, and a ratio of -1 are refused
separately. The messages name the missing field or undefined operation.

Admission tests cover the expanded selector list. The newly admitted NSSL and
Grell 3D combinations have not yet been run on a device against the WRF 4.6.1
diagnostic, so no end-to-end parity is claimed for them.

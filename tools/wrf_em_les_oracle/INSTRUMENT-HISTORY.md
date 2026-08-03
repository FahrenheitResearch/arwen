# Instrument change history (WP-L7 oracle side)

Recorded because a scorer edited after seeing its own output is the classic
way a comparison becomes unfalsifiable. No band is cut anywhere in this lane,
so none of these changes can move a gate; they are logged so the order is
auditable rather than asserted.

**Written and validated before any matched run existed.** All metric
definitions -- z_i as the total-flux minimum, w* from the prescribed Q_s, the
0.1-0.7 z_i fit band, the 1:int(0.7*nz) index band for the resolved fraction,
the fnm/fnp SGS flux, the final-30-minute window, the Parseval-checked
spectrum -- were authored against the ArWen case source and validated on the
stock `em_les` smoke run, which completed before the first matched run.

**Changed after seeing `match_km3_100m` output:**

1. `resolved_fraction_ml` now returns null under `km_opt=3` instead of the
   value it actually produced, which was exactly **1.0**. Under a diagnostic
   Smagorinsky closure there is no prognostic SGS TKE, so that ratio has no
   denominator; 1.0 was an absent-denominator artifact claiming "100 %
   resolved". ArWen's own case module makes the same call for the same reason.
   This removes a number, it does not improve one.
2. Added a `final_sample` block. ArWen's committed receipt entries
   (`wth_res_max_over_qs`, `resolved_fraction_ml`, `tke_res_max`) are computed
   from the **last sample**, while this scorer's primary values are
   30-minute-window averages. Comparing one against the other would have been
   comparing an average with an instantaneous value. Both conventions are now
   emitted and the comparator selects per metric to match ArWen's convention.
3. `e_sgs_ml_mean` reports null under `km_opt=3` for the same reason as (1).

**Not changed after seeing output:** every band, window, level and fit range.

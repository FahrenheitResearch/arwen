# Upper Midwest packed ensemble, 2022-06-13

The showcase config retains the full regional HRRR recipe and its declared engine differences on a 1078 x 634 mass grid at 3 km and 50 levels. Sixteen ordered members combine eight HRRR native cycles and eight RAP native cycles (18Z through 11Z) at a common 18Z start, with distinct fixed soil moisture scales. Earlier cycles initialize from their forecast at the common start. Every member retains aerosol-aware microphysics, MYNN, nine-layer RUC and legacy RRTMG.

The small config has two members on a 101 x 101 grid for three hours. The restart config is a one-hour, two-card behavior check with positive restart interval. The 6 km config checks carrying the same settings to another spacing; the ordinary domain wizard does not yet emit every full-recipe setting.

Use the released door, with the shipped free-threaded interpreter and PYTHON_GIL=0:

    gpuwm ensemble CONFIG --dry-run
    gpuwm ensemble CONFIG --prepare-only --outdir CASE_ROOT --geog-root GEOG --products refl
    gpuwm ensemble CONFIG --outdir PREPARED_STAMPED_ROOT --geog-root GEOG --products refl

Acquire all seven MRMS hourly frames before GPU work. Prepare on CPU without a queue hold. Forecast only through the authorized box queue. The visible cards determine the member device roster; a per-member spatial --devices override disables ordinary member concurrency.

The private run's scoring tools (kept with its run report outside the shipped tree, because they name the rented boxes it ran on) score stored member histories and complete-roster ensemble products through the existing observation bridges. FSS uses fixed dBZ thresholds, nearest radar matching within 4 km, and a ten-cell boundary exclusion. Reliability and Brier are descriptive; these fixed sensitivity members have no calibrated probability claim. Matplotlib draws score charts only. Comparison maps use gpuwm render --engine rust --products refc --compare mrms --compare-offline. Ensemble maps use the door's own Rust renderer.

Do not change the source preparation context while a prepared run is waiting to launch. Stochastic rows remain a future extension of this ordered roster under the owning lane's qualified admission rules; none is enabled here.

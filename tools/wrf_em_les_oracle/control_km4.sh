#!/bin/bash
# Instrument-qualification control: WRF em_les with km_opt=4.
#
# The spec calls its ArWen equivalent "the program's standing proof that the
# ladder can fail" (R1.5 / AC-L2.4): if the instruments cannot tell a closure
# that should not produce a credible CBL from one that should, the instruments
# are rejected, not the engine passed.
#
# This is the oracle-side version of that check, run on the same case, same
# forcing, same sounding, same grid, changing only km_opt 3 -> 4. WRF's
# km_opt=4 is the 2-D (horizontal) Smagorinsky closure intended for
# real-data runs; on a doubly periodic dry CBL it should carry little or no
# resolved-scale vertical SGS heat flux, and this lane's scorer -- which reads
# XKHV for the SGS term -- must SHOW that rather than quietly reporting a
# healthy-looking partition. Whatever it shows is committed.
B=${LES_ORACLE_ROOT:-$HOME/weather/les-oracle-wpl7}
R=control_km4_100m

# wait for the box: the 50 m run and then the novec arm both outrank this
while ! grep -q "SUCCESS COMPLETE WRF" "$B/runs/match_km2_50m/rsl.error.0000" 2>/dev/null; do
  sleep 120
done
while ! grep -q "NOVEC_RUNS_DONE" "$B/provenance/novec_run.log" 2>/dev/null; do
  sleep 120
done

bash "$B/assets/run_les.sh" "$R" "$B/assets/namelist.control_km4_100m" 24
python3 "$B/assets/score_wrf_les.py" "$B/runs/$R" "$B/scores/$R" \
    --window-min 30 > "$B/scores/$R.txt" 2>&1
cat "$B/scores/$R.txt"
echo "########## CONTROL_KM4_DONE $(date -u +%H:%M:%S) ##########"

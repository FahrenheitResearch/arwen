Lane 6 (radar latent heating) cycled-arm kit, as run on box N 2026-10-06.

cycle-l6.sh ARM [smoke]  one cycled arm through the GPU mutex: lane 0's 10-01 hydro case plan
                         (mkargv-hydro.py from /work/da-line), three observed legs, three 1 h free legs,
                         radar heating flags per arm (off, clear, heat, heat-clear, heat-strict, lhn-clear,
                         heat-weak = the lane 3 field-rules tree). Box paths are N's.
chain-l6.sh ARM...       the arms in order, one 8-card --min-cards 4 request each, yielding to da-obs-full.
l6_sheets.py ARM...      five-column sheets (MRMS | DA member 0 | DA mean PMM | no DA | HRRR) from the
                         arms' native REFL_10CM frames through rw_compare and the da-tune panel cutter,
                         plus 15/25/35/45 dBZ coverage bias, FSS35 27 km, object ratio, spurious echo.

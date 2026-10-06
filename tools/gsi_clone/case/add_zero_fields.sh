#!/usr/bin/env bash
# Copy a WRF-format file and add each named field as zeros on the mass grid
# (shaped like QCLOUD). A test shim for fields GSI's WRF interface reads that
# WOOF's files do not carry; every receipt that uses it names it.
#
#   add_zero_fields.sh <input> <output> FIELD [FIELD...]
#
# Used by the single-observation test for (MEASURED 2026-10-03):
#   QNCLOUD in members and background: GSI db90edf reads it unconditionally
#     once hydrometeors are in the table (members: cplr_get_wrf_mass_ensperts
#     general_read_wrf_mass2; background: the guess converter), and WOOF's
#     history and WRF input export do not write it (mp_physics 8 or 28).
#   REFL_10CM in the background: read when if_model_dbz is on; WOOF's WRF
#     input export and its first history write do not carry it.
# These zeros let the temperature-only test open the files. They are test
# placeholders, not valid inputs for reflectivity or cloud analysis.
set -euo pipefail
in="$1"; out="$2"; shift 2
expr=""
for f in "$@"; do expr+="${f}=QCLOUD*0.0f;"; done
ncap2 -O -s "$expr" "$in" "$out"

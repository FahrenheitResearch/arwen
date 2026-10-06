# Operational HRRR v4 namelists

Verbatim copies of two public files from the NOAA-EMC/HRRR repository at
tag v4.1.21 (the operational HRRR v4 source):

| file | source path | sha256 |
|---|---|---|
| hrrr_wrf.nl | parm/conus/hrrr_wrf.nl | 50ac01dbeaca863dfc313eae7dd53865458b2bffdfcc1e402d350d860bef5694 |
| hrrr_namelist.wps | parm/conus/hrrr_namelist.wps | 7b78a6e816aabc16e741ef5f3bfef286bbd119d7a84ea1dfa9cb703a141910d1 |

`scripts/conus/exhrrr_fcst.sh` copies `hrrr_wrf.nl` to `namelist.input`
unchanged apart from the start, end, run-length and interval times, so
this is the &dynamics block the operational model integrates.  It omits
both `mix_full_fields` (WRF Registry default `.false.`) and `use_theta_m`
(the fork's WRFV3.9 Registry default 0, dry theta).  The tests in
`tests/test_mix_full_fields_hrrr.py` read these files.

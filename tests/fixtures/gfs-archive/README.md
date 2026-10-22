# Historical GFS embedded missing cells

One unchanged GRIB2 message, index 250 (zero based), from
https://data.gdex.ucar.edu/d084001/2017/20170918/gfs.0p25.2017091812.f000.grib2
retrieved 2026-10-20. Soil temperature, 0 to 0.1 m, global 1440 by 721.
DRT 5.3, second-order differencing, missing-value management 1, no bitmap.
687174 missing cells and 351066 finite cells. This tests a real historical
encoding that modern bitmap-only samples do not exercise.

245399 bytes. SHA256:
5cc4fc37e3ecfca0a51690ff0f88d2dc831ac0a738e1ad8dbd27794f29626169

Terms checked at https://gdex.ucar.edu/datasets/d084001/:
"This work is licensed under a Creative Commons Attribution 4.0 International License."
Attribution: National Centers for Environmental Prediction / NOAA,
NCEP GFS 0.25 Degree Global Forecast Grids Historical Archive,
Research Data Archive / Geoscience Data Exchange, DOI 10.5065/D65D8PWK.
https://creativecommons.org/licenses/by/4.0/

This small regression fixture is retained as evidence; full downloaded cycles
and prepared arrays are disposable.

Licence provenance, checked at source on 2026-10-20
=================================================

This package is original code under the repository licence. Published equations
were implemented independently; no external wave-model source was copied.

WW3 source was excluded. Its actual LICENSE.md states GNU Lesser General Public
License version 3: https://raw.githubusercontent.com/NOAA-EMC/WW3/develop/LICENSE.md
This fails this workspace's permissive-code shipping rule. The NOAA technical
manual was consulted for equations only.

ecWAM's actual LICENSE is Apache License, Version 2.0:
https://raw.githubusercontent.com/ecmwf-ifs/ecwam/main/LICENSE
No ecWAM implementation is included. SWAN technical-manual equations were
consulted without using its model source or implementation.

The native boundary dynamically links the system netCDF-C library, BSD-3-Clause
according to its actual COPYRIGHT file:
https://raw.githubusercontent.com/Unidata/netcdf-c/main/COPYRIGHT
Its complete notice is retained in native/NETCDF-COPYRIGHT. The library is not
vendored. CuPy and NumPy are existing project dependencies, not copied source.

NOAA data terms: https://www.weather.gov/disclaimer
NOAA's disclaimer states that its information is in the public domain unless
otherwise noted. ETOPO1 from NOAA CoastWatch ERDDAP has its own dataset licence
attribute permitting free use and redistribution:
https://coastwatch.pfeg.noaa.gov/erddap/info/etopo180/index.html
NDBC observation records are NOAA data: https://www.ndbc.noaa.gov/.
Retain any dataset attribution and check dataset-specific terms when acquiring.

ERA5 single-level reanalysis is distributed under CC-BY-4.0, with attribution to
Copernicus Climate Change Service/ECMWF:
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview
The demonstration uses the public ARCO ERA5 store:
https://cloud.google.com/storage/docs/public-datasets/era5
ERA5 is research-only under this workspace's stricter data rule; it requires an
explicit named licence opt-in in each configuration and is not bundled here.

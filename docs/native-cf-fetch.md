# Historical ensemble mean acquisition

`gpuwm fetch --source 20crv3-cf --cycle 1925-03-18T12 --hours 3 --cadence 3 --area 25,-110,50,-75 --out INPUTS` downloads the NOAA PSL 20CRv3 ensemble mean analysis. No account or key is required. Coverage is 1836-01-01 00 UTC through 2015-12-31 21 UTC, including the final boundary time. Analyses are published every three hours. `--cycle latest` resolves the latest complete window within that coverage.

The annual archive switches from SI directories through 1980 to MO directories from 1981. Acquisition splits year boundaries, retries temporary HTTP failures, checks returned valid times and the Ensemble Mean attribute through Rust, and preserves prior requests when the same output directory is reused with changed settings. Matching files are reused only after their receipt hashes are verified.

The table in `gpuwm/authorities/native-cf-fetch.v1.json` declares every variable, archive era, invariant and surface policy. Rust reads NetCDF and binds the published invariant surface height and land fraction to the exact primary grid. It repeats those invariant planes at the primary valid times and writes `invariant.provenance.json`. Native acquisition does not reconstruct terrain from pressure levels.

Published tsoil and soilw supply all four Noah layers, at 0, 10, 40 and 100 cm coordinates. PSL has no separate sub-daily SST variable in this distribution. Ocean initialization uses the source skin temperature as a water boundary proxy, as the profile does for soil missing over water. The acquisition receipt declares this policy; it is not a dedicated SST observation, and no synthetic soil or SST climatology is substituted. Published snow and sea ice fields remain outside the existing profile's semantic bindings, which the preparation receipt declares.

The output includes `fetch-manifest.json`, `SHA256SUMS`, `inputs.txt`, `prep-command.txt` and `prep-arguments.json`. `gpuwm go` consumes the last document automatically through the existing staged preparation chain. A standalone preparation uses `gpuwm prep --source 20crv3-cf --input-list INPUTS/inputs.txt --supplement INPUTS/invariant.nc --author-input-manifest INPUTS/inputs.json` followed by the namelist, experiment configuration, geography root and prepared output directory.

20CRv3 is a coarse analysis on a one-degree delivered grid. A fine nested forecast does not reconstruct the historical tornado. Start with a parent around 9 to 15 km and nest gradually to the desired spacing. The NetCDF route uses the ensemble mean, which smooths member variability. It is distinct from the every-member GRIB2 route `20crv3`, whose files are not anonymously readable.

Data provided by NOAA PSL. Archive and variable descriptions: [NOAA PSL 20CRv3 distribution](https://psl.noaa.gov/data/20thC_Rean/20crv3.subdir.table.html). Method citation: DOI [10.1002/qj.3598](https://doi.org/10.1002/qj.3598).

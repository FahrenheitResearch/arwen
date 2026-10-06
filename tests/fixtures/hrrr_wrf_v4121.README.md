# HRRR v4.1.21 namelist fixture

Source: https://raw.githubusercontent.com/NOAA-EMC/HRRR/v4.1.21/parm/conus/hrrr_wrf.nl

Source SHA256: `50ac01dbeaca863dfc313eae7dd53865458b2bffdfcc1e402d350d860bef5694`

The fixture keeps the public namelist values and removes trailing whitespace.
Its canonical LF SHA256 is
`b244df1bff090f6ba7bb43817b91c842ae9b1add1bb6f8214fbb6403382c38c0`.
Its fork-only keys
identify the requested RUC lineage without relying on its filename.

The complete namelist also requests controls outside this RUC change,
including active additional diagnostics, solar albedo handling and chemistry.
Their existing importer refusals remain. The end-to-end RUC test projects the
source's land-surface selector, soil-layer count, mosaic, LAI and albedo vectors into the
supported importer fixture, explicitly disables the additional diagnostics,
and checks the resulting experiment. It is not a full operational namelist
import or forecast-equivalence claim.

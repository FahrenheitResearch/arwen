# Noah LSM parameter tables — provenance

Copied verbatim (byte-identical) from the `run/` directory of the
provenance-bound WRF v4.6.1 source tree. The canonical relative paths are
`run/{LANDUSE.TBL,VEGPARM.TBL,SOILPARM.TBL,GENPARM.TBL}`; machine-local
staging paths are intentionally not part of the distributable authority.

SHA-256 of the byte-identical WRF v4.6.1 assets (LANDUSE added 2026-07-18):

    cafdb5f4982b88c93f2cb321f18d9559e7c03b6d213388d5ae3d07b7280caa08  LANDUSE.TBL
    4612c5c2d7e5398ae925ff363634ff98d6587a4aaa7440eaf1bbcc248dad6090  VEGPARM.TBL
    efa3e1ad5dc6bd64665fcf8a2336ea4e4a2ca7fc8190a9305d363a4b13b2045b  SOILPARM.TBL
    b8d1829a745cb266a9e253e2ea13f2ae2493f05c1c40c62d785c794d4ee30a9a  GENPARM.TBL

They are parsed by `gpuwm.core.noah.load_tables`, which transcribes the
list-directed READ sequence of `SOIL_VEG_GEN_PARM`
(phys/module_sf_noahdrv.F, WRF v4.6.1): the VEGPARM section for a given
land-use classification (gpuwm default `MODIFIED_IGBP_MODIS_NOAH`, the
bundle geo_em's MMINLU), the SOILPARM `STAS` section, and GENPARM.
Columns beyond the ones the Fortran READ list consumes (e.g. the SOILPARM
BVIC/AXAJ/... columns) are ignored, exactly as list-directed input
ignores the rest of the record.

`gpuwm.core.landuse.load_landuse_table` parses the matching LANDUSE section
in the list-directed order used by `landuse_init`
(`phys/module_physics_init.F`, WRF v4.6.1).  Its cold-start values supply
roughness, surface optics, soil availability, and the raw-lake-to-water
category routing before the first surface/radiation call.

# P3 lookup table provenance

`tables/p3_lookupTable_1.dat-v5.4_2momI` is the byte-identical copy of
`WRF_source_v4.6.1_group/run/p3_lookupTable_1.dat-v5.4_2momI` from the
frozen WRF v4.6.1 reference bundle (P3 v4.5.2 intends table-1 version
5.4_2momI, phys/module_mp_p3.F:138).  SHA-256 pinned in
`tables/MANIFEST.sha256` and in `gpuwm/core/p3_tables.py`; the loader
validates size and digest before parsing.

The 3-moment table (`p3_lookupTable_1.dat-v5.4_3momI`) and the
multi-category interaction table (`p3_lookupTable_2.dat-v5.3`) are NOT
packaged: they belong to the unported mp_physics=53 / 52 options, which
gpuwm refuses by name.

# Source and licence notices

New code inherits the Rust workspace MIT licence. No third-party fire-index
implementation is incorporated. Equations, constants and fuel tables are
independently expressed and discretized.

- Nelson (2000), “Prediction of diurnal change in 10-h fuel stick moisture
  content”, https://www.frames.gov/documents/catalog/spa/nelson_2000.pdf .
  The paper says “© 2000 NRC Canada”. It is a mathematical reference, not a
  licence to redistribute its figures or a third-party implementation.
- Carlson et al. (2007), https://research.fs.usda.gov/download/treesearch/27712.pdf .
  Mathematical reference and published four-class fuel-stick parameters;
  the paper says “© IAWF 2007”. No article text/figures/code are incorporated.
- Jolly et al. (2024), https://research.fs.usda.gov/download/treesearch/68223.pdf .
  The paper is CC-BY-4.0. GSI and simplified fuel-bed parameter reference.
- USFS NFDRS4 at revision 84223b95fd395a62ea344257f80a9949cee71e6e:
  https://github.com/firelab/NFDRS4 . README states
  “NFDRS4 is public domain software”. The index-only nfdrs4.cpp module supplies
  the USFS fuel tables and independent test vectors. Its index equations were
  compiled separately for reference vectors. The deadfuelmoisture.cpp and
  livefuelmoisture.cpp files carry specific GPL notices; none of their code is
  copied, linked, translated or distributed here. The repository-level notice
  is not used to override those file notices.
- Van Wagner (1987) FTR35 and Van Wagner/Pickett (1985) FTR33:
  https://cfs.nrcan.gc.ca/publications?id=19927 and id=19973 . Equations only.
  The later NOR-X-424 listing has a noncommercial reproduction notice and is
  not used as a code source.
- Keetch/Byram (1968), https://research.fs.usda.gov/download/treesearch/40.pdf .
  US Forest Service federal publication; independent KBDI equations.

Weather/reference data are explicit research inputs, never bundled defaults:
ERA5 CC-BY-4.0; GPCP NOAA PSL public domain (unless otherwise annotated);
CWFIS station archive licence not independently established by its README
(the linked daily-map metadata identifies OGL-Canada); and the public USFS
NFDRS4-TechDoc station-group CSV, whose repository contains no explicit data
licence. The latter is research-only with that uncertainty named in the
request/receipt. Neither reference archive is redistributed with this crate.

Native dependency notices remain in their vendored directories: netcrust,
netcdf-writer, bzip2, CSV, chrono, rayon, serde, serde_json, sha2
and cudarc (MIT and/or Apache-2.0). No new non-permissive library is added.

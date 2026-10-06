# prepbufr test fixture

`rap-t12z-fixture.bufr` (115,722 bytes, SHA-256
`6990bf368929cb1b3990bc5ff0a3586d3d4fe140938593ec35e8b7aeba9c1e22`) is cut
from NOAA's public rapid-refresh prepbufr for 2026-10-03 12Z:
`https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/rap.20261003/rap.t12z.prepbufr.tm00.nr`
(9,555,520 bytes, SHA-256
`0f6b607e7b578656f46c7968777e4ac969e1c13d82ae07785a35cef1beb4fa16`,
Last-Modified 2026-10-03 13:07:52 GMT).  NOAA observation data are a
work of the United States government and in the public domain.

The cut keeps whole messages, unchanged and in file order: the seven
dictionary messages at the head of the file (the last is the empty end
marker), then the shortest data message of each type present except the
satellite winds: ADPUPA (4 subsets), AIRCFT (29), PROFLR (2), ADPSFC (89),
SFCSHP (7), RASSDA (15) and ASCATW (5); for VADWND, the shortest message
with a report inside GSI's VAD time windows whose superobs pass their
marks (2 subsets, message at byte 4,097,536 of the source).

`rap-t12z-fixture.oracle.txt.gz` is the listing a Fortran program on
NCEPLIBS-bufr v12.3.0 (commit `328f2294082ad1fac27525859b8422882295671a`)
printed for this fixture, making GSI's own `ufbint` and `ufbevn` calls
(the header, observation, mark, error, drift and background lists, the
subtype and the program codes); `rw_obs::prepbufr::dump` must reproduce
it byte for byte.  The listing format is stated in
`rw_obs::prepbufr::dump`.  NCEPLIBS-bufr is a test oracle only; nothing
in this crate links it.

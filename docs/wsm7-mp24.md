# WSM7, mp_physics = 24

WSM7 carries qv, qc, qr, qi, qs, qg and qh. The WSM-family table chooses its
CUDA column kernel, reflectivity kernel, frozen constants, precipitating
classes and level tiers. Both default and GPUWM_WRF_EXACT=1 compile this
unit with --fmad=false --ftz=false through the no-FTZ NVRTC route.

The column acceptance command is:
GPUWM_WRF_EXACT=1 python -m tools.wsm7_wrf461_column_oracle.run_gpu check-fixture tests/data/oracles/wsm7

The reference comes from pinned public WRF v4.6.1 source. Build it with:
bash tools/wsm7_wrf461_column_oracle/build.sh /work/wrf/src/WRF-4.6.1 /work/port-mp24-wsm7/oracle-build

## Hail inventory

The source diff is generated with:
diff -u module_mp_wsm6.F module_mp_wsm7.F

The hail terms in module_mp_wsm7.F and their CUDA blocks are:

| Term | WRF line | CUDA line | Process |
|---|---:|---:|---|
| phaci | 1137 | 808 | Hail collects cloud ice |
| phacw | 1177 | 832 | Hail collects cloud water |
| phacr | 1251 | 883 | Hail collects rain |
| phacs | 1265 | 894 | Hail collects snow |
| phacg | 1277 | 903 | Hail collects graupel |
| phaci_w | 1318 | 929 | Wet-growth cloud-ice helper |
| phwet | 1314 | 934 | Wet-growth limit |
| pheml | 1353 | 954 | Hail melts |
| phdep | 1414 | 993 | Hail deposition and sublimation |
| phaut | 1460 | 1031 | Graupel converts to hail |
| phevp | 1490 | 1053 | Hail evaporates |

WRF initializes pvaph and primh to zero and never assigns a rate. The
kernel retains their zero contributions in the source and sink sums.
Hail also has its own slope and fall speed, fourth PLM sedimenting
species, HAILNC and HAILNCV, and frozen-fraction contribution.

## Declared divergences

D1 is default-on: the PLM remap search reaches the last arrival interface
and never leaves kt below kb. Stock WRF stops one interface short and can
drop arriving mass. The remap budget check includes a planted reversion.

D2 is default-on: reflectivity includes hail's dry-ice Rayleigh term using
WSM7's hail distribution. Stock WRF sums rain, snow and graupel only.
This is a diagnostic extension, with no change to prognostic fields.

The defined Fortran arm changes only D1 and D2. The stock, D1-only and
probe arms attribute differences. Default and strict kernels must match
every defined-arm output word at every step. A one-bit mutation of pacrh
must fail. Fixtures cover 72 columns at 45 levels and 36 at 72 levels,
with 30, 60, 150 and 300 second timesteps. This is a column implementation
check. It does not establish forecast skill or whole-model WRF parity.

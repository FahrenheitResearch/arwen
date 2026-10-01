# WRF v4.7.1 Noah mosaic oracle provenance

The `ucm` and `ucm_lcz` extension adds four steps each, 48 columns and three
tiles. The unchanged upstream routines initialize and integrate the urban
state. Surface-driver lines 3004-3016 are copied verbatim and hash checked.
The replay reads WRF's dumped table, globals and switches directly. The two
families measure 143 arrays each, 114,432 words total, at max ULP 0 with
raw-word equality and no held-out pairs. D1/D2 retain the original mosaic
port's corrected grid reduction contract. All remaining expected outputs
are the byte-unmodified WRF words. GROPTION is 0/1 for the two families;
CH_SCHEME=1, TS_SCHEME=1, AHOPTION=1, ALHOPTION=1, IMP_SCHEME=2 and
IRI_SCHEME=0. OASIS is 1.0. Night humidity keeps the green roof on positive
EPGR rather than WRF's undefined ETR read under dew. Exact source, raw-array,
ULP, frame, mutation and run receipts are under the harness's `receipts/`.

Upstream tag: `v4.7.1`.
Upstream commit: `f52c197ed39d12e087d02c50f412d90d418f6186`.
Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
C library: Ubuntu GLIBC 2.43-2ubuntu2.4, version 2.43.
Host: Ubuntu Linux x86-64, little endian.

All upstream sources were read-only and byte-unmodified. The SHA-256 list
below is copied from the supplied upstream pin list. The build verifies it
before compilation. The runtime oracle calls public WRF routines.

Build command (run on a Linux host with gfortran 15.2; the WRF tree is a
clone of the tag above, read only):

```sh
bash tools/noah_mosaic_wrf471_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR
```

Compiler flags:

```
-O0 -cpp -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4
-DDWORDSIZE=8 -DLWORDSIZE=4 -ffree-form -ffree-line-length-none
-fallow-argument-mismatch
```

No vector libm symbols in the -O0 objects. The positive control emitted
`_ZGVbN4v_expf`. Compiler, source/harness hashes and the full libm report are
in `tools/noah_mosaic_wrf471_oracle/receipts/`.

`mosaic_init` has 16 fixtures. `base`, `cats`, `lcz`, `twins`, and `ftz`
have 16, 8, 4, 3, and 1 step receipts respectively. Binary files and manifests
are about 5 MB total. Input and output arrays use WRF's Fortran index order.

The fixtures include independently-oracled per-tile increments on every step,
with unit area and zero cell accumulators. The initial increment expansion preserved all 7,968 prior runtime binaries.
Coverage was then extended with an actual retained water tile at a land point.
The complete binary/manifest corpus is 5,084,116 bytes before provenance.
Integrity hashes are in receipts/fixture-sha256sums.txt.

GPU qualification: CuPy 14.2.0, sm_89, -std=c++17 --fmad=false, unconditional
CuPy FTZ. Every non-held-out word matches: 72 output fields, 228,990 words,
max ULP 0. The 46 exact FTZ column/field pairs and GPU word hashes are pinned
in tests/test_noah_mosaic_wrf471_parity.py. D3's undefined/stale RC_mosaic is
not dumped or ported. D1's WRF control needs pre-step state for lower tiles
that have not yet updated; the corrected port uses final tile state.

Upstream source and table hashes:

```
79403e10104e23fc2a44eb7cf9c0a33c10fabed3b963c9ccb0ea0bfd29072b6d  phys/module_bep_bem_helper.F
5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062  share/module_model_constants.F
7bc761ee592feadbaeb974b4271b956044eb02c5a4f3bf0e77af4fbabf0de8eb  phys/module_sf_bem.F
c51ddd86871f81d5f132e63ddab2240e23886891356a7964debd4eafda9e500f  phys/module_sf_bep.F
42fe129dde0ccba24b84a64c1393e582204a7ced938568527beeb0c4f2a20b36  phys/module_sf_bep_bem.F
bde4ecc9a63c9a57c1ca40408eec1703881749fefd8698a7a1a9260cfaba06b1  phys/module_sf_noahdrv.F
034bf1d5f48b0b734702f200e3bcefe67db441b170cdcec8592992eb7d358e8c  phys/module_sf_noahlsm.F
f43c6482c17c5dff132921c0909ed5d706d5c7b0921902744eea17ef82a1d99f  phys/module_sf_noahlsm_glacial_only.F
623868c74c4b9d579e9c3811e9c334d731394c2afbfea7d693221626fbf0b0ea  phys/module_sf_urban.F
2ed7dc6e90e0fe442ffee84512b4998d31c8ec3400d3a7ab6078404065f784a6  frame/module_wrf_error.F
9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2  run/GENPARM.TBL
7f661318ff5f06aed6b4b5508e4d077dc794cff03d5fb4447fe59c88d0ed2ece  run/LANDUSE.TBL
1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f  run/SOILPARM.TBL
5811226b3db503ae02d8b35cb5cb10e0e90804e451f0f4a5b76a274ee64773d0  run/URBPARM.TBL
ab08e3f79d2f5d9d329aa2c953de4e7d94a71741baa0a50f1ecc145c6f81d9ab  run/URBPARM_LCZ.TBL
ed5478afbe49af51492256c1eb6cf88b3948590308525b32df1ddec28687b40e  run/VEGPARM.TBL
b6b2e9e006eff0b61f637a30e31a70f21b3c2bd8275182acb3c49aaf3fac0b4a  phys/module_ra_gfdleta.F
```

The raw writer output (one directory per case, one ``.bin`` per array) is
committed packed, one ``<case>.npz`` per case, by
``tools/noah_mosaic_wrf471_oracle/pack_fixtures.py``.  Each pack holds the
case's ``MANIFEST.txt`` bytes and every array's words unchanged; the packer
checks every ``.bin`` against ``receipts/fixture-sha256sums.txt`` first.
``gpuwm.verify.noah_mosaic_oracle.load`` reads either form.

The `ucm_wrfinit` and `ucm_lcz_wrfinit` families (2026-10-01) are `ucm` and
`ucm_lcz` with FRC_URB2D left as `urban_var_init` writes it: WRF's
dominant-urban rule, zero fraction in the urban-secondary cells. Same build,
same pins; the rebuild reproduced all 13,808 recorded raw arrays and 48
manifests of the earlier families word for word
(tools/noah_mosaic_wrf471_oracle/receipts/ucm-wrfinit-rebuild.txt).

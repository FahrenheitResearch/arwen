# RUC mosaic column oracles

The oracle compiles the unmodified WRF v4.6.1
[`phys/module_sf_ruclsm.F`](https://github.com/wrf-model/WRF/blob/v4.6.1/phys/module_sf_ruclsm.F).
Its SHA-256 is `3265f810d08dcbddfaf198371dc7f652e78e8d3a788f703a515c555a3bbb2a12`.
WRF is public domain. The notice is in `licenses/LICENSE-WRF-public-domain.txt`.
The port calls the existing shared glibc float32 log and exp transcriptions;
their Arm MIT notice remains in `glibc_flt32.cuh` and the repository NOTICE.

Run on a Linux CPU with gfortran and Python:

```sh
export CUDA_VISIBLE_DEVICES= GPUWM_NO_LOCAL_GPU=1
export TMPDIR="$PWD/scratch/tmp"
mkdir -p "$TMPDIR"
bash tools/ruc_mosaic_wrf461_oracle/build.sh /path/to/module_sf_ruclsm.F "$PWD/scratch/oracle"
cmp scratch/oracle/surface.csv gpuwm/data/ruc/oracle/mosaic_surface.csv
cmp scratch/oracle/driver.csv gpuwm/data/ruc/oracle/mosaic_driver.csv
python -m pytest tests/test_ruc_mosaic.py
```

`run_surface.F90` calls SOILVEGIN on 12 columns. It covers seasonal greenness,
prescribed LAI, incomplete and excess land area, water-soil exclusion,
dominant-soil fallback, water land use, and each independent mosaic switch.
The reference expects source category fractions, not reconstructed one-hot
dominant categories. Its effective roughness is the WRF 5 m blending-height
formula and is not divided by land area.

`generate.py` changes only the initial conditions and switches of the existing
full LSMRUC harness. It tests both vegetation tables on 12 warm columns for
two steps, recording every returned state and flux carrier. The columns cover
dry irrigation, wet soil, exactly 0.75 and lower greenness, absent crop cover,
incomplete and excess area, and water-soil fallback. It compiles with
`EM_CORE=0`: the mosaic and irrigation statements are common to both cores.
The CUDA runtime comparison separately exercises the ARW forcing path.
Warm soil and no snow avoid the upstream undefined thin-snow stack state.
This fixture does not remove the previously declared RUC snow/libm divergences.

The fixture was generated with GNU Fortran 13.3.0 on glibc 2.39 (box W1,
lane/verify-ruc-lsm), `-O0`, default REAL32, and the packaged WRF tables:
the toolchain every RUC WRF oracle is pinned on, whose float32 libm words
WOOF reproduces.  An earlier build with GNU Fortran 15.2.0 on another C
library recorded that library's words instead. Its byte counts and hashes are pinned in
`gpuwm.core.ruc_contract`. CPU and CUDA surface/column comparisons require
bitwise equality; the runtime test compares every device field for three
successive calls over mixed land, water, snow and ice columns.

Irrigation follows LSMRUC after SFCTMP. It adds water only in the table's root
layers when the greenness factor is greater than 0.75. WRF neither updates
liquid-water state nor adds an irrigation budget accumulator in this block;
the port preserves that behavior. WRF defaults `mosaic_lu=0` and
`mosaic_soil=0` remain unchanged.

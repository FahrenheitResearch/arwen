# Noah-MP column oracle fixtures

Produced by `tools/noahmp_wrf461_oracle/column` on box W1 (Ubuntu 24.04,
gfortran 13.3.0, glibc 2.39, x86-64 baseline, no FMA in the binary), CPU only.

WRF sources, unmodified, from WRF v4.6.1 (commit
d66e442fccc04111067e29274c9f9eaccc3cef28, submodule phys/noahmp at 848f54a):

| file | sha256 |
|---|---|
| phys/noahmp/src/module_sf_noahmplsm.F | bd592a5b7db29000e715250e3a7c779ffb5e0dcc356f6b5a7d9e1c9f69c55282 |
| phys/noahmp/drivers/wrf/module_sf_noahmpdrv.F | 9010a757da994ed8796c63ca97da354eaf60c5c732df4ea9acad5bc62a973890 |
| phys/noahmp/src/module_sf_noahmp_glacier.F | bf94f3522c3b9c2c9cfbb34fa7e485ff58519106db434520968793409a520579 |
| phys/noahmp/parameters/MPTABLE.TBL | 7fae6a77660c90ad80845565ecfb057093c100de41f35f25a7ffa63f41c19e5d |
| run/SOILPARM.TBL | 1e2275a32d8cd3b48ca693d22c0816df0013f83b6594ac632716361db337d58f |
| run/GENPARM.TBL | 9c02832a0e4a2ecaf47fcee485539aad95cd732c379c5c258161a88eb3d25ea2 |

Build: `build_column.sh <wrf> <dir> noopt` (-O0; no vector-math call in any
Noah-MP object).

| fixture (gzip -9 -n of) | uncompressed sha256 |
|---|---|
| in-summer60.bin | d7c27893a250c58b77b221c272cddb6a3132b16e48e8528ce720b219d7f4d84d |
| in-winter300.bin | d2951adc02291910bf5038fc108efd73c6d34590851c6225e2fed1048e9d0710 |
| wrf461-O0-summer60.bin | ab744240d8aa023663d4f086d6e49bd8a0dfb6a878b02965d35138bd00aa01fd |
| wrf461-O0-winter300.bin | f0fa1e9087355034331f4037a9e211890a66f321937e2869cc811ca2296162ce |

The input files regenerate byte-identically from `gen_columns.py` (checked on
the box's Linux numpy and the desktop's Windows numpy). `regimes-*.txt` names
each column's regime, vegetation and soil class.

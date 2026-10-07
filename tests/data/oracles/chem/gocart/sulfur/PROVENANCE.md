# sulfur WRF 4.7.1 oracle provenance

WRF commit: `f52c197ed39d12e087d02c50f412d90d418f6186`.
Compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
C library: Ubuntu glibc 2.43-2ubuntu2.4.
Host: x86-64 Linux. Reference uses scalar libm, SSE float32/float64, -O0, no FMA.
No WRF source bytes, stub declarations, or oracle_io.F90 bytes were changed.

Command, from a folder holding the pinned WRF sources as wrf-src/:

```
cp -r wrf-src/extra/* wrf-src/
bash tools/chem_wrf471_oracle/gocart/build_sulfur.sh wrf-src ../build ../fixtures
```

Shared build flags:
`-O0 -cpp -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -ffree-form -ffree-line-length-none -fallow-argument-mismatch`.
The build refuses `_ZGV*` in -O0 objects. Its -Ofast positive control calls `_ZGVbN4v_expf`.
The build writes its log, libmvec-report.txt, undefined-O0.txt and
undefined-control.txt into its build directory.

48 cases, three consecutive chemistry steps per case.
The driver is `tools/chem_wrf471_oracle/gocart/run_gocart_sulfur.F90`.
Arrays are REAL(4) or INTEGER(4), little endian, Fortran (i,k,j) or (i,j) order.
The top mass level is included and untouched. `oracle-sha256sums.txt` pins every
MANIFEST and binary array. Tests pin the sha256 of that checksum list too.

Compiled WRF source pins (all are checked by the unchanged shared build):

```
# sha256  path  (WRF v4.7.1, f52c197ed39d12e087d02c50f412d90d418f6186)
8694100875645b54a58bcff38d462ad3c9fd115261fb39941358466c84b62329  phys/module_data_gocart_dust.F
d7288531253596075ab0703448b164a083fb943802655adf924cc62feb078cf0  chem/module_data_gocart_seas.F
efc13a98a4899babee8cf420b23ffb9bc4de8049426bc78d85a7fb2ed3b2b3c2  chem/module_data_gocartchem.F
4e5d1c9cc2a23057f3a91631bd28626c960539387b9afe7c4aad19725b9d9e4b  chem/module_data_radm2.F
67ffc5b574ad6994e572890aefa3b9069843af5e4b08fd68583437baf4300709  chem/module_data_sorgam.F
1192e6228d7f8066a60138bda012c6e34956fd09d60bc46b802aec2d4f349987  chem/module_gocart_dust.F
71f0640f4a4634d3516c4b1c8f8dbc3c45c9d9e6aab7b92b203668c88071aabb  chem/module_gocart_dust_afwa.F
b697f7f38b1b3ffb5d4666a1233c3ffb32ed9003dc44edb6165197aeb1cd7046  chem/module_gocart_seasalt.F
28e8a7509e3ffa3782accbc912952eed45cdab60b1e9de8bbe230473106f0e0d  chem/module_gocart_settling.F
e8264b6cb243120dbe891b500cce68b47778a796ec54073ff434eb3c9fdd7935  chem/module_gocart_drydep.F
3f810ea786d3c8fe99b839e858b31f254b0a6c33ef6087be8ddf3765a3c81143  chem/module_gocart_aerosols.F
8d1e10e4df059e68b192510014b83ccb8edfd848625e5f8e7c882cd38c704441  chem/module_gocart_chem.F
5b80377fecdc18a5f0ad38d3b6c15cfc86ad5d76701adbbbb08a08698d0f7062  share/module_model_constants.F
```

The init path uses `sed -n '1765,1791p' wrf-src/chem/chemics_init.F`
into `chemics_init_solar.inc`, not a retyped implementation.
Full source sha256: `f8c37bd236d2fcaeb9361e8bde5865066dd3487e4105ca4e0aaaea69dd8147b4`.
Extract sha256: `c68194fd2e84cb0c5a2e9e1b533bdf0010fa94c19b161ecf88ff0a8e8ee2ca29`.
The range starts with ENDIF and omits the outer ENDDO; the harness supplies those
surroundings only. Original declarations are REAL for xtime/xhour, INTEGER for ixhour
(chemics_init.F:150-152).

Cases: dt 30, 36, 60, 7.5; julday 172/355/80; gmt 12/0/23.75;
eight latitudes/longitudes including polar night and midnight sun. Cloud fraction
is absent or supplied as 0/0.4, with qc and qi overrides. Temperatures include 250 K,
H2O2 is 1e-8 or 2e-10 mol/mol. Half the cases start with exactly zero DMS and MSA.
The third clock family crosses midnight over the consecutive steps.
`branch_input_*` labels are input-state classifications: 0 cold/top, 1 clear,
2 warm cloudy H2O2-limited, 3 warm cloudy H2O2 sufficient. They use incoming SO2,
not an instrumented internal SO2 value. Limiting and sufficient inputs are widely separated.

The solar-geometry residual (cos(SZA) and its daily sum, tcosz) was
attributed word for word during the port: link-time sinf/cosf/acosf wrappers
on the same -O0 WRF object recorded every trig input and glibc output word
(the instrumented fixtures equal these by their checksum list), and replaying
those glibc words through the CUDA arithmetic reproduced the Fortran. The
residual is the kernel's CUDA sinf/cosf/acosf stubs, which retire when the
engine's glibc float32 trig header lands. That instrumentation is not shipped.

GPU comparison, on a CUDA card:

```
python -m pytest tests/test_chem_ageing_wrf471_parity.py tests/test_chem_sulfur_wrf471_parity.py -q
```

Device: RTX 4090, sm_89. CuPy 14.0.1, NumPy 2.2.6, Python 3.13.15,
CUDA 13 NVRTC. The launchers use get_kernel, standard -std=c++17 and CuPy's
-ftz=true default. No fast math; all float and double arithmetic uses separate
round-to-nearest intrinsics. Float64 exp/log10/pow remain CUDA libm.
`ULP_BASELINE.json` is the measured exact per-case/per-output table, including
unchanged top levels.

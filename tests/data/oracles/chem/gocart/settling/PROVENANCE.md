# GOCART settling Fortran oracle provenance

WRF v4.7.1 commit: `f52c197ed39d12e087d02c50f412d90d418f6186`.
Sources are the exported `wrf-src/` bytes, sha256 checked before compilation.
No WRF source was edited. `stub_gocart.F90` and `oracle_io.F90` are unchanged.
No additional source was fetched.

Reference compiler: GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0.
Reference libc: Ubuntu GLIBC 2.43-2ubuntu2.4.
GPU: NVIDIA GeForce RTX 4090; CuPy 14; default NVRTC float32 FTZ.

Build, from a folder holding the pinned WRF sources as wrf-src/:

```sh
mkdir -p wrf-src/share
cp wrf-src/extra/share/module_model_constants.F wrf-src/share/
bash tools/chem_wrf471_oracle/gocart/build.sh wrf-src build gpuwm/data/chem/oracle/gocart > oracle-build.log 2>&1
```

Flags: `-O0 -cpp -Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4
-DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -ffree-form
-ffree-line-length-none -fallow-argument-mismatch`.
No `_ZGV*` reference occurs in an -O0 object. The positive control references
`_ZGVbN4v_expf`. The build writes its log and
libmvec-report.txt into its build directory.

For the aer_res input, `build.sh` extracts the exact lines 1520..1616 of
`chem/module_dep_simple.F` using
`sed -n '1520,1616p' "$src/chem/module_dep_simple.F" > "$build/sink_depvel.inc"`.
The harness includes it unchanged inside `sink_wesely_oracle`.
The harness's `scpr23` declaration feeds only the discarded gas velocities.
The extracted routine is called with `zr=2.0`, `z0=znt`, `ustar=ust`.
The 300 arm reads its locally zeroed rmol. The 401 arm skips Wesely,
matching `chem/dry_dep_driver.F:376-414`.

Pinned sources checked by the common build:

```text
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
cc6539387999e2d2e56c3038ae8cf1d865813658b77c822e4002d6477b438894  chem/module_dep_simple.F
```

Fixture writer format: unchanged `oracle_io.F90`, REAL(4)/INTEGER(4),
little endian, Fortran order. Settling row radius/density REAL*8 parameters
are recorded losslessly as two INTEGER(4) words per value.

GPU assertions, on a CUDA card:

```sh
python -m pytest tests/test_chem_settling_wrf471_parity.py tests/test_chem_drydep_wrf471_parity.py -q
```

`ULP.json` pins each output's measured table per case. Every measured table
has max_ulp 0 and n_nonzero 0. The two pytest files assert the full tables.

Coverage: 36 cases, 48 mass levels, all five dust and four sea-salt
bins, RH 5/10/30/80/95/99 percent, 7.5/36/60-second steps, 300/401 switches,
negative input reset, 0.05 m bottom layer reaching the 12-substep cap,
and three consecutive steps. The RH 95/99 columns both hit WRF's 0.95 bound.
The thin layer is a deliberate branch stress case. The reference itself
can produce negative values there; the port preserves them until WRF's
next entry reset. This is source parity, not a positivity qualification.

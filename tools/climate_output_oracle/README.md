# Climate-output native oracle

Authority: NOAA-EMC/HRRR `40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827`,
`sorc/hrrr_wrfarw.fd/WRFV3.9/phys/module_diag_cl.F` and its diagnostics driver.

Run `python tools/climate_output_oracle/run.py SOURCE_DIR OUTPUT_DIR` on the
Windows desktop with Ubuntu-24.04 WSL and gfortran. SOURCE_DIR holds the two
unmodified Fortran files. The script compiles the complete climate module,
with framework-only stubs, and derives its argument mapping from the driver.
Copy the generated vectors.npz and receipt.json to tests/data/oracles/climate_output
after reviewing the receipt. No arithmetic statements are rewritten.

106 calls compare all 48 fields, including unmodified reset timestamps,
strict ties, in-progress sums, both window closures, changed/nondivisible dt,
and the native restart flag. The GPU test compares binary32 words under the
repository strict-numerics test policy; default execution retains fast numerics.

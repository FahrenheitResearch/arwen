The oracle compiles the sun-angle albedo and surrounding sea-ice albedo
blocks from NOAA-EMC/HRRR v4.1.21, pinned to commit
`40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827` and full-file SHA256 hashes.
The WRF public-domain notice is in
`licenses/LICENSE-WRF-public-domain.txt`.

Run these commands on a node CPU with the lane's TMPDIR and one thread:

```sh
python build.py source build
python run_oracle.py build solar_albedo_fork.npz
```

The builder adds array bounds and C bindings. It extracts the original
arithmetic without substituting a handwritten formula. The fixture covers
all 21 MODIS classes, three radiation calls, first-call copies, retained
night/water/snow/ice values, the ALBSOL-only cap, and both sea-ice thresholds.
Its embedded receipt records the source hashes, extracted blocks, compiler
and flags. Replaying the fixture tests the host twin and CUDA kernel
against compiled Fortran words.

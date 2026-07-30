# RRTMGP cloud-optics reference outputs

The two `cloud-optics-reference-*.csv` files are raw stdout from the vendored
RTE+RRTMGP Fortran cloud-optics implementation at commit
`fa107a16120051c4124305c6b3d4c87059119f58`. They are reference outputs, not
values evaluated by gpuwm or reconstructed from the shipped lookup tables.

## Reference example and diagnostic run

The pinned checkout may live anywhere; its commit, compiler, libraries, and
commands are the authority rather than a developer's staging directory. On
Ubuntu 24.04, GNU Fortran 13.3.0 and netCDF-Fortran 4.5.4, the unmodified
all-sky example was built and run with the commands below after setting the
three absolute roots (line breaks added only for readability):

```sh
GPUWM_ROOT=/absolute/path/to/gpuwm
RRTMGP_ROOT=/absolute/path/to/rte-rrtmgp
ORACLE_ROOT=/absolute/path/to/cloud-optics-oracle
cd "$RRTMGP_ROOT"
cmake --build build-wsl --target rrtmgp_allsky -j2

cd "$ORACLE_ROOT"
"$RRTMGP_ROOT/build-wsl/examples/all-sky/rrtmgp_allsky" \
  4 30 1 allsky-lw.nc \
  "$GPUWM_ROOT/gpuwm/data/rrtmgp/rrtmgp-gas-lw-g256.nc" \
  "$GPUWM_ROOT/gpuwm/data/rrtmgp/rrtmgp-clouds-lw-bnd.nc"
"$RRTMGP_ROOT/build-wsl/examples/all-sky/rrtmgp_allsky" \
  4 30 1 allsky-sw.nc \
  "$GPUWM_ROOT/gpuwm/data/rrtmgp/rrtmgp-gas-sw-g224.nc" \
  "$GPUWM_ROOT/gpuwm/data/rrtmgp/rrtmgp-clouds-sw-bnd.nc"
```

The example NetCDF output contains cloud inputs and broadband fluxes but does
not expose the intermediate cloud `tau`, `ssa`, and `g` arrays. The vendored
`cloud-optics-reference-driver.F90` therefore executes the same reference
loader, medium-roughness selection, two-stream allocation, and
`cloud_optics()` call used at
`examples/all-sky/rrtmgp_allsky.F90:211-214,597-655,340-345`, then prints the
intermediate arrays. It was compiled and run exactly as follows:

```sh
GPUWM_ROOT=/absolute/path/to/gpuwm
RRTMGP_ROOT=/absolute/path/to/rte-rrtmgp
ORACLE_ROOT=/absolute/path/to/cloud-optics-oracle
cd "$RRTMGP_ROOT"
/usr/bin/f95 -O2 \
  -Ibuild-wsl/modules \
  -Ibuild-wsl/rrtmgp/data-loading-examples/modules \
  "$GPUWM_ROOT/gpuwm/data/rrtmgp/cloud-optics-reference-driver.F90" \
  -o "$ORACLE_ROOT/cloud_optics_oracle" \
  build-wsl/rrtmgp/data-loading-examples/librrtmgp-load-data.a \
  build-wsl/examples/shared-utils/libshared-utils.a \
  build-wsl/rte/frontend/librte.a \
  build-wsl/rrtmgp/frontend/librrtmgp.a \
  build-wsl/ssm/libssm.a \
  /usr/lib/x86_64-linux-gnu/libnetcdff.so \
  build-wsl/rte/frontend/librte.a

"$ORACLE_ROOT/cloud_optics_oracle" \
  lw "$GPUWM_ROOT/gpuwm/data/rrtmgp/rrtmgp-clouds-lw-bnd.nc" \
  > "$GPUWM_ROOT/gpuwm/data/rrtmgp/cloud-optics-reference-lw.csv"
"$ORACLE_ROOT/cloud_optics_oracle" \
  sw "$GPUWM_ROOT/gpuwm/data/rrtmgp/rrtmgp-clouds-sw-bnd.nc" \
  > "$GPUWM_ROOT/gpuwm/data/rrtmgp/cloud-optics-reference-sw.csv"
```

## Inputs and integrity

Both runs use ice roughness category 2 (medium), as selected by the reference
all-sky example. Each spot has one layer; sizes are micrometres and water
paths are g m-2.

| Column | Regime | LWP | IWP | liquid effective radius | ice effective diameter |
|---:|---|---:|---:|---:|---:|
| 0 | liquid | 12.0 | 0.0 | 7.125 | 50.0 |
| 1 | ice | 0.0 | 8.5 | 10.0 | 57.5 |
| 2 | mixed | 4.25 | 6.75 | 14.375 | 123.75 |
| 3 | clear | 0.0 | 0.0 | 10.0 | 50.0 |

| File | SHA256 | License |
|---|---|---|
| `cloud-optics-reference-driver.F90` | `3996197f0e712f4f0cb881d954140e63b527a2f4eb207d031d4ce3853b4906cc` | BSD-3-Clause (`LICENSE`) |
| `cloud-optics-reference-lw.csv` | `654ee18ac84d92d0471bc80862af389f85506882a730d18e6ed920e24b97043d` | BSD-3-Clause (`LICENSE`) |
| `cloud-optics-reference-sw.csv` | `3f065cba7546a783d3736e5ef17a42ebfa8e3c08c42f25ae529c4385a1603399` | BSD-3-Clause (`LICENSE`) |

Against these double-precision Fortran outputs, the measured FP32 device
maximum residuals were `2.5332e-7` (`tau`), `1.0442e-7` (`ssa`), and
`1.4095e-7` (`g`); the corresponding largest RMSE values were `6.3236e-8`,
`2.4933e-8`, and `3.3593e-8`. The committed gates allow measured headroom:
RMSE `2e-7/1e-7/1.2e-7` and maximum `8e-7/4e-7/5e-7` for `tau/ssa/g`.

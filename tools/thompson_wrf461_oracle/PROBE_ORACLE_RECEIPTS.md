# Per-kernel aerosol oracle receipts

Byte receipts for the Fortran output the three committed aerosol probes
produce, so a regeneration can be compared without diffing megabytes of CSV.
The CSVs themselves are **not** committed: they are 12 MB of derived data that
the committed sources reproduce exactly, and the point of this file is that
"exactly" is checkable.

Environment for every number below:

| item | value |
| --- | --- |
| WRF | v4.6.1, commit `d66e442fccc04111067e29274c9f9eaccc3cef28`, unmodified |
| compiler | GNU Fortran 13.3.0, `-O2 -ffree-form -ffree-line-length-none`, baseline x86-64 (no FMA instruction) |
| host | Ubuntu 24.04 on WSL2, x86-64 |
| date | 2026-07-31 |
| tables | the four classic assets pinned by `thompson_contract.py`, plus `CCN_ACTIVATE.BIN` at `f2b8d391…` |

## `build_aero_probes.sh` — `probe-oracle-aero/`

| file | rows | sha256 |
| --- | --- | --- |
| `aero-warm-rates.csv` | 12348 | `7477a384cb7c8e4710c1f2bf13b7e27fa0428c76a683dd80bd25388e40968cab` |
| `aero-ncten-balance.csv` | 11025 | `40a41d6a296f90bf587991223ad130c6af85c1bcc5fdaf4871b740333fdad225` |
| `aero-cold-warm-loop.csv` | 11340 | `fff44927dab5e89aef7b5b879a10875741f2492212c587a2df619c1c9462c3a2` |

The first two digests are also the digests of the pre-existing scratch output
that produced `_WARM_RATE_ORACLE` and `_NCTEN_BALANCE_ORACLE`, which is what
makes `probe_warm_rates_aero.F90` a *recovery* rather than a reconstruction.
`aero-cold-warm-loop.csv` has no such predecessor — its driver was lost — so
its digest pins this repository's reconstruction, not the original run.  See
the header of `probe_cold_warm_loop_aero.F90`.

## `build_aero_instrumented.sh` — `intermediates/`

Only the files the committed tests consume are listed; the script emits one
per scenario per anchor.

| file | sha256 |
| --- | --- |
| `aero-ice-demott-idxin-cold-network.csv` | `b2056c1161ed5f09e64f232338a3ec2381dcdcf8989650d36396475f09ee9ea6` |
| `aero-cloud-freeze-nc-cold-network.csv` | `a23b675faa712786c321eb2c4d13beb32254b9102f08b550ca22552ae41d2679` |
| `aero-cold-overlap-cold-network.csv` | `97637bc2e44103b5a91c0aab47dae81b96de66562391750a20e00b22cdab0014` |
| `aero-nc-sed-cloud-sed.csv` | `c4705dfc5503424a9a13c88a67c6fa471a3027dcaf228de9dc571fe4c3d9c16d` |
| `aero-reduces-to-classic-phase-cleanup.csv` | `6cd53f3a42137953c8b4a2e8b4cd906d1872d7127e59e07b8beb539169f437cb` |

A digest mismatch here is not automatically a defect — a different gfortran
version legitimately moves the last digits — but it does mean the two
verification scripts must be re-run and believed instead of these receipts:

    python3 check_probe_oracles_aero.py       <build>/probe-oracle-aero
    python3 check_instrumented_tables_aero.py <build>/intermediates

Both compare against the literals the tests actually assert, so they are the
authority; this file is a shortcut, not a substitute.

## Verification status, 2026-07-31

| table | where | result |
| --- | --- | --- |
| `_WARM_RATE_ORACLE` | `test_thompson_aerosol_warm_gpu.py` | 124/124 rows x 26 fields reproduced |
| `_NCTEN_BALANCE_ORACLE` | same | 68/68 rows x 8 fields reproduced |
| `_WRF_COLD_WARM_LOOP` | `test_thompson_aerosol_cold_gpu.py` | 54/54 rows x 20 fields reproduced |
| `_WRF_COLD_REFERENCE` | same | 360/360 values reproduced |
| `SED_AERO_NC_SED` | `test_thompson_aerosol_sed_gpu.py` | 384/384 values reproduced bitwise |
| `CLEAN_CLASSIC` | same | 408/408 values reproduced bitwise |
| `SED_NU_SWEEP`, `CLEAN_MELT`, `CLEAN_FREEZE` | same | **not verified** — need three scratch scenarios that are not in `run_column_aero.F90`; see PROVENANCE.md |

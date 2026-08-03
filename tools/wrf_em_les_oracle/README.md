# WRF v4.6.1 `em_les` oracle — reproduction

Everything needed to rebuild the oracle and regenerate its receipts. Nothing
here imports gpuwm; the oracle side has to be able to disagree with the engine
it scores.

## Build

```sh
curl -sL -o v4.6.1.tar.gz \
  https://github.com/wrf-model/WRF/releases/download/v4.6.1/v4.6.1.tar.gz
# expect sha256 b8ec11b240a3cf1274b2bd609700191c6ec84628e4c991d3ab562ce9dc50b5f2
tar xzf v4.6.1.tar.gz && cd WRFV4.6.1

source /opt/intel/oneapi/setvars.sh
export NETCDF=<prefix with include/netcdf.inc and lib/libnetcdff.so>
export WRF_EM_CORE=1 WRF_NMM_CORE=0 WRF_DA_CORE=0

printf '34\n1\n' | ./configure          # 34 = GNU gfortran/gcc dmpar
sed -i 's/-w -O3 -c/-w -O3 -std=gnu17 -c/' configure.wrf   # GCC-15 K&R fix
./compile em_les -j 24                  # ~11 min, ~600 MB tree
```

On Ubuntu the system netCDF lives in `/usr/include` + `/usr/lib/<triplet>`,
which is not the `$NETCDF/lib` layout `configure` expects. Build a shim prefix
of symlinks rather than a second netCDF.

**Do not rebuild the pinned campaign build.** It is identified by its binary
hashes, not by a path: `wrf.exe` `bda5e5e6…`, `real.exe` `a300ada0…`.
`PINNED-BUILD-BASELINE-sha256.txt` (in the receipts directory) is the manifest
to diff against before and after any work on these nodes.

## Run

```sh
python3 gen_namelist.py .               # writes namelist.* + run_matrix.json
./run_les.sh match_km3_100m ./namelist.match_km3_100m 24
```

`run_les.sh` stages the WRF `run/` directory with **`cp -L`**. That is not
cosmetic: `run/MPTABLE.TBL` is a relative symlink, and a plain `cp` leaves a
dangling link that makes Noah-MP stop inside `read_mp_veg_parameters` with an
error that reads like a physics failure and is not one.

`iofields_les.txt` adds `tke, xkmv, xkmh, xkhv, xkhh` to history stream 0.
Without it they are restart-only and every SGS quantity is silently
unavailable after the run finishes. WRF writes them **uppercase** even though
the iofields file names them lowercase.

## Score

```sh
python3 score_wrf_les.py <run_dir> <out_prefix> --window-min 30
python3 verify_independent.py <run_dir> <out_prefix>.json   # second opinion
python3 same_instrument.py arwen_kmN.npz <out_prefix>_profiles.npz <label>
python3 spec_compare.py arwen_kmN.npz wrf_wslab.npy <dx> <outdir>
python3 window_spread.py <run_dir> <out.json> --win-min 15 --spinup-min 60
```

`same_instrument.py` is the one that matters for the comparison: it reduces
**both** models' raw per-minute profiles with a single routine, so a
difference in the output is a difference in the models rather than in two
codebases' conventions. It validates on the way in by reproducing ArWen's own
published fit from ArWen's own arrays.

`INSTRUMENT-HISTORY.md` records which parts of the scorer were changed after
output had been seen, and why. No band is cut anywhere in this lane.

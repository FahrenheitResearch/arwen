# WRF v4.7.1 ndown oracle (rebalance and blend_terrain)

Column oracle for the port of WRF's offline one-way nesting step
(`main/ndown_em.F`) that puts a downscaled child on its own terrain:
`gpuwm/offline_child_geography.py` (`rebalance_to_child_terrain`,
`blend_child_terrain`).  The reference is WRF's own Fortran, byte-unmodified,
compiled by gfortran on the oracle host.  The fixture of record is
`tests/data/ndown_wrf471.npz`, graded by `tests/test_offline_child_geography.py`.

## Files

| file | what it is |
|---|---|
| `SOURCES.sha256` | sha256 of every WRF v4.7.1 file used or cited (tag v4.7.1, commit f52c197ed39d12e087d02c50f412d90d418f6186) |
| `fetch_sources.sh DEST` | fetches those files from github.com/wrf-model/WRF at the tag and refuses any that differ from its pin |
| `make_cases.py write DIR` | writes the cases: 12 rebalance (two ladders x two hypsometric forms x coast, canyon and high terrain) and 3 blend |
| `make_cases.py collect DIR NPZ` | reads every case and the oracle's output into the fixture |
| `stub_wrf.F90` | the framework names the two routines reference and that carry no number they compute with (see its header) |
| `rebalance_host.F90`, `blend_host.F90` | the scopes the two cut routines sit in |
| `dummy_new_args.inc`, `dummy_new_decl.inc` | stand-ins for the Registry's generated argument list: the one 4-D array `rebalance` reads, `moist` |
| `run_ndown.F90` | the driver: one case file in, one output file out |
| `build.sh WRF_SRC BUILD CASES` | checks every pin, cuts the two routines out of their files by line range (checked to start at the SUBROUTINE line and end at its END SUBROUTINE), builds two variants, runs every case |

## Build variants

* **pristine**: the reference.  gfortran -O0, WRF's preprocessor defines for an
  EM RWORDSIZE=4 build.  Its outputs are the fixture of record.
* **snan**: the reference with every local real initialised to a signalling NaN
  and every local integer to -999999.  Required byte-identical on every case: no
  uninitialised local reaches an output.

## How the port is graded

The port computes in float64, the engine's preparation arithmetic (the analytic
base state every real-data run is built on); WRF computes in REAL (float32).  The
test feeds the port the exact float32 inputs and coefficients the oracle read and
holds each output to a few float32 epsilons of the total it is part of: base and
perturbation pressure of the total pressure, theta of the total potential
temperature, inverse density of itself, geopotential of the column-top
geopotential.  Largest measured distances (in those epsilons): pb 4.2, t_init 1.5,
alb 4.1, phb 4.5, mub 3.9, t_2 2.1, p 0.03, alt 4.7, ph_2 4.6, psfc 4.5, against
bounds of 4 (theta) and 8 (the rest); in physical units, geopotential within 1.1 cm
and the pressure perturbation within 0.0003 Pa.  Four single changes of the method
(the two terrains swapped, a dry column, the other hypsometric form, no theta
shift) land 250 to 57,000 times outside the bounds.  The blend is a copy, bit for bit, outside the blend band, and within
WRF's own float32 rounding inside it.

## Running

    bash fetch_sources.sh ~/wrf471
    python make_cases.py write ~/ndown-cases
    bash build.sh ~/wrf471 ~/ndown-build ~/ndown-cases
    python make_cases.py collect ~/ndown-cases tests/data/ndown_wrf471.npz

`make_cases.py` imports gpuwm (for the engine's vertical coordinate), so run it
in an environment with the tree on the path.  Measured on gfortran 15.2.0 and
glibc 2.43.

# Third-party notices

This project is dual-licensed under MIT or Apache-2.0, at your option, with one
addition that cannot be removed: the solver is a derivative of Py-ART, so
BSD-3-Clause terms apply to it as well. The SPDX expression for the whole
project is therefore `(MIT OR Apache-2.0) AND BSD-3-Clause`.

## Py-ART (Python ARM Radar Toolkit)

- **License:** BSD-3-Clause, with a U.S. DOE / Argonne government-rights notice
  prepended
- **Copyright:** © 2013, UChicago Argonne, LLC. All rights reserved.
- **Source:** <https://github.com/ARM-DOE/pyart>

`src/solver.rs` is a Rust port of Py-ART's `dealias_region_based` per-sweep
core (`pyart/correct/region_dealias.py`), mirroring its interval splitting,
region finding, edge accounting, and dynamic network reduction. The complete
verbatim upstream notice ships beside it as
[`PYART-LICENSE.txt`](PYART-LICENSE.txt); the excerpt below does not replace
it. Redistributing this project — in source form or as the compiled
`dealias.wasm` — means shipping that file with it.

> Copyright 2013 UChicago Argonne, LLC. This software was produced under U.S.
> Government contract DE-AC02-06CH11357 for Argonne National Laboratory (ANL),
> which is operated by UChicago Argonne, LLC for the U.S. Department of Energy.
> The U.S. Government has rights to use, reproduce, and distribute this
> software.  NEITHER THE GOVERNMENT NOR UCHICAGO ARGONNE, LLC MAKES ANY
> WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
> SOFTWARE.  If software is modified to produce derivative works, such modified
> software should be clearly marked, so as not to confuse it with the version
> available from ANL.

### This is modified software

Per the notice above, the changes are stated plainly. This is **not** the
Py-ART distributed by ANL:

- It is a Rust translation of a single solver, not the toolkit.
- It works on flat `f32` arrays rather than Py-ART's radar objects, and it has
  no file-format, gridding, or plotting code.
- `MAX_EXTRA_INTERVALS` bounds the number of extra velocity intervals added for
  data outside ±Nyquist. Python's arbitrary-precision integers do not need such
  a bound; `i32` does, and without it a corrupt feed carrying non-physical
  velocities overflows the interval count.
- The merge loop selects the strongest edge with a lazy-deletion max-heap, and
  tracks shared neighbours with an epoch counter, instead of rescanning the
  whole edge list and clearing an array on every merge. Both were quadratic in
  the region count. Which edge is selected is unchanged — verified
  edge-for-edge against the original implementation on 86 real sweeps.

Neither UChicago Argonne, LLC, Argonne National Laboratory, the U.S.
Government, nor the Py-ART contributors endorse or promote this port.

### Citation

Please cite the Py-ART paper when citing this functionality:

> Helmus, J. J., and S. M. Collis, 2016: The Python ARM Radar Toolkit (Py-ART),
> a Library for Working with Weather Radar Data in the Python Programming
> Language. *J. Open Res. Softw.*, **4**(1), p.e25,
> <https://doi.org/10.5334/jors.119>.

## Method lineage

The region-based approach these engines share is described in:

- Jing, Z., and G. Wiener, 1993: Two-dimensional dealiasing of Doppler
  velocities. *J. Atmos. Oceanic Technol.*, **10**, 798–808.
- Besag, J., 1986: On the statistical analysis of dirty pictures.
  *J. R. Stat. Soc. B*, **48**, 259–302.
- Feldmann, M., C. N. James, M. Boscacci, D. Leuenberger, M. Gabella,
  U. Germann, D. Wolfensberger, and A. Berne, 2020: R2D2: A region-based
  recursive Doppler dealiasing algorithm for operational weather radar.
  *J. Atmos. Oceanic Technol.*, **37**, 2341–2356.

# Third-party notices

`hfutils` is licensed under **Apache-2.0** (© Lynker Spatial).
That license applies to Lynker Spatial's own code in this repository.

This package **depends on** third-party R packages that carry their own
licenses. Those packages are **not redistributed** with this source; they are
installed separately from CRAN (or their upstream source) by the end user, so
their licenses govern the user's use of those packages, independent of the
Apache-2.0 grant here. Several are copyleft (GPL/LGPL); using `hfutils` requires
installing them and complying with their terms.

This notice is provided for transparency and to answer reuse questions; it adds
no constraint to the Apache-2.0 grant on Lynker Spatial's code.

## Direct dependencies

| License family | Packages |
|---|---|
| **GPL-2 / GPL-3 (≥)** | `igraph` (GPL ≥2), `terra` (GPL ≥3), `lwgeom` (GPL-2), `units` (GPL-2) |
| **GPL-2 \| MIT (dual)** | `sf`, may be used under MIT |
| **LGPL (≥2.1)** | `DBI`, `RSQLite` |
| **MIT** | `cli`, `dbplyr`, `dplyr`, `duckdb`, `glue`, `httr2`, `jsonlite`, `rlang`, `rmapshaper`, `yyjsonr` |
| **Apache-2.0** | `arrow` |
| **Part of R (base)** | `methods`, `utils` |

## Notes for redistributors

- The copyleft licenses above (GPL/LGPL) impose obligations on **distribution of
  those packages**. Distributing `hfutils`'s own source does not redistribute
  them. If you bundle these dependencies into a combined distribution (e.g. a
  container image or a vendored archive), comply with each package's license for
  that distribution.
- `sf` offers a permissive option (MIT).
- This list reflects direct dependencies. Transitive dependencies carry their
  own licenses; standard CRAN tooling resolves them at install time.

_Generated from the package `DESCRIPTION` dependency sets; regenerate when
dependencies change._

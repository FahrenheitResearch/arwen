# Migrating a WPS case to RW-WPS

RW-WPS keeps `namelist.wps`, `namelist.input`, and `WPS_GEOG` as authorities,
but replaces the normal `geogrid`/`ungrib`/`metgrid`/`real.exe` execution
chain. Migration should begin as an independent output directory; do not
replace a working WPS case in place.

1. Run `rw-wps --namelist-support-report` on the unchanged namelist pair.
   Resolve every preprocessing/physics issue explicitly. Runtime-output-only
   WRF settings remain visible but do not become preprocessing controls.
2. Select a named source only when the exact product/cadence/level contract
   matches. Otherwise author a mapped descriptor, Vtable-derived selector
   set, composition, and hash manifest.
3. Point RW-WPS at the existing `WPS_GEOG` tree. Static fields are built
   natively and receive their own geometry/provenance receipt.
4. Run with `--dry-run`, then publish to a new output root. Preserve every
   receipt and SHA-256 manifest.
5. Compare RW-WPS `wrfinput_d0*`/`wrfbdy_d01` against the existing WPS/real
   inventory. Numerical identity is not promised: interpolation and masked
   donor policies are explicit and may differ while remaining structurally
   valid.
6. Launch an unchanged stock WRF executable against the new directory for a
   short finite-state gate before any production forecast. Record its exact
   executable hash and output evidence.

RW-WPS does not consume `met_em*` in its normal native route and does not
silently translate unsupported namelist settings. `ungrib`/`metgrid` table
controls may remain in `namelist.wps` for provenance, but source mapping and
composition documents replace their runtime role.

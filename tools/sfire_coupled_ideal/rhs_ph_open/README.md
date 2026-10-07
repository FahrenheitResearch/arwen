# Open-boundary geopotential oracle

The unchanged `tools/bigstep_wrf471_oracle/prep_build.py` source-pinned
extraction supplies the original WRF v4.7.1 `rhs_ph` body, compiled
without modification. The wrapper uses the compiled WRF configuration type
and gravity constant.

WRF evaluates the west top row at `kz=kde`. Every WRF initializer assigns
`fnm` and `fnp` only for `k=2..kde-1`, and the top `U` level is never
written, so at run time all three hold their zero allocation value: the
donor speed is zero and the term subtracts a signed zero. The corpus feeds
`fnm(kde)=fnp(kde)=0`, as WRF holds them. Control 1 holds the top `U`
level at zero; control 2 sets it to a -17 sentinel, and `grade.py` requires
every native word to be independent of it.

The corpus covers second- and fifth-order horizontal advection, four
periodic/open x/y combinations, vertical terms on and off, nonuniform
terrain, spatially varying isotropic map factors, mixed inflow/outflow,
every physical interface and corner, and the two top-level controls.
`ph_old` aliases `ph` at the native call. `grade.py` invokes the production
`_launch_open_geopotential` path with native coupled face masses and
compares every output word with no allowed-difference mask. Default and
strict processes grade separately.

Commands:

```
build.sh WRF_SOURCE CONFIGURE_MOD NATIVE_FIRE_ORACLE_BUILD NEW_BUILD
python -m tools.sfire_coupled_ideal.rhs_ph_open.pack NEW_BUILD CORPUS
python -m tools.sfire_coupled_ideal.rhs_ph_open.grade CORPUS RECEIPT
GPUWM_WRF_EXACT=1 GPUWM_WRF_EXACT_BIGSTEP=1 python -m tools.sfire_coupled_ideal.rhs_ph_open.grade CORPUS STRICT_RECEIPT
```

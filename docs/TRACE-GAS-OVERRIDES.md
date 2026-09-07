# Declared radiation gases

`case_data.co2_vmr` keeps the requested positive CO₂ mole fraction through
classic RRTM, legacy RRTMG and RRTMGP radiation. The shared adapter API also
accepts a `trace_gas_overrides` mapping for implemented scalar gas operands:

| Absorption engine | Operands |
| --- | --- |
| Classic RRTM longwave | CO₂, N₂O, CH₄ |
| Legacy RRTMG longwave | CO₂, N₂O, CH₄, O₂, CFC11, CFC12, CFC22, CCl₄ |
| Legacy RRTMG shortwave | CO₂, CH₄, O₂; N₂O is a retained, inactive WRF interface operand |
| RRTMGP | The selected packaged coefficient tables' gas inventory |

Mapping keys use lowercase names (`co2`, `cfc11`, etc.). Values must be
finite positive mole fractions no greater than one, representable by the
FP32 solvers. Ordinary oxygen concentrations are valid. This representation
check does not establish coefficient accuracy outside a table's training
range. The public high-CO₂ advisory remains in place; zero retains the existing
unsupported explicit-override contract.

Only declared gases replace defaults. Classic and legacy radiation retain
their existing WRF year formulas and rounding; RRTMGP retains its existing
date policy and experiment-zero defaults. The declared scalar fills model
and above-model gas layers through the existing preparation routines.
`trace_gas_override_consumption` identifies the active spectra per gas;
its empty tuple means inactive. N₂O remains active when longwave is selected.
Selected spectra consume only their implemented operands. Unknown gases or
gases absent from all selected absorption engines receive a named error;
off, analytic and Dudhia-only radiation report valid declarations as inactive.

Overrides are owned constructor inputs, preserved by tile reconstruction
and bound into restart setup identity. A different concentration cannot
resume a checkpoint under the old setup. This adds no CAM greenhouse-gas
file reader and changes no solver, ozone policy or default concentration.

# Source-dependent release leg

Run `run_source_leg.py` inside an authorized GPU queue. Supply the actual
WRF 4.6.1 tree, the WRF 4.8.0 MMM `bl_ysu.F90` source, the regenerated YSU
step-zero fixture, and the coupled cloud/radiation fixture. The two WRF trees
must be siblings named `v4.6.1` and `v4.8.0`, as required by the source test.

```sh
python tools/battery/run_source_leg.py \
  --wrf461 "$SOURCES/v4.6.1" \
  --wrf480-source "$SOURCES/v4.8.0/phys/physics_mmm/bl_ysu.F90" \
  --ysu-fixture "$YSU_BUILD/wrf_480-step0" \
  --coupled-fixture "$CLOUD_BUILD" --out "$EVIDENCE/source-leg"
```

The YSU tools are `tools/ysu_generations_oracle/fetch_sources.sh` (the four
pinned public releases, each with the MMM physics it names),
`build.sh`, `make_columns.py` and `gate.py`; the cloud fixture builder is
`tools/cldfra2_wrf461_oracle/coupled_build.sh`. Their source hashes and
build receipts remain required evidence. Supply their outputs explicitly;
this leg never substitutes a Python reference or downloads a different source.

The runner executes all five named suites in default and strict modes without
a marker filter, writes per-arm logs and JUnit reports, and refuses any skip
or empty collection. `source-leg.json` records the source paths, commands and
results. A help invocation or an absent-input refusal is not a passing leg.

Outside this leg the YSU generation tests do not skip: they replay the
committed WRF references in `tests/data/oracles/ysu_generations` on every
GPU leg, in that leg's arithmetic mode. Here, with `YSU_COLUMN_FIXTURE` set,
`gate.py --fixture verify` rebuilds every reference from the oracles and
also requires the committed words, so a reference that can no longer be
rebuilt from the public sources fails the leg.

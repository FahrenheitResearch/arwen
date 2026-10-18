# YSU generations: unmodified WRF references

Every word unmodified WRF `bl_ysu_run` wrote on the 1,188-column corpus of
`tools/ysu_generations_oracle/make_columns.py` (seed 20261008), three steps
per release, xz-compressed. `tests/test_ysu_generation_batch_gpu.py` grades
the YSU generations against them on every GPU leg, in that leg's arithmetic
mode, with no Fortran toolchain.

| directory | release | source |
|---|---|---|
| `wrf_480/` | WRF v4.8.0 `06d4240a`, MMM physics `550b5b49` | `bl_ysu.F90` sha256 `15d76892...11ea7e` |
| `wrf_461/` | WRF v4.6.1 `d66e442f` | `bl_ysu.F90` sha256 `a695daa7...3d77d8` |

Each `stepN/` holds:

- `ysu-wide-wrf.bin.xz`: the reference. Four-level columns carry the
  declared bounded reference instead (the unmodified `get_pblh` reads past
  the top of a 4-level column; see `shallow/`), and every other record is
  checked unchanged by the traced build.
- `declared-reference.bin.xz`: the same release with the declared BEP drag
  correction applied, which mixed urban columns are graded against.
- `trace.txt.xz`: the top-down diagnostics (`tmp1`, `ent_eff`, radiative sum,
  `wstar3_2`, cloud flag) the traced build printed, word for word.
- `inputs.sha256`: the step's input records. Step 0 reads `columns.bin.xz`;
  steps 1 and 2 are rebuilt from the previous step's reference, and replay
  refuses inputs whose hash differs.

`shallow/` is the four-level probe: its input, the bounded reference, and the
checked build's rejection of the unmodified source (`qke1d` index 5 above its
bound of 4).

## Rebuilding

From the repository root, with `PYTHONPATH=.` and a CUDA device for the gate:

```sh
bash tools/ysu_generations_oracle/fetch_sources.sh SRC
bash tools/ysu_generations_oracle/build.sh SRC BUILD v4.6.1 v4.8.0
python tools/ysu_generations_oracle/make_columns.py BUILD/columns.bin
python -m tools.ysu_generations_oracle.gate --root BUILD --version wrf_480 --out OUT --fixture verify
python -m tools.ysu_generations_oracle.gate --root BUILD --version wrf_461 --out OUT --fixture verify
python tools/ysu_generations_oracle/check_shallow.py BUILD OUT/shallow.json --fixture verify
```

`--fixture record` writes this directory; `verify` requires its bytes. Built
with GNU Fortran 13.3.0 (glibc 2.39, x86_64), `-O2 -fno-tree-vectorize
-ffp-contract=off`, `kind_phys = 4`. Two independent builds from separate public
clones wrote the same input, reference and declared-reference bytes at every
step of both releases.

# Kessler refute harness

New columns for the WRF v4.6.1 Kessler oracle, graded the same way as
`tools/kessler_wrf461_oracle` (same `run_kessler_O0`, same production
launcher, bitwise compare).

    python make_refute_columns.py SETS          # setA..setE
    python make_refute_columns.py SETS setF     # high-top + libm witnesses
    for s in SETS/set*; do (cd $s && ORACLE/run_kessler_O0 \
        kessler-inputs.bin kessler-wrf-O0.bin); done
    # the same reference under glibc's generic powf/expf:
    GLIBC_TUNABLES=glibc.cpu.hwcaps=-AVX2,-FMA ORACLE/run_kessler_O0 ...
    GPUWM_WRF_EXACT=1 python compare_refute.py SETS setA setB ... --json OUT
    python make_fixture.py SETS tests/data/oracles/kessler

`libm_sweep.py` compares WOOF's own `gfk_pow` (for Kessler's five constant
exponents) and `gfk_exp` with the box's C-library `powf`/`expf` on every
float32 input; `libref.c` is the scalar C side (built
`gcc -O1 -fno-builtin -fno-tree-vectorize -fopenmp -shared -fPIC`), and
`probe.c` prints the C library's words for the witness inputs. `coverage.py`
reports what the sets reach (es >= p cells, Exner above one, split counts).

The pytest gate is `tests/test_kessler_refute_columns.py`; the fixtures and
their provenance are in `tests/data/oracles/kessler/` (`REFUTE-PROVENANCE.md`).

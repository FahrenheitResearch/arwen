When porting logic to rust, use the C implementations in the `c-blosc` and `c-blosc2` subdirectories.

## Porting from C to Rust

When porting the implementation, try to reuse variable names and function names when possible, and use rust idioms otherwise.

The constants defined in `src/internal/constants.rs` are correct and verified against the reference implementation. Use these constants in your code.

## Note taking

Leave notes about the logic of BLOSC compression and de-compression in the `BLOSC_NOTES.md` file in for future reference. Also leave notes when identifying critical lines of code in the reference implementations, with file names, function names, and corresponding line number ranges.

Leave comments in the Rust code reflecting the logic used in the C implementation for future reference.


## Unit testing

Be sure to run unit tests in single-threaded mode.

```sh
cargo test -- --test-threads=1
```

ALWAYS RUN ALL TESTS TO PREVENT REGRESSIONS. DO NOT FILTER TO INDIVIDUAL TESTS.


## Debugging

Debug by printing and comparing the return values from blosc2_src (correct) and blusc (incorrect, needs fixing) crates.

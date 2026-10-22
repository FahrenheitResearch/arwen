# unsafe_cell_slice

[![Latest Version](https://img.shields.io/crates/v/unsafe_cell_slice.svg)](https://crates.io/crates/unsafe_cell_slice)
[![unsafe_cell_slice documentation](https://docs.rs/unsafe_cell_slice/badge.svg)](https://docs.rs/unsafe_cell_slice)
![msrv](https://img.shields.io/crates/msrv/unsafe_cell_slice)
[![build](https://github.com/LDeakin/unsafe_cell_slice/actions/workflows/ci.yml/badge.svg)](https://github.com/LDeakin/unsafe_cell_slice/actions/workflows/ci.yml)

A Rust microlibrary for creating multiple mutable references to a `slice`.

### Motivation
The rust borrow checker forbids creating multiple mutable references of a `slice`.
For example, this fails to compile with ```cannot borrow `data` as mutable more than once at a time```:
```rust
let mut data = vec![0u8; 2];
let data_a: &mut u8 = &mut data[0];
let data_b: &mut u8 = &mut data[1];
*data_a = 0;
*data_b = 1;
```

There are use cases for acquiring multiple mutable references of a `slice`, such as for writing independent elements in parallel.
A safe approach is to borrow non-overlapping slices via `slice::split_at_mut`, `slice::chunks_mut`, etc.
However, such approaches may not be applicable in complicated use cases, such as writing to interleaved elements.

### `UnsafeCellSlice`
An `UnsafeCellSlice` can be created from a mutable slice or the spare capacity in a `Vec`.
It has unsafe `get_mut` and `index_mut` methods that permit creating multiple mutable references of subslices or elements.

```rust
let mut data = vec![0u8; 2];
{
    let data = UnsafeCellSlice::new(&mut data);
    let data_a: &mut u8 = unsafe { data.index_mut(0) };
    let data_b: &mut u8 = unsafe { data.index_mut(1) };
    *data_a = 0;
    *data_b = 1;
}
assert_eq!(data[0], 0);
assert_eq!(data[1], 1);
```

Note that this is very unsafe and bypasses Rust's safety guarantees!
It is the responsibility of the caller of `UnsafeCellSlice` methods to avoid data races and undefined behavior by not requesting overlapping subslices/elements.

Under the hood, `UnsafeCellSlice` is a reference to a `std::cell::UnsafeCell` slice, hence the name of the crate.

## Licence
`unsafe_cell_slice` is licensed under either of
 - the Apache License, Version 2.0 [LICENSE-APACHE](./LICENCE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0> or
 - the MIT license [LICENSE-MIT](./LICENCE-MIT) or <http://opensource.org/licenses/MIT>, at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.

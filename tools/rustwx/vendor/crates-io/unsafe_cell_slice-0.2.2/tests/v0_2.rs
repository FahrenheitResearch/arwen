#[path = "common.rs"]
mod common;
use common::{decr, incr, N_REPETITIONS};

use unsafe_cell_slice::UnsafeCellSlice;

#[test]
fn smoke_test_ser_0_2() {
    let mut data = vec![0i64; 2];
    {
        let data = UnsafeCellSlice::new(&mut data);
        let data_a: &mut [i64] = unsafe { data.index_mut(0..1) };
        let data_b: &mut [i64] = unsafe { data.index_mut(1..2) };
        decr(&mut data_a[0]);
        incr(&mut data_b[0]);
        decr(&mut data_a[0]);
        incr(&mut data_b[0]);
    }
    assert_eq!(data[0], -2 * N_REPETITIONS);
    assert_eq!(data[1], 2 * N_REPETITIONS);
}

#[ignore] // https://github.com/crossbeam-rs/crossbeam/pull/871?
#[test]
fn smoke_test_par_0_2_par() {
    let mut data = vec![0i64; 2];
    {
        let data = UnsafeCellSlice::new(&mut data);
        let data_a: &mut [i64] = unsafe { data.index_mut(0..1) };
        let data_b: &mut [i64] = unsafe { data.index_mut(1..2) };
        rayon::join(|| decr(&mut data_a[0]), || incr(&mut data_b[0]));
        rayon::join(|| decr(&mut data_a[0]), || incr(&mut data_b[0]));
    }
    assert_eq!(data[0], -2 * N_REPETITIONS);
    assert_eq!(data[1], 2 * N_REPETITIONS);
}

#[test]
fn index_full() {
    let mut data = vec![0i64; 2];
    {
        let data = UnsafeCellSlice::new(&mut data);
        let data_a: &mut i64 = &mut unsafe { data.index_mut(..) }[0];
        decr(data_a);
        decr(data_a);
    }
    assert_eq!(data[0], -2 * N_REPETITIONS);
    assert_eq!(data[1], 0);
}

#[test]
fn index_usize() {
    let mut data = vec![0i64; 2];
    {
        let data = UnsafeCellSlice::new(&mut data);
        let data_a: &mut i64 = unsafe { data.index_mut(0) };
        let data_b: &mut i64 = unsafe { data.index_mut(1) };
        decr(data_a);
        incr(data_b);
        decr(data_a);
        incr(data_b);
    }
    assert_eq!(data[0], -2 * N_REPETITIONS);
    assert_eq!(data[1], 2 * N_REPETITIONS);
}

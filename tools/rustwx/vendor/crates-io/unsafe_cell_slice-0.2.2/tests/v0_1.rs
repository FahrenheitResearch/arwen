// #[path = "common.rs"]
// mod common;
// use common::{decr, incr, N_REPETITIONS};

// use unsafe_cell_slice::UnsafeCellSlice;

// #[test]
// fn smoke_test_ser_0_1() {
//     let mut data = vec![0i64; 2];
//     {
//         let data = UnsafeCellSlice::new(&mut data);
//         let data_a: &mut [i64] = unsafe { data.as_mut_slice() };
//         let data_b: &mut [i64] = unsafe { data.as_mut_slice() };
//         decr(&mut data_a[0]);
//         incr(&mut data_b[1]);
//         decr(&mut data_a[0]);
//         incr(&mut data_b[1]);
//     }
//     assert_eq!(data[0], -2 * N_REPETITIONS);
//     assert_eq!(data[1], 2 * N_REPETITIONS);
// }

// #[test]
// fn smoke_test_par_0_1() {
//     let mut data = vec![0i64; 2];
//     {
//         let data = UnsafeCellSlice::new(&mut data);
//         let data_a: &mut [i64] = unsafe { data.as_mut_slice() };
//         let data_b: &mut [i64] = unsafe { data.as_mut_slice() };
//         rayon::join(|| decr(&mut data_a[0]), || incr(&mut data_b[1]));
//         rayon::join(|| decr(&mut data_a[0]), || incr(&mut data_b[1]));
//     }
//     assert_eq!(data[0], -2 * N_REPETITIONS);
//     assert_eq!(data[1], 2 * N_REPETITIONS);
// }

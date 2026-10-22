pub const N_REPETITIONS: i64 = 1_000;

pub fn decr(data: &mut i64) {
    for _ in 0..N_REPETITIONS {
        *data -= 1;
    }
}

pub fn incr(data: &mut i64) {
    for _ in 0..N_REPETITIONS {
        *data += 1;
    }
}

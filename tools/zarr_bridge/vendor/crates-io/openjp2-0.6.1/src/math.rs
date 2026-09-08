#[inline]
pub(crate) fn opj_int_max(mut a: i32, mut b: i32) -> i32 {
  a.max(b)
}

#[inline]
pub(crate) fn opj_int_min(mut a: i32, mut b: i32) -> i32 {
  a.min(b)
}

#[inline]
pub(crate) fn opj_uint_min(mut a: u32, mut b: u32) -> u32 {
  a.min(b)
}

#[inline]
pub(crate) fn opj_uint_max(mut a: u32, mut b: u32) -> u32 {
  a.max(b)
}

#[inline]
pub(crate) fn opj_int_floordivpow2(mut a: i32, mut b: i32) -> i32 {
  a >> b
}

#[inline]
pub(crate) fn opj_int_floorlog2(mut a: i32) -> i32 {
  let mut l: i32 = 0;
  l = 0;
  while a > 1 {
    a >>= 1;
    l += 1
  }
  l
}

#[inline]
pub(crate) fn opj_uint_floordivpow2(mut a: u32, mut b: u32) -> u32 {
  a >> b
}

#[inline]
pub(crate) fn opj_uint_floorlog2(mut a: u32) -> u32 {
  let mut l: u32 = 0;
  l = 0_u32;
  while a > 1 {
    a >>= 1;
    l += 1;
  }
  l
}

#[inline]
pub(crate) fn opj_uint_ceildivpow2(mut a: u32, mut b: u32) -> u32 {
  ((a as u64).wrapping_add(1_u64 << b).wrapping_sub(1_u64) >> b) as u32
}

#[inline]
pub(crate) fn opj_uint_ceildiv(mut a: u32, mut b: u32) -> u32 {
  assert!(b != 0);
  (a as u64)
    .wrapping_add(b as u64)
    .wrapping_sub(1_u64)
    .wrapping_div(b as u64) as u32
}

#[inline]
pub(crate) fn opj_int64_ceildivpow2(mut a: i64, mut b: i32) -> i32 {
  ((a + (1_i64 << b) - 1_i64) >> b) as i32
}

#[inline]
pub(crate) fn opj_int_ceildiv(mut a: i32, mut b: i32) -> i32 {
  assert!(b != 0);
  ((a as i64 + b as i64 - 1_i64) / b as i64) as i32
}

#[inline]
pub(crate) fn opj_uint64_ceildiv_res_uint32(mut a: u64, mut b: u64) -> u32 {
  assert!(b != 0);
  ((a + b - 1) / b) as u32
}

#[inline]
pub(crate) fn opj_int_ceildivpow2(mut a: i32, mut b: i32) -> i32 {
  ((a as i64 + (1_i64 << b) - 1_i64) >> b) as i32
}

#[inline]
pub(crate) fn opj_uint_adds(mut a: u32, mut b: u32) -> u32 {
  a.saturating_add(b)
}

#[inline]
pub(crate) fn opj_int_abs(mut a: i32) -> i32 {
  a.abs()
}

#[inline]
pub(crate) fn opj_uint_subs(mut a: u32, mut b: u32) -> u32 {
  a.saturating_sub(b)
}

#[inline]
pub(crate) fn opj_lrintf(mut f: f32) -> i64 {
  f.round_ties_even() as i64
}

#[inline]
pub(crate) fn opj_int64_clamp(mut a: i64, mut min: i64, mut max: i64) -> i64 {
  a.clamp(min, max)
}

#[inline]
pub(crate) fn opj_int_clamp(mut a: i32, mut min: i32, mut max: i32) -> i32 {
  a.clamp(min, max)
}

#[inline]
pub(crate) fn opj_int_sub_no_overflow(mut a: i32, mut b: i32) -> i32 {
  a.wrapping_sub(b)
}

#[inline]
pub(crate) fn opj_int_add_no_overflow(mut a: i32, mut b: i32) -> i32 {
  a.wrapping_add(b)
}

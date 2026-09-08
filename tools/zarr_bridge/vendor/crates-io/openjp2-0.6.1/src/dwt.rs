use std::alloc::{alloc, dealloc, Layout};

use super::math::*;
use super::openjpeg::*;
use super::sparse_array::*;

extern "C" {
  fn floor(_: core::ffi::c_double) -> core::ffi::c_double;

  fn memcpy(
    _: *mut core::ffi::c_void,
    _: *const core::ffi::c_void,
    _: usize,
  ) -> *mut core::ffi::c_void;
}

/* Where void* is a OPJ_INT32* for 5x3 and OPJ_FLOAT32* for 9x7 */
pub type opj_encode_and_deinterleave_h_one_row_fnptr_type = Option<
  unsafe extern "C" fn(
    _: *mut core::ffi::c_void,
    _: *mut core::ffi::c_void,
    _: OPJ_UINT32,
    _: OPJ_BOOL,
  ) -> (),
>;

/* Forward transform, for the vertical pass, processing cols columns */
/* where cols <= NB_ELTS_V8 */
/* Where void* is a OPJ_INT32* for 5x3 and OPJ_FLOAT32* for 9x7 */
pub type opj_encode_and_deinterleave_v_fnptr_type = Option<
  unsafe extern "C" fn(
    _: *mut core::ffi::c_void,
    _: *mut core::ffi::c_void,
    _: OPJ_UINT32,
    _: OPJ_BOOL,
    _: OPJ_UINT32,
    _: OPJ_UINT32,
  ) -> (),
>;

/* * @name Local data structures */
/*@{*/

pub const VREG_INT_COUNT: usize = 4;

pub const PARALLEL_COLS_53: usize = 2 * VREG_INT_COUNT;

pub type opj_dwt_t = dwt_local;
#[derive(Copy, Clone)]
pub struct dwt_local {
  pub mem: *mut OPJ_INT32,
  /// number of elements in high pass band
  pub dn: OPJ_INT32,
  /// number of elements in low pass band
  pub sn: OPJ_INT32,
  /// 0 = start on even coord, 1 = start on odd coord
  pub cas: OPJ_INT32,
}

impl dwt_local {
  fn sn_idx(&self, i: i32) -> i32 {
    i.clamp(0, self.sn - 1)
  }

  fn dn_idx(&self, i: i32) -> i32 {
    i.clamp(0, self.dn - 1)
  }

  fn s_idx(&self, i: i32, chunk: isize) -> isize {
    (i * 2) as isize * chunk
  }

  fn d_idx(&self, i: i32, chunk: isize) -> isize {
    (1 + (i * 2)) as isize * chunk
  }

  fn s__idx(&self, i: i32, chunk: isize) -> isize {
    self.s_idx(self.sn_idx(i), chunk)
  }

  fn d__idx(&self, i: i32, chunk: isize) -> isize {
    self.d_idx(self.dn_idx(i), chunk)
  }

  fn ss__idx(&self, i: i32, chunk: isize) -> isize {
    self.s_idx(self.dn_idx(i), chunk)
  }

  fn dd__idx(&self, i: i32, chunk: isize) -> isize {
    self.d_idx(self.sn_idx(i), chunk)
  }

  fn m(&self, i: isize) -> i32 {
    unsafe { *self.mem.offset(i) }
  }

  fn m_mut(&self, i: isize) -> &mut i32 {
    unsafe { &mut *self.mem.offset(i) }
  }

  fn s(&self, i: i32) -> i32 {
    self.m(self.s_idx(i, 1))
  }

  fn d(&self, i: i32) -> i32 {
    self.m(self.d_idx(i, 1))
  }

  fn s_mut(&self, i: i32) -> &mut i32 {
    self.m_mut(self.s_idx(i, 1))
  }

  fn d_mut(&self, i: i32) -> &mut i32 {
    self.m_mut(self.d_idx(i, 1))
  }

  fn s_(&self, i: i32) -> i32 {
    self.m(self.s__idx(i, 1))
  }

  fn d_(&self, i: i32) -> i32 {
    self.m(self.d__idx(i, 1))
  }

  fn ss_(&self, i: i32) -> i32 {
    self.m(self.ss__idx(i, 1))
  }

  fn dd_(&self, i: i32) -> i32 {
    self.m(self.dd__idx(i, 1))
  }

  /// Inverse 5-3 wavelet transform in 1-D.
  fn decode_1(&self) {
    if self.cas == 0 {
      if self.dn > 0 || self.sn > 1 {
        for i in 0..self.sn {
          let val = (self.d_(i - 1) + self.d_(i) + 2) >> 2;
          *self.s_mut(i) -= val;
        }
        for i in 0..self.dn {
          let val = (self.s_(i) + self.s_(i + 1)) >> 1;
          *self.d_mut(i) += val;
        }
      }
    } else {
      if self.sn == 0 && self.dn == 1 {
        *self.s_mut(0) /= 2;
      } else {
        for i in 0..self.sn {
          let val = (self.ss_(i) + self.ss_(i + 1) + 2) >> 2;
          *self.d_mut(i) -= val;
        }
        for i in 0..self.dn {
          let val = (self.dd_(i) + self.dd_(i - 1)) >> 1;
          *self.s_mut(i) += val;
        }
      }
    }
  }

  fn decode_partial_1(
    &self,
    win_l_x0: OPJ_INT32,
    win_l_x1: OPJ_INT32,
    win_h_x0: OPJ_INT32,
    win_h_x1: OPJ_INT32,
  ) {
    if self.cas == 0 {
      if self.dn > 0 || self.sn > 1 {
        for i in win_l_x0..win_l_x1 {
          let val = (self.d_(i - 1) + self.d_(i) + 2) >> 2;
          *self.s_mut(i) -= val;
        }
        for i in win_h_x0..win_h_x1 {
          let val = (self.s_(i) + self.s_(i + 1)) >> 1;
          *self.d_mut(i) += val;
        }
      }
    } else {
      if self.sn == 0 && self.dn == 1 {
        *self.s_mut(0) /= 2;
      } else {
        for i in win_l_x0..win_l_x1 {
          let val = self
            .d(i)
            .wrapping_sub(self.ss_(i).wrapping_add(self.ss_(i + 1)).wrapping_add(2) >> 2);
          *self.d_mut(i) = val;
        }
        for i in win_h_x0..win_h_x1 {
          let val = self
            .s(i)
            .wrapping_add(self.dd_(i).wrapping_add(self.dd_(i - 1)) >> 1);
          *self.s_mut(i) = val;
        }
      }
    }
  }

  fn decode_partial_1_parallel(
    &self,
    win_l_x0: OPJ_INT32,
    win_l_x1: OPJ_INT32,
    win_h_x0: OPJ_INT32,
    win_h_x1: OPJ_INT32,
  ) {
    if self.cas == 0 {
      if self.dn > 0 || self.sn > 1 {
        for i in win_l_x0..win_l_x1 {
          let s_idx = self.s_idx(i, 4);
          let d0_idx = self.d__idx(i - 1, 4);
          let d1_idx = self.d__idx(i, 4);
          for off in 0..4 {
            let val = self.m(s_idx + off).wrapping_sub(
              self
                .m(d0_idx + off)
                .wrapping_add(self.m(d1_idx + off))
                .wrapping_add(2)
                >> 2,
            );
            *self.m_mut(s_idx + off) = val;
          }
        }
        for i in win_h_x0..win_h_x1 {
          let d_idx = self.d_idx(i, 4);
          let s0_idx = self.s__idx(i, 4);
          let s1_idx = self.s__idx(i + 1, 4);
          for off in 0..4 {
            let val = self
              .m(d_idx + off)
              .wrapping_add(self.m(s0_idx + off).wrapping_add(self.m(s1_idx + off)) >> 1);
            *self.m_mut(d_idx + off) = val;
          }
        }
      }
    } else {
      if self.sn == 0 && self.dn == 1 {
        let s_idx = self.s_idx(0, 4);
        for off in 0..4 {
          *self.m_mut(s_idx + off) /= 2;
        }
      } else {
        for i in win_l_x0..win_l_x1 {
          let d_idx = self.d_idx(i, 4);
          let s0_idx = self.ss__idx(i, 4);
          let s1_idx = self.ss__idx(i + 1, 4);
          for off in 0..4 {
            let val = self.m(d_idx + off).wrapping_sub(
              self
                .m(s0_idx + off)
                .wrapping_add(self.m(s1_idx + off))
                .wrapping_add(2)
                >> 2,
            );
            *self.m_mut(d_idx + off) = val;
          }
        }
        for i in win_h_x0..win_h_x1 {
          let s_idx = self.s_idx(i, 4);
          let d0_idx = self.dd__idx(i, 4);
          let d1_idx = self.dd__idx(i - 1, 4);
          for off in 0..4 {
            let val = self
              .m(s_idx + off)
              .wrapping_add(self.m(d0_idx + off).wrapping_add(self.m(d1_idx + off)) >> 1);
            *self.m_mut(s_idx + off) = val;
          }
        }
      }
    }
  }
}

pub const NB_ELTS_V8: u32 = 8;
#[repr(C)]
#[derive(Copy, Clone)]
pub union opj_v8_t {
  pub f: [OPJ_FLOAT32; NB_ELTS_V8 as usize],
}

pub type opj_v8dwt_t = v8dwt_local;
#[repr(C)]
#[derive(Copy, Clone)]
pub struct v8dwt_local {
  pub wavelet: *mut opj_v8_t,
  /// number of elements in high pass band
  pub dn: OPJ_INT32,
  /// number of elements in low pass band
  pub sn: OPJ_INT32,
  /// 0 = start on even coord, 1 = start on odd coord
  pub cas: OPJ_INT32,
  /// start coord in low pass band
  pub win_l_x0: OPJ_UINT32,
  /// end coord in low pass band
  pub win_l_x1: OPJ_UINT32,
  /// start coord in high pass band
  pub win_h_x0: OPJ_UINT32,
  /// end coord in high pass band
  pub win_h_x1: OPJ_UINT32,
}

/* From table F.4 from the standard */
const opj_dwt_alpha: f32 = -1.586_134_3;
const opj_dwt_beta: f32 = -0.052980118;
const opj_dwt_gamma: f32 = 0.882_911_1;
const opj_dwt_delta: f32 = 0.443_506_87;
const opj_K: f32 = 1.230_174_1;
const opj_invK: f32 = 1.0 / 1.230_174_1;

/* <summary>                                                              */
/* This table contains the norms of the 5-3 wavelets for different bands. */
/* </summary>                                                             */
/* FIXME! the array should really be extended up to 33 resolution levels */
/* See https://github.com/uclouvain/openjpeg/issues/493 */
const opj_dwt_norms: [[f64; 10]; 4] = [
  [
    1.000, 1.500, 2.750, 5.375, 10.68, 21.34, 42.67, 85.33, 170.7, 341.3,
  ],
  [
    1.038, 1.592, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9, 0.,
  ],
  [
    1.038, 1.592, 2.919, 5.703, 11.33, 22.64, 45.25, 90.48, 180.9, 0.,
  ],
  [
    0.7186, 0.9218, 1.586, 3.043, 6.019, 12.01, 24.00, 47.97, 95.93, 0.,
  ],
];
/* <summary>                                                              */
/* This table contains the norms of the 9-7 wavelets for different bands. */
/* </summary>                                                             */
/* FIXME! the array should really be extended up to 33 resolution levels */
/* See https://github.com/uclouvain/openjpeg/issues/493 */
const opj_dwt_norms_real: [[f64; 10]; 4] = [
  [
    1.000, 1.965, 4.177, 8.403, 16.90, 33.84, 67.69, 135.3, 270.6, 540.9,
  ],
  [
    2.022, 3.989, 8.355, 17.04, 34.27, 68.63, 137.3, 274.6, 549.0, 0.,
  ],
  [
    2.022, 3.989, 8.355, 17.04, 34.27, 68.63, 137.3, 274.6, 549.0, 0.,
  ],
  [
    2.080, 3.865, 8.307, 17.18, 34.71, 69.59, 139.3, 278.6, 557.2, 0.,
  ],
];

/*@}*/
/* * @name Local static functions */
/*@{*/
/* *
Forward lazy transform (horizontal)
*/
/*
==========================================================
   local functions
==========================================================
*/
/* <summary>                             */
/* Forward lazy transform (horizontal).  */
/* </summary>                            */
fn opj_dwt_deinterleave_h(
  mut a: *const OPJ_INT32,
  mut b: *mut OPJ_INT32,
  mut dn: OPJ_INT32,
  mut sn: OPJ_INT32,
  mut cas: OPJ_INT32,
) {
  unsafe {
    let mut i: OPJ_INT32 = 0;
    let mut l_dest = b;
    let mut l_src = a.offset(cas as isize);
    i = 0i32;
    while i < sn {
      let fresh0 = l_dest;
      l_dest = l_dest.offset(1);
      *fresh0 = *l_src;
      l_src = l_src.offset(2);
      i += 1
    }
    l_dest = b.offset(sn as isize);
    l_src = a.offset(1).offset(-(cas as isize));
    i = 0i32;
    while i < dn {
      let fresh1 = l_dest;
      l_dest = l_dest.offset(1);
      *fresh1 = *l_src;
      l_src = l_src.offset(2);
      i += 1
    }
  }
}
/* <summary>                             */
/* Inverse lazy transform (horizontal).  */
/* </summary>                            */
fn opj_dwt_interleave_h(mut h: &opj_dwt_t, mut a: *mut OPJ_INT32) {
  unsafe {
    let mut ai: *const OPJ_INT32 = a;
    let mut bi = h.mem.offset(h.cas as isize);
    let mut i = h.sn;
    loop {
      let fresh2 = i;
      i -= 1;
      if fresh2 == 0 {
        break;
      }
      let fresh3 = ai;
      ai = ai.offset(1);
      *bi = *fresh3;
      bi = bi.offset(2)
    }
    ai = a.offset(h.sn as isize);
    bi = h.mem.offset(1).offset(-(h.cas as isize));
    i = h.dn;
    loop {
      let fresh4 = i;
      i -= 1;
      if fresh4 == 0 {
        break;
      }
      let fresh5 = ai;
      ai = ai.offset(1);
      *bi = *fresh5;
      bi = bi.offset(2)
    }
  }
}

/* <summary>                             */
/* Inverse lazy transform (vertical).    */
/* </summary>                            */
fn opj_dwt_interleave_v(mut v: &opj_dwt_t, mut a: *mut OPJ_INT32, mut x: OPJ_INT32) {
  unsafe {
    let mut ai: *const OPJ_INT32 = a;
    let mut bi = v.mem.offset(v.cas as isize);
    let mut i = v.sn;
    loop {
      let fresh6 = i;
      i -= 1;
      if fresh6 == 0 {
        break;
      }
      *bi = *ai;
      bi = bi.offset(2);
      ai = ai.offset(x as isize)
    }
    ai = a.add((v.sn as usize).wrapping_mul(x as OPJ_SIZE_T));
    bi = v.mem.offset(1).offset(-(v.cas as isize));
    i = v.dn;
    loop {
      let fresh7 = i;
      i -= 1;
      if fresh7 == 0 {
        break;
      }
      *bi = *ai;
      bi = bi.offset(2);
      ai = ai.offset(x as isize)
    }
  }
}

/* <summary>                            */
/* Inverse 5-3 wavelet transform in 1-D for one row. */
/* </summary>                           */
/* Performs interleave, inverse wavelet transform and copy back to buffer */
fn opj_idwt53_h(mut dwt: &opj_dwt_t, mut tiledp: *mut OPJ_INT32) {
  unsafe {
    /* For documentation purpose */
    opj_dwt_interleave_h(dwt, tiledp);
    dwt.decode_1();
    memcpy(
      tiledp as *mut core::ffi::c_void,
      dwt.mem as *const core::ffi::c_void,
      ((dwt.sn + dwt.dn) as OPJ_UINT32 as usize).wrapping_mul(core::mem::size_of::<OPJ_INT32>()),
    );
  }
}

/* <summary>                            */
/* Inverse vertical 5-3 wavelet transform in 1-D for several columns. */
/* </summary>                           */
/* Performs interleave, inverse wavelet transform and copy back to buffer */
fn opj_idwt53_v(
  mut dwt: &opj_dwt_t,
  mut tiledp_col: *mut OPJ_INT32,
  mut stride: OPJ_SIZE_T,
  mut nb_cols: OPJ_INT32,
) {
  unsafe {
    /* For documentation purpose */
    let mut k = 0i32;
    let mut c = 0i32;
    while c < nb_cols {
      opj_dwt_interleave_v(dwt, tiledp_col.offset(c as isize), stride as OPJ_INT32);
      dwt.decode_1();
      k = 0;
      while k < dwt.sn + dwt.dn {
        *tiledp_col.add((c as usize).wrapping_add((k as usize).wrapping_mul(stride))) =
          *dwt.mem.offset(k as isize);
        k += 1
      }
      c += 1
    }
  }
}

fn opj_dwt_encode_step1_combined(
  mut fw: *mut OPJ_FLOAT32,
  mut iters_c1: OPJ_UINT32,
  mut iters_c2: OPJ_UINT32,
  c1: OPJ_FLOAT32,
  c2: OPJ_FLOAT32,
) {
  unsafe {
    let mut i = 0 as OPJ_UINT32;
    let iters_common = opj_uint_min(iters_c1, iters_c2);

    assert!(fw as OPJ_SIZE_T & 0xfusize == 0usize);
    assert!(opj_int_abs(iters_c1 as OPJ_INT32 - iters_c2 as OPJ_INT32) <= 1i32);
    while i.wrapping_add(3u32) < iters_common {
      let fresh13 = &mut (*fw.offset(0));
      *fresh13 *= c1;
      let fresh14 = &mut (*fw.offset(1));
      *fresh14 *= c2;
      let fresh15 = &mut (*fw.offset(2));
      *fresh15 *= c1;
      let fresh16 = &mut (*fw.offset(3));
      *fresh16 *= c2;
      let fresh17 = &mut (*fw.offset(4));
      *fresh17 *= c1;
      let fresh18 = &mut (*fw.offset(5));
      *fresh18 *= c2;
      let fresh19 = &mut (*fw.offset(6));
      *fresh19 *= c1;
      let fresh20 = &mut (*fw.offset(7));
      *fresh20 *= c2;
      fw = fw.offset(8);
      i = (i as core::ffi::c_uint).wrapping_add(4u32) as OPJ_UINT32
    }
    while i < iters_common {
      let fresh21 = &mut (*fw.offset(0));
      *fresh21 *= c1;
      let fresh22 = &mut (*fw.offset(1));
      *fresh22 *= c2;
      fw = fw.offset(2);
      i += 1;
    }
    if i < iters_c1 {
      let fresh23 = &mut (*fw.offset(0));
      *fresh23 *= c1
    } else if i < iters_c2 {
      let fresh24 = &mut (*fw.offset(1));
      *fresh24 *= c2
    };
  }
}
fn opj_dwt_encode_step2(
  mut fl: *mut OPJ_FLOAT32,
  mut fw: *mut OPJ_FLOAT32,
  mut end: OPJ_UINT32,
  mut m: OPJ_UINT32,
  mut c: OPJ_FLOAT32,
) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut imax = opj_uint_min(end, m);
    if imax > 0u32 {
      let fresh25 = &mut (*fw.offset(-(1i32) as isize));
      *fresh25 += (*fl.offset(0) + *fw.offset(0)) * c;
      fw = fw.offset(2);
      i = 1 as OPJ_UINT32;
      while i.wrapping_add(3u32) < imax {
        let fresh26 = &mut (*fw.offset(-(1i32) as isize));
        *fresh26 += (*fw.offset(-(2i32) as isize) + *fw.offset(0)) * c;
        let fresh27 = &mut (*fw.offset(1));
        *fresh27 += (*fw.offset(0) + *fw.offset(2)) * c;
        let fresh28 = &mut (*fw.offset(3));
        *fresh28 += (*fw.offset(2) + *fw.offset(4)) * c;
        let fresh29 = &mut (*fw.offset(5));
        *fresh29 += (*fw.offset(4) + *fw.offset(6)) * c;
        fw = fw.offset(8);
        i = (i as core::ffi::c_uint).wrapping_add(4u32) as OPJ_UINT32 as OPJ_UINT32
      }
      while i < imax {
        let fresh30 = &mut (*fw.offset(-(1i32) as isize));
        *fresh30 += (*fw.offset(-(2i32) as isize) + *fw.offset(0)) * c;
        fw = fw.offset(2);
        i += 1;
      }
    }
    if m < end {
      assert!(m.wrapping_add(1u32) == end);
      let fresh31 = &mut (*fw.offset(-(1i32) as isize));
      *fresh31 += 2 as core::ffi::c_float * *fw.offset(-(2i32) as isize) * c
    };
  }
}
/* *
Forward 9-7 wavelet transform in 1-D
*/
fn opj_dwt_encode_1_real(
  mut aIn: *mut core::ffi::c_void,
  mut dn: OPJ_INT32,
  mut sn: OPJ_INT32,
  mut cas: OPJ_INT32,
) {
  unsafe {
    let mut w = aIn as *mut OPJ_FLOAT32;
    let mut a: OPJ_INT32 = 0;
    let mut b: OPJ_INT32 = 0;
    assert!(dn + sn > 1i32);
    if cas == 0i32 {
      a = 0i32;
      b = 1i32
    } else {
      a = 1i32;
      b = 0i32
    }
    opj_dwt_encode_step2(
      w.offset(a as isize),
      w.offset(b as isize).offset(1),
      dn as OPJ_UINT32,
      opj_int_min(dn, sn - b) as OPJ_UINT32,
      opj_dwt_alpha,
    );
    opj_dwt_encode_step2(
      w.offset(b as isize),
      w.offset(a as isize).offset(1),
      sn as OPJ_UINT32,
      opj_int_min(sn, dn - a) as OPJ_UINT32,
      opj_dwt_beta,
    );
    opj_dwt_encode_step2(
      w.offset(a as isize),
      w.offset(b as isize).offset(1),
      dn as OPJ_UINT32,
      opj_int_min(dn, sn - b) as OPJ_UINT32,
      opj_dwt_gamma,
    );
    opj_dwt_encode_step2(
      w.offset(b as isize),
      w.offset(a as isize).offset(1),
      sn as OPJ_UINT32,
      opj_int_min(sn, dn - a) as OPJ_UINT32,
      opj_dwt_delta,
    );
    if a == 0i32 {
      opj_dwt_encode_step1_combined(w, sn as OPJ_UINT32, dn as OPJ_UINT32, opj_invK, opj_K);
    } else {
      opj_dwt_encode_step1_combined(w, dn as OPJ_UINT32, sn as OPJ_UINT32, opj_K, opj_invK);
    };
  }
}
/* *
Explicit calculation of the Quantization Stepsizes
*/
fn opj_dwt_encode_stepsize(
  mut stepsize: OPJ_INT32,
  mut numbps: OPJ_INT32,
  mut bandno_stepsize: *mut opj_stepsize_t,
) {
  unsafe {
    let mut p: OPJ_INT32 = 0;
    let mut n: OPJ_INT32 = 0;
    p = opj_int_floorlog2(stepsize) - 13i32;
    n = 11i32 - opj_int_floorlog2(stepsize);
    (*bandno_stepsize).mant = if n < 0i32 {
      (stepsize) >> -n
    } else {
      (stepsize) << n
    } & 0x7ffi32;
    (*bandno_stepsize).expn = numbps - p;
  }
}
/*
==========================================================
   DWT interface
==========================================================
*/
/* * Process one line for the horizontal pass of the 5x3 forward transform */
unsafe extern "C" fn opj_dwt_encode_and_deinterleave_h_one_row(
  mut rowIn: *mut core::ffi::c_void,
  mut tmpIn: *mut core::ffi::c_void,
  mut width: OPJ_UINT32,
  mut even: OPJ_BOOL,
) {
  unsafe {
    let mut row = rowIn as *mut OPJ_INT32;
    let mut tmp = tmpIn as *mut OPJ_INT32;
    let sn = (width.wrapping_add(if even != 0 { 1 } else { 0 }) >> 1i32) as OPJ_INT32;
    let dn = width.wrapping_sub(sn as OPJ_UINT32) as OPJ_INT32;
    if even != 0 {
      if width > 1u32 {
        let mut i: OPJ_INT32 = 0;
        i = 0i32;
        while i < sn - 1i32 {
          *tmp.offset((sn + i) as isize) = *row.offset((2i32 * i + 1i32) as isize)
            - ((*row.offset((i * 2i32) as isize) + *row.offset(((i + 1i32) * 2i32) as isize))
              >> 1i32);
          i += 1
        }
        if width.wrapping_rem(2u32) == 0u32 {
          *tmp.offset((sn + i) as isize) =
            *row.offset((2i32 * i + 1i32) as isize) - *row.offset((i * 2i32) as isize)
        }
        let fresh32 = &mut (*row.offset(0));
        *fresh32 += (*tmp.offset(sn as isize) + *tmp.offset(sn as isize) + 2i32) >> 2i32;
        i = 1i32;
        while i < dn {
          *row.offset(i as isize) = *row.offset((2i32 * i) as isize)
            + ((*tmp.offset((sn + (i - 1i32)) as isize) + *tmp.offset((sn + i) as isize) + 2i32)
              >> 2i32);
          i += 1
        }
        if width.wrapping_rem(2u32) == 1u32 {
          *row.offset(i as isize) = *row.offset((2i32 * i) as isize)
            + ((*tmp.offset((sn + (i - 1i32)) as isize)
              + *tmp.offset((sn + (i - 1i32)) as isize)
              + 2i32)
              >> 2i32)
        }
        memcpy(
          row.offset(sn as isize) as *mut core::ffi::c_void,
          tmp.offset(sn as isize) as *const core::ffi::c_void,
          (dn as OPJ_SIZE_T).wrapping_mul(core::mem::size_of::<OPJ_INT32>()),
        );
      }
    } else if width == 1u32 {
      let fresh33 = &mut (*row.offset(0));
      *fresh33 *= 2i32
    } else {
      let mut i_0: OPJ_INT32 = 0;
      *tmp.offset(sn as isize) = *row.offset(0) - *row.offset(1);
      i_0 = 1i32;
      while i_0 < sn {
        *tmp.offset((sn + i_0) as isize) = *row.offset((2i32 * i_0) as isize)
          - ((*row.offset((2i32 * i_0 + 1i32) as isize)
            + *row.offset((2i32 * (i_0 - 1i32) + 1i32) as isize))
            >> 1i32);
        i_0 += 1
      }
      if width.wrapping_rem(2u32) == 1u32 {
        *tmp.offset((sn + i_0) as isize) =
          *row.offset((2i32 * i_0) as isize) - *row.offset((2i32 * (i_0 - 1i32) + 1i32) as isize)
      }
      i_0 = 0i32;
      while i_0 < dn - 1i32 {
        *row.offset(i_0 as isize) = *row.offset((2i32 * i_0 + 1i32) as isize)
          + ((*tmp.offset((sn + i_0) as isize) + *tmp.offset((sn + i_0 + 1i32) as isize) + 2i32)
            >> 2i32);
        i_0 += 1
      }
      if width.wrapping_rem(2u32) == 0u32 {
        *row.offset(i_0 as isize) = *row.offset((2i32 * i_0 + 1i32) as isize)
          + ((*tmp.offset((sn + i_0) as isize) + *tmp.offset((sn + i_0) as isize) + 2i32) >> 2i32)
      }
      memcpy(
        row.offset(sn as isize) as *mut core::ffi::c_void,
        tmp.offset(sn as isize) as *const core::ffi::c_void,
        (dn as OPJ_SIZE_T).wrapping_mul(core::mem::size_of::<OPJ_INT32>()),
      );
    };
  }
}
/* * Process one line for the horizontal pass of the 9x7 forward transform */
unsafe extern "C" fn opj_dwt_encode_and_deinterleave_h_one_row_real(
  mut rowIn: *mut core::ffi::c_void,
  mut tmpIn: *mut core::ffi::c_void,
  mut width: OPJ_UINT32,
  mut even: OPJ_BOOL,
) {
  let mut row = rowIn as *mut OPJ_FLOAT32;
  let mut tmp = tmpIn as *mut OPJ_FLOAT32;
  let sn = (width.wrapping_add(if even != 0 { 1 } else { 0 }) >> 1i32) as OPJ_INT32;
  let dn = width.wrapping_sub(sn as OPJ_UINT32) as OPJ_INT32;
  if width == 1u32 {
    return;
  }
  memcpy(
    tmp as *mut core::ffi::c_void,
    row as *const core::ffi::c_void,
    (width as usize).wrapping_mul(core::mem::size_of::<OPJ_FLOAT32>()),
  );
  opj_dwt_encode_1_real(
    tmp as *mut core::ffi::c_void,
    dn,
    sn,
    if even != 0 { 0i32 } else { 1i32 },
  );
  opj_dwt_deinterleave_h(
    tmp as *mut OPJ_INT32,
    row as *mut OPJ_INT32,
    dn,
    sn,
    if even != 0 { 0i32 } else { 1i32 },
  );
}

/* * Fetch up to cols <= NB_ELTS_V8 for each line, and put them in tmpOut */
/* that has a NB_ELTS_V8 interleave factor. */
fn opj_dwt_fetch_cols_vertical_pass(
  mut arrayIn: *const core::ffi::c_void,
  mut tmpOut: *mut core::ffi::c_void,
  mut height: OPJ_UINT32,
  mut stride_width: OPJ_UINT32,
  mut cols: OPJ_UINT32,
) {
  unsafe {
    let mut array = arrayIn as *const OPJ_INT32;
    let mut tmp = tmpOut as *mut OPJ_INT32;
    if cols == NB_ELTS_V8 {
      let mut k: OPJ_UINT32 = 0;
      k = 0 as OPJ_UINT32;
      while k < height {
        memcpy(
          tmp.offset((NB_ELTS_V8 * k) as isize) as *mut core::ffi::c_void,
          array.offset(k.wrapping_mul(stride_width) as isize) as *const core::ffi::c_void,
          NB_ELTS_V8 as usize * core::mem::size_of::<OPJ_INT32>(),
        );
        k += 1;
      }
    } else {
      let mut k_0: OPJ_UINT32 = 0;
      k_0 = 0 as OPJ_UINT32;
      while k_0 < height {
        let mut c: OPJ_UINT32 = 0;
        c = 0 as OPJ_UINT32;
        while c < cols {
          *tmp.offset((NB_ELTS_V8 * k_0 + c) as isize) =
            *array.offset(c.wrapping_add(k_0.wrapping_mul(stride_width)) as isize);
          c += 1;
        }
        while c < NB_ELTS_V8 {
          *tmp.offset((NB_ELTS_V8 * k_0 + c) as isize) = 0i32;
          c += 1;
        }
        k_0 += 1;
      }
    };
  }
}

/* Deinterleave result of forward transform, where cols <= NB_ELTS_V8 */
/* and src contains NB_ELTS_V8 consecutive values for up to NB_ELTS_V8 */
/* columns. */
#[inline]
fn opj_dwt_deinterleave_v_cols(
  mut src: *const OPJ_INT32,
  mut dst: *mut OPJ_INT32,
  mut dn: OPJ_INT32,
  mut sn: OPJ_INT32,
  mut stride_width: OPJ_UINT32,
  mut cas: OPJ_INT32,
  mut cols: OPJ_UINT32,
) {
  unsafe {
    let mut k: OPJ_INT32 = 0; /* fallthru */
    let mut i = sn; /* fallthru */
    let mut l_dest = dst; /* fallthru */
    let mut l_src = src.offset((cas * NB_ELTS_V8 as i32) as isize); /* fallthru */
    k = 0i32; /* fallthru */
    while k < 2i32 {
      while i > 0 {
        i -= 1;
        if cols == NB_ELTS_V8 {
          memcpy(
            l_dest as *mut core::ffi::c_void,
            l_src as *const core::ffi::c_void,
            NB_ELTS_V8 as usize * core::mem::size_of::<OPJ_INT32>(),
          );
        } else {
          *l_dest = *l_src;
          if cols <= 7 {
            for c in 1..cols {
              *l_dest.offset(c as isize) = *l_src.offset(c as isize);
            }
          }
        }
        l_dest = l_dest.offset(stride_width as isize);
        l_src = l_src.offset((2 * NB_ELTS_V8) as isize)
      }
      l_dest = dst.add((sn as OPJ_SIZE_T).wrapping_mul(stride_width as OPJ_SIZE_T));
      l_src = src.offset(((1 - cas) * NB_ELTS_V8 as i32) as isize);
      i = dn;
      k += 1
    }
  }
}

/* Forward 5-3 transform, for the vertical pass, processing cols columns */
/* where cols <= NB_ELTS_V8 */
unsafe extern "C" fn opj_dwt_encode_and_deinterleave_v(
  mut arrayIn: *mut core::ffi::c_void,
  mut tmpIn: *mut core::ffi::c_void,
  mut height: OPJ_UINT32,
  mut even: OPJ_BOOL,
  mut stride_width: OPJ_UINT32,
  mut cols: OPJ_UINT32,
) {
  let mut array = arrayIn as *mut OPJ_INT32;
  let sn = (height + if even != 0 { 1 } else { 0 }) >> 1;
  let dn = height - sn;
  opj_dwt_fetch_cols_vertical_pass(arrayIn, tmpIn, height, stride_width, cols);

  let mut tmp = tmpIn as *mut OPJ_INT32;
  let mut c: OPJ_UINT32 = 0;
  let mut i: OPJ_UINT32 = 0;

  let _sc_mut =
    |i: u32, c: u32| -> &'static mut i32 { &mut *(tmp.offset((i * 2 * NB_ELTS_V8 + c) as isize)) };

  let _sc = |i: u32, c: u32| -> i32 { *_sc_mut(i, c) };

  let _dc_mut = |i: u32, c: u32| -> &'static mut i32 {
    &mut *(tmp.offset(((1 + i * 2) * NB_ELTS_V8 + c) as isize))
  };

  let _dc = |i: u32, c: u32| -> i32 { *_dc_mut(i, c) };

  if even != 0 {
    if height > 1 {
      i = 0 as OPJ_UINT32;
      while i + 1 < sn {
        c = 0 as OPJ_UINT32;
        while c < NB_ELTS_V8 {
          *_dc_mut(i, c) -= (_sc(i, c) + _sc(i + 1, c)) >> 1;
          c += 1
        }
        i += 1
      }
      if height % 2 == 0 {
        c = 0 as OPJ_UINT32;
        while c < NB_ELTS_V8 {
          *_dc_mut(i, c) -= _sc(i, c);
          c += 1
        }
      }
      c = 0 as OPJ_UINT32;
      while c < NB_ELTS_V8 {
        *_sc_mut(0, c) += (_dc(0, c) + _dc(0, c) + 2) >> 2;
        c += 1
      }
      i = 1 as OPJ_UINT32;
      while i < dn {
        c = 0 as OPJ_UINT32;
        while c < NB_ELTS_V8 {
          *_sc_mut(i, c) += (_dc(i - 1, c) + _dc(i, c) + 2) >> 2;
          c += 1
        }
        i += 1
      }
      if height % 2 == 1 {
        c = 0 as OPJ_UINT32;
        while c < NB_ELTS_V8 {
          *_sc_mut(i, c) += (_dc(i - 1, c) + _dc(i - 1, c) + 2) >> 2;
          c += 1
        }
      }
    }
  } else if height == 1 {
    c = 0 as OPJ_UINT32;
    while c < NB_ELTS_V8 {
      *_sc_mut(0, c) *= 2;
      c += 1
    }
  } else {
    c = 0 as OPJ_UINT32;
    while c < NB_ELTS_V8 {
      *_sc_mut(0, c) -= _dc(0, c);
      c += 1
    }
    i = 1 as OPJ_UINT32;
    while i < sn {
      c = 0 as OPJ_UINT32;
      while c < NB_ELTS_V8 {
        *_sc_mut(i, c) -= (_dc(i, c) + _dc(i - 1, c)) >> 1;
        c += 1
      }
      i += 1
    }
    if height % 2 == 1 {
      c = 0 as OPJ_UINT32;
      while c < NB_ELTS_V8 {
        *_sc_mut(i, c) -= _dc(i - 1, c);
        c += 1
      }
    }
    i = 0 as OPJ_UINT32;
    while i + 1 < dn {
      c = 0 as OPJ_UINT32;
      while c < NB_ELTS_V8 {
        *_dc_mut(i, c) += (_sc(i, c) + _sc(i + 1, c) + 2) >> 2;
        c += 1
      }
      i += 1
    }
    if height % 2 == 0 {
      c = 0 as OPJ_UINT32;
      while c < NB_ELTS_V8 {
        *_dc_mut(i, c) += (_sc(i, c) + _sc(i, c) + 2) >> 2;
        c += 1
      }
    }
  }
  if cols == NB_ELTS_V8 {
    opj_dwt_deinterleave_v_cols(
      tmp,
      array,
      dn as OPJ_INT32,
      sn as OPJ_INT32,
      stride_width,
      if even != 0 { 0i32 } else { 1i32 },
      NB_ELTS_V8,
    );
  } else {
    opj_dwt_deinterleave_v_cols(
      tmp,
      array,
      dn as OPJ_INT32,
      sn as OPJ_INT32,
      stride_width,
      if even != 0 { 0i32 } else { 1i32 },
      cols,
    );
  };
}

fn opj_v8dwt_encode_step1(mut fw: *mut OPJ_FLOAT32, mut end: OPJ_UINT32, cst: OPJ_FLOAT32) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut c: OPJ_UINT32 = 0;
    i = 0 as OPJ_UINT32;
    while i < end {
      c = 0 as OPJ_UINT32;
      while c < NB_ELTS_V8 {
        let fresh46 = &mut (*fw.offset((i * 2 * NB_ELTS_V8 + c) as isize));
        *fresh46 *= cst;
        c += 1
      }
      i += 1
    }
  }
}

fn opj_v8dwt_encode_step2(
  mut fl: *mut OPJ_FLOAT32,
  mut fw: *mut OPJ_FLOAT32,
  mut end: OPJ_UINT32,
  mut m: OPJ_UINT32,
  mut cst: OPJ_FLOAT32,
) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut imax = opj_uint_min(end, m);
    let mut c: OPJ_INT32 = 0;
    if imax > 0u32 {
      c = 0;
      while c < NB_ELTS_V8 as i32 {
        let fresh47 = &mut (*fw.offset((-(NB_ELTS_V8 as i32) + c) as isize));
        *fresh47 += (*fl.offset((0 * NB_ELTS_V8 as i32 + c) as isize)
          + *fw.offset((0 * NB_ELTS_V8 as i32 + c) as isize))
          * cst;
        c += 1
      }
      fw = fw.offset((2 * NB_ELTS_V8 as i32) as isize);
      i = 1 as OPJ_UINT32;
      while i < imax {
        c = 0;
        while c < NB_ELTS_V8 as i32 {
          let fresh48 = &mut (*fw.offset((-(NB_ELTS_V8 as i32) + c) as isize));
          *fresh48 += (*fw.offset((-2 * NB_ELTS_V8 as i32 + c) as isize)
            + *fw.offset((0 * NB_ELTS_V8 as i32 + c) as isize))
            * cst;
          c += 1
        }
        fw = fw.offset((2 * NB_ELTS_V8 as i32) as isize);
        i += 1;
      }
    }
    if m < end {
      assert!(m.wrapping_add(1u32) == end);
      c = 0;
      while c < NB_ELTS_V8 as i32 {
        let fresh49 = &mut (*fw.offset((-(NB_ELTS_V8 as i32) + c) as isize));
        *fresh49 +=
          2 as core::ffi::c_float * *fw.offset((-2 * NB_ELTS_V8 as i32 + c) as isize) * cst;
        c += 1
      }
    };
  }
}

/* Forward 9-7 transform, for the vertical pass, processing cols columns */
/* where cols <= NB_ELTS_V8 */
unsafe extern "C" fn opj_dwt_encode_and_deinterleave_v_real(
  mut arrayIn: *mut core::ffi::c_void,
  mut tmpIn: *mut core::ffi::c_void,
  mut height: OPJ_UINT32,
  mut even: OPJ_BOOL,
  mut stride_width: OPJ_UINT32,
  mut cols: OPJ_UINT32,
) {
  let mut array = arrayIn as *mut OPJ_FLOAT32;
  let mut tmp = tmpIn as *mut OPJ_FLOAT32;
  let sn = (height.wrapping_add(if even != 0 { 1 } else { 0 }) >> 1i32) as OPJ_INT32;
  let dn = height.wrapping_sub(sn as OPJ_UINT32) as OPJ_INT32;
  let mut a: OPJ_INT32 = 0;
  let mut b: OPJ_INT32 = 0;
  if height == 1u32 {
    return;
  }
  opj_dwt_fetch_cols_vertical_pass(arrayIn, tmpIn, height, stride_width, cols);
  if even != 0 {
    a = 0i32;
    b = 1i32
  } else {
    a = 1i32;
    b = 0i32
  }
  opj_v8dwt_encode_step2(
    tmp.offset((a * NB_ELTS_V8 as i32) as isize),
    tmp.offset(((b + 1i32) * NB_ELTS_V8 as i32) as isize),
    dn as OPJ_UINT32,
    opj_int_min(dn, sn - b) as OPJ_UINT32,
    opj_dwt_alpha,
  );
  opj_v8dwt_encode_step2(
    tmp.offset((b * NB_ELTS_V8 as i32) as isize),
    tmp.offset(((a + 1i32) * NB_ELTS_V8 as i32) as isize),
    sn as OPJ_UINT32,
    opj_int_min(sn, dn - a) as OPJ_UINT32,
    opj_dwt_beta,
  );
  opj_v8dwt_encode_step2(
    tmp.offset((a * NB_ELTS_V8 as i32) as isize),
    tmp.offset(((b + 1i32) * NB_ELTS_V8 as i32) as isize),
    dn as OPJ_UINT32,
    opj_int_min(dn, sn - b) as OPJ_UINT32,
    opj_dwt_gamma,
  );
  opj_v8dwt_encode_step2(
    tmp.offset((b * NB_ELTS_V8 as i32) as isize),
    tmp.offset(((a + 1i32) * NB_ELTS_V8 as i32) as isize),
    sn as OPJ_UINT32,
    opj_int_min(sn, dn - a) as OPJ_UINT32,
    opj_dwt_delta,
  );
  opj_v8dwt_encode_step1(
    tmp.offset((b * NB_ELTS_V8 as i32) as isize),
    dn as OPJ_UINT32,
    opj_K,
  );
  opj_v8dwt_encode_step1(
    tmp.offset((a * NB_ELTS_V8 as i32) as isize),
    sn as OPJ_UINT32,
    opj_invK,
  );
  if cols == NB_ELTS_V8 {
    opj_dwt_deinterleave_v_cols(
      tmp as *mut OPJ_INT32,
      array as *mut OPJ_INT32,
      dn,
      sn,
      stride_width,
      if even != 0 { 0i32 } else { 1i32 },
      NB_ELTS_V8,
    );
  } else {
    opj_dwt_deinterleave_v_cols(
      tmp as *mut OPJ_INT32,
      array as *mut OPJ_INT32,
      dn,
      sn,
      stride_width,
      if even != 0 { 0i32 } else { 1i32 },
      cols,
    );
  };
}

/* <summary>                            */
/* Forward 5-3 wavelet transform in 2-D. */
/* </summary>                           */
#[inline]
fn opj_dwt_encode_procedure(
  mut tilec: *mut opj_tcd_tilecomp_t,
  mut p_encode_and_deinterleave_v: opj_encode_and_deinterleave_v_fnptr_type,
  mut p_encode_and_deinterleave_h_one_row: opj_encode_and_deinterleave_h_one_row_fnptr_type,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_INT32 = 0;
    let mut bj = core::ptr::null_mut::<OPJ_INT32>();
    let mut w: OPJ_UINT32 = 0;
    let mut l: OPJ_INT32 = 0;
    let mut l_data_size: OPJ_SIZE_T = 0;
    let mut l_cur_res = core::ptr::null_mut::<opj_tcd_resolution_t>();
    let mut l_last_res = core::ptr::null_mut::<opj_tcd_resolution_t>();
    let mut tiledp = (*tilec).data;
    w = ((*tilec).x1 - (*tilec).x0) as OPJ_UINT32;
    l = (*tilec).numresolutions as OPJ_INT32 - 1i32;
    l_cur_res = (*tilec).resolutions.offset(l as isize);
    l_last_res = l_cur_res.offset(-1);
    l_data_size =
      opj_dwt_max_resolution((*tilec).resolutions, (*tilec).numresolutions) as OPJ_SIZE_T;
    /* overflow check */
    if l_data_size
      > (usize::MAX).wrapping_div(NB_ELTS_V8 as usize * core::mem::size_of::<OPJ_INT32>())
    {
      /* FIXME event manager error callback */
      return 0i32;
    }
    l_data_size *= NB_ELTS_V8 as usize * core::mem::size_of::<OPJ_INT32>();
    let layout = Layout::from_size_align_unchecked(l_data_size, 32);
    bj = alloc(layout) as *mut OPJ_INT32;
    /* l_data_size is equal to 0 when numresolutions == 1 but bj is not used */
    /* in that case, so do not error out */
    if l_data_size != 0 && bj.is_null() {
      return 0i32;
    } /* width of the resolution level computed   */
    i = l; /* height of the resolution level computed  */
    loop {
      let fresh50 = i; /* width of the resolution level once lower than computed one                                       */
      i -= 1; /* height of the resolution level once lower than computed one                                      */
      if fresh50 == 0 {
        break; /* 0 = non inversion on horizontal filtering 1 = inversion between low-pass and high-pass filtering */
      } /* 0 = non inversion on vertical filtering 1 = inversion between low-pass and high-pass filtering   */
      let mut j: OPJ_UINT32 = 0;
      let mut rw: OPJ_UINT32 = 0;
      let mut rh: OPJ_UINT32 = 0;
      let mut cas_col: OPJ_INT32 = 0;
      let mut cas_row: OPJ_INT32 = 0;
      rw = ((*l_cur_res).x1 - (*l_cur_res).x0) as OPJ_UINT32;
      rh = ((*l_cur_res).y1 - (*l_cur_res).y0) as OPJ_UINT32;
      cas_row = (*l_cur_res).x0 & 1i32;
      cas_col = (*l_cur_res).y0 & 1i32;
      /* Perform vertical pass */
      j = 0 as OPJ_UINT32;
      while j + NB_ELTS_V8 - 1 < rw {
        p_encode_and_deinterleave_v.expect("non-null function pointer")(
          tiledp.offset(j as isize) as *mut core::ffi::c_void,
          bj as *mut core::ffi::c_void,
          rh,
          (cas_col == 0i32) as core::ffi::c_int,
          w,
          NB_ELTS_V8,
        );
        j += NB_ELTS_V8;
      }
      if j < rw {
        p_encode_and_deinterleave_v.expect("non-null function pointer")(
          tiledp.offset(j as isize) as *mut core::ffi::c_void,
          bj as *mut core::ffi::c_void,
          rh,
          (cas_col == 0i32) as core::ffi::c_int,
          w,
          rw.wrapping_sub(j),
        );
      }
      /* Perform horizontal pass */
      j = 0 as OPJ_UINT32; /* this can overflow */
      while j < rh {
        let mut aj = tiledp.offset(j.wrapping_mul(w) as isize);
        p_encode_and_deinterleave_h_one_row.expect("non-null function pointer")(
          aj as *mut core::ffi::c_void,
          bj as *mut core::ffi::c_void,
          rw,
          if cas_row == 0i32 { 1i32 } else { 0i32 },
        );
        j += 1;
      }
      l_cur_res = l_last_res;
      l_last_res = l_last_res.offset(-1)
    }
    dealloc(bj as _, layout);
    1i32
  }
}

/* Forward 5-3 wavelet transform in 2-D. */
/* </summary>                           */

pub(crate) fn opj_dwt_encode(mut tilec: *mut opj_tcd_tilecomp_t) -> OPJ_BOOL {
  opj_dwt_encode_procedure(
    tilec,
    Some(
      opj_dwt_encode_and_deinterleave_v
        as unsafe extern "C" fn(
          _: *mut core::ffi::c_void,
          _: *mut core::ffi::c_void,
          _: OPJ_UINT32,
          _: OPJ_BOOL,
          _: OPJ_UINT32,
          _: OPJ_UINT32,
        ) -> (),
    ),
    Some(
      opj_dwt_encode_and_deinterleave_h_one_row
        as unsafe extern "C" fn(
          _: *mut core::ffi::c_void,
          _: *mut core::ffi::c_void,
          _: OPJ_UINT32,
          _: OPJ_BOOL,
        ) -> (),
    ),
  )
}
/* <summary>                            */
/* Inverse 5-3 wavelet transform in 2-D. */
/* </summary>                           */

pub(crate) fn opj_dwt_decode(
  p_tcd: &opj_tcd,
  mut tilec: *mut opj_tcd_tilecomp_t,
  mut numres: OPJ_UINT32,
) -> OPJ_BOOL {
  if p_tcd.whole_tile_decoding != 0 {
    opj_dwt_decode_tile(tilec, numres)
  } else {
    opj_dwt_decode_partial_tile(tilec, numres)
  }
}

/* <summary>                */
/* Get norm of 5-3 wavelet. */
/* </summary>               */

pub fn opj_dwt_getnorm(mut level: OPJ_UINT32, orient: OPJ_UINT32) -> OPJ_FLOAT64 {
  /* FIXME ! This is just a band-aid to avoid a buffer overflow */
  /* but the array should really be extended up to 33 resolution levels */
  /* See https://github.com/uclouvain/openjpeg/issues/493 */
  if orient == 0 && level >= 10 {
    level = 9
  } else if orient > 0 && level >= 9 {
    level = 8
  }
  opj_dwt_norms[orient as usize][level as usize]
}

/* <summary>                             */
/* Forward 9-7 wavelet transform in 2-D. */
/* </summary>                            */

pub(crate) fn opj_dwt_encode_real(mut tilec: *mut opj_tcd_tilecomp_t) -> OPJ_BOOL {
  opj_dwt_encode_procedure(
    tilec,
    Some(
      opj_dwt_encode_and_deinterleave_v_real
        as unsafe extern "C" fn(
          _: *mut core::ffi::c_void,
          _: *mut core::ffi::c_void,
          _: OPJ_UINT32,
          _: OPJ_BOOL,
          _: OPJ_UINT32,
          _: OPJ_UINT32,
        ) -> (),
    ),
    Some(
      opj_dwt_encode_and_deinterleave_h_one_row_real
        as unsafe extern "C" fn(
          _: *mut core::ffi::c_void,
          _: *mut core::ffi::c_void,
          _: OPJ_UINT32,
          _: OPJ_BOOL,
        ) -> (),
    ),
  )
}

/* <summary>                */
/* Get norm of 9-7 wavelet. */
/* </summary>               */

pub fn opj_dwt_getnorm_real(mut level: OPJ_UINT32, orient: OPJ_UINT32) -> OPJ_FLOAT64 {
  /* FIXME ! This is just a band-aid to avoid a buffer overflow */
  /* but the array should really be extended up to 33 resolution levels */
  /* See https://github.com/uclouvain/openjpeg/issues/493 */
  if orient == 0 && level >= 10 {
    level = 9
  } else if orient > 0 && level >= 9 {
    level = 8
  }
  opj_dwt_norms_real[orient as usize][level as usize]
}

pub(crate) fn opj_dwt_calc_explicit_stepsizes(mut tccp: *mut opj_tccp_t, mut prec: OPJ_UINT32) {
  unsafe {
    let mut numbands: OPJ_UINT32 = 0;
    let mut bandno: OPJ_UINT32 = 0;
    numbands = (3u32)
      .wrapping_mul((*tccp).numresolutions)
      .wrapping_sub(2u32);
    bandno = 0 as OPJ_UINT32;
    while bandno < numbands {
      let mut stepsize: OPJ_FLOAT64 = 0.;
      let mut resno: OPJ_UINT32 = 0;
      let mut level: OPJ_UINT32 = 0;
      let mut orient: OPJ_UINT32 = 0;
      let mut gain: OPJ_UINT32 = 0;
      resno = if bandno == 0u32 {
        0u32
      } else {
        bandno
          .wrapping_sub(1u32)
          .wrapping_div(3u32)
          .wrapping_add(1u32)
      };
      orient = if bandno == 0u32 {
        0u32
      } else {
        bandno
          .wrapping_sub(1u32)
          .wrapping_rem(3u32)
          .wrapping_add(1u32)
      };
      level = (*tccp)
        .numresolutions
        .wrapping_sub(1u32)
        .wrapping_sub(resno);
      gain = if (*tccp).qmfbid == 0u32 {
        0i32
      } else if orient == 0u32 {
        0i32
      } else if orient == 1u32 || orient == 2u32 {
        1i32
      } else {
        2i32
      } as OPJ_UINT32;
      if (*tccp).qntsty == 0u32 {
        stepsize = 1.0f64
      } else {
        let mut norm = opj_dwt_getnorm_real(level, orient);
        stepsize = ((1i32) << gain) as core::ffi::c_double / norm
      }
      opj_dwt_encode_stepsize(
        floor(stepsize * 8192.0f64) as OPJ_INT32,
        prec.wrapping_add(gain) as OPJ_INT32,
        &mut *(*tccp).stepsizes.as_mut_ptr().offset(bandno as isize),
      );
      bandno += 1;
    }
  }
}
/* <summary>                             */
/* Determine maximum computed resolution level for inverse wavelet transform */
/* </summary>                            */
fn opj_dwt_max_resolution(mut r: *mut opj_tcd_resolution_t, mut i: OPJ_UINT32) -> OPJ_UINT32 {
  unsafe {
    let mut mr = 0 as OPJ_UINT32;
    let mut w: OPJ_UINT32 = 0;
    loop {
      i = i.wrapping_sub(1);
      if i == 0 {
        break;
      }
      r = r.offset(1);
      w = ((*r).x1 - (*r).x0) as OPJ_UINT32;
      if mr < w {
        mr = w
      }
      w = ((*r).y1 - (*r).y0) as OPJ_UINT32;
      if mr < w {
        mr = w
      }
    }
    mr
  }
}

/* <summary>                            */
/* Inverse wavelet transform in 2-D.    */
/* </summary>                           */
fn opj_dwt_decode_tile(mut tilec: *mut opj_tcd_tilecomp_t, mut numres: OPJ_UINT32) -> OPJ_BOOL {
  unsafe {
    let mut h = opj_dwt_t {
      mem: core::ptr::null_mut::<OPJ_INT32>(),
      dn: 0,
      sn: 0,
      cas: 0,
    }; /* width of the resolution level computed */
    let mut v = opj_dwt_t {
      mem: core::ptr::null_mut::<OPJ_INT32>(),
      dn: 0,
      sn: 0,
      cas: 0,
    }; /* height of the resolution level computed */
    let mut tr = (*tilec).resolutions;
    let mut rw = ((*tr).x1 - (*tr).x0) as OPJ_UINT32;
    let mut rh = ((*tr).y1 - (*tr).y0) as OPJ_UINT32;
    let mut w = ((*(*tilec)
      .resolutions
      .offset((*tilec).minimum_num_resolutions.wrapping_sub(1u32) as isize))
    .x1
      - (*(*tilec)
        .resolutions
        .offset((*tilec).minimum_num_resolutions.wrapping_sub(1u32) as isize))
      .x0) as OPJ_UINT32;
    let mut h_mem_size: OPJ_SIZE_T = 0;

    // Not entirely sure for the return code of w == 0 which is triggered per
    // https://github.com/uclouvain/openjpeg/issues/1505
    if numres == 1 || w == 0 {
      return 1i32;
    }
    h_mem_size = opj_dwt_max_resolution(tr, numres) as OPJ_SIZE_T;
    /* overflow check */
    if h_mem_size > (usize::MAX / PARALLEL_COLS_53 / core::mem::size_of::<OPJ_INT32>()) {
      /* FIXME event manager error callback */
      return 0i32;
    }
    /* We need PARALLEL_COLS_53 times the height of the array, */
    /* since for the vertical pass */
    /* we process PARALLEL_COLS_53 columns at a time */
    h_mem_size *= PARALLEL_COLS_53 * core::mem::size_of::<OPJ_INT32>();
    let layout = Layout::from_size_align_unchecked(h_mem_size, 32);
    h.mem = alloc(layout) as *mut OPJ_INT32;
    if h.mem.is_null() {
      /* FIXME event manager error callback */
      return 0i32;
    }
    v.mem = h.mem;
    loop {
      numres = numres.wrapping_sub(1);
      if numres == 0 {
        break;
      }
      let mut tiledp = (*tilec).data;
      let mut j: OPJ_UINT32 = 0;
      tr = tr.offset(1);
      h.sn = rw as OPJ_INT32;
      v.sn = rh as OPJ_INT32;
      rw = ((*tr).x1 - (*tr).x0) as OPJ_UINT32;
      rh = ((*tr).y1 - (*tr).y0) as OPJ_UINT32;
      h.dn = rw.wrapping_sub(h.sn as OPJ_UINT32) as OPJ_INT32;
      h.cas = (*tr).x0 % 2i32;
      j = 0 as OPJ_UINT32;
      while j < rh {
        opj_idwt53_h(
          &h,
          &mut *tiledp.add((j as OPJ_SIZE_T).wrapping_mul(w as usize)),
        );
        j += 1;
      }
      v.dn = rh.wrapping_sub(v.sn as OPJ_UINT32) as OPJ_INT32;
      v.cas = (*tr).y0 % 2i32;
      j = 0 as OPJ_UINT32;
      while j.wrapping_add((2i32 * 4i32) as core::ffi::c_uint) <= rw {
        opj_idwt53_v(
          &v,
          &mut *tiledp.offset(j as isize),
          w as OPJ_SIZE_T,
          2i32 * 4i32,
        );
        j = (j as core::ffi::c_uint).wrapping_add((2i32 * 4i32) as core::ffi::c_uint) as OPJ_UINT32
      }
      if j < rw {
        opj_idwt53_v(
          &v,
          &mut *tiledp.offset(j as isize),
          w as OPJ_SIZE_T,
          rw.wrapping_sub(j) as OPJ_INT32,
        );
      }
    }
    dealloc(h.mem as _, layout);
    1i32
  }
}
fn opj_dwt_interleave_partial_h(
  dest: *mut OPJ_INT32,
  cas: OPJ_INT32,
  sa: &SparseArray,
  sa_line: OPJ_UINT32,
  sn: OPJ_UINT32,
  win_l_x0: OPJ_UINT32,
  win_l_x1: OPJ_UINT32,
  win_h_x0: OPJ_UINT32,
  win_h_x1: OPJ_UINT32,
) {
  unsafe {
    let ret = sparse_array_read(
      sa,
      win_l_x0,
      sa_line,
      win_l_x1,
      sa_line.wrapping_add(1u32),
      dest
        .offset(cas as isize)
        .offset((2u32).wrapping_mul(win_l_x0) as isize),
      2 as OPJ_UINT32,
      0 as OPJ_UINT32,
      true,
    );
    assert!(ret);
    let ret = sparse_array_read(
      sa,
      sn.wrapping_add(win_h_x0),
      sa_line,
      sn.wrapping_add(win_h_x1),
      sa_line.wrapping_add(1u32),
      dest
        .offset(1)
        .offset(-(cas as isize))
        .offset((2u32).wrapping_mul(win_h_x0) as isize),
      2 as OPJ_UINT32,
      0 as OPJ_UINT32,
      true,
    );
    assert!(ret);
  }
}
fn opj_dwt_interleave_partial_v(
  dest: *mut OPJ_INT32,
  cas: OPJ_INT32,
  sa: &SparseArray,
  sa_col: OPJ_UINT32,
  nb_cols: OPJ_UINT32,
  sn: OPJ_UINT32,
  win_l_y0: OPJ_UINT32,
  win_l_y1: OPJ_UINT32,
  win_h_y0: OPJ_UINT32,
  win_h_y1: OPJ_UINT32,
) {
  unsafe {
    let ret = sparse_array_read(
      sa,
      sa_col,
      win_l_y0,
      sa_col.wrapping_add(nb_cols),
      win_l_y1,
      dest
        .offset((cas * 4i32) as isize)
        .offset(((2i32 * 4i32) as core::ffi::c_uint).wrapping_mul(win_l_y0) as isize),
      1 as OPJ_UINT32,
      (2i32 * 4i32) as OPJ_UINT32,
      true,
    );
    assert!(ret);
    let ret = sparse_array_read(
      sa,
      sa_col,
      sn.wrapping_add(win_h_y0),
      sa_col.wrapping_add(nb_cols),
      sn.wrapping_add(win_h_y1),
      dest
        .offset(((1i32 - cas) * 4i32) as isize)
        .offset(((2i32 * 4i32) as core::ffi::c_uint).wrapping_mul(win_h_y0) as isize),
      1 as OPJ_UINT32,
      (2i32 * 4i32) as OPJ_UINT32,
      true,
    );
    assert!(ret);
  }
}

fn opj_dwt_get_band_coordinates(
  mut tilec: *mut opj_tcd_tilecomp_t,
  mut resno: OPJ_UINT32,
  mut bandno: OPJ_UINT32,
  mut tcx0: OPJ_UINT32,
  mut tcy0: OPJ_UINT32,
  mut tcx1: OPJ_UINT32,
  mut tcy1: OPJ_UINT32,
  mut tbx0: *mut OPJ_UINT32,
  mut tby0: *mut OPJ_UINT32,
  mut tbx1: *mut OPJ_UINT32,
  mut tby1: *mut OPJ_UINT32,
) {
  unsafe {
    /* Compute number of decomposition for this band. See table F-1 */
    let mut nb = if resno == 0u32 {
      (*tilec).numresolutions.wrapping_sub(1u32)
    } else {
      (*tilec).numresolutions.wrapping_sub(resno)
    };
    /* Map above tile-based coordinates to sub-band-based coordinates per */
    /* equation B-15 of the standard */
    let mut x0b = bandno & 1u32;
    let mut y0b = bandno >> 1i32;
    if !tbx0.is_null() {
      *tbx0 = if nb == 0u32 {
        tcx0
      } else if tcx0 <= ((1u32) << nb.wrapping_sub(1u32)).wrapping_mul(x0b) {
        0u32
      } else {
        opj_uint_ceildivpow2(
          tcx0.wrapping_sub(((1u32) << nb.wrapping_sub(1u32)).wrapping_mul(x0b)),
          nb,
        )
      }
    }
    if !tby0.is_null() {
      *tby0 = if nb == 0u32 {
        tcy0
      } else if tcy0 <= ((1u32) << nb.wrapping_sub(1u32)).wrapping_mul(y0b) {
        0u32
      } else {
        opj_uint_ceildivpow2(
          tcy0.wrapping_sub(((1u32) << nb.wrapping_sub(1u32)).wrapping_mul(y0b)),
          nb,
        )
      }
    }
    if !tbx1.is_null() {
      *tbx1 = if nb == 0u32 {
        tcx1
      } else if tcx1 <= ((1u32) << nb.wrapping_sub(1u32)).wrapping_mul(x0b) {
        0u32
      } else {
        opj_uint_ceildivpow2(
          tcx1.wrapping_sub(((1u32) << nb.wrapping_sub(1u32)).wrapping_mul(x0b)),
          nb,
        )
      }
    }
    if !tby1.is_null() {
      *tby1 = if nb == 0u32 {
        tcy1
      } else if tcy1 <= ((1u32) << nb.wrapping_sub(1u32)).wrapping_mul(y0b) {
        0u32
      } else {
        opj_uint_ceildivpow2(
          tcy1.wrapping_sub(((1u32) << nb.wrapping_sub(1u32)).wrapping_mul(y0b)),
          nb,
        )
      }
    };
  }
}
fn opj_dwt_segment_grow(
  mut filter_width: OPJ_UINT32,
  mut max_size: OPJ_UINT32,
  mut start: *mut OPJ_UINT32,
  mut end: *mut OPJ_UINT32,
) {
  unsafe {
    *start = opj_uint_subs(*start, filter_width);
    *end = opj_uint_adds(*end, filter_width);
    *end = opj_uint_min(*end, max_size);
  }
}
fn opj_dwt_init_sparse_array(
  mut tilec: *mut opj_tcd_tilecomp_t,
  mut numres: OPJ_UINT32,
) -> Option<SparseArray> {
  unsafe {
    let mut tr_max: *mut opj_tcd_resolution_t = &mut *(*tilec)
      .resolutions
      .offset(numres.wrapping_sub(1u32) as isize)
      as *mut opj_tcd_resolution_t;
    let mut w = ((*tr_max).x1 - (*tr_max).x0) as OPJ_UINT32;
    let mut h = ((*tr_max).y1 - (*tr_max).y0) as OPJ_UINT32;
    let mut resno: OPJ_UINT32 = 0;
    let mut bandno: OPJ_UINT32 = 0;
    let mut precno: OPJ_UINT32 = 0;
    let mut cblkno: OPJ_UINT32 = 0;
    let mut sa = SparseArray::new(
      w,
      h,
      opj_uint_min(w, 64 as OPJ_UINT32),
      opj_uint_min(h, 64 as OPJ_UINT32),
    )?;
    resno = 0 as OPJ_UINT32;
    while resno < numres {
      let mut res: *mut opj_tcd_resolution_t =
        &mut *(*tilec).resolutions.offset(resno as isize) as *mut opj_tcd_resolution_t;
      bandno = 0 as OPJ_UINT32;
      while bandno < (*res).numbands {
        let mut band: *mut opj_tcd_band_t =
          &mut *(*res).bands.as_mut_ptr().offset(bandno as isize) as *mut opj_tcd_band_t;
        precno = 0 as OPJ_UINT32;
        while precno < (*res).pw.wrapping_mul((*res).ph) {
          let mut precinct: *mut opj_tcd_precinct_t =
            &mut *(*band).precincts.offset(precno as isize) as *mut opj_tcd_precinct_t;
          cblkno = 0 as OPJ_UINT32;
          while cblkno < (*precinct).cw.wrapping_mul((*precinct).ch) {
            let mut cblk: *mut opj_tcd_cblk_dec_t =
              &mut *(*precinct).cblks.dec.offset(cblkno as isize) as *mut opj_tcd_cblk_dec_t;
            if !(*cblk).decoded_data.is_null() {
              let mut x = ((*cblk).x0 - (*band).x0) as OPJ_UINT32;
              let mut y = ((*cblk).y0 - (*band).y0) as OPJ_UINT32;
              let mut cblk_w = ((*cblk).x1 - (*cblk).x0) as OPJ_UINT32;
              let mut cblk_h = ((*cblk).y1 - (*cblk).y0) as OPJ_UINT32;
              if (*band).bandno & 1u32 != 0 {
                let mut pres: *mut opj_tcd_resolution_t = &mut *(*tilec)
                  .resolutions
                  .offset(resno.wrapping_sub(1u32) as isize)
                  as *mut opj_tcd_resolution_t;
                x = x.wrapping_add(((*pres).x1 - (*pres).x0) as OPJ_UINT32) as OPJ_UINT32
              }
              if (*band).bandno & 2u32 != 0 {
                let mut pres_0: *mut opj_tcd_resolution_t = &mut *(*tilec)
                  .resolutions
                  .offset(resno.wrapping_sub(1u32) as isize)
                  as *mut opj_tcd_resolution_t;
                y = (y as core::ffi::c_uint)
                  .wrapping_add(((*pres_0).y1 - (*pres_0).y0) as OPJ_UINT32)
                  as OPJ_UINT32
              }
              if !sparse_array_write(
                &mut sa,
                x,
                y,
                x.wrapping_add(cblk_w),
                y.wrapping_add(cblk_h),
                (*cblk).decoded_data,
                1 as OPJ_UINT32,
                cblk_w,
                true,
              ) {
                return None;
              }
            }
            cblkno += 1;
          }
          precno += 1;
        }
        bandno += 1;
      }
      resno += 1;
    }
    Some(sa)
  }
}

fn opj_dwt_decode_partial_tile(
  mut tilec: *mut opj_tcd_tilecomp_t,
  mut numres: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut h = opj_dwt_t {
      mem: core::ptr::null_mut::<OPJ_INT32>(),
      dn: 0,
      sn: 0,
      cas: 0,
    };
    let mut v = opj_dwt_t {
      mem: core::ptr::null_mut::<OPJ_INT32>(),
      dn: 0,
      sn: 0,
      cas: 0,
    };
    let mut resno: OPJ_UINT32 = 0;
    /* This value matches the maximum left/right extension given in tables */
    /* F.2 and F.3 of the standard. */
    let filter_width = 2u32; /* width of the resolution level computed */
    let mut tr = (*tilec).resolutions; /* height of the resolution level computed */
    let mut tr_max: *mut opj_tcd_resolution_t = &mut *(*tilec)
      .resolutions
      .offset(numres.wrapping_sub(1u32) as isize)
      as *mut opj_tcd_resolution_t;
    let mut rw = ((*tr).x1 - (*tr).x0) as OPJ_UINT32;
    let mut rh = ((*tr).y1 - (*tr).y0) as OPJ_UINT32;
    let mut h_mem_size: OPJ_SIZE_T = 0;
    /* Compute the intersection of the area of interest, expressed in tile coordinates */
    /* with the tile coordinates */
    let mut win_tcx0 = (*tilec).win_x0;
    let mut win_tcy0 = (*tilec).win_y0;
    let mut win_tcx1 = (*tilec).win_x1;
    let mut win_tcy1 = (*tilec).win_y1;
    if (*tr_max).x0 == (*tr_max).x1 || (*tr_max).y0 == (*tr_max).y1 {
      return 1i32;
    }
    let mut sa = if let Some(sa) = opj_dwt_init_sparse_array(tilec, numres) {
      sa
    } else {
      return 0i32;
    };
    if numres == 1u32 {
      let mut ret = sparse_array_read(
        &sa,
        (*tr_max).win_x0.wrapping_sub((*tr_max).x0 as OPJ_UINT32),
        (*tr_max).win_y0.wrapping_sub((*tr_max).y0 as OPJ_UINT32),
        (*tr_max).win_x1.wrapping_sub((*tr_max).x0 as OPJ_UINT32),
        (*tr_max).win_y1.wrapping_sub((*tr_max).y0 as OPJ_UINT32),
        (*tilec).data_win,
        1 as OPJ_UINT32,
        (*tr_max).win_x1.wrapping_sub((*tr_max).win_x0),
        true,
      );
      assert!(ret);
      return 1i32;
    }
    h_mem_size = opj_dwt_max_resolution(tr, numres) as OPJ_SIZE_T;
    /* overflow check */
    /* in vertical pass, we process 4 columns at a time */
    if h_mem_size > (usize::MAX / 4 * core::mem::size_of::<OPJ_INT32>()) {
      /* FIXME event manager error callback */
      return 0i32;
    }
    h_mem_size *= 4 * core::mem::size_of::<OPJ_INT32>();
    let layout = Layout::from_size_align_unchecked(h_mem_size, 32);
    h.mem = alloc(layout) as *mut OPJ_INT32;
    if h.mem.is_null() {
      /* FIXME event manager error callback */
      return 0i32;
    }
    v.mem = h.mem;
    resno = 1 as OPJ_UINT32;
    while resno < numres {
      let mut i: OPJ_UINT32 = 0;
      let mut j: OPJ_UINT32 = 0;
      /* Window of interest subband-based coordinates */
      let mut win_ll_x0: OPJ_UINT32 = 0;
      let mut win_ll_y0: OPJ_UINT32 = 0;
      let mut win_ll_x1: OPJ_UINT32 = 0;
      let mut win_ll_y1: OPJ_UINT32 = 0;
      let mut win_hl_x0: OPJ_UINT32 = 0;
      let mut win_hl_x1: OPJ_UINT32 = 0;
      let mut win_lh_y0: OPJ_UINT32 = 0;
      let mut win_lh_y1: OPJ_UINT32 = 0;
      /* Window of interest tile-resolution-based coordinates */
      let mut win_tr_x0: OPJ_UINT32 = 0;
      let mut win_tr_x1: OPJ_UINT32 = 0;
      let mut win_tr_y0: OPJ_UINT32 = 0;
      let mut win_tr_y1: OPJ_UINT32 = 0;
      /* Tile-resolution subband-based coordinates */
      let mut tr_ll_x0: OPJ_UINT32 = 0;
      let mut tr_ll_y0: OPJ_UINT32 = 0;
      let mut tr_hl_x0: OPJ_UINT32 = 0;
      let mut tr_lh_y0: OPJ_UINT32 = 0;
      tr = tr.offset(1);
      h.sn = rw as OPJ_INT32;
      v.sn = rh as OPJ_INT32;
      rw = ((*tr).x1 - (*tr).x0) as OPJ_UINT32;
      rh = ((*tr).y1 - (*tr).y0) as OPJ_UINT32;
      h.dn = rw.wrapping_sub(h.sn as OPJ_UINT32) as OPJ_INT32;
      h.cas = (*tr).x0 % 2i32;
      v.dn = rh.wrapping_sub(v.sn as OPJ_UINT32) as OPJ_INT32;
      v.cas = (*tr).y0 % 2i32;
      /* Get the subband coordinates for the window of interest */
      /* LL band */
      opj_dwt_get_band_coordinates(
        tilec,
        resno,
        0 as OPJ_UINT32,
        win_tcx0,
        win_tcy0,
        win_tcx1,
        win_tcy1,
        &mut win_ll_x0,
        &mut win_ll_y0,
        &mut win_ll_x1,
        &mut win_ll_y1,
      );
      /* HL band */
      opj_dwt_get_band_coordinates(
        tilec,
        resno,
        1 as OPJ_UINT32,
        win_tcx0,
        win_tcy0,
        win_tcx1,
        win_tcy1,
        &mut win_hl_x0,
        core::ptr::null_mut::<OPJ_UINT32>(),
        &mut win_hl_x1,
        core::ptr::null_mut::<OPJ_UINT32>(),
      );
      /* LH band */
      opj_dwt_get_band_coordinates(
        tilec,
        resno,
        2 as OPJ_UINT32,
        win_tcx0,
        win_tcy0,
        win_tcx1,
        win_tcy1,
        core::ptr::null_mut::<OPJ_UINT32>(),
        &mut win_lh_y0,
        core::ptr::null_mut::<OPJ_UINT32>(),
        &mut win_lh_y1,
      );
      /* Beware: band index for non-LL0 resolution are 0=HL, 1=LH and 2=HH */
      tr_ll_x0 = (*tr).bands[1_usize].x0 as OPJ_UINT32;
      tr_ll_y0 = (*tr).bands[0_usize].y0 as OPJ_UINT32;
      tr_hl_x0 = (*tr).bands[0_usize].x0 as OPJ_UINT32;
      tr_lh_y0 = (*tr).bands[1_usize].y0 as OPJ_UINT32;
      /* Subtract the origin of the bands for this tile, to the subwindow */
      /* of interest band coordinates, so as to get them relative to the */
      /* tile */
      win_ll_x0 = opj_uint_subs(win_ll_x0, tr_ll_x0);
      win_ll_y0 = opj_uint_subs(win_ll_y0, tr_ll_y0);
      win_ll_x1 = opj_uint_subs(win_ll_x1, tr_ll_x0);
      win_ll_y1 = opj_uint_subs(win_ll_y1, tr_ll_y0);
      win_hl_x0 = opj_uint_subs(win_hl_x0, tr_hl_x0);
      win_hl_x1 = opj_uint_subs(win_hl_x1, tr_hl_x0);
      win_lh_y0 = opj_uint_subs(win_lh_y0, tr_lh_y0);
      win_lh_y1 = opj_uint_subs(win_lh_y1, tr_lh_y0);
      opj_dwt_segment_grow(
        filter_width,
        h.sn as OPJ_UINT32,
        &mut win_ll_x0,
        &mut win_ll_x1,
      );
      opj_dwt_segment_grow(
        filter_width,
        h.dn as OPJ_UINT32,
        &mut win_hl_x0,
        &mut win_hl_x1,
      );
      opj_dwt_segment_grow(
        filter_width,
        v.sn as OPJ_UINT32,
        &mut win_ll_y0,
        &mut win_ll_y1,
      );
      opj_dwt_segment_grow(
        filter_width,
        v.dn as OPJ_UINT32,
        &mut win_lh_y0,
        &mut win_lh_y1,
      );
      /* Compute the tile-resolution-based coordinates for the window of interest */
      if h.cas == 0i32 {
        win_tr_x0 = opj_uint_min(
          (2u32).wrapping_mul(win_ll_x0),
          (2u32).wrapping_mul(win_hl_x0).wrapping_add(1u32),
        );
        win_tr_x1 = opj_uint_min(
          opj_uint_max(
            (2u32).wrapping_mul(win_ll_x1),
            (2u32).wrapping_mul(win_hl_x1).wrapping_add(1u32),
          ),
          rw,
        )
      } else {
        win_tr_x0 = opj_uint_min(
          (2u32).wrapping_mul(win_hl_x0),
          (2u32).wrapping_mul(win_ll_x0).wrapping_add(1u32),
        );
        win_tr_x1 = opj_uint_min(
          opj_uint_max(
            (2u32).wrapping_mul(win_hl_x1),
            (2u32).wrapping_mul(win_ll_x1).wrapping_add(1u32),
          ),
          rw,
        )
      }
      if v.cas == 0i32 {
        win_tr_y0 = opj_uint_min(
          (2u32).wrapping_mul(win_ll_y0),
          (2u32).wrapping_mul(win_lh_y0).wrapping_add(1u32),
        );
        win_tr_y1 = opj_uint_min(
          opj_uint_max(
            (2u32).wrapping_mul(win_ll_y1),
            (2u32).wrapping_mul(win_lh_y1).wrapping_add(1u32),
          ),
          rh,
        )
      } else {
        win_tr_y0 = opj_uint_min(
          (2u32).wrapping_mul(win_lh_y0),
          (2u32).wrapping_mul(win_ll_y0).wrapping_add(1u32),
        );
        win_tr_y1 = opj_uint_min(
          opj_uint_max(
            (2u32).wrapping_mul(win_lh_y1),
            (2u32).wrapping_mul(win_ll_y1).wrapping_add(1u32),
          ),
          rh,
        )
      }
      j = 0 as OPJ_UINT32;
      while j < rh {
        if j >= win_ll_y0 && j < win_ll_y1
          || j >= win_lh_y0.wrapping_add(v.sn as OPJ_UINT32)
            && j < win_lh_y1.wrapping_add(v.sn as OPJ_UINT32)
        {
          /* Avoids dwt.c:1584:44 (in opj_dwt_decode_partial_1): runtime error: */
          /* signed integer overflow: -1094795586 + -1094795586 cannot be represented in type 'int' */
          /* on opj_decompress -i  ../../openjpeg/MAPA.jp2 -o out.tif -d 0,0,256,256 */
          /* This is less extreme than memsetting the whole buffer to 0 */
          /* although we could potentially do better with better handling of edge conditions */
          if win_tr_x1 >= 1u32 && win_tr_x1 < rw {
            *h.mem.offset(win_tr_x1.wrapping_sub(1u32) as isize) = 0i32
          }
          if win_tr_x1 < rw {
            *h.mem.offset(win_tr_x1 as isize) = 0i32
          }
          opj_dwt_interleave_partial_h(
            h.mem,
            h.cas,
            &sa,
            j,
            h.sn as OPJ_UINT32,
            win_ll_x0,
            win_ll_x1,
            win_hl_x0,
            win_hl_x1,
          );
          h.decode_partial_1(
            win_ll_x0 as OPJ_INT32,
            win_ll_x1 as OPJ_INT32,
            win_hl_x0 as OPJ_INT32,
            win_hl_x1 as OPJ_INT32,
          );
          if !sparse_array_write(
            &mut sa,
            win_tr_x0,
            j,
            win_tr_x1,
            j.wrapping_add(1u32),
            h.mem.offset(win_tr_x0 as isize),
            1 as OPJ_UINT32,
            0 as OPJ_UINT32,
            true,
          ) {
            /* FIXME event manager error callback */
            dealloc(h.mem as _, layout);
            return 0i32;
          }
        }
        j += 1;
      }
      i = win_tr_x0;
      while i < win_tr_x1 {
        let mut nb_cols = opj_uint_min(4u32, win_tr_x1.wrapping_sub(i));
        opj_dwt_interleave_partial_v(
          v.mem,
          v.cas,
          &sa,
          i,
          nb_cols,
          v.sn as OPJ_UINT32,
          win_ll_y0,
          win_ll_y1,
          win_lh_y0,
          win_lh_y1,
        );
        v.decode_partial_1_parallel(
          win_ll_y0 as OPJ_INT32,
          win_ll_y1 as OPJ_INT32,
          win_lh_y0 as OPJ_INT32,
          win_lh_y1 as OPJ_INT32,
        );
        if !sparse_array_write(
          &mut sa,
          i,
          win_tr_y0,
          i.wrapping_add(nb_cols),
          win_tr_y1,
          v.mem.offset((4u32).wrapping_mul(win_tr_y0) as isize),
          1 as OPJ_UINT32,
          4 as OPJ_UINT32,
          true,
        ) {
          /* FIXME event manager error callback */
          dealloc(h.mem as _, layout);
          return 0i32;
        }
        i = (i as core::ffi::c_uint).wrapping_add(nb_cols) as OPJ_UINT32
      }
      resno += 1;
    }
    dealloc(h.mem as _, layout);
    let mut ret_0 = sparse_array_read(
      &sa,
      (*tr_max).win_x0.wrapping_sub((*tr_max).x0 as OPJ_UINT32),
      (*tr_max).win_y0.wrapping_sub((*tr_max).y0 as OPJ_UINT32),
      (*tr_max).win_x1.wrapping_sub((*tr_max).x0 as OPJ_UINT32),
      (*tr_max).win_y1.wrapping_sub((*tr_max).y0 as OPJ_UINT32),
      (*tilec).data_win,
      1 as OPJ_UINT32,
      (*tr_max).win_x1.wrapping_sub((*tr_max).win_x0),
      true,
    );
    assert!(ret_0);
    1i32
  }
}

fn opj_v8dwt_interleave_h(
  mut dwt: &opj_v8dwt_t,
  mut a: *mut OPJ_FLOAT32,
  mut width: OPJ_UINT32,
  mut remaining_height: OPJ_UINT32,
) {
  unsafe {
    let mut bi = (*dwt).wavelet.offset((*dwt).cas as isize) as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut k: OPJ_UINT32 = 0;
    let mut x0 = (*dwt).win_l_x0;
    let mut x1 = (*dwt).win_l_x1;
    k = 0 as OPJ_UINT32;
    while k < 2u32 {
      if remaining_height >= NB_ELTS_V8 && a as OPJ_SIZE_T & 0xf == 0 && bi as OPJ_SIZE_T & 0xf == 0
      {
        /* Fast code path */
        i = x0;
        while i < x1 {
          let mut j = i;
          let mut dst = bi.offset((i * 2 * NB_ELTS_V8) as isize);
          *dst.offset(0) = *a.offset(j as isize);
          j += width;
          *dst.offset(1) = *a.offset(j as isize);
          j += width;
          *dst.offset(2) = *a.offset(j as isize);
          j += width;
          *dst.offset(3) = *a.offset(j as isize);
          j += width;
          *dst.offset(4) = *a.offset(j as isize);
          j += width;
          *dst.offset(5) = *a.offset(j as isize);
          j += width;
          *dst.offset(6) = *a.offset(j as isize);
          j += width;
          *dst.offset(7) = *a.offset(j as isize);
          i += 1;
        }
      } else {
        /* Slow code path */
        i = x0;
        while i < x1 {
          let mut j = i;
          let mut dst = bi.offset((i * 2 * NB_ELTS_V8) as isize);
          *dst.offset(0) = *a.offset(j as isize);
          j += width;
          if remaining_height != 1u32 {
            *dst.offset(1) = *a.offset(j as isize);
            j += width;
            if remaining_height != 2u32 {
              *dst.offset(2) = *a.offset(j as isize);
              j += width;
              if remaining_height != 3u32 {
                *dst.offset(3) = *a.offset(j as isize);
                j += width;
                if remaining_height != 4u32 {
                  *dst.offset(4) = *a.offset(j as isize);
                  j += width;
                  if remaining_height != 5u32 {
                    *dst.offset(5) = *a.offset(j as isize);
                    j += width;
                    if remaining_height != 6u32 {
                      *dst.offset(6) = *a.offset(j as isize);
                      j += width;
                      if remaining_height != 7u32 {
                        *dst.offset(7) = *a.offset(j as isize)
                      }
                    }
                  }
                }
              }
            }
          }
          i += 1;
        }
      }
      bi = (*dwt).wavelet.offset(1).offset(-((*dwt).cas as isize)) as *mut OPJ_FLOAT32;
      a = a.offset((*dwt).sn as isize);
      x0 = (*dwt).win_h_x0;
      x1 = (*dwt).win_h_x1;
      k += 1;
    }
  }
}

fn opj_v8dwt_interleave_partial_h(
  mut dwt: &opj_v8dwt_t,
  sa: &SparseArray,
  sa_line: OPJ_UINT32,
  remaining_height: OPJ_UINT32,
) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    i = 0 as OPJ_UINT32;
    while i < remaining_height {
      let ret = sparse_array_read(
        sa,
        (*dwt).win_l_x0,
        sa_line.wrapping_add(i),
        (*dwt).win_l_x1,
        sa_line.wrapping_add(i).wrapping_add(1u32),
        ((*dwt)
          .wavelet
          .offset((*dwt).cas as isize)
          .offset((2u32).wrapping_mul((*dwt).win_l_x0) as isize) as *mut OPJ_INT32)
          .offset(i as isize),
        2 * NB_ELTS_V8,
        0 as OPJ_UINT32,
        true,
      );
      assert!(ret);
      let ret = sparse_array_read(
        sa,
        ((*dwt).sn as OPJ_UINT32).wrapping_add((*dwt).win_h_x0),
        sa_line.wrapping_add(i),
        ((*dwt).sn as OPJ_UINT32).wrapping_add((*dwt).win_h_x1),
        sa_line.wrapping_add(i).wrapping_add(1u32),
        ((*dwt)
          .wavelet
          .offset(1)
          .offset(-((*dwt).cas as isize))
          .offset((2u32).wrapping_mul((*dwt).win_h_x0) as isize) as *mut OPJ_INT32)
          .offset(i as isize),
        2 * NB_ELTS_V8,
        0 as OPJ_UINT32,
        true,
      );
      assert!(ret);
      i += 1;
    }
  }
}

#[inline]
fn opj_v8dwt_interleave_v(
  mut dwt: &opj_v8dwt_t,
  mut a: *mut OPJ_FLOAT32,
  mut width: OPJ_UINT32,
  mut nb_elts_read: OPJ_UINT32,
) {
  unsafe {
    let mut bi = (*dwt).wavelet.offset((*dwt).cas as isize);
    let mut i: OPJ_UINT32 = 0;
    i = (*dwt).win_l_x0;
    while i < (*dwt).win_l_x1 {
      memcpy(
        &mut *bi.offset(i.wrapping_mul(2u32) as isize) as *mut opj_v8_t as *mut core::ffi::c_void,
        &mut *a.add((i as usize).wrapping_mul(width as OPJ_SIZE_T)) as *mut OPJ_FLOAT32
          as *const core::ffi::c_void,
        (nb_elts_read as OPJ_SIZE_T).wrapping_mul(core::mem::size_of::<OPJ_FLOAT32>()),
      );
      i += 1;
    }
    a = a.add(((*dwt).sn as OPJ_UINT32 as usize).wrapping_mul(width as OPJ_SIZE_T));
    bi = (*dwt).wavelet.offset(1).offset(-((*dwt).cas as isize));
    i = (*dwt).win_h_x0;
    while i < (*dwt).win_h_x1 {
      memcpy(
        &mut *bi.offset(i.wrapping_mul(2u32) as isize) as *mut opj_v8_t as *mut core::ffi::c_void,
        &mut *a.add((i as usize).wrapping_mul(width as OPJ_SIZE_T)) as *mut OPJ_FLOAT32
          as *const core::ffi::c_void,
        (nb_elts_read as OPJ_SIZE_T).wrapping_mul(core::mem::size_of::<OPJ_FLOAT32>()),
      );
      i += 1;
    }
  }
}
fn opj_v8dwt_interleave_partial_v(
  mut dwt: &opj_v8dwt_t,
  sa: &SparseArray,
  sa_col: OPJ_UINT32,
  nb_elts_read: OPJ_UINT32,
) {
  unsafe {
    let ret = sparse_array_read(
      sa,
      sa_col,
      (*dwt).win_l_x0,
      sa_col.wrapping_add(nb_elts_read),
      (*dwt).win_l_x1,
      (*dwt)
        .wavelet
        .offset((*dwt).cas as isize)
        .offset((2u32).wrapping_mul((*dwt).win_l_x0) as isize) as *mut OPJ_INT32,
      1 as OPJ_UINT32,
      2 * NB_ELTS_V8,
      true,
    );
    assert!(ret);
    let ret = sparse_array_read(
      sa,
      sa_col,
      ((*dwt).sn as OPJ_UINT32).wrapping_add((*dwt).win_h_x0),
      sa_col.wrapping_add(nb_elts_read),
      ((*dwt).sn as OPJ_UINT32).wrapping_add((*dwt).win_h_x1),
      (*dwt)
        .wavelet
        .offset(1)
        .offset(-((*dwt).cas as isize))
        .offset((2u32).wrapping_mul((*dwt).win_h_x0) as isize) as *mut OPJ_INT32,
      1 as OPJ_UINT32,
      2 * NB_ELTS_V8,
      true,
    );
    assert!(ret);
  }
}

fn opj_v8dwt_decode_step1(
  mut w: *mut opj_v8_t,
  mut start: OPJ_UINT32,
  mut end: OPJ_UINT32,
  c: OPJ_FLOAT32,
) {
  unsafe {
    let mut fw = w as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    /* To be adapted if NB_ELTS_V8 changes */
    i = start;
    while i < end {
      *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32) as isize) =
        *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32) as isize) * c;
      *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(1u32) as isize) =
        *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(1u32) as isize) * c;
      *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(2u32) as isize) =
        *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(2u32) as isize) * c;
      *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(3u32) as isize) =
        *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(3u32) as isize) * c;
      *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(4u32) as isize) =
        *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(4u32) as isize) * c;
      *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(5u32) as isize) =
        *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(5u32) as isize) * c;
      *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(6u32) as isize) =
        *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(6u32) as isize) * c;
      *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(7u32) as isize) =
        *fw.offset(i.wrapping_mul(2u32).wrapping_mul(8u32).wrapping_add(7u32) as isize) * c;
      i += 1;
    }
  }
}

fn opj_v8dwt_decode_step2(
  mut l: *mut opj_v8_t,
  mut w: *mut opj_v8_t,
  mut start: OPJ_UINT32,
  mut end: OPJ_UINT32,
  mut m: OPJ_UINT32,
  mut c: OPJ_FLOAT32,
) {
  unsafe {
    let mut fl = l as *mut OPJ_FLOAT32;
    let mut fw = w as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut imax = opj_uint_min(end, m);
    if start > 0u32 {
      fw = fw.offset((2 * NB_ELTS_V8 * start) as isize);
      fl = fw.offset(-((2 * NB_ELTS_V8) as isize))
    }
    /* To be adapted if NB_ELTS_V8 changes */
    i = start;
    while i < imax {
      *fw.offset(-(8i32) as isize) =
        *fw.offset(-(8i32) as isize) + (*fl.offset(0) + *fw.offset(0)) * c;
      *fw.offset(-(7i32) as isize) =
        *fw.offset(-(7i32) as isize) + (*fl.offset(1) + *fw.offset(1)) * c;
      *fw.offset(-(6i32) as isize) =
        *fw.offset(-(6i32) as isize) + (*fl.offset(2) + *fw.offset(2)) * c;
      *fw.offset(-(5i32) as isize) =
        *fw.offset(-(5i32) as isize) + (*fl.offset(3) + *fw.offset(3)) * c;
      *fw.offset(-(4i32) as isize) =
        *fw.offset(-(4i32) as isize) + (*fl.offset(4) + *fw.offset(4)) * c;
      *fw.offset(-(3i32) as isize) =
        *fw.offset(-(3i32) as isize) + (*fl.offset(5) + *fw.offset(5)) * c;
      *fw.offset(-(2i32) as isize) =
        *fw.offset(-(2i32) as isize) + (*fl.offset(6) + *fw.offset(6)) * c;
      *fw.offset(-(1i32) as isize) =
        *fw.offset(-(1i32) as isize) + (*fl.offset(7) + *fw.offset(7)) * c;
      fl = fw;
      fw = fw.offset((2 * NB_ELTS_V8) as isize);
      i += 1;
    }
    if m < end {
      assert!(m.wrapping_add(1u32) == end);
      c += c;
      *fw.offset(-(8i32) as isize) = *fw.offset(-(8i32) as isize) + *fl.offset(0) * c;
      *fw.offset(-(7i32) as isize) = *fw.offset(-(7i32) as isize) + *fl.offset(1) * c;
      *fw.offset(-(6i32) as isize) = *fw.offset(-(6i32) as isize) + *fl.offset(2) * c;
      *fw.offset(-(5i32) as isize) = *fw.offset(-(5i32) as isize) + *fl.offset(3) * c;
      *fw.offset(-(4i32) as isize) = *fw.offset(-(4i32) as isize) + *fl.offset(4) * c;
      *fw.offset(-(3i32) as isize) = *fw.offset(-(3i32) as isize) + *fl.offset(5) * c;
      *fw.offset(-(2i32) as isize) = *fw.offset(-(2i32) as isize) + *fl.offset(6) * c;
      *fw.offset(-(1i32) as isize) = *fw.offset(-(1i32) as isize) + *fl.offset(7) * c
    };
  }
}
/* <summary>                             */
/* Inverse 9-7 wavelet transform in 1-D. */
/* </summary>                            */
fn opj_v8dwt_decode(mut dwt: &opj_v8dwt_t) {
  unsafe {
    let mut a: OPJ_INT32 = 0;
    let mut b: OPJ_INT32 = 0;
    /* BUG_WEIRD_TWO_INVK (look for this identifier in tcd.c) */
    /* Historic value for 2 / opj_invK */
    /* Normally, we should use invK, but if we do so, we have failures in the */
    /* conformance test, due to MSE and peak errors significantly higher than */
    /* accepted value */
    /* Due to using two_invK instead of invK, we have to compensate in tcd.c */
    /* the computation of the stepsize for the non LL subbands */
    let two_invK = 1.625_732_4_f32;
    if (*dwt).cas == 0i32 {
      if !((*dwt).dn > 0i32 || (*dwt).sn > 1i32) {
        return;
      }
      a = 0i32;
      b = 1i32
    } else {
      if !((*dwt).sn > 0i32 || (*dwt).dn > 1i32) {
        return;
      }
      a = 1i32;
      b = 0i32
    }
    opj_v8dwt_decode_step1(
      (*dwt).wavelet.offset(a as isize),
      (*dwt).win_l_x0,
      (*dwt).win_l_x1,
      opj_K,
    );
    opj_v8dwt_decode_step1(
      (*dwt).wavelet.offset(b as isize),
      (*dwt).win_h_x0,
      (*dwt).win_h_x1,
      two_invK,
    );
    opj_v8dwt_decode_step2(
      (*dwt).wavelet.offset(b as isize),
      (*dwt).wavelet.offset(a as isize).offset(1),
      (*dwt).win_l_x0,
      (*dwt).win_l_x1,
      opj_int_min((*dwt).sn, (*dwt).dn - a) as OPJ_UINT32,
      -opj_dwt_delta,
    );
    opj_v8dwt_decode_step2(
      (*dwt).wavelet.offset(a as isize),
      (*dwt).wavelet.offset(b as isize).offset(1),
      (*dwt).win_h_x0,
      (*dwt).win_h_x1,
      opj_int_min((*dwt).dn, (*dwt).sn - b) as OPJ_UINT32,
      -opj_dwt_gamma,
    );
    opj_v8dwt_decode_step2(
      (*dwt).wavelet.offset(b as isize),
      (*dwt).wavelet.offset(a as isize).offset(1),
      (*dwt).win_l_x0,
      (*dwt).win_l_x1,
      opj_int_min((*dwt).sn, (*dwt).dn - a) as OPJ_UINT32,
      -opj_dwt_beta,
    );
    opj_v8dwt_decode_step2(
      (*dwt).wavelet.offset(a as isize),
      (*dwt).wavelet.offset(b as isize).offset(1),
      (*dwt).win_h_x0,
      (*dwt).win_h_x1,
      opj_int_min((*dwt).dn, (*dwt).sn - b) as OPJ_UINT32,
      -opj_dwt_alpha,
    );
  }
}

/* <summary>                             */
/* Inverse 9-7 wavelet transform in 2-D. */
/* </summary>                            */
fn opj_dwt_decode_tile_97(mut tilec: *mut opj_tcd_tilecomp_t, mut numres: OPJ_UINT32) -> OPJ_BOOL {
  unsafe {
    let mut h = opj_v8dwt_t {
      wavelet: core::ptr::null_mut::<opj_v8_t>(),
      dn: 0,
      sn: 0,
      cas: 0,
      win_l_x0: 0,
      win_l_x1: 0,
      win_h_x0: 0,
      win_h_x1: 0,
    }; /* width of the resolution level computed */
    let mut v = opj_v8dwt_t {
      wavelet: core::ptr::null_mut::<opj_v8_t>(),
      dn: 0,
      sn: 0,
      cas: 0,
      win_l_x0: 0,
      win_l_x1: 0,
      win_h_x0: 0,
      win_h_x1: 0,
    }; /* height of the resolution level computed */
    let mut res = (*tilec).resolutions;
    let mut rw = ((*res).x1 - (*res).x0) as OPJ_UINT32;
    let mut rh = ((*res).y1 - (*res).y0) as OPJ_UINT32;
    let mut w = ((*(*tilec)
      .resolutions
      .offset((*tilec).minimum_num_resolutions.wrapping_sub(1u32) as isize))
    .x1
      - (*(*tilec)
        .resolutions
        .offset((*tilec).minimum_num_resolutions.wrapping_sub(1u32) as isize))
      .x0) as OPJ_UINT32;
    let mut l_data_size: OPJ_SIZE_T = 0;
    if numres == 1u32 {
      return 1i32;
    }
    l_data_size = opj_dwt_max_resolution(res, numres) as OPJ_SIZE_T;
    /* overflow check */
    if l_data_size > (usize::MAX).wrapping_div(core::mem::size_of::<opj_v8_t>()) {
      /* FIXME event manager error callback */
      return 0i32;
    }
    let layout = Layout::from_size_align_unchecked(
      l_data_size.wrapping_mul(core::mem::size_of::<opj_v8_t>()),
      16,
    );
    h.wavelet = alloc(layout) as *mut opj_v8_t;
    if h.wavelet.is_null() {
      /* FIXME event manager error callback */
      return 0i32;
    } /* width of the resolution level computed */
    v.wavelet = h.wavelet; /* height of the resolution level computed */
    loop {
      numres = numres.wrapping_sub(1);
      if numres == 0 {
        break;
      }
      let mut aj = (*tilec).data as *mut OPJ_FLOAT32;
      let mut j: OPJ_UINT32 = 0;
      h.sn = rw as OPJ_INT32;
      v.sn = rh as OPJ_INT32;
      res = res.offset(1);
      rw = ((*res).x1 - (*res).x0) as OPJ_UINT32;
      rh = ((*res).y1 - (*res).y0) as OPJ_UINT32;
      h.dn = rw.wrapping_sub(h.sn as OPJ_UINT32) as OPJ_INT32;
      h.cas = (*res).x0 % 2i32;
      h.win_l_x0 = 0 as OPJ_UINT32;
      h.win_l_x1 = h.sn as OPJ_UINT32;
      h.win_h_x0 = 0 as OPJ_UINT32;
      h.win_h_x1 = h.dn as OPJ_UINT32;
      j = 0 as OPJ_UINT32;
      while j + (NB_ELTS_V8 - 1) < rh {
        let mut k: OPJ_UINT32 = 0;
        opj_v8dwt_interleave_h(&h, aj, w, NB_ELTS_V8);
        opj_v8dwt_decode(&h);
        /* To be adapted if NB_ELTS_V8 changes */
        k = 0 as OPJ_UINT32;
        while k < rw {
          *aj.offset(k as isize) = (*h.wavelet.offset(k as isize)).f[0_usize];
          *aj.add((k as usize).wrapping_add(w as OPJ_SIZE_T)) =
            (*h.wavelet.offset(k as isize)).f[1];
          *aj.add((k as usize).wrapping_add((w as OPJ_SIZE_T).wrapping_mul(2))) =
            (*h.wavelet.offset(k as isize)).f[2];
          *aj.add((k as usize).wrapping_add((w as OPJ_SIZE_T).wrapping_mul(3))) =
            (*h.wavelet.offset(k as isize)).f[3];
          k += 1;
        }
        k = 0 as OPJ_UINT32;
        while k < rw {
          *aj.add((k as usize).wrapping_add((w as OPJ_SIZE_T).wrapping_mul(4))) =
            (*h.wavelet.offset(k as isize)).f[4];
          *aj.add((k as usize).wrapping_add((w as OPJ_SIZE_T).wrapping_mul(5))) =
            (*h.wavelet.offset(k as isize)).f[5];
          *aj.add((k as usize).wrapping_add((w as OPJ_SIZE_T).wrapping_mul(6))) =
            (*h.wavelet.offset(k as isize)).f[6];
          *aj.add((k as usize).wrapping_add((w as OPJ_SIZE_T).wrapping_mul(7))) =
            (*h.wavelet.offset(k as isize)).f[7];
          k += 1;
        }
        aj = aj.offset((w * NB_ELTS_V8) as isize);
        j += NB_ELTS_V8;
      }
      if j < rh {
        let mut k_0: OPJ_UINT32 = 0;
        opj_v8dwt_interleave_h(&h, aj, w, rh.wrapping_sub(j));
        opj_v8dwt_decode(&h);
        k_0 = 0 as OPJ_UINT32;
        while k_0 < rw {
          let mut l: OPJ_UINT32 = 0;
          l = 0 as OPJ_UINT32;
          while l < rh.wrapping_sub(j) {
            *aj.add((k_0 as usize).wrapping_add((w as OPJ_SIZE_T).wrapping_mul(l as usize))) =
              (*h.wavelet.offset(k_0 as isize)).f[l as usize];
            l += 1;
          }
          k_0 += 1;
        }
      }
      v.dn = rh.wrapping_sub(v.sn as OPJ_UINT32) as OPJ_INT32;
      v.cas = (*res).y0 % 2i32;
      v.win_l_x0 = 0 as OPJ_UINT32;
      v.win_l_x1 = v.sn as OPJ_UINT32;
      v.win_h_x0 = 0 as OPJ_UINT32;
      v.win_h_x1 = v.dn as OPJ_UINT32;
      aj = (*tilec).data as *mut OPJ_FLOAT32;
      j = rw;
      while j > (NB_ELTS_V8 - 1) {
        let mut k_1: OPJ_UINT32 = 0;
        opj_v8dwt_interleave_v(&v, aj, w, NB_ELTS_V8);
        opj_v8dwt_decode(&v);
        k_1 = 0 as OPJ_UINT32;
        while k_1 < rh {
          memcpy(
            &mut *aj.add((k_1 as usize).wrapping_mul(w as OPJ_SIZE_T)) as *mut OPJ_FLOAT32
              as *mut core::ffi::c_void,
            &mut *v.wavelet.offset(k_1 as isize) as *mut opj_v8_t as *const core::ffi::c_void,
            NB_ELTS_V8 as usize * core::mem::size_of::<OPJ_FLOAT32>(),
          );
          k_1 += 1;
        }
        aj = aj.offset(8);
        j -= NB_ELTS_V8;
      }
      if rw & (NB_ELTS_V8 - 1) != 0 {
        let mut k_2: OPJ_UINT32 = 0;
        j = rw & (NB_ELTS_V8 - 1);
        opj_v8dwt_interleave_v(&v, aj, w, j);
        opj_v8dwt_decode(&v);
        k_2 = 0 as OPJ_UINT32;
        while k_2 < rh {
          memcpy(
            &mut *aj.add((k_2 as usize).wrapping_mul(w as OPJ_SIZE_T)) as *mut OPJ_FLOAT32
              as *mut core::ffi::c_void,
            &mut *v.wavelet.offset(k_2 as isize) as *mut opj_v8_t as *const core::ffi::c_void,
            (j as OPJ_SIZE_T).wrapping_mul(core::mem::size_of::<OPJ_FLOAT32>()),
          );
          k_2 += 1;
        }
      }
    }
    dealloc(h.wavelet as _, layout);
    1i32
  }
}
fn opj_dwt_decode_partial_97(
  mut tilec: *mut opj_tcd_tilecomp_t,
  mut numres: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut h = opj_v8dwt_t {
      wavelet: core::ptr::null_mut::<opj_v8_t>(),
      dn: 0,
      sn: 0,
      cas: 0,
      win_l_x0: 0,
      win_l_x1: 0,
      win_h_x0: 0,
      win_h_x1: 0,
    };
    let mut v = opj_v8dwt_t {
      wavelet: core::ptr::null_mut::<opj_v8_t>(),
      dn: 0,
      sn: 0,
      cas: 0,
      win_l_x0: 0,
      win_l_x1: 0,
      win_h_x0: 0,
      win_h_x1: 0,
    };
    let mut resno: OPJ_UINT32 = 0;
    /* This value matches the maximum left/right extension given in tables */
    /* F.2 and F.3 of the standard. Note: in opj_tcd_is_subband_area_of_interest() */
    /* we currently use 3. */
    let filter_width = 4u32; /* width of the resolution level computed */
    let mut tr = (*tilec).resolutions; /* height of the resolution level computed */
    let mut tr_max: *mut opj_tcd_resolution_t = &mut *(*tilec)
      .resolutions
      .offset(numres.wrapping_sub(1u32) as isize)
      as *mut opj_tcd_resolution_t;
    let mut rw = ((*tr).x1 - (*tr).x0) as OPJ_UINT32;
    let mut rh = ((*tr).y1 - (*tr).y0) as OPJ_UINT32;
    let mut l_data_size: OPJ_SIZE_T = 0;
    /* Compute the intersection of the area of interest, expressed in tile coordinates */
    /* with the tile coordinates */
    let mut win_tcx0 = (*tilec).win_x0;
    let mut win_tcy0 = (*tilec).win_y0;
    let mut win_tcx1 = (*tilec).win_x1;
    let mut win_tcy1 = (*tilec).win_y1;
    if (*tr_max).x0 == (*tr_max).x1 || (*tr_max).y0 == (*tr_max).y1 {
      return 1i32;
    }
    let mut sa = if let Some(sa) = opj_dwt_init_sparse_array(tilec, numres) {
      sa
    } else {
      return 0i32;
    };
    if numres == 1u32 {
      let ret = sparse_array_read(
        &sa,
        (*tr_max).win_x0.wrapping_sub((*tr_max).x0 as OPJ_UINT32),
        (*tr_max).win_y0.wrapping_sub((*tr_max).y0 as OPJ_UINT32),
        (*tr_max).win_x1.wrapping_sub((*tr_max).x0 as OPJ_UINT32),
        (*tr_max).win_y1.wrapping_sub((*tr_max).y0 as OPJ_UINT32),
        (*tilec).data_win,
        1 as OPJ_UINT32,
        (*tr_max).win_x1.wrapping_sub((*tr_max).win_x0),
        true,
      );
      assert!(ret);
      return 1i32;
    }
    l_data_size = opj_dwt_max_resolution(tr, numres) as OPJ_SIZE_T;
    /* overflow check */
    if l_data_size > (usize::MAX).wrapping_div(core::mem::size_of::<opj_v8_t>()) {
      /* FIXME event manager error callback */
      return 0i32;
    }
    let layout = Layout::from_size_align_unchecked(
      l_data_size.wrapping_mul(core::mem::size_of::<opj_v8_t>()),
      16,
    );
    h.wavelet = alloc(layout) as *mut opj_v8_t;
    if h.wavelet.is_null() {
      /* FIXME event manager error callback */
      return 0i32;
    }
    v.wavelet = h.wavelet;
    resno = 1 as OPJ_UINT32;
    while resno < numres {
      let mut j: OPJ_UINT32 = 0;
      /* Window of interest subband-based coordinates */
      let mut win_ll_x0: OPJ_UINT32 = 0;
      let mut win_ll_y0: OPJ_UINT32 = 0;
      let mut win_ll_x1: OPJ_UINT32 = 0;
      let mut win_ll_y1: OPJ_UINT32 = 0;
      let mut win_hl_x0: OPJ_UINT32 = 0;
      let mut win_hl_x1: OPJ_UINT32 = 0;
      let mut win_lh_y0: OPJ_UINT32 = 0;
      let mut win_lh_y1: OPJ_UINT32 = 0;
      /* Window of interest tile-resolution-based coordinates */
      let mut win_tr_x0: OPJ_UINT32 = 0;
      let mut win_tr_x1: OPJ_UINT32 = 0;
      let mut win_tr_y0: OPJ_UINT32 = 0;
      let mut win_tr_y1: OPJ_UINT32 = 0;
      /* Tile-resolution subband-based coordinates */
      let mut tr_ll_x0: OPJ_UINT32 = 0;
      let mut tr_ll_y0: OPJ_UINT32 = 0;
      let mut tr_hl_x0: OPJ_UINT32 = 0;
      let mut tr_lh_y0: OPJ_UINT32 = 0;
      tr = tr.offset(1);
      h.sn = rw as OPJ_INT32;
      v.sn = rh as OPJ_INT32;
      rw = ((*tr).x1 - (*tr).x0) as OPJ_UINT32;
      rh = ((*tr).y1 - (*tr).y0) as OPJ_UINT32;
      h.dn = rw.wrapping_sub(h.sn as OPJ_UINT32) as OPJ_INT32;
      h.cas = (*tr).x0 % 2i32;
      v.dn = rh.wrapping_sub(v.sn as OPJ_UINT32) as OPJ_INT32;
      v.cas = (*tr).y0 % 2i32;
      /* Get the subband coordinates for the window of interest */
      /* LL band */
      opj_dwt_get_band_coordinates(
        tilec,
        resno,
        0 as OPJ_UINT32,
        win_tcx0,
        win_tcy0,
        win_tcx1,
        win_tcy1,
        &mut win_ll_x0,
        &mut win_ll_y0,
        &mut win_ll_x1,
        &mut win_ll_y1,
      );
      /* HL band */
      opj_dwt_get_band_coordinates(
        tilec,
        resno,
        1 as OPJ_UINT32,
        win_tcx0,
        win_tcy0,
        win_tcx1,
        win_tcy1,
        &mut win_hl_x0,
        core::ptr::null_mut::<OPJ_UINT32>(),
        &mut win_hl_x1,
        core::ptr::null_mut::<OPJ_UINT32>(),
      );
      /* LH band */
      opj_dwt_get_band_coordinates(
        tilec,
        resno,
        2 as OPJ_UINT32,
        win_tcx0,
        win_tcy0,
        win_tcx1,
        win_tcy1,
        core::ptr::null_mut::<OPJ_UINT32>(),
        &mut win_lh_y0,
        core::ptr::null_mut::<OPJ_UINT32>(),
        &mut win_lh_y1,
      );
      /* Beware: band index for non-LL0 resolution are 0=HL, 1=LH and 2=HH */
      tr_ll_x0 = (*tr).bands[1_usize].x0 as OPJ_UINT32;
      tr_ll_y0 = (*tr).bands[0_usize].y0 as OPJ_UINT32;
      tr_hl_x0 = (*tr).bands[0_usize].x0 as OPJ_UINT32;
      tr_lh_y0 = (*tr).bands[1_usize].y0 as OPJ_UINT32;
      /* Subtract the origin of the bands for this tile, to the subwindow */
      /* of interest band coordinates, so as to get them relative to the */
      /* tile */
      win_ll_x0 = opj_uint_subs(win_ll_x0, tr_ll_x0);
      win_ll_y0 = opj_uint_subs(win_ll_y0, tr_ll_y0);
      win_ll_x1 = opj_uint_subs(win_ll_x1, tr_ll_x0);
      win_ll_y1 = opj_uint_subs(win_ll_y1, tr_ll_y0);
      win_hl_x0 = opj_uint_subs(win_hl_x0, tr_hl_x0);
      win_hl_x1 = opj_uint_subs(win_hl_x1, tr_hl_x0);
      win_lh_y0 = opj_uint_subs(win_lh_y0, tr_lh_y0);
      win_lh_y1 = opj_uint_subs(win_lh_y1, tr_lh_y0);
      opj_dwt_segment_grow(
        filter_width,
        h.sn as OPJ_UINT32,
        &mut win_ll_x0,
        &mut win_ll_x1,
      );
      opj_dwt_segment_grow(
        filter_width,
        h.dn as OPJ_UINT32,
        &mut win_hl_x0,
        &mut win_hl_x1,
      );
      opj_dwt_segment_grow(
        filter_width,
        v.sn as OPJ_UINT32,
        &mut win_ll_y0,
        &mut win_ll_y1,
      );
      opj_dwt_segment_grow(
        filter_width,
        v.dn as OPJ_UINT32,
        &mut win_lh_y0,
        &mut win_lh_y1,
      );
      /* Compute the tile-resolution-based coordinates for the window of interest */
      if h.cas == 0i32 {
        win_tr_x0 = opj_uint_min(
          (2u32).wrapping_mul(win_ll_x0),
          (2u32).wrapping_mul(win_hl_x0).wrapping_add(1u32),
        );
        win_tr_x1 = opj_uint_min(
          opj_uint_max(
            (2u32).wrapping_mul(win_ll_x1),
            (2u32).wrapping_mul(win_hl_x1).wrapping_add(1u32),
          ),
          rw,
        )
      } else {
        win_tr_x0 = opj_uint_min(
          (2u32).wrapping_mul(win_hl_x0),
          (2u32).wrapping_mul(win_ll_x0).wrapping_add(1u32),
        );
        win_tr_x1 = opj_uint_min(
          opj_uint_max(
            (2u32).wrapping_mul(win_hl_x1),
            (2u32).wrapping_mul(win_ll_x1).wrapping_add(1u32),
          ),
          rw,
        )
      }
      if v.cas == 0i32 {
        win_tr_y0 = opj_uint_min(
          (2u32).wrapping_mul(win_ll_y0),
          (2u32).wrapping_mul(win_lh_y0).wrapping_add(1u32),
        );
        win_tr_y1 = opj_uint_min(
          opj_uint_max(
            (2u32).wrapping_mul(win_ll_y1),
            (2u32).wrapping_mul(win_lh_y1).wrapping_add(1u32),
          ),
          rh,
        )
      } else {
        win_tr_y0 = opj_uint_min(
          (2u32).wrapping_mul(win_lh_y0),
          (2u32).wrapping_mul(win_ll_y0).wrapping_add(1u32),
        );
        win_tr_y1 = opj_uint_min(
          opj_uint_max(
            (2u32).wrapping_mul(win_lh_y1),
            (2u32).wrapping_mul(win_ll_y1).wrapping_add(1u32),
          ),
          rh,
        )
      }
      h.win_l_x0 = win_ll_x0;
      h.win_l_x1 = win_ll_x1;
      h.win_h_x0 = win_hl_x0;
      h.win_h_x1 = win_hl_x1;
      j = 0 as OPJ_UINT32;
      while j + (NB_ELTS_V8 - 1) < rh {
        if j + (NB_ELTS_V8 - 1) >= win_ll_y0 && j < win_ll_y1
          || j + (NB_ELTS_V8 - 1) >= win_lh_y0.wrapping_add(v.sn as OPJ_UINT32)
            && j < win_lh_y1.wrapping_add(v.sn as OPJ_UINT32)
        {
          opj_v8dwt_interleave_partial_h(&h, &sa, j, opj_uint_min(NB_ELTS_V8, rh.wrapping_sub(j)));
          opj_v8dwt_decode(&h);
          if !sparse_array_write(
            &mut sa,
            win_tr_x0,
            j,
            win_tr_x1,
            j.wrapping_add(8u32),
            &mut *(*h.wavelet.offset(win_tr_x0 as isize))
              .f
              .as_mut_ptr()
              .offset(0) as *mut OPJ_FLOAT32 as *mut OPJ_INT32,
            NB_ELTS_V8,
            1 as OPJ_UINT32,
            true,
          ) {
            /* FIXME event manager error callback */
            dealloc(h.wavelet as _, layout);
            return 0i32;
          }
        }
        j += NB_ELTS_V8;
      }
      if j < rh
        && (j + (NB_ELTS_V8 - 1) >= win_ll_y0 && j < win_ll_y1
          || j + (NB_ELTS_V8 - 1) >= win_lh_y0.wrapping_add(v.sn as OPJ_UINT32)
            && j < win_lh_y1.wrapping_add(v.sn as OPJ_UINT32))
      {
        opj_v8dwt_interleave_partial_h(&h, &sa, j, rh.wrapping_sub(j));
        opj_v8dwt_decode(&h);
        if !sparse_array_write(
          &mut sa,
          win_tr_x0,
          j,
          win_tr_x1,
          rh,
          &mut *(*h.wavelet.offset(win_tr_x0 as isize))
            .f
            .as_mut_ptr()
            .offset(0) as *mut OPJ_FLOAT32 as *mut OPJ_INT32,
          NB_ELTS_V8,
          1 as OPJ_UINT32,
          true,
        ) {
          /* FIXME event manager error callback */
          dealloc(h.wavelet as _, layout);
          return 0i32;
        }
      }
      v.win_l_x0 = win_ll_y0;
      v.win_l_x1 = win_ll_y1;
      v.win_h_x0 = win_lh_y0;
      v.win_h_x1 = win_lh_y1;
      j = win_tr_x0;
      while j < win_tr_x1 {
        let mut nb_elts = opj_uint_min(NB_ELTS_V8, win_tr_x1.wrapping_sub(j));
        opj_v8dwt_interleave_partial_v(&v, &sa, j, nb_elts);
        opj_v8dwt_decode(&v);
        if !sparse_array_write(
          &mut sa,
          j,
          win_tr_y0,
          j.wrapping_add(nb_elts),
          win_tr_y1,
          &mut *(*h.wavelet.offset(win_tr_y0 as isize))
            .f
            .as_mut_ptr()
            .offset(0) as *mut OPJ_FLOAT32 as *mut OPJ_INT32,
          1 as OPJ_UINT32,
          NB_ELTS_V8,
          true,
        ) {
          /* FIXME event manager error callback */
          dealloc(h.wavelet as _, layout);
          return 0i32;
        }
        j += NB_ELTS_V8;
      }
      resno += 1;
    }
    let ret_0 = sparse_array_read(
      &sa,
      (*tr_max).win_x0.wrapping_sub((*tr_max).x0 as OPJ_UINT32),
      (*tr_max).win_y0.wrapping_sub((*tr_max).y0 as OPJ_UINT32),
      (*tr_max).win_x1.wrapping_sub((*tr_max).x0 as OPJ_UINT32),
      (*tr_max).win_y1.wrapping_sub((*tr_max).y0 as OPJ_UINT32),
      (*tilec).data_win,
      1 as OPJ_UINT32,
      (*tr_max).win_x1.wrapping_sub((*tr_max).win_x0),
      true,
    );
    assert!(ret_0);
    dealloc(h.wavelet as _, layout);
    1i32
  }
}

pub(crate) fn opj_dwt_decode_real(
  p_tcd: &opj_tcd,
  mut tilec: *mut opj_tcd_tilecomp_t,
  mut numres: OPJ_UINT32,
) -> OPJ_BOOL {
  if p_tcd.whole_tile_decoding != 0 {
    opj_dwt_decode_tile_97(tilec, numres)
  } else {
    opj_dwt_decode_partial_97(tilec, numres)
  }
}

use super::openjpeg::*;

use super::malloc::*;

extern "C" {
  fn sqrt(_: core::ffi::c_double) -> core::ffi::c_double;
}

#[inline]
fn opj_int_fix_mul(mut a: OPJ_INT32, mut b: OPJ_INT32) -> OPJ_INT32 {
  let mut temp = a as OPJ_INT64 * b as OPJ_INT64;
  temp += 4096i64;

  assert!(temp >> 13i32 <= 0x7fffffff as OPJ_INT64);
  assert!(temp >> 13i32 >= -(0x7fffffff as OPJ_INT64) - 1 as OPJ_INT64);
  (temp >> 13i32) as OPJ_INT32
}
/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2002-2014, Universite catholique de Louvain (UCL), Belgium
 * Copyright (c) 2002-2014, Professor Benoit Macq
 * Copyright (c) 2001-2003, David Janssens
 * Copyright (c) 2002-2003, Yannick Verschueren
 * Copyright (c) 2003-2007, Francois-Olivier Devaux
 * Copyright (c) 2003-2014, Antonin Descampe
 * Copyright (c) 2005, Herve Drolon, FreeImage Team
 * Copyright (c) 2008, 2011-2012, Centre National d'Etudes Spatiales (CNES), FR
 * Copyright (c) 2012, CS Systemes d'Information, France
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS `AS IS'
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */
/* <summary> */
/* This table contains the norms of the basis function of the reversible MCT. */
/* </summary> */
static opj_mct_norms: [OPJ_FLOAT64; 3] = [1.732f64, 0.8292f64, 0.8292f64];
/* <summary> */
/* This table contains the norms of the basis function of the irreversible MCT. */
/* </summary> */
static opj_mct_norms_real: [OPJ_FLOAT64; 3] = [1.732f64, 1.805f64, 1.573f64];

pub(crate) fn opj_mct_get_mct_norms() -> *const OPJ_FLOAT64 {
  opj_mct_norms.as_ptr()
}

pub(crate) fn opj_mct_get_mct_norms_real() -> *const OPJ_FLOAT64 {
  opj_mct_norms_real.as_ptr()
}
/* <summary> */
/* Forward reversible MCT. */
/* </summary> */

pub(crate) fn opj_mct_encode(
  mut c0: *mut OPJ_INT32,
  mut c1: *mut OPJ_INT32,
  mut c2: *mut OPJ_INT32,
  mut n: OPJ_SIZE_T,
) {
  unsafe {
    let mut i: OPJ_SIZE_T = 0;
    let len = n;
    i = 0 as OPJ_SIZE_T;
    while i < len {
      let mut r = *c0.add(i);
      let mut g = *c1.add(i);
      let mut b = *c2.add(i);
      let mut y = (r + g * 2i32 + b) >> 2i32;
      let mut u = b - g;
      let mut v = r - g;
      *c0.add(i) = y;
      *c1.add(i) = u;
      *c2.add(i) = v;
      i += 1;
    }
  }
}
/* <summary> */
/* Inverse reversible MCT. */
/* </summary> */

pub(crate) fn opj_mct_decode(
  mut c0: *mut OPJ_INT32,
  mut c1: *mut OPJ_INT32,
  mut c2: *mut OPJ_INT32,
  mut n: OPJ_SIZE_T,
) {
  unsafe {
    let mut i: OPJ_SIZE_T = 0;
    i = 0 as OPJ_SIZE_T;
    while i < n {
      let mut y = *c0.add(i);
      let mut u = *c1.add(i);
      let mut v = *c2.add(i);
      let mut g = y - ((u + v) >> 2i32);
      let mut r = v + g;
      let mut b = u + g;
      *c0.add(i) = r;
      *c1.add(i) = g;
      *c2.add(i) = b;
      i += 1;
    }
  }
}
/* <summary> */
/* Get norm of basis function of reversible MCT. */
/* </summary> */

pub(crate) fn opj_mct_getnorm(mut compno: OPJ_UINT32) -> OPJ_FLOAT64 {
  opj_mct_norms[compno as usize]
}
/* <summary> */
/* Forward irreversible MCT. */
/* </summary> */

pub(crate) fn opj_mct_encode_real(
  mut c0: *mut OPJ_FLOAT32,
  mut c1: *mut OPJ_FLOAT32,
  mut c2: *mut OPJ_FLOAT32,
  mut n: OPJ_SIZE_T,
) {
  unsafe {
    let mut i: OPJ_SIZE_T = 0;
    i = 0 as OPJ_SIZE_T;
    while i < n {
      let mut r = *c0.add(i);
      let mut g = *c1.add(i);
      let mut b = *c2.add(i);
      let mut y = 0.299f32 * r + 0.587f32 * g + 0.114f32 * b;
      let mut u = -0.16875f32 * r - 0.331260f32 * g + 0.5f32 * b;
      let mut v = 0.5f32 * r - 0.41869f32 * g - 0.08131f32 * b;
      *c0.add(i) = y;
      *c1.add(i) = u;
      *c2.add(i) = v;
      i += 1;
    }
  }
}
/* <summary> */
/* Inverse irreversible MCT. */
/* </summary> */

pub(crate) fn opj_mct_decode_real(
  mut c0: *mut OPJ_FLOAT32,
  mut c1: *mut OPJ_FLOAT32,
  mut c2: *mut OPJ_FLOAT32,
  mut n: OPJ_SIZE_T,
) {
  unsafe {
    let mut i: OPJ_SIZE_T = 0;
    i = 0 as OPJ_SIZE_T;
    while i < n {
      let mut y = *c0.add(i);
      let mut u = *c1.add(i);
      let mut v = *c2.add(i);
      let mut r = y + v * 1.402f32;
      let mut g = y - u * 0.34413f32 - v * 0.71414f32;
      let mut b = y + u * 1.772f32;
      *c0.add(i) = r;
      *c1.add(i) = g;
      *c2.add(i) = b;
      i += 1;
    }
  }
}
/* <summary> */
/* Get norm of basis function of irreversible MCT. */
/* </summary> */

pub(crate) fn opj_mct_getnorm_real(mut compno: OPJ_UINT32) -> OPJ_FLOAT64 {
  opj_mct_norms_real[compno as usize]
}

pub(crate) fn opj_mct_encode_custom(
  mut pCodingdata: *mut OPJ_BYTE,
  mut n: OPJ_SIZE_T,
  mut pData: *mut *mut OPJ_BYTE,
  mut pNbComp: OPJ_UINT32,
  mut _isSigned: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut lMct = pCodingdata as *mut OPJ_FLOAT32;
    let mut i: OPJ_SIZE_T = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut k: OPJ_UINT32 = 0;
    let mut lNbMatCoeff = pNbComp.wrapping_mul(pNbComp);
    let mut lCurrentData = core::ptr::null_mut::<OPJ_INT32>();
    let mut lCurrentMatrix = core::ptr::null_mut::<OPJ_INT32>();
    let mut lData = pData as *mut *mut OPJ_INT32;
    let mut lMultiplicator = ((1i32) << 13i32) as OPJ_UINT32;
    let mut lMctPtr = core::ptr::null_mut::<OPJ_INT32>();
    lCurrentData = opj_malloc(
      (pNbComp.wrapping_add(lNbMatCoeff) as usize).wrapping_mul(core::mem::size_of::<OPJ_INT32>()),
    ) as *mut OPJ_INT32;
    if lCurrentData.is_null() {
      return 0i32;
    }
    lCurrentMatrix = lCurrentData.offset(pNbComp as isize);
    i = 0 as OPJ_SIZE_T;
    while i < lNbMatCoeff as usize {
      let fresh0 = lMct;
      lMct = lMct.offset(1);
      *lCurrentMatrix.add(i) = (*fresh0 * lMultiplicator as OPJ_FLOAT32) as OPJ_INT32;
      i += 1;
    }
    i = 0 as OPJ_SIZE_T;
    while i < n {
      lMctPtr = lCurrentMatrix;
      j = 0 as OPJ_UINT32;
      while j < pNbComp {
        *lCurrentData.offset(j as isize) = **lData.offset(j as isize);
        j += 1;
      }
      j = 0 as OPJ_UINT32;
      while j < pNbComp {
        **lData.offset(j as isize) = 0i32;
        k = 0 as OPJ_UINT32;
        while k < pNbComp {
          let fresh1 = &mut (**lData.offset(j as isize));
          *fresh1 += opj_int_fix_mul(*lMctPtr, *lCurrentData.offset(k as isize));
          lMctPtr = lMctPtr.offset(1);
          k += 1;
        }
        let fresh2 = &mut (*lData.offset(j as isize));
        *fresh2 = (*fresh2).offset(1);
        j += 1;
      }
      i += 1;
    }
    opj_free(lCurrentData as *mut core::ffi::c_void);
    1i32
  }
}

pub(crate) fn opj_mct_decode_custom(
  mut pDecodingData: *mut OPJ_BYTE,
  mut n: OPJ_SIZE_T,
  mut pData: *mut *mut OPJ_BYTE,
  mut pNbComp: OPJ_UINT32,
  mut _isSigned: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut lMct = core::ptr::null_mut::<OPJ_FLOAT32>();
    let mut i: OPJ_SIZE_T = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut k: OPJ_UINT32 = 0;
    let mut lCurrentData = core::ptr::null_mut::<OPJ_FLOAT32>();
    let mut lCurrentResult = core::ptr::null_mut::<OPJ_FLOAT32>();
    let mut lData = pData as *mut *mut OPJ_FLOAT32;
    lCurrentData = opj_malloc(
      ((2u32).wrapping_mul(pNbComp) as usize).wrapping_mul(core::mem::size_of::<OPJ_FLOAT32>()),
    ) as *mut OPJ_FLOAT32;
    if lCurrentData.is_null() {
      return 0i32;
    }
    lCurrentResult = lCurrentData.offset(pNbComp as isize);
    i = 0 as OPJ_SIZE_T;
    while i < n {
      lMct = pDecodingData as *mut OPJ_FLOAT32;
      j = 0 as OPJ_UINT32;
      while j < pNbComp {
        *lCurrentData.offset(j as isize) = **lData.offset(j as isize);
        j += 1;
      }
      j = 0 as OPJ_UINT32;
      while j < pNbComp {
        *lCurrentResult.offset(j as isize) = 0 as OPJ_FLOAT32;
        k = 0 as OPJ_UINT32;
        while k < pNbComp {
          let fresh3 = lMct;
          lMct = lMct.offset(1);
          let fresh4 = &mut (*lCurrentResult.offset(j as isize));
          *fresh4 += *fresh3 * *lCurrentData.offset(k as isize);
          k += 1;
        }
        let fresh5 = &mut (*lData.offset(j as isize));
        let fresh6 = *fresh5;
        *fresh5 = (*fresh5).offset(1);
        *fresh6 = *lCurrentResult.offset(j as isize);
        j += 1;
      }
      i += 1;
    }
    opj_free(lCurrentData as *mut core::ffi::c_void);
    1i32
  }
}

pub(crate) fn opj_calculate_norms(
  mut pNorms: *mut OPJ_FLOAT64,
  mut pNbComps: OPJ_UINT32,
  mut pMatrix: *mut OPJ_FLOAT32,
) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut lIndex: OPJ_UINT32 = 0;
    let mut lCurrentValue: OPJ_FLOAT32 = 0.;
    let mut lNorms = pNorms;
    let mut lMatrix = pMatrix;
    i = 0 as OPJ_UINT32;
    while i < pNbComps {
      *lNorms.offset(i as isize) = 0 as OPJ_FLOAT64;
      lIndex = i;
      j = 0 as OPJ_UINT32;
      while j < pNbComps {
        lCurrentValue = *lMatrix.offset(lIndex as isize);
        lIndex = (lIndex as core::ffi::c_uint).wrapping_add(pNbComps) as OPJ_UINT32;
        let fresh7 = &mut (*lNorms.offset(i as isize));
        *fresh7 += lCurrentValue as OPJ_FLOAT64 * lCurrentValue as core::ffi::c_double;
        j += 1;
      }
      *lNorms.offset(i as isize) = sqrt(*lNorms.offset(i as isize));
      i += 1;
    }
  }
}

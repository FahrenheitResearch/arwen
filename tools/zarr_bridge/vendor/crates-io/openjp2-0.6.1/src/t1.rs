use super::consts::*;
use super::dwt::*;
use super::event::*;
use super::ht_dec::*;
use super::math::*;
use super::mqc::*;
use super::openjpeg::*;
use super::t1_luts::*;
use super::tcd::*;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use core::{
  cell::RefCell,
  ops::{AddAssign, Deref, DerefMut, Index, IndexMut},
  ptr::null_mut,
};
use std::alloc::{alloc, dealloc, Layout};

use super::malloc::*;

extern "C" {
  fn memset(_: *mut core::ffi::c_void, _: core::ffi::c_int, _: usize) -> *mut core::ffi::c_void;

  fn memcpy(
    _: *mut core::ffi::c_void,
    _: *const core::ffi::c_void,
    _: usize,
  ) -> *mut core::ffi::c_void;
}

#[derive(Default)]
pub(crate) struct T1Flags {
  flags: Vec<opj_flag_t>,
}

#[derive(Copy, Clone)]
pub(crate) struct FlagsPtr {
  flags: *mut opj_flag_t,
}

impl Deref for FlagsPtr {
  type Target = opj_flag_t;

  fn deref(&self) -> &Self::Target {
    unsafe { &*self.flags }
  }
}

impl DerefMut for FlagsPtr {
  fn deref_mut(&mut self) -> &mut Self::Target {
    unsafe { &mut *self.flags }
  }
}

impl Index<isize> for FlagsPtr {
  type Output = opj_flag_t;

  fn index(&self, idx: isize) -> &Self::Output {
    unsafe { &*self.flags.offset(idx) }
  }
}

impl IndexMut<isize> for FlagsPtr {
  fn index_mut(&mut self, idx: isize) -> &mut Self::Output {
    unsafe { &mut *self.flags.offset(idx) }
  }
}

impl AddAssign<usize> for FlagsPtr {
  fn add_assign(&mut self, n: usize) {
    unsafe {
      self.flags = self.flags.add(n);
    }
  }
}

impl FlagsPtr {
  pub fn offset(&self, offset: isize) -> Self {
    unsafe {
      Self {
        flags: self.flags.offset(offset),
      }
    }
  }

  pub fn inc(&mut self, n: usize) {
    unsafe {
      self.flags = self.flags.add(n);
    }
  }
}

impl T1Flags {
  fn new() -> Self {
    Self::default()
  }

  pub fn resize(&mut self, len: usize) {
    if self.flags.len() != len {
      self.flags = vec![0; len];
    } else {
      for flag in &mut self.flags {
        *flag = 0;
      }
    }
  }

  pub fn as_mut_ptr(&mut self) -> *mut opj_flag_t {
    self.flags.as_mut_ptr()
  }

  pub fn offset(&mut self, offset: isize) -> FlagsPtr {
    unsafe {
      FlagsPtr {
        flags: self.as_mut_ptr().offset(offset),
      }
    }
  }
}

#[derive(Default)]
pub(crate) struct T1Data {
  data: Vec<i32>,
}

#[derive(Copy, Clone)]
pub(crate) struct DataPtr {
  data: *mut i32,
}

impl Deref for DataPtr {
  type Target = i32;

  fn deref(&self) -> &Self::Target {
    unsafe { &*self.data }
  }
}

impl DerefMut for DataPtr {
  fn deref_mut(&mut self) -> &mut Self::Target {
    unsafe { &mut *self.data }
  }
}

impl Index<isize> for DataPtr {
  type Output = i32;

  fn index(&self, idx: isize) -> &Self::Output {
    unsafe { &*self.data.offset(idx) }
  }
}

impl IndexMut<isize> for DataPtr {
  fn index_mut(&mut self, idx: isize) -> &mut Self::Output {
    unsafe { &mut *self.data.offset(idx) }
  }
}

impl AddAssign<usize> for DataPtr {
  fn add_assign(&mut self, n: usize) {
    unsafe {
      self.data = self.data.add(n);
    }
  }
}

impl DataPtr {
  pub fn offset(&self, offset: isize) -> Self {
    unsafe {
      Self {
        data: self.data.offset(offset),
      }
    }
  }

  pub fn inc(&mut self, n: usize) {
    unsafe {
      self.data = self.data.add(n);
    }
  }
}

impl T1Data {
  fn new() -> Self {
    Self::default()
  }

  pub fn resize(&mut self, len: usize) {
    if self.data.len() != len {
      self.data = vec![0; len];
    } else {
      for data in &mut self.data {
        *data = 0;
      }
    }
  }

  pub fn as_mut_ptr(&mut self) -> *mut i32 {
    self.data.as_mut_ptr()
  }

  pub fn offset(&mut self, offset: isize) -> DataPtr {
    unsafe {
      DataPtr {
        data: self.as_mut_ptr().offset(offset),
      }
    }
  }
}

pub(crate) struct opj_t1 {
  pub mqc: opj_mqc_t,
  pub data: T1Data,
  decoded_data: Option<*mut OPJ_INT32>,
  pub flags: T1Flags,
  pub w: OPJ_UINT32,
  pub h: OPJ_UINT32,
  pub datasize: OPJ_UINT32,
  pub encoder: OPJ_BOOL,
  pub mustuse_cblkdatabuffer: OPJ_BOOL,
  pub cblkdatabuffer: *mut OPJ_BYTE,
  pub cblkdatabuffersize: OPJ_UINT32,
}
pub(crate) type opj_t1_t = opj_t1;

impl Default for opj_t1 {
  fn default() -> Self {
    Self {
      mqc: opj_mqc::default(),
      data: T1Data::default(),
      decoded_data: None,
      flags: T1Flags::default(),
      w: 0,
      h: 0,
      datasize: 0,
      encoder: 0,
      mustuse_cblkdatabuffer: 0,
      cblkdatabuffer: null_mut(),
      cblkdatabuffersize: 0,
    }
  }
}

impl Drop for opj_t1 {
  fn drop(&mut self) {
    opj_free(self.cblkdatabuffer as *mut core::ffi::c_void);
  }
}

impl opj_t1 {
  fn new() -> Self {
    Self::default()
  }

  fn set_decoded_data(&mut self, decoded_data: *mut OPJ_INT32) {
    self.decoded_data = Some(decoded_data);
  }

  fn reset_decoded_data(&mut self) {
    self.decoded_data = None;
  }

  fn data_offset(&mut self, offset: isize) -> DataPtr {
    if let Some(decoded_data) = self.decoded_data {
      unsafe {
        DataPtr {
          data: decoded_data.offset(offset),
        }
      }
    } else {
      self.data.offset(offset)
    }
  }
}

thread_local! {
  static T1: RefCell<opj_t1> = RefCell::new(opj_t1::new())
}

pub(crate) struct opj_t1_cblk_encode_processing_job_t {
  pub compno: OPJ_UINT32,
  pub resno: OPJ_UINT32,
  pub cblk: *mut opj_tcd_cblk_enc_t,
  pub tile: *mut opj_tcd_tile_t,
  pub band: *mut opj_tcd_band_t,
  pub tilec: *mut opj_tcd_tilecomp_t,
  pub tccp: *mut opj_tccp_t,
  pub mct_norms: *const OPJ_FLOAT64,
  pub mct_numcomps: OPJ_UINT32,
  pub pret: *mut OPJ_BOOL,
}

pub(crate) struct opj_t1_cblk_decode_processing_job_t {
  pub whole_tile_decoding: OPJ_BOOL,
  pub resno: OPJ_UINT32,
  pub cblk: *mut opj_tcd_cblk_dec_t,
  pub band: *mut opj_tcd_band_t,
  pub tilec: *mut opj_tcd_tilecomp_t,
  pub tccp: *mut opj_tccp_t,
  pub mustuse_cblkdatabuffer: OPJ_BOOL,
  pub pret: *mut OPJ_BOOL,
  pub p_manager: opj_event_mgr,
  pub check_pterm: OPJ_BOOL,
}

#[inline]
fn opj_t1_setcurctx(mqc: &mut opj_mqc_t, ctxno: u8) {
  mqc.set_curctx(ctxno);
}

/* Macros to deal with signed integer with just MSB bit set for
 * negative values (smr = signed magnitude representation) */
//#define opj_smr_abs(x)  (((OPJ_UINT32)(x)) & 0x7FFFFFFFU)
fn opj_smr_abs(x: i32) -> u32 {
  x as u32 & 0x7FFFFFFFu32
}

//#define opj_smr_sign(x) (((OPJ_UINT32)(x)) >> 31)
fn opj_smr_sign(x: i32) -> u32 {
  x as u32 >> 31
}

//#define opj_to_smr(x)   ((x) >= 0 ? (OPJ_UINT32)(x) : ((OPJ_UINT32)(-x) | 0x80000000U))
fn opj_to_smr(x: i32) -> u32 {
  if x >= 0 {
    x as u32
  } else {
    -x as u32 | 0x80000000
  }
}

/* * @name Local static functions */
/*@{*/
/*@}*/
/*@}*/
/* ----------------------------------------------------------------------- */

#[inline]
fn opj_t1_getctxno_zc(mut mqc: &mut opj_mqc_t, f: OPJ_UINT32) -> OPJ_BYTE {
  mqc.lut_ctxno_zc_orient[(f & T1_SIGMA_NEIGHBOURS) as usize]
}

#[inline]
fn opj_t1_getctxtno_sc_or_spb_index(
  mut fX: OPJ_UINT32,
  mut pfX: OPJ_UINT32,
  mut nfX: OPJ_UINT32,
  mut ci: OPJ_UINT32,
) -> OPJ_UINT32 {
  /*
    0 pfX T1_CHI_THIS           T1_LUT_SGN_W
    1 tfX T1_SIGMA_1            T1_LUT_SIG_N
    2 nfX T1_CHI_THIS           T1_LUT_SGN_E
    3 tfX T1_SIGMA_3            T1_LUT_SIG_W
    4  fX T1_CHI_(THIS - 1)     T1_LUT_SGN_N
    5 tfX T1_SIGMA_5            T1_LUT_SIG_E
    6  fX T1_CHI_(THIS + 1)     T1_LUT_SGN_S
    7 tfX T1_SIGMA_7            T1_LUT_SIG_S
  */

  let mut lu = fX >> ci.wrapping_mul(3) & (T1_SIGMA_1 | T1_SIGMA_3 | T1_SIGMA_5 | T1_SIGMA_7);

  lu |= (pfX >> T1_CHI_THIS_I.wrapping_add(ci.wrapping_mul(3))) & (1 << 0);
  lu |= (nfX >> (T1_CHI_THIS_I - 2).wrapping_add(ci.wrapping_mul(3))) & (1 << 2);
  if ci == 0 {
    lu |= (fX >> (T1_CHI_0_I - 4)) & (1 << 4);
  } else {
    lu |= (fX >> (T1_CHI_1_I - 4).wrapping_add(ci.wrapping_sub(1).wrapping_mul(3))) & (1 << 4);
  }
  lu |= (fX >> (T1_CHI_2_I - 6).wrapping_add(ci.wrapping_mul(3))) & (1 << 6);
  lu
}

#[inline]
fn opj_t1_getctxno_sc(mut lu: OPJ_UINT32) -> OPJ_BYTE {
  lut_ctxno_sc[lu as usize]
}

#[inline]
fn opj_t1_getctxno_mag(mut f: OPJ_UINT32) -> u8 {
  let tmp = if f & T1_SIGMA_NEIGHBOURS != 0 {
    T1_CTXNO_MAG + 1
  } else {
    T1_CTXNO_MAG
  };

  if f & T1_MU_0 != 0 {
    T1_CTXNO_MAG + 2
  } else {
    tmp
  }
}

#[inline]
fn opj_t1_getspb(mut lu: OPJ_UINT32) -> OPJ_BYTE {
  lut_spb[lu as usize]
}

fn opj_t1_getnmsedec_sig(mut x: OPJ_UINT32, mut bitpos: OPJ_UINT32) -> OPJ_INT16 {
  if bitpos > 0 {
    return lut_nmsedec_sig[((x >> bitpos) & ((1 << T1_NMSEDEC_BITS) - 1)) as usize];
  }
  lut_nmsedec_sig0[(x & ((1 << T1_NMSEDEC_BITS) - 1)) as usize]
}

fn opj_t1_getnmsedec_ref(mut x: OPJ_UINT32, mut bitpos: OPJ_UINT32) -> OPJ_INT16 {
  if bitpos > 0 {
    return lut_nmsedec_ref[((x >> bitpos) & ((1 << T1_NMSEDEC_BITS) - 1)) as usize];
  }
  lut_nmsedec_ref0[(x & ((1 << T1_NMSEDEC_BITS) - 1)) as usize]
}

#[inline]
fn opj_t1_update_flags_macro(
  mut flagsp: FlagsPtr,
  ci: OPJ_UINT32,
  s: OPJ_UINT32,
  stride: OPJ_UINT32,
  vsc: OPJ_UINT32,
) {
  /* east */
  flagsp[-1] |= T1_SIGMA_5 << 3_u32.wrapping_mul(ci);

  /* mark target as significant */
  flagsp[0] |= ((s << T1_CHI_1_I) | T1_SIGMA_4) << 3_u32.wrapping_mul(ci);

  /* west */
  flagsp[1] |= T1_SIGMA_3 << 3_u32.wrapping_mul(ci);

  /* north-west, north, north-east */
  if ci == 0 && vsc == 0 {
    let mut north = flagsp.offset(-(stride as isize));
    north[0] |= (s << T1_CHI_5_I) | T1_SIGMA_16;
    north[-1] |= T1_SIGMA_17;
    north[1] |= T1_SIGMA_15;
  }

  /* south-west, south, south-east */
  if ci == 3 {
    let mut south = flagsp.offset(stride as isize);
    south[0] |= (s << T1_CHI_0_I) | T1_SIGMA_1;
    south[-1] |= T1_SIGMA_2;
    south[1] |= T1_SIGMA_0;
  }
}

#[inline]
fn opj_t1_update_flags(
  mut flagsp: FlagsPtr,
  ci: OPJ_UINT32,
  s: OPJ_UINT32,
  stride: OPJ_UINT32,
  vsc: OPJ_UINT32,
) {
  opj_t1_update_flags_macro(flagsp, ci, s, stride, vsc);
}

/* *
Decode significant pass
*/

/* *
Encode significant pass
*/
#[inline]
fn opj_t1_enc_sigpass_step_macro(
  mqc: &mut opj_mqc_t,
  w: OPJ_UINT32,
  mut flagsp: FlagsPtr,
  mut l_datap: DataPtr,
  bpno: OPJ_INT32,
  one: OPJ_UINT32,
  nmsedec: *mut OPJ_INT32,
  type_0: OPJ_BYTE,
  ci: OPJ_UINT32,
  vsc: OPJ_UINT32,
) {
  unsafe {
    let mut v = 0;
    let flags = flagsp[0];
    if (flags & ((T1_SIGMA_THIS | T1_PI_THIS) << ci.wrapping_mul(3))) == 0
      && (flags & (T1_SIGMA_NEIGHBOURS << ci.wrapping_mul(3))) != 0
    {
      let ctxt1 = opj_t1_getctxno_zc(mqc, flags >> ci.wrapping_mul(3));
      v = if opj_smr_abs(*l_datap) & one != 0 {
        1
      } else {
        0
      };
      log::debug!("   ctxt1={}", ctxt1);
      opj_t1_setcurctx(mqc, ctxt1);

      if type_0 == T1_TYPE_RAW {
        /* BYPASS/LAZY MODE */
        opj_mqc_bypass_enc_macro(mqc, v);
      } else {
        opj_mqc_encode_macro(mqc, v);
      }
      if v != 0 {
        let lu = opj_t1_getctxtno_sc_or_spb_index(flags, flagsp[-1], flagsp[1], ci);
        let ctxt2 = opj_t1_getctxno_sc(lu);
        v = opj_smr_sign(*l_datap);
        *nmsedec += opj_t1_getnmsedec_sig(opj_smr_abs(*l_datap), bpno as u32) as i32;
        log::debug!("   ctxt2={}", ctxt2);
        opj_t1_setcurctx(mqc, ctxt2);
        if type_0 == T1_TYPE_RAW {
          /* BYPASS/LAZY MODE */
          opj_mqc_bypass_enc_macro(mqc, v);
        } else {
          let spb = opj_t1_getspb(lu) as OPJ_UINT32;
          log::debug!("   spb={}", spb);
          opj_mqc_encode_macro(mqc, v ^ spb);
        }
        opj_t1_update_flags(flagsp, ci, v, w.wrapping_add(2), vsc);
      }
      *flagsp |= T1_PI_THIS << ci.wrapping_mul(3);
    }
  }
}

#[inline]
fn opj_t1_dec_sigpass_step_raw(
  mut t1: &mut opj_t1_t,
  mut flagsp: FlagsPtr,
  mut datap: DataPtr,
  mut oneplushalf: OPJ_INT32,
  mut vsc: OPJ_UINT32,
  mut ci: OPJ_UINT32,
) {
  let mut v = 0;
  let mut mqc = &mut t1.mqc; /* RAW component */
  let flags = flagsp[0];
  if (flags & ((T1_SIGMA_THIS | T1_PI_THIS) << ci.wrapping_mul(3))) == 0
    && (flags & (T1_SIGMA_NEIGHBOURS << ci.wrapping_mul(3))) != 0
  {
    if opj_mqc_raw_decode(mqc) != 0 {
      v = opj_mqc_raw_decode(mqc);
      *datap = if v != 0 { -oneplushalf } else { oneplushalf };
      opj_t1_update_flags(flagsp, ci, v, t1.w.wrapping_add(2), vsc);
    }
    *flagsp |= T1_PI_THIS << ci.wrapping_mul(3);
  }
}

#[inline]
fn opj_t1_dec_sigpass_step_mqc_macro(
  mut flagsp: FlagsPtr,
  mut flags_stride: OPJ_UINT32,
  mut datap: DataPtr,
  mut data_stride: OPJ_UINT32,
  mut ci: OPJ_UINT32,
  mut mqc: &mut opj_mqc_t,
  mut v: OPJ_UINT32,
  mut oneplushalf: OPJ_INT32,
  mut vsc: OPJ_UINT32,
) {
  let flags = flagsp[0];
  if (flags & ((T1_SIGMA_THIS | T1_PI_THIS) << ci.wrapping_mul(3))) == 0
    && (flags & (T1_SIGMA_NEIGHBOURS << ci.wrapping_mul(3))) != 0
  {
    let ctxt1 = opj_t1_getctxno_zc(mqc, flags >> ci.wrapping_mul(3));
    opj_t1_setcurctx(mqc, ctxt1);
    opj_mqc_decode_macro(&mut v, mqc);
    if v != 0 {
      let mut lu = opj_t1_getctxtno_sc_or_spb_index(flags, flagsp[-1], flagsp[1], ci);
      let mut ctxt2 = opj_t1_getctxno_sc(lu);
      let mut spb = opj_t1_getspb(lu) as OPJ_UINT32;
      opj_t1_setcurctx(mqc, ctxt2);
      opj_mqc_decode_macro(&mut v, mqc);
      v ^= spb;
      *datap.offset(ci.wrapping_mul(data_stride) as isize) =
        if v != 0 { -oneplushalf } else { oneplushalf };
      opj_t1_update_flags_macro(flagsp, ci, v, flags_stride, vsc);
    }
    *flagsp |= T1_PI_THIS << ci.wrapping_mul(3);
  }
}

#[inline]
fn opj_t1_dec_sigpass_step_mqc(
  mut t1: &mut opj_t1_t,
  mut flagsp: FlagsPtr,
  mut datap: DataPtr,
  mut oneplushalf: OPJ_INT32,
  mut ci: OPJ_UINT32,
  mut flags_stride: OPJ_UINT32,
  mut vsc: OPJ_UINT32,
) {
  let v = 0;
  let mut mqc = &mut t1.mqc; // MQC component
  opj_t1_dec_sigpass_step_mqc_macro(flagsp, flags_stride, datap, 0, ci, mqc, v, oneplushalf, vsc)
}

// #define T1_FLAGS(x, y)
fn t1_flags(t1: &mut opj_t1_t, x: u32, y: u32) -> FlagsPtr {
  t1.flags
    .offset((x + 1).wrapping_add((y / 4 + 1).wrapping_mul(t1.w.wrapping_add(2))) as isize)
}

/* *
Encode significant pass
*/
fn opj_t1_enc_sigpass(
  mut t1: &mut opj_t1_t,
  mut bpno: OPJ_INT32,
  mut nmsedec: *mut OPJ_INT32,
  mut type_0: OPJ_BYTE,
  mut cblksty: OPJ_UINT32,
) {
  unsafe {
    let mut i = 0;
    let mut k = 0;
    let one = 1 << (bpno + T1_NMSEDEC_FRACBITS);
    let mut f = t1_flags(t1, 0, 0);
    let extra = 2;
    let mut datap = t1.data_offset(0);
    let mqc = &mut t1.mqc;

    *nmsedec = 0;
    log::debug!("enc_sigpass: bpno={}", bpno);

    while k < t1.h & !(0x03) {
      let w = t1.w;
      log::debug!(" k={}", k);
      i = 0;
      while i < w {
        log::debug!(" i={}", i);
        if *f == 0 {
          /* Nothing to do for any of the 4 data points */
        } else {
          opj_t1_enc_sigpass_step_macro(
            mqc,
            t1.w,
            f,
            datap,
            bpno,
            one,
            nmsedec,
            type_0,
            0,
            cblksty & J2K_CCP_CBLKSTY_VSC,
          );
          opj_t1_enc_sigpass_step_macro(
            mqc,
            t1.w,
            f,
            datap.offset(1),
            bpno,
            one,
            nmsedec,
            type_0,
            1,
            0,
          );
          opj_t1_enc_sigpass_step_macro(
            mqc,
            t1.w,
            f,
            datap.offset(2),
            bpno,
            one,
            nmsedec,
            type_0,
            2,
            0,
          );
          opj_t1_enc_sigpass_step_macro(
            mqc,
            t1.w,
            f,
            datap.offset(3),
            bpno,
            one,
            nmsedec,
            type_0,
            3,
            0,
          );
        }
        i += 1;
        f += 1;
        datap += 4
      }
      k += 4;
      f += extra;
    }

    if k < t1.h {
      let mut j: OPJ_UINT32 = 0;
      log::debug!(" k={}", k);
      i = 0;
      while i < t1.w {
        log::debug!(" i={}", i);
        if *f == 0 {
          /* Nothing to do for any of the 4 data points */
          datap = datap.offset(t1.h.wrapping_sub(k) as isize)
        } else {
          j = k;
          while j < t1.h {
            opj_t1_enc_sigpass_step_macro(
              mqc,
              t1.w,
              f,
              datap,
              bpno,
              one,
              nmsedec,
              type_0,
              j - k,
              (j == k && (cblksty & J2K_CCP_CBLKSTY_VSC) != 0) as u32,
            );
            j += 1;
            datap += 1
          }
        }
        i += 1;
        f += 1
      }
    }
  }
}

/* *
Decode significant pass
*/
fn opj_t1_dec_sigpass_raw(mut t1: &mut opj_t1_t, mut bpno: OPJ_INT32, mut cblksty: OPJ_INT32) {
  let mut one = 0;
  let mut half = 0;
  let mut oneplushalf = 0;
  let mut i = 0;
  let mut j = 0;
  let mut k = 0;
  let mut data = t1.data_offset(0);
  let mut flagsp = t1_flags(t1, 0, 0);
  let l_w = t1.w;
  one = 1 << bpno;
  half = one >> 1;
  oneplushalf = one | half;

  k = 0;
  while k < t1.h & !(0x3) {
    i = 0;
    while i < l_w {
      let mut flags = *flagsp;
      if flags != 0 {
        opj_t1_dec_sigpass_step_raw(
          t1,
          flagsp,
          data,
          oneplushalf,
          cblksty as u32 & J2K_CCP_CBLKSTY_VSC, /* vsc */
          0,
        );
        opj_t1_dec_sigpass_step_raw(
          t1,
          flagsp,
          data.offset(l_w as isize),
          oneplushalf,
          0, /* vsc */
          1,
        );
        opj_t1_dec_sigpass_step_raw(
          t1,
          flagsp,
          data.offset(2_u32.wrapping_mul(l_w) as isize),
          oneplushalf,
          0, /* vsc */
          2,
        );
        opj_t1_dec_sigpass_step_raw(
          t1,
          flagsp,
          data.offset(3_u32.wrapping_mul(l_w) as isize),
          oneplushalf,
          0, /* vsc */
          3,
        );
      }
      i += 1;
      flagsp += 1;
      data = data.offset(1)
    }
    k += 4;
    flagsp += 2;
    data = data.offset(3_u32.wrapping_mul(l_w) as isize)
  }
  if k < t1.h {
    i = 0;
    while i < l_w {
      j = 0;
      while j < t1.h.wrapping_sub(k) {
        opj_t1_dec_sigpass_step_raw(
          t1,
          flagsp,
          data.offset(j.wrapping_mul(l_w) as isize),
          oneplushalf,
          cblksty as u32 & J2K_CCP_CBLKSTY_VSC,
          j,
        );
        j += 1
      }
      i += 1;
      flagsp += 1;
      data = data.offset(1)
    }
  }
}

fn opj_t1_dec_sigpass_mqc_internal(
  mut t1: &mut opj_t1_t,
  mut bpno: OPJ_INT32,
  vsc: OPJ_UINT32,
  w: OPJ_UINT32,
  h: OPJ_UINT32,
  flags_stride: OPJ_UINT32,
) {
  let mut one = 0;
  let mut half = 0;
  let mut oneplushalf = 0;
  let mut i = 0;
  let mut j = 0;
  let mut k = 0;
  let mut data = t1.data_offset(0);
  let mut flagsp = t1.flags.offset(flags_stride as isize + 1);
  let l_w = w;
  let mqc = &mut t1.mqc;
  let mut v = 0;
  one = 1 << bpno;
  half = one >> 1;
  oneplushalf = one | half;

  k = 0;
  while k < h & !(0x03) {
    i = 0;
    while i < l_w {
      if *flagsp != 0 {
        opj_t1_dec_sigpass_step_mqc_macro(
          flagsp,
          flags_stride,
          data,
          l_w,
          0,
          mqc,
          v,
          oneplushalf,
          vsc,
        );
        opj_t1_dec_sigpass_step_mqc_macro(
          flagsp,
          flags_stride,
          data,
          l_w,
          1,
          mqc,
          v,
          oneplushalf,
          OPJ_FALSE,
        );
        opj_t1_dec_sigpass_step_mqc_macro(
          flagsp,
          flags_stride,
          data,
          l_w,
          2,
          mqc,
          v,
          oneplushalf,
          OPJ_FALSE,
        );
        opj_t1_dec_sigpass_step_mqc_macro(
          flagsp,
          flags_stride,
          data,
          l_w,
          3,
          mqc,
          v,
          oneplushalf,
          OPJ_FALSE,
        );
      }
      i += 1;
      data = data.offset(1);
      flagsp += 1
    }
    k += 4;
    data = data.offset(3_u32.wrapping_mul(l_w) as isize);
    flagsp += 2
  }
  if k < h {
    i = 0;
    while i < l_w {
      j = 0;
      while j < h.wrapping_sub(k) {
        opj_t1_dec_sigpass_step_mqc(
          t1,
          flagsp,
          data.offset(j.wrapping_mul(l_w) as isize),
          oneplushalf,
          j,
          flags_stride,
          vsc,
        );
        j += 1
      }
      i += 1;
      data = data.offset(1);
      flagsp += 1
    }
  }
}

fn opj_t1_dec_sigpass_mqc_64x64_novsc(t1: &mut opj_t1_t, bpno: OPJ_INT32) {
  opj_t1_dec_sigpass_mqc_internal(t1, bpno, OPJ_FALSE, 64, 64, 66);
}

fn opj_t1_dec_sigpass_mqc_64x64_vsc(t1: &mut opj_t1_t, bpno: OPJ_INT32) {
  opj_t1_dec_sigpass_mqc_internal(t1, bpno, OPJ_TRUE, 64, 64, 66);
}

fn opj_t1_dec_sigpass_mqc_generic_novsc(t1: &mut opj_t1_t, bpno: OPJ_INT32) {
  opj_t1_dec_sigpass_mqc_internal(t1, bpno, OPJ_FALSE, t1.w, t1.h, t1.w + 2);
}

fn opj_t1_dec_sigpass_mqc_generic_vsc(t1: &mut opj_t1_t, bpno: OPJ_INT32) {
  opj_t1_dec_sigpass_mqc_internal(t1, bpno, OPJ_TRUE, t1.w, t1.h, t1.w + 2);
}

fn opj_t1_dec_sigpass_mqc(mut t1: &mut opj_t1_t, mut bpno: OPJ_INT32, mut cblksty: OPJ_INT32) {
  if t1.w == 64 && t1.h == 64 {
    if cblksty as u32 & J2K_CCP_CBLKSTY_VSC != 0 {
      opj_t1_dec_sigpass_mqc_64x64_vsc(t1, bpno);
    } else {
      opj_t1_dec_sigpass_mqc_64x64_novsc(t1, bpno);
    }
  } else if cblksty as u32 & J2K_CCP_CBLKSTY_VSC != 0 {
    opj_t1_dec_sigpass_mqc_generic_vsc(t1, bpno);
  } else {
    opj_t1_dec_sigpass_mqc_generic_novsc(t1, bpno);
  }
}

/* *
Decode refinement pass
*/

/**
Encode refinement pass step
*/
#[inline]
fn opj_t1_enc_refpass_step_macro(
  mqc: &mut opj_mqc_t,
  mut flagsp: FlagsPtr,
  l_datap: DataPtr,
  bpno: OPJ_INT32,
  one: OPJ_UINT32,
  nmsedec: *mut OPJ_INT32,
  type_0: OPJ_BYTE,
  ci: OPJ_UINT32,
) {
  unsafe {
    let mut v: OPJ_UINT32 = 0;
    let flags = flagsp[0];
    if (flags & ((T1_SIGMA_THIS | T1_PI_THIS) << ci.wrapping_mul(3)))
      == (T1_SIGMA_THIS << ci.wrapping_mul(3))
    {
      let shift_flags = flags >> ci.wrapping_mul(3);
      let ctxt = opj_t1_getctxno_mag(shift_flags);
      let abs_data = opj_smr_abs(*l_datap);
      *nmsedec += opj_t1_getnmsedec_ref(abs_data, bpno as u32) as i32;
      v = if abs_data & one != 0 { 1 } else { 0 };
      log::debug!("   ctxt={}", ctxt);
      opj_t1_setcurctx(mqc, ctxt);

      if type_0 == T1_TYPE_RAW {
        /* BYPASS/LAZY MODE */
        opj_mqc_bypass_enc_macro(mqc, v);
      } else {
        opj_mqc_encode_macro(mqc, v);
      }
      *flagsp |= T1_MU_THIS << ci.wrapping_mul(3);
    }
  }
}

#[inline]
fn opj_t1_dec_refpass_step_raw(
  mut t1: &mut opj_t1_t,
  mut flagsp: FlagsPtr,
  mut datap: DataPtr,
  mut poshalf: OPJ_INT32,
  mut ci: OPJ_UINT32,
) {
  let mut v = 0;

  let mut mqc = &mut t1.mqc; /* RAW component */

  if (*flagsp & ((T1_SIGMA_THIS | T1_PI_THIS) << ci.wrapping_mul(3)))
    == (T1_SIGMA_THIS << ci.wrapping_mul(3))
  {
    v = opj_mqc_raw_decode(mqc);
    *datap += if v ^ (*datap < 0) as u32 != 0 {
      poshalf
    } else {
      -poshalf
    };
    *flagsp |= T1_MU_THIS << ci.wrapping_mul(3);
  }
}

fn opj_t1_dec_refpass_step_mqc_macro(
  mut flagsp: FlagsPtr,
  mut datap: DataPtr,
  mut data_stride: OPJ_UINT32,
  mut ci: OPJ_UINT32,
  mut mqc: &mut opj_mqc_t,
  mut v: &mut OPJ_UINT32,
  mut poshalf: OPJ_INT32,
) {
  let flags = flagsp[0];
  if (flags & ((T1_SIGMA_THIS | T1_PI_THIS) << ci.wrapping_mul(3)))
    == (T1_SIGMA_THIS << ci.wrapping_mul(3))
  {
    let ctxt = opj_t1_getctxno_mag(flags >> ci.wrapping_mul(3));
    opj_t1_setcurctx(mqc, ctxt);
    opj_mqc_decode_macro(v, mqc);
    let mut datap = datap.offset(ci.wrapping_mul(data_stride) as isize);
    *datap += if *v ^ (*datap < 0) as u32 != 0 {
      poshalf
    } else {
      -poshalf
    };
    *flagsp |= T1_MU_THIS << ci.wrapping_mul(3);
  }
}

#[inline]
fn opj_t1_dec_refpass_step_mqc(
  mut t1: &mut opj_t1_t,
  mut flagsp: FlagsPtr,
  mut datap: DataPtr,
  mut poshalf: OPJ_INT32,
  mut ci: OPJ_UINT32,
) {
  let mut v = 0;

  let mut mqc = &mut t1.mqc; /* MQC component */
  opj_t1_dec_refpass_step_mqc_macro(flagsp, datap, 0, ci, mqc, &mut v, poshalf);
}

/* *
Encode refinement pass
*/
fn opj_t1_enc_refpass(
  mut t1: &mut opj_t1_t,
  mut bpno: OPJ_INT32,
  mut nmsedec: *mut OPJ_INT32,
  mut type_0: OPJ_BYTE,
) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut k: OPJ_UINT32 = 0;
    let one = 1 << (bpno + T1_NMSEDEC_FRACBITS);
    let mut f = t1_flags(t1, 0, 0);
    let extra = 2;
    let mut datap = t1.data_offset(0);
    let mqc = &mut t1.mqc;

    *nmsedec = 0;
    log::debug!("enc_refpass: bpno={}", bpno);

    while k < t1.h & !(0x03) {
      log::debug!(" k={}", k);
      i = 0;
      while i < t1.w {
        let flags = f[0];
        log::debug!(" i={}", i);
        if (flags & (T1_SIGMA_4 | T1_SIGMA_7 | T1_SIGMA_10 | T1_SIGMA_13)) == 0 {
          /* none significant */
        } else if (flags & (T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3))
          == (T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3)
        {
          /* all processed by sigpass */
        } else {
          opj_t1_enc_refpass_step_macro(mqc, f, datap, bpno, one, nmsedec, type_0, 0);
          opj_t1_enc_refpass_step_macro(mqc, f, datap.offset(1), bpno, one, nmsedec, type_0, 1);
          opj_t1_enc_refpass_step_macro(mqc, f, datap.offset(2), bpno, one, nmsedec, type_0, 2);
          opj_t1_enc_refpass_step_macro(mqc, f, datap.offset(3), bpno, one, nmsedec, type_0, 3);
        }
        i += 1;
        f += 1;
        datap += 4
      }
      k += 4;
      f += extra;
    }

    if k < t1.h {
      let mut j: OPJ_UINT32 = 0;
      let remaining_lines = t1.h - k;
      log::debug!(" k={}", k);
      i = 0;
      while i < t1.w {
        log::debug!(" i={}", i);
        if (*f & (T1_SIGMA_4 | T1_SIGMA_7 | T1_SIGMA_10 | T1_SIGMA_13)) == 0 {
          /* none significant */
          datap = datap.offset(remaining_lines as isize);
        } else {
          j = 0;
          while j < remaining_lines {
            opj_t1_enc_refpass_step_macro(mqc, f, datap, bpno, one, nmsedec, type_0, j);
            j += 1;
            datap += 1
          }
        }
        i += 1;
        f += 1
      }
    }
  }
}

/* *
Decode refinement pass
*/
fn opj_t1_dec_refpass_raw(mut t1: &mut opj_t1_t, mut bpno: OPJ_INT32) {
  let mut one: OPJ_INT32 = 0;
  let mut poshalf: OPJ_INT32 = 0;
  let mut i: OPJ_UINT32 = 0;
  let mut j: OPJ_UINT32 = 0;
  let mut k: OPJ_UINT32 = 0;
  let mut data = t1.data_offset(0);
  let mut flagsp = t1_flags(t1, 0, 0);
  let l_w = t1.w;
  one = 1 << bpno;
  poshalf = one >> 1;
  k = 0;
  while k < t1.h & !(0x03) {
    i = 0;
    while i < l_w {
      let mut flags = *flagsp;
      if flags != 0 {
        opj_t1_dec_refpass_step_raw(t1, flagsp, data, poshalf, 0);
        opj_t1_dec_refpass_step_raw(t1, flagsp, data.offset(l_w as isize), poshalf, 1);
        opj_t1_dec_refpass_step_raw(
          t1,
          flagsp,
          data.offset(2_u32.wrapping_mul(l_w) as isize),
          poshalf,
          2,
        );
        opj_t1_dec_refpass_step_raw(
          t1,
          flagsp,
          data.offset(3_u32.wrapping_mul(l_w) as isize),
          poshalf,
          3,
        );
      }
      i += 1;
      flagsp += 1;
      data = data.offset(1)
    }
    k += 4;
    flagsp += 2;
    data = data.offset(3_u32.wrapping_mul(l_w) as isize)
  }
  if k < t1.h {
    i = 0;
    while i < l_w {
      j = 0;
      while j < t1.h.wrapping_sub(k) {
        opj_t1_dec_refpass_step_raw(
          t1,
          flagsp,
          data.offset(j.wrapping_mul(l_w) as isize),
          poshalf,
          j,
        );
        j += 1
      }
      i += 1;
      flagsp += 1;
      data = data.offset(1)
    }
  }
}

fn opj_t1_dec_refpass_mqc_internal(
  mut t1: &mut opj_t1_t,
  mut bpno: OPJ_INT32,
  w: OPJ_UINT32,
  h: OPJ_UINT32,
  flags_stride: OPJ_UINT32,
) {
  let mut one = 0;
  let mut poshalf = 0;
  let mut i = 0;
  let mut j = 0;
  let mut k = 0;
  let mut data = t1.data_offset(0);
  let mut flagsp = t1.flags.offset(flags_stride as isize + 1);
  let l_w = w;
  let mqc = &mut t1.mqc;
  let mut v = 0;
  one = 1 << bpno;
  poshalf = one >> 1;

  k = 0;
  while k < h & !(0x03) {
    i = 0;
    while i < l_w {
      if *flagsp != 0 {
        opj_t1_dec_refpass_step_mqc_macro(flagsp, data, l_w, 0, mqc, &mut v, poshalf);
        opj_t1_dec_refpass_step_mqc_macro(flagsp, data, l_w, 1, mqc, &mut v, poshalf);
        opj_t1_dec_refpass_step_mqc_macro(flagsp, data, l_w, 2, mqc, &mut v, poshalf);
        opj_t1_dec_refpass_step_mqc_macro(flagsp, data, l_w, 3, mqc, &mut v, poshalf);
      }
      i += 1;
      data = data.offset(1);
      flagsp += 1
    }
    k += 4;
    data = data.offset(3_u32.wrapping_mul(l_w) as isize);
    flagsp += 2
  }
  if k < h {
    i = 0;
    while i < l_w {
      j = 0;
      while j < h.wrapping_sub(k) {
        opj_t1_dec_refpass_step_mqc(
          t1,
          flagsp,
          data.offset(j.wrapping_mul(l_w) as isize),
          poshalf,
          j,
        );
        j += 1
      }
      i += 1;
      data = data.offset(1);
      flagsp += 1
    }
  }
}

fn opj_t1_dec_refpass_mqc_64x64(mut t1: &mut opj_t1_t, mut bpno: OPJ_INT32) {
  opj_t1_dec_refpass_mqc_internal(t1, bpno, 64, 64, 66);
}

fn opj_t1_dec_refpass_mqc_generic(mut t1: &mut opj_t1_t, mut bpno: OPJ_INT32) {
  opj_t1_dec_refpass_mqc_internal(t1, bpno, t1.w, t1.h, t1.w + 2);
}

fn opj_t1_dec_refpass_mqc(mut t1: &mut opj_t1_t, mut bpno: OPJ_INT32) {
  if t1.w == 64 && t1.h == 64 {
    opj_t1_dec_refpass_mqc_64x64(t1, bpno);
  } else {
    opj_t1_dec_refpass_mqc_generic(t1, bpno);
  };
}

/**
Encode clean-up pass step
*/
#[inline]
fn opj_t1_enc_clnpass_step_macro(
  mqc: &mut opj_mqc_t,
  w: OPJ_UINT32,
  mut flagsp: FlagsPtr,
  mut l_datap: DataPtr,
  bpno: OPJ_INT32,
  one: OPJ_UINT32,
  nmsedec: *mut OPJ_INT32,
  agg: OPJ_BYTE,
  runlen: OPJ_UINT32,
  lim: OPJ_UINT32,
  cblksty: OPJ_UINT32,
) {
  const CHECK: opj_flag_t =
    T1_SIGMA_4 | T1_SIGMA_7 | T1_SIGMA_10 | T1_SIGMA_13 | T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3;
  unsafe {
    let mut v = 0;
    if (*flagsp & CHECK) == CHECK {
      if runlen == 0 {
        *flagsp &= !(T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
      } else if runlen == 1 {
        *flagsp &= !(T1_PI_1 | T1_PI_2 | T1_PI_3);
      } else if runlen == 2 {
        *flagsp &= !(T1_PI_2 | T1_PI_3);
      } else if runlen == 3 {
        *flagsp &= !(T1_PI_3);
      }
    } else {
      for ci in runlen..lim {
        let mut goto_PARTIAL = false;
        if agg != 0 && ci == runlen {
          goto_PARTIAL = true;
        } else if (*flagsp & ((T1_SIGMA_THIS | T1_PI_THIS) << ci.wrapping_mul(3))) == 0 {
          let ctxt1 = opj_t1_getctxno_zc(mqc, *flagsp >> ci.wrapping_mul(3));
          log::debug!("   ctxt1={}", ctxt1);
          opj_t1_setcurctx(mqc, ctxt1);
          v = if opj_smr_abs(*l_datap) & one != 0 {
            1
          } else {
            0
          };
          opj_mqc_encode_macro(mqc, v);
          if v != 0 {
            goto_PARTIAL = true;
          }
        }
        if goto_PARTIAL {
          let lu = opj_t1_getctxtno_sc_or_spb_index(*flagsp, flagsp[-1], flagsp[1], ci);
          *nmsedec += opj_t1_getnmsedec_sig(opj_smr_abs(*l_datap), bpno as u32) as i32;
          let ctxt2 = opj_t1_getctxno_sc(lu);
          log::debug!("   ctxt2={}", ctxt2);
          opj_t1_setcurctx(mqc, ctxt2);

          v = opj_smr_sign(*l_datap);
          let spb = opj_t1_getspb(lu);
          log::debug!("   spb={}", spb);
          opj_mqc_encode_macro(mqc, v ^ spb as u32);
          let vsc = if (cblksty & J2K_CCP_CBLKSTY_VSC) != 0 && ci == 0 {
            1
          } else {
            0
          };
          opj_t1_update_flags(flagsp, ci, v, w + 2, vsc);
        }
        *flagsp &= !(T1_PI_THIS << ci.wrapping_mul(3));
        l_datap = l_datap.offset(1);
      }
    }
  }
}

#[inline]
fn opj_t1_dec_clnpass_step_macro(
  check_flags: bool,
  partial: bool,
  flagsp: FlagsPtr,
  flags_stride: OPJ_UINT32,
  datap: DataPtr,
  data_stride: OPJ_UINT32,
  ci: OPJ_UINT32,
  mqc: &mut opj_mqc_t,
  mut v: OPJ_UINT32,
  oneplushalf: OPJ_INT32,
  vsc: OPJ_UINT32,
) {
  let flags = flagsp[0];
  if !check_flags || (flags & ((T1_SIGMA_THIS | T1_PI_THIS) << ci.wrapping_mul(3))) == 0 {
    if !partial {
      let ctxt1 = opj_t1_getctxno_zc(mqc, flags >> ci.wrapping_mul(3));
      opj_t1_setcurctx(mqc, ctxt1);
      opj_mqc_decode_macro(&mut v, mqc);
      if v == 0 {
        return;
      }
    }
    let mut lu = opj_t1_getctxtno_sc_or_spb_index(flags, flagsp[-1], flagsp[1], ci);
    opj_t1_setcurctx(mqc, opj_t1_getctxno_sc(lu));
    opj_mqc_decode_macro(&mut v, mqc);
    v ^= opj_t1_getspb(lu) as u32;
    *datap.offset(ci.wrapping_mul(data_stride) as isize) =
      if v != 0 { -oneplushalf } else { oneplushalf };
    opj_t1_update_flags_macro(flagsp, ci, v, flags_stride, vsc);
  }
}

fn opj_t1_dec_clnpass_step(
  mut t1: &mut opj_t1_t,
  mut flagsp: FlagsPtr,
  mut datap: DataPtr,
  mut oneplushalf: OPJ_INT32,
  mut ci: OPJ_UINT32,
  mut vsc: OPJ_UINT32,
) {
  let v = 0;
  let mqc = &mut t1.mqc; /* MQC component */

  opj_t1_dec_clnpass_step_macro(
    true,
    false,
    flagsp,
    t1.w + 2,
    datap,
    0,
    ci,
    mqc,
    v,
    oneplushalf,
    vsc,
  );
}

/* *
Encode clean-up pass
*/
fn opj_t1_enc_clnpass(
  mut t1: &mut opj_t1_t,
  mut bpno: OPJ_INT32,
  mut nmsedec: *mut OPJ_INT32,
  mut cblksty: OPJ_UINT32,
) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut k: OPJ_UINT32 = 0;
    let one = 1 << (bpno + T1_NMSEDEC_FRACBITS);
    let mut f = t1_flags(t1, 0, 0);
    let mut datap = t1.data_offset(0);
    let mqc = &mut t1.mqc;
    let extra = 2;

    *nmsedec = 0;
    log::debug!("enc_clnpass: bpno={}", bpno);
    k = 0;
    while k < (t1.h & !0x03) {
      log::debug!(" k={}", k);
      i = 0;
      while i < t1.w {
        log::debug!(" i={}", i);
        let mut agg = 0;
        let mut runlen = 0u32;
        agg = (*f == 0) as u8;
        log::debug!("   agg={}", agg);
        loop {
          if agg != 0 {
            runlen = 0;
            while runlen < 4 {
              if (opj_smr_abs(*datap) & one) != 0 {
                break;
              }
              runlen = runlen.wrapping_add(1);
              datap += 1
            }
            opj_t1_setcurctx(mqc, T1_CTXNO_AGG);
            opj_mqc_encode_macro(mqc, (runlen != 4) as u32);
            if runlen == 4 {
              break;
            }
            opj_t1_setcurctx(mqc, T1_CTXNO_UNI);
            opj_mqc_encode_macro(mqc, runlen >> 1);
            opj_mqc_encode_macro(mqc, runlen & 1);
          } else {
            runlen = 0;
          }
          opj_t1_enc_clnpass_step_macro(
            mqc, t1.w, f, datap, bpno, one, nmsedec, agg, runlen, 4, cblksty,
          );
          datap = datap.offset(4_u32.wrapping_sub(runlen) as isize);
          break;
        }
        i += 1;
        f += 1
      }
      k += 4;
      f += extra;
    }

    if k < t1.h {
      let agg = 0;
      let runlen = 0;
      log::debug!(" k={}", k);
      i = 0;
      while i < t1.w {
        log::debug!(" i={}", i);
        log::debug!("  agg={}", agg);
        opj_t1_enc_clnpass_step_macro(
          mqc,
          t1.w,
          f,
          datap,
          bpno,
          one,
          nmsedec,
          agg,
          runlen,
          t1.h - k,
          cblksty,
        );
        datap = datap.offset((t1.h - k) as isize);
        i += 1;
        f += 1
      }
    }
  }
}

fn opj_t1_dec_clnpass_internal(
  t1: &mut opj_t1_t,
  bpno: OPJ_INT32,
  vsc: bool,
  w: u32,
  h: u32,
  flags_stride: u32,
) {
  let mut one = 0;
  let mut half = 0;
  let mut oneplushalf = 0;
  let mut runlen = 0u32;
  let mut i = 0;
  let mut j = 0;
  let mut k = 0;
  let mut data = t1.data_offset(0);
  let mqc = &mut t1.mqc;
  let mut flagsp = t1.flags.offset(flags_stride as isize + 1);
  let l_w = w;
  let mut v = 0u32;
  one = 1 << bpno;
  half = one >> 1;
  oneplushalf = one | half;

  k = 0;
  while k < (h & !3) {
    i = 0;
    while i < l_w {
      if *flagsp == 0 {
        let mut partial = true;
        opj_t1_setcurctx(mqc, T1_CTXNO_AGG);
        opj_mqc_decode_macro(&mut v, mqc);
        if v == 0 {
          // continue;
        } else {
          opj_t1_setcurctx(mqc, T1_CTXNO_UNI);
          opj_mqc_decode_macro(&mut runlen, mqc);
          opj_mqc_decode_macro(&mut v, mqc);
          runlen = (runlen << 1) | v;
          if runlen == 0 {
            opj_t1_dec_clnpass_step_macro(
              false,
              true,
              flagsp,
              flags_stride,
              data,
              l_w,
              0,
              mqc,
              v,
              oneplushalf,
              vsc as u32,
            );
            partial = false;
            /* FALLTHRU */
          }
          if runlen <= 1 {
            opj_t1_dec_clnpass_step_macro(
              false,
              partial,
              flagsp,
              flags_stride,
              data,
              l_w,
              1,
              mqc,
              v,
              oneplushalf,
              false as u32,
            );
            partial = false;
            /* FALLTHRU */
          }
          if runlen <= 2 {
            opj_t1_dec_clnpass_step_macro(
              false,
              partial,
              flagsp,
              flags_stride,
              data,
              l_w,
              2,
              mqc,
              v,
              oneplushalf,
              false as u32,
            );
            partial = false;
            /* FALLTHRU */
          }
          if runlen <= 3 {
            opj_t1_dec_clnpass_step_macro(
              false,
              partial,
              flagsp,
              flags_stride,
              data,
              l_w,
              3,
              mqc,
              v,
              oneplushalf,
              false as u32,
            );
          }
          *flagsp &= !(T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
        }
      } else {
        opj_t1_dec_clnpass_step_macro(
          true,
          false,
          flagsp,
          flags_stride,
          data,
          l_w,
          0,
          mqc,
          v,
          oneplushalf,
          vsc as u32,
        );
        opj_t1_dec_clnpass_step_macro(
          true,
          false,
          flagsp,
          flags_stride,
          data,
          l_w,
          1,
          mqc,
          v,
          oneplushalf,
          false as u32,
        );
        opj_t1_dec_clnpass_step_macro(
          true,
          false,
          flagsp,
          flags_stride,
          data,
          l_w,
          2,
          mqc,
          v,
          oneplushalf,
          false as u32,
        );
        opj_t1_dec_clnpass_step_macro(
          true,
          false,
          flagsp,
          flags_stride,
          data,
          l_w,
          3,
          mqc,
          v,
          oneplushalf,
          false as u32,
        );
        *flagsp &= !(T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
      }
      i += 1;
      data = data.offset(1);
      flagsp += 1
    }
    k += 4;
    data = data.offset(3_u32.wrapping_mul(l_w) as isize);
    flagsp += 2;
  }
  if k < h {
    i = 0;
    while i < l_w {
      j = 0;
      while j < h - k {
        opj_t1_dec_clnpass_step(
          t1,
          flagsp,
          data.offset(j.wrapping_mul(l_w) as isize),
          oneplushalf,
          j,
          vsc as u32,
        );
        j += 1
      }
      *flagsp &= !(T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3);
      i += 1;
      flagsp += 1;
      data = data.offset(1)
    }
  }
}

fn opj_t1_dec_clnpass_check_segsym(mut t1: &mut opj_t1_t, mut cblksty: OPJ_INT32) {
  if (cblksty as u32 & J2K_CCP_CBLKSTY_SEGSYM) != 0 {
    let mqc = &mut t1.mqc;
    let mut v = 0;
    let mut v2 = 0;
    opj_t1_setcurctx(mqc, T1_CTXNO_UNI);
    opj_mqc_decode_macro(&mut v, mqc);
    opj_mqc_decode_macro(&mut v2, mqc);
    v = (v << 1) | v2;
    opj_mqc_decode_macro(&mut v2, mqc);
    v = (v << 1) | v2;
    opj_mqc_decode_macro(&mut v2, mqc);
    v = (v << 1) | v2;
    if v != 0xa {
      /*
      event_msg!(t1->cinfo, EVT_WARNING, "Bad segmentation symbol %x\n", v);
      */
    }
  }
}

fn opj_t1_dec_clnpass_64x64_novsc(t1: &mut opj_t1_t, mut bpno: OPJ_INT32) {
  opj_t1_dec_clnpass_internal(t1, bpno, false, 64, 64, 66);
}

fn opj_t1_dec_clnpass_64x64_vsc(t1: &mut opj_t1_t, mut bpno: OPJ_INT32) {
  opj_t1_dec_clnpass_internal(t1, bpno, true, 64, 64, 66);
}

fn opj_t1_dec_clnpass_generic_novsc(t1: &mut opj_t1_t, mut bpno: OPJ_INT32) {
  opj_t1_dec_clnpass_internal(t1, bpno, false, t1.w, t1.h, t1.w + 2);
}

fn opj_t1_dec_clnpass_generic_vsc(t1: &mut opj_t1_t, mut bpno: OPJ_INT32) {
  opj_t1_dec_clnpass_internal(t1, bpno, true, t1.w, t1.h, t1.w + 2);
}

fn opj_t1_dec_clnpass(mut t1: &mut opj_t1_t, mut bpno: OPJ_INT32, mut cblksty: OPJ_INT32) {
  if t1.w == 64 && t1.h == 64 {
    if (cblksty as u32 & J2K_CCP_CBLKSTY_VSC) != 0 {
      opj_t1_dec_clnpass_64x64_vsc(t1, bpno);
    } else {
      opj_t1_dec_clnpass_64x64_novsc(t1, bpno);
    }
  } else if (cblksty as u32 & J2K_CCP_CBLKSTY_VSC) != 0 {
    opj_t1_dec_clnpass_generic_vsc(t1, bpno);
  } else {
    opj_t1_dec_clnpass_generic_novsc(t1, bpno);
  }
  opj_t1_dec_clnpass_check_segsym(t1, cblksty);
}

fn opj_t1_getwmsedec(
  mut nmsedec: OPJ_INT32,
  mut compno: OPJ_UINT32,
  mut level: OPJ_UINT32,
  mut orient: OPJ_UINT32,
  mut bpno: OPJ_INT32,
  mut qmfbid: OPJ_UINT32,
  mut stepsize: OPJ_FLOAT64,
  mut _numcomps: OPJ_UINT32,
  mut mct_norms: *const OPJ_FLOAT64,
  mut mct_numcomps: OPJ_UINT32,
) -> OPJ_FLOAT64 {
  let mut w1 = 1.0;
  let mut w2 = 0.0;
  let mut wmsedec = 0.0;
  unsafe {
    if !mct_norms.is_null() && compno < mct_numcomps {
      w1 = *mct_norms.offset(compno as isize)
    }
  }
  if qmfbid == 1 {
    w2 = opj_dwt_getnorm(level, orient)
  } else {
    /* if (qmfbid == 0) */
    let log2_gain = if orient == 0 {
      0
    } else if orient == 3 {
      2
    } else {
      1
    };
    w2 = opj_dwt_getnorm_real(level, orient);
    /* Not sure this is right. But preserves past behaviour */
    stepsize /= (1 << log2_gain) as f64;
  }
  wmsedec = w1 * w2 * stepsize * (1 << bpno) as f64;
  wmsedec *= wmsedec * nmsedec as f64 / 8192.0f64;
  wmsedec
}

fn opj_t1_allocate_buffers(
  mut t1: &mut opj_t1_t,
  mut w: OPJ_UINT32,
  mut h: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut flagssize: OPJ_UINT32 = 0;
    let mut flags_stride: OPJ_UINT32 = 0;
    /* No risk of overflow. Prior checks ensure those assert are met */
    /* They are per the specification */

    assert!(w <= 1024);
    assert!(h <= 1024);
    assert!(w.wrapping_mul(h) <= 4096);
    /* encoder uses tile buffer, so no need to allocate */
    let datasize = w.wrapping_mul(h);
    t1.data.resize(datasize as usize);
    flags_stride = w.wrapping_add(2u32);
    flagssize = h.wrapping_add(3u32).wrapping_div(4u32).wrapping_add(2u32);
    flagssize = (flagssize as core::ffi::c_uint).wrapping_mul(flags_stride);
    let mut flags_height = h.wrapping_add(3u32).wrapping_div(4u32);
    t1.flags.resize(flagssize as usize);
    let p = t1.flags.as_mut_ptr().offset(0);
    for x in 0..flags_stride as isize {
      /* magic value to hopefully stop any passes being interested in this entry */
      *p.offset(x) = T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3;
    }
    let p = t1
      .flags
      .as_mut_ptr()
      .offset(flags_height.wrapping_add(1).wrapping_mul(flags_stride) as isize);
    for x in 0..flags_stride as isize {
      /* magic value to hopefully stop any passes being interested in this entry */
      *p.offset(x) = T1_PI_0 | T1_PI_1 | T1_PI_2 | T1_PI_3;
    }
    if h % 4 != 0 {
      let p = t1
        .flags
        .as_mut_ptr()
        .offset(flags_height.wrapping_mul(flags_stride) as isize);
      let v = if h % 4 == 1 {
        T1_PI_1 | T1_PI_2 | T1_PI_3
      } else if h % 4 == 2 {
        T1_PI_2 | T1_PI_3
      } else if h % 4 == 3 {
        T1_PI_3
      } else {
        0
      };
      for x in 0..flags_stride as isize {
        *p.offset(x) = v;
      }
    }
    t1.w = w;
    t1.h = h;
    1i32
  }
}

/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */

fn opj_t1_clbl_decode_processor(mut user_data: *mut core::ffi::c_void) {
  unsafe {
    let mut cblk = core::ptr::null_mut::<opj_tcd_cblk_dec_t>();
    let mut band = core::ptr::null_mut::<opj_tcd_band_t>();
    let mut tilec = core::ptr::null_mut::<opj_tcd_tilecomp_t>();
    let mut tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut datap = core::ptr::null_mut::<OPJ_INT32>();
    let mut cblk_w: OPJ_UINT32 = 0;
    let mut cblk_h: OPJ_UINT32 = 0;
    let mut x: OPJ_INT32 = 0;
    let mut y: OPJ_INT32 = 0;
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut job = core::ptr::null_mut::<opj_t1_cblk_decode_processing_job_t>();
    let mut resno: OPJ_UINT32 = 0;
    let mut tile_w: OPJ_UINT32 = 0;
    job = user_data as *mut opj_t1_cblk_decode_processing_job_t;
    cblk = (*job).cblk;
    if (*job).whole_tile_decoding == 0 {
      cblk_w = ((*cblk).x1 - (*cblk).x0) as OPJ_UINT32;
      cblk_h = ((*cblk).y1 - (*cblk).y0) as OPJ_UINT32;
      let layout = Layout::from_size_align_unchecked(
        core::mem::size_of::<OPJ_INT32>() * cblk_w as usize * cblk_h as usize,
        16,
      );
      (*cblk).decoded_data = alloc(layout) as *mut OPJ_INT32;
      (*cblk).decoded_data_layout = layout;
      if (*cblk).decoded_data.is_null() {
        event_msg!(
          (*job).p_manager,
          EVT_ERROR,
          "Cannot allocate cblk->decoded_data\n",
        );
        core::ptr::write_volatile((*job).pret, 0i32);
        opj_free(job as *mut core::ffi::c_void);
        return;
      }
      /* Zero-init required */
      memset(
        (*cblk).decoded_data as *mut core::ffi::c_void,
        0i32,
        core::mem::size_of::<OPJ_INT32>()
          .wrapping_mul(cblk_w as usize)
          .wrapping_mul(cblk_h as usize),
      );
    } else if !(*cblk).decoded_data.is_null() {
      /* Not sure if that code path can happen, but better be */
      /* safe than sorry */
      dealloc((*cblk).decoded_data as _, (*cblk).decoded_data_layout);
      (*cblk).decoded_data = core::ptr::null_mut::<OPJ_INT32>()
    }
    resno = (*job).resno;
    band = (*job).band;
    tilec = (*job).tilec;
    tccp = (*job).tccp;
    tile_w = ((*(*tilec)
      .resolutions
      .offset((*tilec).minimum_num_resolutions.wrapping_sub(1) as isize))
    .x1
      - (*(*tilec)
        .resolutions
        .offset((*tilec).minimum_num_resolutions.wrapping_sub(1) as isize))
      .x0) as OPJ_UINT32;
    if *(*job).pret == 0 {
      opj_free(job as *mut core::ffi::c_void);
      return;
    }

    // Use thread local t1 instance.
    T1.with(|l_t1| {
      let mut ref_t1 = l_t1.borrow_mut();
      let t1 = ref_t1.deref_mut();

      t1.mustuse_cblkdatabuffer = (*job).mustuse_cblkdatabuffer;
      if (*tccp).cblksty & J2K_CCP_CBLKSTY_HT != 0 {
        if 0i32
          == opj_t1_ht_decode_cblk(
            t1,
            cblk,
            (*band).bandno,
            (*tccp).roishift as OPJ_UINT32,
            (*tccp).cblksty,
            &mut (*job).p_manager,
            (*job).check_pterm,
          )
        {
          core::ptr::write_volatile((*job).pret, 0i32);
          opj_free(job as *mut core::ffi::c_void);
          return;
        }
      } else if 0i32
        == opj_t1_decode_cblk(
          t1,
          cblk,
          (*band).bandno,
          (*tccp).roishift as OPJ_UINT32,
          (*tccp).cblksty,
          &mut (*job).p_manager,
          (*job).check_pterm,
        )
      {
        core::ptr::write_volatile((*job).pret, 0i32);
        opj_free(job as *mut core::ffi::c_void);
        return;
      }
      x = (*cblk).x0 - (*band).x0;
      y = (*cblk).y0 - (*band).y0;
      if (*band).bandno & 1 != 0 {
        let mut pres: *mut opj_tcd_resolution_t = &mut *(*tilec)
          .resolutions
          .offset(resno.wrapping_sub(1) as isize)
          as *mut opj_tcd_resolution_t;
        x += (*pres).x1 - (*pres).x0
      }
      if (*band).bandno & 2 != 0 {
        let mut pres_0: *mut opj_tcd_resolution_t = &mut *(*tilec)
          .resolutions
          .offset(resno.wrapping_sub(1) as isize)
          as *mut opj_tcd_resolution_t;
        y += (*pres_0).y1 - (*pres_0).y0
      }
      datap = if !(*cblk).decoded_data.is_null() {
        (*cblk).decoded_data
      } else {
        t1.data.as_mut_ptr()
      };
      cblk_w = t1.w;
      cblk_h = t1.h;
      if (*tccp).roishift != 0 {
        if (*tccp).roishift >= 31i32 {
          j = 0 as OPJ_UINT32;
          while j < cblk_h {
            i = 0 as OPJ_UINT32;
            while i < cblk_w {
              *datap.offset(j.wrapping_mul(cblk_w).wrapping_add(i) as isize) = 0i32;
              i += 1
            }
            j += 1
          }
        } else {
          let mut thresh = (1i32) << (*tccp).roishift;
          j = 0 as OPJ_UINT32;
          while j < cblk_h {
            i = 0 as OPJ_UINT32;
            while i < cblk_w {
              let mut val = *datap.offset(j.wrapping_mul(cblk_w).wrapping_add(i) as isize);
              let mut mag = val.abs();
              if mag >= thresh {
                mag >>= (*tccp).roishift;
                *datap.offset(j.wrapping_mul(cblk_w).wrapping_add(i) as isize) =
                  if val < 0i32 { -mag } else { mag }
              }
              i += 1
            }
            j += 1
          }
        }
      }
      /* Both can be non NULL if for example decoding a full tile and then */
      /* partially a tile. In which case partial decoding should be the */
      /* priority */
      assert!(!(*cblk).decoded_data.is_null() || !(*tilec).data.is_null()); /* if (tccp->qmfbid == 0) */
      if !(*cblk).decoded_data.is_null() {
        let mut cblk_size = cblk_w.wrapping_mul(cblk_h); /* resno */
        if (*tccp).qmfbid == 1 {
          i = 0 as OPJ_UINT32;
          while i < cblk_size {
            *datap.offset(i as isize) /= 2i32;
            i += 1
          }
        } else {
          let stepsize = 0.5f32 * (*band).stepsize;
          i = 0 as OPJ_UINT32;
          while i < cblk_size {
            let mut tmp = *datap as OPJ_FLOAT32 * stepsize;
            memcpy(
              datap as *mut core::ffi::c_void,
              &mut tmp as *mut OPJ_FLOAT32 as *const core::ffi::c_void,
              core::mem::size_of::<OPJ_FLOAT32>(),
            );
            datap = datap.offset(1);
            i += 1
          }
        }
      } else if (*tccp).qmfbid == 1 {
        let mut tiledp: *mut OPJ_INT32 = &mut *(*tilec).data.add((y as OPJ_SIZE_T)
            .wrapping_mul(tile_w as usize)
            .wrapping_add(x as OPJ_SIZE_T)) as *mut OPJ_INT32;
        j = 0 as OPJ_UINT32;
        while j < cblk_h {
          i = 0 as OPJ_UINT32;
          while i < cblk_w & !(3u32) {
            let mut tmp0 = *datap.offset(
              j.wrapping_mul(cblk_w)
                .wrapping_add(i)
                .wrapping_add(0u32) as isize,
            );
            let mut tmp1 = *datap.offset(
              j.wrapping_mul(cblk_w)
                .wrapping_add(i)
                .wrapping_add(1u32) as isize,
            );
            let mut tmp2 = *datap.offset(
              j.wrapping_mul(cblk_w)
                .wrapping_add(i)
                .wrapping_add(2u32) as isize,
            );
            let mut tmp3 = *datap.offset(
              j.wrapping_mul(cblk_w)
                .wrapping_add(i)
                .wrapping_add(3u32) as isize,
            );
            *tiledp.add((j as usize)
                .wrapping_mul(tile_w as OPJ_SIZE_T)
                .wrapping_add(i as usize)
                .wrapping_add(0)) = tmp0 / 2i32;
            *tiledp.add((j as usize)
                .wrapping_mul(tile_w as OPJ_SIZE_T)
                .wrapping_add(i as usize)
                .wrapping_add(1)) = tmp1 / 2i32;
            *tiledp.add((j as usize)
                .wrapping_mul(tile_w as OPJ_SIZE_T)
                .wrapping_add(i as usize)
                .wrapping_add(2)) = tmp2 / 2i32;
            *tiledp.add((j as usize)
                .wrapping_mul(tile_w as OPJ_SIZE_T)
                .wrapping_add(i as usize)
                .wrapping_add(3)) = tmp3 / 2i32;
            i = (i as core::ffi::c_uint).wrapping_add(4u32)
          }
          while i < cblk_w {
            let mut tmp_0 = *datap.offset(j.wrapping_mul(cblk_w).wrapping_add(i) as isize);
            *tiledp.add((j as usize)
                .wrapping_mul(tile_w as OPJ_SIZE_T)
                .wrapping_add(i as usize)) = tmp_0 / 2i32;
            i += 1
          }
          j += 1
        }
      } else {
        let stepsize_0 = 0.5f32 * (*band).stepsize;
        let mut tiledp_0 = &mut *(*tilec).data.add((y as OPJ_SIZE_T)
            .wrapping_mul(tile_w as usize)
            .wrapping_add(x as OPJ_SIZE_T)) as *mut OPJ_INT32 as *mut OPJ_FLOAT32;
        j = 0 as OPJ_UINT32;
        while j < cblk_h {
          let mut tiledp2 = tiledp_0;
          i = 0 as OPJ_UINT32;
          while i < cblk_w {
            let mut tmp_1 = *datap as OPJ_FLOAT32 * stepsize_0;
            *tiledp2 = tmp_1;
            datap = datap.offset(1);
            tiledp2 = tiledp2.offset(1);
            i += 1
          }
          tiledp_0 = tiledp_0.offset(tile_w as isize);
          j += 1
        }
      }
      opj_free(job as *mut core::ffi::c_void);
    })
  }
}

pub(crate) fn opj_t1_decode_cblks(
  mut tcd: &mut opj_tcd,
  mut pret: *mut OPJ_BOOL,
  mut tilec: *mut opj_tcd_tilecomp_t,
  mut tccp: *mut opj_tccp_t,
  mut p_manager: &mut opj_event_mgr,
  mut check_pterm: OPJ_BOOL,
) {
  unsafe {
    let mut resno: OPJ_UINT32 = 0;
    let mut bandno: OPJ_UINT32 = 0;
    let mut precno: OPJ_UINT32 = 0;
    let mut cblkno: OPJ_UINT32 = 0;
    resno = 0 as OPJ_UINT32;
    while resno < (*tilec).minimum_num_resolutions {
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
          if opj_tcd_is_subband_area_of_interest(
            tcd,
            (*tilec).compno,
            resno,
            (*band).bandno,
            (*precinct).x0 as OPJ_UINT32,
            (*precinct).y0 as OPJ_UINT32,
            (*precinct).x1 as OPJ_UINT32,
            (*precinct).y1 as OPJ_UINT32,
          ) == 0
          {
            cblkno = 0 as OPJ_UINT32;
            while cblkno < (*precinct).cw.wrapping_mul((*precinct).ch) {
              let mut cblk: *mut opj_tcd_cblk_dec_t =
                &mut *(*precinct).cblks.dec.offset(cblkno as isize) as *mut opj_tcd_cblk_dec_t;
              if !(*cblk).decoded_data.is_null() {
                dealloc((*cblk).decoded_data as _, (*cblk).decoded_data_layout);
                (*cblk).decoded_data = core::ptr::null_mut::<OPJ_INT32>()
              }
              cblkno += 1;
            }
          } else {
            let mut current_block_34: u64;
            cblkno = 0 as OPJ_UINT32;
            while cblkno < (*precinct).cw.wrapping_mul((*precinct).ch) {
              let mut cblk_0: *mut opj_tcd_cblk_dec_t =
                &mut *(*precinct).cblks.dec.offset(cblkno as isize) as *mut opj_tcd_cblk_dec_t;
              let mut job = core::ptr::null_mut::<opj_t1_cblk_decode_processing_job_t>();
              if opj_tcd_is_subband_area_of_interest(
                tcd,
                (*tilec).compno,
                resno,
                (*band).bandno,
                (*cblk_0).x0 as OPJ_UINT32,
                (*cblk_0).y0 as OPJ_UINT32,
                (*cblk_0).x1 as OPJ_UINT32,
                (*cblk_0).y1 as OPJ_UINT32,
              ) == 0
              {
                if !(*cblk_0).decoded_data.is_null() {
                  dealloc((*cblk_0).decoded_data as _, (*cblk_0).decoded_data_layout);
                  (*cblk_0).decoded_data = core::ptr::null_mut::<OPJ_INT32>()
                }
              } else {
                if (*tcd).whole_tile_decoding == 0 {
                  let mut cblk_w = ((*cblk_0).x1 - (*cblk_0).x0) as OPJ_UINT32;
                  let mut cblk_h = ((*cblk_0).y1 - (*cblk_0).y0) as OPJ_UINT32;
                  if !(*cblk_0).decoded_data.is_null() {
                    current_block_34 = 2370887241019905314;
                  } else if cblk_w == 0 || cblk_h == 0 {
                    current_block_34 = 2370887241019905314;
                  } else {
                    current_block_34 = 11913429853522160501;
                  }
                } else {
                  current_block_34 = 11913429853522160501;
                }
                match current_block_34 {
                  2370887241019905314 => {}
                  _ => {
                    job = opj_calloc(
                      1i32 as size_t,
                      core::mem::size_of::<opj_t1_cblk_decode_processing_job_t>(),
                    ) as *mut opj_t1_cblk_decode_processing_job_t;
                    if job.is_null() {
                      core::ptr::write_volatile(pret, 0i32);
                      return;
                    }
                    (*job).whole_tile_decoding = (*tcd).whole_tile_decoding;
                    (*job).resno = resno;
                    (*job).cblk = cblk_0;
                    (*job).band = band;
                    (*job).tilec = tilec;
                    (*job).tccp = tccp;
                    (*job).pret = pret;
                    (*job).p_manager = *p_manager;
                    (*job).check_pterm = check_pterm;
                    (*job).mustuse_cblkdatabuffer = 0;
                    opj_t1_clbl_decode_processor(job as _);
                    if *pret == 0 {
                      return;
                    }
                  }
                }
              }
              cblkno += 1;
            }
          }
          precno += 1;
          /* bandno */
          /* precno */
          /* cblkno */
        }
        bandno += 1;
      }
      resno += 1;
    }
  }
}

/* *
Decode 1 code-block
@param t1 T1 handle
@param cblk Code-block coding parameters
@param orient
@param roishift Region of interest shifting value
@param cblksty Code-block style
@param p_manager the event manager
@param check_pterm whether PTERM correct termination should be checked
*/
fn opj_t1_decode_cblk(
  mut t1: &mut opj_t1_t,
  mut cblk: *mut opj_tcd_cblk_dec_t,
  mut orient: OPJ_UINT32,
  mut roishift: OPJ_UINT32,
  mut cblksty: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
  mut check_pterm: OPJ_BOOL,
) -> OPJ_BOOL {
  unsafe {
    let mut bpno_plus_one: OPJ_INT32 = 0; /* BYPASS mode */
    let mut passtype: OPJ_UINT32 = 0;
    let mut segno: OPJ_UINT32 = 0;
    let mut passno: OPJ_UINT32 = 0;
    let mut cblkdata = core::ptr::null_mut::<OPJ_BYTE>();
    let mut cblkdataindex = 0 as OPJ_UINT32;
    let mut type_0 = 0 as OPJ_BYTE;
    t1.mqc.lut_ctxno_zc_orient = &lut_ctxno_zc[orient as usize];
    if opj_t1_allocate_buffers(
      t1,
      ((*cblk).x1 - (*cblk).x0) as OPJ_UINT32,
      ((*cblk).y1 - (*cblk).y0) as OPJ_UINT32,
    ) == 0
    {
      return 0i32;
    }
    bpno_plus_one = roishift.wrapping_add((*cblk).numbps) as OPJ_INT32;
    if bpno_plus_one >= 31i32 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "opj_t1_decode_cblk(): unsupported bpno_plus_one = %d >= 31\n",
        bpno_plus_one,
      );
      return 0i32;
    }
    passtype = 2;

    opj_mqc_resetstates(&mut t1.mqc);
    opj_mqc_setstate(&mut t1.mqc, T1_CTXNO_UNI, 0, 46);
    opj_mqc_setstate(&mut t1.mqc, T1_CTXNO_AGG, 0, 3);
    opj_mqc_setstate(&mut t1.mqc, T1_CTXNO_ZC, 0, 4);

    /* Even if we have a single chunk, in multi-threaded decoding */
    /* the insertion of our synthetic marker might potentially override */
    /* valid codestream of other codeblocks decoded in parallel. */
    if (*cblk).numchunks > 1 || t1.mustuse_cblkdatabuffer != 0 {
      let mut i: OPJ_UINT32 = 0;
      let mut cblk_len: OPJ_UINT32 = 0;
      /* Compute whole codeblock length from chunk lengths */
      cblk_len = 0 as OPJ_UINT32;
      i = 0 as OPJ_UINT32;
      while i < (*cblk).numchunks {
        cblk_len = (cblk_len as core::ffi::c_uint)
          .wrapping_add((*(*cblk).chunks.offset(i as isize)).len) as OPJ_UINT32;
        i += 1
      }
      /* Allocate temporary memory if needed */
      if cblk_len.wrapping_add(2) > t1.cblkdatabuffersize {
        cblkdata = opj_realloc(
          t1.cblkdatabuffer as *mut core::ffi::c_void,
          cblk_len.wrapping_add(2) as size_t,
        ) as *mut OPJ_BYTE;
        if cblkdata.is_null() {
          return 0i32;
        }
        t1.cblkdatabuffer = cblkdata;
        memset(
          t1.cblkdatabuffer.offset(cblk_len as isize) as *mut core::ffi::c_void,
          0,
          OPJ_COMMON_CBLK_DATA_EXTRA as usize,
        );
        t1.cblkdatabuffersize = cblk_len.wrapping_add(2)
      }
      /* Concatenate all chunks */
      cblkdata = t1.cblkdatabuffer;
      cblk_len = 0 as OPJ_UINT32;
      i = 0 as OPJ_UINT32;
      while i < (*cblk).numchunks {
        memcpy(
          cblkdata.offset(cblk_len as isize) as *mut core::ffi::c_void,
          (*(*cblk).chunks.offset(i as isize)).data as *const core::ffi::c_void,
          (*(*cblk).chunks.offset(i as isize)).len as usize,
        );
        cblk_len = (cblk_len as core::ffi::c_uint)
          .wrapping_add((*(*cblk).chunks.offset(i as isize)).len) as OPJ_UINT32;
        i += 1
      }
    } else if (*cblk).numchunks == 1 {
      cblkdata = (*(*cblk).chunks.offset(0)).data
    } else {
      /* Not sure if that can happen in practice, but avoid Coverity to */
      /* think we will dereference a null cblkdta pointer */
      return 1i32;
    }
    /* For subtile decoding, directly decode in the decoded_data buffer of */
    /* the code-block. Hack t1->data to point to it, and restore it later */
    if !(*cblk).decoded_data.is_null() {
      t1.set_decoded_data((*cblk).decoded_data);
    }
    segno = 0 as OPJ_UINT32;
    while segno < (*cblk).real_num_segs {
      let mut seg: *mut opj_tcd_seg_t =
        &mut *(*cblk).segs.offset(segno as isize) as *mut opj_tcd_seg_t;
      /* BYPASS mode */
      type_0 = if bpno_plus_one <= (*cblk).numbps as OPJ_INT32 - 4i32
        && passtype < 2
        && cblksty & 0x1 != 0
      {
        1i32
      } else {
        0i32
      } as OPJ_BYTE;
      if type_0 as core::ffi::c_int == 1i32 {
        opj_mqc_raw_init_dec(
          &mut t1.mqc,
          cblkdata.offset(cblkdataindex as isize),
          (*seg).len,
          2 as OPJ_UINT32,
        );
      } else {
        opj_mqc_init_dec(
          &mut t1.mqc,
          cblkdata.offset(cblkdataindex as isize),
          (*seg).len,
          2 as OPJ_UINT32,
        );
      }
      cblkdataindex = (cblkdataindex as core::ffi::c_uint).wrapping_add((*seg).len);
      passno = 0 as OPJ_UINT32;
      while passno < (*seg).real_num_passes && bpno_plus_one >= 1i32 {
        match passtype {
          0 => {
            if type_0 as core::ffi::c_int == 1i32 {
              opj_t1_dec_sigpass_raw(t1, bpno_plus_one, cblksty as OPJ_INT32);
            } else {
              opj_t1_dec_sigpass_mqc(t1, bpno_plus_one, cblksty as OPJ_INT32);
            }
          }
          1 => {
            if type_0 as core::ffi::c_int == 1i32 {
              opj_t1_dec_refpass_raw(t1, bpno_plus_one);
            } else {
              opj_t1_dec_refpass_mqc(t1, bpno_plus_one);
            }
          }
          2 => {
            opj_t1_dec_clnpass(t1, bpno_plus_one, cblksty as OPJ_INT32);
          }
          _ => {}
        }
        if (cblksty & J2K_CCP_CBLKSTY_RESET) != 0 && type_0 == T1_TYPE_MQ {
          opj_mqc_resetstates(&mut t1.mqc);
          opj_mqc_setstate(&mut t1.mqc, T1_CTXNO_UNI, 0, 46);
          opj_mqc_setstate(&mut t1.mqc, T1_CTXNO_AGG, 0, 3);
          opj_mqc_setstate(&mut t1.mqc, T1_CTXNO_ZC, 0, 4);
        }
        passtype = passtype.wrapping_add(1);
        if passtype == 3 {
          passtype = 0;
          bpno_plus_one -= 1
        }
        passno += 1;
      }
      opq_mqc_finish_dec(&mut t1.mqc);
      segno += 1;
    }
    if check_pterm != 0 {
      let mqc = &mut t1.mqc; /* MQC component */
      if mqc.bp.offset(2) < mqc.end {
        event_msg!(
          p_manager,
          EVT_WARNING,
          "PTERM check failure: %d remaining bytes in code block (%d used / %d)\n",
          mqc.end.offset_from(mqc.bp) as core::ffi::c_int - 2i32,
          mqc.bp.offset_from(mqc.start) as core::ffi::c_int,
          mqc.end.offset_from(mqc.start) as core::ffi::c_int,
        );
      } else if mqc.end_of_byte_stream_counter > 2 {
        event_msg!(
          p_manager,
          EVT_WARNING,
          "PTERM check failure: %d synthetized 0xFF markers read\n",
          mqc.end_of_byte_stream_counter,
        );
      }
    }
    /* Restore original t1->data is needed */
    if !(*cblk).decoded_data.is_null() {
      t1.reset_decoded_data();
    }
    1i32
  }
}

/* * Procedure to deal with a asynchronous code-block encoding job.
 *
 * @param user_data Pointer to a opj_t1_cblk_encode_processing_job_t* structure
 * @param tls       TLS handle.
 */
fn opj_t1_cblk_encode_processor(mut user_data: *mut core::ffi::c_void) {
  unsafe {
    let mut job = user_data as *mut opj_t1_cblk_encode_processing_job_t; /* OPJ_TRUE == T1 for encoding */
    let mut cblk = (*job).cblk; /* if (tccp->qmfbid == 0) */
    let mut band: *const opj_tcd_band_t = (*job).band;
    let mut tilec: *const opj_tcd_tilecomp_t = (*job).tilec;
    let mut tccp: *const opj_tccp_t = (*job).tccp;
    let resno = (*job).resno;
    let tile_w = ((*tilec).x1 - (*tilec).x0) as OPJ_UINT32;
    let mut tiledp = core::ptr::null_mut::<OPJ_INT32>();
    let mut cblk_w: OPJ_UINT32 = 0;
    let mut cblk_h: OPJ_UINT32 = 0;
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut x = (*cblk).x0 - (*band).x0;
    let mut y = (*cblk).y0 - (*band).y0;
    if *(*job).pret == 0 {
      opj_free(job as *mut core::ffi::c_void);
      return;
    }

    // Use thread local t1 instance.
    T1.with(|l_t1| {
      let mut ref_t1 = l_t1.borrow_mut();
      let t1 = ref_t1.deref_mut();

      if (*band).bandno & 1 != 0 {
        let mut pres: *mut opj_tcd_resolution_t =
          &mut *(*tilec).resolutions.offset(resno.wrapping_sub(1) as isize)
            as *mut opj_tcd_resolution_t;
        x += (*pres).x1 - (*pres).x0
      }
      if (*band).bandno & 2 != 0 {
        let mut pres_0: *mut opj_tcd_resolution_t =
          &mut *(*tilec).resolutions.offset(resno.wrapping_sub(1) as isize)
            as *mut opj_tcd_resolution_t;
        y += (*pres_0).y1 - (*pres_0).y0
      }
      if opj_t1_allocate_buffers(
        t1,
        ((*cblk).x1 - (*cblk).x0) as OPJ_UINT32,
        ((*cblk).y1 - (*cblk).y0) as OPJ_UINT32,
      ) == 0
      {
        core::ptr::write_volatile((*job).pret, 0i32);
        opj_free(job as *mut core::ffi::c_void);
        return;
      }
      cblk_w = t1.w;
      cblk_h = t1.h;
      tiledp = &mut *(*tilec).data.add(
        (y as OPJ_SIZE_T)
          .wrapping_mul(tile_w as usize)
          .wrapping_add(x as OPJ_SIZE_T),
      ) as *mut OPJ_INT32;
      if (*tccp).qmfbid == 1 {
        let mut tiledp_u = tiledp as *mut OPJ_UINT32;
        let mut t1data = t1.data.as_mut_ptr() as *mut OPJ_UINT32;
        /* Do multiplication on unsigned type, even if the
         * underlying type is signed, to avoid potential
         * int overflow on large value (the output will be
         * incorrect in such situation, but whatever...)
         * This assumes complement-to-2 signed integer
         * representation
         * Fixes https://github.com/uclouvain/openjpeg/issues/1053
         */
        j = 0 as OPJ_UINT32;
        while j < cblk_h & !(3u32) {
          i = 0 as OPJ_UINT32;
          while i < cblk_w {
            *t1data.offset(0) = *tiledp_u
              .offset(j.wrapping_add(0).wrapping_mul(tile_w).wrapping_add(i) as isize)
              << (7i32 - 1i32);
            *t1data.offset(1) = *tiledp_u
              .offset(j.wrapping_add(1).wrapping_mul(tile_w).wrapping_add(i) as isize)
              << (7i32 - 1i32);
            *t1data.offset(2) = *tiledp_u
              .offset(j.wrapping_add(2).wrapping_mul(tile_w).wrapping_add(i) as isize)
              << (7i32 - 1i32);
            *t1data.offset(3) = *tiledp_u
              .offset(j.wrapping_add(3).wrapping_mul(tile_w).wrapping_add(i) as isize)
              << (7i32 - 1i32);
            t1data = t1data.offset(4);
            i += 1
          }
          j = (j as core::ffi::c_uint).wrapping_add(4) as OPJ_UINT32 as OPJ_UINT32
        }
        if j < cblk_h {
          i = 0 as OPJ_UINT32;
          while i < cblk_w {
            let mut k: OPJ_UINT32 = 0;
            k = j;
            while k < cblk_h {
              *t1data.offset(0) =
                *tiledp_u.offset(k.wrapping_mul(tile_w).wrapping_add(i) as isize) << (7i32 - 1i32);
              t1data = t1data.offset(1);
              k += 1
            }
            i += 1
          }
        }
      } else {
        let mut tiledp_f = tiledp as *mut OPJ_FLOAT32;
        let mut t1data_0 = t1.data_offset(0);
        /* Change from "natural" order to "zigzag" order of T1 passes */
        /* Change from "natural" order to "zigzag" order of T1 passes */
        j = 0 as OPJ_UINT32; /* fixed_quality */
        while j < cblk_h & !(3u32) {
          i = 0 as OPJ_UINT32; /* compno  */
          while i < cblk_w {
            *t1data_0.offset(0) = opj_lrintf(
              *tiledp_f.offset(j.wrapping_add(0).wrapping_mul(tile_w).wrapping_add(i) as isize)
                / (*band).stepsize
                * ((1i32) << (7i32 - 1i32)) as core::ffi::c_float,
            ) as OPJ_INT32;
            *t1data_0.offset(1) = opj_lrintf(
              *tiledp_f.offset(j.wrapping_add(1).wrapping_mul(tile_w).wrapping_add(i) as isize)
                / (*band).stepsize
                * ((1i32) << (7i32 - 1i32)) as core::ffi::c_float,
            ) as OPJ_INT32;
            *t1data_0.offset(2) = opj_lrintf(
              *tiledp_f.offset(j.wrapping_add(2).wrapping_mul(tile_w).wrapping_add(i) as isize)
                / (*band).stepsize
                * ((1i32) << (7i32 - 1i32)) as core::ffi::c_float,
            ) as OPJ_INT32;
            *t1data_0.offset(3) = opj_lrintf(
              *tiledp_f.offset(j.wrapping_add(3).wrapping_mul(tile_w).wrapping_add(i) as isize)
                / (*band).stepsize
                * ((1i32) << (7i32 - 1i32)) as core::ffi::c_float,
            ) as OPJ_INT32;
            t1data_0 = t1data_0.offset(4);
            i += 1
          }
          j = (j as core::ffi::c_uint).wrapping_add(4) as OPJ_UINT32 as OPJ_UINT32
        }
        if j < cblk_h {
          i = 0 as OPJ_UINT32;
          while i < cblk_w {
            let mut k_0: OPJ_UINT32 = 0;
            k_0 = j;
            while k_0 < cblk_h {
              *t1data_0.offset(0) = opj_lrintf(
                *tiledp_f.offset(k_0.wrapping_mul(tile_w).wrapping_add(i) as isize)
                  / (*band).stepsize
                  * ((1i32) << (7i32 - 1i32)) as core::ffi::c_float,
              ) as OPJ_INT32;
              t1data_0 = t1data_0.offset(1);
              k_0 += 1;
            }
            i += 1
          }
        }
      }
      let mut cumwmsedec = opj_t1_encode_cblk(
        t1,
        cblk,
        (*band).bandno,
        (*job).compno,
        (*tilec).numresolutions.wrapping_sub(1).wrapping_sub(resno),
        (*tccp).qmfbid,
        (*band).stepsize as OPJ_FLOAT64,
        (*tccp).cblksty,
        (*(*job).tile).numcomps,
        (*job).mct_norms,
        (*job).mct_numcomps,
      );
      (*(*job).tile).distotile += cumwmsedec;
      opj_free(job as *mut core::ffi::c_void);
    })
  }
}

pub(crate) fn opj_t1_encode_cblks(
  mut tile: *mut opj_tcd_tile_t,
  mut tcp: *mut opj_tcp_t,
  mut mct_norms: *const OPJ_FLOAT64,
  mut mct_numcomps: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut ret = 1i32;
    let mut compno: OPJ_UINT32 = 0;
    let mut resno: OPJ_UINT32 = 0;
    let mut bandno: OPJ_UINT32 = 0;
    let mut precno: OPJ_UINT32 = 0;
    let mut cblkno: OPJ_UINT32 = 0;
    (*tile).distotile = 0 as OPJ_FLOAT64;
    compno = 0 as OPJ_UINT32;
    's_19: while compno < (*tile).numcomps {
      let mut tilec: *mut opj_tcd_tilecomp_t =
        &mut *(*tile).comps.offset(compno as isize) as *mut opj_tcd_tilecomp_t;
      let mut tccp: *mut opj_tccp_t = &mut *(*tcp).tccps.offset(compno as isize) as *mut opj_tccp_t;
      resno = 0 as OPJ_UINT32;
      while resno < (*tilec).numresolutions {
        let mut res: *mut opj_tcd_resolution_t =
          &mut *(*tilec).resolutions.offset(resno as isize) as *mut opj_tcd_resolution_t;
        bandno = 0 as OPJ_UINT32;
        while bandno < (*res).numbands {
          let mut band: *mut opj_tcd_band_t =
            &mut *(*res).bands.as_mut_ptr().offset(bandno as isize) as *mut opj_tcd_band_t;
          /* resno  */
          /* bandno */
          /* precno */
          /* Skip empty bands */
          if opj_tcd_is_band_empty(band) == 0 {
            precno = 0 as OPJ_UINT32;
            while precno < (*res).pw.wrapping_mul((*res).ph) {
              let mut prc: *mut opj_tcd_precinct_t =
                &mut *(*band).precincts.offset(precno as isize) as *mut opj_tcd_precinct_t;
              cblkno = 0 as OPJ_UINT32;
              while cblkno < (*prc).cw.wrapping_mul((*prc).ch) {
                let mut cblk: *mut opj_tcd_cblk_enc_t =
                  &mut *(*prc).cblks.enc.offset(cblkno as isize) as *mut opj_tcd_cblk_enc_t;
                let mut job = opj_calloc(
                  1i32 as size_t,
                  core::mem::size_of::<opj_t1_cblk_encode_processing_job_t>(),
                ) as *mut opj_t1_cblk_encode_processing_job_t;
                if job.is_null() {
                  core::ptr::write_volatile(&mut ret as *mut OPJ_BOOL, 0i32);
                  break 's_19;
                } else {
                  (*job).compno = compno;
                  (*job).tile = tile;
                  (*job).resno = resno;
                  (*job).cblk = cblk;
                  (*job).band = band;
                  (*job).tilec = tilec;
                  (*job).tccp = tccp;
                  (*job).mct_norms = mct_norms;
                  (*job).mct_numcomps = mct_numcomps;
                  (*job).pret = &mut ret;
                  opj_t1_cblk_encode_processor(job as _);
                  cblkno += 1;
                }
              }
              precno += 1;
              /* cblkno */
            }
          }
          bandno += 1;
        }
        resno += 1;
      }
      compno += 1;
    }
    ret
  }
}

/* Returns whether the pass (bpno, passtype) is terminated */
fn opj_t1_enc_is_term_pass(
  mut cblk: *mut opj_tcd_cblk_enc_t,
  mut cblksty: OPJ_UINT32,
  mut bpno: OPJ_INT32,
  mut passtype: OPJ_UINT32,
) -> core::ffi::c_int {
  unsafe {
    /* Is it the last cleanup pass ? */
    if passtype == 2 && bpno == 0i32 {
      return 1i32;
    }
    if cblksty & 0x4 != 0 {
      return 1i32;
    }
    if cblksty & 0x1 != 0 {
      /* For bypass arithmetic bypass, terminate the 4th cleanup pass */
      if bpno == (*cblk).numbps as OPJ_INT32 - 4i32 && passtype == 2 {
        return 1i32;
      }
      /* and beyond terminate all the magnitude refinement passes (in raw) */
      /* and cleanup passes (in MQC) */
      if bpno < (*cblk).numbps as OPJ_INT32 - 4i32 && passtype > 0 {
        return 1i32;
      }
    }
    0i32
  }
}

/* * Return "cumwmsedec" that should be used to increase tile->distotile */
/* * mod fixed_quality */
fn opj_t1_encode_cblk(
  mut t1: &mut opj_t1_t,
  mut cblk: *mut opj_tcd_cblk_enc_t,
  mut orient: OPJ_UINT32,
  mut compno: OPJ_UINT32,
  mut level: OPJ_UINT32,
  mut qmfbid: OPJ_UINT32,
  mut stepsize: OPJ_FLOAT64,
  mut cblksty: OPJ_UINT32,
  mut numcomps: OPJ_UINT32,
  mut mct_norms: *const OPJ_FLOAT64,
  mut mct_numcomps: OPJ_UINT32,
) -> core::ffi::c_double {
  unsafe {
    let mut cumwmsedec = 0.0f64; /* MQC component */
    let mut passno: OPJ_UINT32 = 0;
    let mut bpno: OPJ_INT32 = 0;
    let mut passtype: OPJ_UINT32 = 0;
    let mut nmsedec = 0i32;
    let mut max: OPJ_INT32 = 0;
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut type_0 = 0 as OPJ_BYTE;
    let mut tempwmsedec: OPJ_FLOAT64 = 0.;
    t1.mqc.lut_ctxno_zc_orient = &lut_ctxno_zc[orient as usize];
    max = 0i32;
    let mut datap = t1.data.as_mut_ptr();
    j = 0 as OPJ_UINT32;
    while j < t1.h {
      let w = t1.w;
      i = 0 as OPJ_UINT32;
      while i < w {
        let mut tmp = *datap;
        if tmp < 0i32 {
          let mut tmp_unsigned: OPJ_UINT32 = 0;
          if tmp == i32::MIN {
            /* To avoid undefined behaviour when negating INT_MIN */
            /* but if we go here, it means we have supplied an input */
            /* with more bit depth than we we can really support. */
            /* Cf https://github.com/uclouvain/openjpeg/issues/1432 */
            tmp = i32::MIN + 1;
          }
          max = opj_int_max(max, -tmp);
          tmp_unsigned = if tmp >= 0i32 {
            tmp as OPJ_UINT32
          } else {
            (-tmp as OPJ_UINT32) | 0x80000000u32
          };
          memcpy(
            datap as *mut core::ffi::c_void,
            &mut tmp_unsigned as *mut OPJ_UINT32 as *const core::ffi::c_void,
            core::mem::size_of::<OPJ_INT32>(),
          );
        } else {
          max = opj_int_max(max, tmp)
        }
        i += 1;
        datap = datap.offset(1);
      }
      j += 1
    }
    (*cblk).numbps = if max != 0 {
      (opj_int_floorlog2(max) + 1i32 - (7i32 - 1i32)) as OPJ_UINT32
    } else {
      0
    };
    if (*cblk).numbps == 0 {
      (*cblk).totalpasses = 0 as OPJ_UINT32;
      return cumwmsedec;
    }
    bpno = (*cblk).numbps.wrapping_sub(1) as OPJ_INT32;
    passtype = 2 as OPJ_UINT32;

    opj_mqc_resetstates(&mut t1.mqc);
    opj_mqc_setstate(&mut t1.mqc, T1_CTXNO_UNI, 0, 46);
    opj_mqc_setstate(&mut t1.mqc, T1_CTXNO_AGG, 0, 3);
    opj_mqc_setstate(&mut t1.mqc, T1_CTXNO_ZC, 0, 4);
    opj_mqc_init_enc(&mut t1.mqc, (*cblk).data);

    passno = 0 as OPJ_UINT32;
    while bpno >= 0i32 {
      let mut pass: *mut opj_tcd_pass_t =
        &mut *(*cblk).passes.offset(passno as isize) as *mut opj_tcd_pass_t;
      type_0 = if bpno < (*cblk).numbps as OPJ_INT32 - 4i32 && passtype < 2 && cblksty & 0x1 != 0 {
        1i32
      } else {
        0i32
      } as OPJ_BYTE;
      /* If the previous pass was terminating, we need to reset the encoder */
      if passno > 0 && (*(*cblk).passes.offset(passno.wrapping_sub(1) as isize)).term {
        if type_0 as core::ffi::c_int == 1i32 {
          opj_mqc_bypass_init_enc(&mut t1.mqc);
        } else {
          opj_mqc_restart_init_enc(&mut t1.mqc);
        }
      }
      match passtype {
        0 => {
          opj_t1_enc_sigpass(t1, bpno, &mut nmsedec, type_0, cblksty);
        }
        1 => {
          opj_t1_enc_refpass(t1, bpno, &mut nmsedec, type_0);
        }
        2 => {
          opj_t1_enc_clnpass(t1, bpno, &mut nmsedec, cblksty);
          /* code switch SEGMARK (i.e. SEGSYM) */
          if cblksty & 0x20 != 0 {
            opj_mqc_segmark_enc(&mut t1.mqc);
          }
        }
        _ => {}
      }

      tempwmsedec = opj_t1_getwmsedec(
        nmsedec,
        compno,
        level,
        orient,
        bpno,
        qmfbid,
        stepsize,
        numcomps,
        mct_norms,
        mct_numcomps,
      );
      cumwmsedec += tempwmsedec;
      (*pass).distortiondec = cumwmsedec;
      if opj_t1_enc_is_term_pass(cblk, cblksty, bpno, passtype) != 0 {
        /* If it is a terminated pass, terminate it */
        if type_0 as core::ffi::c_int == 1i32 {
          opj_mqc_bypass_flush_enc(&mut t1.mqc, (cblksty & 0x10) as OPJ_BOOL);
        } else if cblksty & 0x10 != 0 {
          opj_mqc_erterm_enc(&mut t1.mqc);
        } else {
          opj_mqc_flush(&mut t1.mqc);
        }
        (*pass).term = true;
        (*pass).rate = opj_mqc_numbytes(&mut t1.mqc)
      } else {
        /* Non terminated pass */
        let mut rate_extra_bytes: OPJ_UINT32 = 0;
        if type_0 as core::ffi::c_int == 1i32 {
          rate_extra_bytes =
            opj_mqc_bypass_get_extra_bytes(&mut t1.mqc, (cblksty & 0x10) as OPJ_BOOL)
        } else {
          rate_extra_bytes = 3 as OPJ_UINT32
        }
        (*pass).term = false;
        (*pass).rate = opj_mqc_numbytes(&mut t1.mqc).wrapping_add(rate_extra_bytes)
      }
      passtype = passtype.wrapping_add(1);
      if passtype == 3 {
        passtype = 0 as OPJ_UINT32;
        bpno -= 1
      }
      /* Code-switch "RESET" */
      if cblksty & 0x2 != 0 {
        opj_mqc_reset_enc(&mut t1.mqc);
      }
      passno += 1;
    }
    (*cblk).totalpasses = passno;
    if (*cblk).totalpasses != 0 {
      /* Make sure that pass rates are increasing */
      let mut last_pass_rate = opj_mqc_numbytes(&mut t1.mqc);
      passno = (*cblk).totalpasses;
      while passno > 0 {
        passno = passno.wrapping_sub(1);
        let mut pass_0: *mut opj_tcd_pass_t =
          &mut *(*cblk).passes.offset(passno as isize) as *mut opj_tcd_pass_t;
        if (*pass_0).rate > last_pass_rate {
          (*pass_0).rate = last_pass_rate
        } else {
          last_pass_rate = (*pass_0).rate
        }
      }
    }
    passno = 0 as OPJ_UINT32;
    while passno < (*cblk).totalpasses {
      let mut pass_1: *mut opj_tcd_pass_t =
        &mut *(*cblk).passes.offset(passno as isize) as *mut opj_tcd_pass_t;
      /* Prevent generation of FF as last data byte of a pass*/
      /* For terminating passes, the flushing procedure ensured this already */
      assert!((*pass_1).rate > 0);
      if *(*cblk).data.offset((*pass_1).rate.wrapping_sub(1) as isize) as core::ffi::c_int
        == 0xffi32
      {
        (*pass_1).rate = (*pass_1).rate.wrapping_sub(1)
      }
      (*pass_1).len = (*pass_1).rate.wrapping_sub(if passno == 0 {
        0
      } else {
        (*(*cblk).passes.offset(passno.wrapping_sub(1) as isize)).rate
      });
      passno += 1;
    }
    cumwmsedec
  }
}

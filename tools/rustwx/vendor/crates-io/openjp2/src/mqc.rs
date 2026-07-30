use super::consts::*;
use super::openjpeg::*;
use super::t1_luts::*;

pub struct opj_mqc_state {
  pub qeval: u32,
  // TODO: use 'bool'
  pub mps: u8,
  pub nmps: u8,
  pub nlps: u8,
}
pub type opj_mqc_state_t = opj_mqc_state;

pub struct opj_mqc {
  pub c: OPJ_UINT32,
  pub a: OPJ_UINT32,
  pub ct: OPJ_UINT32,
  pub end_of_byte_stream_counter: OPJ_UINT32,
  pub bp: *mut OPJ_BYTE,
  pub start: *mut OPJ_BYTE,
  pub end: *mut OPJ_BYTE,
  pub ctxs: [&'static opj_mqc_state_t; MQC_NUMCTXS],
  pub curctx: u8,
  pub lut_ctxno_zc_orient: &'static [OPJ_BYTE; 512],
  pub backup: [OPJ_BYTE; 2],
}
pub type opj_mqc_t = opj_mqc;

impl Default for opj_mqc {
  fn default() -> Self {
    use core::ptr::null_mut;
    let ctx = &mqc_states[0];
    Self {
      c: 0,
      a: 0,
      ct: 0,
      end_of_byte_stream_counter: 0,
      bp: null_mut(),
      start: null_mut(),
      end: null_mut(),
      ctxs: [ctx; MQC_NUMCTXS],
      curctx: 0,
      lut_ctxno_zc_orient: &lut_ctxno_zc[0],
      backup: [0; 2],
    }
  }
}

impl opj_mqc {
  pub(crate) fn set_curctx(&mut self, ctxno: u8) {
    self.curctx = ctxno;
  }

  pub(crate) fn set_curctx_nlps(&mut self) {
    let curctx = self.curctx();
    self.ctxs[self.curctx as usize] = &mqc_states[curctx.nlps as usize];
  }

  pub(crate) fn set_curctx_nmps(&mut self) {
    let curctx = self.curctx();
    self.ctxs[self.curctx as usize] = &mqc_states[curctx.nmps as usize];
  }

  pub(crate) fn curctx(&self) -> &'static opj_mqc_state_t {
    self.ctxs[self.curctx as usize]
  }

  pub(crate) fn write_byte(&mut self, b: OPJ_BYTE) {
    unsafe {
      *self.bp = b;
    }
  }

  pub(crate) fn bp_mut(&mut self) -> &mut OPJ_BYTE {
    unsafe { &mut *self.bp }
  }

  pub(crate) fn check_start(&self, offset: isize) -> bool {
    self.bp >= unsafe { self.start.offset(offset) }
  }

  pub(crate) fn bp(&self) -> OPJ_BYTE {
    unsafe { *self.bp }
  }

  pub(crate) fn bp_peek_offset(&self, offset: isize) -> OPJ_BYTE {
    unsafe { *self.bp.offset(offset) }
  }

  pub(crate) fn bp_peek(&self) -> OPJ_BYTE {
    self.bp_peek_offset(1)
  }

  pub(crate) fn inc_bp(&mut self) {
    self.bp_offset(1)
  }

  pub(crate) fn bp_offset(&mut self, offset: isize) {
    unsafe { self.bp = self.bp.offset(offset) }
  }

  pub(crate) fn backup_extra(&mut self) {
    let len = self.backup.len();
    let extra = unsafe { core::slice::from_raw_parts_mut(self.end, len) };
    // Backup extra bytes and replace with `0xff`
    for idx in 0..len {
      self.backup[idx] = extra[idx];
      extra[idx] = 0xff;
    }
  }

  pub(crate) fn restore_extra(&mut self) {
    let len = self.backup.len();
    let extra = unsafe { core::slice::from_raw_parts_mut(self.end, len) };
    extra.copy_from_slice(&self.backup[..]);
  }
}

#[inline]
pub(crate) fn opj_mqc_setcurctx(mqc: &mut opj_mqc_t, ctxno: u8) {
  mqc.set_curctx(ctxno);
}

/* For internal use of opj_mqc_decode_macro() */
fn opj_mqc_mpsexchange_macro(d: &mut OPJ_UINT32, mqc: &mut opj_mqc_t) {
  let curctx = mqc.curctx();
  if mqc.a < curctx.qeval {
    *d = (curctx.mps == 0) as u32;
    mqc.set_curctx_nlps();
  } else {
    *d = curctx.mps as u32;
    mqc.set_curctx_nmps();
  }
}

/* For internal use of opj_mqc_decode_macro() */
fn opj_mqc_lpsexchange_macro(d: &mut OPJ_UINT32, mqc: &mut opj_mqc_t) {
  let curctx = mqc.curctx();
  if mqc.a < curctx.qeval {
    mqc.a = curctx.qeval;
    *d = curctx.mps as u32;
    mqc.set_curctx_nmps();
  } else {
    mqc.a = curctx.qeval;
    *d = (curctx.mps == 0) as u32;
    mqc.set_curctx_nlps();
  }
}

/**
Decode a symbol using raw-decoder. Cfr p.506 TAUBMAN
@param mqc MQC handle
@return Returns the decoded symbol (0 or 1)
*/
#[inline]
pub(crate) fn opj_mqc_raw_decode(mut mqc: &mut opj_mqc_t) -> OPJ_UINT32 {
  if mqc.ct == 0 {
    /* Given opj_mqc_raw_init_dec() we know that at some point we will */
    /* have a 0xFF 0xFF artificial marker */
    if mqc.c == 0xff {
      if mqc.bp() > 0x8f {
        mqc.c = 0xff;
        mqc.ct = 8;
      } else {
        mqc.c = mqc.bp() as OPJ_UINT32;
        mqc.inc_bp();
        mqc.ct = 7;
      }
    } else {
      mqc.c = mqc.bp() as OPJ_UINT32;
      mqc.inc_bp();
      mqc.ct = 8;
    }
  }
  mqc.ct = mqc.ct.wrapping_sub(1);

  mqc.c >> mqc.ct & 0x1
}

#[inline]
fn opj_mqc_bytein_macro(mqc: &mut opj_mqc_t) {
  let mut l_c: OPJ_UINT32 = 0;
  /* Given opj_mqc_init_dec() we know that at some point we will */
  /* have a 0xFF 0xFF artificial marker */
  l_c = mqc.bp_peek() as OPJ_UINT32;
  if mqc.bp() == 0xff {
    if l_c > 0x8f {
      mqc.c = mqc.c.wrapping_add(0xff00);
      mqc.ct = 8;
      mqc.end_of_byte_stream_counter = mqc.end_of_byte_stream_counter.wrapping_add(1);
    } else {
      mqc.inc_bp();
      mqc.c = mqc.c.wrapping_add(l_c << 9);
      mqc.ct = 7;
    }
  } else {
    mqc.inc_bp();
    mqc.c = mqc.c.wrapping_add(l_c << 8);
    mqc.ct = 8;
  }
}

#[inline]
pub(crate) fn opj_mqc_bytein(mqc: &mut opj_mqc_t) {
  opj_mqc_bytein_macro(mqc);
}

/* For internal use of opj_mqc_decode_macro() */
#[inline]
fn opj_mqc_renormd_macro(mqc: &mut opj_mqc_t) {
  loop {
    if mqc.ct == 0 {
      opj_mqc_bytein_macro(mqc);
    }
    mqc.a <<= 1;
    mqc.c <<= 1;
    mqc.ct = mqc.ct.wrapping_sub(1);
    if mqc.a >= 0x8000 {
      break;
    }
  }
}

#[inline]
pub(crate) fn opj_mqc_decode_macro(d: &mut OPJ_UINT32, mqc: &mut opj_mqc_t) {
  /* Implements ISO 15444-1 C.3.2 Decoding a decision (DECODE) */
  /* Note: alternate "J.2 - Decoding an MPS or an LPS in the */
  /* software-conventions decoder" has been tried, but does not bring any */
  /* improvement. See https://github.com/uclouvain/openjpeg/issues/921 */
  mqc.a = mqc.a.wrapping_sub(mqc.curctx().qeval);
  if (mqc.c >> 16) < mqc.curctx().qeval {
    opj_mqc_lpsexchange_macro(d, mqc);
    opj_mqc_renormd_macro(mqc);
  } else {
    mqc.c = mqc.c.wrapping_sub(mqc.curctx().qeval << 16);
    if mqc.a & 0x8000 == 0 {
      opj_mqc_mpsexchange_macro(d, mqc);
      opj_mqc_renormd_macro(mqc);
    } else {
      *d = mqc.curctx().mps as u32
    }
  }
}

/**
Renormalize mqc->a and mqc->c while encoding, so that mqc->a stays between 0x8000 and 0x10000
@param mqc MQC handle
@param a_ value of mqc->a
@param c_ value of mqc->c_
@param ct_ value of mqc->ct_
*/
#[inline]
fn opj_mqc_renorme_macro(mqc: &mut opj_mqc_t) {
  loop {
    mqc.a <<= 1;
    mqc.c <<= 1;
    mqc.ct = mqc.ct.wrapping_sub(1);
    if mqc.ct == 0 {
      opj_mqc_byteout(mqc);
    }
    if mqc.a & 0x8000 != 0 {
      break;
    }
  }
}

#[inline]
fn opj_mqc_codemps_macro(mqc: &mut opj_mqc_t) {
  mqc.a = mqc.a.wrapping_sub(mqc.curctx().qeval);
  if mqc.a & 0x8000 == 0 {
    if mqc.a < mqc.curctx().qeval {
      mqc.a = mqc.curctx().qeval
    } else {
      mqc.c = mqc.c.wrapping_add(mqc.curctx().qeval);
    }
    mqc.set_curctx_nmps();
    opj_mqc_renorme_macro(mqc);
  } else {
    mqc.c = mqc.c.wrapping_add(mqc.curctx().qeval);
  }
}

#[inline]
fn opj_mqc_codelps_macro(mqc: &mut opj_mqc_t) {
  mqc.a = mqc.a.wrapping_sub(mqc.curctx().qeval);
  if mqc.a < mqc.curctx().qeval {
    mqc.c = mqc.c.wrapping_add(mqc.curctx().qeval);
  } else {
    mqc.a = mqc.curctx().qeval
  }
  mqc.set_curctx_nlps();
  opj_mqc_renorme_macro(mqc);
}

#[inline]
pub(crate) fn opj_mqc_encode_macro(mqc: &mut opj_mqc_t, d: OPJ_UINT32) {
  if mqc.curctx().mps as u32 == d {
    opj_mqc_codemps(mqc);
  } else {
    opj_mqc_codelps(mqc);
  }
}

#[inline]
pub(crate) fn opj_mqc_bypass_enc_macro(mqc: &mut opj_mqc_t, d: OPJ_UINT32) {
  if mqc.ct == BYPASS_CT_INIT {
    mqc.ct = 8;
  }
  mqc.ct = mqc.ct.wrapping_sub(1);
  mqc.c = mqc.c.wrapping_add(d << mqc.ct);
  if mqc.ct == 0 {
    mqc.write_byte(mqc.c as OPJ_BYTE);
    mqc.ct = 8;
    /* If the previous byte was 0xff, make sure that the next msb is 0 */
    if mqc.bp() == 0xff {
      mqc.ct = 7;
    }
    mqc.inc_bp();
    mqc.c = 0;
  }
}

/*@}*/
/*@}*/
/* <summary> */
/* This array defines all the possible states for a context. */
/* </summary> */
const mqc_states: [opj_mqc_state_t; 47 * 2] = [
  opj_mqc_state_t {
    qeval: 0x5601,
    mps: 0,
    nmps: 2,
    nlps: 3,
  },
  opj_mqc_state_t {
    qeval: 0x5601,
    mps: 1,
    nmps: 3,
    nlps: 2,
  },
  opj_mqc_state_t {
    qeval: 0x3401,
    mps: 0,
    nmps: 4,
    nlps: 12,
  },
  opj_mqc_state_t {
    qeval: 0x3401,
    mps: 1,
    nmps: 5,
    nlps: 13,
  },
  opj_mqc_state_t {
    qeval: 0x1801,
    mps: 0,
    nmps: 6,
    nlps: 18,
  },
  opj_mqc_state_t {
    qeval: 0x1801,
    mps: 1,
    nmps: 7,
    nlps: 19,
  },
  opj_mqc_state_t {
    qeval: 0x0ac1,
    mps: 0,
    nmps: 8,
    nlps: 24,
  },
  opj_mqc_state_t {
    qeval: 0x0ac1,
    mps: 1,
    nmps: 9,
    nlps: 25,
  },
  opj_mqc_state_t {
    qeval: 0x0521,
    mps: 0,
    nmps: 10,
    nlps: 58,
  },
  opj_mqc_state_t {
    qeval: 0x0521,
    mps: 1,
    nmps: 11,
    nlps: 59,
  },
  opj_mqc_state_t {
    qeval: 0x0221,
    mps: 0,
    nmps: 76,
    nlps: 66,
  },
  opj_mqc_state_t {
    qeval: 0x0221,
    mps: 1,
    nmps: 77,
    nlps: 67,
  },
  opj_mqc_state_t {
    qeval: 0x5601,
    mps: 0,
    nmps: 14,
    nlps: 13,
  },
  opj_mqc_state_t {
    qeval: 0x5601,
    mps: 1,
    nmps: 15,
    nlps: 12,
  },
  opj_mqc_state_t {
    qeval: 0x5401,
    mps: 0,
    nmps: 16,
    nlps: 28,
  },
  opj_mqc_state_t {
    qeval: 0x5401,
    mps: 1,
    nmps: 17,
    nlps: 29,
  },
  opj_mqc_state_t {
    qeval: 0x4801,
    mps: 0,
    nmps: 18,
    nlps: 28,
  },
  opj_mqc_state_t {
    qeval: 0x4801,
    mps: 1,
    nmps: 19,
    nlps: 29,
  },
  opj_mqc_state_t {
    qeval: 0x3801,
    mps: 0,
    nmps: 20,
    nlps: 28,
  },
  opj_mqc_state_t {
    qeval: 0x3801,
    mps: 1,
    nmps: 21,
    nlps: 29,
  },
  opj_mqc_state_t {
    qeval: 0x3001,
    mps: 0,
    nmps: 22,
    nlps: 34,
  },
  opj_mqc_state_t {
    qeval: 0x3001,
    mps: 1,
    nmps: 23,
    nlps: 35,
  },
  opj_mqc_state_t {
    qeval: 0x2401,
    mps: 0,
    nmps: 24,
    nlps: 36,
  },
  opj_mqc_state_t {
    qeval: 0x2401,
    mps: 1,
    nmps: 25,
    nlps: 37,
  },
  opj_mqc_state_t {
    qeval: 0x1c01,
    mps: 0,
    nmps: 26,
    nlps: 40,
  },
  opj_mqc_state_t {
    qeval: 0x1c01,
    mps: 1,
    nmps: 27,
    nlps: 41,
  },
  opj_mqc_state_t {
    qeval: 0x1601,
    mps: 0,
    nmps: 58,
    nlps: 42,
  },
  opj_mqc_state_t {
    qeval: 0x1601,
    mps: 1,
    nmps: 59,
    nlps: 43,
  },
  opj_mqc_state_t {
    qeval: 0x5601,
    mps: 0,
    nmps: 30,
    nlps: 29,
  },
  opj_mqc_state_t {
    qeval: 0x5601,
    mps: 1,
    nmps: 31,
    nlps: 28,
  },
  opj_mqc_state_t {
    qeval: 0x5401,
    mps: 0,
    nmps: 32,
    nlps: 28,
  },
  opj_mqc_state_t {
    qeval: 0x5401,
    mps: 1,
    nmps: 33,
    nlps: 29,
  },
  opj_mqc_state_t {
    qeval: 0x5101,
    mps: 0,
    nmps: 34,
    nlps: 30,
  },
  opj_mqc_state_t {
    qeval: 0x5101,
    mps: 1,
    nmps: 35,
    nlps: 31,
  },
  opj_mqc_state_t {
    qeval: 0x4801,
    mps: 0,
    nmps: 36,
    nlps: 32,
  },
  opj_mqc_state_t {
    qeval: 0x4801,
    mps: 1,
    nmps: 37,
    nlps: 33,
  },
  opj_mqc_state_t {
    qeval: 0x3801,
    mps: 0,
    nmps: 38,
    nlps: 34,
  },
  opj_mqc_state_t {
    qeval: 0x3801,
    mps: 1,
    nmps: 39,
    nlps: 35,
  },
  opj_mqc_state_t {
    qeval: 0x3401,
    mps: 0,
    nmps: 40,
    nlps: 36,
  },
  opj_mqc_state_t {
    qeval: 0x3401,
    mps: 1,
    nmps: 41,
    nlps: 37,
  },
  opj_mqc_state_t {
    qeval: 0x3001,
    mps: 0,
    nmps: 42,
    nlps: 38,
  },
  opj_mqc_state_t {
    qeval: 0x3001,
    mps: 1,
    nmps: 43,
    nlps: 39,
  },
  opj_mqc_state_t {
    qeval: 0x2801,
    mps: 0,
    nmps: 44,
    nlps: 38,
  },
  opj_mqc_state_t {
    qeval: 0x2801,
    mps: 1,
    nmps: 45,
    nlps: 39,
  },
  opj_mqc_state_t {
    qeval: 0x2401,
    mps: 0,
    nmps: 46,
    nlps: 40,
  },
  opj_mqc_state_t {
    qeval: 0x2401,
    mps: 1,
    nmps: 47,
    nlps: 41,
  },
  opj_mqc_state_t {
    qeval: 0x2201,
    mps: 0,
    nmps: 48,
    nlps: 42,
  },
  opj_mqc_state_t {
    qeval: 0x2201,
    mps: 1,
    nmps: 49,
    nlps: 43,
  },
  opj_mqc_state_t {
    qeval: 0x1c01,
    mps: 0,
    nmps: 50,
    nlps: 44,
  },
  opj_mqc_state_t {
    qeval: 0x1c01,
    mps: 1,
    nmps: 51,
    nlps: 45,
  },
  opj_mqc_state_t {
    qeval: 0x1801,
    mps: 0,
    nmps: 52,
    nlps: 46,
  },
  opj_mqc_state_t {
    qeval: 0x1801,
    mps: 1,
    nmps: 53,
    nlps: 47,
  },
  opj_mqc_state_t {
    qeval: 0x1601,
    mps: 0,
    nmps: 54,
    nlps: 48,
  },
  opj_mqc_state_t {
    qeval: 0x1601,
    mps: 1,
    nmps: 55,
    nlps: 49,
  },
  opj_mqc_state_t {
    qeval: 0x1401,
    mps: 0,
    nmps: 56,
    nlps: 50,
  },
  opj_mqc_state_t {
    qeval: 0x1401,
    mps: 1,
    nmps: 57,
    nlps: 51,
  },
  opj_mqc_state_t {
    qeval: 0x1201,
    mps: 0,
    nmps: 58,
    nlps: 52,
  },
  opj_mqc_state_t {
    qeval: 0x1201,
    mps: 1,
    nmps: 59,
    nlps: 53,
  },
  opj_mqc_state_t {
    qeval: 0x1101,
    mps: 0,
    nmps: 60,
    nlps: 54,
  },
  opj_mqc_state_t {
    qeval: 0x1101,
    mps: 1,
    nmps: 61,
    nlps: 55,
  },
  opj_mqc_state_t {
    qeval: 0x0ac1,
    mps: 0,
    nmps: 62,
    nlps: 56,
  },
  opj_mqc_state_t {
    qeval: 0x0ac1,
    mps: 1,
    nmps: 63,
    nlps: 57,
  },
  opj_mqc_state_t {
    qeval: 0x09c1,
    mps: 0,
    nmps: 64,
    nlps: 58,
  },
  opj_mqc_state_t {
    qeval: 0x09c1,
    mps: 1,
    nmps: 65,
    nlps: 59,
  },
  opj_mqc_state_t {
    qeval: 0x08a1,
    mps: 0,
    nmps: 66,
    nlps: 60,
  },
  opj_mqc_state_t {
    qeval: 0x08a1,
    mps: 1,
    nmps: 67,
    nlps: 61,
  },
  opj_mqc_state_t {
    qeval: 0x0521,
    mps: 0,
    nmps: 68,
    nlps: 62,
  },
  opj_mqc_state_t {
    qeval: 0x0521,
    mps: 1,
    nmps: 69,
    nlps: 63,
  },
  opj_mqc_state_t {
    qeval: 0x0441,
    mps: 0,
    nmps: 70,
    nlps: 64,
  },
  opj_mqc_state_t {
    qeval: 0x0441,
    mps: 1,
    nmps: 71,
    nlps: 65,
  },
  opj_mqc_state_t {
    qeval: 0x02a1,
    mps: 0,
    nmps: 72,
    nlps: 66,
  },
  opj_mqc_state_t {
    qeval: 0x02a1,
    mps: 1,
    nmps: 73,
    nlps: 67,
  },
  opj_mqc_state_t {
    qeval: 0x0221,
    mps: 0,
    nmps: 74,
    nlps: 68,
  },
  opj_mqc_state_t {
    qeval: 0x0221,
    mps: 1,
    nmps: 75,
    nlps: 69,
  },
  opj_mqc_state_t {
    qeval: 0x0141,
    mps: 0,
    nmps: 76,
    nlps: 70,
  },
  opj_mqc_state_t {
    qeval: 0x0141,
    mps: 1,
    nmps: 77,
    nlps: 71,
  },
  opj_mqc_state_t {
    qeval: 0x0111,
    mps: 0,
    nmps: 78,
    nlps: 72,
  },
  opj_mqc_state_t {
    qeval: 0x0111,
    mps: 1,
    nmps: 79,
    nlps: 73,
  },
  opj_mqc_state_t {
    qeval: 0x0085,
    mps: 0,
    nmps: 80,
    nlps: 74,
  },
  opj_mqc_state_t {
    qeval: 0x0085,
    mps: 1,
    nmps: 81,
    nlps: 75,
  },
  opj_mqc_state_t {
    qeval: 0x0049,
    mps: 0,
    nmps: 82,
    nlps: 76,
  },
  opj_mqc_state_t {
    qeval: 0x0049,
    mps: 1,
    nmps: 83,
    nlps: 77,
  },
  opj_mqc_state_t {
    qeval: 0x0025,
    mps: 0,
    nmps: 84,
    nlps: 78,
  },
  opj_mqc_state_t {
    qeval: 0x0025,
    mps: 1,
    nmps: 85,
    nlps: 79,
  },
  opj_mqc_state_t {
    qeval: 0x0015,
    mps: 0,
    nmps: 86,
    nlps: 80,
  },
  opj_mqc_state_t {
    qeval: 0x0015,
    mps: 1,
    nmps: 87,
    nlps: 81,
  },
  opj_mqc_state_t {
    qeval: 0x0009,
    mps: 0,
    nmps: 88,
    nlps: 82,
  },
  opj_mqc_state_t {
    qeval: 0x0009,
    mps: 1,
    nmps: 89,
    nlps: 83,
  },
  opj_mqc_state_t {
    qeval: 0x0005,
    mps: 0,
    nmps: 90,
    nlps: 84,
  },
  opj_mqc_state_t {
    qeval: 0x0005,
    mps: 1,
    nmps: 91,
    nlps: 85,
  },
  opj_mqc_state_t {
    qeval: 0x0001,
    mps: 0,
    nmps: 90,
    nlps: 86,
  },
  opj_mqc_state_t {
    qeval: 0x0001,
    mps: 1,
    nmps: 91,
    nlps: 87,
  },
  opj_mqc_state_t {
    qeval: 0x5601,
    mps: 0,
    nmps: 92,
    nlps: 92,
  },
  opj_mqc_state_t {
    qeval: 0x5601,
    mps: 1,
    nmps: 93,
    nlps: 93,
  },
];

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
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
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

/* * @defgroup MQC MQC - Implementation of an MQ-Coder */
/*@{*/

/* * @name Local static functions */
/*@{*/

/*
==========================================================
   local functions
==========================================================
*/

/* *
Fill mqc->c with 1's for flushing
@param mqc MQC handle
*/
fn opj_mqc_setbits(mut mqc: &mut opj_mqc_t) {
  let mut tempc = mqc.c.wrapping_add(mqc.a);
  mqc.c |= 0xffff;
  if mqc.c >= tempc {
    mqc.c = mqc.c.wrapping_sub(0x8000);
  }
}

/*
==========================================================
   MQ-Coder interface
==========================================================
*/

pub(crate) fn opj_mqc_numbytes(mut mqc: &mut opj_mqc_t) -> OPJ_UINT32 {
  let diff = unsafe { mqc.bp.offset_from(mqc.start) };
  diff as OPJ_UINT32
}

pub(crate) fn opj_mqc_init_enc(mut mqc: &mut opj_mqc_t, mut bp: *mut OPJ_BYTE) {
  /* To avoid the curctx pointer to be dangling, but not strictly */
  /* required as the current context is always set before encoding */
  opj_mqc_setcurctx(mqc, 0);

  /* As specified in Figure C.10 - Initialization of the encoder */
  /* (C.2.8 Initialization of the encoder (INITENC)) */
  mqc.a = 0x8000;
  mqc.c = 0;
  /* Yes, we point before the start of the buffer, but this is safe */
  /* given opj_tcd_code_block_enc_allocate_data() */
  mqc.bp = bp;
  mqc.bp_offset(-1);
  mqc.ct = 12;
  /* At this point we should test *(mqc->bp) against 0xFF, but this is not */
  /* necessary, as this is only used at the beginning of the code block */
  /* and our initial fake byte is set at 0 */
  assert!(mqc.bp() != 0xff);

  mqc.start = bp;
  mqc.end_of_byte_stream_counter = 0;
}

pub(crate) fn opj_mqc_flush(mut mqc: &mut opj_mqc_t) {
  /* C.2.9 Termination of coding (FLUSH) */
  /* Figure C.11 – FLUSH procedure */
  opj_mqc_setbits(mqc);
  mqc.c <<= mqc.ct;
  opj_mqc_byteout(mqc);
  mqc.c <<= mqc.ct;
  opj_mqc_byteout(mqc);

  /* It is forbidden that a coding pass ends with 0xff */
  if mqc.bp() != 0xff {
    /* Advance pointer so that opj_mqc_numbytes() returns a valid value */
    mqc.inc_bp()
  }
}

pub(crate) fn opj_mqc_bypass_init_enc(mut mqc: &mut opj_mqc_t) {
  /* This function is normally called after at least one opj_mqc_flush() */
  /* which will have advance mqc->bp by at least 2 bytes beyond its */
  /* initial position */
  assert!(mqc.check_start(0));
  mqc.c = 0;
  /* in theory we should initialize to 8, but use this special value */
  /* as a hint that opj_mqc_bypass_enc() has never been called, so */
  /* as to avoid the 0xff 0x7f elimination trick in opj_mqc_bypass_flush_enc() */
  /* to trigger when we don't have output any bit during this bypass sequence */
  /* Any value > 8 will do */
  mqc.ct = BYPASS_CT_INIT;
  /* Given that we are called after opj_mqc_flush(), the previous byte */
  /* cannot be 0xff. */
  assert!(mqc.bp_peek_offset(-1) != 0xff);
}

pub(crate) fn opj_mqc_bypass_enc(mut mqc: &mut opj_mqc_t, d: OPJ_UINT32) {
  opj_mqc_bypass_enc_macro(mqc, d)
}

pub(crate) fn opj_mqc_bypass_get_extra_bytes(
  mut mqc: &mut opj_mqc_t,
  mut erterm: OPJ_BOOL,
) -> OPJ_UINT32 {
  if mqc.ct < 7 || mqc.ct == 7 && (erterm != 0 || mqc.bp_peek_offset(-1) != 0xff) {
    1
  } else {
    0
  }
}

pub(crate) fn opj_mqc_bypass_flush_enc(mut mqc: &mut opj_mqc_t, mut erterm: OPJ_BOOL) {
  /* Is there any bit remaining to be flushed ? */
  /* If the last output byte is 0xff, we can discard it, unless */
  /* erterm is required (I'm not completely sure why in erterm */
  /* we must output 0xff 0x2a if the last byte was 0xff instead of */
  /* discarding it, but Kakadu requires it when decoding */
  /* in -fussy mode) */
  if mqc.ct < 7 || mqc.ct == 7 && (erterm != 0 || mqc.bp_peek_offset(-1) != 0xff) {
    let mut bit_value = 0i32;
    /* If so, fill the remaining lsbs with an alternating sequence of */
    /* 0,1,... */
    /* Note: it seems the standard only requires that for a ERTERM flush */
    /* and doesn't specify what to do for a regular BYPASS flush */
    while mqc.ct > 0 {
      mqc.ct = mqc.ct.wrapping_sub(1);
      mqc.c = mqc.c.wrapping_add((bit_value << mqc.ct) as u32);
      bit_value = 1i32.wrapping_sub(bit_value);
    }
    mqc.write_byte(mqc.c as OPJ_BYTE);
    /* Advance pointer so that opj_mqc_numbytes() returns a valid value */
    mqc.inc_bp()
  } else if mqc.ct == 7 && mqc.bp_peek_offset(-1) == 0xff {
    /* Discard last 0xff */
    assert!(erterm == 0);
    mqc.bp_offset(-1)
  } else if mqc.ct == 8
    && erterm == 0
    && mqc.bp_peek_offset(-1) == 0x7f
    && mqc.bp_peek_offset(-2) == 0xff
  {
    /* Tiny optimization: discard terminating 0xff 0x7f since it is */
    /* interpreted as 0xff 0x7f [0xff 0xff] by the decoder, and given */
    /* the bit stuffing, in fact as 0xff 0xff [0xff ..] */
    /* Happens once on opj_compress -i ../MAPA.tif -o MAPA.j2k  -M 1 */
    mqc.bp_offset(-2)
  }

  assert!(mqc.bp_peek_offset(-1) != 0xff);
}

pub(crate) fn opj_mqc_reset_enc(mut mqc: &mut opj_mqc_t) {
  opj_mqc_resetstates(mqc);
  opj_mqc_setstate(mqc, T1_CTXNO_UNI, 0, 46);
  opj_mqc_setstate(mqc, T1_CTXNO_AGG, 0, 3);
  opj_mqc_setstate(mqc, T1_CTXNO_ZC, 0, 4);
}

pub(crate) fn opj_mqc_restart_init_enc(mut mqc: &mut opj_mqc_t) {
  /* <Re-init part> */
  /* As specified in Figure C.10 - Initialization of the encoder */
  /* (C.2.8 Initialization of the encoder (INITENC)) */
  mqc.a = 0x8000;
  mqc.c = 0;
  mqc.ct = 12;
  /* This function is normally called after at least one opj_mqc_flush() */
  /* which will have advance mqc->bp by at least 2 bytes beyond its */
  /* initial position */
  mqc.bp_offset(-1);

  assert!(mqc.check_start(-1));
  assert!(mqc.bp() != 0xff);
  if mqc.bp() == 0xff {
    mqc.ct = 13;
  }
}

pub(crate) fn opj_mqc_erterm_enc(mut mqc: &mut opj_mqc_t) {
  let mut k = (11_i32).wrapping_sub(mqc.ct as i32).wrapping_add(1);
  while k > 0 {
    mqc.c <<= mqc.ct;
    mqc.ct = 0;
    opj_mqc_byteout(mqc);
    k -= mqc.ct as OPJ_INT32
  }
  if mqc.bp() != 0xff {
    opj_mqc_byteout(mqc);
  }
}

/* *
Encode the most probable symbol
@param mqc MQC handle
*/
#[inline]
fn opj_mqc_codemps(mut mqc: &mut opj_mqc_t) {
  opj_mqc_codemps_macro(mqc)
}

/* *
Encode the most least symbol
@param mqc MQC handle
*/
#[inline]
fn opj_mqc_codelps(mut mqc: &mut opj_mqc_t) {
  opj_mqc_codelps_macro(mqc)
}

/* *
Encode a symbol using the MQ-coder
@param mqc MQC handle
@param d The symbol to be encoded (0 or 1)
*/
#[inline]
pub(crate) fn opj_mqc_encode(mut mqc: &mut opj_mqc_t, mut d: OPJ_UINT32) {
  opj_mqc_encode_macro(mqc, d)
}

pub(crate) fn opj_mqc_segmark_enc(mut mqc: &mut opj_mqc_t) {
  opj_mqc_setcurctx(mqc, 18);

  for i in 1u32..5 {
    opj_mqc_encode(mqc, i.wrapping_rem(2));
  }
}

fn opj_mqc_init_dec_common(
  mut mqc: &mut opj_mqc_t,
  mut bp: *mut OPJ_BYTE,
  mut len: OPJ_UINT32,
  mut extra_writable_bytes: OPJ_UINT32,
) {
  assert!(extra_writable_bytes >= OPJ_COMMON_CBLK_DATA_EXTRA);
  mqc.start = bp;
  unsafe {
    mqc.end = bp.offset(len as isize);
  }
  /* Insert an artificial 0xFF 0xFF marker at end of the code block */
  /* data so that the bytein routines stop on it. This saves us comparing */
  /* the bp and end pointers */
  /* But before inserting it, backup th bytes we will overwrite */
  mqc.backup_extra();
  mqc.bp = bp;
}

pub(crate) fn opj_mqc_init_dec(
  mut mqc: &mut opj_mqc_t,
  mut bp: *mut OPJ_BYTE,
  mut len: OPJ_UINT32,
  mut extra_writable_bytes: OPJ_UINT32,
) {
  /* Implements ISO 15444-1 C.3.5 Initialization of the decoder (INITDEC) */
  /* Note: alternate "J.1 - Initialization of the software-conventions */
  /* decoder" has been tried, but does */
  /* not bring any improvement. */
  /* See https://github.com/uclouvain/openjpeg/issues/921 */
  opj_mqc_init_dec_common(mqc, bp, len, extra_writable_bytes);
  opj_mqc_setcurctx(mqc, 0);
  mqc.end_of_byte_stream_counter = 0;
  if len == 0 {
    mqc.c = 0xff << 16;
  } else {
    mqc.c = (mqc.bp() as u32) << 16;
  }
  opj_mqc_bytein(mqc);
  mqc.c <<= 7;
  mqc.ct = mqc.ct.wrapping_sub(7);
  mqc.a = 0x8000;
}

pub(crate) fn opj_mqc_raw_init_dec(
  mqc: &mut opj_mqc_t,
  bp: *mut OPJ_BYTE,
  len: OPJ_UINT32,
  extra_writable_bytes: OPJ_UINT32,
) {
  opj_mqc_init_dec_common(mqc, bp, len, extra_writable_bytes);
  mqc.c = 0;
  mqc.ct = 0;
}

pub(crate) fn opq_mqc_finish_dec(mqc: &mut opj_mqc_t) {
  /* Restore the bytes overwritten by opj_mqc_init_dec_common() */
  mqc.restore_extra();
}

pub(crate) fn opj_mqc_resetstates(mqc: &mut opj_mqc_t) {
  for i in 0..19 {
    mqc.ctxs[i] = &mqc_states[0];
  }
}

pub(crate) fn opj_mqc_setstate(mqc: &mut opj_mqc_t, ctxno: u8, msb: OPJ_UINT32, prob: OPJ_INT32) {
  mqc.ctxs[ctxno as usize] = &mqc_states[msb.wrapping_add((prob << 1) as OPJ_UINT32) as usize];
}

pub(crate) fn opj_mqc_byteout(mut mqc: &mut opj_mqc_t) {
  /* bp is initialized to start - 1 in opj_mqc_init_enc() */
  /* but this is safe, see opj_tcd_code_block_enc_allocate_data() */
  assert!(mqc.check_start(-1));
  if mqc.bp() == 0xff {
    mqc.inc_bp();
    mqc.write_byte((mqc.c >> 20) as OPJ_BYTE);
    mqc.c &= 0xfffff;
    mqc.ct = 7
  } else if mqc.c & 0x8000000 == 0 {
    mqc.inc_bp();
    mqc.write_byte((mqc.c >> 19) as OPJ_BYTE);
    mqc.c &= 0x7ffff;
    mqc.ct = 8
  } else {
    mqc.write_byte(mqc.bp().wrapping_add(1));
    if mqc.bp() == 0xff {
      mqc.c &= 0x7ffffff;
      mqc.inc_bp();
      mqc.write_byte((mqc.c >> 20) as OPJ_BYTE);
      mqc.c &= 0xfffff;
      mqc.ct = 7
    } else {
      mqc.inc_bp();
      mqc.write_byte((mqc.c >> 19) as OPJ_BYTE);
      mqc.c &= 0x7ffff;
      mqc.ct = 8
    }
  }
}

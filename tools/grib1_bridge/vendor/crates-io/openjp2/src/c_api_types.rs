/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
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

pub use super::image::{opj_image, opj_image_t};
use super::malloc::*;

pub type size_t = usize;

pub type OPJ_BOOL = i32;
pub type OPJ_CHAR = core::ffi::c_char;
pub type OPJ_FLOAT32 = f32;
pub type OPJ_FLOAT64 = f64;
pub type OPJ_BYTE = u8;
pub type OPJ_INT8 = i8;
pub type OPJ_UINT8 = u8;
pub type OPJ_INT16 = i16;
pub type OPJ_UINT16 = u16;
pub type OPJ_INT32 = i32;
pub type OPJ_UINT32 = u32;
pub type OPJ_INT64 = i64;
pub type OPJ_UINT64 = u64;
pub type OPJ_OFF_T = i64;
pub type OPJ_SIZE_T = usize;
pub type opj_flag_t = u32;
pub type RSIZ_CAPABILITIES = core::ffi::c_uint;
pub const OPJ_MCT: RSIZ_CAPABILITIES = 33024;
pub const OPJ_CINEMA4K: RSIZ_CAPABILITIES = 4;
pub const OPJ_CINEMA2K: RSIZ_CAPABILITIES = 3;
pub const OPJ_STD_RSIZ: RSIZ_CAPABILITIES = 0;
pub type OPJ_RSIZ_CAPABILITIES = RSIZ_CAPABILITIES;
pub type CINEMA_MODE = core::ffi::c_uint;
pub const OPJ_CINEMA4K_24: CINEMA_MODE = 3;
pub const OPJ_CINEMA2K_48: CINEMA_MODE = 2;
pub const OPJ_CINEMA2K_24: CINEMA_MODE = 1;
pub const OPJ_OFF: CINEMA_MODE = 0;
pub type OPJ_CINEMA_MODE = CINEMA_MODE;
pub type PROG_ORDER = core::ffi::c_int;
pub const OPJ_CPRL: PROG_ORDER = 4;
pub const OPJ_PCRL: PROG_ORDER = 3;
pub const OPJ_RPCL: PROG_ORDER = 2;
pub const OPJ_RLCP: PROG_ORDER = 1;
pub const OPJ_LRCP: PROG_ORDER = 0;
pub const OPJ_PROG_UNKNOWN: PROG_ORDER = -1;
pub type OPJ_PROG_ORDER = PROG_ORDER;

pub type OPJ_BITFIELD = core::ffi::c_uint;

/// Supported image color spaces
#[derive(Debug, Default, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum COLOR_SPACE {
  ///< not supported by the library
  OPJ_CLRSPC_UNKNOWN = -1,
  ///< not specified in the codestream
  #[default]
  OPJ_CLRSPC_UNSPECIFIED = 0,
  ///< sRGB
  OPJ_CLRSPC_SRGB = 1,
  ///< grayscale
  OPJ_CLRSPC_GRAY = 2,
  ///< YUV
  OPJ_CLRSPC_SYCC = 3,
  ///< e-YCC
  OPJ_CLRSPC_EYCC = 4,
  ///< CMYK
  OPJ_CLRSPC_CMYK = 5,
}
/// Supported image color spaces
pub use COLOR_SPACE as OPJ_COLOR_SPACE;
pub use COLOR_SPACE::*;

/// Supported codec
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum CODEC_FORMAT {
  ///< place-holder
  OPJ_CODEC_UNKNOWN = -1,
  ///< JPEG-2000 codestream : read/write
  OPJ_CODEC_J2K = 0,
  ///< JPT-stream (JPEG 2000, JPIP) : read only
  OPJ_CODEC_JPT = 1,
  ///< JP2 file format : read/write
  OPJ_CODEC_JP2 = 2,
  ///< JPP-stream (JPEG 2000, JPIP) : to be coded
  OPJ_CODEC_JPP = 3,
  ///< JPX file format (JPEG 2000 Part-2) : to be coded
  OPJ_CODEC_JPX = 4,
}
/// Supported codec
pub use CODEC_FORMAT as OPJ_CODEC_FORMAT;
pub use CODEC_FORMAT::*;

impl CODEC_FORMAT {
  pub fn from_i32(num: i32) -> Self {
    match num {
      0 => OPJ_CODEC_J2K,
      1 => OPJ_CODEC_JPT,
      2 => OPJ_CODEC_JP2,
      3 => OPJ_CODEC_JPP,
      4 => OPJ_CODEC_JPX,
      _ => OPJ_CODEC_UNKNOWN,
    }
  }
}

pub type opj_msg_callback_fn =
  unsafe extern "C" fn(_: *const core::ffi::c_char, _: *mut core::ffi::c_void) -> ();

pub type opj_msg_callback = Option<opj_msg_callback_fn>;

#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct opj_poc {
  pub resno0: OPJ_UINT32,
  pub compno0: OPJ_UINT32,
  pub layno1: OPJ_UINT32,
  pub resno1: OPJ_UINT32,
  pub compno1: OPJ_UINT32,
  pub layno0: OPJ_UINT32,
  pub precno0: OPJ_UINT32,
  pub precno1: OPJ_UINT32,
  pub prg1: OPJ_PROG_ORDER,
  pub prg: OPJ_PROG_ORDER,
  pub progorder: [OPJ_CHAR; 5],
  pub tile: OPJ_UINT32,
  pub tx0: OPJ_UINT32,
  pub tx1: OPJ_UINT32,
  pub ty0: OPJ_UINT32,
  pub ty1: OPJ_UINT32,
  pub layS: OPJ_UINT32,
  pub resS: OPJ_UINT32,
  pub compS: OPJ_UINT32,
  pub prcS: OPJ_UINT32,
  pub layE: OPJ_UINT32,
  pub resE: OPJ_UINT32,
  pub compE: OPJ_UINT32,
  pub prcE: OPJ_UINT32,
  pub txS: OPJ_UINT32,
  pub txE: OPJ_UINT32,
  pub tyS: OPJ_UINT32,
  pub tyE: OPJ_UINT32,
  pub dx: OPJ_UINT32,
  pub dy: OPJ_UINT32,
  pub lay_t: OPJ_UINT32,
  pub res_t: OPJ_UINT32,
  pub comp_t: OPJ_UINT32,
  pub prc_t: OPJ_UINT32,
  pub tx0_t: OPJ_UINT32,
  pub ty0_t: OPJ_UINT32,
}
pub type opj_poc_t = opj_poc;

#[repr(C)]
#[derive(Clone, Debug)]
pub struct opj_cparameters {
  pub tile_size_on: OPJ_BOOL,
  pub cp_tx0: core::ffi::c_int,
  pub cp_ty0: core::ffi::c_int,
  pub cp_tdx: core::ffi::c_int,
  pub cp_tdy: core::ffi::c_int,
  pub cp_disto_alloc: core::ffi::c_int,
  pub cp_fixed_alloc: core::ffi::c_int,
  pub cp_fixed_quality: core::ffi::c_int,
  pub cp_matrice: *mut core::ffi::c_int,
  pub cp_comment: *mut core::ffi::c_char,
  pub csty: core::ffi::c_int,
  pub prog_order: OPJ_PROG_ORDER,
  pub POC: [opj_poc_t; 32],
  pub numpocs: OPJ_UINT32,
  pub tcp_numlayers: core::ffi::c_int,
  pub tcp_rates: [core::ffi::c_float; 100],
  pub tcp_distoratio: [core::ffi::c_float; 100],
  pub numresolution: core::ffi::c_int,
  pub cblockw_init: core::ffi::c_int,
  pub cblockh_init: core::ffi::c_int,
  pub mode: core::ffi::c_int,
  pub irreversible: core::ffi::c_int,
  pub roi_compno: core::ffi::c_int,
  pub roi_shift: core::ffi::c_int,
  pub res_spec: core::ffi::c_int,
  pub prcw_init: [core::ffi::c_int; 33],
  pub prch_init: [core::ffi::c_int; 33],
  pub infile: [core::ffi::c_char; 4096],
  pub outfile: [core::ffi::c_char; 4096],
  pub index_on: core::ffi::c_int,
  pub index: [core::ffi::c_char; 4096],
  pub image_offset_x0: core::ffi::c_int,
  pub image_offset_y0: core::ffi::c_int,
  pub subsampling_dx: core::ffi::c_int,
  pub subsampling_dy: core::ffi::c_int,
  pub decod_format: core::ffi::c_int,
  pub cod_format: core::ffi::c_int,
  pub jpwl_epc_on: OPJ_BOOL,
  pub jpwl_hprot_MH: core::ffi::c_int,
  pub jpwl_hprot_TPH_tileno: [core::ffi::c_int; 16],
  pub jpwl_hprot_TPH: [core::ffi::c_int; 16],
  pub jpwl_pprot_tileno: [core::ffi::c_int; 16],
  pub jpwl_pprot_packno: [core::ffi::c_int; 16],
  pub jpwl_pprot: [core::ffi::c_int; 16],
  pub jpwl_sens_size: core::ffi::c_int,
  pub jpwl_sens_addr: core::ffi::c_int,
  pub jpwl_sens_range: core::ffi::c_int,
  pub jpwl_sens_MH: core::ffi::c_int,
  pub jpwl_sens_TPH_tileno: [core::ffi::c_int; 16],
  pub jpwl_sens_TPH: [core::ffi::c_int; 16],
  pub cp_cinema: OPJ_CINEMA_MODE,
  pub max_comp_size: core::ffi::c_int,
  pub cp_rsiz: OPJ_RSIZ_CAPABILITIES,
  pub tp_on: core::ffi::c_char,
  pub tp_flag: core::ffi::c_char,
  pub tcp_mct: core::ffi::c_char,
  pub jpip_on: OPJ_BOOL,
  pub mct_data: *mut core::ffi::c_void,
  pub max_cs_size: core::ffi::c_int,
  pub rsiz: OPJ_UINT16,
}
pub type opj_cparameters_t = opj_cparameters;

impl Drop for opj_cparameters {
  fn drop(&mut self) {
    if !self.cp_matrice.is_null() {
      opj_free(self.cp_matrice as *mut core::ffi::c_void);
      self.cp_matrice = core::ptr::null_mut();
    }
    if !self.cp_comment.is_null() {
      opj_free(self.cp_comment as *mut core::ffi::c_void);
      self.cp_comment = core::ptr::null_mut();
    }
    if !self.mct_data.is_null() {
      opj_free(self.mct_data);
      self.mct_data = core::ptr::null_mut();
    }
  }
}

impl Default for opj_cparameters_t {
  fn default() -> Self {
    Self {
      tile_size_on: Default::default(),
      cp_tx0: Default::default(),
      cp_ty0: Default::default(),
      cp_tdx: Default::default(),
      cp_tdy: Default::default(),
      cp_disto_alloc: Default::default(),
      cp_fixed_alloc: Default::default(),
      cp_fixed_quality: Default::default(),
      cp_matrice: core::ptr::null_mut(),
      cp_comment: core::ptr::null_mut(),
      csty: Default::default(),
      prog_order: OPJ_LRCP,
      POC: Default::default(),
      numpocs: Default::default(),
      tcp_numlayers: Default::default(),
      tcp_rates: [Default::default(); 100],
      tcp_distoratio: [Default::default(); 100],
      numresolution: 6i32,
      cblockw_init: 64i32,
      cblockh_init: 64i32,
      mode: Default::default(),
      irreversible: Default::default(),
      roi_compno: -1,
      roi_shift: Default::default(),
      res_spec: Default::default(),
      prcw_init: [Default::default(); 33],
      prch_init: [Default::default(); 33],
      infile: [0; 4096],
      outfile: [0; 4096],
      index_on: Default::default(),
      index: [0; 4096],
      image_offset_x0: Default::default(),
      image_offset_y0: Default::default(),
      subsampling_dx: 1,
      subsampling_dy: 1,
      decod_format: -1,
      cod_format: -1,
      jpwl_epc_on: 0,
      jpwl_hprot_MH: 0,
      jpwl_hprot_TPH_tileno: [0; 16],
      jpwl_hprot_TPH: [0; 16],
      jpwl_pprot_tileno: [0; 16],
      jpwl_pprot_packno: [0; 16],
      jpwl_pprot: [0; 16],
      jpwl_sens_size: 0,
      jpwl_sens_addr: 0,
      jpwl_sens_range: 0,
      jpwl_sens_MH: 0,
      jpwl_sens_TPH_tileno: [0; 16],
      jpwl_sens_TPH: [0; 16],
      cp_cinema: OPJ_OFF,
      max_comp_size: Default::default(),
      cp_rsiz: OPJ_STD_RSIZ,
      tp_on: Default::default(),
      tp_flag: Default::default(),
      tcp_mct: Default::default(),
      jpip_on: Default::default(),
      mct_data: core::ptr::null_mut(),
      max_cs_size: Default::default(),
      rsiz: Default::default(),
    }
  }
}

impl opj_cparameters_t {
  pub fn set_defaults(&mut self) {
    *self = Default::default()
  }

  pub fn set_comment(&mut self, comment: &str) {
    let len = comment.len();
    if !self.cp_comment.is_null() {
      opj_free(self.cp_comment as *mut core::ffi::c_void);
    }
    self.cp_comment = opj_malloc(len + 1) as *mut core::ffi::c_char;
    unsafe {
      core::ptr::copy_nonoverlapping(comment.as_ptr(), self.cp_comment as *mut u8, len);
      *self.cp_comment.offset(len as isize) = 0;
    }
  }

  pub fn set_MCT(
    &mut self,
    mut encoding_matrix: &[f32],
    mut dc_shift: &[i32],
    nb_comps: u32,
  ) -> bool {
    let mut l_matrix_size = encoding_matrix.len() * core::mem::size_of::<OPJ_FLOAT32>();
    let mut l_dc_shift_size = dc_shift.len() * core::mem::size_of::<OPJ_INT32>();
    let mut l_mct_total_size = l_matrix_size + l_dc_shift_size;
    /* add MCT capability */
    if self.rsiz as core::ffi::c_int & 0x8000i32 != 0 {
      self.rsiz = (self.rsiz as core::ffi::c_int | 0x100i32) as OPJ_UINT16
    } else {
      self.rsiz = (0x8000i32 | 0x100i32) as OPJ_UINT16
    }
    self.irreversible = 1i32;
    /* use array based MCT */
    self.tcp_mct = 2 as core::ffi::c_char;
    self.mct_data = opj_malloc(l_mct_total_size);
    if self.mct_data.is_null() {
      return false;
    }
    self
      .get_matrix_mut(nb_comps as usize)
      .copy_from_slice(encoding_matrix);
    self
      .get_dc_shift_mut(nb_comps as usize)
      .copy_from_slice(dc_shift);
    true
  }

  pub fn get_matrix_mut(&mut self, nb_comps: usize) -> &mut [f32] {
    let mut l_matrix_size = nb_comps * nb_comps;
    unsafe { core::slice::from_raw_parts_mut(self.mct_data as *mut f32, l_matrix_size) }
  }

  pub fn get_dc_shift_mut(&mut self, nb_comps: usize) -> &mut [i32] {
    let mut l_matrix_byte_size = nb_comps * nb_comps * core::mem::size_of::<OPJ_FLOAT32>();
    unsafe {
      core::slice::from_raw_parts_mut(
        self.mct_data.offset(l_matrix_byte_size as isize) as *mut i32,
        nb_comps,
      )
    }
  }
}

#[repr(C)]
#[derive(Clone, Debug)]
pub struct opj_dparameters {
  pub cp_reduce: OPJ_UINT32,
  pub cp_layer: OPJ_UINT32,
  pub infile: [core::ffi::c_char; 4096],
  pub outfile: [core::ffi::c_char; 4096],
  pub decod_format: core::ffi::c_int,
  pub cod_format: core::ffi::c_int,
  pub DA_x0: OPJ_UINT32,
  pub DA_x1: OPJ_UINT32,
  pub DA_y0: OPJ_UINT32,
  pub DA_y1: OPJ_UINT32,
  pub m_verbose: OPJ_BOOL,
  pub tile_index: OPJ_UINT32,
  pub nb_tile_to_decode: OPJ_UINT32,
  pub jpwl_correct: OPJ_BOOL,
  pub jpwl_exp_comps: core::ffi::c_int,
  pub jpwl_max_tiles: core::ffi::c_int,
  pub flags: core::ffi::c_uint,
}
pub type opj_dparameters_t = opj_dparameters;

impl Default for opj_dparameters_t {
  fn default() -> Self {
    Self {
      cp_reduce: Default::default(),
      cp_layer: Default::default(),
      infile: [Default::default(); 4096],
      outfile: [Default::default(); 4096],
      decod_format: -1,
      cod_format: -1,
      DA_x0: Default::default(),
      DA_x1: Default::default(),
      DA_y0: Default::default(),
      DA_y1: Default::default(),
      m_verbose: Default::default(),
      tile_index: Default::default(),
      nb_tile_to_decode: Default::default(),
      jpwl_correct: Default::default(),
      jpwl_exp_comps: Default::default(),
      jpwl_max_tiles: Default::default(),
      flags: Default::default(),
    }
  }
}

impl opj_dparameters_t {
  pub fn set_defaults(&mut self) {
    *self = Default::default()
  }
}

pub type opj_codec_t = *mut core::ffi::c_void;
pub type opj_stream_read_fn = Option<
  unsafe extern "C" fn(
    _: *mut core::ffi::c_void,
    _: OPJ_SIZE_T,
    _: *mut core::ffi::c_void,
  ) -> OPJ_SIZE_T,
>;
pub type opj_stream_write_fn = Option<
  unsafe extern "C" fn(
    _: *mut core::ffi::c_void,
    _: OPJ_SIZE_T,
    _: *mut core::ffi::c_void,
  ) -> OPJ_SIZE_T,
>;
pub type opj_stream_skip_fn =
  Option<unsafe extern "C" fn(_: OPJ_OFF_T, _: *mut core::ffi::c_void) -> OPJ_OFF_T>;
pub type opj_stream_seek_fn =
  Option<unsafe extern "C" fn(_: OPJ_OFF_T, _: *mut core::ffi::c_void) -> OPJ_BOOL>;
pub type opj_stream_free_user_data_fn =
  Option<unsafe extern "C" fn(_: *mut core::ffi::c_void) -> ()>;
pub type opj_stream_t = *mut core::ffi::c_void;

#[repr(C)]
#[derive(Copy, Debug)]
pub struct opj_image_comp {
  pub dx: OPJ_UINT32,
  pub dy: OPJ_UINT32,
  pub w: OPJ_UINT32,
  pub h: OPJ_UINT32,
  pub x0: OPJ_UINT32,
  pub y0: OPJ_UINT32,
  pub prec: OPJ_UINT32,
  pub bpp: OPJ_UINT32,
  pub sgnd: OPJ_UINT32,
  pub resno_decoded: OPJ_UINT32,
  pub factor: OPJ_UINT32,
  pub data: *mut OPJ_INT32,
  pub alpha: OPJ_UINT16,
}
pub type opj_image_comp_t = opj_image_comp;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_packet_info {
  pub start_pos: OPJ_OFF_T,
  pub end_ph_pos: OPJ_OFF_T,
  pub end_pos: OPJ_OFF_T,
  pub disto: core::ffi::c_double,
}
pub type opj_packet_info_t = opj_packet_info;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_marker_info {
  pub type_: core::ffi::c_ushort,
  pub pos: OPJ_OFF_T,
  pub len: core::ffi::c_int,
}
pub type opj_marker_info_t = opj_marker_info;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_tccp_info {
  pub compno: OPJ_UINT32,
  pub csty: OPJ_UINT32,
  pub numresolutions: OPJ_UINT32,
  pub cblkw: OPJ_UINT32,
  pub cblkh: OPJ_UINT32,
  pub cblksty: OPJ_UINT32,
  pub qmfbid: OPJ_UINT32,
  pub qntsty: OPJ_UINT32,
  pub stepsizes_mant: [OPJ_UINT32; 97],
  pub stepsizes_expn: [OPJ_UINT32; 97],
  pub numgbits: OPJ_UINT32,
  pub roishift: OPJ_INT32,
  pub prcw: [OPJ_UINT32; 33],
  pub prch: [OPJ_UINT32; 33],
}
pub type opj_tccp_info_t = opj_tccp_info;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_tile_v2_info {
  pub tileno: core::ffi::c_int,
  pub csty: OPJ_UINT32,
  pub prg: OPJ_PROG_ORDER,
  pub numlayers: OPJ_UINT32,
  pub mct: OPJ_UINT32,
  pub tccp_info: *mut opj_tccp_info_t,
}
pub type opj_tile_info_v2_t = opj_tile_v2_info;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_codestream_info_v2 {
  pub tx0: OPJ_UINT32,
  pub ty0: OPJ_UINT32,
  pub tdx: OPJ_UINT32,
  pub tdy: OPJ_UINT32,
  pub tw: OPJ_UINT32,
  pub th: OPJ_UINT32,
  pub nbcomps: OPJ_UINT32,
  pub m_default_tile_info: opj_tile_info_v2_t,
  pub tile_info: *mut opj_tile_info_v2_t,
}
pub type opj_codestream_info_v2_t = opj_codestream_info_v2;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_tp_info {
  pub tp_start_pos: core::ffi::c_int,
  pub tp_end_header: core::ffi::c_int,
  pub tp_end_pos: core::ffi::c_int,
  pub tp_start_pack: core::ffi::c_int,
  pub tp_numpacks: core::ffi::c_int,
}
pub type opj_tp_info_t = opj_tp_info;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_tile_info {
  pub thresh: *mut core::ffi::c_double,
  pub tileno: core::ffi::c_int,
  pub start_pos: core::ffi::c_int,
  pub end_header: core::ffi::c_int,
  pub end_pos: core::ffi::c_int,
  pub pw: [core::ffi::c_int; 33],
  pub ph: [core::ffi::c_int; 33],
  pub pdx: [core::ffi::c_int; 33],
  pub pdy: [core::ffi::c_int; 33],
  pub packet: *mut opj_packet_info_t,
  pub numpix: core::ffi::c_int,
  pub distotile: core::ffi::c_double,
  pub marknum: core::ffi::c_int,
  pub marker: *mut opj_marker_info_t,
  pub maxmarknum: core::ffi::c_int,
  pub num_tps: core::ffi::c_int,
  pub tp: *mut opj_tp_info_t,
}
pub type opj_tile_info_t = opj_tile_info;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_codestream_info {
  pub D_max: core::ffi::c_double,
  pub packno: core::ffi::c_int,
  pub index_write: core::ffi::c_int,
  pub image_w: core::ffi::c_int,
  pub image_h: core::ffi::c_int,
  pub prog: OPJ_PROG_ORDER,
  pub tile_x: core::ffi::c_int,
  pub tile_y: core::ffi::c_int,
  pub tile_Ox: core::ffi::c_int,
  pub tile_Oy: core::ffi::c_int,
  pub tw: core::ffi::c_int,
  pub th: core::ffi::c_int,
  pub numcomps: core::ffi::c_int,
  pub numlayers: core::ffi::c_int,
  pub numdecompos: *mut core::ffi::c_int,
  pub marknum: core::ffi::c_int,
  pub marker: *mut opj_marker_info_t,
  pub maxmarknum: core::ffi::c_int,
  pub main_head_start: core::ffi::c_int,
  pub main_head_end: core::ffi::c_int,
  pub codestream_size: core::ffi::c_int,
  pub tile: *mut opj_tile_info_t,
}
pub type opj_codestream_info_t = opj_codestream_info;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_tp_index {
  pub start_pos: OPJ_OFF_T,
  pub end_header: OPJ_OFF_T,
  pub end_pos: OPJ_OFF_T,
}
pub type opj_tp_index_t = opj_tp_index;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_tile_index {
  pub tileno: OPJ_UINT32,
  pub nb_tps: OPJ_UINT32,
  pub current_nb_tps: OPJ_UINT32,
  pub current_tpsno: OPJ_UINT32,
  pub tp_index: *mut opj_tp_index_t,
  pub marknum: OPJ_UINT32,
  pub marker: *mut opj_marker_info_t,
  pub maxmarknum: OPJ_UINT32,
  pub nb_packet: OPJ_UINT32,
  pub packet_index: *mut opj_packet_info_t,
}
pub type opj_tile_index_t = opj_tile_index;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_codestream_index {
  pub main_head_start: OPJ_OFF_T,
  pub main_head_end: OPJ_OFF_T,
  pub codestream_size: OPJ_UINT64,
  pub marknum: OPJ_UINT32,
  pub marker: *mut opj_marker_info_t,
  pub maxmarknum: OPJ_UINT32,
  pub nb_of_tiles: OPJ_UINT32,
  pub tile_index: *mut opj_tile_index_t,
}
pub type opj_codestream_index_t = opj_codestream_index;

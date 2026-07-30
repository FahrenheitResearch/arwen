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

use core::alloc::Layout;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

use super::c_api_types::*;
use super::j2k::*;

use super::event::opj_event_mgr;

#[derive(Clone)]
pub(crate) struct EncoderParameters {
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
  pub index_on: core::ffi::c_int,
  pub image_offset_x0: core::ffi::c_int,
  pub image_offset_y0: core::ffi::c_int,
  pub subsampling_dx: core::ffi::c_int,
  pub subsampling_dy: core::ffi::c_int,
  pub decod_format: core::ffi::c_int,
  pub cod_format: core::ffi::c_int,
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

impl Default for EncoderParameters {
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
      index_on: Default::default(),
      image_offset_x0: Default::default(),
      image_offset_y0: Default::default(),
      subsampling_dx: 1,
      subsampling_dy: 1,
      decod_format: -1,
      cod_format: -1,
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

impl EncoderParameters {
  pub fn from_c(params: &mut opj_cparameters_t) -> Self {
    Self {
      tile_size_on: params.tile_size_on,
      cp_tx0: params.cp_tx0,
      cp_ty0: params.cp_ty0,
      cp_tdx: params.cp_tdx,
      cp_tdy: params.cp_tdy,
      cp_disto_alloc: params.cp_disto_alloc,
      cp_fixed_alloc: params.cp_fixed_alloc,
      cp_fixed_quality: params.cp_fixed_quality,
      cp_matrice: params.cp_matrice,
      cp_comment: params.cp_comment,
      csty: params.csty,
      prog_order: params.prog_order,
      POC: params.POC,
      numpocs: params.numpocs,
      tcp_numlayers: params.tcp_numlayers,
      tcp_rates: params.tcp_rates,
      tcp_distoratio: params.tcp_distoratio,
      numresolution: params.numresolution,
      cblockw_init: params.cblockw_init,
      cblockh_init: params.cblockh_init,
      mode: params.mode,
      irreversible: params.irreversible,
      roi_compno: params.roi_compno,
      roi_shift: params.roi_shift,
      res_spec: params.res_spec,
      prcw_init: params.prcw_init,
      prch_init: params.prch_init,
      index_on: params.index_on,
      image_offset_x0: params.image_offset_x0,
      image_offset_y0: params.image_offset_y0,
      subsampling_dx: params.subsampling_dx,
      subsampling_dy: params.subsampling_dy,
      decod_format: params.decod_format,
      cod_format: params.cod_format,
      cp_cinema: params.cp_cinema,
      max_comp_size: params.max_comp_size,
      cp_rsiz: params.cp_rsiz,
      tp_on: params.tp_on,
      tp_flag: params.tp_flag,
      tcp_mct: params.tcp_mct,
      jpip_on: params.jpip_on,
      mct_data: params.mct_data,
      max_cs_size: params.max_cs_size,
      rsiz: params.rsiz,
    }
  }
}

#[derive(Clone)]
pub(crate) struct DecoderParameters {
  pub cp_reduce: OPJ_UINT32,
  pub cp_layer: OPJ_UINT32,
  pub decod_format: core::ffi::c_int,
  pub cod_format: core::ffi::c_int,
  pub DA_x0: OPJ_UINT32,
  pub DA_x1: OPJ_UINT32,
  pub DA_y0: OPJ_UINT32,
  pub DA_y1: OPJ_UINT32,
  pub m_verbose: OPJ_BOOL,
  pub tile_index: OPJ_UINT32,
  pub nb_tile_to_decode: OPJ_UINT32,
  pub flags: core::ffi::c_uint,
}

impl Default for DecoderParameters {
  fn default() -> Self {
    Self {
      cp_reduce: Default::default(),
      cp_layer: Default::default(),
      decod_format: -1,
      cod_format: -1,
      DA_x0: Default::default(),
      DA_x1: Default::default(),
      DA_y0: Default::default(),
      DA_y1: Default::default(),
      m_verbose: Default::default(),
      tile_index: Default::default(),
      nb_tile_to_decode: Default::default(),
      flags: Default::default(),
    }
  }
}

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_marker_info {
  pub need_PLT: OPJ_BOOL,
  pub packet_count: OPJ_UINT32,
  pub p_packet_size: *mut OPJ_UINT32,
}
pub(crate) type opj_tcd_marker_info_t = opj_tcd_marker_info;

pub struct Stream {
  pub(crate) m_inner: super::stream::StreamInner,
  pub(crate) m_stream_length: OPJ_UINT64,
  pub(crate) m_byte_offset: OPJ_OFF_T,
}
pub(crate) type opj_stream_private = Stream;
pub(crate) type opj_stream_private_t = Stream;

pub(crate) type opj_jp2_proc =
  fn(_: &mut opj_jp2, _: &mut Stream, _: &mut opj_event_mgr) -> OPJ_BOOL;
pub(crate) type opj_jp2_proc_list_t = super::function_list::ProcedureList<opj_jp2_proc>;

#[derive(Clone)]
pub(crate) struct opj_jp2 {
  pub j2k: opj_j2k,
  pub w: OPJ_UINT32,
  pub h: OPJ_UINT32,
  pub bpc: OPJ_UINT32,
  pub C: OPJ_UINT32,
  pub UnkC: OPJ_UINT32,
  pub IPR: OPJ_UINT32,
  pub meth: OPJ_UINT32,
  pub approx: OPJ_UINT32,
  pub enumcs: OPJ_UINT32,
  pub precedence: OPJ_UINT32,
  pub brand: OPJ_UINT32,
  pub minversion: OPJ_UINT32,
  pub cl: Vec<u32>,
  pub comps: Vec<opj_jp2_comps>,
  pub j2k_codestream_offset: OPJ_OFF_T,
  pub jpip_iptr_offset: OPJ_OFF_T,
  pub jpip_on: OPJ_BOOL,
  pub jp2_state: OPJ_UINT32,
  pub jp2_img_state: OPJ_UINT32,
  pub color: opj_jp2_color,
  pub ignore_pclr_cmap_cdef: OPJ_BOOL,
  pub has_jp2h: OPJ_BYTE,
  pub has_ihdr: OPJ_BYTE,
}

#[derive(Clone)]
pub(crate) struct opj_jp2_color {
  pub icc_profile: Option<ICCProfile>,
  pub jp2_cdef: Option<opj_jp2_cdef>,
  pub jp2_pclr: Option<opj_jp2_pclr>,
  pub jp2_has_colr: OPJ_BYTE,
}

impl opj_jp2_color {
  pub fn icc_profile_len(&self) -> usize {
    match &self.icc_profile {
      Some(ICCProfile::ICC(data)) => data.len(),
      Some(ICCProfile::CIELab(data)) => data.len(),
      None => 0,
    }
  }

  pub fn icc_profile(&self) -> Option<ICCProfileRef> {
    self.icc_profile.as_ref().map(|profile| match profile {
      ICCProfile::ICC(data) => ICCProfileRef::ICC(data),
      ICCProfile::CIELab(data) => ICCProfileRef::CIELab(data),
    })
  }

  pub fn set_cielab_profile(&mut self, data: Vec<u8>) {
    self.icc_profile = Some(ICCProfile::CIELab(data));
  }

  pub fn set_icc_profile(&mut self, data: Vec<u8>) {
    self.icc_profile = Some(ICCProfile::ICC(data));
  }
}

#[derive(Clone)]
pub enum ICCProfile {
  ICC(Vec<u8>),
  CIELab(Vec<u8>),
}
pub const CIE_LAB_BYTE_SIZE: usize = 36;

impl ICCProfile {
  pub fn new_icc(data: &[u8]) -> Self {
    ICCProfile::ICC(data.to_vec())
  }

  pub fn new_cielab(data: &[u8]) -> Self {
    ICCProfile::CIELab(data.to_vec())
  }
}

#[derive(Debug)]
pub enum ICCProfileRef<'a> {
  ICC(&'a [u8]),
  CIELab(&'a [u8]),
}

impl<'a> ICCProfileRef<'a> {
  pub fn as_slice(&self) -> &[u8] {
    match self {
      ICCProfileRef::ICC(data) => data,
      ICCProfileRef::CIELab(data) => data,
    }
  }

  pub fn is_cielab(&self) -> bool {
    match self {
      ICCProfileRef::CIELab(_) => true,
      _ => false,
    }
  }

  pub fn len(&self) -> usize {
    match self {
      ICCProfileRef::ICC(data) => data.len(),
      ICCProfileRef::CIELab(data) => data.len(),
    }
  }
}

#[derive(Debug)]
pub enum ICCProfileMut<'a> {
  ICC(&'a mut [u8]),
  CIELab(&'a mut [u8]),
}

impl<'a> ICCProfileMut<'a> {
  pub fn as_mut_slice(&mut self) -> &mut [u8] {
    match self {
      ICCProfileMut::ICC(data) => data,
      ICCProfileMut::CIELab(data) => data,
    }
  }

  pub fn is_cielab(&self) -> bool {
    match self {
      ICCProfileMut::CIELab(_) => true,
      _ => false,
    }
  }
}

impl ICCProfile {
  pub fn as_slice(&self) -> &[u8] {
    match self {
      ICCProfile::ICC(data) => data,
      ICCProfile::CIELab(data) => data,
    }
  }
}

#[derive(Copy, Clone)]
pub(crate) struct Jp2ChannelSign {
  pub sign: u8,
  pub size: u8,
}

#[derive(Clone)]
pub(crate) struct opj_jp2_pclr {
  pub entries: Vec<u32>,
  pub channel: Vec<Jp2ChannelSign>,
  pub cmap: Vec<opj_jp2_cmap_comp>,
  pub nr_entries: u16,
  pub nr_channels: u8,
}

#[derive(Copy, Clone)]
pub(crate) struct opj_jp2_cmap_comp {
  pub cmp: u16,
  pub mtyp: u8,
  pub pcol: u8,
}

#[derive(Clone)]
pub(crate) struct opj_jp2_cdef {
  pub info: Vec<opj_jp2_cdef_info>,
}
pub(crate) type opj_jp2_cdef_t = opj_jp2_cdef;

#[derive(Copy, Clone)]
pub(crate) struct opj_jp2_cdef_info {
  pub cn: OPJ_UINT16,
  pub typ: OPJ_UINT16,
  pub asoc: OPJ_UINT16,
}

#[derive(Copy, Clone, Default)]
pub(crate) struct opj_jp2_comps {
  //pub depth: OPJ_UINT32,
  //pub sgnd: OPJ_UINT32,
  pub bpcc: OPJ_UINT32,
}

pub(crate) type opj_j2k_proc =
  fn(_: &mut opj_j2k, _: &mut Stream, _: &mut opj_event_mgr) -> OPJ_BOOL;
pub(crate) type opj_j2k_proc_list_t = super::function_list::ProcedureList<opj_j2k_proc>;

#[derive(Clone)]
pub(crate) struct opj_j2k {
  pub m_is_decoder: OPJ_BOOL,
  pub m_specific_param: C2RustUnnamed_2,
  pub m_private_image: *mut opj_image_t,
  pub m_output_image: *mut opj_image_t,
  pub m_cp: opj_cp_t,
  pub cstr_index: *mut opj_codestream_index_t,
  pub m_current_tile_number: OPJ_UINT32,
  pub m_tcd: opj_tcd,
  pub ihdr_w: OPJ_UINT32,
  pub ihdr_h: OPJ_UINT32,
  pub dump_state: core::ffi::c_uint,
}

#[derive(Clone)]
pub(crate) struct opj_tcd {
  pub tp_pos: OPJ_INT32,
  pub tp_num: OPJ_UINT32,
  pub cur_tp_num: OPJ_UINT32,
  pub cur_totnum_tp: OPJ_UINT32,
  pub cur_pino: OPJ_UINT32,
  pub tcd_image: opj_tcd_image,
  pub image: *mut opj_image_t,
  pub cp: *mut opj_cp_t,
  pub tcp: *mut opj_tcp_t,
  pub tcd_tileno: OPJ_UINT32,
  pub m_is_decoder: bool,
  pub win_x0: OPJ_UINT32,
  pub win_y0: OPJ_UINT32,
  pub win_x1: OPJ_UINT32,
  pub win_y1: OPJ_UINT32,
  pub whole_tile_decoding: OPJ_BOOL,
  pub used_component: *mut OPJ_BOOL,
}

impl Default for opj_tcd {
  fn default() -> Self {
    Self {
      tp_pos: Default::default(),
      tp_num: Default::default(),
      cur_tp_num: Default::default(),
      cur_totnum_tp: Default::default(),
      cur_pino: Default::default(),
      tcd_image: Default::default(),
      image: core::ptr::null_mut(),
      cp: core::ptr::null_mut(),
      tcp: core::ptr::null_mut(),
      tcd_tileno: Default::default(),
      m_is_decoder: Default::default(),
      win_x0: Default::default(),
      win_y0: Default::default(),
      win_x1: Default::default(),
      win_y1: Default::default(),
      whole_tile_decoding: Default::default(),
      used_component: core::ptr::null_mut(),
    }
  }
}

#[derive(Copy, Clone)]
pub(crate) struct opj_tcp {
  pub csty: OPJ_UINT32,
  pub prg: OPJ_PROG_ORDER,
  pub numlayers: OPJ_UINT32,
  pub num_layers_to_decode: OPJ_UINT32,
  pub mct: OPJ_UINT32,
  pub rates: [OPJ_FLOAT32; 100],
  pub numpocs: OPJ_UINT32,
  pub pocs: [opj_poc_t; 32],
  pub ppt_markers_count: OPJ_UINT32,
  pub ppt_markers: *mut opj_ppx,
  pub ppt_data: *mut OPJ_BYTE,
  pub ppt_buffer: *mut OPJ_BYTE,
  pub ppt_data_size: OPJ_UINT32,
  pub ppt_len: OPJ_UINT32,
  pub distoratio: [OPJ_FLOAT32; 100],
  pub tccps: *mut opj_tccp_t,
  pub m_current_tile_part_number: OPJ_INT32,
  pub m_nb_tile_parts: OPJ_UINT32,
  pub m_data: *mut OPJ_BYTE,
  pub m_data_size: OPJ_UINT32,
  pub mct_norms: *mut OPJ_FLOAT64,
  pub m_mct_decoding_matrix: *mut OPJ_FLOAT32,
  pub m_mct_coding_matrix: *mut OPJ_FLOAT32,
  pub m_mct_records: *mut opj_mct_data_t,
  pub m_nb_mct_records: OPJ_UINT32,
  pub m_nb_max_mct_records: OPJ_UINT32,
  pub m_mcc_records: *mut opj_simple_mcc_decorrelation_data_t,
  pub m_nb_mcc_records: OPJ_UINT32,
  pub m_nb_max_mcc_records: OPJ_UINT32,
  pub cod: bool,
  pub ppt: bool,
  pub POC: bool,
}
pub(crate) type opj_tcp_t = opj_tcp;

#[derive(Copy, Clone)]
pub(crate) struct opj_simple_mcc_decorrelation_data {
  pub m_index: OPJ_UINT32,
  pub m_nb_comps: OPJ_UINT32,
  pub m_decorrelation_array: *mut opj_mct_data_t,
  pub m_offset_array: *mut opj_mct_data_t,
  pub m_is_irreversible: bool,
}
pub(crate) type opj_simple_mcc_decorrelation_data_t = opj_simple_mcc_decorrelation_data;

pub type J2K_MCT_ARRAY_TYPE = MCT_ARRAY_TYPE;
pub type MCT_ARRAY_TYPE = core::ffi::c_uint;
pub const MCT_TYPE_OFFSET: MCT_ARRAY_TYPE = 2;
pub const MCT_TYPE_DECORRELATION: MCT_ARRAY_TYPE = 1;
pub const MCT_TYPE_DEPENDENCY: MCT_ARRAY_TYPE = 0;

#[derive(Copy, Clone)]
pub(crate) struct opj_mct_data {
  pub m_element_type: MCTElementType,
  pub m_array_type: J2K_MCT_ARRAY_TYPE,
  pub m_index: OPJ_UINT32,
  pub m_data: *mut OPJ_BYTE,
  pub m_data_size: OPJ_UINT32,
}
pub(crate) type opj_mct_data_t = opj_mct_data;

#[derive(Copy, Clone)]
pub(crate) struct opj_tccp {
  pub csty: OPJ_UINT32,
  pub numresolutions: OPJ_UINT32,
  pub cblkw: OPJ_UINT32,
  pub cblkh: OPJ_UINT32,
  pub cblksty: OPJ_UINT32,
  pub qmfbid: OPJ_UINT32,
  pub qntsty: OPJ_UINT32,
  pub stepsizes: [opj_stepsize_t; 97],
  pub numgbits: OPJ_UINT32,
  pub roishift: OPJ_INT32,
  pub prcw: [OPJ_UINT32; 33],
  pub prch: [OPJ_UINT32; 33],
  pub m_dc_level_shift: OPJ_INT32,
}
pub(crate) type opj_tccp_t = opj_tccp;

#[derive(Copy, Clone)]
pub(crate) struct opj_stepsize {
  pub expn: OPJ_INT32,
  pub mant: OPJ_INT32,
}
pub(crate) type opj_stepsize_t = opj_stepsize;

#[derive(Copy, Clone)]
pub(crate) struct opj_ppx_struct {
  pub m_data: *mut OPJ_BYTE,
  pub m_data_size: OPJ_UINT32,
}
pub(crate) type opj_ppx = opj_ppx_struct;

#[derive(Copy, Clone)]
pub(crate) struct opj_cp {
  pub rsiz: OPJ_UINT16,
  pub tx0: OPJ_UINT32,
  pub ty0: OPJ_UINT32,
  pub tdx: OPJ_UINT32,
  pub tdy: OPJ_UINT32,
  pub comment: *mut OPJ_CHAR,
  pub tw: OPJ_UINT32,
  pub th: OPJ_UINT32,
  pub ppm_markers_count: OPJ_UINT32,
  pub ppm_markers: *mut opj_ppx,
  pub ppm_data: *mut OPJ_BYTE,
  pub ppm_len: OPJ_UINT32,
  pub ppm_data_read: OPJ_UINT32,
  pub ppm_data_current: *mut OPJ_BYTE,
  pub ppm_buffer: *mut OPJ_BYTE,
  pub ppm_data_first: *mut OPJ_BYTE,
  pub ppm_data_size: OPJ_UINT32,
  pub ppm_store: OPJ_INT32,
  pub ppm_previous: OPJ_INT32,
  pub tcps: *mut opj_tcp_t,
  pub m_specific_param: C2RustUnnamed_0,
  pub strict: OPJ_BOOL,
  pub ppm: bool,
  pub m_is_decoder: bool,
  pub allow_different_bit_depth_sign: bool,
}
pub(crate) type opj_cp_t = opj_cp;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) union C2RustUnnamed_0 {
  pub m_dec: opj_decoding_param_t,
  pub m_enc: opj_encoding_param_t,
}

/**
Rate allocation strategy
*/
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
#[repr(i32)]
pub(crate) enum J2K_QUALITY_LAYER_ALLOCATION_STRATEGY {
  /// allocation by rate/distortion
  RATE_DISTORTION_RATIO = 0,
  /// allocation by fixed distortion ratio (PSNR) (fixed quality)
  FIXED_DISTORTION_RATIO = 1,
  /// allocation by fixed layer (number of passes per layer / resolution / subband)
  FIXED_LAYER = 2,
}

#[derive(Copy, Clone)]
pub(crate) struct opj_encoding_param {
  pub m_max_comp_size: OPJ_UINT32,
  pub m_tp_pos: OPJ_INT32,
  pub m_matrice: *mut OPJ_INT32,
  pub m_tp_flag: OPJ_BYTE,
  pub m_quality_layer_alloc_strategy: J2K_QUALITY_LAYER_ALLOCATION_STRATEGY,
  pub m_tp_on: bool,
}
pub(crate) type opj_encoding_param_t = opj_encoding_param;

#[derive(Copy, Clone)]
pub(crate) struct opj_decoding_param {
  pub m_reduce: OPJ_UINT32,
  pub m_layer: OPJ_UINT32,
}
pub(crate) type opj_decoding_param_t = opj_decoding_param;

#[derive(Clone)]
pub(crate) struct opj_tcd_image {
  pub tiles: opj_tcd_tile,
}
pub(crate) type opj_tcd_image_t = opj_tcd_image;

impl Default for opj_tcd_image {
  fn default() -> Self {
    Self {
      tiles: Default::default(),
    }
  }
}

#[derive(Clone)]
pub(crate) struct opj_tcd_tile {
  pub x0: OPJ_INT32,
  pub y0: OPJ_INT32,
  pub x1: OPJ_INT32,
  pub y1: OPJ_INT32,
  pub numcomps: OPJ_UINT32,
  pub comps: *mut opj_tcd_tilecomp_t,
  pub numpix: OPJ_INT32,
  pub distotile: OPJ_FLOAT64,
  pub distolayer: [OPJ_FLOAT64; 100],
  pub packno: OPJ_UINT32,
}
pub(crate) type opj_tcd_tile_t = opj_tcd_tile;

impl Default for opj_tcd_tile {
  fn default() -> Self {
    Self {
      x0: Default::default(),
      y0: Default::default(),
      x1: Default::default(),
      y1: Default::default(),
      numcomps: Default::default(),
      comps: core::ptr::null_mut(),
      numpix: Default::default(),
      distotile: Default::default(),
      distolayer: [Default::default(); 100],
      packno: Default::default(),
    }
  }
}

#[derive(Clone, Default)]
pub struct TileInfo {
  pub index: u32,
  pub data_size: Option<u32>,
  pub x0: i32,
  pub y0: i32,
  pub x1: i32,
  pub y1: i32,
  pub nb_comps: u32,
  pub go_on: bool,
}

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_tilecomp {
  pub x0: OPJ_INT32,
  pub y0: OPJ_INT32,
  pub x1: OPJ_INT32,
  pub y1: OPJ_INT32,
  pub compno: OPJ_UINT32,
  pub numresolutions: OPJ_UINT32,
  pub minimum_num_resolutions: OPJ_UINT32,
  pub resolutions: *mut opj_tcd_resolution_t,
  pub resolutions_size: OPJ_UINT32,
  pub data: *mut OPJ_INT32,
  pub ownsData: OPJ_BOOL,
  pub data_size_needed: size_t,
  pub data_size: size_t,
  pub data_win: *mut OPJ_INT32,
  pub win_x0: OPJ_UINT32,
  pub win_y0: OPJ_UINT32,
  pub win_x1: OPJ_UINT32,
  pub win_y1: OPJ_UINT32,
  pub numpix: OPJ_INT32,
}
pub(crate) type opj_tcd_tilecomp_t = opj_tcd_tilecomp;

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_resolution {
  pub x0: OPJ_INT32,
  pub y0: OPJ_INT32,
  pub x1: OPJ_INT32,
  pub y1: OPJ_INT32,
  pub pw: OPJ_UINT32,
  pub ph: OPJ_UINT32,
  pub numbands: OPJ_UINT32,
  pub bands: [opj_tcd_band_t; 3],
  pub win_x0: OPJ_UINT32,
  pub win_y0: OPJ_UINT32,
  pub win_x1: OPJ_UINT32,
  pub win_y1: OPJ_UINT32,
}
pub(crate) type opj_tcd_resolution_t = opj_tcd_resolution;

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_band {
  pub x0: OPJ_INT32,
  pub y0: OPJ_INT32,
  pub x1: OPJ_INT32,
  pub y1: OPJ_INT32,
  pub bandno: OPJ_UINT32,
  pub precincts: *mut opj_tcd_precinct_t,
  pub precincts_data_size: OPJ_UINT32,
  pub numbps: OPJ_INT32,
  pub stepsize: OPJ_FLOAT32,
}
pub(crate) type opj_tcd_band_t = opj_tcd_band;

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_precinct {
  pub x0: OPJ_INT32,
  pub y0: OPJ_INT32,
  pub x1: OPJ_INT32,
  pub y1: OPJ_INT32,
  pub cw: OPJ_UINT32,
  pub ch: OPJ_UINT32,
  pub cblks: C2RustUnnamed_1,
  pub block_size: OPJ_UINT32,
  pub incltree: *mut opj_tgt_tree_t,
  pub imsbtree: *mut opj_tgt_tree_t,
}
pub(crate) type opj_tcd_precinct_t = opj_tcd_precinct;

#[derive(Copy, Clone)]
pub(crate) struct opj_tgt_tree {
  pub numleafsh: OPJ_UINT32,
  pub numleafsv: OPJ_UINT32,
  pub numnodes: OPJ_UINT32,
  pub nodes: *mut opj_tgt_node_t,
  pub nodes_size: OPJ_UINT32,
}
pub(crate) type opj_tgt_tree_t = opj_tgt_tree;

#[derive(Copy, Clone)]
pub(crate) struct opj_tgt_node {
  pub parent: *mut opj_tgt_node,
  pub value: OPJ_INT32,
  pub low: OPJ_INT32,
  pub known: OPJ_UINT32,
}
pub(crate) type opj_tgt_node_t = opj_tgt_node;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) union C2RustUnnamed_1 {
  pub enc: *mut opj_tcd_cblk_enc_t,
  pub dec: *mut opj_tcd_cblk_dec_t,
  pub blocks: *mut core::ffi::c_void,
}

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_cblk_dec {
  pub segs: *mut opj_tcd_seg_t,
  pub chunks: *mut opj_tcd_seg_data_chunk_t,
  pub x0: OPJ_INT32,
  pub y0: OPJ_INT32,
  pub x1: OPJ_INT32,
  pub y1: OPJ_INT32,
  pub Mb: OPJ_UINT32,
  pub numbps: OPJ_UINT32,
  pub numlenbits: OPJ_UINT32,
  pub numnewpasses: OPJ_UINT32,
  pub numsegs: OPJ_UINT32,
  pub real_num_segs: OPJ_UINT32,
  pub m_current_max_segs: OPJ_UINT32,
  pub numchunks: OPJ_UINT32,
  pub numchunksalloc: OPJ_UINT32,
  pub decoded_data: *mut OPJ_INT32,
  pub decoded_data_layout: Layout,
}
pub(crate) type opj_tcd_cblk_dec_t = opj_tcd_cblk_dec;

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_seg_data_chunk {
  pub data: *mut OPJ_BYTE,
  pub len: OPJ_UINT32,
}
pub(crate) type opj_tcd_seg_data_chunk_t = opj_tcd_seg_data_chunk;

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_seg {
  pub len: OPJ_UINT32,
  pub numpasses: OPJ_UINT32,
  pub real_num_passes: OPJ_UINT32,
  pub maxpasses: OPJ_UINT32,
  pub numnewpasses: OPJ_UINT32,
  pub newlen: OPJ_UINT32,
}
pub(crate) type opj_tcd_seg_t = opj_tcd_seg;

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_cblk_enc {
  pub data: *mut OPJ_BYTE,
  pub layers: *mut opj_tcd_layer_t,
  pub passes: *mut opj_tcd_pass_t,
  pub x0: OPJ_INT32,
  pub y0: OPJ_INT32,
  pub x1: OPJ_INT32,
  pub y1: OPJ_INT32,
  pub numbps: OPJ_UINT32,
  pub numlenbits: OPJ_UINT32,
  pub data_size: OPJ_UINT32,
  pub numpasses: OPJ_UINT32,
  pub numpassesinlayers: OPJ_UINT32,
  pub totalpasses: OPJ_UINT32,
}
pub(crate) type opj_tcd_cblk_enc_t = opj_tcd_cblk_enc;

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_pass {
  pub rate: OPJ_UINT32,
  pub distortiondec: OPJ_FLOAT64,
  pub len: OPJ_UINT32,
  pub term: bool,
}
pub(crate) type opj_tcd_pass_t = opj_tcd_pass;

#[derive(Copy, Clone)]
pub(crate) struct opj_tcd_layer {
  pub numpasses: OPJ_UINT32,
  pub len: OPJ_UINT32,
  pub disto: OPJ_FLOAT64,
  pub data: *mut OPJ_BYTE,
}
pub(crate) type opj_tcd_layer_t = opj_tcd_layer;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) union C2RustUnnamed_2 {
  pub m_decoder: opj_j2k_dec_t,
  pub m_encoder: opj_j2k_enc_t,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct opj_j2k_enc {
  pub m_current_poc_tile_part_number: OPJ_UINT32,
  pub m_current_tile_part_number: OPJ_UINT32,
  pub m_TLM: OPJ_BOOL,
  pub m_Ttlmi_is_byte: OPJ_BOOL,
  pub m_tlm_start: OPJ_OFF_T,
  pub m_tlm_sot_offsets_buffer: *mut OPJ_BYTE,
  pub m_tlm_sot_offsets_current: *mut OPJ_BYTE,
  pub m_total_tile_parts: OPJ_UINT32,
  pub m_encoded_tile_data: *mut OPJ_BYTE,
  pub m_encoded_tile_size: OPJ_UINT32,
  pub m_header_tile_data: *mut OPJ_BYTE,
  pub m_header_tile_data_size: OPJ_UINT32,
  pub m_PLT: OPJ_BOOL,
  pub m_reserved_bytes_for_PLT: OPJ_UINT32,
  pub m_nb_comps: OPJ_UINT32,
}
pub(crate) type opj_j2k_enc_t = opj_j2k_enc;

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct opj_j2k_dec {
  pub m_state: J2KState,
  pub m_default_tcp: *mut opj_tcp_t,
  pub m_header_data: *mut OPJ_BYTE,
  pub m_header_data_size: OPJ_UINT32,
  pub m_sot_length: OPJ_UINT32,
  pub m_start_tile_x: OPJ_UINT32,
  pub m_start_tile_y: OPJ_UINT32,
  pub m_end_tile_x: OPJ_UINT32,
  pub m_end_tile_y: OPJ_UINT32,
  pub m_tile_ind_to_dec: OPJ_INT32,
  pub m_last_sot_read_pos: OPJ_OFF_T,
  pub m_last_tile_part: OPJ_BOOL,
  pub m_numcomps_to_decode: OPJ_UINT32,
  pub m_comps_indices_to_decode: *mut OPJ_UINT32,
  pub m_can_decode: bool,
  pub m_discard_tiles: bool,
  pub m_skip_data: bool,
  pub m_nb_tile_parts_correction_checked: bool,
  pub m_nb_tile_parts_correction: bool,
}
pub(crate) type opj_j2k_dec_t = opj_j2k_dec;

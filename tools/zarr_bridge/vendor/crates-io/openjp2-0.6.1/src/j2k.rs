use super::cio::*;
use super::consts::*;
use super::dwt::*;
use super::event::*;
use super::image::*;
use super::invert::*;
use super::math::*;
use super::mct::*;
use super::openjpeg::*;
use super::pi::*;
use super::stream::*;
use super::tcd::*;

use super::malloc::*;

#[cfg(feature = "file-io")]
use ::libc::FILE;

use bitflags::bitflags;

extern "C" {
  fn strcpy(_: *mut core::ffi::c_char, _: *const core::ffi::c_char) -> *mut core::ffi::c_char;

  fn strcmp(_: *const core::ffi::c_char, _: *const core::ffi::c_char) -> core::ffi::c_int;

  fn strncmp(
    _: *const core::ffi::c_char,
    _: *const core::ffi::c_char,
    _: usize,
  ) -> core::ffi::c_int;

  fn strlen(_: *const core::ffi::c_char) -> usize;

  fn atoi(__nptr: *const core::ffi::c_char) -> core::ffi::c_int;

  fn getenv(__name: *const core::ffi::c_char) -> *mut core::ffi::c_char;

  fn floor(_: core::ffi::c_double) -> core::ffi::c_double;

  fn memcpy(
    _: *mut core::ffi::c_void,
    _: *const core::ffi::c_void,
    _: usize,
  ) -> *mut core::ffi::c_void;

  fn memset(_: *mut core::ffi::c_void, _: core::ffi::c_int, _: usize) -> *mut core::ffi::c_void;

  fn memmove(
    _: *mut core::ffi::c_void,
    _: *const core::ffi::c_void,
    _: usize,
  ) -> *mut core::ffi::c_void;
}

bitflags! {
  pub struct J2KState: u32 {
    const ERR = 32768;
    const EOC = 256;
    const DATA = 128;
    const NEOC = 64;
    const MT = 32;
    const TPH = 16;
    const TPHSOT = 8;
    const MH = 4;
    const MHSIZ = 2;
    const MHSOC = 1;
    const NONE = 0;
  }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum MCTElementType {
  DOUBLE = 3,
  FLOAT = 2,
  INT32 = 1,
  INT16 = 0,
}

impl MCTElementType {
  pub fn new(num: u32) -> Self {
    match num & 0b_11_u32 {
      0 => Self::INT16,
      1 => Self::INT32,
      2 => Self::FLOAT,
      3 => Self::DOUBLE,
      _ => Self::INT16,
    }
  }

  pub fn size(&self) -> u32 {
    match self {
      Self::INT16 => 2,
      Self::INT32 => 4,
      Self::FLOAT => 4,
      Self::DOUBLE => 8,
    }
  }

  pub fn read_to_float(
    &self,
    p_src_data: *const core::ffi::c_void,
    p_dest_data: *mut core::ffi::c_void,
    p_nb_elem: OPJ_UINT32,
  ) {
    match self {
      Self::INT16 => opj_j2k_read_int16_to_float(p_src_data, p_dest_data, p_nb_elem),
      Self::INT32 => opj_j2k_read_int32_to_float(p_src_data, p_dest_data, p_nb_elem),
      Self::FLOAT => opj_j2k_read_float32_to_float(p_src_data, p_dest_data, p_nb_elem),
      Self::DOUBLE => opj_j2k_read_float64_to_float(p_src_data, p_dest_data, p_nb_elem),
    }
  }

  pub fn read_to_int32(
    &self,
    p_src_data: *const core::ffi::c_void,
    p_dest_data: *mut core::ffi::c_void,
    p_nb_elem: OPJ_UINT32,
  ) {
    match self {
      Self::INT16 => opj_j2k_read_int16_to_int32(p_src_data, p_dest_data, p_nb_elem),
      Self::INT32 => opj_j2k_read_int32_to_int32(p_src_data, p_dest_data, p_nb_elem),
      Self::FLOAT => opj_j2k_read_float32_to_int32(p_src_data, p_dest_data, p_nb_elem),
      Self::DOUBLE => opj_j2k_read_float64_to_int32(p_src_data, p_dest_data, p_nb_elem),
    }
  }

  pub fn write_from_float(
    &self,
    p_src_data: *const core::ffi::c_void,
    p_dest_data: *mut core::ffi::c_void,
    p_nb_elem: OPJ_UINT32,
  ) {
    match self {
      Self::INT16 => opj_j2k_write_float_to_int16(p_src_data, p_dest_data, p_nb_elem),
      Self::INT32 => opj_j2k_write_float_to_int32(p_src_data, p_dest_data, p_nb_elem),
      Self::FLOAT => opj_j2k_write_float_to_float(p_src_data, p_dest_data, p_nb_elem),
      Self::DOUBLE => opj_j2k_write_float_to_float64(p_src_data, p_dest_data, p_nb_elem),
    }
  }
}

#[repr(C)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) enum ProgressionStep {
  Unknown = 0,
  Component = 67,
  Resolution = 82,
  Precinct = 80,
  Layer = 76,
}

impl ProgressionStep {
  pub fn as_byte(&self) -> u8 {
    match self {
      Self::Component => b'C',
      Self::Resolution => b'R',
      Self::Precinct => b'P',
      Self::Layer => b'L',
      Self::Unknown => 0,
    }
  }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub(crate) enum ProgressionOrder {
  Unknown = 0,
  CPRL,
  PCRL,
  RLCP,
  LRCP,
}

impl ProgressionOrder {
  pub fn from_c_enum(enum_prog: OPJ_PROG_ORDER) -> Self {
    match enum_prog {
      OPJ_CPRL => Self::CPRL,
      OPJ_PCRL => Self::PCRL,
      OPJ_RLCP => Self::RLCP,
      OPJ_LRCP => Self::LRCP,
      _ => Self::Unknown,
    }
  }

  pub fn get_order(&self) -> &'static [ProgressionStep] {
    use ProgressionStep::*;
    match self {
      Self::CPRL => &[Component, Precinct, Resolution, Layer],
      Self::PCRL => &[Precinct, Component, Resolution, Layer],
      Self::RLCP => &[Resolution, Layer, Component, Precinct],
      Self::LRCP => &[Layer, Resolution, Component, Precinct],
      Self::Unknown => &[],
    }
  }

  pub fn get_order_str(&self) -> &'static str {
    match self {
      Self::CPRL => "CPRL",
      Self::PCRL => "PCRL",
      Self::RLCP => "RLCP",
      Self::LRCP => "LRCP",
      Self::Unknown => "",
    }
  }

  pub fn get_step(&self, pos: i32) -> ProgressionStep {
    let steps = self.get_order();
    steps
      .get(pos as usize)
      .cloned()
      .unwrap_or(ProgressionStep::Unknown)
  }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub(crate) enum J2KMarker {
  /// UNKNOWN marker value
  UNK(u32),
  /// SOC marker value
  SOC,
  /// SOT marker value
  SOT,
  /// SOD marker value
  SOD,
  /// EOC marker value
  EOC,
  /// CAP marker value
  CAP,
  /// SIZ marker value
  SIZ,
  /// COD marker value
  COD,
  /// COC marker value
  COC,
  /// CPF marker value
  CPF,
  /// RGN marker value
  RGN,
  /// QCD marker value
  QCD,
  /// QCC marker value
  QCC,
  /// POC marker value
  POC,
  /// TLM marker value
  TLM,
  /// PLM marker value
  PLM,
  /// PLT marker value
  PLT,
  /// PPM marker value
  PPM,
  /// PPT marker value
  PPT,
  /// SOP marker value
  SOP,
  /// EPH marker value
  EPH,
  /// CRG marker value
  CRG,
  /// COM marker value
  COM,
  /// CBD marker value
  CBD,
  /// MCC marker value
  MCC,
  /// MCT marker value
  MCT,
  /// MCO marker value
  MCO,
  /// EPC marker value (Part 11: JPEG 2000 for Wireless)
  #[cfg(feature = "jpwl")]
  EPC,
  /// EPB marker value (Part 11: JPEG 2000 for Wireless)
  #[cfg(feature = "jpwl")]
  EPB,
  /// ESD marker value (Part 11: JPEG 2000 for Wireless)
  #[cfg(feature = "jpwl")]
  ESD,
  /// RED marker value (Part 11: JPEG 2000 for Wireless)
  #[cfg(feature = "jpwl")]
  RED,
  /// SEC marker value (Part 8: Secure JPEG 2000)
  #[cfg(feature = "jpspec")]
  SEC,
  /// INSEC marker value (Part 8: Secure JPEG 2000)
  #[cfg(feature = "jpspec")]
  INSEC,
}

impl From<u32> for J2KMarker {
  fn from(num: u32) -> Self {
    match num {
      0xff4f => Self::SOC,
      0xff90 => Self::SOT,
      0xff93 => Self::SOD,
      0xffd9 => Self::EOC,
      0xff50 => Self::CAP,
      0xff51 => Self::SIZ,
      0xff52 => Self::COD,
      0xff53 => Self::COC,
      0xff59 => Self::CPF,
      0xff5e => Self::RGN,
      0xff5c => Self::QCD,
      0xff5d => Self::QCC,
      0xff5f => Self::POC,
      0xff55 => Self::TLM,
      0xff57 => Self::PLM,
      0xff58 => Self::PLT,
      0xff60 => Self::PPM,
      0xff61 => Self::PPT,
      0xff91 => Self::SOP,
      0xff92 => Self::EPH,
      0xff63 => Self::CRG,
      0xff64 => Self::COM,
      0xff78 => Self::CBD,
      0xff75 => Self::MCC,
      0xff74 => Self::MCT,
      0xff77 => Self::MCO,
      #[cfg(feature = "jpwl")]
      0xff68 => Self::EPC,
      #[cfg(feature = "jpwl")]
      0xff66 => Self::EPB,
      #[cfg(feature = "jpwl")]
      0xff67 => Self::ESD,
      #[cfg(feature = "jpwl")]
      0xff69 => Self::RED,
      #[cfg(feature = "jpspec")]
      0xff65 => Self::SEC,
      #[cfg(feature = "jpspec")]
      0xff94 => Self::INSEC,
      num => Self::UNK(num),
    }
  }
}

impl J2KMarker {
  pub fn from_buffer(p_buffer: *const OPJ_BYTE) -> Self {
    let mut marker: OPJ_UINT32 = 0;
    /* Read 2 bytes as the new marker ID */
    opj_read_bytes(p_buffer, &mut marker, 2 as OPJ_UINT32);
    Self::from(marker)
  }

  pub fn as_u32(&self) -> u32 {
    match self {
      Self::SOC => 0xff4f,
      Self::SOT => 0xff90,
      Self::SOD => 0xff93,
      Self::EOC => 0xffd9,
      Self::CAP => 0xff50,
      Self::SIZ => 0xff51,
      Self::COD => 0xff52,
      Self::COC => 0xff53,
      Self::CPF => 0xff59,
      Self::RGN => 0xff5e,
      Self::QCD => 0xff5c,
      Self::QCC => 0xff5d,
      Self::POC => 0xff5f,
      Self::TLM => 0xff55,
      Self::PLM => 0xff57,
      Self::PLT => 0xff58,
      Self::PPM => 0xff60,
      Self::PPT => 0xff61,
      Self::SOP => 0xff91,
      Self::EPH => 0xff92,
      Self::CRG => 0xff63,
      Self::COM => 0xff64,
      Self::CBD => 0xff78,
      Self::MCC => 0xff75,
      Self::MCT => 0xff74,
      Self::MCO => 0xff77,
      #[cfg(feature = "jpwl")]
      Self::EPC => 0xff68,
      #[cfg(feature = "jpwl")]
      Self::EPB => 0xff66,
      #[cfg(feature = "jpwl")]
      Self::ESD => 0xff67,
      #[cfg(feature = "jpwl")]
      Self::RED => 0xff69,
      #[cfg(feature = "jpspec")]
      Self::SEC => 0xff65,
      #[cfg(feature = "jpspec")]
      Self::INSEC => 0xff94,
      Self::UNK(num) => *num,
    }
  }

  pub fn is_invalid(&self) -> bool {
    self.as_u32() < 0xff00u32
  }

  pub fn is_unknown(&self) -> bool {
    match self {
      Self::UNK(_) => true,
      _ => false,
    }
  }

  pub fn states(&self) -> J2KState {
    match self {
      Self::SOT => J2KState::MH | J2KState::TPHSOT,
      Self::COD => J2KState::MH | J2KState::TPH,
      Self::COC => J2KState::MH | J2KState::TPH,
      Self::RGN => J2KState::MH | J2KState::TPH,
      Self::QCD => J2KState::MH | J2KState::TPH,
      Self::QCC => J2KState::MH | J2KState::TPH,
      Self::POC => J2KState::MH | J2KState::TPH,
      Self::SIZ => J2KState::MHSIZ,
      Self::TLM => J2KState::MH,
      Self::PLM => J2KState::MH,
      Self::PLT => J2KState::TPH,
      Self::PPM => J2KState::MH,
      Self::PPT => J2KState::TPH,
      Self::SOP => J2KState::NONE,
      Self::CRG => J2KState::MH,
      Self::COM => J2KState::MH | J2KState::TPH,
      Self::MCT => J2KState::MH | J2KState::TPH,
      Self::CBD => J2KState::MH,
      Self::CAP => J2KState::MH,
      Self::CPF => J2KState::MH,
      Self::MCC => J2KState::MH | J2KState::TPH,
      Self::MCO => J2KState::MH | J2KState::TPH,
      #[cfg(feature = "jpwl")]
      Self::EPC => J2KState::MH | J2KState::TPH,
      #[cfg(feature = "jpwl")]
      Self::EPB => J2KState::MH | J2KState::TPH,
      #[cfg(feature = "jpwl")]
      Self::ESD => J2KState::MH | J2KState::TPH,
      #[cfg(feature = "jpwl")]
      Self::RED => J2KState::MH | J2KState::TPH,
      #[cfg(feature = "jpspec")]
      Self::SEC => J2K_DEC_STATE_MH,
      #[cfg(feature = "jpspec")]
      Self::INSEC => J2KState::NONE,

      _ => J2KState::MH | J2KState::TPH,
    }
  }

  pub fn handler(
    &self,
    p_j2k: &mut opj_j2k,
    p_header_data: *mut OPJ_BYTE,
    p_header_size: OPJ_UINT32,
    p_manager: &mut opj_event_mgr,
  ) -> OPJ_BOOL {
    match self {
      Self::SOT => opj_j2k_read_sot(p_j2k, p_header_data, p_header_size, p_manager),
      Self::COD => opj_j2k_read_cod(p_j2k, p_header_data, p_header_size, p_manager),
      Self::COC => opj_j2k_read_coc(p_j2k, p_header_data, p_header_size, p_manager),
      Self::RGN => opj_j2k_read_rgn(p_j2k, p_header_data, p_header_size, p_manager),
      Self::QCD => opj_j2k_read_qcd(p_j2k, p_header_data, p_header_size, p_manager),
      Self::QCC => opj_j2k_read_qcc(p_j2k, p_header_data, p_header_size, p_manager),
      Self::POC => opj_j2k_read_poc(p_j2k, p_header_data, p_header_size, p_manager),
      Self::SIZ => opj_j2k_read_siz(p_j2k, p_header_data, p_header_size, p_manager),
      Self::TLM => opj_j2k_read_tlm(p_j2k, p_header_data, p_header_size, p_manager),
      Self::PLM => opj_j2k_read_plm(p_j2k, p_header_data, p_header_size, p_manager),
      Self::PLT => opj_j2k_read_plt(p_j2k, p_header_data, p_header_size, p_manager),
      Self::PPM => opj_j2k_read_ppm(p_j2k, p_header_data, p_header_size, p_manager),
      Self::PPT => opj_j2k_read_ppt(p_j2k, p_header_data, p_header_size, p_manager),
      Self::CRG => opj_j2k_read_crg(p_j2k, p_header_data, p_header_size, p_manager),
      Self::COM => opj_j2k_read_com(p_j2k, p_header_data, p_header_size, p_manager),
      Self::MCT => opj_j2k_read_mct(p_j2k, p_header_data, p_header_size, p_manager),
      Self::CBD => opj_j2k_read_cbd(p_j2k, p_header_data, p_header_size, p_manager),
      Self::CAP => opj_j2k_read_cap(p_j2k, p_header_data, p_header_size, p_manager),
      Self::CPF => opj_j2k_read_cpf(p_j2k, p_header_data, p_header_size, p_manager),
      Self::MCC => opj_j2k_read_mcc(p_j2k, p_header_data, p_header_size, p_manager),
      Self::MCO => opj_j2k_read_mco(p_j2k, p_header_data, p_header_size, p_manager),
      #[cfg(feature = "jpwl")]
      Self::EPC => opj_j2k_read_epc(p_j2k, p_header_data, p_header_size, p_manager),
      #[cfg(feature = "jpwl")]
      Self::EPB => opj_j2k_read_epb(p_j2k, p_header_data, p_header_size, p_manager),
      #[cfg(feature = "jpwl")]
      Self::ESD => opj_j2k_read_esd(p_j2k, p_header_data, p_header_size, p_manager),
      #[cfg(feature = "jpwl")]
      Self::RED => opj_j2k_read_red(p_j2k, p_header_data, p_header_size, p_manager),
      #[cfg(feature = "jpspec")]
      Self::SEC => opj_j2k_read_sec(p_j2k, p_header_data, p_header_size, p_manager),
      #[cfg(feature = "jpspec")]
      Self::INSEC => opj_j2k_read_insec(p_j2k, p_header_data, p_header_size, p_manager),
      _ => {
        // No handler for this marker.
        event_msg!(
          &mut *p_manager,
          EVT_ERROR,
          "No handler for unknown marker.\n",
        );
        0i32
      }
    }
  }
}

/* *
 * Updates the Tile Length Marker.
 */
fn opj_j2k_update_tlm(mut p_j2k: &mut opj_j2k, mut p_tile_part_size: OPJ_UINT32) {
  unsafe {
    if p_j2k.m_specific_param.m_encoder.m_Ttlmi_is_byte != 0 {
      opj_write_bytes(
        p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_current,
        p_j2k.m_current_tile_number,
        1 as OPJ_UINT32,
      ); /* PSOT */
      p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_current = p_j2k
        .m_specific_param
        .m_encoder
        .m_tlm_sot_offsets_current
        .offset(1)
    } else {
      opj_write_bytes(
        p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_current,
        p_j2k.m_current_tile_number,
        2 as OPJ_UINT32,
      );
      p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_current = p_j2k
        .m_specific_param
        .m_encoder
        .m_tlm_sot_offsets_current
        .offset(2)
    }
    opj_write_bytes(
      p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_current,
      p_tile_part_size,
      4 as OPJ_UINT32,
    );
    p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_current = p_j2k
      .m_specific_param
      .m_encoder
      .m_tlm_sot_offsets_current
      .offset(4);
  }
}

fn opj_j2k_read_int16_to_float(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_src_data = p_src_data as *mut OPJ_BYTE;
    let mut l_dest_data = p_dest_data as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_UINT32 = 0;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      opj_read_bytes(l_src_data, &mut l_temp, 2 as OPJ_UINT32);
      l_src_data = l_src_data.add(core::mem::size_of::<OPJ_INT16>());
      let fresh0 = l_dest_data;
      l_dest_data = l_dest_data.offset(1);
      *fresh0 = l_temp as OPJ_FLOAT32;
      i += 1;
    }
  }
}

fn opj_j2k_read_int32_to_float(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_src_data = p_src_data as *mut OPJ_BYTE;
    let mut l_dest_data = p_dest_data as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_UINT32 = 0;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      opj_read_bytes(l_src_data, &mut l_temp, 4 as OPJ_UINT32);
      l_src_data = l_src_data.add(core::mem::size_of::<OPJ_INT32>());
      let fresh1 = l_dest_data;
      l_dest_data = l_dest_data.offset(1);
      *fresh1 = l_temp as OPJ_FLOAT32;
      i += 1;
    }
  }
}

fn opj_j2k_read_float32_to_float(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_src_data = p_src_data as *mut OPJ_BYTE;
    let mut l_dest_data = p_dest_data as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_FLOAT32 = 0.;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      opj_read_float(l_src_data, &mut l_temp);
      l_src_data = l_src_data.add(core::mem::size_of::<OPJ_FLOAT32>());
      let fresh2 = l_dest_data;
      l_dest_data = l_dest_data.offset(1);
      *fresh2 = l_temp;
      i += 1;
    }
  }
}

fn opj_j2k_read_float64_to_float(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_src_data = p_src_data as *mut OPJ_BYTE;
    let mut l_dest_data = p_dest_data as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_FLOAT64 = 0.;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      opj_read_double(l_src_data, &mut l_temp);
      l_src_data = l_src_data.add(core::mem::size_of::<OPJ_FLOAT64>());
      let fresh3 = l_dest_data;
      l_dest_data = l_dest_data.offset(1);
      *fresh3 = l_temp as OPJ_FLOAT32;
      i += 1;
    }
  }
}

fn opj_j2k_read_int16_to_int32(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_src_data = p_src_data as *mut OPJ_BYTE;
    let mut l_dest_data = p_dest_data as *mut OPJ_INT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_UINT32 = 0;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      opj_read_bytes(l_src_data, &mut l_temp, 2 as OPJ_UINT32);
      l_src_data = l_src_data.add(core::mem::size_of::<OPJ_INT16>());
      let fresh4 = l_dest_data;
      l_dest_data = l_dest_data.offset(1);
      *fresh4 = l_temp as OPJ_INT32;
      i += 1;
    }
  }
}

fn opj_j2k_read_int32_to_int32(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_src_data = p_src_data as *mut OPJ_BYTE;
    let mut l_dest_data = p_dest_data as *mut OPJ_INT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_UINT32 = 0;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      opj_read_bytes(l_src_data, &mut l_temp, 4 as OPJ_UINT32);
      l_src_data = l_src_data.add(core::mem::size_of::<OPJ_INT32>());
      let fresh5 = l_dest_data;
      l_dest_data = l_dest_data.offset(1);
      *fresh5 = l_temp as OPJ_INT32;
      i += 1;
    }
  }
}

fn opj_j2k_read_float32_to_int32(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_src_data = p_src_data as *mut OPJ_BYTE;
    let mut l_dest_data = p_dest_data as *mut OPJ_INT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_FLOAT32 = 0.;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      opj_read_float(l_src_data, &mut l_temp);
      l_src_data = l_src_data.add(core::mem::size_of::<OPJ_FLOAT32>());
      let fresh6 = l_dest_data;
      l_dest_data = l_dest_data.offset(1);
      *fresh6 = l_temp as OPJ_INT32;
      i += 1;
    }
  }
}

fn opj_j2k_read_float64_to_int32(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_src_data = p_src_data as *mut OPJ_BYTE;
    let mut l_dest_data = p_dest_data as *mut OPJ_INT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_FLOAT64 = 0.;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      opj_read_double(l_src_data, &mut l_temp);
      l_src_data = l_src_data.add(core::mem::size_of::<OPJ_FLOAT64>());
      let fresh7 = l_dest_data;
      l_dest_data = l_dest_data.offset(1);
      *fresh7 = l_temp as OPJ_INT32;
      i += 1;
    }
  }
}

fn opj_j2k_write_float_to_int16(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_dest_data = p_dest_data as *mut OPJ_BYTE;
    let mut l_src_data = p_src_data as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_UINT32 = 0;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      let fresh8 = l_src_data;
      l_src_data = l_src_data.offset(1);
      l_temp = *fresh8 as OPJ_UINT32;
      opj_write_bytes(
        l_dest_data,
        l_temp,
        core::mem::size_of::<OPJ_INT16>() as OPJ_UINT32,
      );
      l_dest_data = l_dest_data.add(core::mem::size_of::<OPJ_INT16>());
      i += 1;
    }
  }
}

fn opj_j2k_write_float_to_int32(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_dest_data = p_dest_data as *mut OPJ_BYTE;
    let mut l_src_data = p_src_data as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_UINT32 = 0;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      let fresh9 = l_src_data;
      l_src_data = l_src_data.offset(1);
      l_temp = *fresh9 as OPJ_UINT32;
      opj_write_bytes(
        l_dest_data,
        l_temp,
        core::mem::size_of::<OPJ_INT32>() as OPJ_UINT32,
      );
      l_dest_data = l_dest_data.add(core::mem::size_of::<OPJ_INT32>());
      i += 1;
    }
  }
}

fn opj_j2k_write_float_to_float(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_dest_data = p_dest_data as *mut OPJ_BYTE;
    let mut l_src_data = p_src_data as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_FLOAT32 = 0.;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      let fresh10 = l_src_data;
      l_src_data = l_src_data.offset(1);
      l_temp = *fresh10;
      opj_write_float(l_dest_data, l_temp);
      l_dest_data = l_dest_data.add(core::mem::size_of::<OPJ_FLOAT32>());
      i += 1;
    }
  }
}

fn opj_j2k_write_float_to_float64(
  mut p_src_data: *const core::ffi::c_void,
  mut p_dest_data: *mut core::ffi::c_void,
  mut p_nb_elem: OPJ_UINT32,
) {
  unsafe {
    let mut l_dest_data = p_dest_data as *mut OPJ_BYTE;
    let mut l_src_data = p_src_data as *mut OPJ_FLOAT32;
    let mut i: OPJ_UINT32 = 0;
    let mut l_temp: OPJ_FLOAT64 = 0.;
    i = 0 as OPJ_UINT32;
    while i < p_nb_elem {
      let fresh11 = l_src_data;
      l_src_data = l_src_data.offset(1);
      l_temp = *fresh11 as OPJ_FLOAT64;
      opj_write_double(l_dest_data, l_temp);
      l_dest_data = l_dest_data.add(core::mem::size_of::<OPJ_FLOAT64>());
      i += 1;
    }
  }
}

pub(crate) fn opj_j2k_convert_progression_order(prg_order: OPJ_PROG_ORDER) -> ProgressionOrder {
  ProgressionOrder::from_c_enum(prg_order)
}
/* *
 * Checks the progression order changes values. Tells of the poc given as input are valid.
 * A nice message is outputted at errors.
 *
 * @param       p_pocs                  the progression order changes.
 * @param       tileno                  the tile number of interest
 * @param       p_nb_pocs               the number of progression order changes.
 * @param       p_nb_resolutions        the number of resolutions.
 * @param       numcomps                the number of components
 * @param       numlayers               the number of layers.
 * @param       p_manager               the user event manager.
 *
 * @return      true if the pocs are valid.
 */
fn opj_j2k_check_poc_val(
  mut p_pocs: *const opj_poc_t,
  mut tileno: OPJ_UINT32,
  mut p_nb_pocs: OPJ_UINT32,
  mut p_nb_resolutions: OPJ_UINT32,
  mut p_num_comps: OPJ_UINT32,
  mut p_num_layers: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut packet_array = core::ptr::null_mut::<OPJ_UINT32>();
    let mut index: OPJ_UINT32 = 0;
    let mut resno: OPJ_UINT32 = 0;
    let mut compno: OPJ_UINT32 = 0;
    let mut layno: OPJ_UINT32 = 0;
    let mut i: OPJ_UINT32 = 0;
    let mut step_c = 1 as OPJ_UINT32;
    let mut step_r = p_num_comps.wrapping_mul(step_c);
    let mut step_l = p_nb_resolutions.wrapping_mul(step_r);
    let mut loss = 0i32;
    assert!(p_nb_pocs > 0u32);
    packet_array = opj_calloc(
      (step_l as size_t).wrapping_mul(p_num_layers as usize),
      core::mem::size_of::<OPJ_UINT32>(),
    ) as *mut OPJ_UINT32;
    if packet_array.is_null() {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory for checking the poc values.\n",
      );
      return 0i32;
    }
    /* iterate through all the pocs that match our tile of interest. */
    i = 0 as OPJ_UINT32;
    while i < p_nb_pocs {
      let mut poc: *const opj_poc_t = &*p_pocs.offset(i as isize) as *const opj_poc_t;
      if tileno.wrapping_add(1u32) == (*poc).tile {
        index = step_r.wrapping_mul((*poc).resno0);
        /* take each resolution for each poc */
        resno = (*poc).resno0;
        while resno < opj_uint_min((*poc).resno1, p_nb_resolutions) {
          let mut res_index = index.wrapping_add((*poc).compno0.wrapping_mul(step_c));
          /* take each comp of each resolution for each poc */
          compno = (*poc).compno0;
          while compno < opj_uint_min((*poc).compno1, p_num_comps) {
            /* The layer index always starts at zero for every progression. */
            let layno0 = 0 as OPJ_UINT32;
            let mut comp_index = res_index.wrapping_add(layno0.wrapping_mul(step_l));
            /* and finally take each layer of each res of ... */
            layno = layno0;
            while layno < opj_uint_min((*poc).layno1, p_num_layers) {
              *packet_array.offset(comp_index as isize) = 1 as OPJ_UINT32;
              comp_index = (comp_index as core::ffi::c_uint).wrapping_add(step_l) as OPJ_UINT32;
              layno += 1;
            }
            res_index = (res_index as core::ffi::c_uint).wrapping_add(step_c) as OPJ_UINT32;
            compno += 1;
          }
          index = (index as core::ffi::c_uint).wrapping_add(step_r) as OPJ_UINT32;
          resno += 1;
        }
      }
      i += 1;
    }
    index = 0 as OPJ_UINT32;
    layno = 0 as OPJ_UINT32;
    while layno < p_num_layers {
      resno = 0 as OPJ_UINT32;
      while resno < p_nb_resolutions {
        compno = 0 as OPJ_UINT32;
        while compno < p_num_comps {
          loss |= (*packet_array.offset(index as isize) != 1u32) as core::ffi::c_int;
          index = (index as core::ffi::c_uint).wrapping_add(step_c) as OPJ_UINT32;
          compno += 1;
        }
        resno += 1;
      }
      layno += 1;
    }
    if loss != 0 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Missing packets possible loss of data\n",
      );
    }
    opj_free(packet_array as *mut core::ffi::c_void);
    (loss == 0) as core::ffi::c_int
  }
}

/* *
 * Gets the number of tile parts used for the given change of progression (if any) and the given tile.
 *
 * @param               cp                      the coding parameters.
 * @param               pino            the offset of the given poc (i.e. its position in the coding parameter).
 * @param               tileno          the given tile.
 *
 * @return              the number of tile parts.
 */
/* ----------------------------------------------------------------------- */
fn opj_j2k_get_num_tp(
  mut cp: *mut opj_cp_t,
  mut pino: OPJ_UINT32,
  mut tileno: OPJ_UINT32,
) -> OPJ_UINT32 {
  unsafe {
    let mut i: OPJ_INT32 = 0;
    let mut tpnum = 1 as OPJ_UINT32;
    let mut tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_current_poc = core::ptr::null_mut::<opj_poc_t>();
    /*  preconditions */

    assert!(tileno < (*cp).tw.wrapping_mul((*cp).th));
    assert!(
      pino
        < (*(*cp).tcps.offset(tileno as isize))
          .numpocs
          .wrapping_add(1u32)
    );
    /* get the given tile coding parameter */
    tcp = &mut *(*cp).tcps.offset(tileno as isize) as *mut opj_tcp_t;
    assert!(!tcp.is_null());
    l_current_poc = &mut *(*tcp).pocs.as_mut_ptr().offset(pino as isize) as *mut opj_poc_t;
    assert!(!l_current_poc.is_null());
    /* get the progression order as a character string */
    let prog = opj_j2k_convert_progression_order((*tcp).prg);
    assert!(prog != ProgressionOrder::Unknown);
    if (*cp).m_specific_param.m_enc.m_tp_on {
      for step in prog.get_order() {
        match step {
          ProgressionStep::Component => {
            /* component wise */
            tpnum = (tpnum as core::ffi::c_uint).wrapping_mul((*l_current_poc).compE) as OPJ_UINT32
          }
          ProgressionStep::Resolution => {
            /* resolution wise */
            tpnum = (tpnum as core::ffi::c_uint).wrapping_mul((*l_current_poc).resE) as OPJ_UINT32
          }
          ProgressionStep::Precinct => {
            /* precinct wise */
            tpnum = (tpnum as core::ffi::c_uint).wrapping_mul((*l_current_poc).prcE) as OPJ_UINT32
          }
          ProgressionStep::Layer => {
            /* layer wise */
            tpnum = (tpnum as core::ffi::c_uint).wrapping_mul((*l_current_poc).layE) as OPJ_UINT32
          }
          ProgressionStep::Unknown => {}
        }
        /* would we split here ? */
        if (*cp).m_specific_param.m_enc.m_tp_flag == *step as u8 {
          (*cp).m_specific_param.m_enc.m_tp_pos = i;
          break;
        }
      }
    } else {
      tpnum = 1 as OPJ_UINT32
    }
    tpnum
  }
}

/* *
 * Calculates the total number of tile parts needed by the encoder to
 * encode such an image. If not enough memory is available, then the function return false.
 *
 * @param       p_nb_tiles      pointer that will hold the number of tile parts.
 * @param       cp                      the coding parameters for the image.
 * @param       image           the image to encode.
 * @param       p_j2k                   the p_j2k encoder.
 * @param       p_manager       the user event manager.
 *
 * @return true if the function was successful, false else.
 */
fn opj_j2k_calculate_tp(
  mut cp: *mut opj_cp_t,
  mut p_nb_tiles: *mut OPJ_UINT32,
  mut image: &mut opj_image,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut pino: OPJ_UINT32 = 0;
    let mut tileno: OPJ_UINT32 = 0;
    let mut l_nb_tiles: OPJ_UINT32 = 0;
    let mut tcp = core::ptr::null_mut::<opj_tcp_t>();
    /* preconditions */

    assert!(!p_nb_tiles.is_null());
    assert!(!cp.is_null());
    l_nb_tiles = (*cp).tw.wrapping_mul((*cp).th);
    *p_nb_tiles = 0 as OPJ_UINT32;
    tcp = (*cp).tcps;
    /* INDEX >> */
    /* TODO mergeV2: check this part which use cstr_info */
    /*if (p_j2k->cstr_info) {
            opj_tile_info_t * l_info_tile_ptr = p_j2k->cstr_info->tile;

            for (tileno = 0; tileno < l_nb_tiles; ++tileno) {
                    OPJ_UINT32 cur_totnum_tp = 0;

                    opj_pi_update_encoding_parameters(image,cp,tileno);

                    for (pino = 0; pino <= tcp->numpocs; ++pino)
                    {
                            OPJ_UINT32 tp_num = opj_j2k_get_num_tp(cp,pino,tileno);

                            *p_nb_tiles = *p_nb_tiles + tp_num;

                            cur_totnum_tp += tp_num;
                    }

                    tcp->m_nb_tile_parts = cur_totnum_tp;

                    l_info_tile_ptr->tp = (opj_tp_info_t *) opj_malloc(cur_totnum_tp * sizeof(opj_tp_info_t));
                    if (l_info_tile_ptr->tp == 00) {
                            return OPJ_FALSE;
                    }

                    memset(l_info_tile_ptr->tp,0,cur_totnum_tp * sizeof(opj_tp_info_t));

                    l_info_tile_ptr->num_tps = cur_totnum_tp;

                    ++l_info_tile_ptr;
                    ++tcp;
            }
    }
    else */
    tileno = 0 as OPJ_UINT32;
    while tileno < l_nb_tiles {
      let mut cur_totnum_tp = 0 as OPJ_UINT32;
      opj_pi_update_encoding_parameters(image, cp, tileno);
      pino = 0 as OPJ_UINT32;
      while pino <= (*tcp).numpocs {
        let mut tp_num = opj_j2k_get_num_tp(cp, pino, tileno);
        *p_nb_tiles = (*p_nb_tiles).wrapping_add(tp_num);
        cur_totnum_tp = (cur_totnum_tp as core::ffi::c_uint).wrapping_add(tp_num) as OPJ_UINT32;
        pino += 1;
      }
      (*tcp).m_nb_tile_parts = cur_totnum_tp;
      tcp = tcp.offset(1);
      tileno += 1;
    }
    1i32
  }
}

/*
 * -----------------------------------------------------------------------
 * -----------------------------------------------------------------------
 * -----------------------------------------------------------------------
 */
/* *
 * Writes the SOC marker (Start Of Codestream)
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_write_soc(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    /* 2 bytes will be written */
    let mut l_start_stream = core::ptr::null_mut::<OPJ_BYTE>();
    /* preconditions */

    l_start_stream = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    /* write SOC identifier */
    opj_write_bytes(l_start_stream, J2KMarker::SOC.as_u32(), 2 as OPJ_UINT32);
    if opj_stream_write_data(p_stream, l_start_stream, 2 as OPJ_SIZE_T, p_manager) != 2 {
      return 0i32;
    }
    /* UniPG>> */
    /* USE_JPWL */
    /* <<UniPG */
    1i32
  }
}

/* *
 * Reads a SOC marker (Start of Codestream)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_stream        XXX needs data
 * @param       p_manager       the user event manager.
*/
/* *
 * Reads a SOC marker (Start of Codestream)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_stream        FIXME DOC
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_read_soc(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_data: [OPJ_BYTE; 2] = [0; 2];
    /* preconditions */

    if opj_stream_read_data(p_stream, l_data.as_mut_ptr(), 2 as OPJ_SIZE_T, p_manager) != 2 {
      return 0i32;
    }
    let l_marker = J2KMarker::from_buffer(l_data.as_mut_ptr());
    if l_marker != J2KMarker::SOC {
      return 0i32;
    }
    /* Next marker should be a SIZ marker in the main header */
    p_j2k.m_specific_param.m_decoder.m_state = J2KState::MHSIZ;
    /* FIXME move it in a index structure included in p_j2k*/
    (*p_j2k.cstr_index).main_head_start = opj_stream_tell(p_stream) - 2i64;
    event_msg!(
      p_manager,
      EVT_INFO,
      "Start to read j2k main header (%ld).\n",
      (*p_j2k.cstr_index).main_head_start,
    );
    /* Add the marker to the codestream index*/
    if 0i32
      == opj_j2k_add_mhmarker(
        p_j2k.cstr_index,
        J2KMarker::SOC,
        (*p_j2k.cstr_index).main_head_start,
        2 as OPJ_UINT32,
      )
    {
      event_msg!(p_manager, EVT_ERROR, "Not enough memory to add mh marker\n",);
      return 0i32;
    }
    1i32
  }
}

/* *
 * Writes the SIZ marker (image and tile size)
 *
 * @param       p_j2k           J2K codec.
 * @param       p_stream        the stream to write data to.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_write_siz(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_size_len: OPJ_UINT32 = 0;
    let mut l_current_ptr = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_img_comp = core::ptr::null_mut::<opj_image_comp_t>();
    /* preconditions */

    l_image = p_j2k.m_private_image;
    cp = &mut p_j2k.m_cp;
    l_size_len = (40u32).wrapping_add((3u32).wrapping_mul((*l_image).numcomps));
    l_img_comp = (*l_image).comps;
    if l_size_len > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_size_len as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory for the SIZ marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_size_len
    }
    l_current_ptr = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    /* write SOC identifier */
    opj_write_bytes(l_current_ptr, J2KMarker::SIZ.as_u32(), 2 as OPJ_UINT32); /* SIZ */
    l_current_ptr = l_current_ptr.offset(2); /* L_SIZ */
    opj_write_bytes(
      l_current_ptr,
      l_size_len.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    ); /* Rsiz (capabilities) */
    l_current_ptr = l_current_ptr.offset(2); /* Xsiz */
    opj_write_bytes(l_current_ptr, (*cp).rsiz as OPJ_UINT32, 2 as OPJ_UINT32); /* Ysiz */
    l_current_ptr = l_current_ptr.offset(2); /* X0siz */
    opj_write_bytes(l_current_ptr, (*l_image).x1, 4 as OPJ_UINT32); /* Y0siz */
    l_current_ptr = l_current_ptr.offset(4); /* XTsiz */
    opj_write_bytes(l_current_ptr, (*l_image).y1, 4 as OPJ_UINT32); /* YTsiz */
    l_current_ptr = l_current_ptr.offset(4); /* XT0siz */
    opj_write_bytes(l_current_ptr, (*l_image).x0, 4 as OPJ_UINT32); /* YT0siz */
    l_current_ptr = l_current_ptr.offset(4); /* Csiz */
    opj_write_bytes(l_current_ptr, (*l_image).y0, 4 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(4);
    opj_write_bytes(l_current_ptr, (*cp).tdx, 4 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(4);
    opj_write_bytes(l_current_ptr, (*cp).tdy, 4 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(4);
    opj_write_bytes(l_current_ptr, (*cp).tx0, 4 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(4);
    opj_write_bytes(l_current_ptr, (*cp).ty0, 4 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(4);
    opj_write_bytes(l_current_ptr, (*l_image).numcomps, 2 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(2);
    i = 0 as OPJ_UINT32;
    while i < (*l_image).numcomps {
      /* TODO here with MCT ? */
      opj_write_bytes(
        l_current_ptr,
        (*l_img_comp)
          .prec
          .wrapping_sub(1u32)
          .wrapping_add((*l_img_comp).sgnd << 7i32),
        1 as OPJ_UINT32,
      ); /* Ssiz_i */
      l_current_ptr = l_current_ptr.offset(1); /* XRsiz_i */
      opj_write_bytes(l_current_ptr, (*l_img_comp).dx, 1 as OPJ_UINT32); /* YRsiz_i */
      l_current_ptr = l_current_ptr.offset(1);
      opj_write_bytes(l_current_ptr, (*l_img_comp).dy, 1 as OPJ_UINT32);
      l_current_ptr = l_current_ptr.offset(1);
      l_img_comp = l_img_comp.offset(1);
      i += 1;
    }
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_size_len as OPJ_SIZE_T,
      p_manager,
    ) != l_size_len as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Reads a SIZ marker (image and tile size)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_header_data   the data contained in the SIZ box.
 * @param       p_header_size   the size of the data contained in the SIZ marker.
 * @param       p_manager       the user event manager.
*/
/* *
 * Reads a SIZ marker (image and tile size)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_header_data   the data contained in the SIZ box.
 * @param       p_header_size   the size of the data contained in the SIZ marker.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_read_siz(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_nb_comp: OPJ_UINT32 = 0;
    let mut l_nb_comp_remain: OPJ_UINT32 = 0;
    let mut l_remaining_size: OPJ_UINT32 = 0;
    let mut l_nb_tiles: OPJ_UINT32 = 0;
    let mut l_tmp: OPJ_UINT32 = 0;
    let mut l_tx1: OPJ_UINT32 = 0;
    let mut l_ty1: OPJ_UINT32 = 0;
    let mut l_prec0: OPJ_UINT32 = 0;
    let mut l_sgnd0: OPJ_UINT32 = 0;
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_img_comp = core::ptr::null_mut::<opj_image_comp_t>();
    let mut l_current_tile_param = core::ptr::null_mut::<opj_tcp_t>();
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_image = p_j2k.m_private_image;
    l_cp = &mut p_j2k.m_cp;
    /* minimum size == 39 - 3 (= minimum component parameter) */
    if p_header_size < 36u32 {
      event_msg!(p_manager, EVT_ERROR, "Error with SIZ marker size\n",); /* Rsiz (capabilities) */
      return 0i32;
    } /* Xsiz */
    l_remaining_size = p_header_size.wrapping_sub(36u32); /* Ysiz */
    l_nb_comp = l_remaining_size.wrapping_div(3u32); /* X0siz */
    l_nb_comp_remain = l_remaining_size.wrapping_rem(3u32); /* Y0siz */
    if l_nb_comp_remain != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error with SIZ marker size\n",); /* XTsiz */
      return 0i32;
    } /* YTsiz */
    opj_read_bytes(p_header_data, &mut l_tmp, 2 as OPJ_UINT32); /* XT0siz */
    p_header_data = p_header_data.offset(2); /* YT0siz */
    (*l_cp).rsiz = l_tmp as OPJ_UINT16; /* Csiz */
    opj_read_bytes(
      p_header_data,
      &mut (*l_image).x1 as *mut OPJ_UINT32,
      4 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(4);
    opj_read_bytes(
      p_header_data,
      &mut (*l_image).y1 as *mut OPJ_UINT32,
      4 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(4);
    opj_read_bytes(
      p_header_data,
      &mut (*l_image).x0 as *mut OPJ_UINT32,
      4 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(4);
    opj_read_bytes(
      p_header_data,
      &mut (*l_image).y0 as *mut OPJ_UINT32,
      4 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(4);
    opj_read_bytes(
      p_header_data,
      &mut (*l_cp).tdx as *mut OPJ_UINT32,
      4 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(4);
    opj_read_bytes(
      p_header_data,
      &mut (*l_cp).tdy as *mut OPJ_UINT32,
      4 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(4);
    opj_read_bytes(
      p_header_data,
      &mut (*l_cp).tx0 as *mut OPJ_UINT32,
      4 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(4);
    opj_read_bytes(
      p_header_data,
      &mut (*l_cp).ty0 as *mut OPJ_UINT32,
      4 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(4);
    opj_read_bytes(
      p_header_data,
      &mut l_tmp as *mut OPJ_UINT32,
      2 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(2);
    if l_tmp < 16385u32 {
      (*l_image).numcomps = l_tmp as OPJ_UINT16 as OPJ_UINT32
    } else {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Error with SIZ marker: number of component is illegal -> %d\n",
        l_tmp,
      );
      return 0i32;
    }
    if (*l_image).numcomps != l_nb_comp {
      event_msg!(p_manager, EVT_ERROR,
                      "Error with SIZ marker: number of component is not compatible with the remaining number of parameters ( %d vs %d)\n",
                      (*l_image).numcomps, l_nb_comp);
      return 0i32;
    }
    /* testcase 4035.pdf.SIGSEGV.d8b.3375 */
    /* testcase issue427-null-image-size.jp2 */
    if (*l_image).x0 >= (*l_image).x1 || (*l_image).y0 >= (*l_image).y1 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Error with SIZ marker: negative or zero image size (%ld x %ld)\n",
        (*l_image).x1 as OPJ_INT64 - (*l_image).x0 as i64,
        (*l_image).y1 as OPJ_INT64 - (*l_image).y0 as i64,
      );
      return 0i32;
    }
    /* testcase 2539.pdf.SIGFPE.706.1712 (also 3622.pdf.SIGFPE.706.2916 and 4008.pdf.SIGFPE.706.3345 and maybe more) */
    if (*l_cp).tdx == 0u32 || (*l_cp).tdy == 0u32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Error with SIZ marker: invalid tile size (tdx: %d, tdy: %d)\n",
        (*l_cp).tdx,
        (*l_cp).tdy,
      );
      return 0i32;
    }
    /* testcase issue427-illegal-tile-offset.jp2 */
    l_tx1 = opj_uint_adds((*l_cp).tx0, (*l_cp).tdx); /* manage overflow */
    l_ty1 = opj_uint_adds((*l_cp).ty0, (*l_cp).tdy); /* manage overflow */
    if (*l_cp).tx0 > (*l_image).x0
      || (*l_cp).ty0 > (*l_image).y0
      || l_tx1 <= (*l_image).x0
      || l_ty1 <= (*l_image).y0
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Error with SIZ marker: illegal tile offset\n",
      );
      return 0i32;
    }
    if p_j2k.dump_state == 0 {
      let mut siz_w: OPJ_UINT32 = 0;
      let mut siz_h: OPJ_UINT32 = 0;
      siz_w = (*l_image).x1.wrapping_sub((*l_image).x0);
      siz_h = (*l_image).y1.wrapping_sub((*l_image).y0);
      if p_j2k.ihdr_w > 0u32
        && p_j2k.ihdr_h > 0u32
        && (p_j2k.ihdr_w != siz_w || p_j2k.ihdr_h != siz_h)
      {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Error with SIZ marker: IHDR w(%u) h(%u) vs. SIZ w(%u) h(%u)\n",
          p_j2k.ihdr_w,
          p_j2k.ihdr_h,
          siz_w,
          siz_h,
        );
        return 0i32;
      }
    }
    /* USE_JPWL */
    /* Allocate the resulting image components */
    (*l_image).comps = opj_calloc(
      (*l_image).numcomps as size_t,
      core::mem::size_of::<opj_image_comp_t>(),
    ) as *mut opj_image_comp_t;
    if (*l_image).comps.is_null() {
      (*l_image).numcomps = 0 as OPJ_UINT32;
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to take in charge SIZ marker\n",
      );
      return 0i32;
    }
    l_img_comp = (*l_image).comps;
    l_prec0 = 0 as OPJ_UINT32;
    l_sgnd0 = 0 as OPJ_UINT32;
    /* Read the component information */
    i = 0 as OPJ_UINT32; /* Ssiz_i */
    while i < (*l_image).numcomps {
      let mut tmp: OPJ_UINT32 = 0;
      opj_read_bytes(p_header_data, &mut tmp, 1 as OPJ_UINT32);
      p_header_data = p_header_data.offset(1);
      (*l_img_comp).prec = (tmp & 0x7fu32).wrapping_add(1u32);
      (*l_img_comp).sgnd = tmp >> 7i32;
      if p_j2k.dump_state == 0u32 {
        if i == 0u32 {
          l_prec0 = (*l_img_comp).prec;
          l_sgnd0 = (*l_img_comp).sgnd
        } else if !(*l_cp).allow_different_bit_depth_sign
          && ((*l_img_comp).prec != l_prec0 || (*l_img_comp).sgnd != l_sgnd0)
        {
          event_msg!(p_manager, EVT_WARNING,
                              "Despite JP2 BPC!=255, precision and/or sgnd values for comp[%d] is different than comp[0]:\n        [0] prec(%d) sgnd(%d) [%d] prec(%d) sgnd(%d)\n", i,
                              l_prec0, l_sgnd0, i, (*l_img_comp).prec,
                              (*l_img_comp).sgnd);
        }
        /* TODO: we should perhaps also check against JP2 BPCC values */
      } /* XRsiz_i */
      opj_read_bytes(p_header_data, &mut tmp, 1 as OPJ_UINT32); /* should be between 1 and 255 */
      p_header_data = p_header_data.offset(1); /* YRsiz_i */
      (*l_img_comp).dx = tmp; /* should be between 1 and 255 */
      opj_read_bytes(p_header_data, &mut tmp, 1 as OPJ_UINT32);
      p_header_data = p_header_data.offset(1);
      (*l_img_comp).dy = tmp;
      if (*l_img_comp).dx < 1u32
        || (*l_img_comp).dx > 255u32
        || (*l_img_comp).dy < 1u32
        || (*l_img_comp).dy > 255u32
      {
        event_msg!(p_manager, EVT_ERROR,
                          "Invalid values for comp = %d : dx=%u dy=%u (should be between 1 and 255 according to the JPEG2000 norm)\n", i,
                          (*l_img_comp).dx, (*l_img_comp).dy);
        return 0i32;
      }
      /* Avoids later undefined shift in computation of */
      /* p_j2k->m_specific_param.m_decoder.m_default_tcp->tccps[i].m_dc_level_shift = 1
      << (l_image->comps[i].prec - 1); */
      if (*l_img_comp).prec > 31u32 {
        event_msg!(p_manager, EVT_ERROR,
                          "Invalid values for comp = %d : prec=%u (should be between 1 and 38 according to the JPEG2000 norm. OpenJpeg only supports up to 31)\n", i,
                          (*l_img_comp).prec);
        return 0i32;
      }
      /* USE_JPWL */
      (*l_img_comp).resno_decoded = 0 as OPJ_UINT32; /* number of resolution decoded */
      (*l_img_comp).factor = (*l_cp).m_specific_param.m_dec.m_reduce; /* reducing factor per component */
      l_img_comp = l_img_comp.offset(1);
      i += 1;
    }
    if (*l_cp).tdx == 0u32 || (*l_cp).tdy == 0u32 {
      return 0i32;
    }

    /* Compute the number of tiles */
    (*l_cp).tw = opj_uint_ceildiv((*l_image).x1 - ((*l_cp).tx0), (*l_cp).tdx);
    (*l_cp).th = opj_uint_ceildiv((*l_image).y1 - ((*l_cp).ty0), (*l_cp).tdy);
    /* Check that the number of tiles is valid */
    if (*l_cp).tw == 0u32 || (*l_cp).th == 0u32 || (*l_cp).tw > (65535u32).wrapping_div((*l_cp).th)
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid number of tiles : %u x %u (maximum fixed by jpeg2000 norm is 65535 tiles)\n",
        (*l_cp).tw,
        (*l_cp).th,
      );
      return 0i32;
    }
    l_nb_tiles = (*l_cp).tw.wrapping_mul((*l_cp).th);
    /* Define the tiles which will be decoded */
    if p_j2k.m_specific_param.m_decoder.m_discard_tiles {
      p_j2k.m_specific_param.m_decoder.m_start_tile_x = p_j2k
        .m_specific_param
        .m_decoder
        .m_start_tile_x
        .wrapping_sub((*l_cp).tx0)
        .wrapping_div((*l_cp).tdx);
      p_j2k.m_specific_param.m_decoder.m_start_tile_y = p_j2k
        .m_specific_param
        .m_decoder
        .m_start_tile_y
        .wrapping_sub((*l_cp).ty0)
        .wrapping_div((*l_cp).tdy);
      p_j2k.m_specific_param.m_decoder.m_end_tile_x = opj_uint_ceildiv(
        p_j2k.m_specific_param.m_decoder.m_end_tile_x - ((*l_cp).tx0),
        (*l_cp).tdx,
      );
      p_j2k.m_specific_param.m_decoder.m_end_tile_y = opj_uint_ceildiv(
        p_j2k.m_specific_param.m_decoder.m_end_tile_y - ((*l_cp).ty0),
        (*l_cp).tdy,
      );
    } else {
      p_j2k.m_specific_param.m_decoder.m_start_tile_x = 0 as OPJ_UINT32;
      p_j2k.m_specific_param.m_decoder.m_start_tile_y = 0 as OPJ_UINT32;
      p_j2k.m_specific_param.m_decoder.m_end_tile_x = (*l_cp).tw;
      p_j2k.m_specific_param.m_decoder.m_end_tile_y = (*l_cp).th
    }
    /* USE_JPWL */
    /* memory allocations */
    (*l_cp).tcps =
      opj_calloc(l_nb_tiles as size_t, core::mem::size_of::<opj_tcp_t>()) as *mut opj_tcp_t;
    if (*l_cp).tcps.is_null() {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to take in charge SIZ marker\n",
      );
      return 0i32;
    }
    /* USE_JPWL */
    (*p_j2k.m_specific_param.m_decoder.m_default_tcp).tccps = opj_calloc(
      (*l_image).numcomps as size_t,
      core::mem::size_of::<opj_tccp_t>(),
    ) as *mut opj_tccp_t;
    if (*p_j2k.m_specific_param.m_decoder.m_default_tcp)
      .tccps
      .is_null()
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to take in charge SIZ marker\n",
      );
      return 0i32;
    }
    (*p_j2k.m_specific_param.m_decoder.m_default_tcp).m_mct_records =
      opj_calloc(10i32 as size_t, core::mem::size_of::<opj_mct_data_t>()) as *mut opj_mct_data_t;
    if (*p_j2k.m_specific_param.m_decoder.m_default_tcp)
      .m_mct_records
      .is_null()
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to take in charge SIZ marker\n",
      );
      return 0i32;
    }
    (*p_j2k.m_specific_param.m_decoder.m_default_tcp).m_nb_max_mct_records = 10 as OPJ_UINT32;
    (*p_j2k.m_specific_param.m_decoder.m_default_tcp).m_mcc_records = opj_calloc(
      10i32 as size_t,
      core::mem::size_of::<opj_simple_mcc_decorrelation_data_t>(),
    )
      as *mut opj_simple_mcc_decorrelation_data_t;
    if (*p_j2k.m_specific_param.m_decoder.m_default_tcp)
      .m_mcc_records
      .is_null()
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to take in charge SIZ marker\n",
      );
      return 0i32;
    }
    (*p_j2k.m_specific_param.m_decoder.m_default_tcp).m_nb_max_mcc_records = 10 as OPJ_UINT32;
    /* set up default dc level shift */
    i = 0 as OPJ_UINT32;
    while i < (*l_image).numcomps {
      if (*(*l_image).comps.offset(i as isize)).sgnd == 0 {
        (*(*p_j2k.m_specific_param.m_decoder.m_default_tcp)
          .tccps
          .offset(i as isize))
        .m_dc_level_shift = (1i32)
          << (*(*l_image).comps.offset(i as isize))
            .prec
            .wrapping_sub(1u32)
      }
      i += 1;
    }
    l_current_tile_param = (*l_cp).tcps;
    i = 0 as OPJ_UINT32;
    while i < l_nb_tiles {
      (*l_current_tile_param).tccps = opj_calloc(
        (*l_image).numcomps as size_t,
        core::mem::size_of::<opj_tccp_t>(),
      ) as *mut opj_tccp_t;
      if (*l_current_tile_param).tccps.is_null() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to take in charge SIZ marker\n",
        );
        return 0i32;
      }
      l_current_tile_param = l_current_tile_param.offset(1);
      i += 1;
    }
    p_j2k.m_specific_param.m_decoder.m_state = J2KState::MH;
    opj_image_comp_header_update(l_image, l_cp);
    1i32
  }
}

/* *
 * Writes the COM marker (comment)
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_write_com(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_comment_size: OPJ_UINT32 = 0;
    let mut l_total_com_size: OPJ_UINT32 = 0;
    let mut l_comment = core::ptr::null::<OPJ_CHAR>();
    let mut l_current_ptr = core::ptr::null_mut::<OPJ_BYTE>();
    /* preconditions */
    /* L_COM */

    l_comment = p_j2k.m_cp.comment;
    l_comment_size = strlen(l_comment) as OPJ_UINT32;
    l_total_com_size = l_comment_size.wrapping_add(6u32);
    if l_total_com_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_total_com_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write the COM marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_total_com_size
    }
    l_current_ptr = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    opj_write_bytes(l_current_ptr, J2KMarker::COM.as_u32(), 2 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(2);
    opj_write_bytes(
      l_current_ptr,
      l_total_com_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_ptr = l_current_ptr.offset(2);
    opj_write_bytes(l_current_ptr, 1 as OPJ_UINT32, 2 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(2);
    memcpy(
      l_current_ptr as *mut core::ffi::c_void,
      l_comment as *const core::ffi::c_void,
      l_comment_size as usize,
    );
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_total_com_size as OPJ_SIZE_T,
      p_manager,
    ) != l_total_com_size as usize
    {
      return 0i32;
    }
    1i32
  }
}

/* *
 * Reads a COM marker (comments)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_header_data   the data contained in the COM box.
 * @param       p_header_size   the size of the data contained in the COM marker.
 * @param       p_manager       the user event manager.
*/
/* *
 * Reads a COM marker (comments)
 * @param       p_j2k           the jpeg2000 file codec.
 * @param       p_header_data   the data contained in the COM box.
 * @param       p_header_size   the size of the data contained in the COM marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_com(
  mut _p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut _p_header_size: OPJ_UINT32,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* preconditions */

  assert!(!p_header_data.is_null());
  1i32
}

/* *
 * Writes the COD marker (Coding style default)
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_write_cod(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_code_size: OPJ_UINT32 = 0;
    let mut l_remaining_size: OPJ_UINT32 = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    /* preconditions */
    /* L_COD */
    /* SGcod (A) */
    /* SGcod (C) */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t;
    l_code_size = (9u32).wrapping_add(opj_j2k_get_SPCod_SPCoc_size(
      p_j2k,
      p_j2k.m_current_tile_number,
      0 as OPJ_UINT32,
    ));
    l_remaining_size = l_code_size;
    if l_code_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_code_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write COD marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_code_size
    }
    l_current_data = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    opj_write_bytes(l_current_data, J2KMarker::COD.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(
      l_current_data,
      l_code_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(l_current_data, (*l_tcp).csty, 1 as OPJ_UINT32);
    l_current_data = l_current_data.offset(1);
    opj_write_bytes(l_current_data, (*l_tcp).prg as OPJ_UINT32, 1 as OPJ_UINT32);
    l_current_data = l_current_data.offset(1);
    opj_write_bytes(l_current_data, (*l_tcp).numlayers, 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(l_current_data, (*l_tcp).mct, 1 as OPJ_UINT32);
    l_current_data = l_current_data.offset(1);
    l_remaining_size =
      (l_remaining_size as core::ffi::c_uint).wrapping_sub(9u32) as OPJ_UINT32 as OPJ_UINT32;
    if opj_j2k_write_SPCod_SPCoc(
      p_j2k,
      p_j2k.m_current_tile_number,
      0 as OPJ_UINT32,
      l_current_data,
      &mut l_remaining_size,
      p_manager,
    ) == 0
    {
      event_msg!(p_manager, EVT_ERROR, "Error writing COD marker\n",);
      return 0i32;
    }
    if l_remaining_size != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error writing COD marker\n",);
      return 0i32;
    }
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_code_size as OPJ_SIZE_T,
      p_manager,
    ) != l_code_size as usize
    {
      return 0i32;
    }
    1i32
  }
}

/* *
 * Reads a COD marker (Coding style defaults)
 * @param       p_header_data   the data contained in the COD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the COD marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_cod(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    /* loop */
    let mut i: OPJ_UINT32 = 0;
    let mut l_tmp: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_image = p_j2k.m_private_image;
    l_cp = &mut p_j2k.m_cp;
    /* If we are in the first tile-part header of the current tile */
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    (*l_tcp).cod = true;
    /* Make sure room is sufficient */
    if p_header_size < 5u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading COD marker\n",); /* Scod */
      return 0i32;
    }
    opj_read_bytes(p_header_data, &mut (*l_tcp).csty, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    /* Make sure we know how to decode this */
    if (*l_tcp).csty & !((0x1i32 | 0x2i32 | 0x4i32) as OPJ_UINT32) != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Unknown Scod value in COD marker\n",); /* SGcod (A) */
      return 0i32;
    }
    opj_read_bytes(p_header_data, &mut l_tmp, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    (*l_tcp).prg = l_tmp as OPJ_PROG_ORDER;
    /* Make sure progression order is valid */
    if (*l_tcp).prg as core::ffi::c_int > OPJ_CPRL as core::ffi::c_int {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Unknown progression order in COD marker\n",
      ); /* SGcod (B) */
      (*l_tcp).prg = OPJ_PROG_UNKNOWN
    }
    opj_read_bytes(p_header_data, &mut (*l_tcp).numlayers, 2 as OPJ_UINT32);
    p_header_data = p_header_data.offset(2);
    if (*l_tcp).numlayers < 1u32 || (*l_tcp).numlayers > 65535u32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid number of layers in COD marker : %d not in range [1-65535]\n",
        (*l_tcp).numlayers,
      );
      return 0i32;
    }
    /* If user didn't set a number layer to decode take the max specify in the codestream. */
    if (*l_cp).m_specific_param.m_dec.m_layer != 0 {
      (*l_tcp).num_layers_to_decode = (*l_cp).m_specific_param.m_dec.m_layer
    } else {
      (*l_tcp).num_layers_to_decode = (*l_tcp).numlayers
    } /* SGcod (C) */
    opj_read_bytes(p_header_data, &mut (*l_tcp).mct, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    if (*l_tcp).mct > 1u32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid multiple component transformation\n",
      );
      return 0i32;
    }
    p_header_size = (p_header_size as core::ffi::c_uint).wrapping_sub(5u32) as OPJ_UINT32;
    i = 0 as OPJ_UINT32;
    while i < (*l_image).numcomps {
      (*(*l_tcp).tccps.offset(i as isize)).csty = (*l_tcp).csty & 0x1u32;
      i += 1;
    }
    if opj_j2k_read_SPCod_SPCoc(
      p_j2k,
      0 as OPJ_UINT32,
      p_header_data,
      &mut p_header_size,
      p_manager,
    ) == 0
    {
      event_msg!(p_manager, EVT_ERROR, "Error reading COD marker\n",);
      return 0i32;
    }
    if p_header_size != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading COD marker\n",);
      return 0i32;
    }
    /* Apply the coding style to other components of the current tile or the m_default_tcp*/
    opj_j2k_copy_tile_component_parameters(p_j2k);
    /* Index */
    1i32
  }
}

/* *
 * Writes the COC marker (Coding style component)
 *
 * @param       p_j2k       J2K codec.
 * @param       p_comp_no   the index of the component to output.
 * @param       p_stream    the stream to write data to.
 * @param       p_manager   the user event manager.
*/
fn opj_j2k_write_coc(
  mut p_j2k: &mut opj_j2k,
  mut p_comp_no: OPJ_UINT32,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_coc_size: OPJ_UINT32 = 0;
    let mut l_remaining_size: OPJ_UINT32 = 0;
    let mut l_comp_room: OPJ_UINT32 = 0;
    /* preconditions */

    l_comp_room = if (*p_j2k.m_private_image).numcomps <= 256u32 {
      1i32
    } else {
      2i32
    } as OPJ_UINT32;
    l_coc_size = (5u32)
      .wrapping_add(l_comp_room)
      .wrapping_add(opj_j2k_get_SPCod_SPCoc_size(
        p_j2k,
        p_j2k.m_current_tile_number,
        p_comp_no,
      ));
    if l_coc_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
      /*p_j2k->m_specific_param.m_encoder.m_header_tile_data
      = (OPJ_BYTE*)opj_realloc(
              p_j2k->m_specific_param.m_encoder.m_header_tile_data,
              l_coc_size);*/
      new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_coc_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write COC marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_coc_size
    }
    opj_j2k_write_coc_in_memory(
      p_j2k,
      p_comp_no,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      &mut l_remaining_size,
      p_manager,
    );
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_coc_size as OPJ_SIZE_T,
      p_manager,
    ) != l_coc_size as usize
    {
      return 0i32;
    }
    1i32
  }
}

/* *
 * Compares 2 COC markers (Coding style component)
 *
 * @param       p_j2k            J2K codec.
 * @param       p_first_comp_no  the index of the first component to compare.
 * @param       p_second_comp_no the index of the second component to compare.
 *
 * @return      OPJ_TRUE if equals
 */
fn opj_j2k_compare_coc(
  mut p_j2k: &mut opj_j2k,
  mut p_first_comp_no: OPJ_UINT32,
  mut p_second_comp_no: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    /* preconditions */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t;
    if (*(*l_tcp).tccps.offset(p_first_comp_no as isize)).csty
      != (*(*l_tcp).tccps.offset(p_second_comp_no as isize)).csty
    {
      return 0i32;
    }
    opj_j2k_compare_SPCod_SPCoc(
      p_j2k,
      p_j2k.m_current_tile_number,
      p_first_comp_no,
      p_second_comp_no,
    )
  }
}

/* *
 * Writes the COC marker (Coding style component)
 *
 * @param       p_j2k                   J2K codec.
 * @param       p_comp_no               the index of the component to output.
 * @param       p_data          FIXME DOC
 * @param       p_data_written  FIXME DOC
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_coc_in_memory(
  mut p_j2k: &mut opj_j2k,
  mut p_comp_no: OPJ_UINT32,
  mut p_data: *mut OPJ_BYTE,
  mut p_data_written: *mut OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_coc_size: OPJ_UINT32 = 0;
    let mut l_remaining_size: OPJ_UINT32 = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut l_comp_room: OPJ_UINT32 = 0;
    /* preconditions */
    /* L_COC */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t;
    l_image = p_j2k.m_private_image;
    l_comp_room = if (*l_image).numcomps <= 256u32 {
      1i32
    } else {
      2i32
    } as OPJ_UINT32;
    l_coc_size = (5u32)
      .wrapping_add(l_comp_room)
      .wrapping_add(opj_j2k_get_SPCod_SPCoc_size(
        p_j2k,
        p_j2k.m_current_tile_number,
        p_comp_no,
      ));
    l_remaining_size = l_coc_size;
    l_current_data = p_data;
    opj_write_bytes(l_current_data, J2KMarker::COC.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(
      l_current_data,
      l_coc_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(l_current_data, p_comp_no, l_comp_room);
    l_current_data = l_current_data.offset(l_comp_room as isize);
    opj_write_bytes(
      l_current_data,
      (*(*l_tcp).tccps.offset(p_comp_no as isize)).csty,
      1 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(1);
    l_remaining_size = (l_remaining_size as core::ffi::c_uint)
      .wrapping_sub((5u32).wrapping_add(l_comp_room)) as OPJ_UINT32;
    opj_j2k_write_SPCod_SPCoc(
      p_j2k,
      p_j2k.m_current_tile_number,
      0 as OPJ_UINT32,
      l_current_data,
      &mut l_remaining_size,
      p_manager,
    );
    *p_data_written = l_coc_size;
  }
}

/* *
 * Gets the maximum size taken by a coc.
 *
 * @param       p_j2k   the jpeg2000 codec to use.
 */
fn opj_j2k_get_max_coc_size(mut p_j2k: &mut opj_j2k) -> OPJ_UINT32 {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut l_nb_comp: OPJ_UINT32 = 0;
    let mut l_nb_tiles: OPJ_UINT32 = 0;
    let mut l_max = 0 as OPJ_UINT32;
    /* preconditions */
    l_nb_tiles = p_j2k.m_cp.tw.wrapping_mul(p_j2k.m_cp.th);
    l_nb_comp = (*p_j2k.m_private_image).numcomps;
    i = 0 as OPJ_UINT32;
    while i < l_nb_tiles {
      j = 0 as OPJ_UINT32;
      while j < l_nb_comp {
        l_max = opj_uint_max(l_max, opj_j2k_get_SPCod_SPCoc_size(p_j2k, i, j));
        j += 1;
      }
      i += 1;
    }
    (6u32).wrapping_add(l_max)
  }
}

/* *
 * Reads a COC marker (Coding Style Component)
 * @param       p_header_data   the data contained in the COC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the COC marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a COC marker (Coding Style Component)
 * @param       p_header_data   the data contained in the COC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the COC marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_coc(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut l_comp_room: OPJ_UINT32 = 0;
    let mut l_comp_no: OPJ_UINT32 = 0;
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_cp = &mut p_j2k.m_cp;
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    l_image = p_j2k.m_private_image;
    l_comp_room = if (*l_image).numcomps <= 256u32 {
      1i32
    } else {
      2i32
    } as OPJ_UINT32;
    /* make sure room is sufficient*/
    if p_header_size < l_comp_room.wrapping_add(1u32) {
      event_msg!(p_manager, EVT_ERROR, "Error reading COC marker\n",); /* Ccoc */
      return 0i32;
    } /* Scoc */
    p_header_size = (p_header_size as core::ffi::c_uint)
      .wrapping_sub(l_comp_room.wrapping_add(1u32)) as OPJ_UINT32;
    opj_read_bytes(p_header_data, &mut l_comp_no, l_comp_room);
    p_header_data = p_header_data.offset(l_comp_room as isize);
    if l_comp_no >= (*l_image).numcomps {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Error reading COC marker (bad number of components)\n",
      );
      return 0i32;
    }
    opj_read_bytes(
      p_header_data,
      &mut (*(*l_tcp).tccps.offset(l_comp_no as isize)).csty,
      1 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(1);
    if opj_j2k_read_SPCod_SPCoc(
      p_j2k,
      l_comp_no,
      p_header_data,
      &mut p_header_size,
      p_manager,
    ) == 0
    {
      event_msg!(p_manager, EVT_ERROR, "Error reading COC marker\n",);
      return 0i32;
    }
    if p_header_size != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading COC marker\n",);
      return 0i32;
    }
    1i32
  }
}
/* *
 * Writes the QCD marker (quantization default)
 *
 * @param       p_j2k                   J2K codec.
 * @param       p_stream                the stream to write data to.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_qcd(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_qcd_size: OPJ_UINT32 = 0;
    let mut l_remaining_size: OPJ_UINT32 = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    /* preconditions */
    /* L_QCD */

    l_qcd_size = (4u32).wrapping_add(opj_j2k_get_SQcd_SQcc_size(
      p_j2k,
      p_j2k.m_current_tile_number,
      0 as OPJ_UINT32,
    ));
    l_remaining_size = l_qcd_size;
    if l_qcd_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_qcd_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write QCD marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_qcd_size
    }
    l_current_data = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    opj_write_bytes(l_current_data, J2KMarker::QCD.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(
      l_current_data,
      l_qcd_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    l_remaining_size =
      (l_remaining_size as core::ffi::c_uint).wrapping_sub(4u32) as OPJ_UINT32 as OPJ_UINT32;
    if opj_j2k_write_SQcd_SQcc(
      p_j2k,
      p_j2k.m_current_tile_number,
      0 as OPJ_UINT32,
      l_current_data,
      &mut l_remaining_size,
      p_manager,
    ) == 0
    {
      event_msg!(p_manager, EVT_ERROR, "Error writing QCD marker\n",);
      return 0i32;
    }
    if l_remaining_size != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error writing QCD marker\n",);
      return 0i32;
    }
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_qcd_size as OPJ_SIZE_T,
      p_manager,
    ) != l_qcd_size as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Reads a QCD marker (Quantization defaults)
 * @param       p_header_data   the data contained in the QCD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the QCD marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a QCD marker (Quantization defaults)
 * @param       p_header_data   the data contained in the QCD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the QCD marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_qcd(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* preconditions */

  assert!(!p_header_data.is_null());
  if opj_j2k_read_SQcd_SQcc(
    p_j2k,
    0 as OPJ_UINT32,
    p_header_data,
    &mut p_header_size,
    p_manager,
  ) == 0
  {
    event_msg!(p_manager, EVT_ERROR, "Error reading QCD marker\n",);
    return 0i32;
  }
  if p_header_size != 0u32 {
    event_msg!(p_manager, EVT_ERROR, "Error reading QCD marker\n",);
    return 0i32;
  }
  /* Apply the quantization parameters to other components of the current tile or the m_default_tcp */
  opj_j2k_copy_tile_quantization_parameters(p_j2k);
  1i32
}

/* *
 * Writes the QCC marker (quantization component)
 *
 * @param       p_comp_no       the index of the component to output.
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_qcc(
  mut p_j2k: &mut opj_j2k,
  mut p_comp_no: OPJ_UINT32,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_qcc_size: OPJ_UINT32 = 0;
    let mut l_remaining_size: OPJ_UINT32 = 0;
    /* preconditions */

    l_qcc_size = (5u32).wrapping_add(opj_j2k_get_SQcd_SQcc_size(
      p_j2k,
      p_j2k.m_current_tile_number,
      p_comp_no,
    ));
    l_qcc_size =
      (l_qcc_size as core::ffi::c_uint).wrapping_add(if (*p_j2k.m_private_image).numcomps <= 256u32
      {
        0i32
      } else {
        1i32
      } as core::ffi::c_uint) as OPJ_UINT32;
    l_remaining_size = l_qcc_size;
    if l_qcc_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_qcc_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write QCC marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_qcc_size
    }
    opj_j2k_write_qcc_in_memory(
      p_j2k,
      p_comp_no,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      &mut l_remaining_size,
      p_manager,
    );
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_qcc_size as OPJ_SIZE_T,
      p_manager,
    ) != l_qcc_size as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Compare QCC markers (quantization component)
 *
 * @param       p_j2k                 J2K codec.
 * @param       p_first_comp_no       the index of the first component to compare.
 * @param       p_second_comp_no      the index of the second component to compare.
 *
 * @return OPJ_TRUE if equals.
 */
fn opj_j2k_compare_qcc(
  mut p_j2k: &mut opj_j2k,
  mut p_first_comp_no: OPJ_UINT32,
  mut p_second_comp_no: OPJ_UINT32,
) -> OPJ_BOOL {
  opj_j2k_compare_SQcd_SQcc(
    p_j2k,
    p_j2k.m_current_tile_number,
    p_first_comp_no,
    p_second_comp_no,
  )
}

/* *
 * Writes the QCC marker (quantization component)
 *
 * @param       p_j2k           J2K codec.
 * @param       p_comp_no       the index of the component to output.
 * @param       p_data          FIXME DOC
 * @param       p_data_written  the stream to write data to.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_write_qcc_in_memory(
  mut p_j2k: &mut opj_j2k,
  mut p_comp_no: OPJ_UINT32,
  mut p_data: *mut OPJ_BYTE,
  mut p_data_written: *mut OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) {
  unsafe {
    let mut l_qcc_size: OPJ_UINT32 = 0;
    let mut l_remaining_size: OPJ_UINT32 = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    /* preconditions */
    /* L_QCC */
    l_qcc_size = (6u32).wrapping_add(opj_j2k_get_SQcd_SQcc_size(
      p_j2k,
      p_j2k.m_current_tile_number,
      p_comp_no,
    ));
    l_remaining_size = l_qcc_size;
    l_current_data = p_data;
    opj_write_bytes(l_current_data, J2KMarker::QCC.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    if (*p_j2k.m_private_image).numcomps <= 256u32 {
      l_qcc_size = l_qcc_size.wrapping_sub(1);
      opj_write_bytes(
        l_current_data,
        l_qcc_size.wrapping_sub(2u32),
        2 as OPJ_UINT32,
      );
      l_current_data = l_current_data.offset(2);
      opj_write_bytes(l_current_data, p_comp_no, 1 as OPJ_UINT32);
      l_current_data = l_current_data.offset(1);
      /* in the case only one byte is sufficient the last byte allocated is useless -> still do -6 for available */
      l_remaining_size =
        (l_remaining_size as core::ffi::c_uint).wrapping_sub(6u32) as OPJ_UINT32 as OPJ_UINT32
    } else {
      opj_write_bytes(
        l_current_data,
        l_qcc_size.wrapping_sub(2u32),
        2 as OPJ_UINT32,
      ); /* L_QCC */
      l_current_data = l_current_data.offset(2); /* Cqcc */
      opj_write_bytes(l_current_data, p_comp_no, 2 as OPJ_UINT32);
      l_current_data = l_current_data.offset(2);
      l_remaining_size =
        (l_remaining_size as core::ffi::c_uint).wrapping_sub(6u32) as OPJ_UINT32 as OPJ_UINT32
    }
    opj_j2k_write_SQcd_SQcc(
      p_j2k,
      p_j2k.m_current_tile_number,
      p_comp_no,
      l_current_data,
      &mut l_remaining_size,
      p_manager,
    );
    *p_data_written = l_qcc_size;
  }
}

/* *
 * Gets the maximum size taken by a qcc.
 */
fn opj_j2k_get_max_qcc_size(mut p_j2k: &mut opj_j2k) -> OPJ_UINT32 {
  opj_j2k_get_max_coc_size(p_j2k)
}

/* *
 * Reads a QCC marker (Quantization component)
 * @param       p_header_data   the data contained in the QCC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the QCC marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a QCC marker (Quantization component)
 * @param       p_header_data   the data contained in the QCC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the QCC marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_qcc(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_num_comp: OPJ_UINT32 = 0;
    let mut l_comp_no: OPJ_UINT32 = 0;
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_num_comp = (*p_j2k.m_private_image).numcomps;
    if l_num_comp <= 256u32 {
      if p_header_size < 1u32 {
        event_msg!(p_manager, EVT_ERROR, "Error reading QCC marker\n",);
        return 0i32;
      }
      opj_read_bytes(p_header_data, &mut l_comp_no, 1 as OPJ_UINT32);
      p_header_data = p_header_data.offset(1);
      p_header_size = p_header_size.wrapping_sub(1)
    } else {
      if p_header_size < 2u32 {
        event_msg!(p_manager, EVT_ERROR, "Error reading QCC marker\n",);
        return 0i32;
      }
      opj_read_bytes(p_header_data, &mut l_comp_no, 2 as OPJ_UINT32);
      p_header_data = p_header_data.offset(2);
      p_header_size = (p_header_size as core::ffi::c_uint).wrapping_sub(2u32) as OPJ_UINT32
    }
    /* USE_JPWL */
    if l_comp_no >= (*p_j2k.m_private_image).numcomps {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid component number: %d, regarding the number of components %d\n",
        l_comp_no,
        (*p_j2k.m_private_image).numcomps,
      );
      return 0i32;
    }
    if opj_j2k_read_SQcd_SQcc(
      p_j2k,
      l_comp_no,
      p_header_data,
      &mut p_header_size,
      p_manager,
    ) == 0
    {
      event_msg!(p_manager, EVT_ERROR, "Error reading QCC marker\n",);
      return 0i32;
    }
    if p_header_size != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading QCC marker\n",);
      return 0i32;
    }
    1i32
  }
}
/* *
 * Writes the POC marker (Progression Order Change)
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_poc(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_nb_comp: OPJ_UINT32 = 0;
    let mut l_nb_poc: OPJ_UINT32 = 0;
    let mut l_poc_size: OPJ_UINT32 = 0;
    let mut l_written_size = 0 as OPJ_UINT32;
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_poc_room: OPJ_UINT32 = 0;
    /* preconditions */

    l_tcp = &mut *p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t;
    l_nb_comp = (*p_j2k.m_private_image).numcomps;
    l_nb_poc = (1u32).wrapping_add((*l_tcp).numpocs);
    if l_nb_comp <= 256u32 {
      l_poc_room = 1 as OPJ_UINT32
    } else {
      l_poc_room = 2 as OPJ_UINT32
    }
    l_poc_size = (4u32).wrapping_add(
      (5u32)
        .wrapping_add((2u32).wrapping_mul(l_poc_room))
        .wrapping_mul(l_nb_poc),
    );
    if l_poc_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_poc_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write POC marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_poc_size
    }
    opj_j2k_write_poc_in_memory(
      p_j2k,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      &mut l_written_size,
      p_manager,
    );
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_poc_size as OPJ_SIZE_T,
      p_manager,
    ) != l_poc_size as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Writes the POC marker (Progression Order Change)
 *
 * @param       p_j2k          J2K codec.
 * @param       p_data         FIXME DOC
 * @param       p_data_written the stream to write data to.
 * @param       p_manager      the user event manager.
 */
fn opj_j2k_write_poc_in_memory(
  mut p_j2k: &mut opj_j2k,
  mut p_data: *mut OPJ_BYTE,
  mut p_data_written: *mut OPJ_UINT32,
  mut _p_manager: &mut opj_event_mgr,
) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_nb_comp: OPJ_UINT32 = 0;
    let mut l_nb_poc: OPJ_UINT32 = 0;
    let mut l_poc_size: OPJ_UINT32 = 0;
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_current_poc = core::ptr::null_mut::<opj_poc_t>();
    let mut l_poc_room: OPJ_UINT32 = 0;
    /* preconditions */
    /* Lpoc */
    l_tcp = &mut *p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t; /* LYEpoc_i */
    l_tccp = &mut *(*l_tcp).tccps.offset(0) as *mut opj_tccp_t; /* REpoc_i */
    l_image = p_j2k.m_private_image; /* CEpoc_i */
    l_nb_comp = (*l_image).numcomps; /* Ppoc_i */
    l_nb_poc = (1u32).wrapping_add((*l_tcp).numpocs);
    if l_nb_comp <= 256u32 {
      l_poc_room = 1 as OPJ_UINT32
    } else {
      l_poc_room = 2 as OPJ_UINT32
    }
    l_poc_size = (4u32).wrapping_add(
      (5u32)
        .wrapping_add((2u32).wrapping_mul(l_poc_room))
        .wrapping_mul(l_nb_poc),
    );
    l_current_data = p_data;
    opj_write_bytes(l_current_data, J2KMarker::POC.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(
      l_current_data,
      l_poc_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    l_current_poc = (*l_tcp).pocs.as_mut_ptr();
    i = 0 as OPJ_UINT32;
    while i < l_nb_poc {
      opj_write_bytes(l_current_data, (*l_current_poc).resno0, 1 as OPJ_UINT32);
      l_current_data = l_current_data.offset(1);
      opj_write_bytes(l_current_data, (*l_current_poc).compno0, l_poc_room);
      l_current_data = l_current_data.offset(l_poc_room as isize);
      opj_write_bytes(l_current_data, (*l_current_poc).layno1, 2 as OPJ_UINT32);
      l_current_data = l_current_data.offset(2);
      opj_write_bytes(l_current_data, (*l_current_poc).resno1, 1 as OPJ_UINT32);
      l_current_data = l_current_data.offset(1);
      opj_write_bytes(l_current_data, (*l_current_poc).compno1, l_poc_room);
      l_current_data = l_current_data.offset(l_poc_room as isize);
      opj_write_bytes(
        l_current_data,
        (*l_current_poc).prg as OPJ_UINT32,
        1 as OPJ_UINT32,
      );
      l_current_data = l_current_data.offset(1);
      /* change the value of the max layer according to the actual number of layers in the file, components and resolutions*/
      (*l_current_poc).layno1 = opj_int_min(
        (*l_current_poc).layno1 as OPJ_INT32,
        (*l_tcp).numlayers as OPJ_INT32,
      ) as OPJ_UINT32;
      (*l_current_poc).resno1 = opj_int_min(
        (*l_current_poc).resno1 as OPJ_INT32,
        (*l_tccp).numresolutions as OPJ_INT32,
      ) as OPJ_UINT32;
      (*l_current_poc).compno1 = opj_int_min(
        (*l_current_poc).compno1 as OPJ_INT32,
        l_nb_comp as OPJ_INT32,
      ) as OPJ_UINT32;
      l_current_poc = l_current_poc.offset(1);
      i += 1;
    }
    *p_data_written = l_poc_size;
  }
}
/* *
 * Gets the maximum size taken by the writing of a POC.
 */
fn opj_j2k_get_max_poc_size(mut p_j2k: &mut opj_j2k) -> OPJ_UINT32 {
  unsafe {
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_nb_tiles = 0 as OPJ_UINT32;
    let mut l_max_poc = 0 as OPJ_UINT32;
    let mut i: OPJ_UINT32 = 0;
    l_tcp = p_j2k.m_cp.tcps;
    l_nb_tiles = p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw);
    i = 0 as OPJ_UINT32;
    while i < l_nb_tiles {
      l_max_poc = opj_uint_max(l_max_poc, (*l_tcp).numpocs);
      l_tcp = l_tcp.offset(1);
      i += 1;
    }
    l_max_poc = l_max_poc.wrapping_add(1);
    (4u32).wrapping_add((9u32).wrapping_mul(l_max_poc))
  }
}
/* *
 * Gets the maximum size taken by the toc headers of all the tile parts of any given tile.
 */
fn opj_j2k_get_max_toc_size(mut p_j2k: &mut opj_j2k) -> OPJ_UINT32 {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_nb_tiles: OPJ_UINT32 = 0;
    let mut l_max = 0 as OPJ_UINT32;
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    l_tcp = p_j2k.m_cp.tcps;
    l_nb_tiles = p_j2k.m_cp.tw.wrapping_mul(p_j2k.m_cp.th);
    i = 0 as OPJ_UINT32;
    while i < l_nb_tiles {
      l_max = opj_uint_max(l_max, (*l_tcp).m_nb_tile_parts);
      l_tcp = l_tcp.offset(1);
      i += 1;
    }
    (12u32).wrapping_mul(l_max)
  }
}
/* *
 * Gets the maximum size taken by the headers of the SOT.
 *
 * @param       p_j2k   the jpeg2000 codec to use.
 */
fn opj_j2k_get_specific_header_sizes(mut p_j2k: &mut opj_j2k) -> OPJ_UINT32 {
  unsafe {
    let mut l_nb_bytes = 0 as OPJ_UINT32;
    let mut l_nb_comps: OPJ_UINT32 = 0;
    let mut l_coc_bytes: OPJ_UINT32 = 0;
    let mut l_qcc_bytes: OPJ_UINT32 = 0;
    l_nb_comps = (*p_j2k.m_private_image).numcomps.wrapping_sub(1u32);
    l_nb_bytes =
      (l_nb_bytes as core::ffi::c_uint).wrapping_add(opj_j2k_get_max_toc_size(p_j2k)) as OPJ_UINT32;
    if !(p_j2k.m_cp.rsiz as core::ffi::c_int >= 0x3i32
      && p_j2k.m_cp.rsiz as core::ffi::c_int <= 0x6i32)
    {
      l_coc_bytes = opj_j2k_get_max_coc_size(p_j2k);
      l_nb_bytes = (l_nb_bytes as core::ffi::c_uint)
        .wrapping_add(l_nb_comps.wrapping_mul(l_coc_bytes)) as OPJ_UINT32;
      l_qcc_bytes = opj_j2k_get_max_qcc_size(p_j2k);
      l_nb_bytes = (l_nb_bytes as core::ffi::c_uint)
        .wrapping_add(l_nb_comps.wrapping_mul(l_qcc_bytes)) as OPJ_UINT32
    }
    l_nb_bytes =
      (l_nb_bytes as core::ffi::c_uint).wrapping_add(opj_j2k_get_max_poc_size(p_j2k)) as OPJ_UINT32;
    if p_j2k.m_specific_param.m_encoder.m_PLT != 0 {
      /* Reserve space for PLT markers */
      let mut i: OPJ_UINT32 = 0;
      let mut l_cp: *const opj_cp_t = &mut p_j2k.m_cp;
      let mut l_max_packet_count = 0 as OPJ_UINT32;
      i = 0 as OPJ_UINT32;
      while i < (*l_cp).th.wrapping_mul((*l_cp).tw) {
        l_max_packet_count = opj_uint_max(
          l_max_packet_count,
          opj_get_encoding_packet_count(p_j2k.m_private_image, l_cp, i),
        );
        i += 1;
      }
      /* Minimum 6 bytes per PLT marker, and at a minimum (taking a pessimistic */
      /* estimate of 4 bytes for a packet size), one can write */
      /* (65536-6) / 4 = 16382 paquet sizes per PLT marker */
      p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT =
        (6u32).wrapping_mul(opj_uint_ceildiv(l_max_packet_count, 16382 as OPJ_UINT32));
      /* Maximum 5 bytes per packet to encode a full UINT32 */
      l_nb_bytes = (l_nb_bytes as core::ffi::c_uint)
        .wrapping_add((5u32).wrapping_mul(l_max_packet_count)) as OPJ_UINT32;
      p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT =
        (p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT as core::ffi::c_uint)
          .wrapping_add(l_nb_bytes) as OPJ_UINT32;
      p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT =
        (p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT as core::ffi::c_uint)
          .wrapping_add(1u32) as OPJ_UINT32;
      l_nb_bytes = (l_nb_bytes as core::ffi::c_uint)
        .wrapping_add(p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT)
        as OPJ_UINT32
    }
    /* ** DEVELOPER CORNER, Add room for your headers ***/
    l_nb_bytes
  }
}
/* *
 * Reads a POC marker (Progression Order Change)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a POC marker (Progression Order Change)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_poc(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_nb_comp: OPJ_UINT32 = 0;
    let mut l_tmp: OPJ_UINT32 = 0;
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut l_old_poc_nb: OPJ_UINT32 = 0;
    let mut l_current_poc_nb: OPJ_UINT32 = 0;
    let mut l_current_poc_remaining: OPJ_UINT32 = 0;
    let mut l_chunk_size: OPJ_UINT32 = 0;
    let mut l_comp_room: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_current_poc = core::ptr::null_mut::<opj_poc_t>();
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_image = p_j2k.m_private_image;
    l_nb_comp = (*l_image).numcomps;
    if l_nb_comp <= 256u32 {
      l_comp_room = 1 as OPJ_UINT32
    } else {
      l_comp_room = 2 as OPJ_UINT32
    }
    l_chunk_size = (5u32).wrapping_add((2u32).wrapping_mul(l_comp_room));
    l_current_poc_nb = p_header_size.wrapping_div(l_chunk_size);
    l_current_poc_remaining = p_header_size.wrapping_rem(l_chunk_size);
    if l_current_poc_nb <= 0u32 || l_current_poc_remaining != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading POC marker\n",);
      return 0i32;
    }
    l_cp = &mut p_j2k.m_cp;
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    l_old_poc_nb = if (*l_tcp).POC {
      (*l_tcp).numpocs.wrapping_add(1u32)
    } else {
      0u32
    };
    l_current_poc_nb =
      (l_current_poc_nb as core::ffi::c_uint).wrapping_add(l_old_poc_nb) as OPJ_UINT32;
    if l_current_poc_nb >= 32u32 {
      event_msg!(p_manager, EVT_ERROR, "Too many POCs %d\n", l_current_poc_nb,);
      return 0i32;
    }
    /* now poc is in use.*/
    (*l_tcp).POC = true; /* RSpoc_i */
    l_current_poc =
      &mut *(*l_tcp).pocs.as_mut_ptr().offset(l_old_poc_nb as isize) as *mut opj_poc_t; /* CSpoc_i */
    i = l_old_poc_nb; /* LYEpoc_i */
    while i < l_current_poc_nb {
      opj_read_bytes(p_header_data, &mut (*l_current_poc).resno0, 1 as OPJ_UINT32);
      p_header_data = p_header_data.offset(1);
      opj_read_bytes(p_header_data, &mut (*l_current_poc).compno0, l_comp_room);
      p_header_data = p_header_data.offset(l_comp_room as isize);
      opj_read_bytes(p_header_data, &mut (*l_current_poc).layno1, 2 as OPJ_UINT32);
      /* make sure layer end is in acceptable bounds */
      (*l_current_poc).layno1 = opj_uint_min((*l_current_poc).layno1, (*l_tcp).numlayers); /* REpoc_i */
      p_header_data = p_header_data.offset(2); /* CEpoc_i */
      opj_read_bytes(p_header_data, &mut (*l_current_poc).resno1, 1 as OPJ_UINT32); /* Ppoc_i */
      p_header_data = p_header_data.offset(1);
      opj_read_bytes(p_header_data, &mut (*l_current_poc).compno1, l_comp_room);
      p_header_data = p_header_data.offset(l_comp_room as isize);
      opj_read_bytes(p_header_data, &mut l_tmp, 1 as OPJ_UINT32);
      p_header_data = p_header_data.offset(1);
      (*l_current_poc).prg = l_tmp as OPJ_PROG_ORDER;
      /* make sure comp is in acceptable bounds */
      (*l_current_poc).compno1 = opj_uint_min((*l_current_poc).compno1, l_nb_comp);
      l_current_poc = l_current_poc.offset(1);
      i += 1;
    }
    (*l_tcp).numpocs = l_current_poc_nb.wrapping_sub(1u32);
    1i32
  }
}
/* *
 * Reads a CRG marker (Component registration)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a CRG marker (Component registration)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_crg(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_nb_comp: OPJ_UINT32 = 0;
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_nb_comp = (*p_j2k.m_private_image).numcomps;
    if p_header_size != l_nb_comp.wrapping_mul(4u32) {
      event_msg!(p_manager, EVT_ERROR, "Error reading CRG marker\n",);
      return 0i32;
    }
    /* Do not care of this at the moment since only local variables are set here */
    /*
    for
            (i = 0; i < l_nb_comp; ++i)
    {
            opj_read_bytes(p_header_data,&l_Xcrg_i,2);                              // Xcrg_i
            p_header_data+=2;
            opj_read_bytes(p_header_data,&l_Ycrg_i,2);                              // Xcrg_i
            p_header_data+=2;
    }
    */
    1i32
  }
}
/* *
 * Reads a TLM marker (Tile Length Marker)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a TLM marker (Tile Length Marker)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_tlm(
  mut _p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_Ztlm: OPJ_UINT32 = 0;
    let mut l_Stlm: OPJ_UINT32 = 0;
    let mut l_ST: OPJ_UINT32 = 0;
    let mut l_SP: OPJ_UINT32 = 0;
    let mut l_tot_num_tp_remaining: OPJ_UINT32 = 0;
    let mut l_quotient: OPJ_UINT32 = 0;
    let mut l_Ptlm_size: OPJ_UINT32 = 0;
    /* preconditions */
    /* Stlm */

    assert!(!p_header_data.is_null());
    if p_header_size < 2u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading TLM marker\n",);
      return 0i32;
    }
    p_header_size = (p_header_size as core::ffi::c_uint).wrapping_sub(2u32) as OPJ_UINT32;
    opj_read_bytes(p_header_data, &mut l_Ztlm, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    opj_read_bytes(p_header_data, &mut l_Stlm, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    l_ST = l_Stlm >> 4i32 & 0x3u32;
    l_SP = l_Stlm >> 6i32 & 0x1u32;
    l_Ptlm_size = l_SP.wrapping_add(1u32).wrapping_mul(2u32);
    l_quotient = l_Ptlm_size.wrapping_add(l_ST);
    l_tot_num_tp_remaining = p_header_size.wrapping_rem(l_quotient);
    if l_tot_num_tp_remaining != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading TLM marker\n",);
      return 0i32;
    }
    /* FIXME Do not care of this at the moment since only local variables are set here */
    /*
    for
            (i = 0; i < l_tot_num_tp; ++i)
    {
            opj_read_bytes(p_header_data,&l_Ttlm_i,l_ST);                           // Ttlm_i
            p_header_data += l_ST;
            opj_read_bytes(p_header_data,&l_Ptlm_i,l_Ptlm_size);            // Ptlm_i
            p_header_data += l_Ptlm_size;
    }*/
    1i32
  }
}
/* *
 * Reads a PLM marker (Packet length, main header marker)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a PLM marker (Packet length, main header marker)
 *
 * @param       p_header_data   the data contained in the TLM box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the TLM marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_plm(
  mut _p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* preconditions */

  assert!(!p_header_data.is_null());
  if p_header_size < 1u32 {
    event_msg!(p_manager, EVT_ERROR, "Error reading PLM marker\n",);
    return 0i32;
  }
  /* Do not care of this at the moment since only local variables are set here */
  /*
  opj_read_bytes(p_header_data,&l_Zplm,1);                                        // Zplm
  ++p_header_data;
  --p_header_size;

  while
          (p_header_size > 0)
  {
          opj_read_bytes(p_header_data,&l_Nplm,1);                                // Nplm
          ++p_header_data;
          p_header_size -= (1+l_Nplm);
          if
                  (p_header_size < 0)
          {
                  event_msg!(p_manager, EVT_ERROR, "Error reading PLM marker\n");
                  return false;
          }
          for
                  (i = 0; i < l_Nplm; ++i)
          {
                  opj_read_bytes(p_header_data,&l_tmp,1);                         // Iplm_ij
                  ++p_header_data;
                  // take only the last seven bytes
                  l_packet_len |= (l_tmp & 0x7f);
                  if
                          (l_tmp & 0x80)
                  {
                          l_packet_len <<= 7;
                  }
                  else
                  {
          // store packet length and proceed to next packet
                          l_packet_len = 0;
                  }
          }
          if
                  (l_packet_len != 0)
          {
                  event_msg!(p_manager, EVT_ERROR, "Error reading PLM marker\n");
                  return false;
          }
  }
  */
  1i32
}

/* *
 * Reads a PLT marker (Packet length, tile-part header)
 *
 * @param       p_header_data   the data contained in the PLT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PLT marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a PLT marker (Packet length, tile-part header)
 *
 * @param       p_header_data   the data contained in the PLT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PLT marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_plt(
  mut _p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_Zplt: OPJ_UINT32 = 0;
    let mut l_tmp: OPJ_UINT32 = 0;
    let mut l_packet_len = 0 as OPJ_UINT32;
    let mut i: OPJ_UINT32 = 0;
    /* preconditions */
    /* Iplt_ij */

    assert!(!p_header_data.is_null());
    if p_header_size < 1u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading PLT marker\n",);
      return 0i32;
    }
    opj_read_bytes(p_header_data, &mut l_Zplt, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    p_header_size = p_header_size.wrapping_sub(1);
    i = 0 as OPJ_UINT32;
    while i < p_header_size {
      opj_read_bytes(p_header_data, &mut l_tmp, 1 as OPJ_UINT32);
      p_header_data = p_header_data.offset(1);
      /* take only the last seven bytes */
      l_packet_len |= l_tmp & 0x7fu32;
      if l_tmp & 0x80u32 != 0 {
        l_packet_len <<= 7i32
      } else {
        /* store packet length and proceed to next packet */
        l_packet_len = 0 as OPJ_UINT32
      }
      i += 1;
    }
    if l_packet_len != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading PLT marker\n",);
      return 0i32;
    }
    1i32
  }
}
/* *
 * Reads a PPM marker (Packed headers, main header)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
 */
/* *
 * Reads a PPM marker (Packed packet headers, main header)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
 */
fn opj_j2k_read_ppm(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_Z_ppm: OPJ_UINT32 = 0;
    /* preconditions */

    assert!(!p_header_data.is_null());
    /* We need to have the Z_ppm element + 1 byte of Nppm/Ippm at minimum */
    if p_header_size < 2u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading PPM marker\n",); /* Z_ppm */
      return 0i32;
    }
    l_cp = &mut p_j2k.m_cp;
    (*l_cp).ppm = true;
    opj_read_bytes(p_header_data, &mut l_Z_ppm, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    p_header_size = p_header_size.wrapping_sub(1);
    /* check allocation needed */
    if (*l_cp).ppm_markers.is_null() {
      /* first PPM marker */
      let mut l_newCount = l_Z_ppm.wrapping_add(1u32); /* can't overflow, l_Z_ppm is UINT8 */
      assert!((*l_cp).ppm_markers_count == 0u32);
      (*l_cp).ppm_markers =
        opj_calloc(l_newCount as size_t, core::mem::size_of::<opj_ppx>()) as *mut opj_ppx;
      if (*l_cp).ppm_markers.is_null() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to read PPM marker\n",
        );
        return 0i32;
      }
      (*l_cp).ppm_markers_count = l_newCount
    } else if (*l_cp).ppm_markers_count <= l_Z_ppm {
      let mut l_newCount_0 = l_Z_ppm.wrapping_add(1u32);
      let mut new_ppm_markers = core::ptr::null_mut::<opj_ppx>();
      new_ppm_markers = opj_realloc(
        (*l_cp).ppm_markers as *mut core::ffi::c_void,
        (l_newCount_0 as usize).wrapping_mul(core::mem::size_of::<opj_ppx>()),
      ) as *mut opj_ppx;
      if new_ppm_markers.is_null() {
        /* clean up to be done on l_cp destruction */
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to read PPM marker\n",
        );
        return 0i32;
      }
      (*l_cp).ppm_markers = new_ppm_markers;
      memset(
        (*l_cp)
          .ppm_markers
          .offset((*l_cp).ppm_markers_count as isize) as *mut core::ffi::c_void,
        0i32,
        (l_newCount_0.wrapping_sub((*l_cp).ppm_markers_count) as usize)
          .wrapping_mul(core::mem::size_of::<opj_ppx>()),
      );
      (*l_cp).ppm_markers_count = l_newCount_0
    }
    if !(*(*l_cp).ppm_markers.offset(l_Z_ppm as isize))
      .m_data
      .is_null()
    {
      /* clean up to be done on l_cp destruction */
      event_msg!(p_manager, EVT_ERROR, "Zppm %u already read\n", l_Z_ppm,);
      return 0i32;
    }
    let fresh12 = &mut (*(*l_cp).ppm_markers.offset(l_Z_ppm as isize)).m_data;
    *fresh12 = opj_malloc(p_header_size as size_t) as *mut OPJ_BYTE;
    if (*(*l_cp).ppm_markers.offset(l_Z_ppm as isize))
      .m_data
      .is_null()
    {
      /* clean up to be done on l_cp destruction */
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to read PPM marker\n",
      );
      return 0i32;
    }
    (*(*l_cp).ppm_markers.offset(l_Z_ppm as isize)).m_data_size = p_header_size;
    memcpy(
      (*(*l_cp).ppm_markers.offset(l_Z_ppm as isize)).m_data as *mut core::ffi::c_void,
      p_header_data as *const core::ffi::c_void,
      p_header_size as usize,
    );
    1i32
  }
}
/* *
 * Merges all PPM markers read (Packed headers, main header)
 *
 * @param       p_cp      main coding parameters.
 * @param       p_manager the user event manager.
 */
/* *
 * Merges all PPM markers read (Packed headers, main header)
 *
 * @param       p_cp      main coding parameters.
 * @param       p_manager the user event manager.
 */
fn opj_j2k_merge_ppm(mut p_cp: *mut opj_cp_t, mut p_manager: &mut opj_event_mgr) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_ppm_data_size: OPJ_UINT32 = 0;
    let mut l_N_ppm_remaining: OPJ_UINT32 = 0;
    /* preconditions */

    assert!(!p_cp.is_null());
    assert!((*p_cp).ppm_buffer.is_null());
    if !(*p_cp).ppm {
      return 1i32;
    }

    l_ppm_data_size = 0u32;
    l_N_ppm_remaining = 0u32;
    i = 0u32;
    while i < (*p_cp).ppm_markers_count {
      if !(*(*p_cp).ppm_markers.offset(i as isize)).m_data.is_null() {
        /* standard doesn't seem to require contiguous Zppm */
        let mut l_N_ppm: OPJ_UINT32 = 0;
        let mut l_data_size = (*(*p_cp).ppm_markers.offset(i as isize)).m_data_size;
        let mut l_data: *const OPJ_BYTE = (*(*p_cp).ppm_markers.offset(i as isize)).m_data;
        if l_N_ppm_remaining >= l_data_size {
          l_N_ppm_remaining =
            (l_N_ppm_remaining as core::ffi::c_uint).wrapping_sub(l_data_size) as OPJ_UINT32;
          l_data_size = 0u32
        } else {
          l_data = l_data.offset(l_N_ppm_remaining as isize);
          l_data_size =
            (l_data_size as core::ffi::c_uint).wrapping_sub(l_N_ppm_remaining) as OPJ_UINT32;
          l_N_ppm_remaining = 0u32
        }
        if l_data_size > 0u32 {
          loop {
            /* read Nppm */
            if l_data_size < 4u32 {
              /* clean up to be done on l_cp destruction */
              event_msg!(p_manager, EVT_ERROR, "Not enough bytes to read Nppm\n",);
              return 0i32;
            }
            opj_read_bytes(l_data, &mut l_N_ppm, 4 as OPJ_UINT32);
            l_data = l_data.offset(4);
            l_data_size -= 4;

            if l_ppm_data_size > u32::MAX - l_N_ppm {
              event_msg!(p_manager, EVT_ERROR, "Too large value for Nppm\n",);
              return 0i32;
            }
            l_ppm_data_size += l_N_ppm;
            if l_data_size >= l_N_ppm {
              l_data_size = (l_data_size as core::ffi::c_uint).wrapping_sub(l_N_ppm) as OPJ_UINT32;
              l_data = l_data.offset(l_N_ppm as isize)
            } else {
              l_N_ppm_remaining = l_N_ppm.wrapping_sub(l_data_size);
              l_data_size = 0u32
            }
            if l_data_size <= 0u32 {
              break;
            }
          }
        }
      }
      i += 1;
    }
    if l_N_ppm_remaining != 0u32 {
      /* clean up to be done on l_cp destruction */
      event_msg!(p_manager, EVT_ERROR, "Corrupted PPM markers\n",);
      return 0i32;
    }
    (*p_cp).ppm_buffer = opj_malloc(l_ppm_data_size as size_t) as *mut OPJ_BYTE;
    if (*p_cp).ppm_buffer.is_null() {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to read PPM marker\n",
      );
      return 0i32;
    }
    (*p_cp).ppm_len = l_ppm_data_size;
    l_ppm_data_size = 0u32;
    l_N_ppm_remaining = 0u32;
    i = 0u32;
    while i < (*p_cp).ppm_markers_count {
      if !(*(*p_cp).ppm_markers.offset(i as isize)).m_data.is_null() {
        /* standard doesn't seem to require contiguous Zppm */
        let mut l_N_ppm_0: OPJ_UINT32 = 0;
        let mut l_data_size_0 = (*(*p_cp).ppm_markers.offset(i as isize)).m_data_size;
        let mut l_data_0: *const OPJ_BYTE = (*(*p_cp).ppm_markers.offset(i as isize)).m_data;
        if l_N_ppm_remaining >= l_data_size_0 {
          memcpy(
            (*p_cp).ppm_buffer.offset(l_ppm_data_size as isize) as *mut core::ffi::c_void,
            l_data_0 as *const core::ffi::c_void,
            l_data_size_0 as usize,
          );
          l_ppm_data_size =
            (l_ppm_data_size as core::ffi::c_uint).wrapping_add(l_data_size_0) as OPJ_UINT32;
          l_N_ppm_remaining =
            (l_N_ppm_remaining as core::ffi::c_uint).wrapping_sub(l_data_size_0) as OPJ_UINT32;
          l_data_size_0 = 0u32
        } else {
          memcpy(
            (*p_cp).ppm_buffer.offset(l_ppm_data_size as isize) as *mut core::ffi::c_void,
            l_data_0 as *const core::ffi::c_void,
            l_N_ppm_remaining as usize,
          );
          l_ppm_data_size =
            (l_ppm_data_size as core::ffi::c_uint).wrapping_add(l_N_ppm_remaining) as OPJ_UINT32;
          l_data_0 = l_data_0.offset(l_N_ppm_remaining as isize);
          l_data_size_0 =
            (l_data_size_0 as core::ffi::c_uint).wrapping_sub(l_N_ppm_remaining) as OPJ_UINT32;
          l_N_ppm_remaining = 0u32
        }
        if l_data_size_0 > 0u32 {
          loop {
            /* read Nppm */
            if l_data_size_0 < 4u32 {
              /* clean up to be done on l_cp destruction */
              event_msg!(p_manager, EVT_ERROR, "Not enough bytes to read Nppm\n",);
              return 0i32;
            }
            opj_read_bytes(l_data_0, &mut l_N_ppm_0, 4 as OPJ_UINT32);
            l_data_0 = l_data_0.offset(4);
            l_data_size_0 =
              (l_data_size_0 as core::ffi::c_uint).wrapping_sub(4u32) as OPJ_UINT32 as OPJ_UINT32;
            if l_data_size_0 >= l_N_ppm_0 {
              memcpy(
                (*p_cp).ppm_buffer.offset(l_ppm_data_size as isize) as *mut core::ffi::c_void,
                l_data_0 as *const core::ffi::c_void,
                l_N_ppm_0 as usize,
              );
              l_ppm_data_size =
                (l_ppm_data_size as core::ffi::c_uint).wrapping_add(l_N_ppm_0) as OPJ_UINT32;
              l_data_size_0 =
                (l_data_size_0 as core::ffi::c_uint).wrapping_sub(l_N_ppm_0) as OPJ_UINT32;
              l_data_0 = l_data_0.offset(l_N_ppm_0 as isize)
            } else {
              memcpy(
                (*p_cp).ppm_buffer.offset(l_ppm_data_size as isize) as *mut core::ffi::c_void,
                l_data_0 as *const core::ffi::c_void,
                l_data_size_0 as usize,
              );
              l_ppm_data_size =
                (l_ppm_data_size as core::ffi::c_uint).wrapping_add(l_data_size_0) as OPJ_UINT32;
              l_N_ppm_remaining = l_N_ppm_0.wrapping_sub(l_data_size_0);
              l_data_size_0 = 0u32
            }
            if l_data_size_0 <= 0u32 {
              break;
            }
          }
        }
        opj_free((*(*p_cp).ppm_markers.offset(i as isize)).m_data as *mut core::ffi::c_void);
        let fresh13 = &mut (*(*p_cp).ppm_markers.offset(i as isize)).m_data;
        *fresh13 = core::ptr::null_mut::<OPJ_BYTE>();
        (*(*p_cp).ppm_markers.offset(i as isize)).m_data_size = 0u32
      }
      i += 1;
    }
    (*p_cp).ppm_data = (*p_cp).ppm_buffer;
    (*p_cp).ppm_data_size = (*p_cp).ppm_len;
    (*p_cp).ppm_markers_count = 0u32;
    opj_free((*p_cp).ppm_markers as *mut core::ffi::c_void);
    (*p_cp).ppm_markers = core::ptr::null_mut::<opj_ppx>();
    1i32
  }
}
/* *
 * Reads a PPT marker (Packed packet headers, tile-part header)
 *
 * @param       p_header_data   the data contained in the PPT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PPT marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a PPT marker (Packed packet headers, tile-part header)
 *
 * @param       p_header_data   the data contained in the PPT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PPT marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_ppt(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_Z_ppt: OPJ_UINT32 = 0;
    /* preconditions */

    assert!(!p_header_data.is_null());
    /* We need to have the Z_ppt element + 1 byte of Ippt at minimum */
    if p_header_size < 2u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading PPT marker\n",); /* Z_ppt */
      return 0i32;
    }
    l_cp = &mut p_j2k.m_cp;
    if (*l_cp).ppm {
      event_msg!(p_manager, EVT_ERROR,
                      "Error reading PPT marker: packet header have been previously found in the main header (PPM marker).\n");
      return 0i32;
    }
    l_tcp = &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t;
    (*l_tcp).ppt = true;
    opj_read_bytes(p_header_data, &mut l_Z_ppt, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    p_header_size = p_header_size.wrapping_sub(1);
    /* check allocation needed */
    if (*l_tcp).ppt_markers.is_null() {
      /* first PPT marker */
      let mut l_newCount = l_Z_ppt.wrapping_add(1u32); /* can't overflow, l_Z_ppt is UINT8 */
      assert!((*l_tcp).ppt_markers_count == 0u32);
      (*l_tcp).ppt_markers =
        opj_calloc(l_newCount as size_t, core::mem::size_of::<opj_ppx>()) as *mut opj_ppx;
      if (*l_tcp).ppt_markers.is_null() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to read PPT marker\n",
        );
        return 0i32;
      }
      (*l_tcp).ppt_markers_count = l_newCount
    } else if (*l_tcp).ppt_markers_count <= l_Z_ppt {
      let mut l_newCount_0 = l_Z_ppt.wrapping_add(1u32);
      let mut new_ppt_markers = core::ptr::null_mut::<opj_ppx>();
      new_ppt_markers = opj_realloc(
        (*l_tcp).ppt_markers as *mut core::ffi::c_void,
        (l_newCount_0 as usize).wrapping_mul(core::mem::size_of::<opj_ppx>()),
      ) as *mut opj_ppx;
      if new_ppt_markers.is_null() {
        /* clean up to be done on l_tcp destruction */
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to read PPT marker\n",
        );
        return 0i32;
      }
      (*l_tcp).ppt_markers = new_ppt_markers;
      memset(
        (*l_tcp)
          .ppt_markers
          .offset((*l_tcp).ppt_markers_count as isize) as *mut core::ffi::c_void,
        0i32,
        (l_newCount_0.wrapping_sub((*l_tcp).ppt_markers_count) as usize)
          .wrapping_mul(core::mem::size_of::<opj_ppx>()),
      );
      (*l_tcp).ppt_markers_count = l_newCount_0
    }
    if !(*(*l_tcp).ppt_markers.offset(l_Z_ppt as isize))
      .m_data
      .is_null()
    {
      /* clean up to be done on l_tcp destruction */
      event_msg!(p_manager, EVT_ERROR, "Zppt %u already read\n", l_Z_ppt,);
      return 0i32;
    }
    let fresh14 = &mut (*(*l_tcp).ppt_markers.offset(l_Z_ppt as isize)).m_data;
    *fresh14 = opj_malloc(p_header_size as size_t) as *mut OPJ_BYTE;
    if (*(*l_tcp).ppt_markers.offset(l_Z_ppt as isize))
      .m_data
      .is_null()
    {
      /* clean up to be done on l_tcp destruction */
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to read PPT marker\n",
      );
      return 0i32;
    }
    (*(*l_tcp).ppt_markers.offset(l_Z_ppt as isize)).m_data_size = p_header_size;
    memcpy(
      (*(*l_tcp).ppt_markers.offset(l_Z_ppt as isize)).m_data as *mut core::ffi::c_void,
      p_header_data as *const core::ffi::c_void,
      p_header_size as usize,
    );
    1i32
  }
}
/* *
 * Merges all PPT markers read (Packed headers, tile-part header)
 *
 * @param       p_tcp   the tile.
 * @param       p_manager               the user event manager.
 */
/* *
 * Merges all PPT markers read (Packed packet headers, tile-part header)
 *
 * @param       p_tcp   the tile.
 * @param       p_manager               the user event manager.
 */
fn opj_j2k_merge_ppt(mut p_tcp: *mut opj_tcp_t, mut p_manager: &mut opj_event_mgr) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_ppt_data_size: OPJ_UINT32 = 0;
    /* preconditions */

    assert!(!p_tcp.is_null());
    if !(*p_tcp).ppt_buffer.is_null() {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "opj_j2k_merge_ppt() has already been called\n",
      );
      return 0i32;
    }
    if !(*p_tcp).ppt {
      return 1i32;
    }
    l_ppt_data_size = 0u32;
    i = 0u32;
    while i < (*p_tcp).ppt_markers_count {
      l_ppt_data_size = (l_ppt_data_size as core::ffi::c_uint)
        .wrapping_add((*(*p_tcp).ppt_markers.offset(i as isize)).m_data_size)
        as OPJ_UINT32;
      i += 1;
      /* can't overflow, max 256 markers of max 65536 bytes */
    }
    (*p_tcp).ppt_buffer = opj_malloc(l_ppt_data_size as size_t) as *mut OPJ_BYTE;
    if (*p_tcp).ppt_buffer.is_null() {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to read PPT marker\n",
      );
      return 0i32;
    }
    (*p_tcp).ppt_len = l_ppt_data_size;
    l_ppt_data_size = 0u32;
    i = 0u32;
    while i < (*p_tcp).ppt_markers_count {
      if !(*(*p_tcp).ppt_markers.offset(i as isize)).m_data.is_null() {
        /* standard doesn't seem to require contiguous Zppt */
        memcpy(
          (*p_tcp).ppt_buffer.offset(l_ppt_data_size as isize) as *mut core::ffi::c_void,
          (*(*p_tcp).ppt_markers.offset(i as isize)).m_data as *const core::ffi::c_void,
          (*(*p_tcp).ppt_markers.offset(i as isize)).m_data_size as usize,
        ); /* can't overflow, max 256 markers of max 65536 bytes */
        l_ppt_data_size = (l_ppt_data_size as core::ffi::c_uint)
          .wrapping_add((*(*p_tcp).ppt_markers.offset(i as isize)).m_data_size)
          as OPJ_UINT32;
        opj_free((*(*p_tcp).ppt_markers.offset(i as isize)).m_data as *mut core::ffi::c_void);
        let fresh15 = &mut (*(*p_tcp).ppt_markers.offset(i as isize)).m_data;
        *fresh15 = core::ptr::null_mut::<OPJ_BYTE>();
        (*(*p_tcp).ppt_markers.offset(i as isize)).m_data_size = 0u32
      }
      i += 1;
    }
    (*p_tcp).ppt_markers_count = 0u32;
    opj_free((*p_tcp).ppt_markers as *mut core::ffi::c_void);
    (*p_tcp).ppt_markers = core::ptr::null_mut::<opj_ppx>();
    (*p_tcp).ppt_data = (*p_tcp).ppt_buffer;
    (*p_tcp).ppt_data_size = (*p_tcp).ppt_len;
    1i32
  }
}
/* *
 * Writes the TLM marker (Tile Length Marker)
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_tlm(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_tlm_size: OPJ_UINT32 = 0;
    let mut size_per_tile_part: OPJ_UINT32 = 0;
    /* preconditions */

    /* 10921 = (65535 - header_size) / size_per_tile_part where */
    /* header_size = 4 and size_per_tile_part = 6 */
    if p_j2k.m_specific_param.m_encoder.m_total_tile_parts > 10921u32 {
      /* We could do more but it would require writing several TLM markers */
      event_msg!(
        p_manager,
        EVT_ERROR,
        "A maximum of 10921 tile-parts are supported currently when writing TLM marker\n",
      );
      return 0i32;
    }
    if p_j2k.m_specific_param.m_encoder.m_total_tile_parts <= 255u32 {
      size_per_tile_part = 5 as OPJ_UINT32;
      p_j2k.m_specific_param.m_encoder.m_Ttlmi_is_byte = 1i32
    } else {
      size_per_tile_part = 6 as OPJ_UINT32;
      p_j2k.m_specific_param.m_encoder.m_Ttlmi_is_byte = 0i32
    }
    l_tlm_size = ((2i32 + 4i32) as core::ffi::c_uint).wrapping_add(
      size_per_tile_part.wrapping_mul(p_j2k.m_specific_param.m_encoder.m_total_tile_parts),
    );
    if l_tlm_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_tlm_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write TLM marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_tlm_size
    }
    memset(
      p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
      0i32,
      l_tlm_size as usize,
    );
    l_current_data = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    /* change the way data is written to avoid seeking if possible */
    /* TODO */
    p_j2k.m_specific_param.m_encoder.m_tlm_start = opj_stream_tell(p_stream); /* TLM */
    opj_write_bytes(l_current_data, J2KMarker::TLM.as_u32(), 2 as OPJ_UINT32); /* Lpoc */
    l_current_data = l_current_data.offset(2); /* Ztlm=0*/
    opj_write_bytes(
      l_current_data,
      l_tlm_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(l_current_data, 0 as OPJ_UINT32, 1 as OPJ_UINT32);
    l_current_data = l_current_data.offset(1);
    /* Stlm 0x50= ST=1(8bits-255 tiles max),SP=1(Ptlm=32bits) */
    /* Stlm 0x60= ST=2(16bits-65535 tiles max),SP=1(Ptlm=32bits) */
    opj_write_bytes(
      l_current_data,
      if size_per_tile_part == 5u32 {
        0x50i32
      } else {
        0x60i32
      } as OPJ_UINT32,
      1 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(1);
    /* do nothing on the size_per_tile_part * l_j2k->m_specific_param.m_encoder.m_total_tile_parts remaining data */
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_tlm_size as OPJ_SIZE_T,
      p_manager,
    ) != l_tlm_size as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Writes the SOT marker (Start of tile-part)
 *
 * @param       p_j2k            J2K codec.
 * @param       p_data           Output buffer
 * @param       total_data_size  Output buffer size
 * @param       p_data_written   Number of bytes written into stream
 * @param       p_stream         the stream to write data to.
 * @param       p_manager        the user event manager.
*/
fn opj_j2k_write_sot(
  mut p_j2k: &mut opj_j2k,
  mut p_data: *mut OPJ_BYTE,
  mut total_data_size: OPJ_UINT32,
  mut p_data_written: *mut OPJ_UINT32,
  mut _p_stream: &Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    /* preconditions */
    /* Lsot */

    if total_data_size < 12u32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough bytes in output buffer to write SOT marker\n",
      );
      return 0i32;
    }
    opj_write_bytes(p_data, J2KMarker::SOT.as_u32(), 2 as OPJ_UINT32);
    p_data = p_data.offset(2);
    opj_write_bytes(p_data, 10 as OPJ_UINT32, 2 as OPJ_UINT32);
    p_data = p_data.offset(2);
    opj_write_bytes(p_data, p_j2k.m_current_tile_number, 2 as OPJ_UINT32);
    p_data = p_data.offset(2);
    /* Psot  */
    p_data = p_data.offset(4); /* TPsot */
    opj_write_bytes(
      p_data,
      p_j2k.m_specific_param.m_encoder.m_current_tile_part_number,
      1 as OPJ_UINT32,
    ); /* TNsot */
    p_data = p_data.offset(1);
    opj_write_bytes(
      p_data,
      (*p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize)).m_nb_tile_parts,
      1 as OPJ_UINT32,
    );
    p_data = p_data.offset(1);
    /* UniPG>> */
    /* USE_JPWL */
    *p_data_written = 12 as OPJ_UINT32;
    1i32
  }
}
/* *
 * Reads values from a SOT marker (Start of tile-part)
 *
 * the j2k decoder state is not affected. No side effects, no checks except for p_header_size.
 *
 * @param       p_header_data   the data contained in the SOT marker.
 * @param       p_header_size   the size of the data contained in the SOT marker.
 * @param       p_tile_no       Isot.
 * @param       p_tot_len       Psot.
 * @param       p_current_part  TPsot.
 * @param       p_num_parts     TNsot.
 * @param       p_manager       the user event manager.
 */
fn opj_j2k_get_sot_values(
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_tile_no: *mut OPJ_UINT32,
  mut p_tot_len: *mut OPJ_UINT32,
  mut p_current_part: *mut OPJ_UINT32,
  mut p_num_parts: *mut OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    /* preconditions */

    assert!(!p_header_data.is_null());
    /* Size of this marker is fixed = 12 (we have already read marker and its size)*/
    if p_header_size != 8u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading SOT marker\n",); /* Isot */
      return 0i32;
    } /* Psot */
    opj_read_bytes(p_header_data, p_tile_no, 2 as OPJ_UINT32); /* TPsot */
    p_header_data = p_header_data.offset(2); /* TNsot */
    opj_read_bytes(p_header_data, p_tot_len, 4 as OPJ_UINT32);
    p_header_data = p_header_data.offset(4);
    opj_read_bytes(p_header_data, p_current_part, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    opj_read_bytes(p_header_data, p_num_parts, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    1i32
  }
}
/* *
 * Reads a SOT marker (Start of tile-part)
 *
 * @param       p_header_data   the data contained in the SOT marker.
 * @param       p_j2k           the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the PPT marker.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_read_sot(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tot_len: OPJ_UINT32 = 0;
    let mut l_num_parts = 0 as OPJ_UINT32;
    let mut l_current_part: OPJ_UINT32 = 0;
    let mut l_tile_x: OPJ_UINT32 = 0;
    let mut l_tile_y: OPJ_UINT32 = 0;
    /* preconditions */

    if opj_j2k_get_sot_values(
      p_header_data,
      p_header_size,
      &mut p_j2k.m_current_tile_number,
      &mut l_tot_len,
      &mut l_current_part,
      &mut l_num_parts,
      p_manager,
    ) == 0
    {
      event_msg!(p_manager, EVT_ERROR, "Error reading SOT marker\n",);
      return 0i32;
    }
    l_cp = &mut p_j2k.m_cp;
    /* testcase 2.pdf.SIGFPE.706.1112 */
    if p_j2k.m_current_tile_number >= (*l_cp).tw.wrapping_mul((*l_cp).th) {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid tile number %d\n",
        p_j2k.m_current_tile_number,
      );
      return 0i32;
    }
    l_tcp = &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t;
    l_tile_x = p_j2k.m_current_tile_number.wrapping_rem((*l_cp).tw);
    l_tile_y = p_j2k.m_current_tile_number.wrapping_div((*l_cp).tw);
    if p_j2k.m_specific_param.m_decoder.m_tile_ind_to_dec < 0i32
      || p_j2k.m_current_tile_number
        == p_j2k.m_specific_param.m_decoder.m_tile_ind_to_dec as OPJ_UINT32
    {
      /* Do only this check if we decode all tile part headers, or if */
      /* we decode one precise tile. Otherwise the m_current_tile_part_number */
      /* might not be valid */
      /* Fixes issue with id_000020,sig_06,src_001958,op_flip4,pos_149 */
      /* of https://github.com/uclouvain/openjpeg/issues/939 */
      /* We must avoid reading twice the same tile part number for a given tile */
      /* so as to avoid various issues, like opj_j2k_merge_ppt being called */
      /* several times. */
      /* ISO 15444-1 A.4.2 Start of tile-part (SOT) mandates that tile parts */
      /* should appear in increasing order. */
      if (*l_tcp).m_current_tile_part_number + 1i32 != l_current_part as OPJ_INT32 {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Invalid tile part index for tile number %d. Got %d, expected %d\n",
          p_j2k.m_current_tile_number,
          l_current_part,
          (*l_tcp).m_current_tile_part_number + 1i32,
        );
        return 0i32;
      }
    }
    (*l_tcp).m_current_tile_part_number = l_current_part as OPJ_INT32;
    /* USE_JPWL */
    /* look for the tile in the list of already processed tile (in parts). */
    /* Optimization possible here with a more complex data structure and with the removing of tiles */
    /* since the time taken by this function can only grow at the time */
    /* PSot should be equal to zero or >=14 or <= 2^32-1 */
    if l_tot_len != 0u32 && l_tot_len < 14u32 {
      if l_tot_len == 12u32 {
        /* MSD: Special case for the PHR data which are read by kakadu*/
        event_msg!(
          p_manager,
          EVT_WARNING,
          "Empty SOT marker detected: Psot=%d.\n",
          l_tot_len,
        );
      } else {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Psot value is not correct regards to the JPEG2000 norm: %d.\n",
          l_tot_len,
        );
        return 0i32;
      }
    }
    /* USE_JPWL */
    /* Ref A.4.2: Psot could be equal zero if it is the last tile-part of the codestream.*/
    if l_tot_len == 0 {
      event_msg!(p_manager, EVT_INFO,
                      "Psot value of the current tile-part is equal to zero, we assuming it is the last tile-part of the codestream.\n");
      p_j2k.m_specific_param.m_decoder.m_last_tile_part = 1i32
    }
    if (*l_tcp).m_nb_tile_parts != 0u32 && l_current_part >= (*l_tcp).m_nb_tile_parts {
      /* Fixes https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=2851 */
      event_msg!(p_manager, EVT_ERROR,
                      "In SOT marker, TPSot (%d) is not valid regards to the previous number of tile-part (%d), giving up\n", l_current_part,
                      (*l_tcp).m_nb_tile_parts);
      p_j2k.m_specific_param.m_decoder.m_last_tile_part = 1i32;
      return 0i32;
    }
    if l_num_parts != 0u32 {
      /* Number of tile-part header is provided by this tile-part header */
      l_num_parts = (l_num_parts as core::ffi::c_uint)
        .wrapping_add(p_j2k.m_specific_param.m_decoder.m_nb_tile_parts_correction as _)
        as OPJ_UINT32;
      /* Useful to manage the case of textGBR.jp2 file because two values of TNSot are allowed: the correct numbers of
       * tile-parts for that tile and zero (A.4.2 of 15444-1 : 2002). */
      if (*l_tcp).m_nb_tile_parts != 0 && l_current_part >= (*l_tcp).m_nb_tile_parts {
        event_msg!(p_manager, EVT_ERROR,
                            "In SOT marker, TPSot (%d) is not valid regards to the current number of tile-part (%d), giving up\n",
                            l_current_part, (*l_tcp).m_nb_tile_parts);
        p_j2k.m_specific_param.m_decoder.m_last_tile_part = 1i32;
        return 0i32;
      }
      if l_current_part >= l_num_parts {
        /* testcase 451.pdf.SIGSEGV.ce9.3723 */
        event_msg!(p_manager, EVT_ERROR,
                          "In SOT marker, TPSot (%d) is not valid regards to the current number of tile-part (header) (%d), giving up\n",
                          l_current_part, l_num_parts);
        p_j2k.m_specific_param.m_decoder.m_last_tile_part = 1i32;
        return 0i32;
      }
      (*l_tcp).m_nb_tile_parts = l_num_parts
    }
    /* If know the number of tile part header we will check if we didn't read the last*/
    if (*l_tcp).m_nb_tile_parts != 0
      && (*l_tcp).m_nb_tile_parts == l_current_part.wrapping_add(1u32)
    {
      p_j2k.m_specific_param.m_decoder.m_can_decode = true;
      /* Process the last tile-part header*/
    }
    if p_j2k.m_specific_param.m_decoder.m_last_tile_part == 0 {
      /* Keep the size of data to skip after this marker */
      p_j2k.m_specific_param.m_decoder.m_sot_length = l_tot_len.wrapping_sub(12u32)
    /* SOT_marker_size = 12 */
    } else {
      /* FIXME: need to be computed from the number of bytes remaining in the codestream */
      p_j2k.m_specific_param.m_decoder.m_sot_length = 0 as OPJ_UINT32
    }
    p_j2k.m_specific_param.m_decoder.m_state = J2KState::TPH;
    /* Check if the current tile is outside the area we want decode or not corresponding to the tile index*/
    if p_j2k.m_specific_param.m_decoder.m_tile_ind_to_dec == -(1i32) {
      p_j2k.m_specific_param.m_decoder.m_skip_data = l_tile_x
        < p_j2k.m_specific_param.m_decoder.m_start_tile_x
        || l_tile_x >= p_j2k.m_specific_param.m_decoder.m_end_tile_x
        || l_tile_y < p_j2k.m_specific_param.m_decoder.m_start_tile_y
        || l_tile_y >= p_j2k.m_specific_param.m_decoder.m_end_tile_y;
    } else {
      assert!(p_j2k.m_specific_param.m_decoder.m_tile_ind_to_dec >= 0i32);
      p_j2k.m_specific_param.m_decoder.m_skip_data = p_j2k.m_current_tile_number
        != p_j2k.m_specific_param.m_decoder.m_tile_ind_to_dec as OPJ_UINT32;
    }
    /* Index */
    if !p_j2k.cstr_index.is_null() {
      assert!(!(*p_j2k.cstr_index).tile_index.is_null());
      (*(*p_j2k.cstr_index)
        .tile_index
        .offset(p_j2k.m_current_tile_number as isize))
      .tileno = p_j2k.m_current_tile_number;
      (*(*p_j2k.cstr_index)
        .tile_index
        .offset(p_j2k.m_current_tile_number as isize))
      .current_tpsno = l_current_part;
      if l_num_parts != 0u32 {
        (*(*p_j2k.cstr_index)
          .tile_index
          .offset(p_j2k.m_current_tile_number as isize))
        .nb_tps = l_num_parts;
        (*(*p_j2k.cstr_index)
          .tile_index
          .offset(p_j2k.m_current_tile_number as isize))
        .current_nb_tps = l_num_parts;
        if (*(*p_j2k.cstr_index)
          .tile_index
          .offset(p_j2k.m_current_tile_number as isize))
        .tp_index
        .is_null()
        {
          let fresh16 = &mut (*(*p_j2k.cstr_index)
            .tile_index
            .offset(p_j2k.m_current_tile_number as isize))
          .tp_index;
          *fresh16 = opj_calloc(
            l_num_parts as size_t,
            core::mem::size_of::<opj_tp_index_t>(),
          ) as *mut opj_tp_index_t;
          if (*(*p_j2k.cstr_index)
            .tile_index
            .offset(p_j2k.m_current_tile_number as isize))
          .tp_index
          .is_null()
          {
            event_msg!(
              p_manager,
              EVT_ERROR,
              "Not enough memory to read SOT marker. Tile index allocation failed\n",
            );
            return 0i32;
          }
        } else {
          let mut new_tp_index = opj_realloc(
            (*(*p_j2k.cstr_index)
              .tile_index
              .offset(p_j2k.m_current_tile_number as isize))
            .tp_index as *mut core::ffi::c_void,
            (l_num_parts as usize).wrapping_mul(core::mem::size_of::<opj_tp_index_t>()),
          ) as *mut opj_tp_index_t;
          if new_tp_index.is_null() {
            opj_free(
              (*(*p_j2k.cstr_index)
                .tile_index
                .offset(p_j2k.m_current_tile_number as isize))
              .tp_index as *mut core::ffi::c_void,
            );
            let fresh17 = &mut (*(*p_j2k.cstr_index)
              .tile_index
              .offset(p_j2k.m_current_tile_number as isize))
            .tp_index;
            *fresh17 = core::ptr::null_mut::<opj_tp_index_t>();
            event_msg!(
              p_manager,
              EVT_ERROR,
              "Not enough memory to read SOT marker. Tile index allocation failed\n",
            );
            return 0i32;
          }
          let fresh18 = &mut (*(*p_j2k.cstr_index)
            .tile_index
            .offset(p_j2k.m_current_tile_number as isize))
          .tp_index;
          *fresh18 = new_tp_index
        }
      } else {
        /*if (!p_j2k->cstr_index->tile_index[p_j2k->m_current_tile_number].tp_index)*/
        if (*(*p_j2k.cstr_index)
          .tile_index
          .offset(p_j2k.m_current_tile_number as isize))
        .tp_index
        .is_null()
        {
          (*(*p_j2k.cstr_index)
            .tile_index
            .offset(p_j2k.m_current_tile_number as isize))
          .current_nb_tps = 10 as OPJ_UINT32;
          let fresh19 = &mut (*(*p_j2k.cstr_index)
            .tile_index
            .offset(p_j2k.m_current_tile_number as isize))
          .tp_index;
          *fresh19 = opj_calloc(
            (*(*p_j2k.cstr_index)
              .tile_index
              .offset(p_j2k.m_current_tile_number as isize))
            .current_nb_tps as size_t,
            core::mem::size_of::<opj_tp_index_t>(),
          ) as *mut opj_tp_index_t;
          if (*(*p_j2k.cstr_index)
            .tile_index
            .offset(p_j2k.m_current_tile_number as isize))
          .tp_index
          .is_null()
          {
            (*(*p_j2k.cstr_index)
              .tile_index
              .offset(p_j2k.m_current_tile_number as isize))
            .current_nb_tps = 0 as OPJ_UINT32;
            event_msg!(
              p_manager,
              EVT_ERROR,
              "Not enough memory to read SOT marker. Tile index allocation failed\n",
            );
            return 0i32;
          }
        }
        if l_current_part
          >= (*(*p_j2k.cstr_index)
            .tile_index
            .offset(p_j2k.m_current_tile_number as isize))
          .current_nb_tps
        {
          let mut new_tp_index_0 = core::ptr::null_mut::<opj_tp_index_t>();
          (*(*p_j2k.cstr_index)
            .tile_index
            .offset(p_j2k.m_current_tile_number as isize))
          .current_nb_tps = l_current_part.wrapping_add(1u32);
          new_tp_index_0 = opj_realloc(
            (*(*p_j2k.cstr_index)
              .tile_index
              .offset(p_j2k.m_current_tile_number as isize))
            .tp_index as *mut core::ffi::c_void,
            ((*(*p_j2k.cstr_index)
              .tile_index
              .offset(p_j2k.m_current_tile_number as isize))
            .current_nb_tps as usize)
              .wrapping_mul(core::mem::size_of::<opj_tp_index_t>()),
          ) as *mut opj_tp_index_t;
          if new_tp_index_0.is_null() {
            opj_free(
              (*(*p_j2k.cstr_index)
                .tile_index
                .offset(p_j2k.m_current_tile_number as isize))
              .tp_index as *mut core::ffi::c_void,
            );
            let fresh20 = &mut (*(*p_j2k.cstr_index)
              .tile_index
              .offset(p_j2k.m_current_tile_number as isize))
            .tp_index;
            *fresh20 = core::ptr::null_mut::<opj_tp_index_t>();
            (*(*p_j2k.cstr_index)
              .tile_index
              .offset(p_j2k.m_current_tile_number as isize))
            .current_nb_tps = 0 as OPJ_UINT32;
            event_msg!(
              p_manager,
              EVT_ERROR,
              "Not enough memory to read SOT marker. Tile index allocation failed\n",
            );
            return 0i32;
          }
          let fresh21 = &mut (*(*p_j2k.cstr_index)
            .tile_index
            .offset(p_j2k.m_current_tile_number as isize))
          .tp_index;
          *fresh21 = new_tp_index_0
        }
      }
    }
    /* FIXME move this onto a separate method to call before reading any SOT, remove part about main_end header, use a index struct inside p_j2k */
    /* if (p_j2k->cstr_info) {
    if (l_tcp->first) {
    if (tileno == 0) {
    p_j2k->cstr_info->main_head_end = p_stream_tell(p_stream) - 13;
    }

    p_j2k->cstr_info->tile[tileno].tileno = tileno;
    p_j2k->cstr_info->tile[tileno].start_pos = p_stream_tell(p_stream) - 12;
    p_j2k->cstr_info->tile[tileno].end_pos = p_j2k->cstr_info->tile[tileno].start_pos + totlen - 1;
    p_j2k->cstr_info->tile[tileno].num_tps = numparts;

    if (numparts) {
    p_j2k->cstr_info->tile[tileno].tp = (opj_tp_info_t *) opj_malloc(numparts * sizeof(opj_tp_info_t));
    }
    else {
    p_j2k->cstr_info->tile[tileno].tp = (opj_tp_info_t *) opj_malloc(10 * sizeof(opj_tp_info_t)); // Fixme (10)
    }
    }
    else {
    p_j2k->cstr_info->tile[tileno].end_pos += totlen;
    }

    p_j2k->cstr_info->tile[tileno].tp[partno].tp_start_pos = p_stream_tell(p_stream) - 12;
    p_j2k->cstr_info->tile[tileno].tp[partno].tp_end_pos =
    p_j2k->cstr_info->tile[tileno].tp[partno].tp_start_pos + totlen - 1;
    }*/
    1i32
  }
}
/* *
 * Write one or more PLT markers in the provided buffer
 */
fn opj_j2k_write_plt_in_memory(
  mut _p_j2k: &mut opj_j2k,
  mut marker_info: *mut opj_tcd_marker_info_t,
  mut p_data: *mut OPJ_BYTE,
  mut p_data_written: *mut OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut Zplt = 0 as OPJ_BYTE;
    let mut Lplt: OPJ_UINT16 = 0;
    let mut p_data_start = p_data;
    let mut p_data_Lplt = p_data.offset(2);
    let mut i: OPJ_UINT32 = 0;
    opj_write_bytes(p_data, J2KMarker::PLT.as_u32(), 2 as OPJ_UINT32);
    p_data = p_data.offset(2);
    /* Reserve space for Lplt */
    p_data = p_data.offset(2);
    opj_write_bytes(p_data, Zplt as OPJ_UINT32, 1 as OPJ_UINT32);
    p_data = p_data.offset(1);
    Lplt = 3 as OPJ_UINT16;
    i = 0 as OPJ_UINT32;
    while i < (*marker_info).packet_count {
      let mut var_bytes: [OPJ_BYTE; 5] = [0; 5];
      let mut var_bytes_size = 0 as OPJ_UINT8;
      let mut packet_size = *(*marker_info).p_packet_size.offset(i as isize);
      /* Packet size written in variable-length way, starting with LSB */
      var_bytes[var_bytes_size as usize] = (packet_size & 0x7fu32) as OPJ_BYTE;
      var_bytes_size = var_bytes_size.wrapping_add(1);
      packet_size >>= 7i32;
      while packet_size > 0u32 {
        var_bytes[var_bytes_size as usize] = (packet_size & 0x7fu32 | 0x80u32) as OPJ_BYTE;
        var_bytes_size = var_bytes_size.wrapping_add(1);
        packet_size >>= 7i32
      }
      /* Check if that can fit in the current PLT marker. If not, finish */
      /* current one, and start a new one */
      if Lplt as core::ffi::c_int + var_bytes_size as core::ffi::c_int > 65535i32 {
        if Zplt as core::ffi::c_int == 255i32 {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "More than 255 PLT markers would be needed for current tile-part !\n",
          );
          return 0i32;
        }
        /* Patch Lplt */
        opj_write_bytes(p_data_Lplt, Lplt as OPJ_UINT32, 2 as OPJ_UINT32);
        /* Start new segment */
        opj_write_bytes(p_data, J2KMarker::PLT.as_u32(), 2 as OPJ_UINT32);
        p_data = p_data.offset(2);
        /* Reserve space for Lplt */
        p_data_Lplt = p_data;
        p_data = p_data.offset(2);
        Zplt = Zplt.wrapping_add(1);
        opj_write_bytes(p_data, Zplt as OPJ_UINT32, 1 as OPJ_UINT32);
        p_data = p_data.offset(1);
        Lplt = 3 as OPJ_UINT16
      }
      Lplt = (Lplt as core::ffi::c_int + var_bytes_size as core::ffi::c_int) as OPJ_UINT16;
      /* Serialize variable-length packet size, starting with MSB */
      while var_bytes_size as core::ffi::c_int > 0i32 {
        opj_write_bytes(
          p_data,
          var_bytes[(var_bytes_size as core::ffi::c_int - 1i32) as usize] as OPJ_UINT32,
          1 as OPJ_UINT32,
        );
        p_data = p_data.offset(1);
        var_bytes_size = var_bytes_size.wrapping_sub(1)
      }
      i += 1;
    }
    *p_data_written = p_data.offset_from(p_data_start) as OPJ_UINT32;
    /* Patch Lplt */
    opj_write_bytes(p_data_Lplt, Lplt as OPJ_UINT32, 2 as OPJ_UINT32);
    1i32
  }
}
/* *
 * Writes the SOD marker (Start of data)
 *
 * This also writes optional PLT markers (before SOD)
 *
 * @param       p_j2k               J2K codec.
 * @param       p_data              FIXME DOC
 * @param       p_data_written      FIXME DOC
 * @param       total_data_size   FIXME DOC
 * @param       p_stream            the stream to write data to.
 * @param       p_manager           the user event manager.
*/
fn opj_j2k_write_sod(
  mut p_j2k: &mut opj_j2k,
  mut p_data: *mut OPJ_BYTE,
  mut p_data_written: *mut OPJ_UINT32,
  mut total_data_size: OPJ_UINT32,
  mut _p_stream: &Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cstr_info = core::ptr::null_mut::<opj_codestream_info_t>();
    let mut l_remaining_data: OPJ_UINT32 = 0;
    let mut marker_info = core::ptr::null_mut::<opj_tcd_marker_info_t>();
    /* preconditions */

    if total_data_size < 4u32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough bytes in output buffer to write SOD marker\n",
      );
      return 0i32;
    }
    opj_write_bytes(p_data, J2KMarker::SOD.as_u32(), 2 as OPJ_UINT32);
    /* make room for the EOF marker */
    l_remaining_data = total_data_size.wrapping_sub(4u32);
    /* update tile coder */
    p_j2k.m_tcd.tp_num = p_j2k
      .m_specific_param
      .m_encoder
      .m_current_poc_tile_part_number;
    p_j2k.m_tcd.cur_tp_num = p_j2k.m_specific_param.m_encoder.m_current_tile_part_number;
    /* INDEX >> */
    /* TODO mergeV2: check this part which use cstr_info */
    /*l_cstr_info = p_j2k->cstr_info;
    if (l_cstr_info) {
            if (!p_j2k->m_specific_param.m_encoder.m_current_tile_part_number ) {
                    //TODO cstr_info->tile[p_j2k->m_current_tile_number].end_header = p_stream_tell(p_stream) + p_j2k->pos_correction - 1;
                    l_cstr_info->tile[p_j2k->m_current_tile_number].tileno = p_j2k->m_current_tile_number;
            }
            else {*/
    /*
    TODO
    if
            (cstr_info->tile[p_j2k->m_current_tile_number].packet[cstr_info->packno - 1].end_pos < p_stream_tell(p_stream))
    {
            cstr_info->tile[p_j2k->m_current_tile_number].packet[cstr_info->packno].start_pos = p_stream_tell(p_stream);
    }*/
    /*}*/
    /* UniPG>> */
    /* USE_JPWL */
    /* <<UniPG */
    /*}*/
    /* << INDEX */
    if p_j2k.m_specific_param.m_encoder.m_current_tile_part_number == 0u32 {
      p_j2k.m_tcd.tcd_image.tiles.packno = 0 as OPJ_UINT32
    }
    *p_data_written = 0 as OPJ_UINT32;
    if p_j2k.m_specific_param.m_encoder.m_PLT != 0 {
      marker_info = opj_tcd_marker_info_create(p_j2k.m_specific_param.m_encoder.m_PLT);
      if marker_info.is_null() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Cannot encode tile: opj_tcd_marker_info_create() failed\n",
        );
        return 0i32;
      }
    }
    if l_remaining_data < p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough bytes in output buffer to write SOD marker\n",
      );
      opj_tcd_marker_info_destroy(marker_info);
      return 0i32;
    }
    l_remaining_data = (l_remaining_data as core::ffi::c_uint)
      .wrapping_sub(p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT)
      as OPJ_UINT32;
    if opj_tcd_encode_tile(
      &mut p_j2k.m_tcd,
      p_j2k.m_current_tile_number,
      p_data.offset(2),
      p_data_written,
      l_remaining_data,
      l_cstr_info,
      marker_info,
      p_manager,
    ) == 0
    {
      event_msg!(p_manager, EVT_ERROR, "Cannot encode tile\n",);
      opj_tcd_marker_info_destroy(marker_info);
      return 0i32;
    }
    /* For SOD */
    *p_data_written = (*p_data_written as core::ffi::c_uint).wrapping_add(2u32) as OPJ_UINT32;
    if p_j2k.m_specific_param.m_encoder.m_PLT != 0 {
      let mut l_data_written_PLT = 0 as OPJ_UINT32;
      let mut p_PLT_buffer =
        opj_malloc(p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT as size_t)
          as *mut OPJ_BYTE;
      if p_PLT_buffer.is_null() {
        event_msg!(p_manager, EVT_ERROR, "Cannot allocate memory\n",);
        opj_tcd_marker_info_destroy(marker_info);
        return 0i32;
      }
      if opj_j2k_write_plt_in_memory(
        p_j2k,
        marker_info,
        p_PLT_buffer,
        &mut l_data_written_PLT,
        p_manager,
      ) == 0
      {
        opj_tcd_marker_info_destroy(marker_info);
        opj_free(p_PLT_buffer as *mut core::ffi::c_void);
        return 0i32;
      }
      assert!(l_data_written_PLT <= p_j2k.m_specific_param.m_encoder.m_reserved_bytes_for_PLT);
      /* Move PLT marker(s) before SOD */
      memmove(
        p_data.offset(l_data_written_PLT as isize) as *mut core::ffi::c_void,
        p_data as *const core::ffi::c_void,
        *p_data_written as usize,
      );
      memcpy(
        p_data as *mut core::ffi::c_void,
        p_PLT_buffer as *const core::ffi::c_void,
        l_data_written_PLT as usize,
      );
      opj_free(p_PLT_buffer as *mut core::ffi::c_void);
      *p_data_written =
        (*p_data_written as core::ffi::c_uint).wrapping_add(l_data_written_PLT) as OPJ_UINT32
    }
    opj_tcd_marker_info_destroy(marker_info);
    1i32
  }
}
/* *
 * Reads a SOD marker (Start Of Data)
 *
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_stream                FIXME DOC
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_sod(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_current_read_size: OPJ_SIZE_T = 0;
    let mut l_cstr_index = core::ptr::null_mut::<opj_codestream_index_t>();
    let mut l_current_data = core::ptr::null_mut::<*mut OPJ_BYTE>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tile_len = core::ptr::null_mut::<OPJ_UINT32>();
    let mut l_sot_length_pb_detected = 0i32;
    /* preconditions */

    l_tcp = &mut *p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t;
    if p_j2k.m_specific_param.m_decoder.m_last_tile_part != 0 {
      /* opj_stream_get_number_byte_left returns OPJ_OFF_T
      // but we are in the last tile part,
      // so its result will fit on OPJ_UINT32 unless we find
      // a file with a single tile part of more than 4 GB...*/
      p_j2k.m_specific_param.m_decoder.m_sot_length =
        (opj_stream_get_number_byte_left(p_stream) - 2i64) as OPJ_UINT32
    } else if p_j2k.m_specific_param.m_decoder.m_sot_length >= 2u32 {
      p_j2k.m_specific_param.m_decoder.m_sot_length = (p_j2k.m_specific_param.m_decoder.m_sot_length
        as core::ffi::c_uint)
        .wrapping_sub(2u32) as OPJ_UINT32
    }
    l_current_data = &mut (*l_tcp).m_data;
    l_tile_len = &mut (*l_tcp).m_data_size;
    /* Check to avoid pass the limit of OPJ_UINT32 */
    /* Patch to support new PHR data */
    if p_j2k.m_specific_param.m_decoder.m_sot_length != 0 {
      /* If we are here, we'll try to read the data after allocation */
      /* Check enough bytes left in stream before allocation */
      if p_j2k.m_specific_param.m_decoder.m_sot_length as OPJ_OFF_T
        > opj_stream_get_number_byte_left(p_stream)
      {
        if p_j2k.m_cp.strict != 0 {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Tile part length size inconsistent with stream length\n",
          );
          return 0i32;
        } else {
          event_msg!(
            p_manager,
            EVT_WARNING,
            "Tile part length size inconsistent with stream length\n",
          );
        }
      }
      if p_j2k.m_specific_param.m_decoder.m_sot_length
        > (2147483647u32)
          .wrapping_mul(2u32)
          .wrapping_add(1u32)
          .wrapping_sub(2u32)
      {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "p_j2k->m_specific_param.m_decoder.m_sot_length > UINT_MAX - OPJ_COMMON_CBLK_DATA_EXTRA"
        );
        return 0i32;
      }
      /* Add a margin of OPJ_COMMON_CBLK_DATA_EXTRA to the allocation we */
      /* do so that opj_mqc_init_dec_common() can safely add a synthetic */
      /* 0xFFFF marker. */
      if (*l_current_data).is_null() {
        /* LH: oddly enough, in this path, l_tile_len!=0.
         * TODO: If this was consistent, we could simplify the code to only use realloc(), as realloc(0,...) default to malloc(0,...).
         */
        *l_current_data = opj_malloc(
          p_j2k
            .m_specific_param
            .m_decoder
            .m_sot_length
            .wrapping_add(2u32) as size_t,
        ) as *mut OPJ_BYTE
      } else {
        let mut l_new_current_data = core::ptr::null_mut::<OPJ_BYTE>();
        if *l_tile_len
          > (2147483647u32)
            .wrapping_mul(2u32)
            .wrapping_add(1u32)
            .wrapping_sub(2u32)
            .wrapping_sub(p_j2k.m_specific_param.m_decoder.m_sot_length)
        {
          event_msg!(p_manager, EVT_ERROR,
                              "*l_tile_len > UINT_MAX - OPJ_COMMON_CBLK_DATA_EXTRA - p_j2k->m_specific_param.m_decoder.m_sot_length");
          return 0i32;
        }
        l_new_current_data = opj_realloc(
          *l_current_data as *mut core::ffi::c_void,
          (*l_tile_len)
            .wrapping_add(p_j2k.m_specific_param.m_decoder.m_sot_length)
            .wrapping_add(2u32) as size_t,
        ) as *mut OPJ_BYTE;
        if l_new_current_data.is_null() {
          opj_free(*l_current_data as *mut core::ffi::c_void);
          /*nothing more is done as l_current_data will be set to null, and just
          afterward we enter in the error path
          and the actual tile_len is updated (committed) at the end of the
          function. */
        }
        *l_current_data = l_new_current_data
      }
      if (*l_current_data).is_null() {
        event_msg!(p_manager, EVT_ERROR, "Not enough memory to decode tile\n",);
        return 0i32;
      }
    } else {
      l_sot_length_pb_detected = 1i32
    }
    /* Index */
    l_cstr_index = p_j2k.cstr_index;
    if !l_cstr_index.is_null() {
      let mut l_current_pos = opj_stream_tell(p_stream) - 2i64;
      let mut l_current_tile_part = (*(*l_cstr_index)
        .tile_index
        .offset(p_j2k.m_current_tile_number as isize))
      .current_tpsno;
      (*(*(*l_cstr_index)
        .tile_index
        .offset(p_j2k.m_current_tile_number as isize))
      .tp_index
      .offset(l_current_tile_part as isize))
      .end_header = l_current_pos;
      (*(*(*l_cstr_index)
        .tile_index
        .offset(p_j2k.m_current_tile_number as isize))
      .tp_index
      .offset(l_current_tile_part as isize))
      .end_pos = l_current_pos + p_j2k.m_specific_param.m_decoder.m_sot_length as i64 + 2i64;
      if 0i32
        == opj_j2k_add_tlmarker(
          p_j2k.m_current_tile_number,
          l_cstr_index,
          J2KMarker::SOD,
          l_current_pos,
          p_j2k
            .m_specific_param
            .m_decoder
            .m_sot_length
            .wrapping_add(2u32),
        )
      {
        event_msg!(p_manager, EVT_ERROR, "Not enough memory to add tl marker\n",);
        return 0i32;
      }
      /*l_cstr_index->packno = 0;*/
    }
    /* Patch to support new PHR data */
    if l_sot_length_pb_detected == 0 {
      l_current_read_size = opj_stream_read_data(
        p_stream,
        (*l_current_data).offset(*l_tile_len as isize),
        p_j2k.m_specific_param.m_decoder.m_sot_length as OPJ_SIZE_T,
        p_manager,
      )
    } else {
      l_current_read_size = 0 as OPJ_SIZE_T
    }
    if l_current_read_size != p_j2k.m_specific_param.m_decoder.m_sot_length as usize {
      p_j2k.m_specific_param.m_decoder.m_state = J2KState::NEOC
    } else {
      p_j2k.m_specific_param.m_decoder.m_state = J2KState::TPHSOT
    }
    *l_tile_len = (*l_tile_len as core::ffi::c_uint).wrapping_add(l_current_read_size as OPJ_UINT32)
      as OPJ_UINT32;
    1i32
  }
}
/* *
 * Writes the RGN marker (Region Of Interest)
 *
 * @param       p_tile_no               the tile to output
 * @param       p_comp_no               the component to output
 * @param       nb_comps                the number of components
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_rgn(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_no: OPJ_UINT32,
  mut p_comp_no: OPJ_UINT32,
  mut nb_comps: OPJ_UINT32,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_rgn_size: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_comp_room: OPJ_UINT32 = 0;
    /* preconditions */
    /* Lrgn */
    /* Srgn */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    l_tccp = &mut *(*l_tcp).tccps.offset(p_comp_no as isize) as *mut opj_tccp_t;
    if nb_comps <= 256u32 {
      l_comp_room = 1 as OPJ_UINT32
    } else {
      l_comp_room = 2 as OPJ_UINT32
    }
    l_rgn_size = (6u32).wrapping_add(l_comp_room);
    l_current_data = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    opj_write_bytes(l_current_data, J2KMarker::RGN.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(
      l_current_data,
      l_rgn_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(l_current_data, p_comp_no, l_comp_room);
    l_current_data = l_current_data.offset(l_comp_room as isize);
    opj_write_bytes(l_current_data, 0 as OPJ_UINT32, 1 as OPJ_UINT32);
    l_current_data = l_current_data.offset(1);
    opj_write_bytes(
      l_current_data,
      (*l_tccp).roishift as OPJ_UINT32,
      1 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(1);
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_rgn_size as OPJ_SIZE_T,
      p_manager,
    ) != l_rgn_size as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Writes the EOC marker (End of Codestream)
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_eoc(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    /* preconditions */

    opj_write_bytes(
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      J2KMarker::EOC.as_u32(),
      2 as OPJ_UINT32,
    );
    /* UniPG>> */
    /* USE_JPWL */
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      2 as OPJ_SIZE_T,
      p_manager,
    ) != 2
    {
      return 0i32;
    }
    if opj_stream_flush(p_stream, p_manager) == 0 {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Reads a RGN marker (Region Of Interest)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a RGN marker (Region Of Interest)
 *
 * @param       p_header_data   the data contained in the POC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the POC marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_rgn(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_nb_comp: OPJ_UINT32 = 0;
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_comp_room: OPJ_UINT32 = 0;
    let mut l_comp_no: OPJ_UINT32 = 0;
    let mut l_roi_sty: OPJ_UINT32 = 0;
    /* preconditions*/
    /* Srgn */

    assert!(!p_header_data.is_null());
    l_image = p_j2k.m_private_image;
    l_nb_comp = (*l_image).numcomps;
    if l_nb_comp <= 256u32 {
      l_comp_room = 1 as OPJ_UINT32
    } else {
      l_comp_room = 2 as OPJ_UINT32
    }
    if p_header_size != (2u32).wrapping_add(l_comp_room) {
      event_msg!(p_manager, EVT_ERROR, "Error reading RGN marker\n",);
      return 0i32;
    }
    l_cp = &mut p_j2k.m_cp;
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    opj_read_bytes(p_header_data, &mut l_comp_no, l_comp_room);
    p_header_data = p_header_data.offset(l_comp_room as isize);
    opj_read_bytes(p_header_data, &mut l_roi_sty, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    /* USE_JPWL */
    /* testcase 3635.pdf.asan.77.2930 */
    if l_comp_no >= l_nb_comp {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "bad component number in RGN (%d when there are only %d)\n",
        l_comp_no,
        l_nb_comp,
      ); /* SPrgn */
      return 0i32;
    }
    opj_read_bytes(
      p_header_data,
      &mut (*(*l_tcp).tccps.offset(l_comp_no as isize)).roishift as *mut OPJ_INT32
        as *mut OPJ_UINT32,
      1 as OPJ_UINT32,
    );
    p_header_data = p_header_data.offset(1);
    1i32
  }
}
fn opj_j2k_get_tp_stride(mut p_tcp: *mut opj_tcp_t) -> OPJ_FLOAT32 {
  unsafe {
    (*p_tcp)
      .m_nb_tile_parts
      .wrapping_sub(1u32)
      .wrapping_mul(14u32) as OPJ_FLOAT32
  }
}

fn opj_j2k_get_default_stride(mut _p_tcp: *mut opj_tcp_t) -> OPJ_FLOAT32 {
  0 as OPJ_FLOAT32
}

/* *
 * Updates the rates of the tcp.
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_update_rates(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_img_comp = core::ptr::null_mut::<opj_image_comp_t>();
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut k: OPJ_UINT32 = 0;
    let mut l_x0: OPJ_INT32 = 0;
    let mut l_y0: OPJ_INT32 = 0;
    let mut l_x1: OPJ_INT32 = 0;
    let mut l_y1: OPJ_INT32 = 0;
    let mut l_rates = core::ptr::null_mut::<OPJ_FLOAT32>();
    let mut l_sot_remove: OPJ_FLOAT32 = 0.;
    let mut l_bits_empty: OPJ_UINT32 = 0;
    let mut l_size_pixel: OPJ_UINT32 = 0;
    let mut l_last_res: OPJ_UINT32 = 0;
    let mut l_tp_stride_func: Option<fn(_: *mut opj_tcp_t) -> OPJ_FLOAT32> = None;
    /* preconditions */

    l_cp = &mut p_j2k.m_cp;
    l_image = p_j2k.m_private_image;
    l_tcp = (*l_cp).tcps;
    l_bits_empty = (8u32)
      .wrapping_mul((*(*l_image).comps).dx)
      .wrapping_mul((*(*l_image).comps).dy);
    l_size_pixel = (*l_image).numcomps.wrapping_mul((*(*l_image).comps).prec);
    l_sot_remove =
      opj_stream_tell(p_stream) as OPJ_FLOAT32 / (*l_cp).th.wrapping_mul((*l_cp).tw) as OPJ_FLOAT32;
    if (*l_cp).m_specific_param.m_enc.m_tp_on {
      l_tp_stride_func = Some(opj_j2k_get_tp_stride as fn(_: *mut opj_tcp_t) -> OPJ_FLOAT32)
    } else {
      l_tp_stride_func = Some(opj_j2k_get_default_stride as fn(_: *mut opj_tcp_t) -> OPJ_FLOAT32)
    }
    i = 0 as OPJ_UINT32;
    while i < (*l_cp).th {
      j = 0 as OPJ_UINT32;
      while j < (*l_cp).tw {
        let mut l_offset = l_tp_stride_func.expect("non-null function pointer")(l_tcp)
          / (*l_tcp).numlayers as OPJ_FLOAT32;
        /* 4 borders of the tile rescale on the image if necessary */
        l_x0 = opj_int_max(
          (*l_cp).tx0.wrapping_add(j.wrapping_mul((*l_cp).tdx)) as OPJ_INT32,
          (*l_image).x0 as OPJ_INT32,
        );
        l_y0 = opj_int_max(
          (*l_cp).ty0.wrapping_add(i.wrapping_mul((*l_cp).tdy)) as OPJ_INT32,
          (*l_image).y0 as OPJ_INT32,
        );
        l_x1 = opj_int_min(
          (*l_cp)
            .tx0
            .wrapping_add(j.wrapping_add(1u32).wrapping_mul((*l_cp).tdx)) as OPJ_INT32,
          (*l_image).x1 as OPJ_INT32,
        );
        l_y1 = opj_int_min(
          (*l_cp)
            .ty0
            .wrapping_add(i.wrapping_add(1u32).wrapping_mul((*l_cp).tdy)) as OPJ_INT32,
          (*l_image).y1 as OPJ_INT32,
        );
        l_rates = (*l_tcp).rates.as_mut_ptr();
        /* Modification of the RATE >> */
        k = 0 as OPJ_UINT32;
        while k < (*l_tcp).numlayers {
          if *l_rates > 0.0f32 {
            *l_rates = (l_size_pixel as OPJ_FLOAT64
              * (l_x1 - l_x0) as OPJ_UINT32 as core::ffi::c_double
              * (l_y1 - l_y0) as OPJ_UINT32 as core::ffi::c_double
              / (*l_rates * l_bits_empty as OPJ_FLOAT32) as core::ffi::c_double)
              as OPJ_FLOAT32
              - l_offset
          }
          l_rates = l_rates.offset(1);
          k += 1;
        }
        l_tcp = l_tcp.offset(1);
        j += 1;
      }
      i += 1;
    }
    l_tcp = (*l_cp).tcps;
    i = 0 as OPJ_UINT32;
    while i < (*l_cp).th {
      j = 0 as OPJ_UINT32;
      while j < (*l_cp).tw {
        l_rates = (*l_tcp).rates.as_mut_ptr();
        if *l_rates > 0.0f32 {
          *l_rates -= l_sot_remove;
          if *l_rates < 30.0f32 {
            *l_rates = 30.0f32
          }
        }
        l_rates = l_rates.offset(1);
        l_last_res = (*l_tcp).numlayers.wrapping_sub(1u32);
        k = 1 as OPJ_UINT32;
        while k < l_last_res {
          if *l_rates > 0.0f32 {
            *l_rates -= l_sot_remove;
            if *l_rates < *l_rates.offset(-1) + 10.0f32 {
              *l_rates = *l_rates.offset(-1) + 20.0f32
            }
          }
          l_rates = l_rates.offset(1);
          k += 1;
        }
        if *l_rates > 0.0f32 {
          *l_rates -= l_sot_remove + 2.0f32;
          if *l_rates < *l_rates.offset(-1) + 10.0f32 {
            *l_rates = *l_rates.offset(-1) + 20.0f32
          }
        }
        l_tcp = l_tcp.offset(1);
        j += 1;
      }
      i += 1;
    }
    l_img_comp = (*l_image).comps;
    let mut l_tile_size = 0u64;
    i = 0 as OPJ_UINT32;
    while i < (*l_image).numcomps {
      l_tile_size += opj_uint_ceildiv((*l_cp).tdx, (*l_img_comp).dx) as u64
        * opj_uint_ceildiv((*l_cp).tdy, (*l_img_comp).dy) as u64
        * (*l_img_comp).prec as u64;
      l_img_comp = l_img_comp.offset(1);
      i += 1;
    }
    /* TODO: where does this magic value come from ? */
    /* This used to be 1.3 / 8, but with random data and very small code */
    /* block sizes, this is not enough. For example with */
    /* bin/test_tile_encoder 1 256 256 32 32 8 0 reversible_with_precinct.j2k 4 4 3 0 0 1 16 16 */
    /* TODO revise this to take into account the overhead linked to the */
    /* number of packets and number of code blocks in packets */
    l_tile_size = (l_tile_size as f64 * 1.4 / 8.0) as u64;

    /* Arbitrary amount to make the following work: */
    /* bin/test_tile_encoder 1 256 256 17 16 8 0 reversible_no_precinct.j2k 4 4 3 0 0 1 */
    l_tile_size += 500;

    l_tile_size += opj_j2k_get_specific_header_sizes(p_j2k) as u64;

    if l_tile_size > u32::MAX as u64 {
      l_tile_size = u32::MAX as u64;
    }

    p_j2k.m_specific_param.m_encoder.m_encoded_tile_size = l_tile_size as OPJ_UINT32;
    p_j2k.m_specific_param.m_encoder.m_encoded_tile_data =
      opj_malloc(p_j2k.m_specific_param.m_encoder.m_encoded_tile_size as size_t) as *mut OPJ_BYTE;
    if p_j2k
      .m_specific_param
      .m_encoder
      .m_encoded_tile_data
      .is_null()
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to allocate m_encoded_tile_data. %u MB required\n",
        l_tile_size.wrapping_div(1024u64).wrapping_div(1024u64) as OPJ_UINT32,
      );
      return 0i32;
    }
    if p_j2k.m_specific_param.m_encoder.m_TLM != 0 {
      p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_buffer = opj_malloc(
        (6u32).wrapping_mul(p_j2k.m_specific_param.m_encoder.m_total_tile_parts) as size_t,
      ) as *mut OPJ_BYTE;
      if p_j2k
        .m_specific_param
        .m_encoder
        .m_tlm_sot_offsets_buffer
        .is_null()
      {
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_current =
        p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_buffer
    }
    1i32
  }
}
/* *
 * Gets the offset of the header.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_get_end_header(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    /* preconditions */

    (*p_j2k.cstr_index).main_head_end = opj_stream_tell(p_stream);
    1i32
  }
}
/* *
 * Writes the CBD-MCT-MCC-MCO markers (Multi components transform)
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_write_mct_data_group(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_mcc_record = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
    let mut l_mct_record = core::ptr::null_mut::<opj_mct_data_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    /* preconditions */

    if opj_j2k_write_cbd(p_j2k, p_stream, p_manager) == 0 {
      return 0i32;
    }
    l_tcp = &mut *p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t;
    l_mct_record = (*l_tcp).m_mct_records;
    i = 0 as OPJ_UINT32;
    while i < (*l_tcp).m_nb_mct_records {
      if opj_j2k_write_mct_record(p_j2k, l_mct_record, p_stream, p_manager) == 0 {
        return 0i32;
      }
      l_mct_record = l_mct_record.offset(1);
      i += 1;
    }
    l_mcc_record = (*l_tcp).m_mcc_records;
    i = 0 as OPJ_UINT32;
    while i < (*l_tcp).m_nb_mcc_records {
      if opj_j2k_write_mcc_record(p_j2k, l_mcc_record, p_stream, p_manager) == 0 {
        return 0i32;
      }
      l_mcc_record = l_mcc_record.offset(1);
      i += 1;
    }
    if opj_j2k_write_mco(p_j2k, p_stream, p_manager) == 0 {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Writes COC marker for each component.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_all_coc(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut compno: OPJ_UINT32 = 0;
    /* preconditions */

    compno = 1 as OPJ_UINT32;
    while compno < (*p_j2k.m_private_image).numcomps {
      /* cod is first component of first tile */
      if opj_j2k_compare_coc(p_j2k, 0 as OPJ_UINT32, compno) == 0
        && opj_j2k_write_coc(p_j2k, compno, p_stream, p_manager) == 0
      {
        return 0i32;
      }
      compno += 1;
    }
    1i32
  }
}
/* *
 * Writes QCC marker for each component.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_all_qcc(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut compno: OPJ_UINT32 = 0;
    /* preconditions */

    compno = 1 as OPJ_UINT32;
    while compno < (*p_j2k.m_private_image).numcomps {
      /* qcd is first component of first tile */
      if opj_j2k_compare_qcc(p_j2k, 0 as OPJ_UINT32, compno) == 0
        && opj_j2k_write_qcc(p_j2k, compno, p_stream, p_manager) == 0
      {
        return 0i32;
      }
      compno += 1;
    }
    1i32
  }
}
/* *
 * Writes regions of interests.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_regions(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut compno: OPJ_UINT32 = 0;
    let mut l_tccp = core::ptr::null::<opj_tccp_t>();
    /* preconditions */

    l_tccp = (*p_j2k.m_cp.tcps).tccps;
    compno = 0 as OPJ_UINT32;
    while compno < (*p_j2k.m_private_image).numcomps {
      if (*l_tccp).roishift != 0
        && opj_j2k_write_rgn(
          p_j2k,
          0 as OPJ_UINT32,
          compno,
          (*p_j2k.m_private_image).numcomps,
          p_stream,
          p_manager,
        ) == 0
      {
        return 0i32;
      }
      l_tccp = l_tccp.offset(1);
      compno += 1;
    }
    1i32
  }
}
/* *
 * Writes EPC ????
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_epc(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cstr_index = core::ptr::null_mut::<opj_codestream_index_t>();
    /* preconditions */

    l_cstr_index = p_j2k.cstr_index;
    if !l_cstr_index.is_null() {
      (*l_cstr_index).codestream_size = opj_stream_tell(p_stream) as OPJ_UINT64;
      /* UniPG>> */
      /* The following adjustment is done to adjust the codestream size */
      /* if SOD is not at 0 in the buffer. Useful in case of JP2, where */
      /* the first bunch of bytes is not in the codestream              */
      (*l_cstr_index).codestream_size -= (*l_cstr_index).main_head_start as OPJ_UINT64;
      /* <<UniPG */
    }
    /* USE_JPWL */
    1i32
  }
}
/* *
 * Reads an unknown marker
 *
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_stream                the stream object to read from.
 * @param       output_marker           FIXME DOC
 * @param       p_manager               the user event manager.
 *
 * @return      true                    if the marker could be deduced.
*/
fn opj_j2k_read_unk(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut output_marker: &mut J2KMarker,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_unknown_marker: OPJ_UINT32 = 0;
    let mut l_size_unk = 2 as OPJ_UINT32;
    let mut l_marker_handler = J2KMarker::UNK(0);
    /* preconditions*/

    event_msg!(p_manager, EVT_WARNING, "Unknown marker\n",);
    loop {
      /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer*/
      if opj_stream_read_data(
        p_stream,
        p_j2k.m_specific_param.m_decoder.m_header_data,
        2 as OPJ_SIZE_T,
        p_manager,
      ) != 2
      {
        event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
        return 0i32;
      }
      /* read 2 bytes as the new marker ID*/
      opj_read_bytes(
        p_j2k.m_specific_param.m_decoder.m_header_data,
        &mut l_unknown_marker,
        2 as OPJ_UINT32,
      );
      if l_unknown_marker < 0xff00u32 {
        continue;
      }
      /* Get the marker handler from the marker ID*/
      l_marker_handler = J2KMarker::from(l_unknown_marker);
      if p_j2k.m_specific_param.m_decoder.m_state & l_marker_handler.states() == J2KState::NONE {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Marker is not compliant with its position\n",
        );
        return 0i32;
      } else if !l_marker_handler.is_unknown() {
        /* Add the marker to the codestream index*/
        if l_marker_handler != J2KMarker::SOT {
          let mut res = opj_j2k_add_mhmarker(
            p_j2k.cstr_index,
            J2KMarker::UNK(0),
            (opj_stream_tell(p_stream) as OPJ_UINT32).wrapping_sub(l_size_unk) as OPJ_OFF_T,
            l_size_unk,
          );
          if res == 0i32 {
            event_msg!(p_manager, EVT_ERROR, "Not enough memory to add mh marker\n",);
            return 0i32;
          }
        }
        break;
      /* next marker is known and well located */
      } else {
        l_size_unk = (l_size_unk as core::ffi::c_uint).wrapping_add(2u32) as OPJ_UINT32
      }
    }
    *output_marker = l_marker_handler;
    1i32
  }
}
/* *
 * Writes the MCT marker (Multiple Component Transform)
 *
 * @param       p_j2k           J2K codec.
 * @param       p_mct_record    FIXME DOC
 * @param       p_stream        the stream to write data to.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_write_mct_record(
  mut p_j2k: &mut opj_j2k,
  mut p_mct_record: *mut opj_mct_data_t,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_mct_size: OPJ_UINT32 = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_tmp: OPJ_UINT32 = 0;
    /* preconditions */
    /* Lmct */

    l_mct_size = (10u32).wrapping_add((*p_mct_record).m_data_size);
    if l_mct_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_mct_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write MCT marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_mct_size
    }
    l_current_data = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    opj_write_bytes(l_current_data, J2KMarker::MCT.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(
      l_current_data,
      l_mct_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(l_current_data, 0 as OPJ_UINT32, 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    /* only one marker atm */
    l_tmp = (*p_mct_record).m_index & 0xffu32
      | ((*p_mct_record).m_array_type as core::ffi::c_uint) << 8i32
      | ((*p_mct_record).m_element_type as core::ffi::c_uint) << 10i32; /* Ymct */
    opj_write_bytes(l_current_data, l_tmp, 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(l_current_data, 0 as OPJ_UINT32, 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    memcpy(
      l_current_data as *mut core::ffi::c_void,
      (*p_mct_record).m_data as *const core::ffi::c_void,
      (*p_mct_record).m_data_size as usize,
    );
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_mct_size as OPJ_SIZE_T,
      p_manager,
    ) != l_mct_size as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Reads a MCT marker (Multiple Component Transform)
 *
 * @param       p_header_data   the data contained in the MCT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCT marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a MCT marker (Multiple Component Transform)
 *
 * @param       p_header_data   the data contained in the MCT box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCT marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_mct(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tmp: OPJ_UINT32 = 0;
    let mut l_indix: OPJ_UINT32 = 0;
    let mut l_mct_data = core::ptr::null_mut::<opj_mct_data_t>();
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    if p_header_size < 2u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading MCT marker\n",);
      return 0i32;
    }
    /* first marker */
    opj_read_bytes(p_header_data, &mut l_tmp, 2 as OPJ_UINT32); /* Zmct */
    p_header_data = p_header_data.offset(2);
    if l_tmp != 0u32 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Cannot take in charge mct data within multiple MCT records\n",
      );
      return 1i32;
    }
    if p_header_size <= 6u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading MCT marker\n",);
      return 0i32;
    }
    /* Imct -> no need for other values, take the first, type is double with decorrelation x0000 1101 0000 0000*/
    opj_read_bytes(p_header_data, &mut l_tmp, 2 as OPJ_UINT32); /* Imct */
    p_header_data = p_header_data.offset(2);
    l_indix = l_tmp & 0xffu32;
    l_mct_data = (*l_tcp).m_mct_records;
    i = 0 as OPJ_UINT32;
    while i < (*l_tcp).m_nb_mct_records {
      if (*l_mct_data).m_index == l_indix {
        break;
      }
      l_mct_data = l_mct_data.offset(1);
      i += 1;
    }
    /* NOT FOUND */
    if i == (*l_tcp).m_nb_mct_records {
      if (*l_tcp).m_nb_mct_records == (*l_tcp).m_nb_max_mct_records {
        let mut new_mct_records = core::ptr::null_mut::<opj_mct_data_t>();
        (*l_tcp).m_nb_max_mct_records =
          ((*l_tcp).m_nb_max_mct_records as core::ffi::c_uint).wrapping_add(10u32) as OPJ_UINT32;
        new_mct_records = opj_realloc(
          (*l_tcp).m_mct_records as *mut core::ffi::c_void,
          ((*l_tcp).m_nb_max_mct_records as usize)
            .wrapping_mul(core::mem::size_of::<opj_mct_data_t>()),
        ) as *mut opj_mct_data_t;
        if new_mct_records.is_null() {
          opj_free((*l_tcp).m_mct_records as *mut core::ffi::c_void);
          (*l_tcp).m_mct_records = core::ptr::null_mut::<opj_mct_data_t>();
          (*l_tcp).m_nb_max_mct_records = 0 as OPJ_UINT32;
          (*l_tcp).m_nb_mct_records = 0 as OPJ_UINT32;
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Not enough memory to read MCT marker\n",
          );
          return 0i32;
        }
        /* Update m_mcc_records[].m_offset_array and m_decorrelation_array
         * to point to the new addresses */
        if new_mct_records != (*l_tcp).m_mct_records {
          i = 0 as OPJ_UINT32; /* Ymct */
          while i < (*l_tcp).m_nb_mcc_records {
            let mut l_mcc_record: *mut opj_simple_mcc_decorrelation_data_t =
              &mut *(*l_tcp).m_mcc_records.offset(i as isize)
                as *mut opj_simple_mcc_decorrelation_data_t;
            if !(*l_mcc_record).m_decorrelation_array.is_null() {
              (*l_mcc_record).m_decorrelation_array = new_mct_records.offset(
                (*l_mcc_record)
                  .m_decorrelation_array
                  .offset_from((*l_tcp).m_mct_records) as isize,
              )
            }
            if !(*l_mcc_record).m_offset_array.is_null() {
              (*l_mcc_record).m_offset_array = new_mct_records.offset(
                (*l_mcc_record)
                  .m_offset_array
                  .offset_from((*l_tcp).m_mct_records) as isize,
              )
            }
            i += 1;
          }
        }
        (*l_tcp).m_mct_records = new_mct_records;
        l_mct_data = (*l_tcp)
          .m_mct_records
          .offset((*l_tcp).m_nb_mct_records as isize);
        memset(
          l_mct_data as *mut core::ffi::c_void,
          0i32,
          ((*l_tcp)
            .m_nb_max_mct_records
            .wrapping_sub((*l_tcp).m_nb_mct_records) as usize)
            .wrapping_mul(core::mem::size_of::<opj_mct_data_t>()),
        );
      }
      l_mct_data = (*l_tcp)
        .m_mct_records
        .offset((*l_tcp).m_nb_mct_records as isize);
      (*l_tcp).m_nb_mct_records = (*l_tcp).m_nb_mct_records.wrapping_add(1)
    }
    if !(*l_mct_data).m_data.is_null() {
      opj_free((*l_mct_data).m_data as *mut core::ffi::c_void);
      (*l_mct_data).m_data = core::ptr::null_mut::<OPJ_BYTE>();
      (*l_mct_data).m_data_size = 0 as OPJ_UINT32
    }
    (*l_mct_data).m_index = l_indix;
    (*l_mct_data).m_array_type = (l_tmp >> 8i32 & 3u32) as J2K_MCT_ARRAY_TYPE;
    (*l_mct_data).m_element_type = MCTElementType::new(l_tmp >> 10i32 & 3u32);
    opj_read_bytes(p_header_data, &mut l_tmp, 2 as OPJ_UINT32);
    p_header_data = p_header_data.offset(2);
    if l_tmp != 0u32 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Cannot take in charge multiple MCT markers\n",
      );
      return 1i32;
    }
    p_header_size = (p_header_size as core::ffi::c_uint).wrapping_sub(6u32) as OPJ_UINT32;
    (*l_mct_data).m_data = opj_malloc(p_header_size as size_t) as *mut OPJ_BYTE;
    if (*l_mct_data).m_data.is_null() {
      event_msg!(p_manager, EVT_ERROR, "Error reading MCT marker\n",);
      return 0i32;
    }
    memcpy(
      (*l_mct_data).m_data as *mut core::ffi::c_void,
      p_header_data as *const core::ffi::c_void,
      p_header_size as usize,
    );
    (*l_mct_data).m_data_size = p_header_size;
    1i32
  }
}
/* *
 * Writes the MCC marker (Multiple Component Collection)
 *
 * @param       p_j2k                   J2K codec.
 * @param       p_mcc_record            FIXME DOC
 * @param       p_stream                the stream to write data to.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_mcc_record(
  mut p_j2k: &mut opj_j2k,
  mut p_mcc_record: *mut opj_simple_mcc_decorrelation_data,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_mcc_size: OPJ_UINT32 = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_nb_bytes_for_comp: OPJ_UINT32 = 0;
    let mut l_mask: OPJ_UINT32 = 0;
    let mut l_tmcc: OPJ_UINT32 = 0;
    /* preconditions */
    /* Lmcc */

    if (*p_mcc_record).m_nb_comps > 255u32 {
      l_nb_bytes_for_comp = 2 as OPJ_UINT32;
      l_mask = 0x8000 as OPJ_UINT32
    } else {
      l_nb_bytes_for_comp = 1 as OPJ_UINT32;
      l_mask = 0 as OPJ_UINT32
    }
    l_mcc_size = (*p_mcc_record)
      .m_nb_comps
      .wrapping_mul(2u32)
      .wrapping_mul(l_nb_bytes_for_comp)
      .wrapping_add(19u32);
    if l_mcc_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_mcc_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write MCC marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_mcc_size
    }
    l_current_data = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    opj_write_bytes(l_current_data, J2KMarker::MCC.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(
      l_current_data,
      l_mcc_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    /* first marker */
    opj_write_bytes(l_current_data, 0 as OPJ_UINT32, 2 as OPJ_UINT32); /* Zmcc */
    l_current_data = l_current_data.offset(2); /* Imcc -> no need for other values, take the first */
    opj_write_bytes(l_current_data, (*p_mcc_record).m_index, 1 as OPJ_UINT32);
    l_current_data = l_current_data.offset(1);
    /* only one marker atm */
    opj_write_bytes(l_current_data, 0 as OPJ_UINT32, 2 as OPJ_UINT32); /* Ymcc */
    l_current_data = l_current_data.offset(2); /* Qmcc -> number of collections -> 1 */
    opj_write_bytes(l_current_data, 1 as OPJ_UINT32, 2 as OPJ_UINT32); /* Xmcci type of component transformation -> array based decorrelation */
    l_current_data = l_current_data.offset(2); /* Nmcci number of input components involved and size for each component offset = 8 bits */
    opj_write_bytes(l_current_data, 0x1 as OPJ_UINT32, 1 as OPJ_UINT32); /* Cmccij Component offset*/
    l_current_data = l_current_data.offset(1); /* Mmcci number of output components involved and size for each component offset = 8 bits */
    opj_write_bytes(
      l_current_data,
      (*p_mcc_record).m_nb_comps | l_mask,
      2 as OPJ_UINT32,
    ); /* Wmccij Component offset*/
    l_current_data = l_current_data.offset(2); /* Tmcci : use MCT defined as number 1 and irreversible array based. */
    i = 0 as OPJ_UINT32;
    while i < (*p_mcc_record).m_nb_comps {
      opj_write_bytes(l_current_data, i, l_nb_bytes_for_comp);
      l_current_data = l_current_data.offset(l_nb_bytes_for_comp as isize);
      i += 1;
    }
    opj_write_bytes(
      l_current_data,
      (*p_mcc_record).m_nb_comps | l_mask,
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    i = 0 as OPJ_UINT32;
    while i < (*p_mcc_record).m_nb_comps {
      opj_write_bytes(l_current_data, i, l_nb_bytes_for_comp);
      l_current_data = l_current_data.offset(l_nb_bytes_for_comp as isize);
      i += 1;
    }
    l_tmcc = ((!(*p_mcc_record).m_is_irreversible) as core::ffi::c_uint & 1u32) << 16i32;
    if !(*p_mcc_record).m_decorrelation_array.is_null() {
      l_tmcc |= (*(*p_mcc_record).m_decorrelation_array).m_index
    }
    if !(*p_mcc_record).m_offset_array.is_null() {
      l_tmcc |= (*(*p_mcc_record).m_offset_array).m_index << 8i32
    }
    opj_write_bytes(l_current_data, l_tmcc, 3 as OPJ_UINT32);
    l_current_data = l_current_data.offset(3);
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_mcc_size as OPJ_SIZE_T,
      p_manager,
    ) != l_mcc_size as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Reads a MCC marker (Multiple Component Collection)
 *
 * @param       p_header_data   the data contained in the MCC box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCC marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_mcc(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut l_tmp: OPJ_UINT32 = 0;
    let mut l_indix: OPJ_UINT32 = 0;
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_mcc_record = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
    let mut l_mct_data = core::ptr::null_mut::<opj_mct_data_t>();
    let mut l_nb_collections: OPJ_UINT32 = 0;
    let mut l_nb_comps: OPJ_UINT32 = 0;
    let mut l_nb_bytes_by_comp: OPJ_UINT32 = 0;
    let mut l_new_mcc = 0i32;
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    if p_header_size < 2u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading MCC marker\n",);
      return 0i32;
    }
    /* first marker */
    opj_read_bytes(p_header_data, &mut l_tmp, 2 as OPJ_UINT32); /* Zmcc */
    p_header_data = p_header_data.offset(2); /* Imcc -> no need for other values, take the first */
    if l_tmp != 0u32 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Cannot take in charge multiple data spanning\n",
      );
      return 1i32;
    }
    if p_header_size < 7u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading MCC marker\n",);
      return 0i32;
    }
    opj_read_bytes(p_header_data, &mut l_indix, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    l_mcc_record = (*l_tcp).m_mcc_records;
    i = 0 as OPJ_UINT32;
    while i < (*l_tcp).m_nb_mcc_records {
      if (*l_mcc_record).m_index == l_indix {
        break;
      }
      l_mcc_record = l_mcc_record.offset(1);
      i += 1;
    }
    /* * NOT FOUND */
    if i == (*l_tcp).m_nb_mcc_records {
      if (*l_tcp).m_nb_mcc_records == (*l_tcp).m_nb_max_mcc_records {
        let mut new_mcc_records = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
        (*l_tcp).m_nb_max_mcc_records =
          ((*l_tcp).m_nb_max_mcc_records as core::ffi::c_uint).wrapping_add(10u32) as OPJ_UINT32;
        new_mcc_records = opj_realloc(
          (*l_tcp).m_mcc_records as *mut core::ffi::c_void,
          ((*l_tcp).m_nb_max_mcc_records as usize)
            .wrapping_mul(core::mem::size_of::<opj_simple_mcc_decorrelation_data_t>()),
        ) as *mut opj_simple_mcc_decorrelation_data_t;
        if new_mcc_records.is_null() {
          opj_free((*l_tcp).m_mcc_records as *mut core::ffi::c_void);
          (*l_tcp).m_mcc_records = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
          (*l_tcp).m_nb_max_mcc_records = 0 as OPJ_UINT32;
          (*l_tcp).m_nb_mcc_records = 0 as OPJ_UINT32;
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Not enough memory to read MCC marker\n",
          );
          return 0i32;
        }
        (*l_tcp).m_mcc_records = new_mcc_records;
        l_mcc_record = (*l_tcp)
          .m_mcc_records
          .offset((*l_tcp).m_nb_mcc_records as isize);
        memset(
          l_mcc_record as *mut core::ffi::c_void,
          0i32,
          ((*l_tcp)
            .m_nb_max_mcc_records
            .wrapping_sub((*l_tcp).m_nb_mcc_records) as usize)
            .wrapping_mul(core::mem::size_of::<opj_simple_mcc_decorrelation_data_t>()),
        );
      }
      l_mcc_record = (*l_tcp)
        .m_mcc_records
        .offset((*l_tcp).m_nb_mcc_records as isize);
      l_new_mcc = 1i32
    }
    (*l_mcc_record).m_index = l_indix;
    /* only one marker atm */
    opj_read_bytes(p_header_data, &mut l_tmp, 2 as OPJ_UINT32); /* Ymcc */
    p_header_data = p_header_data.offset(2); /* Qmcc -> number of collections -> 1 */
    if l_tmp != 0u32 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Cannot take in charge multiple data spanning\n",
      ); /* Xmcci type of component transformation -> array based decorrelation */
      return 1i32;
    } /* Cmccij Component offset*/
    opj_read_bytes(p_header_data, &mut l_nb_collections, 2 as OPJ_UINT32); /* Wmccij Component offset*/
    p_header_data = p_header_data.offset(2); /* Wmccij Component offset*/
    if l_nb_collections > 1u32 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Cannot take in charge multiple collections\n",
      );
      return 1i32;
    }
    p_header_size = (p_header_size as core::ffi::c_uint).wrapping_sub(7u32) as OPJ_UINT32;
    i = 0 as OPJ_UINT32;
    while i < l_nb_collections {
      if p_header_size < 3u32 {
        event_msg!(p_manager, EVT_ERROR, "Error reading MCC marker\n",);
        return 0i32;
      }
      opj_read_bytes(p_header_data, &mut l_tmp, 1 as OPJ_UINT32);
      p_header_data = p_header_data.offset(1);
      if l_tmp != 1u32 {
        event_msg!(
          p_manager,
          EVT_WARNING,
          "Cannot take in charge collections other than array decorrelation\n",
        );
        return 1i32;
      }
      opj_read_bytes(p_header_data, &mut l_nb_comps, 2 as OPJ_UINT32);
      p_header_data = p_header_data.offset(2);
      p_header_size = (p_header_size as core::ffi::c_uint).wrapping_sub(3u32) as OPJ_UINT32;
      l_nb_bytes_by_comp = (1u32).wrapping_add(l_nb_comps >> 15i32);
      (*l_mcc_record).m_nb_comps = l_nb_comps & 0x7fffu32;
      if p_header_size
        < l_nb_bytes_by_comp
          .wrapping_mul((*l_mcc_record).m_nb_comps)
          .wrapping_add(2u32)
      {
        event_msg!(p_manager, EVT_ERROR, "Error reading MCC marker\n",);
        return 0i32;
      }
      p_header_size = (p_header_size as core::ffi::c_uint).wrapping_sub(
        l_nb_bytes_by_comp
          .wrapping_mul((*l_mcc_record).m_nb_comps)
          .wrapping_add(2u32),
      ) as OPJ_UINT32;
      j = 0 as OPJ_UINT32;
      while j < (*l_mcc_record).m_nb_comps {
        opj_read_bytes(p_header_data, &mut l_tmp, l_nb_bytes_by_comp);
        p_header_data = p_header_data.offset(l_nb_bytes_by_comp as isize);
        if l_tmp != j {
          event_msg!(
            p_manager,
            EVT_WARNING,
            "Cannot take in charge collections with indix shuffle\n",
          );
          return 1i32;
        }
        j += 1;
      }
      opj_read_bytes(p_header_data, &mut l_nb_comps, 2 as OPJ_UINT32);
      p_header_data = p_header_data.offset(2);
      l_nb_bytes_by_comp = (1u32).wrapping_add(l_nb_comps >> 15i32);
      l_nb_comps &= 0x7fffu32;
      if l_nb_comps != (*l_mcc_record).m_nb_comps {
        event_msg!(
          p_manager,
          EVT_WARNING,
          "Cannot take in charge collections without same number of indixes\n",
        );
        return 1i32;
      }
      if p_header_size
        < l_nb_bytes_by_comp
          .wrapping_mul((*l_mcc_record).m_nb_comps)
          .wrapping_add(3u32)
      {
        event_msg!(p_manager, EVT_ERROR, "Error reading MCC marker\n",);
        return 0i32;
      }
      p_header_size = (p_header_size as core::ffi::c_uint).wrapping_sub(
        l_nb_bytes_by_comp
          .wrapping_mul((*l_mcc_record).m_nb_comps)
          .wrapping_add(3u32),
      ) as OPJ_UINT32;
      j = 0 as OPJ_UINT32;
      while j < (*l_mcc_record).m_nb_comps {
        opj_read_bytes(p_header_data, &mut l_tmp, l_nb_bytes_by_comp);
        p_header_data = p_header_data.offset(l_nb_bytes_by_comp as isize);
        if l_tmp != j {
          event_msg!(
            p_manager,
            EVT_WARNING,
            "Cannot take in charge collections with indix shuffle\n",
          );
          return 1i32;
        }
        j += 1;
      }
      opj_read_bytes(p_header_data, &mut l_tmp, 3 as OPJ_UINT32);
      p_header_data = p_header_data.offset(3);
      (*l_mcc_record).m_is_irreversible = l_tmp >> 16i32 & 1u32 == 0;
      (*l_mcc_record).m_decorrelation_array = core::ptr::null_mut::<opj_mct_data_t>();
      (*l_mcc_record).m_offset_array = core::ptr::null_mut::<opj_mct_data_t>();
      l_indix = l_tmp & 0xffu32;
      if l_indix != 0u32 {
        l_mct_data = (*l_tcp).m_mct_records;
        j = 0 as OPJ_UINT32;
        while j < (*l_tcp).m_nb_mct_records {
          if (*l_mct_data).m_index == l_indix {
            (*l_mcc_record).m_decorrelation_array = l_mct_data;
            break;
          } else {
            l_mct_data = l_mct_data.offset(1);
            j += 1;
          }
        }
        if (*l_mcc_record).m_decorrelation_array.is_null() {
          event_msg!(p_manager, EVT_ERROR, "Error reading MCC marker\n",);
          return 0i32;
        }
      }
      l_indix = l_tmp >> 8i32 & 0xffu32;
      if l_indix != 0u32 {
        l_mct_data = (*l_tcp).m_mct_records;
        j = 0 as OPJ_UINT32;
        while j < (*l_tcp).m_nb_mct_records {
          if (*l_mct_data).m_index == l_indix {
            (*l_mcc_record).m_offset_array = l_mct_data;
            break;
          } else {
            l_mct_data = l_mct_data.offset(1);
            j += 1;
          }
        }
        if (*l_mcc_record).m_offset_array.is_null() {
          event_msg!(p_manager, EVT_ERROR, "Error reading MCC marker\n",);
          return 0i32;
        }
      }
      i += 1;
    }
    if p_header_size != 0u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading MCC marker\n",);
      return 0i32;
    }
    if l_new_mcc != 0 {
      (*l_tcp).m_nb_mcc_records = (*l_tcp).m_nb_mcc_records.wrapping_add(1)
    }
    1i32
  }
}
/* *
 * Writes the MCO marker (Multiple component transformation ordering)
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_mco(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_mco_size: OPJ_UINT32 = 0;
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_mcc_record = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
    let mut i: OPJ_UINT32 = 0;
    /* preconditions */
    /* Lmco */
    /* Imco -> use the mcc indicated by 1*/
    l_tcp = &mut *p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t;
    l_mco_size = (5u32).wrapping_add((*l_tcp).m_nb_mcc_records);
    if l_mco_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_mco_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write MCO marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_mco_size
    }
    l_current_data = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    opj_write_bytes(l_current_data, J2KMarker::MCO.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(
      l_current_data,
      l_mco_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(l_current_data, (*l_tcp).m_nb_mcc_records, 1 as OPJ_UINT32);
    l_current_data = l_current_data.offset(1);
    l_mcc_record = (*l_tcp).m_mcc_records;
    i = 0 as OPJ_UINT32;
    while i < (*l_tcp).m_nb_mcc_records {
      opj_write_bytes(l_current_data, (*l_mcc_record).m_index, 1 as OPJ_UINT32);
      l_current_data = l_current_data.offset(1);
      l_mcc_record = l_mcc_record.offset(1);
      i += 1;
    }
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_mco_size as OPJ_SIZE_T,
      p_manager,
    ) != l_mco_size as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Reads a MCO marker (Multiple Component Transform Ordering)
 *
 * @param       p_header_data   the data contained in the MCO box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCO marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a MCO marker (Multiple Component Transform Ordering)
 *
 * @param       p_header_data   the data contained in the MCO box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the MCO marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_mco(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_tmp: OPJ_UINT32 = 0;
    let mut i: OPJ_UINT32 = 0;
    let mut l_nb_stages: OPJ_UINT32 = 0;
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_image = p_j2k.m_private_image;
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    if p_header_size < 1u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading MCO marker\n",);
      return 0i32;
    }
    opj_read_bytes(p_header_data, &mut l_nb_stages, 1 as OPJ_UINT32);
    p_header_data = p_header_data.offset(1);
    if l_nb_stages > 1u32 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Cannot take in charge multiple transformation stages.\n",
      );
      return 1i32;
    }
    if p_header_size != l_nb_stages.wrapping_add(1u32) {
      event_msg!(p_manager, EVT_WARNING, "Error reading MCO marker\n",);
      return 0i32;
    }
    l_tccp = (*l_tcp).tccps;
    i = 0 as OPJ_UINT32;
    while i < (*l_image).numcomps {
      (*l_tccp).m_dc_level_shift = 0i32;
      l_tccp = l_tccp.offset(1);
      i += 1;
    }
    if !(*l_tcp).m_mct_decoding_matrix.is_null() {
      opj_free((*l_tcp).m_mct_decoding_matrix as *mut core::ffi::c_void);
      (*l_tcp).m_mct_decoding_matrix = core::ptr::null_mut::<OPJ_FLOAT32>()
    }
    i = 0 as OPJ_UINT32;
    while i < l_nb_stages {
      opj_read_bytes(p_header_data, &mut l_tmp, 1 as OPJ_UINT32);
      p_header_data = p_header_data.offset(1);
      if opj_j2k_add_mct(l_tcp, &mut *p_j2k.m_private_image, l_tmp) == 0 {
        return 0i32;
      }
      i += 1;
    }
    1i32
  }
}
fn opj_j2k_add_mct(
  mut p_tcp: *mut opj_tcp_t,
  mut p_image: &mut opj_image,
  mut p_index: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_mcc_record = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
    let mut l_deco_array = core::ptr::null_mut::<opj_mct_data_t>();
    let mut l_offset_array = core::ptr::null_mut::<opj_mct_data_t>();
    let mut l_data_size: OPJ_UINT32 = 0;
    let mut l_mct_size: OPJ_UINT32 = 0;
    let mut l_offset_size: OPJ_UINT32 = 0;
    let mut l_nb_elem: OPJ_UINT32 = 0;
    let mut l_offset_data = core::ptr::null_mut::<OPJ_UINT32>();
    let mut l_current_offset_data = core::ptr::null_mut::<OPJ_UINT32>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    /* preconditions */
    assert!(!p_tcp.is_null());
    l_mcc_record = (*p_tcp).m_mcc_records;
    i = 0 as OPJ_UINT32;
    while i < (*p_tcp).m_nb_mcc_records {
      if (*l_mcc_record).m_index == p_index {
        break;
      }
      i += 1;
    }
    if i == (*p_tcp).m_nb_mcc_records {
      /* * element discarded **/
      return 1i32;
    }
    if (*l_mcc_record).m_nb_comps != p_image.numcomps {
      /* * do not support number of comps != image */
      return 1i32;
    }
    l_deco_array = (*l_mcc_record).m_decorrelation_array;
    if !l_deco_array.is_null() {
      l_data_size = (*l_deco_array)
        .m_element_type
        .size()
        .wrapping_mul(p_image.numcomps)
        .wrapping_mul(p_image.numcomps);
      if (*l_deco_array).m_data_size != l_data_size {
        return 0i32;
      }
      l_nb_elem = p_image.numcomps.wrapping_mul(p_image.numcomps);
      l_mct_size = l_nb_elem.wrapping_mul(core::mem::size_of::<OPJ_FLOAT32>() as OPJ_UINT32);
      (*p_tcp).m_mct_decoding_matrix = opj_malloc(l_mct_size as size_t) as *mut OPJ_FLOAT32;
      if (*p_tcp).m_mct_decoding_matrix.is_null() {
        return 0i32;
      }
      (*l_deco_array).m_element_type.read_to_float(
        (*l_deco_array).m_data as *const core::ffi::c_void,
        (*p_tcp).m_mct_decoding_matrix as *mut core::ffi::c_void,
        l_nb_elem,
      );
    }
    l_offset_array = (*l_mcc_record).m_offset_array;
    if !l_offset_array.is_null() {
      l_data_size = (*l_offset_array)
        .m_element_type
        .size()
        .wrapping_mul(p_image.numcomps);
      if (*l_offset_array).m_data_size != l_data_size {
        return 0i32;
      }
      l_nb_elem = p_image.numcomps;
      l_offset_size = l_nb_elem.wrapping_mul(core::mem::size_of::<OPJ_UINT32>() as OPJ_UINT32);
      l_offset_data = opj_malloc(l_offset_size as size_t) as *mut OPJ_UINT32;
      if l_offset_data.is_null() {
        return 0i32;
      }
      (*l_offset_array).m_element_type.read_to_int32(
        (*l_offset_array).m_data as *const core::ffi::c_void,
        l_offset_data as *mut core::ffi::c_void,
        l_nb_elem,
      );
      l_tccp = (*p_tcp).tccps;
      l_current_offset_data = l_offset_data;
      i = 0 as OPJ_UINT32;
      while i < p_image.numcomps {
        let fresh22 = l_current_offset_data;
        l_current_offset_data = l_current_offset_data.offset(1);
        (*l_tccp).m_dc_level_shift = *fresh22 as OPJ_INT32;
        l_tccp = l_tccp.offset(1);
        i += 1;
      }
      opj_free(l_offset_data as *mut core::ffi::c_void);
    }
    1i32
  }
}
/* *
 * Writes the CBD marker (Component bit depth definition)
 *
 * @param       p_stream                                the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_cbd(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_cbd_size: OPJ_UINT32 = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut l_comp = core::ptr::null_mut::<opj_image_comp_t>();
    /* preconditions */
    /* L_CBD */
    /* Component bit depth */
    l_image = p_j2k.m_private_image;
    l_cbd_size = (6u32).wrapping_add((*p_j2k.m_private_image).numcomps);
    if l_cbd_size > p_j2k.m_specific_param.m_encoder.m_header_tile_data_size {
      let mut new_header_tile_data = opj_realloc(
        p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void,
        l_cbd_size as size_t,
      ) as *mut OPJ_BYTE;
      if new_header_tile_data.is_null() {
        opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
        p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
        p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to write CBD marker\n",
        );
        return 0i32;
      }
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = new_header_tile_data;
      p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = l_cbd_size
    }
    l_current_data = p_j2k.m_specific_param.m_encoder.m_header_tile_data;
    opj_write_bytes(l_current_data, J2KMarker::CBD.as_u32(), 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(
      l_current_data,
      l_cbd_size.wrapping_sub(2u32),
      2 as OPJ_UINT32,
    );
    l_current_data = l_current_data.offset(2);
    opj_write_bytes(l_current_data, (*l_image).numcomps, 2 as OPJ_UINT32);
    l_current_data = l_current_data.offset(2);
    l_comp = (*l_image).comps;
    i = 0 as OPJ_UINT32;
    while i < (*l_image).numcomps {
      opj_write_bytes(
        l_current_data,
        (*l_comp).sgnd << 7i32 | (*l_comp).prec.wrapping_sub(1u32),
        1 as OPJ_UINT32,
      );
      l_current_data = l_current_data.offset(1);
      l_comp = l_comp.offset(1);
      i += 1;
    }
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_header_tile_data,
      l_cbd_size as OPJ_SIZE_T,
      p_manager,
    ) != l_cbd_size as usize
    {
      return 0i32;
    }
    1i32
  }
}
/* *
 * Reads a CBD marker (Component bit depth definition)
 * @param       p_header_data   the data contained in the CBD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CBD marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a CBD marker (Component bit depth definition)
 * @param       p_header_data   the data contained in the CBD box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CBD marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_cbd(
  mut p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_nb_comp: OPJ_UINT32 = 0;
    let mut l_num_comp: OPJ_UINT32 = 0;
    let mut l_comp_def: OPJ_UINT32 = 0;
    let mut i: OPJ_UINT32 = 0;
    let mut l_comp = core::ptr::null_mut::<opj_image_comp_t>();
    /* preconditions */
    /* Component bit depth */

    assert!(!p_header_data.is_null());
    l_num_comp = (*p_j2k.m_private_image).numcomps;
    if p_header_size != (*p_j2k.m_private_image).numcomps.wrapping_add(2u32) {
      event_msg!(p_manager, EVT_ERROR, "Crror reading CBD marker\n",);
      return 0i32;
    }
    opj_read_bytes(p_header_data, &mut l_nb_comp, 2 as OPJ_UINT32);
    p_header_data = p_header_data.offset(2);
    if l_nb_comp != l_num_comp {
      event_msg!(p_manager, EVT_ERROR, "Crror reading CBD marker\n",);
      return 0i32;
    }
    l_comp = (*p_j2k.m_private_image).comps;
    i = 0 as OPJ_UINT32;
    while i < l_num_comp {
      opj_read_bytes(p_header_data, &mut l_comp_def, 1 as OPJ_UINT32);
      p_header_data = p_header_data.offset(1);
      (*l_comp).sgnd = l_comp_def >> 7i32 & 1u32;
      (*l_comp).prec = (l_comp_def & 0x7fu32).wrapping_add(1u32);
      if (*l_comp).prec > 31u32 {
        event_msg!(p_manager, EVT_ERROR,
                          "Invalid values for comp = %d : prec=%u (should be between 1 and 38 according to the JPEG2000 norm. OpenJpeg only supports up to 31)\n", i,
                          (*l_comp).prec);
        return 0i32;
      }
      l_comp = l_comp.offset(1);
      i += 1;
    }
    1i32
  }
}
/* *
 * Reads a CAP marker (extended capabilities definition). Empty implementation.
 * Found in HTJ2K files
 *
 * @param       p_header_data   the data contained in the CAP box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CAP marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a CAP marker (extended capabilities definition). Empty implementation.
 * Found in HTJ2K files.
 *
 * @param       p_header_data   the data contained in the CAP box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CAP marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_cap(
  mut _p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut _p_header_size: OPJ_UINT32,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* preconditions */

  assert!(!p_header_data.is_null());
  1i32
}

/* *
 * Reads a CPF marker (corresponding profile). Empty implementation. Found in HTJ2K files
 * @param       p_header_data   the data contained in the CPF box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CPF marker.
 * @param       p_manager               the user event manager.
*/
/* *
 * Reads a CPF marker (corresponding profile). Empty implementation. Found in HTJ2K files
 * @param       p_header_data   the data contained in the CPF box.
 * @param       p_j2k                   the jpeg2000 codec.
 * @param       p_header_size   the size of the data contained in the CPF marker.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_read_cpf(
  mut _p_j2k: &mut opj_j2k,
  mut p_header_data: *mut OPJ_BYTE,
  mut _p_header_size: OPJ_UINT32,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* preconditions */

  assert!(!p_header_data.is_null());
  1i32
}

/* ----------------------------------------------------------------------- */
/* J2K / JPT decoder interface                                             */
/* ----------------------------------------------------------------------- */
pub(crate) fn opj_j2k_setup_decoder(mut j2k: &mut opj_j2k, mut parameters: &mut opj_dparameters_t) {
  j2k.m_cp.m_specific_param.m_dec.m_layer = parameters.cp_layer;
  j2k.m_cp.m_specific_param.m_dec.m_reduce = parameters.cp_reduce;
  j2k.dump_state = parameters.flags & 0x2u32
}

pub(crate) fn opj_j2k_decoder_set_strict_mode(mut j2k: &mut opj_j2k, mut strict: OPJ_BOOL) {
  j2k.m_cp.strict = strict
}

pub(crate) fn opj_j2k_set_threads(
  mut _j2k: &mut opj_j2k,
  mut _num_threads: OPJ_UINT32,
) -> OPJ_BOOL {
  0i32
}

/* ----------------------------------------------------------------------- */
/* J2K encoder interface                                                       */
/* ----------------------------------------------------------------------- */
pub(crate) fn opj_j2k_create_compress() -> Option<opj_j2k> {
  let mut l_j2k = opj_j2k::new(0);
  l_j2k.m_cp.m_is_decoder = false;
  unsafe {
    l_j2k.m_specific_param.m_encoder.m_header_tile_data =
      opj_malloc(1000i32 as size_t) as *mut OPJ_BYTE;
    if l_j2k
      .m_specific_param
      .m_encoder
      .m_header_tile_data
      .is_null()
    {
      return None;
    }
    l_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 1000 as OPJ_UINT32;
  }
  Some(l_j2k)
}

fn opj_j2k_initialise_4K_poc(
  mut POC: *mut opj_poc_t,
  mut numres: core::ffi::c_int,
) -> core::ffi::c_int {
  unsafe {
    (*POC.offset(0)).tile = 1 as OPJ_UINT32;
    (*POC.offset(0)).resno0 = 0 as OPJ_UINT32;
    (*POC.offset(0)).compno0 = 0 as OPJ_UINT32;
    (*POC.offset(0)).layno1 = 1 as OPJ_UINT32;
    (*POC.offset(0)).resno1 = (numres - 1i32) as OPJ_UINT32;
    (*POC.offset(0)).compno1 = 3 as OPJ_UINT32;
    (*POC.offset(0)).prg1 = OPJ_CPRL;
    (*POC.offset(1)).tile = 1 as OPJ_UINT32;
    (*POC.offset(1)).resno0 = (numres - 1i32) as OPJ_UINT32;
    (*POC.offset(1)).compno0 = 0 as OPJ_UINT32;
    (*POC.offset(1)).layno1 = 1 as OPJ_UINT32;
    (*POC.offset(1)).resno1 = numres as OPJ_UINT32;
    (*POC.offset(1)).compno1 = 3 as OPJ_UINT32;
    (*POC.offset(1)).prg1 = OPJ_CPRL;
    2i32
  }
}
fn opj_j2k_set_cinema_parameters(
  mut parameters: &mut opj_cparameters_t,
  mut image: &mut opj_image,
  mut p_manager: &mut opj_event_mgr,
) {
  unsafe {
    /* Configure cinema parameters */
    let mut i: core::ffi::c_int = 0;
    /* No tiling */
    parameters.tile_size_on = 0i32;
    parameters.cp_tdx = 1i32;
    parameters.cp_tdy = 1i32;
    /* One tile part for each component */
    parameters.tp_flag = 'C' as i32 as core::ffi::c_char;
    parameters.tp_on = 1 as core::ffi::c_char;
    /* Tile and Image shall be at (0,0) */
    parameters.cp_tx0 = 0i32;
    parameters.cp_ty0 = 0i32;
    parameters.image_offset_x0 = 0i32;
    parameters.image_offset_y0 = 0i32;
    /* Codeblock size= 32*32 */
    parameters.cblockw_init = 32i32;
    parameters.cblockh_init = 32i32;
    /* Codeblock style: no mode switch enabled */
    parameters.mode = 0i32;
    /* No ROI */
    parameters.roi_compno = -(1i32);
    /* No subsampling */
    parameters.subsampling_dx = 1i32;
    parameters.subsampling_dy = 1i32;
    /* 9-7 transform */
    parameters.irreversible = 1i32;
    /* Number of layers */
    if parameters.tcp_numlayers > 1i32 {
      event_msg!(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\n1 single quality layer-> Number of layers forced to 1 (rather than %d)\n-> Rate of the last layer (%3.1f) will be used",
                      parameters.tcp_numlayers,
                      parameters.tcp_rates[(parameters.tcp_numlayers -
                                                   1i32) as usize]
                          as core::ffi::c_double);
      parameters.tcp_rates[0_usize] =
        parameters.tcp_rates[(parameters.tcp_numlayers - 1i32) as usize];
      parameters.tcp_numlayers = 1i32
    }
    /* Resolution levels */
    match parameters.rsiz as core::ffi::c_int {
      3 => {
        if parameters.numresolution > 6i32 {
          event_msg!(p_manager, EVT_WARNING,
                              "JPEG 2000 Profile-3 (2k dc profile) requires:\nNumber of decomposition levels <= 5\n-> Number of decomposition levels forced to 5 (rather than %d)\n",
                              parameters.numresolution + 1i32);
          parameters.numresolution = 6i32
        }
      }
      4 => {
        if parameters.numresolution < 2i32 {
          event_msg!(p_manager, EVT_WARNING,
                              "JPEG 2000 Profile-4 (4k dc profile) requires:\nNumber of decomposition levels >= 1 && <= 6\n-> Number of decomposition levels forced to 1 (rather than %d)\n",
                              parameters.numresolution + 1i32);
          parameters.numresolution = 1i32
        } else if parameters.numresolution > 7i32 {
          event_msg!(p_manager, EVT_WARNING,
                              "JPEG 2000 Profile-4 (4k dc profile) requires:\nNumber of decomposition levels >= 1 && <= 6\n-> Number of decomposition levels forced to 6 (rather than %d)\n",
                              parameters.numresolution + 1i32);
          parameters.numresolution = 7i32
        }
      }
      _ => {}
    }
    /* Precincts */
    parameters.csty |= 0x1i32;
    if parameters.numresolution == 1i32 {
      parameters.res_spec = 1i32;
      parameters.prcw_init[0_usize] = 128i32;
      parameters.prch_init[0_usize] = 128i32
    } else {
      parameters.res_spec = parameters.numresolution - 1i32;
      i = 0i32;
      while i < parameters.res_spec {
        parameters.prcw_init[i as usize] = 256i32;
        parameters.prch_init[i as usize] = 256i32;
        i += 1
      }
    }
    /* The progression order shall be CPRL */
    parameters.prog_order = OPJ_CPRL;
    /* Progression order changes for 4K, disallowed for 2K */
    if parameters.rsiz as core::ffi::c_int == 0x4i32 {
      parameters.numpocs =
        opj_j2k_initialise_4K_poc(parameters.POC.as_mut_ptr(), parameters.numresolution)
          as OPJ_UINT32
    } else {
      parameters.numpocs = 0 as OPJ_UINT32
    }
    /* Limited bit-rate */
    parameters.cp_disto_alloc = 1i32;
    if parameters.max_cs_size <= 0i32 {
      /* No rate has been introduced, 24 fps is assumed */
      parameters.max_cs_size = 1302083i32;
      event_msg!(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\nMaximum 1302083 compressed bytes @ 24fps\nAs no rate has been given, this limit will be used.\n");
    } else if parameters.max_cs_size > 1302083i32 {
      event_msg!(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\nMaximum 1302083 compressed bytes @ 24fps\n-> Specified rate exceeds this limit. Rate will be forced to 1302083 bytes.\n");
      parameters.max_cs_size = 1302083i32
    }
    if parameters.max_comp_size <= 0i32 {
      /* No rate has been introduced, 24 fps is assumed */
      parameters.max_comp_size = 1041666i32;
      event_msg!(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\nMaximum 1041666 compressed bytes @ 24fps\nAs no rate has been given, this limit will be used.\n");
    } else if parameters.max_comp_size > 1041666i32 {
      event_msg!(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 and 4 (2k/4k dc profile) requires:\nMaximum 1041666 compressed bytes @ 24fps\n-> Specified rate exceeds this limit. Rate will be forced to 1041666 bytes.\n");
      parameters.max_comp_size = 1041666i32
    }
    parameters.tcp_rates[0_usize] = image
      .numcomps
      .wrapping_mul((*image.comps.offset(0)).w)
      .wrapping_mul((*image.comps.offset(0)).h)
      .wrapping_mul((*image.comps.offset(0)).prec)
      as OPJ_FLOAT32
      / (parameters.max_cs_size as OPJ_UINT32)
        .wrapping_mul(8u32)
        .wrapping_mul((*image.comps.offset(0)).dx)
        .wrapping_mul((*image.comps.offset(0)).dy) as OPJ_FLOAT32;
  }
}
fn opj_j2k_is_cinema_compliant(
  mut image: &mut opj_image,
  mut rsiz: OPJ_UINT16,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    /* Number of components */
    if image.numcomps != 3u32 {
      event_msg!(p_manager, EVT_WARNING,
                      "JPEG 2000 Profile-3 (2k dc profile) requires:\n3 components-> Number of components of input image (%d) is not compliant\n-> Non-profile-3 codestream will be generated\n",
                      image.numcomps);
      return 0i32;
    }
    /* Bitdepth */
    i = 0 as OPJ_UINT32;
    while i < image.numcomps {
      if ((*image.comps.offset(i as isize)).prec != 12u32) as core::ffi::c_uint
        | (*image.comps.offset(i as isize)).sgnd
        != 0
      {
        let tmp_str = if (*image.comps.offset(i as isize)).sgnd != 0 {
          "signed"
        } else {
          "unsigned"
        };
        event_msg!(p_manager, EVT_WARNING,
                          "JPEG 2000 Profile-3 (2k dc profile) requires:\nPrecision of each component shall be 12 bits unsigned-> At least component %d of input image (%d bits, %s) is not compliant\n-> Non-profile-3 codestream will be generated\n", i,
                          (*image.comps.offset(i as isize)).prec, tmp_str);
        return 0i32;
      }
      i += 1;
    }
    /* Image size */
    match rsiz as core::ffi::c_int {
      3 => {
        if ((*image.comps.offset(0)).w > 2048u32) as core::ffi::c_int
          | ((*image.comps.offset(0)).h > 1080u32) as core::ffi::c_int
          != 0
        {
          event_msg!(p_manager, EVT_WARNING,
                              "JPEG 2000 Profile-3 (2k dc profile) requires:\nwidth <= 2048 and height <= 1080\n-> Input image size %d x %d is not compliant\n-> Non-profile-3 codestream will be generated\n",
                              (*image.comps.offset(0i32 as
                                                          isize)).w,
                              (*image.comps.offset(0i32 as
                                                          isize)).h);
          return 0i32;
        }
      }
      4 => {
        if ((*image.comps.offset(0)).w > 4096u32) as core::ffi::c_int
          | ((*image.comps.offset(0)).h > 2160u32) as core::ffi::c_int
          != 0
        {
          event_msg!(p_manager, EVT_WARNING,
                              "JPEG 2000 Profile-4 (4k dc profile) requires:\nwidth <= 4096 and height <= 2160\n-> Image size %d x %d is not compliant\n-> Non-profile-4 codestream will be generated\n",
                              (*image.comps.offset(0i32 as
                                                          isize)).w,
                              (*image.comps.offset(0i32 as
                                                          isize)).h);
          return 0i32;
        }
      }
      _ => {}
    }
    1i32
  }
}
fn opj_j2k_get_imf_max_NL(
  mut parameters: &mut opj_cparameters_t,
  mut image: &mut opj_image,
) -> core::ffi::c_int {
  /* Decomposition levels */
  let rsiz = parameters.rsiz;
  let profile = (rsiz as core::ffi::c_int & 0xff00i32) as OPJ_UINT16;
  let XTsiz = if parameters.tile_size_on != 0 {
    parameters.cp_tdx as OPJ_UINT32
  } else {
    image.x1
  };
  match profile as core::ffi::c_int {
    1024 => return 5i32,
    1280 => return 6i32,
    1536 => return 7i32,
    1792 => {
      if XTsiz >= 2048u32 {
        return 5i32;
      } else if XTsiz >= 1024u32 {
        return 4i32;
      }
    }
    2048 => {
      if XTsiz >= 4096u32 {
        return 6i32;
      } else if XTsiz >= 2048u32 {
        return 5i32;
      } else if XTsiz >= 1024u32 {
        return 4i32;
      }
    }
    2304 => {
      if XTsiz >= 8192u32 {
        return 7i32;
      } else if XTsiz >= 4096u32 {
        return 6i32;
      } else if XTsiz >= 2048u32 {
        return 5i32;
      } else if XTsiz >= 1024u32 {
        return 4i32;
      }
    }
    _ => {}
  }
  -(1i32)
}
fn opj_j2k_set_imf_parameters(
  mut parameters: &mut opj_cparameters_t,
  mut image: &mut opj_image,
  mut _p_manager: &mut opj_event_mgr,
) {
  let rsiz = parameters.rsiz;
  let profile = (rsiz as core::ffi::c_int & 0xff00i32) as OPJ_UINT16;
  /* Override defaults set by opj_set_default_encoder_parameters */
  if parameters.cblockw_init == 64i32 && parameters.cblockh_init == 64i32 {
    parameters.cblockw_init = 32i32;
    parameters.cblockh_init = 32i32
  }
  /* One tile part for each component */
  parameters.tp_flag = 'C' as i32 as core::ffi::c_char;
  parameters.tp_on = 1 as core::ffi::c_char;
  if parameters.prog_order as core::ffi::c_int == OPJ_LRCP as core::ffi::c_int {
    parameters.prog_order = OPJ_CPRL
  }
  if profile as core::ffi::c_int == 0x400i32
    || profile as core::ffi::c_int == 0x500i32
    || profile as core::ffi::c_int == 0x600i32
  {
    /* 9-7 transform */
    parameters.irreversible = 1i32
  }
  /* Adjust the number of resolutions if set to its defaults */
  if parameters.numresolution == 6i32 && image.x0 == 0u32 && image.y0 == 0u32 {
    let max_NL = opj_j2k_get_imf_max_NL(parameters, image);
    if max_NL >= 0i32 && parameters.numresolution > max_NL {
      parameters.numresolution = max_NL + 1i32
    }
    /* Note: below is generic logic */
    if parameters.tile_size_on == 0 {
      while parameters.numresolution > 0i32 {
        if image.x1 < (1u32) << (parameters.numresolution as OPJ_UINT32).wrapping_sub(1u32) {
          parameters.numresolution -= 1
        } else {
          if image.y1 >= (1u32) << (parameters.numresolution as OPJ_UINT32).wrapping_sub(1u32) {
            break;
          }
          parameters.numresolution -= 1
        }
      }
    }
  }
  /* Set defaults precincts */
  if parameters.csty == 0i32 {
    parameters.csty |= 0x1i32;
    if parameters.numresolution == 1i32 {
      parameters.res_spec = 1i32;
      parameters.prcw_init[0_usize] = 128i32;
      parameters.prch_init[0_usize] = 128i32
    } else {
      let mut i: core::ffi::c_int = 0;
      parameters.res_spec = parameters.numresolution - 1i32;
      i = 0i32;
      while i < parameters.res_spec {
        parameters.prcw_init[i as usize] = 256i32;
        parameters.prch_init[i as usize] = 256i32;
        i += 1
      }
    }
  };
}
/* Table A.53 from JPEG2000 standard */
static mut tabMaxSubLevelFromMainLevel: [OPJ_UINT16; 12] = [
  15 as OPJ_UINT16,
  1 as OPJ_UINT16,
  1 as OPJ_UINT16,
  1 as OPJ_UINT16,
  2 as OPJ_UINT16,
  3 as OPJ_UINT16,
  4 as OPJ_UINT16,
  5 as OPJ_UINT16,
  6 as OPJ_UINT16,
  7 as OPJ_UINT16,
  8 as OPJ_UINT16,
  9 as OPJ_UINT16,
];
fn opj_j2k_is_imf_compliant(
  mut parameters: &mut opj_cparameters_t,
  mut image: &mut opj_image,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let rsiz = parameters.rsiz;
    let profile = (rsiz as core::ffi::c_int & 0xff00i32) as OPJ_UINT16;
    let mainlevel = (rsiz as core::ffi::c_int & 0xfi32) as OPJ_UINT16;
    let sublevel = (rsiz as core::ffi::c_int >> 4i32 & 0xfi32) as OPJ_UINT16;
    let NL = parameters.numresolution - 1i32;
    let XTsiz = if parameters.tile_size_on != 0 {
      parameters.cp_tdx as OPJ_UINT32
    } else {
      image.x1
    };
    let mut ret = 1i32;
    /* Validate mainlevel */
    if mainlevel as core::ffi::c_int > 11i32 {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF profile require mainlevel <= 11.\n-> %d is thus not compliant\n-> Non-IMF codestream will be generated\n",
                      mainlevel as core::ffi::c_int);
      ret = 0i32
    } else {
      /* Validate sublevel */
      assert!(
        core::mem::size_of::<[OPJ_UINT16; 12]>()
          == ((11i32 + 1i32) as usize).wrapping_mul(core::mem::size_of::<OPJ_UINT16>())
      );
      if sublevel as core::ffi::c_int
        > tabMaxSubLevelFromMainLevel[mainlevel as usize] as core::ffi::c_int
      {
        event_msg!(p_manager, EVT_WARNING,
                          "IMF profile require sublevel <= %d for mainlevel = %d.\n-> %d is thus not compliant\n-> Non-IMF codestream will be generated\n",
                          tabMaxSubLevelFromMainLevel[mainlevel as usize] as
                              core::ffi::c_int, mainlevel as core::ffi::c_int,
                          sublevel as core::ffi::c_int);
        ret = 0i32
      }
    }
    /* Number of components */
    if image.numcomps > 3u32 {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF profiles require at most 3 components.\n-> Number of components of input image (%d) is not compliant\n-> Non-IMF codestream will be generated\n",
                      image.numcomps);
      ret = 0i32
    }
    if image.x0 != 0u32 || image.y0 != 0u32 {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF profiles require image origin to be at 0,0.\n-> %d,%d is not compliant\n-> Non-IMF codestream will be generated\n", image.x0,
                      (image.y0 != 0u32) as
                          core::ffi::c_int);
      ret = 0i32
    }
    if parameters.cp_tx0 != 0i32 || parameters.cp_ty0 != 0i32 {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF profiles require tile origin to be at 0,0.\n-> %d,%d is not compliant\n-> Non-IMF codestream will be generated\n",
                      parameters.cp_tx0, parameters.cp_ty0);
      ret = 0i32
    }
    if parameters.tile_size_on != 0 {
      if profile as core::ffi::c_int == 0x400i32
        || profile as core::ffi::c_int == 0x500i32
        || profile as core::ffi::c_int == 0x600i32
      {
        if (parameters.cp_tdx as OPJ_UINT32) < image.x1
          || (parameters.cp_tdy as OPJ_UINT32) < image.y1
        {
          event_msg!(p_manager, EVT_WARNING,
                              "IMF 2K/4K/8K single tile profiles require tile to be greater or equal to image size.\n-> %d,%d is lesser than %d,%d\n-> Non-IMF codestream will be generated\n",
                              parameters.cp_tdx, parameters.cp_tdy,
                              image.x1, image.y1);
          ret = 0i32
        }
      } else if !(parameters.cp_tdx as OPJ_UINT32 >= image.x1
        && parameters.cp_tdy as OPJ_UINT32 >= image.y1)
        && !(parameters.cp_tdx == 1024i32 && parameters.cp_tdy == 1024i32)
        && !(parameters.cp_tdx == 2048i32
          && parameters.cp_tdy == 2048i32
          && (profile as core::ffi::c_int == 0x500i32 || profile as core::ffi::c_int == 0x600i32))
        && !(parameters.cp_tdx == 4096i32
          && parameters.cp_tdy == 4096i32
          && profile as core::ffi::c_int == 0x600i32)
      {
        event_msg!(p_manager, EVT_WARNING,
                                "IMF 2K_R/4K_R/8K_R single/multiple tile profiles require tile to be greater or equal to image size,\nor to be (1024,1024), or (2048,2048) for 4K_R/8K_R or (4096,4096) for 8K_R.\n-> %d,%d is non conformant\n-> Non-IMF codestream will be generated\n",
                                parameters.cp_tdx,
                                parameters.cp_tdy);
        ret = 0i32
      }
    }
    /* Bitdepth */
    i = 0 as OPJ_UINT32;
    while i < image.numcomps {
      if !((*image.comps.offset(i as isize)).prec >= 8u32
        && (*image.comps.offset(i as isize)).prec <= 16u32)
        || (*image.comps.offset(i as isize)).sgnd != 0
      {
        let tmp_str = if (*image.comps.offset(i as isize)).sgnd != 0 {
          "signed"
        } else {
          "unsigned"
        };
        event_msg!(p_manager, EVT_WARNING,
                          "IMF profiles require precision of each component to b in [8-16] bits unsigned-> At least component %d of input image (%d bits, %s) is not compliant\n-> Non-IMF codestream will be generated\n", i,
                          (*image.comps.offset(i as isize)).prec, tmp_str);
        ret = 0i32
      }
      i += 1;
    }
    /* Sub-sampling */
    i = 0 as OPJ_UINT32;
    while i < image.numcomps {
      if i == 0u32 && (*image.comps.offset(i as isize)).dx != 1u32 {
        event_msg!(p_manager, EVT_WARNING,
                          "IMF profiles require XRSiz1 == 1. Here it is set to %d.\n-> Non-IMF codestream will be generated\n",
                          (*image.comps.offset(i as isize)).dx);
        ret = 0i32
      }
      if i == 1u32
        && (*image.comps.offset(i as isize)).dx != 1u32
        && (*image.comps.offset(i as isize)).dx != 2u32
      {
        event_msg!(p_manager, EVT_WARNING,
                          "IMF profiles require XRSiz2 == 1 or 2. Here it is set to %d.\n-> Non-IMF codestream will be generated\n",
                          (*image.comps.offset(i as isize)).dx);
        ret = 0i32
      }
      if i > 1u32
        && (*image.comps.offset(i as isize)).dx
          != (*image.comps.offset(i.wrapping_sub(1u32) as isize)).dx
      {
        event_msg!(p_manager, EVT_WARNING,
                          "IMF profiles require XRSiz%d to be the same as XRSiz2. Here it is set to %d instead of %d.\n-> Non-IMF codestream will be generated\n",
                          i.wrapping_add(1u32),
                          (*image.comps.offset(i as isize)).dx,
                          (*image.comps.offset(i.wrapping_sub(1 as
                                                                     core::ffi::c_int
                                                                     as
                                                                     core::ffi::c_uint)
                                                      as isize)).dx);
        ret = 0i32
      }
      if (*image.comps.offset(i as isize)).dy != 1u32 {
        event_msg!(p_manager, EVT_WARNING,
                          "IMF profiles require YRsiz == 1. Here it is set to %d for component %d.\n-> Non-IMF codestream will be generated\n",
                          (*image.comps.offset(i as isize)).dy, i);
        ret = 0i32
      }
      i += 1;
    }
    /* Image size */
    match profile as core::ffi::c_int {
      1024 | 1792 => {
        if ((*image.comps.offset(0)).w > 2048u32) as core::ffi::c_int
          | ((*image.comps.offset(0)).h > 1556u32) as core::ffi::c_int
          != 0
        {
          event_msg!(p_manager, EVT_WARNING,
                              "IMF 2K/2K_R profile require:\nwidth <= 2048 and height <= 1556\n-> Input image size %d x %d is not compliant\n-> Non-IMF codestream will be generated\n",
                              (*image.comps.offset(0i32 as
                                                          isize)).w,
                              (*image.comps.offset(0i32 as
                                                          isize)).h);
          ret = 0i32
        }
      }
      1280 | 2048 => {
        if ((*image.comps.offset(0)).w > 4096u32) as core::ffi::c_int
          | ((*image.comps.offset(0)).h > 3112u32) as core::ffi::c_int
          != 0
        {
          event_msg!(p_manager, EVT_WARNING,
                              "IMF 4K/4K_R profile require:\nwidth <= 4096 and height <= 3112\n-> Input image size %d x %d is not compliant\n-> Non-IMF codestream will be generated\n",
                              (*image.comps.offset(0i32 as
                                                          isize)).w,
                              (*image.comps.offset(0i32 as
                                                          isize)).h);
          ret = 0i32
        }
      }
      1536 | 2304 => {
        if ((*image.comps.offset(0)).w > 8192u32) as core::ffi::c_int
          | ((*image.comps.offset(0)).h > 6224u32) as core::ffi::c_int
          != 0
        {
          event_msg!(p_manager, EVT_WARNING,
                              "IMF 8K/8K_R profile require:\nwidth <= 8192 and height <= 6224\n-> Input image size %d x %d is not compliant\n-> Non-IMF codestream will be generated\n",
                              (*image.comps.offset(0i32 as
                                                          isize)).w,
                              (*image.comps.offset(0i32 as
                                                          isize)).h);
          ret = 0i32
        }
      }
      _ => {
        panic!("Unknown OPJ_PROFILE");
        //C: assert(0);
      }
    }
    if parameters.roi_compno != -(1i32) {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF profile forbid RGN / region of interest marker.\n-> Compression parameters specify a ROI\n-> Non-IMF codestream will be generated\n");
      ret = 0i32
    }
    if parameters.cblockw_init != 32i32 || parameters.cblockh_init != 32i32 {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF profile require code block size to be 32x32.\n-> Compression parameters set it to %dx%d.\n-> Non-IMF codestream will be generated\n",
                      parameters.cblockw_init, parameters.cblockh_init);
      ret = 0i32
    }
    if parameters.prog_order as core::ffi::c_int != OPJ_CPRL as core::ffi::c_int {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF profile require progression order to be CPRL.\n-> Compression parameters set it to %d.\n-> Non-IMF codestream will be generated\n",
                      parameters.prog_order as core::ffi::c_int);
      ret = 0i32
    }
    if parameters.numpocs != 0u32 {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF profile forbid POC markers.\n-> Compression parameters set %d POC.\n-> Non-IMF codestream will be generated\n",
                      parameters.numpocs);
      ret = 0i32
    }
    /* Codeblock style: no mode switch enabled */
    if parameters.mode != 0i32 {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF profile forbid mode switch in code block style.\n-> Compression parameters set code block style to %d.\n-> Non-IMF codestream will be generated\n",
                      parameters.mode);
      ret = 0i32
    }
    if profile as core::ffi::c_int == 0x400i32
      || profile as core::ffi::c_int == 0x500i32
      || profile as core::ffi::c_int == 0x600i32
    {
      /* Expect 9-7 transform */
      if parameters.irreversible != 1i32 {
        event_msg!(p_manager, EVT_WARNING,
                          "IMF 2K/4K/8K profiles require 9-7 Irreversible Transform.\n-> Compression parameters set it to reversible.\n-> Non-IMF codestream will be generated\n");
        ret = 0i32
      }
    } else if parameters.irreversible != 0i32 {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF 2K/4K/8K profiles require 5-3 reversible Transform.\n-> Compression parameters set it to irreversible.\n-> Non-IMF codestream will be generated\n");
      ret = 0i32
    }
    /* Expect 5-3 transform */
    /* Number of layers */
    if parameters.tcp_numlayers != 1i32 {
      event_msg!(p_manager, EVT_WARNING,
                      "IMF 2K/4K/8K profiles require 1 single quality layer.\n-> Number of layers is %d.\n-> Non-IMF codestream will be generated\n",
                      parameters.tcp_numlayers);
      ret = 0i32
    }
    /* Decomposition levels */
    match profile as core::ffi::c_int {
      1024 => {
        if !(1i32..=5i32).contains(&NL) {
          event_msg!(p_manager, EVT_WARNING,
                              "IMF 2K profile requires 1 <= NL <= 5:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n", NL);
          ret = 0i32
        }
      }
      1280 => {
        if !(1i32..=6i32).contains(&NL) {
          event_msg!(p_manager, EVT_WARNING,
                              "IMF 4K profile requires 1 <= NL <= 6:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n", NL);
          ret = 0i32
        }
      }
      1536 => {
        if !(1i32..=7i32).contains(&NL) {
          event_msg!(p_manager, EVT_WARNING,
                              "IMF 8K profile requires 1 <= NL <= 7:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n", NL);
          ret = 0i32
        }
      }
      1792 => {
        if XTsiz >= 2048u32 {
          if !(1i32..=5i32).contains(&NL) {
            event_msg!(p_manager, EVT_WARNING,
                                  "IMF 2K_R profile requires 1 <= NL <= 5 for XTsiz >= 2048:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n",
                                  NL);
            ret = 0i32
          }
        } else if XTsiz >= 1024u32 && !(1i32..=4i32).contains(&NL) {
          event_msg!(p_manager, EVT_WARNING,
                                "IMF 2K_R profile requires 1 <= NL <= 4 for XTsiz in [1024,2048[:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n",
                                NL);
          ret = 0i32
        }
      }
      2048 => {
        if XTsiz >= 4096u32 {
          if !(1i32..=6i32).contains(&NL) {
            event_msg!(p_manager, EVT_WARNING,
                                  "IMF 4K_R profile requires 1 <= NL <= 6 for XTsiz >= 4096:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n",
                                  NL);
            ret = 0i32
          }
        } else if XTsiz >= 2048u32 {
          if !(1i32..=5i32).contains(&NL) {
            event_msg!(p_manager, EVT_WARNING,
                                  "IMF 4K_R profile requires 1 <= NL <= 5 for XTsiz in [2048,4096[:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n",
                                  NL);
            ret = 0i32
          }
        } else if XTsiz >= 1024u32 && !(1i32..=4i32).contains(&NL) {
          event_msg!(p_manager, EVT_WARNING,
                                "IMF 4K_R profile requires 1 <= NL <= 4 for XTsiz in [1024,2048[:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n",
                                NL);
          ret = 0i32
        }
      }
      2304 => {
        if XTsiz >= 8192u32 {
          if !(1i32..=7i32).contains(&NL) {
            event_msg!(p_manager, EVT_WARNING,
                                  "IMF 4K_R profile requires 1 <= NL <= 7 for XTsiz >= 8192:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n",
                                  NL);
            ret = 0i32
          }
        } else if XTsiz >= 4096u32 {
          if !(1i32..=6i32).contains(&NL) {
            event_msg!(p_manager, EVT_WARNING,
                                  "IMF 4K_R profile requires 1 <= NL <= 6 for XTsiz in [4096,8192[:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n",
                                  NL);
            ret = 0i32
          }
        } else if XTsiz >= 2048u32 {
          if !(1i32..=5i32).contains(&NL) {
            event_msg!(p_manager, EVT_WARNING,
                                  "IMF 4K_R profile requires 1 <= NL <= 5 for XTsiz in [2048,4096[:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n",
                                  NL);
            ret = 0i32
          }
        } else if XTsiz >= 1024u32 && !(1i32..=4i32).contains(&NL) {
          event_msg!(p_manager, EVT_WARNING,
                                "IMF 4K_R profile requires 1 <= NL <= 4 for XTsiz in [1024,2048[:\n-> Number of decomposition levels is %d.\n-> Non-IMF codestream will be generated\n",
                                NL);
          ret = 0i32
        }
      }
      _ => {}
    }
    if parameters.numresolution == 1i32 {
      if parameters.res_spec != 1i32
        || parameters.prcw_init[0_usize] != 128i32
        || parameters.prch_init[0_usize] != 128i32
      {
        event_msg!(p_manager, EVT_WARNING,
                          "IMF profiles require PPx = PPy = 7 for NLLL band, else 8.\n-> Supplied values are different from that.\n-> Non-IMF codestream will be generated\n");
        ret = 0i32
      }
    } else {
      let mut i_0: core::ffi::c_int = 0;
      i_0 = 0i32;
      while i_0 < parameters.res_spec {
        if parameters.prcw_init[i_0 as usize] != 256i32
          || parameters.prch_init[i_0 as usize] != 256i32
        {
          event_msg!(p_manager, EVT_WARNING,
                              "IMF profiles require PPx = PPy = 7 for NLLL band, else 8.\n-> Supplied values are different from that.\n-> Non-IMF codestream will be generated\n");
          ret = 0i32
        }
        i_0 += 1
      }
    }
    ret
  }
}

pub(crate) fn opj_j2k_setup_encoder(
  mut p_j2k: &mut opj_j2k,
  mut parameters: &mut opj_cparameters_t,
  mut image: &mut opj_image,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut tileno: OPJ_UINT32 = 0;
    let mut numpocs_tile: OPJ_UINT32 = 0;
    let mut cp = core::ptr::null_mut::<opj_cp_t>();
    let mut cblkw: OPJ_UINT32 = 0;
    let mut cblkh: OPJ_UINT32 = 0;
    if parameters.numresolution <= 0i32 || parameters.numresolution > 33i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid number of resolutions : %d not in range [1,%d]\n",
        parameters.numresolution,
        33i32,
      );
      return 0i32;
    }
    if parameters.cblockw_init < 4i32 || parameters.cblockw_init > 1024i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid value for cblockw_init: %d not a power of 2 in range [4,1024]\n",
        parameters.cblockw_init,
      );
      return 0i32;
    }
    if parameters.cblockh_init < 4i32 || parameters.cblockh_init > 1024i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid value for cblockh_init: %d not a power of 2 not in range [4,1024]\n",
        parameters.cblockh_init,
      );
      return 0i32;
    }
    if parameters.cblockw_init * parameters.cblockh_init > 4096i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid value for cblockw_init * cblockh_init: should be <= 4096\n",
      );
      return 0i32;
    }
    cblkw = opj_int_floorlog2(parameters.cblockw_init) as OPJ_UINT32;
    cblkh = opj_int_floorlog2(parameters.cblockh_init) as OPJ_UINT32;
    if parameters.cblockw_init != (1i32) << cblkw {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid value for cblockw_init: %d not a power of 2 in range [4,1024]\n",
        parameters.cblockw_init,
      );
      return 0i32;
    }
    if parameters.cblockh_init != (1i32) << cblkh {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid value for cblockw_init: %d not a power of 2 in range [4,1024]\n",
        parameters.cblockh_init,
      );
      return 0i32;
    }

    if parameters.cp_fixed_alloc != 0 {
      if parameters.cp_matrice.is_null() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "cp_fixed_alloc set, but cp_matrice missing\n"
        );
        return 0;
      }

      if parameters.tcp_numlayers > j2k::J2K_TCD_MATRIX_MAX_LAYER_COUNT {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "tcp_numlayers when cp_fixed_alloc set should not exceed %d\n",
          j2k::J2K_TCD_MATRIX_MAX_LAYER_COUNT
        );
        return 0;
      }
      if parameters.numresolution > j2k::J2K_TCD_MATRIX_MAX_RESOLUTION_COUNT {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "numresolution when cp_fixed_alloc set should not exceed %d\n",
          j2k::J2K_TCD_MATRIX_MAX_RESOLUTION_COUNT
        );
        return 0;
      }
    }

    p_j2k.m_specific_param.m_encoder.m_nb_comps = image.numcomps;
    /* keep a link to cp so that we can destroy it later in j2k_destroy_compress */
    cp = &mut p_j2k.m_cp;
    /* set default values for cp */
    (*cp).tw = 1 as OPJ_UINT32;
    (*cp).th = 1 as OPJ_UINT32;
    /* FIXME ADE: to be removed once deprecated cp_cinema and cp_rsiz have been removed */
    if parameters.rsiz as core::ffi::c_int == 0i32 {
      /* consider deprecated fields only if RSIZ has not been set */
      let mut deprecated_used = 0i32;
      match parameters.cp_cinema as core::ffi::c_uint {
        1 => {
          parameters.rsiz = 0x3 as OPJ_UINT16;
          parameters.max_cs_size = 1302083i32;
          parameters.max_comp_size = 1041666i32;
          deprecated_used = 1i32
        }
        2 => {
          parameters.rsiz = 0x3 as OPJ_UINT16;
          parameters.max_cs_size = 651041i32;
          parameters.max_comp_size = 520833i32;
          deprecated_used = 1i32
        }
        3 => {
          parameters.rsiz = 0x4 as OPJ_UINT16;
          parameters.max_cs_size = 1302083i32;
          parameters.max_comp_size = 1041666i32;
          deprecated_used = 1i32
        }
        0 | _ => {}
      }
      match parameters.cp_rsiz as core::ffi::c_uint {
        3 => {
          parameters.rsiz = 0x3 as OPJ_UINT16;
          deprecated_used = 1i32
        }
        4 => {
          parameters.rsiz = 0x4 as OPJ_UINT16;
          deprecated_used = 1i32
        }
        33024 => {
          parameters.rsiz = (0x8000i32 | 0x100i32) as OPJ_UINT16;
          deprecated_used = 1i32
        }
        0 | _ => {}
      }
      if deprecated_used != 0 {
        event_msg!(p_manager, EVT_WARNING,
                          "Deprecated fields cp_cinema or cp_rsiz are used\nPlease consider using only the rsiz field\nSee openjpeg.h documentation for more details\n");
      }
    }
    /* If no explicit layers are provided, use lossless settings */
    if parameters.tcp_numlayers == 0i32 {
      parameters.tcp_numlayers = 1i32;
      parameters.cp_disto_alloc = 1i32;
      parameters.tcp_rates[0_usize] = 0 as core::ffi::c_float
    }
    if parameters.cp_disto_alloc != 0 {
      /* Emit warnings if tcp_rates are not decreasing */
      i = 1 as OPJ_UINT32;
      while i < parameters.tcp_numlayers as OPJ_UINT32 {
        let mut rate_i_corr = parameters.tcp_rates[i as usize];
        let mut rate_i_m_1_corr = parameters.tcp_rates[i.wrapping_sub(1u32) as usize];
        if rate_i_corr as core::ffi::c_double <= 1.0f64 {
          rate_i_corr = 1.0f64 as OPJ_FLOAT32
        }
        if rate_i_m_1_corr as core::ffi::c_double <= 1.0f64 {
          rate_i_m_1_corr = 1.0f64 as OPJ_FLOAT32
        }
        if rate_i_corr >= rate_i_m_1_corr {
          if rate_i_corr != parameters.tcp_rates[i as usize]
            && rate_i_m_1_corr != parameters.tcp_rates[i.wrapping_sub(1u32) as usize]
          {
            event_msg!(p_manager, EVT_WARNING,
                                  "tcp_rates[%d]=%f (corrected as %f) should be strictly lesser than tcp_rates[%d]=%f (corrected as %f)\n", i,
                                  parameters.tcp_rates[i as usize] as
                                      core::ffi::c_double,
                                  rate_i_corr as core::ffi::c_double,
                                  i.wrapping_sub(1i32 as
                                                     core::ffi::c_uint),
                                  parameters.tcp_rates[i.wrapping_sub(1 as
                                                                             core::ffi::c_int
                                                                             as
                                                                             core::ffi::c_uint)
                                                              as usize] as
                                      core::ffi::c_double,
                                  rate_i_m_1_corr as core::ffi::c_double);
          } else if rate_i_corr != parameters.tcp_rates[i as usize] {
            event_msg!(
            p_manager,
            EVT_WARNING,
            "tcp_rates[%d]=%f (corrected as %f) should be strictly lesser than tcp_rates[%d]=%f\n",
            i,
            parameters.tcp_rates[i as usize] as core::ffi::c_double,
            rate_i_corr as core::ffi::c_double,
            i.wrapping_sub(1i32 as core::ffi::c_uint),
            parameters.tcp_rates
              [i.wrapping_sub(1 as core::ffi::c_int as core::ffi::c_uint) as usize]
              as core::ffi::c_double
          );
          } else if rate_i_m_1_corr != parameters.tcp_rates[i.wrapping_sub(1u32) as usize] {
            event_msg!(
            p_manager,
            EVT_WARNING,
            "tcp_rates[%d]=%f should be strictly lesser than tcp_rates[%d]=%f (corrected as %f)\n",
            i,
            parameters.tcp_rates[i as usize] as core::ffi::c_double,
            i.wrapping_sub(1i32 as core::ffi::c_uint),
            parameters.tcp_rates
              [i.wrapping_sub(1 as core::ffi::c_int as core::ffi::c_uint) as usize]
              as core::ffi::c_double,
            rate_i_m_1_corr as core::ffi::c_double
          );
          } else {
            event_msg!(
              p_manager,
              EVT_WARNING,
              "tcp_rates[%d]=%f should be strictly lesser than tcp_rates[%d]=%f\n",
              i,
              parameters.tcp_rates[i as usize] as core::ffi::c_double,
              i.wrapping_sub(1u32),
              parameters.tcp_rates[i.wrapping_sub(1u32) as usize] as core::ffi::c_double,
            );
          }
        }
        i += 1;
      }
    } else if parameters.cp_fixed_quality != 0 {
      /* Emit warnings if tcp_distoratio are not increasing */
      i = 1 as OPJ_UINT32;
      while i < parameters.tcp_numlayers as OPJ_UINT32 {
        if parameters.tcp_distoratio[i as usize]
          < parameters.tcp_distoratio[i.wrapping_sub(1u32) as usize]
          && !(i == (parameters.tcp_numlayers as OPJ_UINT32).wrapping_sub(1u32)
            && parameters.tcp_distoratio[i as usize] == 0 as core::ffi::c_float)
        {
          event_msg!(
            p_manager,
            EVT_WARNING,
            "tcp_distoratio[%d]=%f should be strictly greater than tcp_distoratio[%d]=%f\n",
            i,
            parameters.tcp_distoratio[i as usize] as core::ffi::c_double,
            i.wrapping_sub(1u32),
            parameters.tcp_distoratio[i.wrapping_sub(1u32) as usize] as core::ffi::c_double,
          );
        }
        i += 1;
      }
    }
    /* see if max_codestream_size does limit input rate */
    if parameters.max_cs_size <= 0i32 {
      if parameters.tcp_rates[(parameters.tcp_numlayers - 1i32) as usize] > 0 as core::ffi::c_float
      {
        let mut temp_size: OPJ_FLOAT32 = 0.;
        temp_size = (image.numcomps as core::ffi::c_double
          * (*image.comps.offset(0)).w as core::ffi::c_double
          * (*image.comps.offset(0)).h as core::ffi::c_double
          * (*image.comps.offset(0)).prec as core::ffi::c_double
          / (parameters.tcp_rates[(parameters.tcp_numlayers - 1i32) as usize]
            as core::ffi::c_double
            * 8 as core::ffi::c_double
            * (*image.comps.offset(0)).dx as core::ffi::c_double
            * (*image.comps.offset(0)).dy as core::ffi::c_double))
          as OPJ_FLOAT32;
        if temp_size > 2147483647 as core::ffi::c_float {
          parameters.max_cs_size = 2147483647i32
        } else {
          parameters.max_cs_size = floor(temp_size as core::ffi::c_double) as core::ffi::c_int
        }
      } else {
        parameters.max_cs_size = 0i32
      }
    } else {
      let mut temp_rate: OPJ_FLOAT32 = 0.;
      let mut cap = 0i32;
      if parameters.rsiz as core::ffi::c_int >= 0x400i32
        && parameters.rsiz as core::ffi::c_int <= 0x900i32 | 0x9bi32
        && parameters.max_cs_size > 0i32
        && parameters.tcp_numlayers == 1i32
        && parameters.tcp_rates[0_usize] == 0 as core::ffi::c_float
      {
        parameters.tcp_rates[0_usize] = image
          .numcomps
          .wrapping_mul((*image.comps.offset(0)).w)
          .wrapping_mul((*image.comps.offset(0)).h)
          .wrapping_mul((*image.comps.offset(0)).prec)
          as OPJ_FLOAT32
          / (parameters.max_cs_size as OPJ_UINT32)
            .wrapping_mul(8u32)
            .wrapping_mul((*image.comps.offset(0)).dx)
            .wrapping_mul((*image.comps.offset(0)).dy) as OPJ_FLOAT32
      }
      temp_rate = (image.numcomps as core::ffi::c_double
        * (*image.comps.offset(0)).w as core::ffi::c_double
        * (*image.comps.offset(0)).h as core::ffi::c_double
        * (*image.comps.offset(0)).prec as core::ffi::c_double
        / (parameters.max_cs_size as core::ffi::c_double
          * 8 as core::ffi::c_double
          * (*image.comps.offset(0)).dx as core::ffi::c_double
          * (*image.comps.offset(0)).dy as core::ffi::c_double)) as OPJ_FLOAT32;
      i = 0 as OPJ_UINT32;
      while i < parameters.tcp_numlayers as OPJ_UINT32 {
        if parameters.tcp_rates[i as usize] < temp_rate {
          parameters.tcp_rates[i as usize] = temp_rate;
          cap = 1i32
        }
        i += 1;
      }
      if cap != 0 {
        event_msg!(p_manager, EVT_WARNING,
                          "The desired maximum codestream size has limited\nat least one of the desired quality layers\n");
      }
    }
    if parameters.rsiz as core::ffi::c_int >= 0x3i32
      && parameters.rsiz as core::ffi::c_int <= 0x6i32
      || parameters.rsiz as core::ffi::c_int >= 0x400i32
        && parameters.rsiz as core::ffi::c_int <= 0x900i32 | 0x9bi32
    {
      p_j2k.m_specific_param.m_encoder.m_TLM = 1i32
    }
    /* Manage profiles and applications and set RSIZ */
    /* set cinema parameters if required */
    if parameters.rsiz as core::ffi::c_int >= 0x3i32
      && parameters.rsiz as core::ffi::c_int <= 0x6i32
    {
      if parameters.rsiz as core::ffi::c_int == 0x5i32
        || parameters.rsiz as core::ffi::c_int == 0x6i32
      {
        event_msg!(
          p_manager,
          EVT_WARNING,
          "JPEG 2000 Scalable Digital Cinema profiles not yet supported\n",
        );
        parameters.rsiz = 0 as OPJ_UINT16
      } else {
        opj_j2k_set_cinema_parameters(parameters, image, p_manager);
        if opj_j2k_is_cinema_compliant(image, parameters.rsiz, p_manager) == 0 {
          parameters.rsiz = 0 as OPJ_UINT16
        }
      }
    } else if parameters.rsiz as core::ffi::c_int == 0x7i32 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "JPEG 2000 Long Term Storage profile not yet supported\n",
      );
      parameters.rsiz = 0 as OPJ_UINT16
    } else if parameters.rsiz as core::ffi::c_int >= 0x100i32
      && parameters.rsiz as core::ffi::c_int <= 0x300i32 | 0xbi32
    {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "JPEG 2000 Broadcast profiles not yet supported\n",
      );
      parameters.rsiz = 0 as OPJ_UINT16
    } else if parameters.rsiz as core::ffi::c_int >= 0x400i32
      && parameters.rsiz as core::ffi::c_int <= 0x900i32 | 0x9bi32
    {
      opj_j2k_set_imf_parameters(parameters, image, p_manager);
      if opj_j2k_is_imf_compliant(parameters, image, p_manager) == 0 {
        parameters.rsiz = 0 as OPJ_UINT16
      }
    } else if parameters.rsiz as core::ffi::c_int & 0x8000i32 != 0 {
      if parameters.rsiz as core::ffi::c_int == 0x8000i32 {
        event_msg!(p_manager, EVT_WARNING,
                          "JPEG 2000 Part-2 profile defined\nbut no Part-2 extension enabled.\nProfile set to NONE.\n");
        parameters.rsiz = 0 as OPJ_UINT16
      } else if parameters.rsiz as core::ffi::c_int != 0x8000i32 | 0x100i32 {
        event_msg!(
          p_manager,
          EVT_WARNING,
          "Unsupported Part-2 extension enabled\nProfile set to NONE.\n",
        );
        parameters.rsiz = 0 as OPJ_UINT16
      }
    }
    /*
    copy user encoding parameters
    */
    (*cp).m_specific_param.m_enc.m_max_comp_size = parameters.max_comp_size as OPJ_UINT32;
    (*cp).rsiz = parameters.rsiz;

    if parameters.cp_fixed_alloc != 0 {
      (*cp).m_specific_param.m_enc.m_quality_layer_alloc_strategy =
        J2K_QUALITY_LAYER_ALLOCATION_STRATEGY::FIXED_LAYER;
    } else if parameters.cp_fixed_quality != 0 {
      (*cp).m_specific_param.m_enc.m_quality_layer_alloc_strategy =
        J2K_QUALITY_LAYER_ALLOCATION_STRATEGY::FIXED_DISTORTION_RATIO;
    } else {
      (*cp).m_specific_param.m_enc.m_quality_layer_alloc_strategy =
        J2K_QUALITY_LAYER_ALLOCATION_STRATEGY::RATE_DISTORTION_RATIO;
    }

    if parameters.cp_fixed_alloc != 0 {
      let mut array_size = (parameters.tcp_numlayers as size_t)
        .wrapping_mul(parameters.numresolution as size_t)
        .wrapping_mul(3)
        .wrapping_mul(core::mem::size_of::<OPJ_INT32>());
      (*cp).m_specific_param.m_enc.m_matrice = opj_malloc(array_size) as *mut OPJ_INT32;
      if (*cp).m_specific_param.m_enc.m_matrice.is_null() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to allocate copy of user encoding parameters matrix \n",
        );
        return 0i32;
      }
      memcpy(
        (*cp).m_specific_param.m_enc.m_matrice as *mut core::ffi::c_void,
        parameters.cp_matrice as *const core::ffi::c_void,
        array_size,
      );
    }
    /* tiles */
    (*cp).tdx = parameters.cp_tdx as OPJ_UINT32;
    (*cp).tdy = parameters.cp_tdy as OPJ_UINT32;
    /* tile offset */
    (*cp).tx0 = parameters.cp_tx0 as OPJ_UINT32;
    (*cp).ty0 = parameters.cp_ty0 as OPJ_UINT32;
    /* comment string */
    if !parameters.cp_comment.is_null() {
      (*cp).comment =
        opj_malloc(strlen(parameters.cp_comment).wrapping_add(1)) as *mut core::ffi::c_char;
      if (*cp).comment.is_null() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to allocate copy of comment string\n",
        );
        return 0i32;
      }
      strcpy((*cp).comment, parameters.cp_comment);
    } else {
      /* Create default comment for codestream */
      let comment = format!("Created by OpenJPEG version {}", OPJ_VERSION);
      let c_comment = alloc::ffi::CString::new(comment).unwrap();
      /* UniPG>> */
      (*cp).comment = c_comment.into_raw();
      if (*cp).comment.is_null() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to allocate comment string\n",
        );
        return 0i32;
      }
    }
    /*
    calculate other encoding parameters
    */
    if parameters.tile_size_on != 0 {
      if (*cp).tdx == 0u32 {
        event_msg!(p_manager, EVT_ERROR, "Invalid tile width\n",);
        return 0i32;
      }
      if (*cp).tdy == 0u32 {
        event_msg!(p_manager, EVT_ERROR, "Invalid tile height\n",);
        return 0i32;
      }
      (*cp).tw = opj_uint_ceildiv(image.x1 - ((*cp).tx0), (*cp).tdx);
      (*cp).th = opj_uint_ceildiv(image.y1 - ((*cp).ty0), (*cp).tdy);
      /* Check that the number of tiles is valid */
      if (*cp).tw > (65535u32).wrapping_div((*cp).th) {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Invalid number of tiles : %u x %u (maximum fixed by jpeg2000 norm is 65535 tiles)\n",
          (*cp).tw,
          (*cp).th,
        );
        return 0i32;
      }
    } else {
      (*cp).tdx = image.x1.wrapping_sub((*cp).tx0);
      (*cp).tdy = image.y1.wrapping_sub((*cp).ty0)
    }
    if parameters.tp_on != 0 {
      (*cp).m_specific_param.m_enc.m_tp_flag = parameters.tp_flag as OPJ_BYTE;
      (*cp).m_specific_param.m_enc.m_tp_on = true;
    }
    /* USE_JPWL */
    /* initialize the multiple tiles */
    /* ---------------------------- */
    (*cp).tcps = opj_calloc(
      (*cp).tw.wrapping_mul((*cp).th) as size_t,
      core::mem::size_of::<opj_tcp_t>(),
    ) as *mut opj_tcp_t;
    if (*cp).tcps.is_null() {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to allocate tile coding parameters\n",
      );
      return 0i32;
    }
    tileno = 0 as OPJ_UINT32;
    while tileno < (*cp).tw.wrapping_mul((*cp).th) {
      let mut tcp: *mut opj_tcp_t = &mut *(*cp).tcps.offset(tileno as isize) as *mut opj_tcp_t;
      let fixed_distoratio = (*cp).m_specific_param.m_enc.m_quality_layer_alloc_strategy
        == J2K_QUALITY_LAYER_ALLOCATION_STRATEGY::FIXED_DISTORTION_RATIO;
      (*tcp).numlayers = parameters.tcp_numlayers as OPJ_UINT32;

      j = 0 as OPJ_UINT32;
      while j < (*tcp).numlayers {
        if (*cp).rsiz as core::ffi::c_int >= 0x3i32 && (*cp).rsiz as core::ffi::c_int <= 0x6i32
          || (*cp).rsiz as core::ffi::c_int >= 0x400i32
            && (*cp).rsiz as core::ffi::c_int <= 0x900i32 | 0x9bi32
        {
          if fixed_distoratio {
            (*tcp).distoratio[j as usize] = parameters.tcp_distoratio[j as usize]
          }
          (*tcp).rates[j as usize] = parameters.tcp_rates[j as usize]
        } else if fixed_distoratio {
          /* add fixed_quality */
          (*tcp).distoratio[j as usize] = parameters.tcp_distoratio[j as usize]
        } else {
          (*tcp).rates[j as usize] = parameters.tcp_rates[j as usize]
        }
        if !fixed_distoratio && (*tcp).rates[j as usize] as core::ffi::c_double <= 1.0f64 {
          (*tcp).rates[j as usize] = 0.0f64 as OPJ_FLOAT32
          /* force lossless */
        }
        j += 1;
      }
      (*tcp).csty = parameters.csty as OPJ_UINT32;
      (*tcp).prg = parameters.prog_order;
      (*tcp).mct = parameters.tcp_mct as OPJ_UINT32;
      numpocs_tile = 0 as OPJ_UINT32;
      (*tcp).POC = false;
      if parameters.numpocs != 0 {
        /* initialisation of POC */
        i = 0 as OPJ_UINT32;
        while i < parameters.numpocs {
          if tileno.wrapping_add(1u32) == parameters.POC[i as usize].tile {
            let mut tcp_poc: *mut opj_poc_t =
              &mut *(*tcp).pocs.as_mut_ptr().offset(numpocs_tile as isize) as *mut opj_poc_t;
            if parameters.POC[numpocs_tile as usize].compno0 >= image.numcomps {
              event_msg!(p_manager, EVT_ERROR, "Invalid compno0 for POC %d\n", i,);
              return 0i32;
            }
            (*tcp_poc).resno0 = parameters.POC[numpocs_tile as usize].resno0;
            (*tcp_poc).compno0 = parameters.POC[numpocs_tile as usize].compno0;
            (*tcp_poc).layno1 = parameters.POC[numpocs_tile as usize].layno1;
            (*tcp_poc).resno1 = parameters.POC[numpocs_tile as usize].resno1;
            (*tcp_poc).compno1 = opj_uint_min(
              parameters.POC[numpocs_tile as usize].compno1,
              image.numcomps,
            );
            (*tcp_poc).prg1 = parameters.POC[numpocs_tile as usize].prg1;
            (*tcp_poc).tile = parameters.POC[numpocs_tile as usize].tile;
            numpocs_tile += 1;
          }
          i += 1;
        }
        if numpocs_tile != 0 {
          /* TODO MSD use the return value*/
          opj_j2k_check_poc_val(
            parameters.POC.as_mut_ptr(),
            tileno,
            parameters.numpocs,
            parameters.numresolution as OPJ_UINT32,
            image.numcomps,
            parameters.tcp_numlayers as OPJ_UINT32,
            p_manager,
          );
          (*tcp).POC = true;
          (*tcp).numpocs = numpocs_tile.wrapping_sub(1u32)
        }
      } else {
        (*tcp).numpocs = 0 as OPJ_UINT32
      }
      (*tcp).tccps =
        opj_calloc(image.numcomps as size_t, core::mem::size_of::<opj_tccp_t>()) as *mut opj_tccp_t;
      if (*tcp).tccps.is_null() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Not enough memory to allocate tile component coding parameters\n",
        );
        return 0i32;
      }
      if !parameters.mct_data.is_null() {
        let numcomps = image.numcomps;
        let mut lMctLen = numcomps * numcomps;
        let mut lMctSize = lMctLen.wrapping_mul(core::mem::size_of::<OPJ_FLOAT32>() as OPJ_UINT32);
        let mut lTmpBuf = opj_malloc(lMctSize as size_t) as *mut OPJ_FLOAT32;
        let mut l_dc_shift =
          (parameters.mct_data as *mut OPJ_BYTE).offset(lMctSize as isize) as *mut OPJ_INT32;
        if lTmpBuf.is_null() {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Not enough memory to allocate temp buffer\n",
          );
          return 0i32;
        }
        (*tcp).mct = 2 as OPJ_UINT32;
        (*tcp).m_mct_coding_matrix = opj_malloc(lMctSize as size_t) as *mut OPJ_FLOAT32;
        if (*tcp).m_mct_coding_matrix.is_null() {
          opj_free(lTmpBuf as *mut core::ffi::c_void);
          lTmpBuf = core::ptr::null_mut::<OPJ_FLOAT32>();
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Not enough memory to allocate encoder MCT coding matrix \n",
          );
          return 0i32;
        }
        memcpy(
          (*tcp).m_mct_coding_matrix as *mut core::ffi::c_void,
          parameters.mct_data,
          lMctSize as usize,
        );
        memcpy(
          lTmpBuf as *mut core::ffi::c_void,
          parameters.mct_data,
          lMctSize as usize,
        );
        (*tcp).m_mct_decoding_matrix = opj_malloc(lMctSize as size_t) as *mut OPJ_FLOAT32;
        if (*tcp).m_mct_decoding_matrix.is_null() {
          opj_free(lTmpBuf as *mut core::ffi::c_void);
          lTmpBuf = core::ptr::null_mut::<OPJ_FLOAT32>();
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Not enough memory to allocate encoder MCT decoding matrix \n",
          );
          return 0i32;
        }
        if !opj_matrix_inversion_f(
          core::slice::from_raw_parts_mut(lTmpBuf as *mut f32, lMctLen as usize),
          core::slice::from_raw_parts_mut(
            (*tcp).m_mct_decoding_matrix as *mut f32,
            lMctLen as usize,
          ),
          numcomps as usize,
        ) {
          opj_free(lTmpBuf as *mut core::ffi::c_void);
          lTmpBuf = core::ptr::null_mut::<OPJ_FLOAT32>();
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Failed to inverse encoder MCT decoding matrix \n",
          );
          return 0i32;
        }
        (*tcp).mct_norms =
          opj_malloc((image.numcomps as usize).wrapping_mul(core::mem::size_of::<OPJ_FLOAT64>()))
            as *mut OPJ_FLOAT64;
        if (*tcp).mct_norms.is_null() {
          opj_free(lTmpBuf as *mut core::ffi::c_void);
          lTmpBuf = core::ptr::null_mut::<OPJ_FLOAT32>();
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Not enough memory to allocate encoder MCT norms \n",
          );
          return 0i32;
        }
        opj_calculate_norms(
          (*tcp).mct_norms,
          image.numcomps,
          (*tcp).m_mct_decoding_matrix,
        );
        opj_free(lTmpBuf as *mut core::ffi::c_void);
        i = 0 as OPJ_UINT32;
        while i < image.numcomps {
          let mut tccp: *mut opj_tccp_t = &mut *(*tcp).tccps.offset(i as isize) as *mut opj_tccp_t;
          (*tccp).m_dc_level_shift = *l_dc_shift.offset(i as isize);
          i += 1;
        }
        if opj_j2k_setup_mct_encoding(tcp, image) == 0i32 {
          /* free will be handled by opj_j2k_destroy */
          event_msg!(p_manager, EVT_ERROR, "Failed to setup j2k mct encoding\n",);
          return 0i32;
        }
      } else {
        if (*tcp).mct == 1u32 && image.numcomps >= 3u32 {
          /* RGB->YCC MCT is enabled */
          if (*image.comps.offset(0)).dx != (*image.comps.offset(1)).dx
            || (*image.comps.offset(0)).dx != (*image.comps.offset(2)).dx
            || (*image.comps.offset(0)).dy != (*image.comps.offset(1)).dy
            || (*image.comps.offset(0)).dy != (*image.comps.offset(2)).dy
          {
            event_msg!(
              p_manager,
              EVT_WARNING,
              "Cannot perform MCT on components with different sizes. Disabling MCT.\n",
            ); /* 0 => one precinct || 1 => custom precinct  */
            (*tcp).mct = 0 as OPJ_UINT32
          }
        }
        i = 0 as OPJ_UINT32;
        while i < image.numcomps {
          let mut tccp_0: *mut opj_tccp_t =
            &mut *(*tcp).tccps.offset(i as isize) as *mut opj_tccp_t;
          let mut l_comp: *mut opj_image_comp_t =
            &mut *image.comps.offset(i as isize) as *mut opj_image_comp_t;
          if (*l_comp).sgnd == 0 {
            (*tccp_0).m_dc_level_shift = (1i32) << (*l_comp).prec.wrapping_sub(1u32)
          }
          i += 1;
        }
      }
      i = 0 as OPJ_UINT32;
      while i < image.numcomps {
        let mut tccp_1: *mut opj_tccp_t = &mut *(*tcp).tccps.offset(i as isize) as *mut opj_tccp_t;
        (*tccp_1).csty = (parameters.csty & 0x1i32) as OPJ_UINT32;
        (*tccp_1).numresolutions = parameters.numresolution as OPJ_UINT32;
        (*tccp_1).cblkw = opj_int_floorlog2(parameters.cblockw_init) as OPJ_UINT32;
        (*tccp_1).cblkh = opj_int_floorlog2(parameters.cblockh_init) as OPJ_UINT32;
        (*tccp_1).cblksty = parameters.mode as OPJ_UINT32;
        (*tccp_1).qmfbid = if parameters.irreversible != 0 {
          0i32
        } else {
          1i32
        } as OPJ_UINT32;
        (*tccp_1).qntsty = if parameters.irreversible != 0 {
          2i32
        } else {
          0i32
        } as OPJ_UINT32;
        (*tccp_1).numgbits = 2 as OPJ_UINT32;
        if i as OPJ_INT32 == parameters.roi_compno {
          (*tccp_1).roishift = parameters.roi_shift
        } else {
          (*tccp_1).roishift = 0i32
        }
        if parameters.csty & 0x1i32 != 0 {
          let mut p = 0i32;
          let mut it_res: OPJ_INT32 = 0;
          assert!((*tccp_1).numresolutions > 0u32);
          it_res = (*tccp_1).numresolutions as OPJ_INT32 - 1i32;
          while it_res >= 0i32 {
            if p < parameters.res_spec {
              if parameters.prcw_init[p as usize] < 1i32 {
                (*tccp_1).prcw[it_res as usize] = 1 as OPJ_UINT32
              } else {
                (*tccp_1).prcw[it_res as usize] =
                  opj_int_floorlog2(parameters.prcw_init[p as usize]) as OPJ_UINT32
              }
              if parameters.prch_init[p as usize] < 1i32 {
                (*tccp_1).prch[it_res as usize] = 1 as OPJ_UINT32
              } else {
                (*tccp_1).prch[it_res as usize] =
                  opj_int_floorlog2(parameters.prch_init[p as usize]) as OPJ_UINT32
              }
            } else {
              let mut res_spec = parameters.res_spec;
              let mut size_prcw = 0i32;
              let mut size_prch = 0i32;
              /*end for*/
              assert!(res_spec > 0i32);
              size_prcw =
                parameters.prcw_init[(res_spec - 1i32) as usize] >> (p - (res_spec - 1i32));
              size_prch =
                parameters.prch_init[(res_spec - 1i32) as usize] >> (p - (res_spec - 1i32));
              if size_prcw < 1i32 {
                (*tccp_1).prcw[it_res as usize] = 1 as OPJ_UINT32
              } else {
                (*tccp_1).prcw[it_res as usize] = opj_int_floorlog2(size_prcw) as OPJ_UINT32
              }
              if size_prch < 1i32 {
                (*tccp_1).prch[it_res as usize] = 1 as OPJ_UINT32
              } else {
                (*tccp_1).prch[it_res as usize] = opj_int_floorlog2(size_prch) as OPJ_UINT32
              }
            }
            p += 1;
            it_res -= 1
            /*printf("\nsize precinct for level %d : %d,%d\n", it_res,tccp->prcw[it_res], tccp->prch[it_res]); */
          }
        } else {
          j = 0 as OPJ_UINT32;
          while j < (*tccp_1).numresolutions {
            (*tccp_1).prcw[j as usize] = 15 as OPJ_UINT32;
            (*tccp_1).prch[j as usize] = 15 as OPJ_UINT32;
            j += 1;
          }
        }
        opj_dwt_calc_explicit_stepsizes(tccp_1, (*image.comps.offset(i as isize)).prec);
        i += 1;
      }
      tileno += 1;
    }
    if !parameters.mct_data.is_null() {
      opj_free(parameters.mct_data);
      parameters.mct_data = core::ptr::null_mut::<core::ffi::c_void>()
    }
    1i32
  }
}
/* *
Add main header marker information
@param cstr_index    Codestream information structure
@param type         marker type
@param pos          byte offset of marker segment
@param len          length of marker segment
 */
fn opj_j2k_add_mhmarker(
  mut cstr_index: *mut opj_codestream_index_t,
  mut marker: J2KMarker,
  mut pos: OPJ_OFF_T,
  mut len: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    assert!(!cstr_index.is_null());
    /* expand the list? */
    if (*cstr_index).marknum.wrapping_add(1u32) > (*cstr_index).maxmarknum {
      let mut new_marker = core::ptr::null_mut::<opj_marker_info_t>();
      (*cstr_index).maxmarknum =
        (100 as core::ffi::c_float + (*cstr_index).maxmarknum as OPJ_FLOAT32) as OPJ_UINT32;
      new_marker = opj_realloc(
        (*cstr_index).marker as *mut core::ffi::c_void,
        ((*cstr_index).maxmarknum as usize).wrapping_mul(core::mem::size_of::<opj_marker_info_t>()),
      ) as *mut opj_marker_info_t;
      if new_marker.is_null() {
        opj_free((*cstr_index).marker as *mut core::ffi::c_void);
        (*cstr_index).marker = core::ptr::null_mut::<opj_marker_info_t>();
        (*cstr_index).maxmarknum = 0 as OPJ_UINT32;
        (*cstr_index).marknum = 0 as OPJ_UINT32;
        /* event_msg!(p_manager, EVT_ERROR, "Not enough memory to add mh marker\n"); */
        return 0i32;
      }
      (*cstr_index).marker = new_marker
    }
    /* add the marker */
    (*(*cstr_index).marker.offset((*cstr_index).marknum as isize)).type_ =
      marker.as_u32() as OPJ_UINT16;
    (*(*cstr_index).marker.offset((*cstr_index).marknum as isize)).pos = pos as OPJ_OFF_T;
    (*(*cstr_index).marker.offset((*cstr_index).marknum as isize)).len = len as OPJ_INT32;
    (*cstr_index).marknum = (*cstr_index).marknum.wrapping_add(1);
    1i32
  }
}
/* *
Add tile header marker information
@param tileno       tile index number
@param cstr_index   Codestream information structure
@param type         marker type
@param pos          byte offset of marker segment
@param len          length of marker segment
 */
fn opj_j2k_add_tlmarker(
  mut tileno: OPJ_UINT32,
  mut cstr_index: *mut opj_codestream_index_t,
  mut marker: J2KMarker,
  mut pos: OPJ_OFF_T,
  mut len: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    assert!(!cstr_index.is_null());
    assert!(!(*cstr_index).tile_index.is_null());
    /* expand the list? */
    if (*(*cstr_index).tile_index.offset(tileno as isize))
      .marknum
      .wrapping_add(1u32)
      > (*(*cstr_index).tile_index.offset(tileno as isize)).maxmarknum
    {
      let mut new_marker = core::ptr::null_mut::<opj_marker_info_t>();
      (*(*cstr_index).tile_index.offset(tileno as isize)).maxmarknum = (100i32
        as core::ffi::c_float
        + (*(*cstr_index).tile_index.offset(tileno as isize)).maxmarknum as OPJ_FLOAT32)
        as OPJ_UINT32;
      new_marker = opj_realloc(
        (*(*cstr_index).tile_index.offset(tileno as isize)).marker as *mut core::ffi::c_void,
        ((*(*cstr_index).tile_index.offset(tileno as isize)).maxmarknum as usize)
          .wrapping_mul(core::mem::size_of::<opj_marker_info_t>()),
      ) as *mut opj_marker_info_t;
      if new_marker.is_null() {
        opj_free(
          (*(*cstr_index).tile_index.offset(tileno as isize)).marker as *mut core::ffi::c_void,
        );
        let fresh23 = &mut (*(*cstr_index).tile_index.offset(tileno as isize)).marker;
        *fresh23 = core::ptr::null_mut::<opj_marker_info_t>();
        (*(*cstr_index).tile_index.offset(tileno as isize)).maxmarknum = 0 as OPJ_UINT32;
        (*(*cstr_index).tile_index.offset(tileno as isize)).marknum = 0 as OPJ_UINT32;
        /* event_msg!(p_manager, EVT_ERROR, "Not enough memory to add tl marker\n"); */
        return 0i32;
      }
      let fresh24 = &mut (*(*cstr_index).tile_index.offset(tileno as isize)).marker;
      *fresh24 = new_marker
    }
    /* add the marker */
    (*(*(*cstr_index).tile_index.offset(tileno as isize))
      .marker
      .offset((*(*cstr_index).tile_index.offset(tileno as isize)).marknum as isize))
    .type_ = marker.as_u32() as OPJ_UINT16;
    (*(*(*cstr_index).tile_index.offset(tileno as isize))
      .marker
      .offset((*(*cstr_index).tile_index.offset(tileno as isize)).marknum as isize))
    .pos = pos as OPJ_OFF_T;
    (*(*(*cstr_index).tile_index.offset(tileno as isize))
      .marker
      .offset((*(*cstr_index).tile_index.offset(tileno as isize)).marknum as isize))
    .len = len as OPJ_INT32;
    let fresh25 = &mut (*(*cstr_index).tile_index.offset(tileno as isize)).marknum;
    *fresh25 = (*fresh25).wrapping_add(1);
    if marker == J2KMarker::SOT {
      let mut l_current_tile_part =
        (*(*cstr_index).tile_index.offset(tileno as isize)).current_tpsno;
      if !(*(*cstr_index).tile_index.offset(tileno as isize))
        .tp_index
        .is_null()
      {
        (*(*(*cstr_index).tile_index.offset(tileno as isize))
          .tp_index
          .offset(l_current_tile_part as isize))
        .start_pos = pos
      }
    }
    1i32
  }
}
/*
 * -----------------------------------------------------------------------
 * -----------------------------------------------------------------------
 * -----------------------------------------------------------------------
 */

pub(crate) fn opj_j2k_end_decompress(
  mut _p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  1i32
}

pub(crate) fn opj_j2k_read_header(
  mut p_stream: &mut Stream,
  mut p_j2k: &mut opj_j2k,
  mut p_image: *mut *mut opj_image_t,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  let mut validation_list = opj_j2k_proc_list_t::new();
  let mut procedure_list = opj_j2k_proc_list_t::new();
  unsafe {
    /* preconditions */

    /* create an empty image header */
    p_j2k.m_private_image = opj_image_create0();
    if p_j2k.m_private_image.is_null() {
      return 0i32;
    }
    /* customization of the validation */
    if opj_j2k_setup_decoding_validation(p_j2k, &mut validation_list, p_manager) == 0 {
      opj_image_destroy(p_j2k.m_private_image);
      p_j2k.m_private_image = core::ptr::null_mut::<opj_image_t>();
      return 0i32;
    }
    /* validation of the parameters codec */
    if opj_j2k_exec(p_j2k, &mut validation_list, p_stream, p_manager) == 0 {
      opj_image_destroy(p_j2k.m_private_image);
      p_j2k.m_private_image = core::ptr::null_mut::<opj_image_t>();
      return 0i32;
    }
    /* customization of the encoding */
    if opj_j2k_setup_header_reading(p_j2k, &mut procedure_list, p_manager) == 0 {
      opj_image_destroy(p_j2k.m_private_image);
      p_j2k.m_private_image = core::ptr::null_mut::<opj_image_t>();
      return 0i32;
    }
    /* read header */
    if opj_j2k_exec(p_j2k, &mut procedure_list, p_stream, p_manager) == 0 {
      opj_image_destroy(p_j2k.m_private_image);
      p_j2k.m_private_image = core::ptr::null_mut::<opj_image_t>();
      return 0i32;
    }
    *p_image = opj_image_create0();
    if (*p_image).is_null() {
      return 0i32;
    }
    /* Copy codestream image information to the output image */
    opj_copy_image_header(p_j2k.m_private_image, *p_image);
    /*Allocate and initialize some elements of codestrem index*/
    if opj_j2k_allocate_tile_element_cstr_index(p_j2k) == 0 {
      opj_image_destroy(*p_image);
      *p_image = core::ptr::null_mut::<opj_image_t>();
      return 0i32;
    }
    1i32
  }
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
 * Copyright (c) 2008, Jerome Fimes, Communications & Systemes <jerome.fimes@c-s.fr>
 * Copyright (c) 2006-2007, Parvatha Elangovan
 * Copyright (c) 2010-2011, Kaori Hagihara
 * Copyright (c) 2011-2012, Centre National d'Etudes Spatiales (CNES), France
 * Copyright (c) 2012, CS Systemes d'Information, France
 * Copyright (c) 2017, IntoPIX SA <support@intopix.com>
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
/* * @defgroup J2K J2K - JPEG-2000 codestream reader/writer */
/*@{*/
/* * @name Local static functions */
/*@{*/
/* *
 * Sets up the procedures to do on reading header. Developers wanting to extend the library can add their own reading procedures.
 */
fn opj_j2k_setup_header_reading(
  _p_j2k: &mut opj_j2k,
  list: &mut opj_j2k_proc_list_t,
  _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  list.add(opj_j2k_read_header_procedure);
  /* DEVELOPER CORNER, add your custom procedures */
  list.add(opj_j2k_copy_default_tcp_and_create_tcd);
  1i32
}
/* *
 * Sets up the validation ,i.e. adds the procedures to launch to make sure the codec parameters
 * are valid. Developers wanting to extend the library can add their own validation procedures.
 */
fn opj_j2k_setup_decoding_validation(
  _p_j2k: &mut opj_j2k,
  list: &mut opj_j2k_proc_list_t,
  _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  list.add(opj_j2k_build_decoder);
  list.add(opj_j2k_decoding_validation);
  /* DEVELOPER CORNER, add your custom validation procedure */
  1i32
}

/* *
 * The mct encoding validation procedure.
 *
 * @param       p_j2k                   the jpeg2000 codec to validate.
 * @param       p_stream                                the input stream to validate.
 * @param       p_manager               the user event manager.
 *
 * @return true if the parameters are correct.
 */
fn opj_j2k_mct_validation(
  mut p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_is_valid = 1i32;
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    /* preconditions */

    if p_j2k.m_cp.rsiz as core::ffi::c_int & 0x8200i32 == 0x8200i32 {
      let mut l_nb_tiles = p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw);
      let mut l_tcp = p_j2k.m_cp.tcps;
      i = 0 as OPJ_UINT32;
      while i < l_nb_tiles {
        if (*l_tcp).mct == 2u32 {
          let mut l_tccp = (*l_tcp).tccps;
          l_is_valid &= ((*l_tcp).m_mct_coding_matrix != core::ptr::null_mut::<OPJ_FLOAT32>())
            as core::ffi::c_int;
          j = 0 as OPJ_UINT32;
          while j < (*p_j2k.m_private_image).numcomps {
            l_is_valid &= ((*l_tccp).qmfbid & 1u32 == 0) as core::ffi::c_int;
            l_tccp = l_tccp.offset(1);
            j += 1;
          }
        }
        l_tcp = l_tcp.offset(1);
        i += 1;
      }
    }
    l_is_valid
  }
}

pub(crate) fn opj_j2k_setup_mct_encoding(
  mut p_tcp: *mut opj_tcp_t,
  mut p_image: &mut opj_image,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_indix = 1 as OPJ_UINT32;
    let mut l_mct_deco_data = core::ptr::null_mut::<opj_mct_data_t>();
    let mut l_mct_offset_data = core::ptr::null_mut::<opj_mct_data_t>();
    let mut l_mcc_data = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
    let mut l_mct_size: OPJ_UINT32 = 0;
    let mut l_nb_elem: OPJ_UINT32 = 0;
    let mut l_data = core::ptr::null_mut::<OPJ_FLOAT32>();
    let mut l_current_data = core::ptr::null_mut::<OPJ_FLOAT32>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    /* preconditions */
    assert!(!p_tcp.is_null());
    if (*p_tcp).mct != 2u32 {
      return 1i32;
    }
    if !(*p_tcp).m_mct_decoding_matrix.is_null() {
      if (*p_tcp).m_nb_mct_records == (*p_tcp).m_nb_max_mct_records {
        let mut new_mct_records = core::ptr::null_mut::<opj_mct_data_t>();
        (*p_tcp).m_nb_max_mct_records =
          ((*p_tcp).m_nb_max_mct_records as core::ffi::c_uint).wrapping_add(10u32) as OPJ_UINT32;
        new_mct_records = opj_realloc(
          (*p_tcp).m_mct_records as *mut core::ffi::c_void,
          ((*p_tcp).m_nb_max_mct_records as usize)
            .wrapping_mul(core::mem::size_of::<opj_mct_data_t>()),
        ) as *mut opj_mct_data_t;
        if new_mct_records.is_null() {
          opj_free((*p_tcp).m_mct_records as *mut core::ffi::c_void);
          (*p_tcp).m_mct_records = core::ptr::null_mut::<opj_mct_data_t>();
          (*p_tcp).m_nb_max_mct_records = 0 as OPJ_UINT32;
          (*p_tcp).m_nb_mct_records = 0 as OPJ_UINT32;
          /* event_msg!(p_manager, EVT_ERROR, "Not enough memory to setup mct encoding\n"); */
          return 0i32;
        }
        (*p_tcp).m_mct_records = new_mct_records;
        l_mct_deco_data = (*p_tcp)
          .m_mct_records
          .offset((*p_tcp).m_nb_mct_records as isize);
        memset(
          l_mct_deco_data as *mut core::ffi::c_void,
          0i32,
          ((*p_tcp)
            .m_nb_max_mct_records
            .wrapping_sub((*p_tcp).m_nb_mct_records) as usize)
            .wrapping_mul(core::mem::size_of::<opj_mct_data_t>()),
        );
      }
      l_mct_deco_data = (*p_tcp)
        .m_mct_records
        .offset((*p_tcp).m_nb_mct_records as isize);
      if !(*l_mct_deco_data).m_data.is_null() {
        opj_free((*l_mct_deco_data).m_data as *mut core::ffi::c_void);
        (*l_mct_deco_data).m_data = core::ptr::null_mut::<OPJ_BYTE>()
      }
      let fresh26 = l_indix;
      l_indix = l_indix.wrapping_add(1);
      (*l_mct_deco_data).m_index = fresh26;
      (*l_mct_deco_data).m_array_type = MCT_TYPE_DECORRELATION;
      (*l_mct_deco_data).m_element_type = MCTElementType::FLOAT;
      l_nb_elem = p_image.numcomps.wrapping_mul(p_image.numcomps);
      l_mct_size = l_nb_elem.wrapping_mul((*l_mct_deco_data).m_element_type.size());
      (*l_mct_deco_data).m_data = opj_malloc(l_mct_size as size_t) as *mut OPJ_BYTE;
      if (*l_mct_deco_data).m_data.is_null() {
        return 0i32;
      }
      (*l_mct_deco_data).m_element_type.write_from_float(
        (*p_tcp).m_mct_decoding_matrix as *const core::ffi::c_void,
        (*l_mct_deco_data).m_data as *mut core::ffi::c_void,
        l_nb_elem,
      );
      (*l_mct_deco_data).m_data_size = l_mct_size;
      (*p_tcp).m_nb_mct_records = (*p_tcp).m_nb_mct_records.wrapping_add(1)
    }
    if (*p_tcp).m_nb_mct_records == (*p_tcp).m_nb_max_mct_records {
      let mut new_mct_records_0 = core::ptr::null_mut::<opj_mct_data_t>();
      (*p_tcp).m_nb_max_mct_records =
        ((*p_tcp).m_nb_max_mct_records as core::ffi::c_uint).wrapping_add(10u32) as OPJ_UINT32;
      new_mct_records_0 = opj_realloc(
        (*p_tcp).m_mct_records as *mut core::ffi::c_void,
        ((*p_tcp).m_nb_max_mct_records as usize)
          .wrapping_mul(core::mem::size_of::<opj_mct_data_t>()),
      ) as *mut opj_mct_data_t;
      if new_mct_records_0.is_null() {
        opj_free((*p_tcp).m_mct_records as *mut core::ffi::c_void);
        (*p_tcp).m_mct_records = core::ptr::null_mut::<opj_mct_data_t>();
        (*p_tcp).m_nb_max_mct_records = 0 as OPJ_UINT32;
        (*p_tcp).m_nb_mct_records = 0 as OPJ_UINT32;
        /* event_msg!(p_manager, EVT_ERROR, "Not enough memory to setup mct encoding\n"); */
        return 0i32;
      }
      (*p_tcp).m_mct_records = new_mct_records_0;
      l_mct_offset_data = (*p_tcp)
        .m_mct_records
        .offset((*p_tcp).m_nb_mct_records as isize);
      memset(
        l_mct_offset_data as *mut core::ffi::c_void,
        0i32,
        ((*p_tcp)
          .m_nb_max_mct_records
          .wrapping_sub((*p_tcp).m_nb_mct_records) as usize)
          .wrapping_mul(core::mem::size_of::<opj_mct_data_t>()),
      );
      if !l_mct_deco_data.is_null() {
        l_mct_deco_data = l_mct_offset_data.offset(-1)
      }
    }
    l_mct_offset_data = (*p_tcp)
      .m_mct_records
      .offset((*p_tcp).m_nb_mct_records as isize);
    if !(*l_mct_offset_data).m_data.is_null() {
      opj_free((*l_mct_offset_data).m_data as *mut core::ffi::c_void);
      (*l_mct_offset_data).m_data = core::ptr::null_mut::<OPJ_BYTE>()
    }
    let fresh27 = l_indix;
    l_indix = l_indix.wrapping_add(1);
    (*l_mct_offset_data).m_index = fresh27;
    (*l_mct_offset_data).m_array_type = MCT_TYPE_OFFSET;
    (*l_mct_offset_data).m_element_type = MCTElementType::FLOAT;
    l_nb_elem = p_image.numcomps;
    l_mct_size = l_nb_elem.wrapping_mul((*l_mct_offset_data).m_element_type.size());
    (*l_mct_offset_data).m_data = opj_malloc(l_mct_size as size_t) as *mut OPJ_BYTE;
    if (*l_mct_offset_data).m_data.is_null() {
      return 0i32;
    }
    l_data = opj_malloc((l_nb_elem as usize).wrapping_mul(core::mem::size_of::<OPJ_FLOAT32>()))
      as *mut OPJ_FLOAT32;
    if l_data.is_null() {
      opj_free((*l_mct_offset_data).m_data as *mut core::ffi::c_void);
      (*l_mct_offset_data).m_data = core::ptr::null_mut::<OPJ_BYTE>();
      return 0i32;
    }
    l_tccp = (*p_tcp).tccps;
    l_current_data = l_data;
    i = 0 as OPJ_UINT32;
    while i < l_nb_elem {
      let fresh28 = l_current_data;
      l_current_data = l_current_data.offset(1);
      *fresh28 = (*l_tccp).m_dc_level_shift as OPJ_FLOAT32;
      l_tccp = l_tccp.offset(1);
      i += 1;
    }
    (*l_mct_offset_data).m_element_type.write_from_float(
      l_data as *const core::ffi::c_void,
      (*l_mct_offset_data).m_data as *mut core::ffi::c_void,
      l_nb_elem,
    );
    opj_free(l_data as *mut core::ffi::c_void);
    (*l_mct_offset_data).m_data_size = l_mct_size;
    (*p_tcp).m_nb_mct_records = (*p_tcp).m_nb_mct_records.wrapping_add(1);
    if (*p_tcp).m_nb_mcc_records == (*p_tcp).m_nb_max_mcc_records {
      let mut new_mcc_records = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
      (*p_tcp).m_nb_max_mcc_records =
        ((*p_tcp).m_nb_max_mcc_records as core::ffi::c_uint).wrapping_add(10u32) as OPJ_UINT32;
      new_mcc_records = opj_realloc(
        (*p_tcp).m_mcc_records as *mut core::ffi::c_void,
        ((*p_tcp).m_nb_max_mcc_records as usize)
          .wrapping_mul(core::mem::size_of::<opj_simple_mcc_decorrelation_data_t>()),
      ) as *mut opj_simple_mcc_decorrelation_data_t;
      if new_mcc_records.is_null() {
        opj_free((*p_tcp).m_mcc_records as *mut core::ffi::c_void);
        (*p_tcp).m_mcc_records = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
        (*p_tcp).m_nb_max_mcc_records = 0 as OPJ_UINT32;
        (*p_tcp).m_nb_mcc_records = 0 as OPJ_UINT32;
        /* event_msg!(p_manager, EVT_ERROR, "Not enough memory to setup mct encoding\n"); */
        return 0i32;
      }
      (*p_tcp).m_mcc_records = new_mcc_records;
      l_mcc_data = (*p_tcp)
        .m_mcc_records
        .offset((*p_tcp).m_nb_mcc_records as isize);
      memset(
        l_mcc_data as *mut core::ffi::c_void,
        0i32,
        ((*p_tcp)
          .m_nb_max_mcc_records
          .wrapping_sub((*p_tcp).m_nb_mcc_records) as usize)
          .wrapping_mul(core::mem::size_of::<opj_simple_mcc_decorrelation_data_t>()),
      );
    }
    l_mcc_data = (*p_tcp)
      .m_mcc_records
      .offset((*p_tcp).m_nb_mcc_records as isize);
    (*l_mcc_data).m_decorrelation_array = l_mct_deco_data;
    (*l_mcc_data).m_is_irreversible = true;
    (*l_mcc_data).m_nb_comps = p_image.numcomps;
    let fresh29 = l_indix;
    l_indix = l_indix.wrapping_add(1);
    (*l_mcc_data).m_index = fresh29;
    (*l_mcc_data).m_offset_array = l_mct_offset_data;
    (*p_tcp).m_nb_mcc_records = (*p_tcp).m_nb_mcc_records.wrapping_add(1);
    1i32
  }
}

/* *
 * Builds the tcd decoder to use to decode tile.
 */
fn opj_j2k_build_decoder(
  mut _p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* add here initialization of cp
  copy paste of setup_decoder */
  1i32
}

/* *
 * Builds the tcd encoder to use to encode tile.
 */
fn opj_j2k_build_encoder(
  mut _p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* add here initialization of cp
  copy paste of setup_encoder */
  1i32
}

/* *
 * The default encoding validation procedure without any extension.
 *
 * @param       p_j2k                   the jpeg2000 codec to validate.
 * @param       p_stream                the input stream to validate.
 * @param       p_manager               the user event manager.
 *
 * @return true if the parameters are correct.
 */
fn opj_j2k_encoding_validation(
  mut p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_is_valid = 1i32;
    /* preconditions */

    /* STATE checking */
    /* make sure the state is at 0 */
    l_is_valid &= (p_j2k.m_specific_param.m_decoder.m_state == J2KState::NONE) as core::ffi::c_int;
    /* ISO 15444-1:2004 states between 1 & 33 (0 -> 32) */
    /* 33 (32) would always fail the check below (if a cast to 64bits was done) */
    /* FIXME Shall we change OPJ_J2K_MAXRLVLS to 32 ? */
    if (*(*p_j2k.m_cp.tcps).tccps).numresolutions <= 0u32
      || (*(*p_j2k.m_cp.tcps).tccps).numresolutions > 32u32
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Number of resolutions is too high in comparison to the size of tiles\n",
      );
      return 0i32;
    }
    if p_j2k.m_cp.tdx
      < ((1i32)
        << (*(*p_j2k.m_cp.tcps).tccps)
          .numresolutions
          .wrapping_sub(1u32)) as OPJ_UINT32
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Number of resolutions is too high in comparison to the size of tiles\n",
      );
      return 0i32;
    }
    if p_j2k.m_cp.tdy
      < ((1i32)
        << (*(*p_j2k.m_cp.tcps).tccps)
          .numresolutions
          .wrapping_sub(1u32)) as OPJ_UINT32
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Number of resolutions is too high in comparison to the size of tiles\n",
      );
      return 0i32;
    }
    /* PARAMETER VALIDATION */
    l_is_valid
  }
}
/* *
 * The default decoding validation procedure without any extension.
 *
 * @param       p_j2k                   the jpeg2000 codec to validate.
 * @param       p_stream                                the input stream to validate.
 * @param       p_manager               the user event manager.
 *
 * @return true if the parameters are correct.
 */
fn opj_j2k_decoding_validation(
  mut p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_is_valid = 1i32;
    /* preconditions*/

    /* STATE checking */
    /* make sure the state is at 0 */
    l_is_valid &= (p_j2k.m_specific_param.m_decoder.m_state == J2KState::NONE) as core::ffi::c_int;
    /* PARAMETER VALIDATION */
    l_is_valid
  }
}
/* *
 * The read header procedure.
 */
fn opj_j2k_read_header_procedure(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_marker_size: OPJ_UINT32 = 0;
    let mut l_has_siz = 0i32;
    let mut l_has_cod = 0i32;
    let mut l_has_qcd = 0i32;
    /* preconditions */

    /*  We enter in the main header */
    p_j2k.m_specific_param.m_decoder.m_state = J2KState::MHSOC;
    /* Try to read the SOC marker, the codestream must begin with SOC marker */
    if opj_j2k_read_soc(p_j2k, p_stream, p_manager) == 0 {
      event_msg!(p_manager, EVT_ERROR, "Expected a SOC marker \n",);
      return 0i32;
    }
    /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer */
    if opj_stream_read_data(
      p_stream,
      p_j2k.m_specific_param.m_decoder.m_header_data,
      2 as OPJ_SIZE_T,
      p_manager,
    ) != 2
    {
      event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
      return 0i32;
    }
    /* Read 2 bytes as the new marker ID */
    let mut l_current_marker =
      J2KMarker::from_buffer(p_j2k.m_specific_param.m_decoder.m_header_data);
    /* Try to read until the SOT is detected */
    while l_current_marker != J2KMarker::SOT {
      /* Check if the current marker ID is valid */
      if l_current_marker.is_invalid() {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "A marker ID was expected (0xff--) instead of %.8x\n",
          l_current_marker.as_u32(),
        );
        return 0i32;
      }
      /* Get the marker handler from the marker ID */
      let mut l_marker_handler = l_current_marker;
      /* Manage case where marker is unknown */
      if l_marker_handler.is_unknown() {
        if opj_j2k_read_unk(p_j2k, p_stream, &mut l_current_marker, p_manager) == 0 {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Unknown marker has been detected and generated error.\n",
          );
          return 0i32;
        }
        if l_current_marker == J2KMarker::SOT {
          break;
        }
        l_marker_handler = l_current_marker;
      }
      if l_marker_handler == J2KMarker::SIZ {
        /* Mark required SIZ marker as found */
        l_has_siz = 1i32
      }
      if l_marker_handler == J2KMarker::COD {
        /* Mark required COD marker as found */
        l_has_cod = 1i32
      }
      if l_marker_handler == J2KMarker::QCD {
        /* Mark required QCD marker as found */
        l_has_qcd = 1i32
      }
      /* Check if the marker is known and if it is the right place to find it */
      if p_j2k.m_specific_param.m_decoder.m_state & l_marker_handler.states() == J2KState::NONE {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Marker is not compliant with its position\n",
        );
        return 0i32;
      }
      /* Try to read 2 bytes (the marker size) from stream and copy them into the buffer */
      if opj_stream_read_data(
        p_stream,
        p_j2k.m_specific_param.m_decoder.m_header_data,
        2 as OPJ_SIZE_T,
        p_manager,
      ) != 2
      {
        event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
        return 0i32;
      }
      /* read 2 bytes as the marker size */
      opj_read_bytes(
        p_j2k.m_specific_param.m_decoder.m_header_data,
        &mut l_marker_size,
        2 as OPJ_UINT32,
      ); /* Subtract the size of the marker ID already read */
      if l_marker_size < 2u32 {
        event_msg!(p_manager, EVT_ERROR, "Invalid marker size\n",);
        return 0i32;
      }
      l_marker_size = (l_marker_size as core::ffi::c_uint).wrapping_sub(2u32) as OPJ_UINT32;
      /* Check if the marker size is compatible with the header data size */
      if l_marker_size > p_j2k.m_specific_param.m_decoder.m_header_data_size {
        let mut new_header_data = opj_realloc(
          p_j2k.m_specific_param.m_decoder.m_header_data as *mut core::ffi::c_void,
          l_marker_size as size_t,
        ) as *mut OPJ_BYTE;
        if new_header_data.is_null() {
          opj_free(p_j2k.m_specific_param.m_decoder.m_header_data as *mut core::ffi::c_void);
          p_j2k.m_specific_param.m_decoder.m_header_data = core::ptr::null_mut::<OPJ_BYTE>();
          p_j2k.m_specific_param.m_decoder.m_header_data_size = 0 as OPJ_UINT32;
          event_msg!(p_manager, EVT_ERROR, "Not enough memory to read header\n",);
          return 0i32;
        }
        p_j2k.m_specific_param.m_decoder.m_header_data = new_header_data;
        p_j2k.m_specific_param.m_decoder.m_header_data_size = l_marker_size
      }
      /* Try to read the rest of the marker segment from stream and copy them into the buffer */
      if opj_stream_read_data(
        p_stream,
        p_j2k.m_specific_param.m_decoder.m_header_data,
        l_marker_size as OPJ_SIZE_T,
        p_manager,
      ) != l_marker_size as usize
      {
        event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
        return 0i32;
      }
      /* Read the marker segment with the correct marker handler */
      if l_marker_handler.handler(
        p_j2k,
        p_j2k.m_specific_param.m_decoder.m_header_data,
        l_marker_size,
        p_manager,
      ) == 0
      {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Marker handler function failed to read the marker segment\n",
        );
        return 0i32;
      }
      /* Add the marker to the codestream index*/
      if 0i32
        == opj_j2k_add_mhmarker(
          p_j2k.cstr_index,
          l_marker_handler,
          (opj_stream_tell(p_stream) as OPJ_UINT32)
            .wrapping_sub(l_marker_size)
            .wrapping_sub(4u32) as OPJ_OFF_T,
          l_marker_size.wrapping_add(4u32),
        )
      {
        event_msg!(p_manager, EVT_ERROR, "Not enough memory to add mh marker\n",);
        return 0i32;
      }
      /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer */
      if opj_stream_read_data(
        p_stream,
        p_j2k.m_specific_param.m_decoder.m_header_data,
        2 as OPJ_SIZE_T,
        p_manager,
      ) != 2
      {
        event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
        return 0i32;
      }
      /* read 2 bytes as the new marker ID */
      l_current_marker = J2KMarker::from_buffer(p_j2k.m_specific_param.m_decoder.m_header_data);
    }
    if l_has_siz == 0i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "required SIZ marker not found in main header\n",
      );
      return 0i32;
    }
    if l_has_cod == 0i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "required COD marker not found in main header\n",
      );
      return 0i32;
    }
    if l_has_qcd == 0i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "required QCD marker not found in main header\n",
      );
      return 0i32;
    }
    if opj_j2k_merge_ppm(&mut p_j2k.m_cp, p_manager) == 0 {
      event_msg!(p_manager, EVT_ERROR, "Failed to merge PPM data\n",);
      return 0i32;
    }
    event_msg!(
      p_manager,
      EVT_INFO,
      "Main header has been correctly decoded.\n",
    );
    /* Position of the last element if the main header */
    (*p_j2k.cstr_index).main_head_end =
      (opj_stream_tell(p_stream) as OPJ_UINT32).wrapping_sub(2u32) as OPJ_OFF_T;
    /* Next step: read a tile-part header */
    p_j2k.m_specific_param.m_decoder.m_state = J2KState::TPHSOT;
    1i32
  }
}
/* *
 * Executes the given procedures on the given codec.
 *
 * @param       p_procedure_list        the list of procedures to execute
 * @param       p_j2k                           the jpeg2000 codec to execute the procedures on.
 * @param       p_stream                        the stream to execute the procedures on.
 * @param       p_manager                       the user manager.
 *
 * @return      true                            if all the procedures were successfully executed.
 */
fn opj_j2k_exec(
  p_j2k: &mut opj_j2k,
  list: &mut opj_j2k_proc_list_t,
  stream: &mut Stream,
  p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  list.execute(|p| (p)(p_j2k, stream, p_manager) != 0) as i32
}

/* *
 * Copies the decoding tile parameters onto all the tile parameters.
 * Creates also the tile decoder.
 */
/* FIXME DOC*/
fn opj_j2k_copy_default_tcp_and_create_tcd(
  mut p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_default_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_nb_tiles: OPJ_UINT32 = 0;
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut l_current_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_tccp_size: OPJ_UINT32 = 0;
    let mut l_mct_size: OPJ_UINT32 = 0;
    let mut l_image = core::ptr::null_mut::<opj_image_t>();
    let mut l_mcc_records_size: OPJ_UINT32 = 0;
    let mut l_mct_records_size: OPJ_UINT32 = 0;
    let mut l_src_mct_rec = core::ptr::null_mut::<opj_mct_data_t>();
    let mut l_dest_mct_rec = core::ptr::null_mut::<opj_mct_data_t>();
    let mut l_src_mcc_rec = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
    let mut l_dest_mcc_rec = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
    let mut l_offset: OPJ_UINT32 = 0;
    /* preconditions */

    l_image = p_j2k.m_private_image;
    l_nb_tiles = p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw);
    l_tcp = p_j2k.m_cp.tcps;
    l_tccp_size = (*l_image)
      .numcomps
      .wrapping_mul(core::mem::size_of::<opj_tccp_t>() as OPJ_UINT32);
    l_default_tcp = p_j2k.m_specific_param.m_decoder.m_default_tcp;
    l_mct_size = (*l_image)
      .numcomps
      .wrapping_mul((*l_image).numcomps)
      .wrapping_mul(core::mem::size_of::<OPJ_FLOAT32>() as OPJ_UINT32);
    /* For each tile */
    i = 0 as OPJ_UINT32;
    while i < l_nb_tiles {
      /* keep the tile-compo coding parameters pointer of the current tile coding parameters*/
      l_current_tccp = (*l_tcp).tccps;
      /*Copy default coding parameters into the current tile coding parameters*/
      memcpy(
        l_tcp as *mut core::ffi::c_void,
        l_default_tcp as *const core::ffi::c_void,
        core::mem::size_of::<opj_tcp_t>(),
      );
      /* Initialize some values of the current tile coding parameters*/
      (*l_tcp).cod = false;
      (*l_tcp).ppt = false;
      (*l_tcp).ppt_data = core::ptr::null_mut::<OPJ_BYTE>();
      (*l_tcp).m_current_tile_part_number = -(1i32);
      /* Remove memory not owned by this tile in case of early error return. */
      (*l_tcp).m_mct_decoding_matrix = core::ptr::null_mut::<OPJ_FLOAT32>();
      (*l_tcp).m_nb_max_mct_records = 0 as OPJ_UINT32;
      (*l_tcp).m_mct_records = core::ptr::null_mut::<opj_mct_data_t>();
      (*l_tcp).m_nb_max_mcc_records = 0 as OPJ_UINT32;
      (*l_tcp).m_mcc_records = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
      /* Reconnect the tile-compo coding parameters pointer to the current tile coding parameters*/
      (*l_tcp).tccps = l_current_tccp;
      /* Get the mct_decoding_matrix of the dflt_tile_cp and copy them into the current tile cp*/
      if !(*l_default_tcp).m_mct_decoding_matrix.is_null() {
        (*l_tcp).m_mct_decoding_matrix = opj_malloc(l_mct_size as size_t) as *mut OPJ_FLOAT32;
        if (*l_tcp).m_mct_decoding_matrix.is_null() {
          return 0i32;
        }
        memcpy(
          (*l_tcp).m_mct_decoding_matrix as *mut core::ffi::c_void,
          (*l_default_tcp).m_mct_decoding_matrix as *const core::ffi::c_void,
          l_mct_size as usize,
        );
      }
      /* Get the mct_record of the dflt_tile_cp and copy them into the current tile cp*/
      l_mct_records_size = (*l_default_tcp)
        .m_nb_max_mct_records
        .wrapping_mul(core::mem::size_of::<opj_mct_data_t>() as OPJ_UINT32);
      (*l_tcp).m_mct_records = opj_malloc(l_mct_records_size as size_t) as *mut opj_mct_data_t;
      if (*l_tcp).m_mct_records.is_null() {
        return 0i32;
      }
      memcpy(
        (*l_tcp).m_mct_records as *mut core::ffi::c_void,
        (*l_default_tcp).m_mct_records as *const core::ffi::c_void,
        l_mct_records_size as usize,
      );
      /* Copy the mct record data from dflt_tile_cp to the current tile*/
      l_src_mct_rec = (*l_default_tcp).m_mct_records;
      l_dest_mct_rec = (*l_tcp).m_mct_records;
      j = 0 as OPJ_UINT32;
      while j < (*l_default_tcp).m_nb_mct_records {
        if !(*l_src_mct_rec).m_data.is_null() {
          (*l_dest_mct_rec).m_data =
            opj_malloc((*l_src_mct_rec).m_data_size as size_t) as *mut OPJ_BYTE;
          if (*l_dest_mct_rec).m_data.is_null() {
            return 0i32;
          }
          memcpy(
            (*l_dest_mct_rec).m_data as *mut core::ffi::c_void,
            (*l_src_mct_rec).m_data as *const core::ffi::c_void,
            (*l_src_mct_rec).m_data_size as usize,
          );
        }
        l_src_mct_rec = l_src_mct_rec.offset(1);
        l_dest_mct_rec = l_dest_mct_rec.offset(1);
        /* Update with each pass to free exactly what has been allocated on early return. */
        (*l_tcp).m_nb_max_mct_records =
          ((*l_tcp).m_nb_max_mct_records as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32;
        j += 1;
      }
      /* Get the mcc_record of the dflt_tile_cp and copy them into the current tile cp*/
      l_mcc_records_size = (*l_default_tcp)
        .m_nb_max_mcc_records
        .wrapping_mul(core::mem::size_of::<opj_simple_mcc_decorrelation_data_t>() as OPJ_UINT32);
      (*l_tcp).m_mcc_records =
        opj_malloc(l_mcc_records_size as size_t) as *mut opj_simple_mcc_decorrelation_data_t;
      if (*l_tcp).m_mcc_records.is_null() {
        return 0i32;
      }
      memcpy(
        (*l_tcp).m_mcc_records as *mut core::ffi::c_void,
        (*l_default_tcp).m_mcc_records as *const core::ffi::c_void,
        l_mcc_records_size as usize,
      );
      (*l_tcp).m_nb_max_mcc_records = (*l_default_tcp).m_nb_max_mcc_records;
      /* Copy the mcc record data from dflt_tile_cp to the current tile*/
      l_src_mcc_rec = (*l_default_tcp).m_mcc_records;
      l_dest_mcc_rec = (*l_tcp).m_mcc_records;
      j = 0 as OPJ_UINT32;
      while j < (*l_default_tcp).m_nb_max_mcc_records {
        if !(*l_src_mcc_rec).m_decorrelation_array.is_null() {
          l_offset = (*l_src_mcc_rec)
            .m_decorrelation_array
            .offset_from((*l_default_tcp).m_mct_records) as core::ffi::c_long
            as OPJ_UINT32;
          (*l_dest_mcc_rec).m_decorrelation_array = (*l_tcp).m_mct_records.offset(l_offset as isize)
        }
        if !(*l_src_mcc_rec).m_offset_array.is_null() {
          l_offset = (*l_src_mcc_rec)
            .m_offset_array
            .offset_from((*l_default_tcp).m_mct_records) as core::ffi::c_long
            as OPJ_UINT32;
          (*l_dest_mcc_rec).m_offset_array = (*l_tcp).m_mct_records.offset(l_offset as isize)
        }
        l_src_mcc_rec = l_src_mcc_rec.offset(1);
        l_dest_mcc_rec = l_dest_mcc_rec.offset(1);
        j += 1;
      }
      /* Copy all the dflt_tile_compo_cp to the current tile cp */
      memcpy(
        l_current_tccp as *mut core::ffi::c_void,
        (*l_default_tcp).tccps as *const core::ffi::c_void,
        l_tccp_size as usize,
      );
      /* Move to next tile cp*/
      l_tcp = l_tcp.offset(1);
      i += 1;
    }
    // Init the current tile decoder
    if opj_tcd_init(&mut p_j2k.m_tcd, l_image, &mut p_j2k.m_cp) == 0 {
      event_msg!(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n",);
      return 0i32;
    }
    1i32
  }
}

impl Drop for opj_j2k {
  fn drop(&mut self) {
    unsafe {
      if self.m_is_decoder != 0 {
        if !self.m_specific_param.m_decoder.m_default_tcp.is_null() {
          opj_j2k_tcp_destroy(self.m_specific_param.m_decoder.m_default_tcp);
          opj_free(self.m_specific_param.m_decoder.m_default_tcp as *mut core::ffi::c_void);
          self.m_specific_param.m_decoder.m_default_tcp = core::ptr::null_mut::<opj_tcp_t>()
        }
        if !self.m_specific_param.m_decoder.m_header_data.is_null() {
          opj_free(self.m_specific_param.m_decoder.m_header_data as *mut core::ffi::c_void);
          self.m_specific_param.m_decoder.m_header_data = core::ptr::null_mut::<OPJ_BYTE>();
          self.m_specific_param.m_decoder.m_header_data_size = 0 as OPJ_UINT32
        }
        opj_free(
          self.m_specific_param.m_decoder.m_comps_indices_to_decode as *mut core::ffi::c_void,
        );
        self.m_specific_param.m_decoder.m_comps_indices_to_decode =
          core::ptr::null_mut::<OPJ_UINT32>();
        self.m_specific_param.m_decoder.m_numcomps_to_decode = 0 as OPJ_UINT32
      } else {
        if !self
          .m_specific_param
          .m_encoder
          .m_encoded_tile_data
          .is_null()
        {
          opj_free(self.m_specific_param.m_encoder.m_encoded_tile_data as *mut core::ffi::c_void);
          self.m_specific_param.m_encoder.m_encoded_tile_data = core::ptr::null_mut::<OPJ_BYTE>()
        }
        if !self
          .m_specific_param
          .m_encoder
          .m_tlm_sot_offsets_buffer
          .is_null()
        {
          opj_free(
            self.m_specific_param.m_encoder.m_tlm_sot_offsets_buffer as *mut core::ffi::c_void,
          );
          self.m_specific_param.m_encoder.m_tlm_sot_offsets_buffer =
            core::ptr::null_mut::<OPJ_BYTE>();
          self.m_specific_param.m_encoder.m_tlm_sot_offsets_current =
            core::ptr::null_mut::<OPJ_BYTE>()
        }
        if !self.m_specific_param.m_encoder.m_header_tile_data.is_null() {
          opj_free(self.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
          self.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>();
          self.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32
        }
      }
      opj_j2k_cp_destroy(&mut self.m_cp);
      memset(
        &mut self.m_cp as *mut opj_cp_t as *mut core::ffi::c_void,
        0i32,
        core::mem::size_of::<opj_cp_t>(),
      );
      j2k_destroy_cstr_index(self.cstr_index);
      self.cstr_index = core::ptr::null_mut::<opj_codestream_index_t>();
      opj_image_destroy(self.m_private_image);
      self.m_private_image = core::ptr::null_mut::<opj_image_t>();
      opj_image_destroy(self.m_output_image);
      self.m_output_image = core::ptr::null_mut::<opj_image_t>();
    }
  }
}

pub(crate) fn j2k_destroy_cstr_index(mut p_cstr_ind: *mut opj_codestream_index_t) {
  unsafe {
    if !p_cstr_ind.is_null() {
      if !(*p_cstr_ind).marker.is_null() {
        opj_free((*p_cstr_ind).marker as *mut core::ffi::c_void);
        (*p_cstr_ind).marker = core::ptr::null_mut::<opj_marker_info_t>()
      }
      if !(*p_cstr_ind).tile_index.is_null() {
        let mut it_tile = 0 as OPJ_UINT32;
        it_tile = 0 as OPJ_UINT32;
        while it_tile < (*p_cstr_ind).nb_of_tiles {
          if !(*(*p_cstr_ind).tile_index.offset(it_tile as isize))
            .packet_index
            .is_null()
          {
            opj_free(
              (*(*p_cstr_ind).tile_index.offset(it_tile as isize)).packet_index
                as *mut core::ffi::c_void,
            );
            let fresh30 = &mut (*(*p_cstr_ind).tile_index.offset(it_tile as isize)).packet_index;
            *fresh30 = core::ptr::null_mut::<opj_packet_info_t>()
          }
          if !(*(*p_cstr_ind).tile_index.offset(it_tile as isize))
            .tp_index
            .is_null()
          {
            opj_free(
              (*(*p_cstr_ind).tile_index.offset(it_tile as isize)).tp_index
                as *mut core::ffi::c_void,
            );
            let fresh31 = &mut (*(*p_cstr_ind).tile_index.offset(it_tile as isize)).tp_index;
            *fresh31 = core::ptr::null_mut::<opj_tp_index_t>()
          }
          if !(*(*p_cstr_ind).tile_index.offset(it_tile as isize))
            .marker
            .is_null()
          {
            opj_free(
              (*(*p_cstr_ind).tile_index.offset(it_tile as isize)).marker as *mut core::ffi::c_void,
            );
            let fresh32 = &mut (*(*p_cstr_ind).tile_index.offset(it_tile as isize)).marker;
            *fresh32 = core::ptr::null_mut::<opj_marker_info_t>()
          }
          it_tile += 1;
        }
        opj_free((*p_cstr_ind).tile_index as *mut core::ffi::c_void);
        (*p_cstr_ind).tile_index = core::ptr::null_mut::<opj_tile_index_t>()
      }
      opj_free(p_cstr_ind as *mut core::ffi::c_void);
    };
  }
}

/* *
 * Destroys a tile coding parameter structure.
 *
 * @param       p_tcp           the tile coding parameter to destroy.
 */
fn opj_j2k_tcp_destroy(mut p_tcp: *mut opj_tcp_t) {
  unsafe {
    if p_tcp.is_null() {
      return;
    }
    if !(*p_tcp).ppt_markers.is_null() {
      let mut i: OPJ_UINT32 = 0;
      i = 0u32;
      while i < (*p_tcp).ppt_markers_count {
        if !(*(*p_tcp).ppt_markers.offset(i as isize)).m_data.is_null() {
          opj_free((*(*p_tcp).ppt_markers.offset(i as isize)).m_data as *mut core::ffi::c_void);
        }
        i += 1;
      }
      (*p_tcp).ppt_markers_count = 0u32;
      opj_free((*p_tcp).ppt_markers as *mut core::ffi::c_void);
      (*p_tcp).ppt_markers = core::ptr::null_mut::<opj_ppx>()
    }
    if !(*p_tcp).ppt_buffer.is_null() {
      opj_free((*p_tcp).ppt_buffer as *mut core::ffi::c_void);
      (*p_tcp).ppt_buffer = core::ptr::null_mut::<OPJ_BYTE>()
    }
    if !(*p_tcp).tccps.is_null() {
      opj_free((*p_tcp).tccps as *mut core::ffi::c_void);
      (*p_tcp).tccps = core::ptr::null_mut::<opj_tccp_t>()
    }
    if !(*p_tcp).m_mct_coding_matrix.is_null() {
      opj_free((*p_tcp).m_mct_coding_matrix as *mut core::ffi::c_void);
      (*p_tcp).m_mct_coding_matrix = core::ptr::null_mut::<OPJ_FLOAT32>()
    }
    if !(*p_tcp).m_mct_decoding_matrix.is_null() {
      opj_free((*p_tcp).m_mct_decoding_matrix as *mut core::ffi::c_void);
      (*p_tcp).m_mct_decoding_matrix = core::ptr::null_mut::<OPJ_FLOAT32>()
    }
    if !(*p_tcp).m_mcc_records.is_null() {
      opj_free((*p_tcp).m_mcc_records as *mut core::ffi::c_void);
      (*p_tcp).m_mcc_records = core::ptr::null_mut::<opj_simple_mcc_decorrelation_data_t>();
      (*p_tcp).m_nb_max_mcc_records = 0 as OPJ_UINT32;
      (*p_tcp).m_nb_mcc_records = 0 as OPJ_UINT32
    }
    if !(*p_tcp).m_mct_records.is_null() {
      let mut l_mct_data = (*p_tcp).m_mct_records;
      let mut i_0: OPJ_UINT32 = 0;
      i_0 = 0 as OPJ_UINT32;
      while i_0 < (*p_tcp).m_nb_mct_records {
        if !(*l_mct_data).m_data.is_null() {
          opj_free((*l_mct_data).m_data as *mut core::ffi::c_void);
          (*l_mct_data).m_data = core::ptr::null_mut::<OPJ_BYTE>()
        }
        l_mct_data = l_mct_data.offset(1);
        i_0 += 1;
      }
      opj_free((*p_tcp).m_mct_records as *mut core::ffi::c_void);
      (*p_tcp).m_mct_records = core::ptr::null_mut::<opj_mct_data_t>()
    }
    if !(*p_tcp).mct_norms.is_null() {
      opj_free((*p_tcp).mct_norms as *mut core::ffi::c_void);
      (*p_tcp).mct_norms = core::ptr::null_mut::<OPJ_FLOAT64>()
    }
    opj_j2k_tcp_data_destroy(p_tcp);
  }
}
/* *
 * Destroys the data inside a tile coding parameter structure.
 *
 * @param       p_tcp           the tile coding parameter which contain data to destroy.
 */
fn opj_j2k_tcp_data_destroy(mut p_tcp: *mut opj_tcp_t) {
  unsafe {
    if !(*p_tcp).m_data.is_null() {
      opj_free((*p_tcp).m_data as *mut core::ffi::c_void);
      (*p_tcp).m_data = core::ptr::null_mut::<OPJ_BYTE>();
      (*p_tcp).m_data_size = 0 as OPJ_UINT32
    };
  }
}
/* *
 * Destroys a coding parameter structure.
 *
 * @param       p_cp            the coding parameter to destroy.
 */
fn opj_j2k_cp_destroy(mut p_cp: *mut opj_cp_t) {
  unsafe {
    let mut l_nb_tiles: OPJ_UINT32 = 0; /* ppm_data belongs to the allocated buffer pointed by ppm_buffer */
    let mut l_current_tile = core::ptr::null_mut::<opj_tcp_t>();
    if p_cp.is_null() {
      return;
    }
    if !(*p_cp).tcps.is_null() {
      let mut i: OPJ_UINT32 = 0;
      l_current_tile = (*p_cp).tcps;
      l_nb_tiles = (*p_cp).th.wrapping_mul((*p_cp).tw);
      i = 0u32;
      while i < l_nb_tiles {
        opj_j2k_tcp_destroy(l_current_tile);
        l_current_tile = l_current_tile.offset(1);
        i += 1;
      }
      opj_free((*p_cp).tcps as *mut core::ffi::c_void);
      (*p_cp).tcps = core::ptr::null_mut::<opj_tcp_t>()
    }
    if !(*p_cp).ppm_markers.is_null() {
      let mut i_0: OPJ_UINT32 = 0;
      i_0 = 0u32;
      while i_0 < (*p_cp).ppm_markers_count {
        if !(*(*p_cp).ppm_markers.offset(i_0 as isize)).m_data.is_null() {
          opj_free((*(*p_cp).ppm_markers.offset(i_0 as isize)).m_data as *mut core::ffi::c_void);
        }
        i_0 += 1;
      }
      (*p_cp).ppm_markers_count = 0u32;
      opj_free((*p_cp).ppm_markers as *mut core::ffi::c_void);
      (*p_cp).ppm_markers = core::ptr::null_mut::<opj_ppx>()
    }
    opj_free((*p_cp).ppm_buffer as *mut core::ffi::c_void);
    (*p_cp).ppm_buffer = core::ptr::null_mut::<OPJ_BYTE>();
    (*p_cp).ppm_data = core::ptr::null_mut::<OPJ_BYTE>();
    opj_free((*p_cp).comment as *mut core::ffi::c_void);
    (*p_cp).comment = core::ptr::null_mut::<OPJ_CHAR>();
    if !(*p_cp).m_is_decoder {
      opj_free((*p_cp).m_specific_param.m_enc.m_matrice as *mut core::ffi::c_void);
      (*p_cp).m_specific_param.m_enc.m_matrice = core::ptr::null_mut::<OPJ_INT32>()
    };
  }
}
/* *
 * Checks for invalid number of tile-parts in SOT marker (TPsot==TNsot). See issue 254.
 *
 * @param       p_stream            the stream to read data from.
 * @param       tile_no             tile number we're looking for.
 * @param       p_correction_needed output value. if true, non conformant codestream needs TNsot correction.
 * @param       p_manager       the user event manager.
 *
 * @return true if the function was successful, false else.
 */
fn opj_j2k_need_nb_tile_parts_correction(
  mut p_stream: &mut Stream,
  mut tile_no: OPJ_UINT32,
  mut p_correction_needed: *mut OPJ_BOOL,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_header_data: [OPJ_BYTE; 10] = [0; 10];
    let mut l_stream_pos_backup: OPJ_OFF_T = 0;
    let mut l_current_marker = J2KMarker::UNK(0);
    let mut l_marker_size: OPJ_UINT32 = 0;
    let mut l_tile_no: OPJ_UINT32 = 0;
    let mut l_tot_len: OPJ_UINT32 = 0;
    let mut l_current_part: OPJ_UINT32 = 0;
    let mut l_num_parts: OPJ_UINT32 = 0;
    /* initialize to no correction needed */
    *p_correction_needed = 0i32;
    if opj_stream_has_seek(p_stream) == 0 {
      /* We can't do much in this case, seek is needed */
      return 1i32;
    }
    l_stream_pos_backup = opj_stream_tell(p_stream);
    if l_stream_pos_backup == -(1i32) as i64 {
      /* let's do nothing */
      return 1i32;
    }
    loop {
      /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer */
      if opj_stream_read_data(
        p_stream,
        l_header_data.as_mut_ptr(),
        2 as OPJ_SIZE_T,
        p_manager,
      ) != 2
      {
        /* assume all is OK */
        if opj_stream_seek(p_stream, l_stream_pos_backup, p_manager) == 0 {
          return 0i32;
        }
        return 1i32;
      }
      /* Read 2 bytes from buffer as the new marker ID */
      l_current_marker = J2KMarker::from_buffer(l_header_data.as_mut_ptr());
      if l_current_marker != J2KMarker::SOT {
        /* assume all is OK */
        if opj_stream_seek(p_stream, l_stream_pos_backup, p_manager) == 0 {
          return 0i32;
        }
        return 1i32;
      }
      /* Try to read 2 bytes (the marker size) from stream and copy them into the buffer */
      if opj_stream_read_data(
        p_stream,
        l_header_data.as_mut_ptr(),
        2 as OPJ_SIZE_T,
        p_manager,
      ) != 2
      {
        event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
        return 0i32;
      }
      /* Read 2 bytes from the buffer as the marker size */
      opj_read_bytes(
        l_header_data.as_mut_ptr(),
        &mut l_marker_size,
        2 as OPJ_UINT32,
      );
      /* Check marker size for SOT Marker */
      if l_marker_size != 10u32 {
        event_msg!(p_manager, EVT_ERROR, "Inconsistent marker size\n",);
        return 0i32;
      }
      l_marker_size = (l_marker_size as core::ffi::c_uint).wrapping_sub(2u32) as OPJ_UINT32;
      if opj_stream_read_data(
        p_stream,
        l_header_data.as_mut_ptr(),
        l_marker_size as OPJ_SIZE_T,
        p_manager,
      ) != l_marker_size as usize
      {
        event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
        return 0i32;
      }
      if opj_j2k_get_sot_values(
        l_header_data.as_mut_ptr(),
        l_marker_size,
        &mut l_tile_no,
        &mut l_tot_len,
        &mut l_current_part,
        &mut l_num_parts,
        p_manager,
      ) == 0
      {
        return 0i32;
      }
      if l_tile_no == tile_no {
        break;
      }
      if l_tot_len < 14u32 {
        /* last SOT until EOC or invalid Psot value */
        /* assume all is OK */
        if opj_stream_seek(p_stream, l_stream_pos_backup, p_manager) == 0 {
          return 0i32;
        }
        return 1i32;
      }
      l_tot_len = (l_tot_len as core::ffi::c_uint).wrapping_sub(12u32) as OPJ_UINT32;
      /* look for next SOT marker */
      if opj_stream_skip(p_stream, l_tot_len as OPJ_OFF_T, p_manager) != l_tot_len as OPJ_OFF_T {
        /* assume all is OK */
        if opj_stream_seek(p_stream, l_stream_pos_backup, p_manager) == 0 {
          return 0i32;
        }
        return 1i32;
      }
    }
    /* check for correction */
    if l_current_part == l_num_parts {
      *p_correction_needed = 1i32
    }
    if opj_stream_seek(p_stream, l_stream_pos_backup, p_manager) == 0 {
      return 0i32;
    }
    1i32
  }
}

pub(crate) fn opj_j2k_read_tile_header(
  p_j2k: &mut opj_j2k,
  p_stream: &mut Stream,
  tile_info: &mut TileInfo,
  p_manager: &mut opj_event_mgr,
) -> bool {
  unsafe {
    let mut l_current_marker = J2KMarker::SOT;
    let mut l_marker_size: OPJ_UINT32 = 0;
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let l_nb_tiles = p_j2k.m_cp.tw.wrapping_mul(p_j2k.m_cp.th);
    /* preconditions */

    /* Reach the End Of Codestream ?*/
    if p_j2k.m_specific_param.m_decoder.m_state == J2KState::EOC {
      l_current_marker = J2KMarker::EOC;
    } else if p_j2k.m_specific_param.m_decoder.m_state != J2KState::TPHSOT {
      return false;
    }
    /* We need to encounter a SOT marker (a new tile-part header) */
    /* Read into the codestream until reach the EOC or ! can_decode ??? FIXME */
    while !p_j2k.m_specific_param.m_decoder.m_can_decode && l_current_marker != J2KMarker::EOC {
      /* Try to read until the Start Of Data is detected */
      while l_current_marker != J2KMarker::SOD {
        if opj_stream_get_number_byte_left(p_stream) == 0i64 {
          p_j2k.m_specific_param.m_decoder.m_state = J2KState::NEOC;
          break;
        } else {
          /* Try to read 2 bytes (the marker size) from stream and copy them into the buffer */
          if opj_stream_read_data(
            p_stream,
            p_j2k.m_specific_param.m_decoder.m_header_data,
            2 as OPJ_SIZE_T,
            p_manager,
          ) != 2
          {
            event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
            return false;
          }
          /* Read 2 bytes from the buffer as the marker size */
          opj_read_bytes(
            p_j2k.m_specific_param.m_decoder.m_header_data,
            &mut l_marker_size,
            2 as OPJ_UINT32,
          );
          /* Check marker size (does not include marker ID but includes marker size) */
          if l_marker_size < 2u32 {
            event_msg!(p_manager, EVT_ERROR, "Inconsistent marker size\n",);
            return false;
          }
          /* cf. https://code.google.com/p/openjpeg/issues/detail?id=226 */
          if l_current_marker == J2KMarker::UNK(0x8080u32)
            && opj_stream_get_number_byte_left(p_stream) == 0i64
          {
            p_j2k.m_specific_param.m_decoder.m_state = J2KState::NEOC;
            break;
          } else {
            /* Why this condition? FIXME */
            if p_j2k.m_specific_param.m_decoder.m_state & J2KState::TPH != J2KState::NONE {
              p_j2k.m_specific_param.m_decoder.m_sot_length =
                (p_j2k.m_specific_param.m_decoder.m_sot_length as core::ffi::c_uint)
                  .wrapping_sub(l_marker_size.wrapping_add(2u32)) as OPJ_UINT32
            } /* Subtract the size of the marker ID already read */
            l_marker_size =
              (l_marker_size as core::ffi::c_uint).wrapping_sub(2u32) as OPJ_UINT32 as OPJ_UINT32;
            /* Get the marker handler from the marker ID */
            let l_marker_handler = l_current_marker;
            /* Check if the marker is known and if it is the right place to find it */
            if p_j2k.m_specific_param.m_decoder.m_state & l_marker_handler.states()
              == J2KState::NONE
            {
              event_msg!(
                p_manager,
                EVT_ERROR,
                "Marker is not compliant with its position\n",
              );
              return false;
            }
            /* FIXME manage case of unknown marker as in the main header ? */
            /* Check if the marker size is compatible with the header data size */
            if l_marker_size > p_j2k.m_specific_param.m_decoder.m_header_data_size {
              let mut new_header_data = core::ptr::null_mut::<OPJ_BYTE>();
              /* If we are here, this means we consider this marker as known & we will read it */
              /* Check enough bytes left in stream before allocation */
              if l_marker_size as OPJ_OFF_T > opj_stream_get_number_byte_left(p_stream) {
                event_msg!(
                  p_manager,
                  EVT_ERROR,
                  "Marker size inconsistent with stream length\n",
                );
                return false;
              }
              new_header_data = opj_realloc(
                p_j2k.m_specific_param.m_decoder.m_header_data as *mut core::ffi::c_void,
                l_marker_size as size_t,
              ) as *mut OPJ_BYTE;
              if new_header_data.is_null() {
                opj_free(p_j2k.m_specific_param.m_decoder.m_header_data as *mut core::ffi::c_void);
                p_j2k.m_specific_param.m_decoder.m_header_data = core::ptr::null_mut::<OPJ_BYTE>();
                p_j2k.m_specific_param.m_decoder.m_header_data_size = 0 as OPJ_UINT32;
                event_msg!(p_manager, EVT_ERROR, "Not enough memory to read header\n",);
                return false;
              }
              p_j2k.m_specific_param.m_decoder.m_header_data = new_header_data;
              p_j2k.m_specific_param.m_decoder.m_header_data_size = l_marker_size
            }
            /* Try to read the rest of the marker segment from stream and copy them into the buffer */
            if opj_stream_read_data(
              p_stream,
              p_j2k.m_specific_param.m_decoder.m_header_data,
              l_marker_size as OPJ_SIZE_T,
              p_manager,
            ) != l_marker_size as usize
            {
              event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
              return false;
            }
            /* Read the marker segment with the correct marker handler */
            if l_marker_handler.handler(
              p_j2k,
              p_j2k.m_specific_param.m_decoder.m_header_data,
              l_marker_size,
              p_manager,
            ) == 0
            {
              event_msg!(
                p_manager,
                EVT_ERROR,
                "Fail to read the current marker segment (%#x)\n",
                l_current_marker.as_u32(),
              );
              return false;
            }
            /* Add the marker to the codestream index*/
            if 0i32
              == opj_j2k_add_tlmarker(
                p_j2k.m_current_tile_number,
                p_j2k.cstr_index,
                l_marker_handler,
                (opj_stream_tell(p_stream) as OPJ_UINT32)
                  .wrapping_sub(l_marker_size)
                  .wrapping_sub(4u32) as OPJ_OFF_T,
                l_marker_size.wrapping_add(4u32),
              )
            {
              event_msg!(p_manager, EVT_ERROR, "Not enough memory to add tl marker\n",);
              return false;
            }
            /* Keep the position of the last SOT marker read */
            if l_marker_handler == J2KMarker::SOT {
              let mut sot_pos = (opj_stream_tell(p_stream) as OPJ_UINT32)
                .wrapping_sub(l_marker_size)
                .wrapping_sub(4u32);
              if sot_pos as i64 > p_j2k.m_specific_param.m_decoder.m_last_sot_read_pos {
                p_j2k.m_specific_param.m_decoder.m_last_sot_read_pos = sot_pos as OPJ_OFF_T
              }
            }
            if p_j2k.m_specific_param.m_decoder.m_skip_data {
              /* Skip the rest of the tile part header*/
              if opj_stream_skip(
                p_stream,
                p_j2k.m_specific_param.m_decoder.m_sot_length as OPJ_OFF_T,
                p_manager,
              ) != p_j2k.m_specific_param.m_decoder.m_sot_length as i64
              {
                event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
                return false;
              }
              l_current_marker = J2KMarker::SOD
            /* Normally we reached a SOD */
            } else {
              /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer*/
              if opj_stream_read_data(
                p_stream,
                p_j2k.m_specific_param.m_decoder.m_header_data,
                2 as OPJ_SIZE_T,
                p_manager,
              ) != 2
              {
                event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
                return false;
              }
              /* Read 2 bytes from the buffer as the new marker ID */
              l_current_marker =
                J2KMarker::from_buffer(p_j2k.m_specific_param.m_decoder.m_header_data);
            }
          }
        }
      }
      if opj_stream_get_number_byte_left(p_stream) == 0i64
        && p_j2k.m_specific_param.m_decoder.m_state == J2KState::NEOC
      {
        break;
      }
      /* If we didn't skip data before, we need to read the SOD marker*/
      if !p_j2k.m_specific_param.m_decoder.m_skip_data {
        /* Try to read the SOD marker and skip data ? FIXME */
        if opj_j2k_read_sod(p_j2k, p_stream, p_manager) == 0 {
          return false;
        }
        if p_j2k.m_specific_param.m_decoder.m_can_decode
          && !p_j2k
            .m_specific_param
            .m_decoder
            .m_nb_tile_parts_correction_checked
        {
          /* Issue 254 */
          let mut l_correction_needed: OPJ_BOOL = 0;
          p_j2k
            .m_specific_param
            .m_decoder
            .m_nb_tile_parts_correction_checked = true;
          if opj_j2k_need_nb_tile_parts_correction(
            p_stream,
            p_j2k.m_current_tile_number,
            &mut l_correction_needed,
            p_manager,
          ) == 0
          {
            event_msg!(
              p_manager,
              EVT_ERROR,
              "opj_j2k_apply_nb_tile_parts_correction error\n",
            );
            return false;
          }
          if l_correction_needed != 0 {
            let mut l_tile_no: OPJ_UINT32 = 0;
            p_j2k.m_specific_param.m_decoder.m_can_decode = false;
            p_j2k.m_specific_param.m_decoder.m_nb_tile_parts_correction = true;
            /* correct tiles */
            l_tile_no = 0u32;
            while l_tile_no < l_nb_tiles {
              if (*p_j2k.m_cp.tcps.offset(l_tile_no as isize)).m_nb_tile_parts != 0u32 {
                let fresh33 = &mut (*p_j2k.m_cp.tcps.offset(l_tile_no as isize)).m_nb_tile_parts;
                *fresh33 = (*fresh33 as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32
              }
              l_tile_no += 1;
            }
            event_msg!(
              p_manager,
              EVT_WARNING,
              "Non conformant codestream TPsot==TNsot.\n",
            );
          }
        }
      } else {
        /* Indicate we will try to read a new tile-part header*/
        p_j2k.m_specific_param.m_decoder.m_skip_data = false;
        p_j2k.m_specific_param.m_decoder.m_can_decode = false;
        p_j2k.m_specific_param.m_decoder.m_state = J2KState::TPHSOT
      }
      if p_j2k.m_specific_param.m_decoder.m_can_decode {
        continue;
      }
      /* Try to read 2 bytes (the next marker ID) from stream and copy them into the buffer */
      if opj_stream_read_data(
        p_stream,
        p_j2k.m_specific_param.m_decoder.m_header_data,
        2 as OPJ_SIZE_T,
        p_manager,
      ) != 2
      {
        /* Deal with likely non conformant SPOT6 files, where the last */
        /* row of tiles have TPsot == 0 and TNsot == 0, and missing EOC, */
        /* but no other tile-parts were found. */
        if p_j2k.m_current_tile_number.wrapping_add(1u32) == l_nb_tiles {
          let mut l_tile_no_0: OPJ_UINT32 = 0;
          l_tile_no_0 = 0u32;
          while l_tile_no_0 < l_nb_tiles {
            if (*p_j2k.m_cp.tcps.offset(l_tile_no_0 as isize)).m_current_tile_part_number == 0i32
              && (*p_j2k.m_cp.tcps.offset(l_tile_no_0 as isize)).m_nb_tile_parts == 0u32
            {
              break;
            }
            l_tile_no_0 += 1;
          }
          if l_tile_no_0 < l_nb_tiles {
            event_msg!(p_manager, EVT_INFO,
                                  "Tile %u has TPsot == 0 and TNsot == 0, but no other tile-parts were found. EOC is also missing.\n",
                                  l_tile_no_0);
            p_j2k.m_current_tile_number = l_tile_no_0;
            l_current_marker = J2KMarker::EOC;
            p_j2k.m_specific_param.m_decoder.m_state = J2KState::EOC;
            break;
          }
        }
        event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
        return false;
      } else {
        /* Read 2 bytes from buffer as the new marker ID */
        l_current_marker = J2KMarker::from_buffer(p_j2k.m_specific_param.m_decoder.m_header_data);
      }
    }
    /* Current marker is the EOC marker ?*/
    if l_current_marker == J2KMarker::EOC
      && p_j2k.m_specific_param.m_decoder.m_state != J2KState::EOC
    {
      p_j2k.m_current_tile_number = 0 as OPJ_UINT32;
      p_j2k.m_specific_param.m_decoder.m_state = J2KState::EOC
    }
    /* Deal with tiles that have a single tile-part with TPsot == 0 and TNsot == 0 */
    if !p_j2k.m_specific_param.m_decoder.m_can_decode {
      l_tcp = p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize);
      while p_j2k.m_current_tile_number < l_nb_tiles && (*l_tcp).m_data.is_null() {
        p_j2k.m_current_tile_number = p_j2k.m_current_tile_number.wrapping_add(1);
        l_tcp = l_tcp.offset(1)
      }
      if p_j2k.m_current_tile_number == l_nb_tiles {
        tile_info.go_on = false;
        return true;
      }
    }
    if opj_j2k_merge_ppt(
      p_j2k.m_cp.tcps.offset(p_j2k.m_current_tile_number as isize),
      p_manager,
    ) == 0
    {
      event_msg!(p_manager, EVT_ERROR, "Failed to merge PPT data\n",);
      return false;
    }
    /*FIXME ???*/
    if opj_tcd_init_decode_tile(&mut p_j2k.m_tcd, p_j2k.m_current_tile_number, p_manager) == 0 {
      event_msg!(p_manager, EVT_ERROR, "Cannot decode tile, memory error\n",);
      return false;
    }
    event_msg!(
      p_manager,
      EVT_INFO,
      "Header of tile %d / %d has been read.\n",
      p_j2k.m_current_tile_number.wrapping_add(1u32),
      p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw),
    );
    tile_info.index = p_j2k.m_current_tile_number;
    tile_info.go_on = true;
    /* For internal use in j2k.c, we don't need this */
    /* This is just needed for folks using the opj_read_tile_header() / opj_decode_tile_data() combo */
    if let Some(data_size) = &mut tile_info.data_size {
      *data_size = opj_tcd_get_decoded_tile_size(&mut p_j2k.m_tcd, 0i32);
      if *data_size == (2147483647u32).wrapping_mul(2u32).wrapping_add(1u32) {
        return false;
      }
    }
    tile_info.x0 = p_j2k.m_tcd.tcd_image.tiles.x0;
    tile_info.y0 = p_j2k.m_tcd.tcd_image.tiles.y0;
    tile_info.x1 = p_j2k.m_tcd.tcd_image.tiles.x1;
    tile_info.y1 = p_j2k.m_tcd.tcd_image.tiles.y1;
    tile_info.nb_comps = p_j2k.m_tcd.tcd_image.tiles.numcomps;
    p_j2k.m_specific_param.m_decoder.m_state |= J2KState::DATA;
    true
  }
}

pub(crate) fn opj_j2k_decode_tile(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_index: OPJ_UINT32,
  mut p_data: Option<&mut [u8]>,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_current_marker = J2KMarker::UNK(0);
    let mut l_data: [OPJ_BYTE; 2] = [0; 2];
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_image_for_bounds = core::ptr::null_mut::<opj_image_t>();
    /* preconditions */

    if p_j2k.m_specific_param.m_decoder.m_state & J2KState::DATA == J2KState::NONE
      || p_tile_index != p_j2k.m_current_tile_number
    {
      return 0i32;
    }
    l_tcp = &mut *p_j2k.m_cp.tcps.offset(p_tile_index as isize) as *mut opj_tcp_t;
    if (*l_tcp).m_data.is_null() {
      opj_j2k_tcp_destroy(l_tcp);
      return 0i32;
    }
    /* When using the opj_read_tile_header / opj_decode_tile_data API */
    /* such as in test_tile_decoder, m_output_image is NULL, so fall back */
    /* to the full image dimension. This is a bit surprising that */
    /* opj_set_decode_area() is only used to determine intersecting tiles, */
    /* but full tile decoding is done */
    l_image_for_bounds = if !p_j2k.m_output_image.is_null() {
      p_j2k.m_output_image
    } else {
      p_j2k.m_private_image
    };
    if opj_tcd_decode_tile(
      &mut p_j2k.m_tcd,
      (*l_image_for_bounds).x0,
      (*l_image_for_bounds).y0,
      (*l_image_for_bounds).x1,
      (*l_image_for_bounds).y1,
      p_j2k.m_specific_param.m_decoder.m_numcomps_to_decode,
      p_j2k.m_specific_param.m_decoder.m_comps_indices_to_decode,
      (*l_tcp).m_data,
      (*l_tcp).m_data_size,
      p_tile_index,
      p_j2k.cstr_index,
      p_manager,
    ) == 0
    {
      opj_j2k_tcp_destroy(l_tcp);
      p_j2k.m_specific_param.m_decoder.m_state |= J2KState::ERR;
      event_msg!(p_manager, EVT_ERROR, "Failed to decode.\n",);
      return 0i32;
    }
    /* p_data can be set to NULL when the call will take care of using */
    /* itself the TCD data. This is typically the case for whole single */
    /* tile decoding optimization. */
    if let Some(p_data) = p_data {
      if opj_tcd_update_tile_data(&mut p_j2k.m_tcd, p_data) == 0 {
        return 0i32;
      }
      /* To avoid to destroy the tcp which can be useful when we try to decode a tile decoded before (cf j2k_random_tile_access)
       * we destroy just the data which will be re-read in read_tile_header*/
      /*opj_j2k_tcp_destroy(l_tcp);
      p_j2k->m_tcd->tcp = 0;*/
      opj_j2k_tcp_data_destroy(l_tcp);
    }
    p_j2k.m_specific_param.m_decoder.m_can_decode = false;
    p_j2k.m_specific_param.m_decoder.m_state &= !(J2KState::DATA);
    if opj_stream_get_number_byte_left(p_stream) == 0i64
      && p_j2k.m_specific_param.m_decoder.m_state == J2KState::NEOC
    {
      return 1i32;
    }
    if p_j2k.m_specific_param.m_decoder.m_state != J2KState::EOC {
      if opj_stream_read_data(p_stream, l_data.as_mut_ptr(), 2 as OPJ_SIZE_T, p_manager) != 2 {
        event_msg!(p_manager, EVT_ERROR, "Stream too short\n",);
        return 0i32;
      }
      l_current_marker = J2KMarker::from_buffer(l_data.as_mut_ptr());
      if l_current_marker == J2KMarker::EOC {
        p_j2k.m_current_tile_number = 0 as OPJ_UINT32;
        p_j2k.m_specific_param.m_decoder.m_state = J2KState::EOC
      } else if l_current_marker == J2KMarker::UNK(0x8080u32) {
        /* cf. https://code.google.com/p/openjpeg/issues/detail?id=226 */
        if opj_stream_get_number_byte_left(p_stream) == 2i64 {
          p_j2k.m_specific_param.m_decoder.m_state = J2KState::NEOC;
          event_msg!(p_manager, EVT_WARNING, "Expected EOC or SOT marker, got unknown marker 0x8080.  Stream does not end with EOC\n",);
          return 1i32;
        }
      } else if l_current_marker != J2KMarker::SOT {
        if opj_stream_get_number_byte_left(p_stream) == 0i64 {
          p_j2k.m_specific_param.m_decoder.m_state = J2KState::NEOC;
          event_msg!(p_manager, EVT_WARNING, "Stream does not end with EOC\n",);
          return 1i32;
        }
        log::error!(
          "Expected SOT marker: got {:?}, bytes left: {}",
          l_current_marker,
          opj_stream_get_number_byte_left(p_stream)
        );
        event_msg!(p_manager, EVT_ERROR, "Stream too short, expected SOT\n",);
        return 0i32;
      }
    }
    1i32
  }
}
fn opj_j2k_update_image_data(
  mut p_tcd: &mut opj_tcd,
  mut p_output_image: &mut opj_image,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut l_width_src: OPJ_UINT32 = 0;
    let mut l_height_src: OPJ_UINT32 = 0;
    let mut l_width_dest: OPJ_UINT32 = 0;
    let mut l_height_dest: OPJ_UINT32 = 0;
    let mut l_offset_x0_src: OPJ_INT32 = 0;
    let mut l_offset_y0_src: OPJ_INT32 = 0;
    let mut l_offset_x1_src: OPJ_INT32 = 0;
    let mut l_offset_y1_src: OPJ_INT32 = 0;
    let mut l_start_offset_src: OPJ_SIZE_T = 0;
    let mut l_start_x_dest: OPJ_UINT32 = 0;
    let mut l_start_y_dest: OPJ_UINT32 = 0;
    let mut l_x0_dest: OPJ_UINT32 = 0;
    let mut l_y0_dest: OPJ_UINT32 = 0;
    let mut l_x1_dest: OPJ_UINT32 = 0;
    let mut l_y1_dest: OPJ_UINT32 = 0;
    let mut l_start_offset_dest: OPJ_SIZE_T = 0;
    let mut l_img_comp_src = core::ptr::null_mut::<opj_image_comp_t>();
    let mut l_img_comp_dest = core::ptr::null_mut::<opj_image_comp_t>();
    let mut l_tilec = core::ptr::null_mut::<opj_tcd_tilecomp_t>();
    let mut l_image_src = core::ptr::null_mut::<opj_image_t>();
    let mut l_dest_ptr = core::ptr::null_mut::<OPJ_INT32>();
    l_tilec = p_tcd.tcd_image.tiles.comps;
    l_image_src = (*p_tcd).image;
    l_img_comp_src = (*l_image_src).comps;
    l_img_comp_dest = (*p_output_image).comps;
    i = 0 as OPJ_UINT32;
    while i < (*l_image_src).numcomps {
      let mut res_x0: OPJ_INT32 = 0;
      let mut res_x1: OPJ_INT32 = 0;
      let mut res_y0: OPJ_INT32 = 0;
      let mut res_y1: OPJ_INT32 = 0;
      let mut src_data_stride: OPJ_UINT32 = 0;
      let mut p_src_data = core::ptr::null::<OPJ_INT32>();
      /* Copy info from decoded comp image to output image */
      (*l_img_comp_dest).resno_decoded = (*l_img_comp_src).resno_decoded;
      if (*p_tcd).whole_tile_decoding != 0 {
        let mut l_res = (*l_tilec)
          .resolutions
          .offset((*l_img_comp_src).resno_decoded as isize);
        res_x0 = (*l_res).x0;
        res_y0 = (*l_res).y0;
        res_x1 = (*l_res).x1;
        res_y1 = (*l_res).y1;
        src_data_stride = ((*(*l_tilec)
          .resolutions
          .offset((*l_tilec).minimum_num_resolutions.wrapping_sub(1u32) as isize))
        .x1
          - (*(*l_tilec)
            .resolutions
            .offset((*l_tilec).minimum_num_resolutions.wrapping_sub(1u32) as isize))
          .x0) as OPJ_UINT32;
        p_src_data = (*l_tilec).data
      } else {
        let mut l_res_0 = (*l_tilec)
          .resolutions
          .offset((*l_img_comp_src).resno_decoded as isize);
        res_x0 = (*l_res_0).win_x0 as OPJ_INT32;
        res_y0 = (*l_res_0).win_y0 as OPJ_INT32;
        res_x1 = (*l_res_0).win_x1 as OPJ_INT32;
        res_y1 = (*l_res_0).win_y1 as OPJ_INT32;
        src_data_stride = (*l_res_0).win_x1.wrapping_sub((*l_res_0).win_x0);
        p_src_data = (*l_tilec).data_win
      }
      if !p_src_data.is_null() {
        l_width_src = (res_x1 - res_x0) as OPJ_UINT32;
        l_height_src = (res_y1 - res_y0) as OPJ_UINT32;
        /* Current tile component size*/
        /*if (i == 0) {
        fprintf!(stdout, "SRC: l_res_x0=%d, l_res_x1=%d, l_res_y0=%d, l_res_y1=%d\n",
                        res_x0, res_x1, res_y0, res_y1);
        }*/
        /* Border of the current output component*/
        l_x0_dest = opj_uint_ceildivpow2((*l_img_comp_dest).x0, (*l_img_comp_dest).factor); /* can't overflow given that image->x1 is uint32 */
        l_y0_dest = opj_uint_ceildivpow2((*l_img_comp_dest).y0, (*l_img_comp_dest).factor);
        l_x1_dest = l_x0_dest.wrapping_add((*l_img_comp_dest).w);
        l_y1_dest = l_y0_dest.wrapping_add((*l_img_comp_dest).h);
        /*if (i == 0) {
        fprintf!(stdout, "DEST: l_x0_dest=%d, l_x1_dest=%d, l_y0_dest=%d, l_y1_dest=%d (%d)\n",
                        l_x0_dest, l_x1_dest, l_y0_dest, l_y1_dest, l_img_comp_dest->factor );
        }*/
        /*-----*/
        /* Compute the area (l_offset_x0_src, l_offset_y0_src, l_offset_x1_src, l_offset_y1_src)
         * of the input buffer (decoded tile component) which will be move
         * in the output buffer. Compute the area of the output buffer (l_start_x_dest,
         * l_start_y_dest, l_width_dest, l_height_dest)  which will be modified
         * by this input area.
         * */

        assert!(res_x0 >= 0i32);
        assert!(res_x1 >= 0i32);
        if l_x0_dest < res_x0 as OPJ_UINT32 {
          l_start_x_dest = (res_x0 as OPJ_UINT32).wrapping_sub(l_x0_dest);
          l_offset_x0_src = 0i32;
          if l_x1_dest >= res_x1 as OPJ_UINT32 {
            l_width_dest = l_width_src;
            l_offset_x1_src = 0i32
          } else {
            l_width_dest = l_x1_dest.wrapping_sub(res_x0 as OPJ_UINT32);
            l_offset_x1_src = l_width_src.wrapping_sub(l_width_dest) as OPJ_INT32
          }
        } else {
          l_start_x_dest = 0u32;
          l_offset_x0_src = l_x0_dest as OPJ_INT32 - res_x0;
          if l_x1_dest >= res_x1 as OPJ_UINT32 {
            l_width_dest = l_width_src.wrapping_sub(l_offset_x0_src as OPJ_UINT32);
            l_offset_x1_src = 0i32
          } else {
            l_width_dest = (*l_img_comp_dest).w;
            l_offset_x1_src = res_x1 - l_x1_dest as OPJ_INT32
          }
        }
        if l_y0_dest < res_y0 as OPJ_UINT32 {
          l_start_y_dest = (res_y0 as OPJ_UINT32).wrapping_sub(l_y0_dest);
          l_offset_y0_src = 0i32;
          if l_y1_dest >= res_y1 as OPJ_UINT32 {
            l_height_dest = l_height_src;
            l_offset_y1_src = 0i32
          } else {
            l_height_dest = l_y1_dest.wrapping_sub(res_y0 as OPJ_UINT32);
            l_offset_y1_src = l_height_src.wrapping_sub(l_height_dest) as OPJ_INT32
          }
        } else {
          l_start_y_dest = 0u32;
          l_offset_y0_src = l_y0_dest as OPJ_INT32 - res_y0;
          if l_y1_dest >= res_y1 as OPJ_UINT32 {
            l_height_dest = l_height_src.wrapping_sub(l_offset_y0_src as OPJ_UINT32);
            l_offset_y1_src = 0i32
          } else {
            l_height_dest = (*l_img_comp_dest).h;
            l_offset_y1_src = res_y1 - l_y1_dest as OPJ_INT32
          }
        }
        if l_offset_x0_src < 0i32
          || l_offset_y0_src < 0i32
          || l_offset_x1_src < 0i32
          || l_offset_y1_src < 0i32
        {
          return 0i32;
        }
        /* testcase 2977.pdf.asan.67.2198 */
        if (l_width_dest as OPJ_INT32) < 0i32 || (l_height_dest as OPJ_INT32) < 0i32 {
          return 0i32;
        }
        /*-----*/
        /* Compute the input buffer offset */
        l_start_offset_src = (l_offset_x0_src as OPJ_SIZE_T).wrapping_add(
          (l_offset_y0_src as OPJ_SIZE_T).wrapping_mul(src_data_stride as OPJ_SIZE_T),
        );
        /* Compute the output buffer offset */
        l_start_offset_dest = (l_start_x_dest as OPJ_SIZE_T).wrapping_add(
          (l_start_y_dest as OPJ_SIZE_T).wrapping_mul((*l_img_comp_dest).w as OPJ_SIZE_T),
        );
        /* Allocate output component buffer if necessary */
        if (*l_img_comp_dest).data.is_null()
          && l_start_offset_src == 0
          && l_start_offset_dest == 0
          && src_data_stride == (*l_img_comp_dest).w
          && l_width_dest == (*l_img_comp_dest).w
          && l_height_dest == (*l_img_comp_dest).h
        {
          /* If the final image matches the tile buffer, then borrow it */
          /* directly to save a copy */
          if (*p_tcd).whole_tile_decoding != 0 {
            (*l_img_comp_dest).data = (*l_tilec).data;
            (*l_tilec).data = core::ptr::null_mut::<OPJ_INT32>()
          } else {
            (*l_img_comp_dest).data = (*l_tilec).data_win;
            (*l_tilec).data_win = core::ptr::null_mut::<OPJ_INT32>()
          }
        } else {
          if (*l_img_comp_dest).data.is_null() {
            let mut l_width = (*l_img_comp_dest).w as OPJ_SIZE_T;
            let mut l_height = (*l_img_comp_dest).h as OPJ_SIZE_T;
            if l_height == 0
              || l_width > (usize::MAX).wrapping_div(l_height)
              || l_width.wrapping_mul(l_height)
                > (usize::MAX).wrapping_div(core::mem::size_of::<OPJ_INT32>())
            {
              /* would overflow */
              return 0i32;
            }
            (*l_img_comp_dest).data = opj_image_data_alloc(
              l_width
                .wrapping_mul(l_height)
                .wrapping_mul(core::mem::size_of::<OPJ_INT32>()),
            ) as *mut OPJ_INT32;
            if (*l_img_comp_dest).data.is_null() {
              return 0i32;
            }
            if (*l_img_comp_dest).w != l_width_dest || (*l_img_comp_dest).h != l_height_dest {
              memset(
                (*l_img_comp_dest).data as *mut core::ffi::c_void,
                0i32,
                ((*l_img_comp_dest).w as OPJ_SIZE_T)
                  .wrapping_mul((*l_img_comp_dest).h as usize)
                  .wrapping_mul(core::mem::size_of::<OPJ_INT32>()),
              );
            }
          }
          /* Move the output buffer to the first place where we will write*/
          l_dest_ptr = (*l_img_comp_dest).data.add(l_start_offset_dest);
          let mut l_src_ptr = p_src_data;
          l_src_ptr = l_src_ptr.add(l_start_offset_src);
          j = 0 as OPJ_UINT32;
          while j < l_height_dest {
            memcpy(
              l_dest_ptr as *mut core::ffi::c_void,
              l_src_ptr as *const core::ffi::c_void,
              (l_width_dest as usize).wrapping_mul(core::mem::size_of::<OPJ_INT32>()),
            );
            l_dest_ptr = l_dest_ptr.offset((*l_img_comp_dest).w as isize);
            l_src_ptr = l_src_ptr.offset(src_data_stride as isize);
            j += 1;
          }
        }
      }
      /* Happens for partial component decoding */
      i = i.wrapping_add(1);
      l_img_comp_dest = l_img_comp_dest.offset(1);
      l_img_comp_src = l_img_comp_src.offset(1);
      l_tilec = l_tilec.offset(1)
    }
    1i32
  }
}

fn opj_j2k_update_image_dimensions(
  mut p_image: &mut opj_image,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut it_comp: OPJ_UINT32 = 0;
    let mut l_comp_x1: OPJ_INT32 = 0;
    let mut l_comp_y1: OPJ_INT32 = 0;
    let mut l_img_comp = core::ptr::null_mut::<opj_image_comp_t>();
    l_img_comp = p_image.comps;
    it_comp = 0 as OPJ_UINT32;
    while it_comp < p_image.numcomps {
      let mut l_h: OPJ_INT32 = 0;
      let mut l_w: OPJ_INT32 = 0;
      if p_image.x0 > 2147483647 as OPJ_UINT32
        || p_image.y0 > 2147483647 as OPJ_UINT32
        || p_image.x1 > 2147483647 as OPJ_UINT32
        || p_image.y1 > 2147483647 as OPJ_UINT32
      {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Image coordinates above INT_MAX are not supported\n",
        );
        return 0i32;
      }

      (*l_img_comp).x0 = opj_uint_ceildiv(p_image.x0, (*l_img_comp).dx);
      (*l_img_comp).y0 = opj_uint_ceildiv(p_image.y0, (*l_img_comp).dy);
      l_comp_x1 = opj_int_ceildiv(p_image.x1 as OPJ_INT32, (*l_img_comp).dx as OPJ_INT32);
      l_comp_y1 = opj_int_ceildiv(p_image.y1 as OPJ_INT32, (*l_img_comp).dy as OPJ_INT32);

      l_w = opj_int_ceildivpow2(l_comp_x1, (*l_img_comp).factor as OPJ_INT32)
        - opj_int_ceildivpow2(
          (*l_img_comp).x0 as OPJ_INT32,
          (*l_img_comp).factor as OPJ_INT32,
        );
      if l_w < 0i32 {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Size x of the decoded component image is incorrect (comp[%d].w=%d).\n",
          it_comp,
          l_w,
        );
        return 0i32;
      }
      (*l_img_comp).w = l_w as OPJ_UINT32;
      l_h = opj_int_ceildivpow2(l_comp_y1, (*l_img_comp).factor as OPJ_INT32)
        - opj_int_ceildivpow2(
          (*l_img_comp).y0 as OPJ_INT32,
          (*l_img_comp).factor as OPJ_INT32,
        );
      if l_h < 0i32 {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Size y of the decoded component image is incorrect (comp[%d].h=%d).\n",
          it_comp,
          l_h,
        );
        return 0i32;
      }
      (*l_img_comp).h = l_h as OPJ_UINT32;
      l_img_comp = l_img_comp.offset(1);
      it_comp += 1;
    }
    1i32
  }
}

pub(crate) fn opj_j2k_set_decoded_components(
  p_j2k: &mut opj_j2k,
  compenents: &[u32],
  p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let numcomps = compenents.len() as u32;
    let comps_indices = compenents.as_ptr();
    if p_j2k.m_private_image.is_null() {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "opj_read_header() should be called before opj_set_decoded_components().\n",
      );
      return 0i32;
    }
    let mut already_mapped = alloc::collections::BTreeSet::new();
    for comp in compenents {
      if *comp >= (*p_j2k.m_private_image).numcomps {
        event_msg!(p_manager, EVT_ERROR, "Invalid component index: %u\n", *comp,);
        return 0i32;
      }
      if !already_mapped.insert(*comp) {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Component index %u used several times\n",
          *comp,
        );
        return 0i32;
      }
    }
    let mut comps_indices_to_decode = p_j2k.m_specific_param.m_decoder.m_comps_indices_to_decode;
    opj_free(comps_indices_to_decode as *mut core::ffi::c_void);
    comps_indices_to_decode = core::ptr::null_mut::<OPJ_UINT32>();
    if numcomps != 0 {
      comps_indices_to_decode =
        opj_malloc((numcomps as usize).wrapping_mul(core::mem::size_of::<OPJ_UINT32>()))
          as *mut OPJ_UINT32;
      if comps_indices_to_decode.is_null() {
        p_j2k.m_specific_param.m_decoder.m_numcomps_to_decode = 0 as OPJ_UINT32;
        return 0i32;
      }
      memcpy(
        comps_indices_to_decode as *mut core::ffi::c_void,
        comps_indices as *const core::ffi::c_void,
        (numcomps as usize).wrapping_mul(core::mem::size_of::<OPJ_UINT32>()),
      );
    }
    p_j2k.m_specific_param.m_decoder.m_comps_indices_to_decode = comps_indices_to_decode;
    p_j2k.m_specific_param.m_decoder.m_numcomps_to_decode = numcomps;
    1i32
  }
}

pub(crate) fn opj_j2k_set_decode_area(
  mut p_j2k: &mut opj_j2k,
  mut p_image: &mut opj_image,
  mut p_start_x: OPJ_INT32,
  mut p_start_y: OPJ_INT32,
  mut p_end_x: OPJ_INT32,
  mut p_end_y: OPJ_INT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cp: *mut opj_cp_t = &mut p_j2k.m_cp;
    let mut l_image = p_j2k.m_private_image;
    let mut ret: OPJ_BOOL = 0;
    let mut it_comp: OPJ_UINT32 = 0;
    if !(p_j2k.m_cp.tw == 1u32
      && p_j2k.m_cp.th == 1u32
      && !(*p_j2k.m_cp.tcps.offset(0)).m_data.is_null())
    {
      /* Check if we are read the main header */
      if p_j2k.m_specific_param.m_decoder.m_state != J2KState::TPHSOT {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Need to decode the main header before begin to decode the remaining codestream.\n",
        );
        return 0i32;
      }
    }
    /* Update the comps[].factor member of the output image with the one */
    /* of m_reduce */
    it_comp = 0 as OPJ_UINT32;
    while it_comp < p_image.numcomps {
      (*p_image.comps.offset(it_comp as isize)).factor = p_j2k.m_cp.m_specific_param.m_dec.m_reduce;
      it_comp += 1;
    }
    if p_start_x == 0 && p_start_y == 0 && p_end_x == 0 && p_end_y == 0 {
      event_msg!(
        p_manager,
        EVT_INFO,
        "No decoded area parameters, set the decoded area to the whole image\n",
      );
      p_j2k.m_specific_param.m_decoder.m_start_tile_x = 0 as OPJ_UINT32;
      p_j2k.m_specific_param.m_decoder.m_start_tile_y = 0 as OPJ_UINT32;
      p_j2k.m_specific_param.m_decoder.m_end_tile_x = (*l_cp).tw;
      p_j2k.m_specific_param.m_decoder.m_end_tile_y = (*l_cp).th;
      p_image.x0 = (*l_image).x0;
      p_image.y0 = (*l_image).y0;
      p_image.x1 = (*l_image).x1;
      p_image.y1 = (*l_image).y1;
      return opj_j2k_update_image_dimensions(p_image, p_manager);
    }
    /* ----- */
    /* Check if the positions provided by the user are correct */
    /* Left */
    if p_start_x < 0i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Left position of the decoded area (region_x0=%d) should be >= 0.\n",
        p_start_x,
      );
      return 0i32;
    } else if p_start_x as OPJ_UINT32 > (*l_image).x1 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Left position of the decoded area (region_x0=%d) is outside the image area (Xsiz=%d).\n",
        p_start_x,
        (*l_image).x1
      );
      return 0i32;
    } else if (p_start_x as OPJ_UINT32) < (*l_image).x0 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Left position of the decoded area (region_x0=%d) is outside the image area (XOsiz=%d).\n",
        p_start_x,
        (*l_image).x0
      );
      p_j2k.m_specific_param.m_decoder.m_start_tile_x = 0 as OPJ_UINT32;
      p_image.x0 = (*l_image).x0
    } else {
      p_j2k.m_specific_param.m_decoder.m_start_tile_x = (p_start_x as OPJ_UINT32)
        .wrapping_sub((*l_cp).tx0)
        .wrapping_div((*l_cp).tdx);
      p_image.x0 = p_start_x as OPJ_UINT32
    }
    /* Up */
    if p_start_y < 0i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Up position of the decoded area (region_y0=%d) should be >= 0.\n",
        p_start_y,
      );
      return 0i32;
    } else if p_start_y as OPJ_UINT32 > (*l_image).y1 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Up position of the decoded area (region_y0=%d) is outside the image area (Ysiz=%d).\n",
        p_start_y,
        (*l_image).y1,
      );
      return 0i32;
    } else if (p_start_y as OPJ_UINT32) < (*l_image).y0 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Up position of the decoded area (region_y0=%d) is outside the image area (YOsiz=%d).\n",
        p_start_y,
        (*l_image).y0
      );
      p_j2k.m_specific_param.m_decoder.m_start_tile_y = 0 as OPJ_UINT32;
      p_image.y0 = (*l_image).y0
    } else {
      p_j2k.m_specific_param.m_decoder.m_start_tile_y = (p_start_y as OPJ_UINT32)
        .wrapping_sub((*l_cp).ty0)
        .wrapping_div((*l_cp).tdy);
      p_image.y0 = p_start_y as OPJ_UINT32
    }
    /* Right */
    if p_end_x <= 0i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Right position of the decoded area (region_x1=%d) should be > 0.\n",
        p_end_x,
      );
      return 0i32;
    } else if (p_end_x as OPJ_UINT32) < (*l_image).x0 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Right position of the decoded area (region_x1=%d) is outside the image area (XOsiz=%d).\n",
        p_end_x,
        (*l_image).x0
      );
      return 0i32;
    } else if p_end_x as OPJ_UINT32 > (*l_image).x1 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Right position of the decoded area (region_x1=%d) is outside the image area (Xsiz=%d).\n",
        p_end_x,
        (*l_image).x1
      );
      p_j2k.m_specific_param.m_decoder.m_end_tile_x = (*l_cp).tw;
      p_image.x1 = (*l_image).x1
    } else {
      p_j2k.m_specific_param.m_decoder.m_end_tile_x =
        opj_uint_ceildiv(p_end_x as OPJ_UINT32 - (*l_cp).tx0, (*l_cp).tdx);
      p_image.x1 = p_end_x as OPJ_UINT32
    }
    /* Bottom */
    if p_end_y <= 0i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Bottom position of the decoded area (region_y1=%d) should be > 0.\n",
        p_end_y,
      );
      return 0i32;
    } else if (p_end_y as OPJ_UINT32) < (*l_image).y0 {
      event_msg!(
      p_manager,
      EVT_ERROR,
      "Bottom position of the decoded area (region_y1=%d) is outside the image area (YOsiz=%d).\n",
      p_end_y,
      (*l_image).y0
    );
      return 0i32;
    }
    if p_end_y as OPJ_UINT32 > (*l_image).y1 {
      event_msg!(
        p_manager,
        EVT_WARNING,
        "Bottom position of the decoded area (region_y1=%d) is outside the image area (Ysiz=%d).\n",
        p_end_y,
        (*l_image).y1
      );
      p_j2k.m_specific_param.m_decoder.m_end_tile_y = (*l_cp).th;
      p_image.y1 = (*l_image).y1
    } else {
      p_j2k.m_specific_param.m_decoder.m_end_tile_y =
        opj_uint_ceildiv(p_end_y as OPJ_UINT32 - (*l_cp).ty0, (*l_cp).tdy);
      p_image.y1 = p_end_y as OPJ_UINT32
    }
    /* ----- */
    p_j2k.m_specific_param.m_decoder.m_discard_tiles = true;
    ret = opj_j2k_update_image_dimensions(p_image, p_manager);
    if ret != 0 {
      event_msg!(
        p_manager,
        EVT_INFO,
        "Setting decoding area to %d,%d,%d,%d\n",
        p_image.x0,
        p_image.y0,
        p_image.x1,
        p_image.y1,
      );
    }
    ret
  }
}

impl opj_j2k {
  pub fn new(m_is_decoder: i32) -> Self {
    unsafe {
      Self {
        m_is_decoder,
        m_specific_param: core::mem::zeroed(),
        m_private_image: core::ptr::null_mut(),
        m_output_image: core::ptr::null_mut(),
        m_cp: core::mem::zeroed(),
        cstr_index: core::ptr::null_mut(),
        m_current_tile_number: 0,
        m_tcd: opj_tcd::new(m_is_decoder != 0),
        ihdr_w: 0,
        ihdr_h: 0,
        dump_state: 0,
      }
    }
  }
}

pub(crate) fn opj_j2k_create_decompress() -> Option<opj_j2k> {
  let mut l_j2k = opj_j2k::new(1);
  l_j2k.m_cp.m_is_decoder = true;
  /* in the absence of JP2 boxes, consider different bit depth / sign */
  /* per component is allowed */
  unsafe {
    l_j2k.m_cp.allow_different_bit_depth_sign = true;
    /* Default to using strict mode. */
    l_j2k.m_cp.strict = 1i32;
    l_j2k.m_specific_param.m_decoder.m_default_tcp =
      opj_calloc(1i32 as size_t, core::mem::size_of::<opj_tcp_t>()) as *mut opj_tcp_t;
    if l_j2k.m_specific_param.m_decoder.m_default_tcp.is_null() {
      return None;
    }
    l_j2k.m_specific_param.m_decoder.m_header_data =
      opj_calloc(1i32 as size_t, 1000i32 as size_t) as *mut OPJ_BYTE;
    if l_j2k.m_specific_param.m_decoder.m_header_data.is_null() {
      return None;
    }
    l_j2k.m_specific_param.m_decoder.m_header_data_size = 1000 as OPJ_UINT32;
    l_j2k.m_specific_param.m_decoder.m_tile_ind_to_dec = -(1i32);
    l_j2k.m_specific_param.m_decoder.m_last_sot_read_pos = 0 as OPJ_OFF_T;
    /* codestream index creation */
    l_j2k.cstr_index = opj_j2k_create_cstr_index();
    if l_j2k.cstr_index.is_null() {
      return None;
    }
  }
  Some(l_j2k)
}

fn opj_j2k_create_cstr_index() -> *mut opj_codestream_index_t {
  unsafe {
    let mut cstr_index = opj_calloc(
      1i32 as size_t,
      core::mem::size_of::<opj_codestream_index_t>(),
    ) as *mut opj_codestream_index_t;
    if cstr_index.is_null() {
      return core::ptr::null_mut::<opj_codestream_index_t>();
    }
    (*cstr_index).maxmarknum = 100 as OPJ_UINT32;
    (*cstr_index).marknum = 0 as OPJ_UINT32;
    (*cstr_index).marker = opj_calloc(
      (*cstr_index).maxmarknum as size_t,
      core::mem::size_of::<opj_marker_info_t>(),
    ) as *mut opj_marker_info_t;
    if (*cstr_index).marker.is_null() {
      opj_free(cstr_index as *mut core::ffi::c_void);
      return core::ptr::null_mut::<opj_codestream_index_t>();
    }
    (*cstr_index).tile_index = core::ptr::null_mut::<opj_tile_index_t>();
    cstr_index
  }
}
/* *
 * Gets the size taken by writing a SPCod or SPCoc for the given tile and component.
 *
 * @param       p_j2k                   the J2K codec.
 * @param       p_tile_no               the tile index.
 * @param       p_comp_no               the component being outputted.
 *
 * @return      the number of bytes taken by the SPCod element.
 */
fn opj_j2k_get_SPCod_SPCoc_size(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_no: OPJ_UINT32,
  mut p_comp_no: OPJ_UINT32,
) -> OPJ_UINT32 {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    /* preconditions */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    l_tccp = &mut *(*l_tcp).tccps.offset(p_comp_no as isize) as *mut opj_tccp_t;
    /* preconditions again */

    assert!(p_tile_no < (*l_cp).tw.wrapping_mul((*l_cp).th));
    assert!(p_comp_no < (*p_j2k.m_private_image).numcomps);
    if (*l_tccp).csty & 0x1u32 != 0 {
      (5u32).wrapping_add((*l_tccp).numresolutions)
    } else {
      5 as OPJ_UINT32
    }
  }
}
/* *
 * Compare 2 a SPCod/ SPCoc elements, i.e. the coding style of a given component of a tile.
 *
 * @param       p_j2k            J2K codec.
 * @param       p_tile_no        Tile number
 * @param       p_first_comp_no  The 1st component number to compare.
 * @param       p_second_comp_no The 1st component number to compare.
 *
 * @return OPJ_TRUE if SPCdod are equals.
 */
fn opj_j2k_compare_SPCod_SPCoc(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_no: OPJ_UINT32,
  mut p_first_comp_no: OPJ_UINT32,
  mut p_second_comp_no: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp0 = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_tccp1 = core::ptr::null_mut::<opj_tccp_t>();
    /* preconditions */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    l_tccp0 = &mut *(*l_tcp).tccps.offset(p_first_comp_no as isize) as *mut opj_tccp_t;
    l_tccp1 = &mut *(*l_tcp).tccps.offset(p_second_comp_no as isize) as *mut opj_tccp_t;
    if (*l_tccp0).numresolutions != (*l_tccp1).numresolutions {
      return 0i32;
    }
    if (*l_tccp0).cblkw != (*l_tccp1).cblkw {
      return 0i32;
    }
    if (*l_tccp0).cblkh != (*l_tccp1).cblkh {
      return 0i32;
    }
    if (*l_tccp0).cblksty != (*l_tccp1).cblksty {
      return 0i32;
    }
    if (*l_tccp0).qmfbid != (*l_tccp1).qmfbid {
      return 0i32;
    }
    if (*l_tccp0).csty & 0x1u32 != (*l_tccp1).csty & 0x1u32 {
      return 0i32;
    }
    i = 0u32;
    while i < (*l_tccp0).numresolutions {
      if (*l_tccp0).prcw[i as usize] != (*l_tccp1).prcw[i as usize] {
        return 0i32;
      }
      if (*l_tccp0).prch[i as usize] != (*l_tccp1).prch[i as usize] {
        return 0i32;
      }
      i += 1;
    }
    1i32
  }
}
/* *
 * Writes a SPCod or SPCoc element, i.e. the coding style of a given component of a tile.
 *
 * @param       p_j2k           J2K codec.
 * @param       p_tile_no       FIXME DOC
 * @param       p_comp_no       the component number to output.
 * @param       p_data          FIXME DOC
 * @param       p_header_size   FIXME DOC
 * @param       p_manager       the user event manager.
 *
 * @return FIXME DOC
*/
fn opj_j2k_write_SPCod_SPCoc(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_no: OPJ_UINT32,
  mut p_comp_no: OPJ_UINT32,
  mut p_data: *mut OPJ_BYTE,
  mut p_header_size: *mut OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    /* preconditions */

    assert!(!p_header_size.is_null());
    assert!(!p_data.is_null());
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    l_tccp = &mut *(*l_tcp).tccps.offset(p_comp_no as isize) as *mut opj_tccp_t;
    /* preconditions again */
    /* SPcoc (E) */
    assert!(p_tile_no < (*l_cp).tw.wrapping_mul((*l_cp).th));
    assert!(p_comp_no < (*p_j2k.m_private_image).numcomps); /* SPcoc (G) */
    if *p_header_size < 5u32 {
      event_msg!(p_manager, EVT_ERROR, "Error writing SPCod SPCoc element\n",); /* SPcoc (H) */
      return 0i32;
    } /* SPcoc (I_i) */
    opj_write_bytes(
      p_data,
      (*l_tccp).numresolutions.wrapping_sub(1u32),
      1 as OPJ_UINT32,
    );
    p_data = p_data.offset(1);
    opj_write_bytes(p_data, (*l_tccp).cblkw.wrapping_sub(2u32), 1 as OPJ_UINT32);
    p_data = p_data.offset(1);
    opj_write_bytes(p_data, (*l_tccp).cblkh.wrapping_sub(2u32), 1 as OPJ_UINT32);
    p_data = p_data.offset(1);
    opj_write_bytes(p_data, (*l_tccp).cblksty, 1 as OPJ_UINT32);
    p_data = p_data.offset(1);
    opj_write_bytes(p_data, (*l_tccp).qmfbid, 1 as OPJ_UINT32);
    p_data = p_data.offset(1);
    *p_header_size = (*p_header_size).wrapping_sub(5u32);
    if (*l_tccp).csty & 0x1u32 != 0 {
      if *p_header_size < (*l_tccp).numresolutions {
        event_msg!(p_manager, EVT_ERROR, "Error writing SPCod SPCoc element\n",);
        return 0i32;
      }
      i = 0 as OPJ_UINT32;
      while i < (*l_tccp).numresolutions {
        opj_write_bytes(
          p_data,
          (*l_tccp).prcw[i as usize].wrapping_add((*l_tccp).prch[i as usize] << 4i32),
          1 as OPJ_UINT32,
        );
        p_data = p_data.offset(1);
        i += 1;
      }
      *p_header_size = (*p_header_size).wrapping_sub((*l_tccp).numresolutions)
    }
    1i32
  }
}
/* *
 * Reads a SPCod or SPCoc element, i.e. the coding style of a given component of a tile.
 * @param       p_j2k           the jpeg2000 codec.
 * @param       compno          FIXME DOC
 * @param       p_header_data   the data contained in the COM box.
 * @param       p_header_size   the size of the data contained in the COM marker.
 * @param       p_manager       the user event manager.
*/
fn opj_j2k_read_SPCod_SPCoc(
  mut p_j2k: &mut opj_j2k,
  mut compno: OPJ_UINT32,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: *mut OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_tmp: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_current_ptr = core::ptr::null_mut::<OPJ_BYTE>();
    /* preconditions */

    assert!(!p_header_data.is_null());
    l_cp = &mut p_j2k.m_cp;
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    /* precondition again */
    assert!(compno < (*p_j2k.m_private_image).numcomps);
    l_tccp = &mut *(*l_tcp).tccps.offset(compno as isize) as *mut opj_tccp_t;
    l_current_ptr = p_header_data;
    /* make sure room is sufficient */
    if *p_header_size < 5u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading SPCod SPCoc element\n",);
      return 0i32;
    }
    /* SPcod (D) / SPcoc (A) */
    opj_read_bytes(
      l_current_ptr,
      &mut (*l_tccp).numresolutions,
      1 as OPJ_UINT32,
    ); /* tccp->numresolutions = read() + 1 */
    (*l_tccp).numresolutions = (*l_tccp).numresolutions.wrapping_add(1);
    if (*l_tccp).numresolutions > 33u32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Invalid value for numresolutions : %d, max value is set in openjpeg.h at %d\n",
        (*l_tccp).numresolutions,
        33i32,
      );
      return 0i32;
    }
    l_current_ptr = l_current_ptr.offset(1);
    /* If user wants to remove more resolutions than the codestream contains, return error */
    if (*l_cp).m_specific_param.m_dec.m_reduce >= (*l_tccp).numresolutions {
      event_msg!(p_manager, EVT_ERROR,
                      "Error decoding component %d.\nThe number of resolutions to remove (%d) is greater or equal than the number of resolutions of this component (%d)\nModify the cp_reduce parameter.\n\n", compno,
                      (*l_cp).m_specific_param.m_dec.m_reduce,
                      (*l_tccp).numresolutions);
      p_j2k.m_specific_param.m_decoder.m_state |= J2KState::ERR;
      return 0i32;
    }
    /* SPcod (E) / SPcoc (B) */
    opj_read_bytes(l_current_ptr, &mut (*l_tccp).cblkw, 1 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(1);
    (*l_tccp).cblkw = ((*l_tccp).cblkw as core::ffi::c_uint).wrapping_add(2u32) as OPJ_UINT32;
    /* SPcod (F) / SPcoc (C) */
    opj_read_bytes(l_current_ptr, &mut (*l_tccp).cblkh, 1 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(1);
    (*l_tccp).cblkh = ((*l_tccp).cblkh as core::ffi::c_uint).wrapping_add(2u32) as OPJ_UINT32;
    if (*l_tccp).cblkw > 10u32
      || (*l_tccp).cblkh > 10u32
      || (*l_tccp).cblkw.wrapping_add((*l_tccp).cblkh) > 12u32
    {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Error reading SPCod SPCoc element, Invalid cblkw/cblkh combination\n",
      );
      return 0i32;
    }
    /* SPcod (G) / SPcoc (D) */
    opj_read_bytes(l_current_ptr, &mut (*l_tccp).cblksty, 1 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(1);
    if (*l_tccp).cblksty & 0x80u32 != 0u32 {
      /* We do not support HT mixed mode yet.  For conformance, it should be supported.*/
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Error reading SPCod SPCoc element. Unsupported Mixed HT code-block style found\n",
      );
      return 0i32;
    }
    /* SPcod (H) / SPcoc (E) */
    opj_read_bytes(l_current_ptr, &mut (*l_tccp).qmfbid, 1 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(1);
    if (*l_tccp).qmfbid > 1u32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Error reading SPCod SPCoc element, Invalid transformation found\n",
      );
      return 0i32;
    }
    *p_header_size = (*p_header_size).wrapping_sub(5u32);
    /* use custom precinct size ? */
    if (*l_tccp).csty & 0x1u32 != 0 {
      if *p_header_size < (*l_tccp).numresolutions {
        event_msg!(p_manager, EVT_ERROR, "Error reading SPCod SPCoc element\n",);
        return 0i32;
      }
      /* SPcod (I_i) / SPcoc (F_i) */
      i = 0 as OPJ_UINT32;
      while i < (*l_tccp).numresolutions {
        opj_read_bytes(l_current_ptr, &mut l_tmp, 1 as OPJ_UINT32);
        l_current_ptr = l_current_ptr.offset(1);
        /* Precinct exponent 0 is only allowed for lowest resolution level (Table A.21) */
        if i != 0u32 && (l_tmp & 0xfu32 == 0u32 || l_tmp >> 4i32 == 0u32) {
          event_msg!(p_manager, EVT_ERROR, "Invalid precinct size\n",);
          return 0i32;
        }
        (*l_tccp).prcw[i as usize] = l_tmp & 0xfu32;
        (*l_tccp).prch[i as usize] = l_tmp >> 4i32;
        i += 1;
      }
      *p_header_size = (*p_header_size).wrapping_sub((*l_tccp).numresolutions)
    } else {
      /* set default size for the precinct width and height */
      i = 0 as OPJ_UINT32;
      while i < (*l_tccp).numresolutions {
        (*l_tccp).prcw[i as usize] = 15 as OPJ_UINT32;
        (*l_tccp).prch[i as usize] = 15 as OPJ_UINT32;
        i += 1;
      }
    }
    1i32
  }
}
/* *
 * Copies the tile component parameters of all the component from the first tile component.
 *
 * @param               p_j2k           the J2k codec.
 */
fn opj_j2k_copy_tile_component_parameters(mut p_j2k: &mut opj_j2k) {
  unsafe {
    /* loop */
    let mut i: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_ref_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_copied_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_prc_size: OPJ_UINT32 = 0;
    /* preconditions */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    l_ref_tccp = &mut *(*l_tcp).tccps.offset(0) as *mut opj_tccp_t;
    l_copied_tccp = l_ref_tccp.offset(1);
    l_prc_size = (*l_ref_tccp)
      .numresolutions
      .wrapping_mul(core::mem::size_of::<OPJ_UINT32>() as OPJ_UINT32);
    i = 1 as OPJ_UINT32;
    while i < (*p_j2k.m_private_image).numcomps {
      (*l_copied_tccp).numresolutions = (*l_ref_tccp).numresolutions;
      (*l_copied_tccp).cblkw = (*l_ref_tccp).cblkw;
      (*l_copied_tccp).cblkh = (*l_ref_tccp).cblkh;
      (*l_copied_tccp).cblksty = (*l_ref_tccp).cblksty;
      (*l_copied_tccp).qmfbid = (*l_ref_tccp).qmfbid;
      memcpy(
        (*l_copied_tccp).prcw.as_mut_ptr() as *mut core::ffi::c_void,
        (*l_ref_tccp).prcw.as_mut_ptr() as *const core::ffi::c_void,
        l_prc_size as usize,
      );
      memcpy(
        (*l_copied_tccp).prch.as_mut_ptr() as *mut core::ffi::c_void,
        (*l_ref_tccp).prch.as_mut_ptr() as *const core::ffi::c_void,
        l_prc_size as usize,
      );
      l_copied_tccp = l_copied_tccp.offset(1);
      i += 1;
    }
  }
}
/* *
 * Gets the size taken by writing SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param       p_tile_no               the tile index.
 * @param       p_comp_no               the component being outputted.
 * @param       p_j2k                   the J2K codec.
 *
 * @return      the number of bytes taken by the SPCod element.
 */
fn opj_j2k_get_SQcd_SQcc_size(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_no: OPJ_UINT32,
  mut p_comp_no: OPJ_UINT32,
) -> OPJ_UINT32 {
  unsafe {
    let mut l_num_bands: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    /* preconditions */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    l_tccp = &mut *(*l_tcp).tccps.offset(p_comp_no as isize) as *mut opj_tccp_t;
    /* preconditions again */

    assert!(p_tile_no < (*l_cp).tw.wrapping_mul((*l_cp).th));
    assert!(p_comp_no < (*p_j2k.m_private_image).numcomps);
    l_num_bands = if (*l_tccp).qntsty == 1u32 {
      1u32
    } else {
      (*l_tccp)
        .numresolutions
        .wrapping_mul(3u32)
        .wrapping_sub(2u32)
    };
    if (*l_tccp).qntsty == 0u32 {
      (1u32).wrapping_add(l_num_bands)
    } else {
      (1u32).wrapping_add((2u32).wrapping_mul(l_num_bands))
    }
  }
}
/* *
 * Compares 2 SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param       p_j2k                   J2K codec.
 * @param       p_tile_no               the tile to output.
 * @param       p_first_comp_no         the first component number to compare.
 * @param       p_second_comp_no        the second component number to compare.
 *
 * @return OPJ_TRUE if equals.
 */
fn opj_j2k_compare_SQcd_SQcc(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_no: OPJ_UINT32,
  mut p_first_comp_no: OPJ_UINT32,
  mut p_second_comp_no: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp0 = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_tccp1 = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_band_no: OPJ_UINT32 = 0;
    let mut l_num_bands: OPJ_UINT32 = 0;
    /* preconditions */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    l_tccp0 = &mut *(*l_tcp).tccps.offset(p_first_comp_no as isize) as *mut opj_tccp_t;
    l_tccp1 = &mut *(*l_tcp).tccps.offset(p_second_comp_no as isize) as *mut opj_tccp_t;
    if (*l_tccp0).qntsty != (*l_tccp1).qntsty {
      return 0i32;
    }
    if (*l_tccp0).numgbits != (*l_tccp1).numgbits {
      return 0i32;
    }
    if (*l_tccp0).qntsty == 1u32 {
      l_num_bands = 1u32
    } else {
      l_num_bands = (*l_tccp0)
        .numresolutions
        .wrapping_mul(3u32)
        .wrapping_sub(2u32);
      if l_num_bands
        != (*l_tccp1)
          .numresolutions
          .wrapping_mul(3u32)
          .wrapping_sub(2u32)
      {
        return 0i32;
      }
    }
    l_band_no = 0 as OPJ_UINT32;
    while l_band_no < l_num_bands {
      if (*l_tccp0).stepsizes[l_band_no as usize].expn
        != (*l_tccp1).stepsizes[l_band_no as usize].expn
      {
        return 0i32;
      }
      l_band_no += 1;
    }
    if (*l_tccp0).qntsty != 0u32 {
      l_band_no = 0 as OPJ_UINT32;
      while l_band_no < l_num_bands {
        if (*l_tccp0).stepsizes[l_band_no as usize].mant
          != (*l_tccp1).stepsizes[l_band_no as usize].mant
        {
          return 0i32;
        }
        l_band_no += 1;
      }
    }
    1i32
  }
}
/* *
 * Writes a SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param       p_tile_no               the tile to output.
 * @param       p_comp_no               the component number to output.
 * @param       p_data                  the data buffer.
 * @param       p_header_size   pointer to the size of the data buffer, it is changed by the function.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
 *
*/
fn opj_j2k_write_SQcd_SQcc(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_no: OPJ_UINT32,
  mut p_comp_no: OPJ_UINT32,
  mut p_data: *mut OPJ_BYTE,
  mut p_header_size: *mut OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_header_size: OPJ_UINT32 = 0;
    let mut l_band_no: OPJ_UINT32 = 0;
    let mut l_num_bands: OPJ_UINT32 = 0;
    let mut l_expn: OPJ_UINT32 = 0;
    let mut l_mant: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    /* preconditions */

    assert!(!p_header_size.is_null());
    assert!(!p_data.is_null());
    l_cp = &mut p_j2k.m_cp;
    l_tcp = &mut *(*l_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    l_tccp = &mut *(*l_tcp).tccps.offset(p_comp_no as isize) as *mut opj_tccp_t;
    /* preconditions again */
    /* SPqcx_i */
    assert!(p_tile_no < (*l_cp).tw.wrapping_mul((*l_cp).th));
    assert!(p_comp_no < (*p_j2k.m_private_image).numcomps); /* SPqcx_i */
    l_num_bands = if (*l_tccp).qntsty == 1u32 {
      1u32
    } else {
      (*l_tccp)
        .numresolutions
        .wrapping_mul(3u32)
        .wrapping_sub(2u32)
    };
    if (*l_tccp).qntsty == 0u32 {
      l_header_size = (1u32).wrapping_add(l_num_bands);
      if *p_header_size < l_header_size {
        event_msg!(p_manager, EVT_ERROR, "Error writing SQcd SQcc element\n",);
        return 0i32;
      }
      opj_write_bytes(
        p_data,
        (*l_tccp).qntsty.wrapping_add((*l_tccp).numgbits << 5i32),
        1 as OPJ_UINT32,
      );
      p_data = p_data.offset(1);
      l_band_no = 0 as OPJ_UINT32;
      while l_band_no < l_num_bands {
        l_expn = (*l_tccp).stepsizes[l_band_no as usize].expn as OPJ_UINT32;
        opj_write_bytes(p_data, l_expn << 3i32, 1 as OPJ_UINT32);
        p_data = p_data.offset(1);
        l_band_no += 1;
      }
    } else {
      l_header_size = (1u32).wrapping_add((2u32).wrapping_mul(l_num_bands));
      if *p_header_size < l_header_size {
        event_msg!(p_manager, EVT_ERROR, "Error writing SQcd SQcc element\n",);
        return 0i32;
      }
      opj_write_bytes(
        p_data,
        (*l_tccp).qntsty.wrapping_add((*l_tccp).numgbits << 5i32),
        1 as OPJ_UINT32,
      );
      p_data = p_data.offset(1);
      l_band_no = 0 as OPJ_UINT32;
      while l_band_no < l_num_bands {
        l_expn = (*l_tccp).stepsizes[l_band_no as usize].expn as OPJ_UINT32;
        l_mant = (*l_tccp).stepsizes[l_band_no as usize].mant as OPJ_UINT32;
        opj_write_bytes(
          p_data,
          (l_expn << 11i32).wrapping_add(l_mant),
          2 as OPJ_UINT32,
        );
        p_data = p_data.offset(2);
        l_band_no += 1;
      }
    }
    *p_header_size = (*p_header_size).wrapping_sub(l_header_size);
    1i32
  }
}
/* *
 * Reads a SQcd or SQcc element, i.e. the quantization values of a band in the QCD or QCC.
 *
 * @param       p_j2k           J2K codec.
 * @param       compno          the component number to output.
 * @param       p_header_data   the data buffer.
 * @param       p_header_size   pointer to the size of the data buffer, it is changed by the function.
 * @param       p_manager       the user event manager.
 *
*/
fn opj_j2k_read_SQcd_SQcc(
  mut p_j2k: &mut opj_j2k,
  mut p_comp_no: OPJ_UINT32,
  mut p_header_data: *mut OPJ_BYTE,
  mut p_header_size: *mut OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    /* loop*/
    let mut l_band_no: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_current_ptr = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_tmp: OPJ_UINT32 = 0;
    let mut l_num_band: OPJ_UINT32 = 0;
    /* preconditions*/

    assert!(!p_header_data.is_null());
    l_cp = &mut p_j2k.m_cp;
    /* come from tile part header or main header ?*/
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    /* precondition again*/
    assert!(p_comp_no < (*p_j2k.m_private_image).numcomps);
    l_tccp = &mut *(*l_tcp).tccps.offset(p_comp_no as isize) as *mut opj_tccp_t;
    l_current_ptr = p_header_data;
    if *p_header_size < 1u32 {
      event_msg!(p_manager, EVT_ERROR, "Error reading SQcd or SQcc element\n",);
      return 0i32;
    }
    *p_header_size = (*p_header_size as core::ffi::c_uint).wrapping_sub(1u32) as OPJ_UINT32;
    opj_read_bytes(l_current_ptr, &mut l_tmp, 1 as OPJ_UINT32);
    l_current_ptr = l_current_ptr.offset(1);
    (*l_tccp).qntsty = l_tmp & 0x1fu32;
    (*l_tccp).numgbits = l_tmp >> 5i32;
    if (*l_tccp).qntsty == 1u32 {
      l_num_band = 1 as OPJ_UINT32
    } else {
      l_num_band = if (*l_tccp).qntsty == 0u32 {
        *p_header_size
      } else {
        (*p_header_size).wrapping_div(2u32)
      };
      if l_num_band > (3i32 * 33i32 - 2i32) as core::ffi::c_uint {
        event_msg!(p_manager, EVT_WARNING,
                          "While reading CCP_QNTSTY element inside QCD or QCC marker segment, number of subbands (%d) is greater to OPJ_J2K_MAXBANDS (%d). So we limit the number of elements stored to OPJ_J2K_MAXBANDS (%d) and skip the rest. \n", l_num_band,
                          3i32 * 33i32 -
                              2i32,
                          3i32 * 33i32 -
                              2i32);
        /*return OPJ_FALSE;*/
      }
    }
    /* USE_JPWL */

    if (*l_tccp).qntsty == 0u32 {
      l_band_no = 0 as OPJ_UINT32; /* SPqcx_i */
      while l_band_no < l_num_band {
        opj_read_bytes(l_current_ptr, &mut l_tmp, 1 as OPJ_UINT32); /* SPqcx_i */
        l_current_ptr = l_current_ptr.offset(1);
        if l_band_no < (3i32 * 33i32 - 2i32) as core::ffi::c_uint {
          (*l_tccp).stepsizes[l_band_no as usize].expn = (l_tmp >> 3i32) as OPJ_INT32;
          (*l_tccp).stepsizes[l_band_no as usize].mant = 0i32
        }
        l_band_no += 1;
      }
      if *p_header_size < l_num_band {
        return 0i32;
      }
      *p_header_size = (*p_header_size).wrapping_sub(l_num_band)
    } else {
      l_band_no = 0 as OPJ_UINT32;
      while l_band_no < l_num_band {
        opj_read_bytes(l_current_ptr, &mut l_tmp, 2 as OPJ_UINT32);
        l_current_ptr = l_current_ptr.offset(2);
        if l_band_no < (3i32 * 33i32 - 2i32) as core::ffi::c_uint {
          (*l_tccp).stepsizes[l_band_no as usize].expn = (l_tmp >> 11i32) as OPJ_INT32;
          (*l_tccp).stepsizes[l_band_no as usize].mant = (l_tmp & 0x7ffu32) as OPJ_INT32
        }
        l_band_no += 1;
      }
      if *p_header_size < 2 * l_num_band {
        return 0i32;
      }
      *p_header_size = (*p_header_size).wrapping_sub((2u32).wrapping_mul(l_num_band))
    }
    /* Add Antonin : if scalar_derived -> compute other stepsizes */
    if (*l_tccp).qntsty == 1u32 {
      l_band_no = 1 as OPJ_UINT32;
      while l_band_no < (3i32 * 33i32 - 2i32) as core::ffi::c_uint {
        (*l_tccp).stepsizes[l_band_no as usize].expn = if (*l_tccp).stepsizes[0_usize].expn
          - l_band_no.wrapping_sub(1u32).wrapping_div(3u32) as OPJ_INT32
          > 0i32
        {
          ((*l_tccp).stepsizes[0_usize].expn)
            - l_band_no.wrapping_sub(1u32).wrapping_div(3u32) as OPJ_INT32
        } else {
          0i32
        };
        (*l_tccp).stepsizes[l_band_no as usize].mant = (*l_tccp).stepsizes[0_usize].mant;
        l_band_no += 1;
      }
    }
    1i32
  }
}
/* *
 * Copies the tile quantization parameters of all the component from the first tile component.
 *
 * @param               p_j2k           the J2k codec.
 */
fn opj_j2k_copy_tile_quantization_parameters(mut p_j2k: &mut opj_j2k) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_ref_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_copied_tccp = core::ptr::null_mut::<opj_tccp_t>();
    let mut l_size: OPJ_UINT32 = 0;
    /* preconditions */
    l_cp = &mut p_j2k.m_cp;
    l_tcp = if p_j2k.m_specific_param.m_decoder.m_state == J2KState::TPH {
      &mut *(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize) as *mut opj_tcp_t
    } else {
      p_j2k.m_specific_param.m_decoder.m_default_tcp
    };
    l_ref_tccp = &mut *(*l_tcp).tccps.offset(0) as *mut opj_tccp_t;
    l_copied_tccp = l_ref_tccp.offset(1);
    l_size = ((3i32 * 33i32 - 2i32) as usize).wrapping_mul(core::mem::size_of::<opj_stepsize_t>())
      as OPJ_UINT32;
    i = 1 as OPJ_UINT32;
    while i < (*p_j2k.m_private_image).numcomps {
      (*l_copied_tccp).qntsty = (*l_ref_tccp).qntsty;
      (*l_copied_tccp).numgbits = (*l_ref_tccp).numgbits;
      memcpy(
        (*l_copied_tccp).stepsizes.as_mut_ptr() as *mut core::ffi::c_void,
        (*l_ref_tccp).stepsizes.as_mut_ptr() as *const core::ffi::c_void,
        l_size as usize,
      );
      l_copied_tccp = l_copied_tccp.offset(1);
      i += 1;
    }
  }
}

#[cfg(feature = "file-io")]
fn opj_j2k_dump_tile_info(
  mut l_default_tile: *mut opj_tcp_t,
  mut numcomps: OPJ_INT32,
  mut out_stream: *mut FILE,
) {
  unsafe {
    if !l_default_tile.is_null() {
      let mut compno: OPJ_INT32 = 0;
      fprintf!(out_stream, "\t default tile {\n",);
      if (*l_default_tile).csty != 0 {
        fprintf!(out_stream, "\t\t csty=%#x\n", (*l_default_tile).csty,);
      } else {
        fprintf!(out_stream, "\t\t csty=0\n",);
      }
      if (*l_default_tile).prg != 0 {
        fprintf!(
          out_stream,
          "\t\t prg=%#x\n",
          (*l_default_tile).prg as core::ffi::c_int,
        );
      } else {
        fprintf!(out_stream, "\t\t prg=0\n",);
      }
      fprintf!(
        out_stream,
        "\t\t numlayers=%d\n",
        (*l_default_tile).numlayers,
      );
      fprintf!(out_stream, "\t\t mct=%x\n", (*l_default_tile).mct,);
      /*end of default tile*/
      compno = 0i32; /*end of component of default tile*/
      while compno < numcomps {
        let mut l_tccp: *mut opj_tccp_t =
          &mut *(*l_default_tile).tccps.offset(compno as isize) as *mut opj_tccp_t;
        let mut resno: OPJ_UINT32 = 0;
        let mut bandno: OPJ_INT32 = 0;
        let mut numbands: OPJ_INT32 = 0;
        /* coding style*/
        fprintf!(out_stream, "\t\t comp %d {\n", compno,);
        if (*l_tccp).csty != 0 {
          fprintf!(out_stream, "\t\t\t csty=%#x\n", (*l_tccp).csty,);
        } else {
          fprintf!(out_stream, "\t\t\t csty=0\n",);
        }
        fprintf!(
          out_stream,
          "\t\t\t numresolutions=%d\n",
          (*l_tccp).numresolutions,
        );
        fprintf!(out_stream, "\t\t\t cblkw=2^%d\n", (*l_tccp).cblkw,);
        fprintf!(out_stream, "\t\t\t cblkh=2^%d\n", (*l_tccp).cblkh,);
        if (*l_tccp).cblksty != 0 {
          fprintf!(out_stream, "\t\t\t cblksty=%#x\n", (*l_tccp).cblksty,);
        } else {
          fprintf!(out_stream, "\t\t\t cblksty=0\n",);
        }
        fprintf!(out_stream, "\t\t\t qmfbid=%d\n", (*l_tccp).qmfbid,);
        fprintf!(out_stream, "\t\t\t preccintsize (w,h)=",);
        resno = 0 as OPJ_UINT32;
        while resno < (*l_tccp).numresolutions {
          fprintf!(
            out_stream,
            "(%d,%d) ",
            (*l_tccp).prcw[resno as usize],
            (*l_tccp).prch[resno as usize],
          );
          resno += 1;
        }
        fprintf!(out_stream, "\n",);
        /* quantization style*/
        fprintf!(out_stream, "\t\t\t qntsty=%d\n", (*l_tccp).qntsty,);
        fprintf!(out_stream, "\t\t\t numgbits=%d\n", (*l_tccp).numgbits,);
        fprintf!(out_stream, "\t\t\t stepsizes (m,e)=",);
        numbands = if (*l_tccp).qntsty == 1u32 {
          1i32
        } else {
          ((*l_tccp).numresolutions as OPJ_INT32 * 3i32) - 2i32
        };
        bandno = 0i32;
        while bandno < numbands {
          fprintf!(
            out_stream,
            "(%d,%d) ",
            (*l_tccp).stepsizes[bandno as usize].mant,
            (*l_tccp).stepsizes[bandno as usize].expn,
          );
          bandno += 1
        }
        fprintf!(out_stream, "\n",);
        /* RGN value*/
        fprintf!(out_stream, "\t\t\t roishift=%d\n", (*l_tccp).roishift,);
        fprintf!(out_stream, "\t\t }\n",);
        compno += 1
      }
      fprintf!(out_stream, "\t }\n",);
    };
  }
}

#[cfg(feature = "file-io")]
pub(crate) fn j2k_dump(mut p_j2k: &mut opj_j2k, mut flag: OPJ_INT32, mut out_stream: *mut FILE) {
  unsafe {
    /* Check if the flag is compatible with j2k file*/
    if flag & 128i32 != 0 || flag & 256i32 != 0 {
      fprintf!(out_stream, "Wrong flag\n",);
      return;
    }
    /* Dump the image_header */
    if flag & 1i32 != 0 && !p_j2k.m_private_image.is_null() {
      j2k_dump_image_header(&mut *p_j2k.m_private_image, 0i32, out_stream);
    }
    /* Dump the codestream info from main header */
    if flag & 2i32 != 0 && !p_j2k.m_private_image.is_null() {
      opj_j2k_dump_MH_info(p_j2k, out_stream);
    }
    /* Dump all tile/codestream info */
    if flag & 8i32 != 0 {
      let mut l_nb_tiles = p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw);
      let mut i: OPJ_UINT32 = 0;
      let mut l_tcp = p_j2k.m_cp.tcps;
      if !p_j2k.m_private_image.is_null() {
        i = 0 as OPJ_UINT32;
        while i < l_nb_tiles {
          opj_j2k_dump_tile_info(
            l_tcp,
            (*p_j2k.m_private_image).numcomps as OPJ_INT32,
            out_stream,
          );
          l_tcp = l_tcp.offset(1);
          i += 1;
        }
      }
    }
    /* Dump the codestream info of the current tile */
    if flag & 4i32 != 0 {};
    /* Dump the codestream index from main header */
    if flag & 16i32 != 0 {
      opj_j2k_dump_MH_index(p_j2k, out_stream);
    }
    /* Dump the codestream index of the current tile */
    if flag & 32i32 != 0 {}
  }
}

#[cfg(feature = "file-io")]
fn opj_j2k_dump_MH_index(mut p_j2k: &mut opj_j2k, mut out_stream: *mut FILE) {
  unsafe {
    let mut cstr_index = p_j2k.cstr_index;
    let mut it_marker: OPJ_UINT32 = 0;
    let mut it_tile: OPJ_UINT32 = 0;
    let mut it_tile_part: OPJ_UINT32 = 0;
    fprintf!(out_stream, "Codestream index from main header: {\n",);
    fprintf!(
      out_stream,
      "\t Main header start position=%li\n\t Main header end position=%li\n",
      (*cstr_index).main_head_start,
      (*cstr_index).main_head_end,
    );
    fprintf!(out_stream, "\t Marker list: {\n",);
    if !(*cstr_index).marker.is_null() {
      it_marker = 0 as OPJ_UINT32;
      while it_marker < (*cstr_index).marknum {
        let marker = *(*cstr_index).marker.offset(it_marker as isize);
        let ty = marker.type_ as i32;
        if ty != 0 {
          fprintf!(
            out_stream,
            "\t\t type=%#x, pos=%li, len=%d\n",
            ty,
            marker.pos,
            marker.len,
          );
        } else {
          fprintf!(
            out_stream,
            "\t\t type=%x, pos=%li, len=%d\n",
            ty,
            marker.pos,
            marker.len,
          );
        }
        it_marker += 1;
      }
    }
    fprintf!(out_stream, "\t }\n",);
    if !(*cstr_index).tile_index.is_null() {
      /* Simple test to avoid to write empty information*/
      let mut l_acc_nb_of_tile_part = 0 as OPJ_UINT32; /* Not fill from the main header*/
      it_tile = 0 as OPJ_UINT32;
      while it_tile < (*cstr_index).nb_of_tiles {
        l_acc_nb_of_tile_part = (l_acc_nb_of_tile_part as core::ffi::c_uint)
          .wrapping_add((*(*cstr_index).tile_index.offset(it_tile as isize)).nb_tps)
          as OPJ_UINT32;
        it_tile += 1;
      }
      if l_acc_nb_of_tile_part != 0 {
        fprintf!(out_stream, "\t Tile index: {\n",);
        it_tile = 0 as OPJ_UINT32;
        while it_tile < (*cstr_index).nb_of_tiles {
          let mut nb_of_tile_part = (*(*cstr_index).tile_index.offset(it_tile as isize)).nb_tps;
          fprintf!(
            out_stream,
            "\t\t nb of tile-part in tile [%d]=%d\n",
            it_tile,
            nb_of_tile_part,
          );
          if !(*(*cstr_index).tile_index.offset(it_tile as isize))
            .tp_index
            .is_null()
          {
            it_tile_part = 0 as OPJ_UINT32;
            while it_tile_part < nb_of_tile_part {
              fprintf!(
                out_stream,
                "\t\t\t tile-part[%d]: star_pos=%li, end_header=%li, end_pos=%li.\n",
                it_tile_part,
                (*(*(*cstr_index).tile_index.offset(it_tile as isize))
                  .tp_index
                  .offset(it_tile_part as isize))
                .start_pos,
                (*(*(*cstr_index).tile_index.offset(it_tile as isize))
                  .tp_index
                  .offset(it_tile_part as isize))
                .end_header,
                (*(*(*cstr_index).tile_index.offset(it_tile as isize))
                  .tp_index
                  .offset(it_tile_part as isize))
                .end_pos,
              );
              it_tile_part += 1;
            }
          }
          if !(*(*cstr_index).tile_index.offset(it_tile as isize))
            .marker
            .is_null()
          {
            it_marker = 0 as OPJ_UINT32;
            while it_marker < (*(*cstr_index).tile_index.offset(it_tile as isize)).marknum {
              fprintf!(
                out_stream,
                "\t\t type=%#x, pos=%li, len=%d\n",
                (*(*(*cstr_index).tile_index.offset(it_tile as isize))
                  .marker
                  .offset(it_marker as isize))
                .type_ as core::ffi::c_int,
                (*(*(*cstr_index).tile_index.offset(it_tile as isize))
                  .marker
                  .offset(it_marker as isize))
                .pos,
                (*(*(*cstr_index).tile_index.offset(it_tile as isize))
                  .marker
                  .offset(it_marker as isize))
                .len,
              );
              it_marker += 1;
            }
          }
          it_tile += 1;
        }
        fprintf!(out_stream, "\t }\n",);
      }
    }
    fprintf!(out_stream, "}\n",);
  }
}

#[cfg(feature = "file-io")]
fn opj_j2k_dump_MH_info(mut p_j2k: &mut opj_j2k, mut out_stream: *mut FILE) {
  unsafe {
    fprintf!(out_stream, "Codestream info from main header: {\n",);
    fprintf!(
      out_stream,
      "\t tx0=%u, ty0=%u\n",
      p_j2k.m_cp.tx0,
      p_j2k.m_cp.ty0,
    );
    fprintf!(
      out_stream,
      "\t tdx=%u, tdy=%u\n",
      p_j2k.m_cp.tdx,
      p_j2k.m_cp.tdy,
    );
    fprintf!(
      out_stream,
      "\t tw=%u, th=%u\n",
      p_j2k.m_cp.tw,
      p_j2k.m_cp.th,
    );
    opj_j2k_dump_tile_info(
      p_j2k.m_specific_param.m_decoder.m_default_tcp,
      (*p_j2k.m_private_image).numcomps as OPJ_INT32,
      out_stream,
    );
    fprintf!(out_stream, "}\n",);
  }
}

#[cfg(feature = "file-io")]

pub(crate) fn j2k_dump_image_header(
  mut img_header: &mut opj_image,
  mut dev_dump_flag: OPJ_BOOL,
  mut out_stream: *mut FILE,
) {
  unsafe {
    let mut tab = "";
    if dev_dump_flag != 0 {
      fprintf!(out_stream, "[DEV] Dump an image_header struct {\n",);
    } else {
      fprintf!(out_stream, "Image info {\n",);
      tab = "\t";
    }
    fprintf!(
      out_stream,
      "%s x0=%d, y0=%d\n",
      tab,
      (*img_header).x0,
      (*img_header).y0,
    );
    fprintf!(
      out_stream,
      "%s x1=%d, y1=%d\n",
      tab,
      (*img_header).x1,
      (*img_header).y1,
    );
    fprintf!(out_stream, "%s numcomps=%d\n", tab, (*img_header).numcomps,);
    if !(*img_header).comps.is_null() {
      let mut compno: OPJ_UINT32 = 0;
      compno = 0 as OPJ_UINT32;
      while compno < (*img_header).numcomps {
        fprintf!(out_stream, "%s\t component %d {\n", tab, compno,);
        j2k_dump_image_comp_header(
          &mut *(*img_header).comps.offset(compno as isize),
          dev_dump_flag,
          out_stream,
        );
        fprintf!(out_stream, "%s}\n", tab,);
        compno += 1;
      }
    }
    fprintf!(out_stream, "}\n",);
  }
}

#[cfg(feature = "file-io")]

pub(crate) fn j2k_dump_image_comp_header(
  mut comp_header: *mut opj_image_comp_t,
  mut dev_dump_flag: OPJ_BOOL,
  mut out_stream: *mut FILE,
) {
  unsafe {
    let mut tab = "";
    if dev_dump_flag != 0 {
      fprintf!(out_stream, "[DEV] Dump an image_comp_header struct {\n",);
    } else {
      tab = "\t\t";
    }
    fprintf!(
      out_stream,
      "%s dx=%d, dy=%d\n",
      tab,
      (*comp_header).dx,
      (*comp_header).dy,
    );
    fprintf!(out_stream, "%s prec=%d\n", tab, (*comp_header).prec,);
    fprintf!(out_stream, "%s sgnd=%d\n", tab, (*comp_header).sgnd,);
    if dev_dump_flag != 0 {
      fprintf!(out_stream, "}\n",);
    };
  }
}

pub(crate) fn j2k_get_cstr_info(mut p_j2k: &mut opj_j2k) -> *mut opj_codestream_info_v2_t {
  unsafe {
    let mut compno: OPJ_UINT32 = 0;
    let mut numcomps = (*p_j2k.m_private_image).numcomps;
    let mut l_default_tile = core::ptr::null_mut::<opj_tcp_t>();
    let mut cstr_info = opj_calloc(
      1i32 as size_t,
      core::mem::size_of::<opj_codestream_info_v2_t>(),
    ) as *mut opj_codestream_info_v2_t;
    if cstr_info.is_null() {
      return core::ptr::null_mut::<opj_codestream_info_v2_t>();
    }
    (*cstr_info).nbcomps = (*p_j2k.m_private_image).numcomps;
    (*cstr_info).tx0 = p_j2k.m_cp.tx0;
    (*cstr_info).ty0 = p_j2k.m_cp.ty0;
    (*cstr_info).tdx = p_j2k.m_cp.tdx;
    (*cstr_info).tdy = p_j2k.m_cp.tdy;
    (*cstr_info).tw = p_j2k.m_cp.tw;
    (*cstr_info).th = p_j2k.m_cp.th;
    (*cstr_info).tile_info = core::ptr::null_mut::<opj_tile_info_v2_t>();
    l_default_tile = p_j2k.m_specific_param.m_decoder.m_default_tcp;
    (*cstr_info).m_default_tile_info.csty = (*l_default_tile).csty;
    (*cstr_info).m_default_tile_info.prg = (*l_default_tile).prg;
    (*cstr_info).m_default_tile_info.numlayers = (*l_default_tile).numlayers;
    (*cstr_info).m_default_tile_info.mct = (*l_default_tile).mct;
    (*cstr_info).m_default_tile_info.tccp_info = opj_calloc(
      (*cstr_info).nbcomps as size_t,
      core::mem::size_of::<opj_tccp_info_t>(),
    ) as *mut opj_tccp_info_t;
    if (*cstr_info).m_default_tile_info.tccp_info.is_null() {
      opj_destroy_cstr_info(&mut cstr_info);
      return core::ptr::null_mut::<opj_codestream_info_v2_t>();
    }
    compno = 0 as OPJ_UINT32;
    while compno < numcomps {
      let mut l_tccp: *mut opj_tccp_t =
        &mut *(*l_default_tile).tccps.offset(compno as isize) as *mut opj_tccp_t;
      let mut l_tccp_info: *mut opj_tccp_info_t = &mut *(*cstr_info)
        .m_default_tile_info
        .tccp_info
        .offset(compno as isize)
        as *mut opj_tccp_info_t;
      let mut bandno: OPJ_INT32 = 0;
      let mut numbands: OPJ_INT32 = 0;
      /* coding style*/
      (*l_tccp_info).csty = (*l_tccp).csty;
      (*l_tccp_info).numresolutions = (*l_tccp).numresolutions;
      (*l_tccp_info).cblkw = (*l_tccp).cblkw;
      (*l_tccp_info).cblkh = (*l_tccp).cblkh;
      (*l_tccp_info).cblksty = (*l_tccp).cblksty;
      (*l_tccp_info).qmfbid = (*l_tccp).qmfbid;
      if (*l_tccp).numresolutions < 33u32 {
        memcpy(
          (*l_tccp_info).prch.as_mut_ptr() as *mut core::ffi::c_void,
          (*l_tccp).prch.as_mut_ptr() as *const core::ffi::c_void,
          (*l_tccp).numresolutions as usize,
        );
        memcpy(
          (*l_tccp_info).prcw.as_mut_ptr() as *mut core::ffi::c_void,
          (*l_tccp).prcw.as_mut_ptr() as *const core::ffi::c_void,
          (*l_tccp).numresolutions as usize,
        );
      }
      /* quantization style*/
      (*l_tccp_info).qntsty = (*l_tccp).qntsty;
      (*l_tccp_info).numgbits = (*l_tccp).numgbits;
      numbands = if (*l_tccp).qntsty == 1u32 {
        1i32
      } else {
        ((*l_tccp).numresolutions as OPJ_INT32 * 3i32) - 2i32
      };
      if numbands < 3i32 * 33i32 - 2i32 {
        bandno = 0i32;
        while bandno < numbands {
          (*l_tccp_info).stepsizes_mant[bandno as usize] =
            (*l_tccp).stepsizes[bandno as usize].mant as OPJ_UINT32;
          (*l_tccp_info).stepsizes_expn[bandno as usize] =
            (*l_tccp).stepsizes[bandno as usize].expn as OPJ_UINT32;
          bandno += 1
        }
      }
      /* RGN value*/
      (*l_tccp_info).roishift = (*l_tccp).roishift;
      compno += 1;
    }
    cstr_info
  }
}

pub(crate) fn j2k_get_cstr_index(mut p_j2k: &mut opj_j2k) -> *mut opj_codestream_index_t {
  unsafe {
    let mut l_cstr_index = opj_calloc(
      1i32 as size_t,
      core::mem::size_of::<opj_codestream_index_t>(),
    ) as *mut opj_codestream_index_t;
    if l_cstr_index.is_null() {
      return core::ptr::null_mut::<opj_codestream_index_t>();
    }
    (*l_cstr_index).main_head_start = (*p_j2k.cstr_index).main_head_start;
    (*l_cstr_index).main_head_end = (*p_j2k.cstr_index).main_head_end;
    (*l_cstr_index).codestream_size = (*p_j2k.cstr_index).codestream_size;
    (*l_cstr_index).marknum = (*p_j2k.cstr_index).marknum;
    (*l_cstr_index).marker = opj_malloc(
      ((*l_cstr_index).marknum as usize).wrapping_mul(core::mem::size_of::<opj_marker_info_t>()),
    ) as *mut opj_marker_info_t;
    if (*l_cstr_index).marker.is_null() {
      opj_free(l_cstr_index as *mut core::ffi::c_void);
      return core::ptr::null_mut::<opj_codestream_index_t>();
    }
    if !(*p_j2k.cstr_index).marker.is_null() {
      memcpy(
        (*l_cstr_index).marker as *mut core::ffi::c_void,
        (*p_j2k.cstr_index).marker as *const core::ffi::c_void,
        ((*l_cstr_index).marknum as usize).wrapping_mul(core::mem::size_of::<opj_marker_info_t>()),
      );
    } else {
      opj_free((*l_cstr_index).marker as *mut core::ffi::c_void);
      (*l_cstr_index).marker = core::ptr::null_mut::<opj_marker_info_t>()
    }
    (*l_cstr_index).nb_of_tiles = (*p_j2k.cstr_index).nb_of_tiles;
    (*l_cstr_index).tile_index = opj_calloc(
      (*l_cstr_index).nb_of_tiles as size_t,
      core::mem::size_of::<opj_tile_index_t>(),
    ) as *mut opj_tile_index_t;
    if (*l_cstr_index).tile_index.is_null() {
      opj_free((*l_cstr_index).marker as *mut core::ffi::c_void);
      opj_free(l_cstr_index as *mut core::ffi::c_void);
      return core::ptr::null_mut::<opj_codestream_index_t>();
    }
    if (*p_j2k.cstr_index).tile_index.is_null() {
      opj_free((*l_cstr_index).tile_index as *mut core::ffi::c_void);
      (*l_cstr_index).tile_index = core::ptr::null_mut::<opj_tile_index_t>()
    } else {
      let mut it_tile = 0 as OPJ_UINT32;
      it_tile = 0 as OPJ_UINT32;
      while it_tile < (*l_cstr_index).nb_of_tiles {
        /* Tile Marker*/
        (*(*l_cstr_index).tile_index.offset(it_tile as isize)).marknum =
          (*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize)).marknum;
        let fresh34 = &mut (*(*l_cstr_index).tile_index.offset(it_tile as isize)).marker;
        *fresh34 = opj_malloc(
          ((*(*l_cstr_index).tile_index.offset(it_tile as isize)).marknum as usize)
            .wrapping_mul(core::mem::size_of::<opj_marker_info_t>()),
        ) as *mut opj_marker_info_t;
        if (*(*l_cstr_index).tile_index.offset(it_tile as isize))
          .marker
          .is_null()
        {
          let mut it_tile_free: OPJ_UINT32 = 0;
          it_tile_free = 0 as OPJ_UINT32;
          while it_tile_free < it_tile {
            opj_free(
              (*(*l_cstr_index).tile_index.offset(it_tile_free as isize)).marker
                as *mut core::ffi::c_void,
            );
            it_tile_free += 1;
          }
          opj_free((*l_cstr_index).tile_index as *mut core::ffi::c_void);
          opj_free((*l_cstr_index).marker as *mut core::ffi::c_void);
          opj_free(l_cstr_index as *mut core::ffi::c_void);
          return core::ptr::null_mut::<opj_codestream_index_t>();
        }
        if !(*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize))
          .marker
          .is_null()
        {
          memcpy(
            (*(*l_cstr_index).tile_index.offset(it_tile as isize)).marker as *mut core::ffi::c_void,
            (*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize)).marker
              as *const core::ffi::c_void,
            ((*(*l_cstr_index).tile_index.offset(it_tile as isize)).marknum as usize)
              .wrapping_mul(core::mem::size_of::<opj_marker_info_t>()),
          );
        } else {
          opj_free(
            (*(*l_cstr_index).tile_index.offset(it_tile as isize)).marker as *mut core::ffi::c_void,
          );
          let fresh35 = &mut (*(*l_cstr_index).tile_index.offset(it_tile as isize)).marker;
          *fresh35 = core::ptr::null_mut::<opj_marker_info_t>()
        }
        /* Tile part index*/
        (*(*l_cstr_index).tile_index.offset(it_tile as isize)).nb_tps =
          (*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize)).nb_tps;
        let fresh36 = &mut (*(*l_cstr_index).tile_index.offset(it_tile as isize)).tp_index;
        *fresh36 = opj_malloc(
          ((*(*l_cstr_index).tile_index.offset(it_tile as isize)).nb_tps as usize)
            .wrapping_mul(core::mem::size_of::<opj_tp_index_t>()),
        ) as *mut opj_tp_index_t;
        if (*(*l_cstr_index).tile_index.offset(it_tile as isize))
          .tp_index
          .is_null()
        {
          let mut it_tile_free_0: OPJ_UINT32 = 0;
          it_tile_free_0 = 0 as OPJ_UINT32;
          while it_tile_free_0 < it_tile {
            opj_free(
              (*(*l_cstr_index).tile_index.offset(it_tile_free_0 as isize)).marker
                as *mut core::ffi::c_void,
            );
            opj_free(
              (*(*l_cstr_index).tile_index.offset(it_tile_free_0 as isize)).tp_index
                as *mut core::ffi::c_void,
            );
            it_tile_free_0 += 1;
          }
          opj_free((*l_cstr_index).tile_index as *mut core::ffi::c_void);
          opj_free((*l_cstr_index).marker as *mut core::ffi::c_void);
          opj_free(l_cstr_index as *mut core::ffi::c_void);
          return core::ptr::null_mut::<opj_codestream_index_t>();
        }
        if !(*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize))
          .tp_index
          .is_null()
        {
          memcpy(
            (*(*l_cstr_index).tile_index.offset(it_tile as isize)).tp_index
              as *mut core::ffi::c_void,
            (*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize)).tp_index
              as *const core::ffi::c_void,
            ((*(*l_cstr_index).tile_index.offset(it_tile as isize)).nb_tps as usize)
              .wrapping_mul(core::mem::size_of::<opj_tp_index_t>()),
          );
        } else {
          opj_free(
            (*(*l_cstr_index).tile_index.offset(it_tile as isize)).tp_index
              as *mut core::ffi::c_void,
          );
          let fresh37 = &mut (*(*l_cstr_index).tile_index.offset(it_tile as isize)).tp_index;
          *fresh37 = core::ptr::null_mut::<opj_tp_index_t>()
        }
        /* Packet index (NOT USED)*/
        (*(*l_cstr_index).tile_index.offset(it_tile as isize)).nb_packet = 0 as OPJ_UINT32;
        let fresh38 = &mut (*(*l_cstr_index).tile_index.offset(it_tile as isize)).packet_index;
        *fresh38 = core::ptr::null_mut::<opj_packet_info_t>();
        it_tile += 1;
      }
    }
    l_cstr_index
  }
}

fn opj_j2k_allocate_tile_element_cstr_index(mut p_j2k: &mut opj_j2k) -> OPJ_BOOL {
  unsafe {
    let mut it_tile = 0 as OPJ_UINT32;
    (*p_j2k.cstr_index).nb_of_tiles = p_j2k.m_cp.tw.wrapping_mul(p_j2k.m_cp.th);
    (*p_j2k.cstr_index).tile_index = opj_calloc(
      (*p_j2k.cstr_index).nb_of_tiles as size_t,
      core::mem::size_of::<opj_tile_index_t>(),
    ) as *mut opj_tile_index_t;
    if (*p_j2k.cstr_index).tile_index.is_null() {
      return 0i32;
    }
    it_tile = 0 as OPJ_UINT32;
    while it_tile < (*p_j2k.cstr_index).nb_of_tiles {
      (*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize)).maxmarknum = 100 as OPJ_UINT32;
      (*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize)).marknum = 0 as OPJ_UINT32;
      let fresh39 = &mut (*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize)).marker;
      *fresh39 = opj_calloc(
        (*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize)).maxmarknum as size_t,
        core::mem::size_of::<opj_marker_info_t>(),
      ) as *mut opj_marker_info_t;
      if (*(*p_j2k.cstr_index).tile_index.offset(it_tile as isize))
        .marker
        .is_null()
      {
        return 0i32;
      }
      it_tile += 1;
    }
    1i32
  }
}
fn opj_j2k_are_all_used_components_decoded(
  mut p_j2k: &mut opj_j2k,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut compno: OPJ_UINT32 = 0;
    let mut decoded_all_used_components = 1i32;
    if p_j2k.m_specific_param.m_decoder.m_numcomps_to_decode != 0 {
      compno = 0 as OPJ_UINT32;
      while compno < p_j2k.m_specific_param.m_decoder.m_numcomps_to_decode {
        let mut dec_compno = *p_j2k
          .m_specific_param
          .m_decoder
          .m_comps_indices_to_decode
          .offset(compno as isize);
        if (*(*p_j2k.m_output_image).comps.offset(dec_compno as isize))
          .data
          .is_null()
        {
          event_msg!(
            p_manager,
            EVT_WARNING,
            "Failed to decode component %d\n",
            dec_compno,
          );
          decoded_all_used_components = 0i32
        }
        compno += 1;
      }
    } else {
      compno = 0 as OPJ_UINT32;
      while compno < (*p_j2k.m_output_image).numcomps {
        if (*(*p_j2k.m_output_image).comps.offset(compno as isize))
          .data
          .is_null()
        {
          event_msg!(
            p_manager,
            EVT_WARNING,
            "Failed to decode component %d\n",
            compno,
          );
          decoded_all_used_components = 0i32
        }
        compno += 1;
      }
    }
    if decoded_all_used_components == 0i32 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Failed to decode all used components\n",
      );
      return 0i32;
    }
    1i32
  }
}
/* *
 * Reads the tiles.
 */
fn opj_j2k_decode_tiles(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut tile_info = TileInfo::default();
    let mut nr_tiles = 0 as OPJ_UINT32;
    /* Particular case for whole single tile decoding */
    /* We can avoid allocating intermediate tile buffers */
    if p_j2k.m_cp.tw == 1u32
      && p_j2k.m_cp.th == 1u32
      && p_j2k.m_cp.tx0 == 0u32
      && p_j2k.m_cp.ty0 == 0u32
      && (*p_j2k.m_output_image).x0 == 0u32
      && (*p_j2k.m_output_image).y0 == 0u32
      && (*p_j2k.m_output_image).x1 == p_j2k.m_cp.tdx
      && (*p_j2k.m_output_image).y1 == p_j2k.m_cp.tdy
    {
      let mut i: OPJ_UINT32 = 0;
      if !opj_j2k_read_tile_header(p_j2k, p_stream, &mut tile_info, p_manager) {
        return 0i32;
      }
      if opj_j2k_decode_tile(p_j2k, tile_info.index, None, p_stream, p_manager) == 0 {
        event_msg!(p_manager, EVT_ERROR, "Failed to decode tile 1/1\n",);
        return 0i32;
      }
      /* Transfer TCD data to output image data */
      i = 0 as OPJ_UINT32;
      while i < (*p_j2k.m_output_image).numcomps {
        opj_image_data_free(
          (*(*p_j2k.m_output_image).comps.offset(i as isize)).data as *mut core::ffi::c_void,
        );
        let fresh40 = &mut (*(*p_j2k.m_output_image).comps.offset(i as isize)).data;
        *fresh40 = (*p_j2k.m_tcd.tcd_image.tiles.comps.offset(i as isize)).data;
        (*(*p_j2k.m_output_image).comps.offset(i as isize)).resno_decoded =
          (*(*p_j2k.m_tcd.image).comps.offset(i as isize)).resno_decoded;
        let fresh41 = &mut (*p_j2k.m_tcd.tcd_image.tiles.comps.offset(i as isize)).data;
        *fresh41 = core::ptr::null_mut::<OPJ_INT32>();
        i += 1;
      }
      return 1i32;
    }
    loop {
      if p_j2k.m_cp.tw == 1u32
        && p_j2k.m_cp.th == 1u32
        && !(*p_j2k.m_cp.tcps.offset(0)).m_data.is_null()
      {
        tile_info.index = 0 as OPJ_UINT32;
        p_j2k.m_current_tile_number = 0 as OPJ_UINT32;
        p_j2k.m_specific_param.m_decoder.m_state |= J2KState::DATA
      } else {
        if !opj_j2k_read_tile_header(p_j2k, p_stream, &mut tile_info, p_manager) {
          return 0i32;
        }
        if !tile_info.go_on {
          break;
        }
      }
      if opj_j2k_decode_tile(p_j2k, tile_info.index, None, p_stream, p_manager) == 0 {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Failed to decode tile %d/%d\n",
          tile_info.index.wrapping_add(1u32),
          p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw),
        );
        return 0i32;
      }
      event_msg!(
        p_manager,
        EVT_INFO,
        "Tile %d/%d has been decoded.\n",
        tile_info.index.wrapping_add(1u32),
        p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw),
      );
      if opj_j2k_update_image_data(&mut p_j2k.m_tcd, &mut *p_j2k.m_output_image) == 0 {
        return 0i32;
      }
      if !(p_j2k.m_cp.tw == 1u32
        && p_j2k.m_cp.th == 1u32
        && !((*p_j2k.m_output_image).x0 == (*p_j2k.m_private_image).x0
          && (*p_j2k.m_output_image).y0 == (*p_j2k.m_private_image).y0
          && (*p_j2k.m_output_image).x1 == (*p_j2k.m_private_image).x1
          && (*p_j2k.m_output_image).y1 == (*p_j2k.m_private_image).y1))
      {
        opj_j2k_tcp_data_destroy(&mut *p_j2k.m_cp.tcps.offset(tile_info.index as isize));
      }
      event_msg!(
        p_manager,
        EVT_INFO,
        "Image data has been updated with tile %d.\n\n",
        tile_info.index.wrapping_add(1u32),
      );
      if opj_stream_get_number_byte_left(p_stream) == 0i64
        && p_j2k.m_specific_param.m_decoder.m_state == J2KState::NEOC
      {
        break;
      }
      nr_tiles = nr_tiles.wrapping_add(1);
      if nr_tiles == p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw) {
        break;
      }
    }
    if opj_j2k_are_all_used_components_decoded(p_j2k, p_manager) == 0 {
      return 0i32;
    }
    1i32
  }
}

/* *
 * Sets up the procedures to do on decoding data. Developers wanting to extend the library can add their own reading procedures.
 */
fn opj_j2k_setup_decoding(
  _p_j2k: &mut opj_j2k,
  list: &mut opj_j2k_proc_list_t,
  _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  list.add(opj_j2k_decode_tiles);
  /* DEVELOPER CORNER, add your custom procedures */
  1i32
}

/*
 * Read and decode one tile.
 */
fn opj_j2k_decode_one_tile(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut tile_info = TileInfo::default();
    let mut l_nb_tiles: OPJ_UINT32 = 0;
    let mut i: OPJ_UINT32 = 0;
    /*Allocate and initialize some elements of codestrem index if not already done*/
    if (*p_j2k.cstr_index).tile_index.is_null()
      && opj_j2k_allocate_tile_element_cstr_index(p_j2k) == 0
    {
      return 0i32;
    }
    /* Move into the codestream to the first SOT used to decode the desired tile */
    let l_tile_no_to_dec = p_j2k.m_specific_param.m_decoder.m_tile_ind_to_dec as OPJ_UINT32;
    if !(*p_j2k.cstr_index).tile_index.is_null()
      && !(*(*p_j2k.cstr_index).tile_index).tp_index.is_null()
    {
      if (*(*p_j2k.cstr_index)
        .tile_index
        .offset(l_tile_no_to_dec as isize))
      .nb_tps
        == 0
      {
        /* the index for this tile has not been built,
         *  so move to the last SOT read */
        if opj_stream_seek(
          p_stream,
          p_j2k.m_specific_param.m_decoder.m_last_sot_read_pos + 2i64,
          p_manager,
        ) == 0
        {
          event_msg!(p_manager, EVT_ERROR, "Problem with seek function\n",);
          return 0i32;
        }
      } else if opj_stream_seek(
        p_stream,
        (*(*(*p_j2k.cstr_index)
          .tile_index
          .offset(l_tile_no_to_dec as isize))
        .tp_index
        .offset(0))
        .start_pos
          + 2i64,
        p_manager,
      ) == 0
      {
        event_msg!(p_manager, EVT_ERROR, "Problem with seek function\n",);
        return 0i32;
      }
      /* Special case if we have previously read the EOC marker (if the previous tile getted is the last ) */
      if p_j2k.m_specific_param.m_decoder.m_state == J2KState::EOC {
        p_j2k.m_specific_param.m_decoder.m_state = J2KState::TPHSOT
      }
    }
    /* Reset current tile part number for all tiles, and not only the one */
    /* of interest. */
    /* Not completely sure this is always correct but required for */
    /* ./build/bin/j2k_random_tile_access ./build/tests/tte1.j2k */
    l_nb_tiles = p_j2k.m_cp.tw.wrapping_mul(p_j2k.m_cp.th);
    i = 0 as OPJ_UINT32;
    while i < l_nb_tiles {
      (*p_j2k.m_cp.tcps.offset(i as isize)).m_current_tile_part_number = -(1i32);
      i += 1;
    }
    loop {
      if !opj_j2k_read_tile_header(p_j2k, p_stream, &mut tile_info, p_manager) {
        return 0i32;
      }
      if !tile_info.go_on {
        break;
      }
      if opj_j2k_decode_tile(p_j2k, tile_info.index, None, p_stream, p_manager) == 0 {
        return 0i32;
      }
      event_msg!(
        p_manager,
        EVT_INFO,
        "Tile %d/%d has been decoded.\n",
        tile_info.index.wrapping_add(1u32),
        p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw),
      );
      if opj_j2k_update_image_data(&mut p_j2k.m_tcd, &mut *p_j2k.m_output_image) == 0 {
        return 0i32;
      }
      opj_j2k_tcp_data_destroy(&mut *p_j2k.m_cp.tcps.offset(tile_info.index as isize));
      event_msg!(
        p_manager,
        EVT_INFO,
        "Image data has been updated with tile %d.\n\n",
        tile_info.index.wrapping_add(1u32),
      );
      if tile_info.index == l_tile_no_to_dec {
        /* move into the codestream to the first SOT (FIXME or not move?)*/
        if opj_stream_seek(
          p_stream,
          (*p_j2k.cstr_index).main_head_end + 2i64,
          p_manager,
        ) == 0
        {
          event_msg!(p_manager, EVT_ERROR, "Problem with seek function\n",);
          return 0i32;
        }
        break;
      } else {
        event_msg!(
          p_manager,
          EVT_WARNING,
          "Tile read, decoded and updated is not the desired one (%d vs %d).\n",
          tile_info.index.wrapping_add(1u32),
          l_tile_no_to_dec.wrapping_add(1u32),
        );
      }
    }
    if opj_j2k_are_all_used_components_decoded(p_j2k, p_manager) == 0 {
      return 0i32;
    }
    1i32
  }
}

/* *
 * Sets up the procedures to do on decoding one tile. Developers wanting to extend the library can add their own reading procedures.
 */
fn opj_j2k_setup_decoding_tile(
  _p_j2k: &mut opj_j2k,
  list: &mut opj_j2k_proc_list_t,
  _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  list.add(opj_j2k_decode_one_tile);
  /* DEVELOPER CORNER, add your custom procedures */
  1i32
}

fn opj_j2k_move_data_from_codec_to_output_image(
  mut p_j2k: &mut opj_j2k,
  mut p_image: &mut opj_image,
) -> OPJ_BOOL {
  unsafe {
    let mut compno: OPJ_UINT32 = 0;
    /* Move data and copy one information from codec to output image*/
    if p_j2k.m_specific_param.m_decoder.m_numcomps_to_decode > 0u32 {
      let mut newcomps = opj_malloc(
        (p_j2k.m_specific_param.m_decoder.m_numcomps_to_decode as usize)
          .wrapping_mul(core::mem::size_of::<opj_image_comp_t>()),
      ) as *mut opj_image_comp_t;
      if newcomps.is_null() {
        opj_image_destroy(p_j2k.m_private_image);
        p_j2k.m_private_image = core::ptr::null_mut::<opj_image_t>();
        return 0i32;
      }
      compno = 0 as OPJ_UINT32;
      while compno < p_image.numcomps {
        opj_image_data_free(
          (*p_image.comps.offset(compno as isize)).data as *mut core::ffi::c_void,
        );
        let fresh42 = &mut (*p_image.comps.offset(compno as isize)).data;
        *fresh42 = core::ptr::null_mut::<OPJ_INT32>();
        compno += 1;
      }
      compno = 0 as OPJ_UINT32;
      while compno < p_j2k.m_specific_param.m_decoder.m_numcomps_to_decode {
        let mut src_compno = *p_j2k
          .m_specific_param
          .m_decoder
          .m_comps_indices_to_decode
          .offset(compno as isize);
        memcpy(
          &mut *newcomps.offset(compno as isize) as *mut opj_image_comp_t as *mut core::ffi::c_void,
          &mut *(*p_j2k.m_output_image).comps.offset(src_compno as isize) as *mut opj_image_comp_t
            as *const core::ffi::c_void,
          core::mem::size_of::<opj_image_comp_t>(),
        );
        (*newcomps.offset(compno as isize)).resno_decoded =
          (*(*p_j2k.m_output_image).comps.offset(src_compno as isize)).resno_decoded;
        let fresh43 = &mut (*newcomps.offset(compno as isize)).data;
        *fresh43 = (*(*p_j2k.m_output_image).comps.offset(src_compno as isize)).data;
        let fresh44 = &mut (*(*p_j2k.m_output_image).comps.offset(src_compno as isize)).data;
        *fresh44 = core::ptr::null_mut::<OPJ_INT32>();
        compno += 1;
      }
      compno = 0 as OPJ_UINT32;
      while compno < p_image.numcomps {
        assert!((*(*p_j2k.m_output_image).comps.offset(compno as isize))
          .data
          .is_null());
        opj_image_data_free(
          (*(*p_j2k.m_output_image).comps.offset(compno as isize)).data as *mut core::ffi::c_void,
        );
        let fresh45 = &mut (*(*p_j2k.m_output_image).comps.offset(compno as isize)).data;
        *fresh45 = core::ptr::null_mut::<OPJ_INT32>();
        compno += 1;
      }
      p_image.numcomps = p_j2k.m_specific_param.m_decoder.m_numcomps_to_decode;
      opj_free(p_image.comps as *mut core::ffi::c_void);
      p_image.comps = newcomps
    } else {
      compno = 0 as OPJ_UINT32;
      while compno < p_image.numcomps {
        (*p_image.comps.offset(compno as isize)).resno_decoded =
          (*(*p_j2k.m_output_image).comps.offset(compno as isize)).resno_decoded;
        opj_image_data_free(
          (*p_image.comps.offset(compno as isize)).data as *mut core::ffi::c_void,
        );
        let fresh46 = &mut (*p_image.comps.offset(compno as isize)).data;
        *fresh46 = (*(*p_j2k.m_output_image).comps.offset(compno as isize)).data;
        let fresh47 = &mut (*(*p_j2k.m_output_image).comps.offset(compno as isize)).data;
        *fresh47 = core::ptr::null_mut::<OPJ_INT32>();
        compno += 1;
      }
    }
    1i32
  }
}

pub(crate) fn opj_j2k_decode(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_image: &mut opj_image,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    /* Heuristics to detect sequence opj_read_header(), opj_set_decoded_resolution_factor() */
    /* and finally opj_decode_image() without manual setting of comps[].factor */
    /* We could potentially always execute it, if we don't allow people to do */
    /* opj_read_header(), modify x0,y0,x1,y1 of returned image an call opj_decode_image() */
    if p_j2k.m_cp.m_specific_param.m_dec.m_reduce > 0u32
      && !p_j2k.m_private_image.is_null()
      && (*p_j2k.m_private_image).numcomps > 0u32
      && (*(*p_j2k.m_private_image).comps.offset(0)).factor
        == p_j2k.m_cp.m_specific_param.m_dec.m_reduce
      && p_image.numcomps > 0u32
      && (*p_image.comps.offset(0)).factor == 0u32
      && (*p_image.comps.offset(0)).data.is_null()
    {
      let mut it_comp: OPJ_UINT32 = 0;
      /* Update the comps[].factor member of the output image with the one */
      /* of m_reduce */
      it_comp = 0 as OPJ_UINT32;
      while it_comp < p_image.numcomps {
        (*p_image.comps.offset(it_comp as isize)).factor =
          p_j2k.m_cp.m_specific_param.m_dec.m_reduce;
        it_comp += 1;
      }
      if opj_j2k_update_image_dimensions(p_image, p_manager) == 0 {
        return 0i32;
      }
    }
    if p_j2k.m_output_image.is_null() {
      p_j2k.m_output_image = opj_image_create0();
      if p_j2k.m_output_image.is_null() {
        return 0i32;
      }
    }
    opj_copy_image_header(p_image, p_j2k.m_output_image);
    /* customization of the decoding */
    let mut procedure_list = opj_j2k_proc_list_t::new();
    if opj_j2k_setup_decoding(p_j2k, &mut procedure_list, p_manager) == 0 {
      return 0i32;
    }
    /* Decode the codestream */
    if opj_j2k_exec(p_j2k, &mut procedure_list, p_stream, p_manager) == 0 {
      opj_image_destroy(p_j2k.m_private_image);
      p_j2k.m_private_image = core::ptr::null_mut::<opj_image_t>();
      return 0i32;
    }
    /* Move data and copy one information from codec to output image*/
    opj_j2k_move_data_from_codec_to_output_image(p_j2k, p_image)
  }
}

pub(crate) fn opj_j2k_get_tile(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_image: &mut opj_image,
  mut p_manager: &mut opj_event_mgr,
  mut tile_index: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut compno: OPJ_UINT32 = 0;
    let mut l_tile_x: OPJ_UINT32 = 0;
    let mut l_tile_y: OPJ_UINT32 = 0;
    let mut l_img_comp = core::ptr::null_mut::<opj_image_comp_t>();
    if p_image.numcomps < (*p_j2k.m_private_image).numcomps {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Image has less components than codestream.\n",
      );
      return 0i32;
    }
    if tile_index >= p_j2k.m_cp.tw.wrapping_mul(p_j2k.m_cp.th) {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Tile index provided by the user is incorrect %d (max = %d) \n",
        tile_index,
        p_j2k.m_cp.tw.wrapping_mul(p_j2k.m_cp.th).wrapping_sub(1u32),
      );
      return 0i32;
    }
    /* Compute the dimension of the desired tile*/
    l_tile_x = tile_index.wrapping_rem(p_j2k.m_cp.tw);
    l_tile_y = tile_index.wrapping_div(p_j2k.m_cp.tw);
    p_image.x0 = l_tile_x
      .wrapping_mul(p_j2k.m_cp.tdx)
      .wrapping_add(p_j2k.m_cp.tx0);
    if p_image.x0 < (*p_j2k.m_private_image).x0 {
      p_image.x0 = (*p_j2k.m_private_image).x0
    }
    p_image.x1 = l_tile_x
      .wrapping_add(1u32)
      .wrapping_mul(p_j2k.m_cp.tdx)
      .wrapping_add(p_j2k.m_cp.tx0);
    if p_image.x1 > (*p_j2k.m_private_image).x1 {
      p_image.x1 = (*p_j2k.m_private_image).x1
    }
    p_image.y0 = l_tile_y
      .wrapping_mul(p_j2k.m_cp.tdy)
      .wrapping_add(p_j2k.m_cp.ty0);
    if p_image.y0 < (*p_j2k.m_private_image).y0 {
      p_image.y0 = (*p_j2k.m_private_image).y0
    }
    p_image.y1 = l_tile_y
      .wrapping_add(1u32)
      .wrapping_mul(p_j2k.m_cp.tdy)
      .wrapping_add(p_j2k.m_cp.ty0);
    if p_image.y1 > (*p_j2k.m_private_image).y1 {
      p_image.y1 = (*p_j2k.m_private_image).y1
    }

    l_img_comp = p_image.comps;
    compno = 0 as OPJ_UINT32;
    while compno < (*p_j2k.m_private_image).numcomps {
      let mut l_comp_x1: OPJ_INT32 = 0;
      let mut l_comp_y1: OPJ_INT32 = 0;

      (*l_img_comp).factor = (*(*p_j2k.m_private_image).comps.offset(compno as isize)).factor;

      (*l_img_comp).x0 = opj_uint_ceildiv(p_image.x0, (*l_img_comp).dx);
      (*l_img_comp).y0 = opj_uint_ceildiv(p_image.y0, (*l_img_comp).dy);
      l_comp_x1 = opj_int_ceildiv(p_image.x1 as OPJ_INT32, (*l_img_comp).dx as OPJ_INT32);
      l_comp_y1 = opj_int_ceildiv(p_image.y1 as OPJ_INT32, (*l_img_comp).dy as OPJ_INT32);

      (*l_img_comp).w = (opj_int_ceildivpow2(l_comp_x1, (*l_img_comp).factor as OPJ_INT32)
        - opj_int_ceildivpow2(
          (*l_img_comp).x0 as OPJ_INT32,
          (*l_img_comp).factor as OPJ_INT32,
        )) as OPJ_UINT32;
      (*l_img_comp).h = (opj_int_ceildivpow2(l_comp_y1, (*l_img_comp).factor as OPJ_INT32)
        - opj_int_ceildivpow2(
          (*l_img_comp).y0 as OPJ_INT32,
          (*l_img_comp).factor as OPJ_INT32,
        )) as OPJ_UINT32;
      l_img_comp = l_img_comp.offset(1);
      compno += 1;
    }
    if p_image.numcomps > (*p_j2k.m_private_image).numcomps {
      /* Can happen when calling repeatdly opj_get_decoded_tile() on an
       * image with a color palette, where color palette expansion is done
       * later in jp2.c */
      compno = (*p_j2k.m_private_image).numcomps;
      while compno < p_image.numcomps {
        opj_image_data_free(
          (*p_image.comps.offset(compno as isize)).data as *mut core::ffi::c_void,
        );
        let fresh48 = &mut (*p_image.comps.offset(compno as isize)).data;
        *fresh48 = core::ptr::null_mut::<OPJ_INT32>();
        compno += 1;
      }
      p_image.numcomps = (*p_j2k.m_private_image).numcomps
    }
    /* Destroy the previous output image*/
    if !p_j2k.m_output_image.is_null() {
      opj_image_destroy(p_j2k.m_output_image);
    }
    /* Create the output image from the information previously computed*/
    p_j2k.m_output_image = opj_image_create0();
    if p_j2k.m_output_image.is_null() {
      return 0i32;
    }
    opj_copy_image_header(p_image, p_j2k.m_output_image);
    p_j2k.m_specific_param.m_decoder.m_tile_ind_to_dec = tile_index as OPJ_INT32;
    let mut procedure_list = opj_j2k_proc_list_t::new();
    /* customization of the decoding */
    if opj_j2k_setup_decoding_tile(p_j2k, &mut procedure_list, p_manager) == 0 {
      return 0i32;
    }
    /* Decode the codestream */
    if opj_j2k_exec(p_j2k, &mut procedure_list, p_stream, p_manager) == 0 {
      opj_image_destroy(p_j2k.m_private_image);
      p_j2k.m_private_image = core::ptr::null_mut::<opj_image_t>();
      return 0i32;
    }
    /* Move data and copy one information from codec to output image*/
    opj_j2k_move_data_from_codec_to_output_image(p_j2k, p_image)
  }
}

pub(crate) fn opj_j2k_set_decoded_resolution_factor(
  mut p_j2k: &mut opj_j2k,
  mut res_factor: OPJ_UINT32,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut it_comp: OPJ_UINT32 = 0;
    p_j2k.m_cp.m_specific_param.m_dec.m_reduce = res_factor;
    if !p_j2k.m_private_image.is_null()
      && !(*p_j2k.m_private_image).comps.is_null()
      && !p_j2k.m_specific_param.m_decoder.m_default_tcp.is_null()
      && !(*p_j2k.m_specific_param.m_decoder.m_default_tcp)
        .tccps
        .is_null()
    {
      it_comp = 0 as OPJ_UINT32;
      while it_comp < (*p_j2k.m_private_image).numcomps {
        let mut max_res = (*(*p_j2k.m_specific_param.m_decoder.m_default_tcp)
          .tccps
          .offset(it_comp as isize))
        .numresolutions;
        if res_factor >= max_res {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Resolution factor is greater than the maximum resolution in the component.\n",
          );
          return 0i32;
        }
        (*(*p_j2k.m_private_image).comps.offset(it_comp as isize)).factor = res_factor;
        it_comp += 1;
      }
      return 1i32;
    }
    0i32
  }
}

pub(crate) fn opj_j2k_encoder_set_extra_options(
  p_j2k: &mut opj_j2k,
  options: &[&str],
  p_manager: &mut opj_event_mgr,
) -> bool {
  for option in options {
    if option.starts_with("PLT=") {
      if *option == "PLT=YES" {
        p_j2k.m_specific_param.m_encoder.m_PLT = 1i32
      } else if *option == "PLT=NO" {
        p_j2k.m_specific_param.m_encoder.m_PLT = 0i32
      } else {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Invalid value for option: %s.\n",
          *option,
        );
        return false;
      }
    } else if option.starts_with("TLM=") {
      if *option == "TLM=YES" {
        p_j2k.m_specific_param.m_encoder.m_TLM = 1i32
      } else if *option == "TLM=NO" {
        p_j2k.m_specific_param.m_encoder.m_TLM = 0i32
      } else {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Invalid value for option: %s.\n",
          *option,
        );
        return false;
      }
    } else if option.starts_with("GUARD_BITS=") {
      let mut tileno: OPJ_UINT32 = 0;
      let mut cp = &mut p_j2k.m_cp;
      let mut numgbits = option[11..].parse::<i32>().unwrap_or_default();
      if !(0..=7).contains(&numgbits) {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Invalid value for option: %s. Should be in [0,7]\n",
          *option,
        );
        return false;
      }
      unsafe {
        tileno = 0 as OPJ_UINT32;
        while tileno < cp.tw.wrapping_mul(cp.th) {
          let mut i: OPJ_UINT32 = 0;
          let mut tcp: *mut opj_tcp_t = &mut *cp.tcps.offset(tileno as isize) as *mut opj_tcp_t;
          i = 0 as OPJ_UINT32;
          while i < p_j2k.m_specific_param.m_encoder.m_nb_comps {
            let mut tccp: *mut opj_tccp_t =
              &mut *(*tcp).tccps.offset(i as isize) as *mut opj_tccp_t;
            (*tccp).numgbits = numgbits as OPJ_UINT32;
            i += 1;
          }
          tileno += 1;
        }
      }
    } else {
      event_msg!(p_manager, EVT_ERROR, "Invalid option: %s.\n", *option);
      return false;
    }
  }
  true
}

pub(crate) fn opj_j2k_encode(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_UINT32 = 0;
    let mut l_nb_tiles: OPJ_UINT32 = 0;
    let mut l_max_tile_size = 0 as OPJ_SIZE_T;
    let mut l_current_tile_size: OPJ_SIZE_T = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_reuse_data = 0i32;
    /* preconditions */

    l_nb_tiles = p_j2k.m_cp.th.wrapping_mul(p_j2k.m_cp.tw);
    if l_nb_tiles == 1u32 {
      l_reuse_data = 1i32
    }
    i = 0 as OPJ_UINT32;
    while i < l_nb_tiles {
      if opj_j2k_pre_write_tile(p_j2k, i, p_stream, p_manager) == 0 {
        if !l_current_data.is_null() {
          opj_free(l_current_data as *mut core::ffi::c_void);
        }
        return 0i32;
      }
      /* if we only have one tile, then simply set tile component data equal to image component data */
      /* otherwise, allocate the data */
      j = 0 as OPJ_UINT32;
      while j < (*p_j2k.m_tcd.image).numcomps {
        let mut l_tilec = p_j2k.m_tcd.tcd_image.tiles.comps.offset(j as isize);
        if l_reuse_data != 0 {
          let mut l_img_comp = (*p_j2k.m_tcd.image).comps.offset(j as isize);
          (*l_tilec).data = (*l_img_comp).data;
          (*l_tilec).ownsData = 0i32
        } else if opj_alloc_tile_component_data(l_tilec) == 0 {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Error allocating tile component data.",
          );
          if !l_current_data.is_null() {
            opj_free(l_current_data as *mut core::ffi::c_void);
          }
          return 0i32;
        }
        j += 1;
      }
      l_current_tile_size = opj_tcd_get_encoder_input_buffer_size(&mut p_j2k.m_tcd);
      if l_reuse_data == 0 {
        if l_current_tile_size > l_max_tile_size {
          let mut l_new_current_data = opj_realloc(
            l_current_data as *mut core::ffi::c_void,
            l_current_tile_size,
          ) as *mut OPJ_BYTE;
          if l_new_current_data.is_null() {
            if !l_current_data.is_null() {
              opj_free(l_current_data as *mut core::ffi::c_void);
            }
            event_msg!(
              p_manager,
              EVT_ERROR,
              "Not enough memory to encode all tiles\n",
            );
            return 0i32;
          }
          l_current_data = l_new_current_data;
          l_max_tile_size = l_current_tile_size
        }
        if l_current_data.is_null() {
          /* Should not happen in practice, but will avoid Coverity to */
          /* complain about a null pointer dereference */
          panic!("");
          // C: assert(0);
        }
        /* copy image data (32 bit) to l_current_data as contiguous, all-component, zero offset buffer */
        /* 32 bit components @ 8 bit precision get converted to 8 bit */
        /* 32 bit components @ 16 bit precision get converted to 16 bit */
        let p_data =
          core::slice::from_raw_parts_mut(l_current_data as *mut u8, l_current_tile_size as usize);
        opj_j2k_get_tile_data(&mut p_j2k.m_tcd, p_data);
        /* now copy this data into the tile component */
        if opj_tcd_copy_tile_data(&mut p_j2k.m_tcd, p_data) == 0 {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Size mismatch between tile data and sent data.",
          );
          opj_free(l_current_data as *mut core::ffi::c_void);
          return 0i32;
        }
      }
      if opj_j2k_post_write_tile(p_j2k, p_stream, p_manager) == 0 {
        if !l_current_data.is_null() {
          opj_free(l_current_data as *mut core::ffi::c_void);
        }
        return 0i32;
      }
      i += 1;
    }
    if !l_current_data.is_null() {
      opj_free(l_current_data as *mut core::ffi::c_void);
    }
    1i32
  }
}

pub(crate) fn opj_j2k_end_compress(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  let mut procedure_list = opj_j2k_proc_list_t::new();
  /* customization of the encoding */
  if opj_j2k_setup_end_compress(p_j2k, &mut procedure_list, p_manager) == 0 {
    return 0i32;
  }
  if opj_j2k_exec(p_j2k, &mut procedure_list, p_stream, p_manager) == 0 {
    return 0i32;
  }
  1i32
}

pub(crate) fn opj_j2k_start_compress(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_image: &mut opj_image,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  let mut validation_list = opj_j2k_proc_list_t::new();
  let mut procedure_list = opj_j2k_proc_list_t::new();
  /* preconditions */

  unsafe {
    p_j2k.m_private_image = opj_image_create0();
    if p_j2k.m_private_image.is_null() {
      event_msg!(p_manager, EVT_ERROR, "Failed to allocate image header.",);
      return 0i32;
    }
    opj_copy_image_header(p_image, p_j2k.m_private_image);
    /* TODO_MSD: Find a better way */
    if !p_image.comps.is_null() {
      let mut it_comp: OPJ_UINT32 = 0;
      it_comp = 0 as OPJ_UINT32;
      while it_comp < p_image.numcomps {
        if !(*p_image.comps.offset(it_comp as isize)).data.is_null() {
          let fresh49 = &mut (*(*p_j2k.m_private_image).comps.offset(it_comp as isize)).data;
          *fresh49 = (*p_image.comps.offset(it_comp as isize)).data;
          let fresh50 = &mut (*p_image.comps.offset(it_comp as isize)).data;
          *fresh50 = core::ptr::null_mut::<OPJ_INT32>()
        }
        it_comp += 1;
      }
    }
    /* customization of the validation */
    if opj_j2k_setup_encoding_validation(p_j2k, &mut validation_list, p_manager) == 0 {
      return 0i32;
    }
    /* validation of the parameters codec */
    if opj_j2k_exec(p_j2k, &mut validation_list, p_stream, p_manager) == 0 {
      return 0i32;
    }
    /* customization of the encoding */
    if opj_j2k_setup_header_writing(p_j2k, &mut procedure_list, p_manager) == 0 {
      return 0i32;
    }
    /* write header */
    if opj_j2k_exec(p_j2k, &mut procedure_list, p_stream, p_manager) == 0 {
      return 0i32;
    }
    1i32
  }
}

fn opj_j2k_pre_write_tile(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_index: OPJ_UINT32,
  mut _p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    if p_tile_index != p_j2k.m_current_tile_number {
      event_msg!(p_manager, EVT_ERROR, "The given tile index does not match.",);
      return 0i32;
    }
    event_msg!(
      p_manager,
      EVT_INFO,
      "tile number %d / %d\n",
      p_j2k.m_current_tile_number.wrapping_add(1u32),
      p_j2k.m_cp.tw.wrapping_mul(p_j2k.m_cp.th),
    );
    p_j2k.m_specific_param.m_encoder.m_current_tile_part_number = 0 as OPJ_UINT32;
    p_j2k.m_tcd.cur_totnum_tp = (*p_j2k.m_cp.tcps.offset(p_tile_index as isize)).m_nb_tile_parts;
    p_j2k
      .m_specific_param
      .m_encoder
      .m_current_poc_tile_part_number = 0 as OPJ_UINT32;
    /* initialisation before tile encoding  */
    if opj_tcd_init_encode_tile(&mut p_j2k.m_tcd, p_j2k.m_current_tile_number, p_manager) == 0 {
      return 0i32;
    } /* (/8) */
    1i32 /* (%8) */
  }
}

fn opj_get_tile_dimensions(
  mut l_image: &mut opj_image,
  mut l_tilec: &opj_tcd_tilecomp_t,
  mut l_img_comp: &opj_image_comp_t,
  mut l_size_comp: *mut OPJ_UINT32,
  mut l_width: *mut OPJ_UINT32,
  mut l_height: *mut OPJ_UINT32,
  mut l_offset_x: *mut OPJ_UINT32,
  mut l_offset_y: *mut OPJ_UINT32,
  mut l_image_width: *mut OPJ_UINT32,
  mut l_stride: *mut OPJ_UINT32,
  mut l_tile_offset: *mut OPJ_UINT32,
) {
  unsafe {
    let mut l_remaining: OPJ_UINT32 = 0;
    *l_size_comp = (*l_img_comp).prec >> 3i32;
    l_remaining = (*l_img_comp).prec & 7u32;
    if l_remaining != 0 {
      *l_size_comp = (*l_size_comp as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32
    }
    if *l_size_comp == 3u32 {
      *l_size_comp = 4 as OPJ_UINT32
    }

    *l_width = ((*l_tilec).x1 - (*l_tilec).x0) as OPJ_UINT32;
    *l_height = ((*l_tilec).y1 - (*l_tilec).y0) as OPJ_UINT32;
    *l_offset_x = opj_uint_ceildiv((*l_image).x0, (*l_img_comp).dx);
    *l_offset_y = opj_uint_ceildiv((*l_image).y0, (*l_img_comp).dy);
    *l_image_width = opj_uint_ceildiv((*l_image).x1 - (*l_image).x0, (*l_img_comp).dx);
    *l_stride = (*l_image_width).wrapping_sub(*l_width);
    *l_tile_offset = ((*l_tilec).x0 as OPJ_UINT32)
      .wrapping_sub(*l_offset_x)
      .wrapping_add(
        ((*l_tilec).y0 as OPJ_UINT32)
          .wrapping_sub(*l_offset_y)
          .wrapping_mul(*l_image_width),
      );
  }
}

fn opj_j2k_get_tile_data(mut p_tcd: &mut opj_tcd, mut p_data: &mut [u8]) {
  unsafe {
    let mut l_image = (*p_tcd).image;
    let numcomps = (*p_tcd.image).numcomps as usize;
    let mut l_tilec = core::slice::from_raw_parts(p_tcd.tcd_image.tiles.comps, numcomps);
    let mut l_img_comp = core::slice::from_raw_parts((*l_image).comps, numcomps);
    for (l_tilec, l_img_comp) in l_tilec.iter().zip(l_img_comp.iter()) {
      let mut l_size_comp: OPJ_UINT32 = 0;
      let mut l_width: OPJ_UINT32 = 0;
      let mut l_height: OPJ_UINT32 = 0;
      let mut l_offset_x: OPJ_UINT32 = 0;
      let mut l_offset_y: OPJ_UINT32 = 0;
      let mut l_image_width: OPJ_UINT32 = 0;
      let mut l_stride: OPJ_UINT32 = 0;
      let mut l_tile_offset: OPJ_UINT32 = 0;
      opj_get_tile_dimensions(
        &mut *l_image,
        l_tilec,
        l_img_comp,
        &mut l_size_comp,
        &mut l_width,
        &mut l_height,
        &mut l_offset_x,
        &mut l_offset_y,
        &mut l_image_width,
        &mut l_stride,
        &mut l_tile_offset,
      );
      let mut l_src_ptr = (*l_img_comp).data.offset(l_tile_offset as isize);
      let l_height = l_height as usize;
      let l_width = l_width as usize;
      let l_stride = l_stride as usize;
      let l_nb_elem = l_height * l_width;
      let mut l_src = core::slice::from_raw_parts(l_src_ptr, l_nb_elem + (l_height * l_stride));
      match l_size_comp {
        1 => {
          let (dest, remain) = p_data.split_at_mut(l_nb_elem);
          p_data = remain;
          if l_img_comp.sgnd != 0 {
            for (src, dest) in l_src
              .chunks_exact(l_width + l_stride)
              .zip(dest.chunks_exact_mut(l_width))
            {
              let src = &src[0..l_width];
              for (src, dest) in src.iter().zip(dest.iter_mut()) {
                *dest = *src as i8 as u8;
              }
            }
          } else {
            for (src, dest) in l_src
              .chunks_exact(l_width + l_stride)
              .zip(dest.chunks_exact_mut(l_width))
            {
              let src = &src[0..l_width];
              for (src, dest) in src.iter().zip(dest.iter_mut()) {
                *dest = (*src & 0xffi32) as u8;
              }
            }
          }
        }
        2 => {
          let (dest, remain) = p_data.split_at_mut(l_nb_elem as usize * 2);
          p_data = remain;
          if l_img_comp.sgnd != 0 {
            for (src, dest) in l_src
              .chunks_exact(l_width + l_stride)
              .zip(dest.chunks_exact_mut(l_width * 2))
            {
              let src = &src[0..l_width];
              for (src, dest) in src.iter().zip(dest.chunks_exact_mut(2)) {
                let val = *src as i16;
                dest.copy_from_slice(&val.to_ne_bytes());
              }
            }
          } else {
            for (src, dest) in l_src
              .chunks_exact(l_width + l_stride)
              .zip(dest.chunks_exact_mut(l_width * 2))
            {
              let src = &src[0..l_width];
              for (src, dest) in src.iter().zip(dest.chunks_exact_mut(2)) {
                let val = (*src & 0xffffi32) as i16;
                dest.copy_from_slice(&val.to_ne_bytes());
              }
            }
          }
        }
        4 => {
          let (dest, remain) = p_data.split_at_mut(l_nb_elem as usize * 4);
          p_data = remain;
          for (src, dest) in l_src
            .chunks_exact(l_width + l_stride)
            .zip(dest.chunks_exact_mut(l_width * 4))
          {
            let src = &src[0..l_width];
            for (src, dest) in src.iter().zip(dest.chunks_exact_mut(4)) {
              dest.copy_from_slice(&src.to_ne_bytes());
            }
          }
        }
        _ => {}
      }
    }
  }
}

fn opj_j2k_post_write_tile(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_nb_bytes_written: OPJ_UINT32 = 0;
    let mut l_current_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_tile_size = 0 as OPJ_UINT32;
    let mut l_available_data: OPJ_UINT32 = 0;
    /* preconditions */
    assert!(!p_j2k
      .m_specific_param
      .m_encoder
      .m_encoded_tile_data
      .is_null());
    l_tile_size = p_j2k.m_specific_param.m_encoder.m_encoded_tile_size;
    l_available_data = l_tile_size;
    l_current_data = p_j2k.m_specific_param.m_encoder.m_encoded_tile_data;
    l_nb_bytes_written = 0 as OPJ_UINT32;
    if opj_j2k_write_first_tile_part(
      p_j2k,
      l_current_data,
      &mut l_nb_bytes_written,
      l_available_data,
      p_stream,
      p_manager,
    ) == 0
    {
      return 0i32;
    }
    l_current_data = l_current_data.offset(l_nb_bytes_written as isize);
    l_available_data =
      (l_available_data as core::ffi::c_uint).wrapping_sub(l_nb_bytes_written) as OPJ_UINT32;
    l_nb_bytes_written = 0 as OPJ_UINT32;
    if opj_j2k_write_all_tile_parts(
      p_j2k,
      l_current_data,
      &mut l_nb_bytes_written,
      l_available_data,
      p_stream,
      p_manager,
    ) == 0
    {
      return 0i32;
    }
    l_available_data =
      (l_available_data as core::ffi::c_uint).wrapping_sub(l_nb_bytes_written) as OPJ_UINT32;
    l_nb_bytes_written = l_tile_size.wrapping_sub(l_available_data);
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_encoded_tile_data,
      l_nb_bytes_written as OPJ_SIZE_T,
      p_manager,
    ) != l_nb_bytes_written as usize
    {
      return 0i32;
    }
    p_j2k.m_current_tile_number = p_j2k.m_current_tile_number.wrapping_add(1);
    1i32
  }
}

/* *
 * Sets up the validation ,i.e. adds the procedures to launch to make sure the codec parameters
 * are valid. Developers wanting to extend the library can add their own validation procedures.
 */
fn opj_j2k_setup_end_compress(
  p_j2k: &mut opj_j2k,
  list: &mut opj_j2k_proc_list_t,
  _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* DEVELOPER CORNER, insert your custom procedures */
  list.add(opj_j2k_write_eoc);
  if unsafe { p_j2k.m_specific_param.m_encoder.m_TLM } != 0 {
    list.add(opj_j2k_write_updated_tlm);
  }
  list.add(opj_j2k_write_epc);
  list.add(opj_j2k_end_encoding);
  list.add(opj_j2k_destroy_header_memory);
  1i32
}

/* *
 * Sets up the validation ,i.e. adds the procedures to launch to make sure the codec parameters
 * are valid. Developers wanting to extend the library can add their own validation procedures.
 */
fn opj_j2k_setup_encoding_validation(
  _p_j2k: &mut opj_j2k,
  list: &mut opj_j2k_proc_list_t,
  _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  list.add(opj_j2k_build_encoder);
  list.add(opj_j2k_encoding_validation);
  /* DEVELOPER CORNER, add your custom validation procedure */
  list.add(opj_j2k_mct_validation);
  1i32
}

/* *
 * Sets up the procedures to do on writing header.
 * Developers wanting to extend the library can add their own writing procedures.
 */
fn opj_j2k_setup_header_writing(
  p_j2k: &mut opj_j2k,
  list: &mut opj_j2k_proc_list_t,
  _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  list.add(opj_j2k_init_info);
  list.add(opj_j2k_write_soc);
  list.add(opj_j2k_write_siz);
  list.add(opj_j2k_write_cod);
  list.add(opj_j2k_write_qcd);
  list.add(opj_j2k_write_all_coc);
  list.add(opj_j2k_write_all_qcc);
  if unsafe { p_j2k.m_specific_param.m_encoder.m_TLM } != 0 {
    list.add(opj_j2k_write_tlm);
    if p_j2k.m_cp.rsiz as core::ffi::c_int == 0x4i32 {
      list.add(opj_j2k_write_poc);
    }
  }
  list.add(opj_j2k_write_regions);
  if !p_j2k.m_cp.comment.is_null() {
    list.add(opj_j2k_write_com);
  }
  /* DEVELOPER CORNER, insert your custom procedures */
  if p_j2k.m_cp.rsiz as core::ffi::c_int & (0x8000i32 | 0x100i32) == 0x8000i32 | 0x100i32 {
    list.add(opj_j2k_write_mct_data_group);
  }
  /* End of Developer Corner */
  if !p_j2k.cstr_index.is_null() {
    list.add(opj_j2k_get_end_header);
  }
  list.add(opj_j2k_create_tcd);
  list.add(opj_j2k_update_rates);
  1i32
}

fn opj_j2k_write_first_tile_part(
  mut p_j2k: &mut opj_j2k,
  mut p_data: *mut OPJ_BYTE,
  mut p_data_written: *mut OPJ_UINT32,
  mut total_data_size: OPJ_UINT32,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_nb_bytes_written = 0 as OPJ_UINT32;
    let mut l_current_nb_bytes_written: OPJ_UINT32 = 0;
    let mut l_begin_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    l_cp = &mut p_j2k.m_cp;
    p_j2k.m_tcd.cur_pino = 0 as OPJ_UINT32;
    /*Get number of tile parts*/
    p_j2k
      .m_specific_param
      .m_encoder
      .m_current_poc_tile_part_number = 0 as OPJ_UINT32;
    /* INDEX >> */
    /* << INDEX */
    l_current_nb_bytes_written = 0 as OPJ_UINT32;
    l_begin_data = p_data;
    if opj_j2k_write_sot(
      p_j2k,
      p_data,
      total_data_size,
      &mut l_current_nb_bytes_written,
      p_stream,
      p_manager,
    ) == 0
    {
      return 0i32;
    }
    l_nb_bytes_written = (l_nb_bytes_written as core::ffi::c_uint)
      .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32;
    p_data = p_data.offset(l_current_nb_bytes_written as isize);
    total_data_size =
      (total_data_size as core::ffi::c_uint).wrapping_sub(l_current_nb_bytes_written) as OPJ_UINT32;
    if !((*l_cp).rsiz as core::ffi::c_int >= 0x3i32 && (*l_cp).rsiz as core::ffi::c_int <= 0x6i32)
      && (*(*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize)).POC
    {
      l_current_nb_bytes_written = 0 as OPJ_UINT32;
      opj_j2k_write_poc_in_memory(p_j2k, p_data, &mut l_current_nb_bytes_written, p_manager);
      l_nb_bytes_written = (l_nb_bytes_written as core::ffi::c_uint)
        .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32
        as OPJ_UINT32;
      p_data = p_data.offset(l_current_nb_bytes_written as isize);
      total_data_size = (total_data_size as core::ffi::c_uint)
        .wrapping_sub(l_current_nb_bytes_written) as OPJ_UINT32
    }
    l_current_nb_bytes_written = 0 as OPJ_UINT32;
    if opj_j2k_write_sod(
      p_j2k,
      p_data,
      &mut l_current_nb_bytes_written,
      total_data_size,
      p_stream,
      p_manager,
    ) == 0
    {
      return 0i32;
    }
    l_nb_bytes_written = (l_nb_bytes_written as core::ffi::c_uint)
      .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32;
    *p_data_written = l_nb_bytes_written;
    /* Writing Psot in SOT marker */
    opj_write_bytes(l_begin_data.offset(6), l_nb_bytes_written, 4 as OPJ_UINT32); /* PSOT */
    if p_j2k.m_specific_param.m_encoder.m_TLM != 0 {
      opj_j2k_update_tlm(p_j2k, l_nb_bytes_written);
    }
    1i32
  }
}

fn opj_j2k_write_all_tile_parts(
  mut p_j2k: &mut opj_j2k,
  mut p_data: *mut OPJ_BYTE,
  mut p_data_written: *mut OPJ_UINT32,
  mut total_data_size: OPJ_UINT32,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut tilepartno = 0 as OPJ_UINT32;
    let mut l_nb_bytes_written = 0 as OPJ_UINT32;
    let mut l_current_nb_bytes_written: OPJ_UINT32 = 0;
    let mut l_part_tile_size: OPJ_UINT32 = 0;
    let mut tot_num_tp: OPJ_UINT32 = 0;
    let mut pino: OPJ_UINT32 = 0;
    let mut l_begin_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_cp = core::ptr::null_mut::<opj_cp_t>();
    l_cp = &mut p_j2k.m_cp;
    l_tcp = (*l_cp).tcps.offset(p_j2k.m_current_tile_number as isize);
    /*Get number of tile parts*/
    tot_num_tp = opj_j2k_get_num_tp(l_cp, 0 as OPJ_UINT32, p_j2k.m_current_tile_number);
    /* start writing remaining tile parts */
    p_j2k.m_specific_param.m_encoder.m_current_tile_part_number = p_j2k
      .m_specific_param
      .m_encoder
      .m_current_tile_part_number
      .wrapping_add(1);
    tilepartno = 1 as OPJ_UINT32;
    while tilepartno < tot_num_tp {
      p_j2k
        .m_specific_param
        .m_encoder
        .m_current_poc_tile_part_number = tilepartno;
      l_current_nb_bytes_written = 0 as OPJ_UINT32;
      l_part_tile_size = 0 as OPJ_UINT32;
      l_begin_data = p_data;
      if opj_j2k_write_sot(
        p_j2k,
        p_data,
        total_data_size,
        &mut l_current_nb_bytes_written,
        p_stream,
        p_manager,
      ) == 0
      {
        return 0i32;
      }
      l_nb_bytes_written = (l_nb_bytes_written as core::ffi::c_uint)
        .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32
        as OPJ_UINT32;
      p_data = p_data.offset(l_current_nb_bytes_written as isize);
      total_data_size = (total_data_size as core::ffi::c_uint)
        .wrapping_sub(l_current_nb_bytes_written) as OPJ_UINT32;
      l_part_tile_size = (l_part_tile_size as core::ffi::c_uint)
        .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32;
      l_current_nb_bytes_written = 0 as OPJ_UINT32;
      if opj_j2k_write_sod(
        p_j2k,
        p_data,
        &mut l_current_nb_bytes_written,
        total_data_size,
        p_stream,
        p_manager,
      ) == 0
      {
        return 0i32;
      }
      p_data = p_data.offset(l_current_nb_bytes_written as isize);
      l_nb_bytes_written = (l_nb_bytes_written as core::ffi::c_uint)
        .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32
        as OPJ_UINT32;
      total_data_size = (total_data_size as core::ffi::c_uint)
        .wrapping_sub(l_current_nb_bytes_written) as OPJ_UINT32;
      l_part_tile_size = (l_part_tile_size as core::ffi::c_uint)
        .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32;
      /* Writing Psot in SOT marker */
      opj_write_bytes(l_begin_data.offset(6), l_part_tile_size, 4 as OPJ_UINT32); /* PSOT */
      if p_j2k.m_specific_param.m_encoder.m_TLM != 0 {
        opj_j2k_update_tlm(p_j2k, l_part_tile_size);
      }
      p_j2k.m_specific_param.m_encoder.m_current_tile_part_number = p_j2k
        .m_specific_param
        .m_encoder
        .m_current_tile_part_number
        .wrapping_add(1);
      tilepartno += 1;
    }
    pino = 1 as OPJ_UINT32;
    while pino <= (*l_tcp).numpocs {
      p_j2k.m_tcd.cur_pino = pino;
      /*Get number of tile parts*/
      tot_num_tp = opj_j2k_get_num_tp(l_cp, pino, p_j2k.m_current_tile_number);
      tilepartno = 0 as OPJ_UINT32;
      while tilepartno < tot_num_tp {
        p_j2k
          .m_specific_param
          .m_encoder
          .m_current_poc_tile_part_number = tilepartno;
        l_current_nb_bytes_written = 0 as OPJ_UINT32;
        l_part_tile_size = 0 as OPJ_UINT32;
        l_begin_data = p_data;
        if opj_j2k_write_sot(
          p_j2k,
          p_data,
          total_data_size,
          &mut l_current_nb_bytes_written,
          p_stream,
          p_manager,
        ) == 0
        {
          return 0i32;
        }
        l_nb_bytes_written = (l_nb_bytes_written as core::ffi::c_uint)
          .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32
          as OPJ_UINT32;
        p_data = p_data.offset(l_current_nb_bytes_written as isize);
        total_data_size = (total_data_size as core::ffi::c_uint)
          .wrapping_sub(l_current_nb_bytes_written) as OPJ_UINT32;
        l_part_tile_size = (l_part_tile_size as core::ffi::c_uint)
          .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32;
        l_current_nb_bytes_written = 0 as OPJ_UINT32;
        if opj_j2k_write_sod(
          p_j2k,
          p_data,
          &mut l_current_nb_bytes_written,
          total_data_size,
          p_stream,
          p_manager,
        ) == 0
        {
          return 0i32;
        }
        l_nb_bytes_written = (l_nb_bytes_written as core::ffi::c_uint)
          .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32
          as OPJ_UINT32;
        p_data = p_data.offset(l_current_nb_bytes_written as isize);
        total_data_size = (total_data_size as core::ffi::c_uint)
          .wrapping_sub(l_current_nb_bytes_written) as OPJ_UINT32;
        l_part_tile_size = (l_part_tile_size as core::ffi::c_uint)
          .wrapping_add(l_current_nb_bytes_written) as OPJ_UINT32;
        /* Writing Psot in SOT marker */
        opj_write_bytes(l_begin_data.offset(6), l_part_tile_size, 4 as OPJ_UINT32); /* PSOT */
        if p_j2k.m_specific_param.m_encoder.m_TLM != 0 {
          opj_j2k_update_tlm(p_j2k, l_part_tile_size);
        }
        p_j2k.m_specific_param.m_encoder.m_current_tile_part_number = p_j2k
          .m_specific_param
          .m_encoder
          .m_current_tile_part_number
          .wrapping_add(1);
        tilepartno += 1;
      }
      pino += 1;
    }
    *p_data_written = l_nb_bytes_written;
    1i32
  }
}

/* *
 * Writes the updated tlm.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_write_updated_tlm(
  mut p_j2k: &mut opj_j2k,
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_tlm_size: OPJ_UINT32 = 0;
    let mut l_tlm_position: OPJ_OFF_T = 0;
    let mut l_current_position: OPJ_OFF_T = 0;
    let mut size_per_tile_part: OPJ_UINT32 = 0;
    /* preconditions */

    size_per_tile_part = if p_j2k.m_specific_param.m_encoder.m_Ttlmi_is_byte != 0 {
      5i32
    } else {
      6i32
    } as OPJ_UINT32;
    l_tlm_size =
      size_per_tile_part.wrapping_mul(p_j2k.m_specific_param.m_encoder.m_total_tile_parts);
    l_tlm_position = 6i64 + p_j2k.m_specific_param.m_encoder.m_tlm_start;
    l_current_position = opj_stream_tell(p_stream);
    if opj_stream_seek(p_stream, l_tlm_position, p_manager) == 0 {
      return 0i32;
    }
    if opj_stream_write_data(
      p_stream,
      p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_buffer,
      l_tlm_size as OPJ_SIZE_T,
      p_manager,
    ) != l_tlm_size as usize
    {
      return 0i32;
    }
    if opj_stream_seek(p_stream, l_current_position, p_manager) == 0 {
      return 0i32;
    }
    1i32
  }
}

/* *
 * Ends the encoding, i.e. frees memory.
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_end_encoding(
  mut p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* preconditions */
  unsafe {
    if !p_j2k
      .m_specific_param
      .m_encoder
      .m_tlm_sot_offsets_buffer
      .is_null()
    {
      opj_free(p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_buffer as *mut core::ffi::c_void);
      p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_buffer = core::ptr::null_mut::<OPJ_BYTE>();
      p_j2k.m_specific_param.m_encoder.m_tlm_sot_offsets_current = core::ptr::null_mut::<OPJ_BYTE>()
    }
    if !p_j2k
      .m_specific_param
      .m_encoder
      .m_encoded_tile_data
      .is_null()
    {
      opj_free(p_j2k.m_specific_param.m_encoder.m_encoded_tile_data as *mut core::ffi::c_void);
      p_j2k.m_specific_param.m_encoder.m_encoded_tile_data = core::ptr::null_mut::<OPJ_BYTE>()
    }
    p_j2k.m_specific_param.m_encoder.m_encoded_tile_size = 0 as OPJ_UINT32;
    1i32
  }
}

/* *
 * Destroys the memory associated with the decoding of headers.
 */
/* *
 * Destroys the memory associated with the decoding of headers.
 */
fn opj_j2k_destroy_header_memory(
  mut p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* preconditions */

  unsafe {
    if !p_j2k
      .m_specific_param
      .m_encoder
      .m_header_tile_data
      .is_null()
    {
      opj_free(p_j2k.m_specific_param.m_encoder.m_header_tile_data as *mut core::ffi::c_void);
      p_j2k.m_specific_param.m_encoder.m_header_tile_data = core::ptr::null_mut::<OPJ_BYTE>()
    }
    p_j2k.m_specific_param.m_encoder.m_header_tile_data_size = 0 as OPJ_UINT32;
    1i32
  }
}

/* *
 * Inits the Info
 *
 * @param       p_stream                the stream to write data to.
 * @param       p_j2k                   J2K codec.
 * @param       p_manager               the user event manager.
*/
fn opj_j2k_init_info(
  mut p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* preconditions */

  unsafe {
    /* TODO mergeV2: check this part which use cstr_info */
    /*
    let mut l_cstr_info = 0 as *mut opj_codestream_info_t;
    l_cstr_info = p_j2k->cstr_info;

    if (l_cstr_info)  {
            OPJ_UINT32 compno;
            l_cstr_info->tile = (opj_tile_info_t *) opj_malloc(p_j2k->m_cp.tw * p_j2k->m_cp.th * sizeof(opj_tile_info_t));

            l_cstr_info->image_w = p_j2k->m_image->x1 - p_j2k->m_image->x0;
            l_cstr_info->image_h = p_j2k->m_image->y1 - p_j2k->m_image->y0;

            l_cstr_info->prog = (&p_j2k->m_cp.tcps[0])->prg;

            l_cstr_info->tw = p_j2k->m_cp.tw;
            l_cstr_info->th = p_j2k->m_cp.th;

            l_cstr_info->tile_x = p_j2k->m_cp.tdx;*/
    /* new version parser */
    /*l_cstr_info->tile_y = p_j2k->m_cp.tdy;*/
    /* new version parser */
    /*l_cstr_info->tile_Ox = p_j2k->m_cp.tx0;*/
    /* new version parser */
    /*l_cstr_info->tile_Oy = p_j2k->m_cp.ty0;*/
    /* new version parser */
    /*l_cstr_info->numcomps = p_j2k->m_image->numcomps;

    l_cstr_info->numlayers = (&p_j2k->m_cp.tcps[0])->numlayers;

    l_cstr_info->numdecompos = (OPJ_INT32*) opj_malloc(p_j2k->m_image->numcomps * sizeof(OPJ_INT32));

    for (compno=0; compno < p_j2k->m_image->numcomps; compno++) {
            l_cstr_info->numdecompos[compno] = (&p_j2k->m_cp.tcps[0])->tccps->numresolutions - 1;
    }

    l_cstr_info->D_max = 0.0;       */
    /* ADD Marcela */
    /*l_cstr_info->main_head_start = opj_stream_tell(p_stream);*/
    /* position of SOC */
    /*l_cstr_info->maxmarknum = 100;
    l_cstr_info->marker = (opj_marker_info_t *) opj_malloc(l_cstr_info->maxmarknum * sizeof(opj_marker_info_t));
    l_cstr_info->marknum = 0;
    }*/
    opj_j2k_calculate_tp(
      &mut p_j2k.m_cp,
      &mut p_j2k.m_specific_param.m_encoder.m_total_tile_parts,
      &mut *p_j2k.m_private_image,
      p_manager,
    )
  }
}

/* *
 * Creates a tile-coder encoder.
 *
 * @param       p_stream                        the stream to write data to.
 * @param       p_j2k                           J2K codec.
 * @param       p_manager                   the user event manager.
*/
fn opj_j2k_create_tcd(
  mut p_j2k: &mut opj_j2k,
  mut _p_stream: &mut Stream,
  mut _p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  /* preconditions */

  if opj_tcd_init(&mut p_j2k.m_tcd, p_j2k.m_private_image, &mut p_j2k.m_cp) == 0 {
    return 0i32;
  }
  1i32
}

pub(crate) fn opj_j2k_write_tile(
  mut p_j2k: &mut opj_j2k,
  mut p_tile_index: OPJ_UINT32,
  mut p_data: &[u8],
  mut p_stream: &mut Stream,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    if opj_j2k_pre_write_tile(p_j2k, p_tile_index, p_stream, p_manager) == 0 {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Error while opj_j2k_pre_write_tile with tile index = %d\n",
        p_tile_index,
      );
      return 0i32;
    } else {
      let mut j: OPJ_UINT32 = 0;
      /* Allocate data */
      j = 0 as OPJ_UINT32;
      while j < (*p_j2k.m_tcd.image).numcomps {
        let mut l_tilec = p_j2k.m_tcd.tcd_image.tiles.comps.offset(j as isize);
        if opj_alloc_tile_component_data(l_tilec) == 0 {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Error allocating tile component data.",
          );
          return 0i32;
        }
        j += 1;
      }
      /* now copy data into the tile component */
      if opj_tcd_copy_tile_data(&mut p_j2k.m_tcd, p_data) == 0 {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Size mismatch between tile data and sent data.",
        );
        return 0i32;
      }
      if opj_j2k_post_write_tile(p_j2k, p_stream, p_manager) == 0 {
        event_msg!(
          p_manager,
          EVT_ERROR,
          "Error while opj_j2k_post_write_tile with tile index = %d\n",
          p_tile_index,
        );
        return 0i32;
      }
    }
    1i32
  }
}

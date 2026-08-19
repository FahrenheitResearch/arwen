pub mod cio {
  pub const OPJ_STREAM_STATUS_OUTPUT: u32 = 0x1u32;
  pub const OPJ_STREAM_STATUS_INPUT: u32 = 0x2u32;
  pub const OPJ_STREAM_STATUS_END: u32 = 0x4u32;
  pub const OPJ_STREAM_STATUS_ERROR: u32 = 0x8u32;
}

pub mod event {
  #[derive(Copy, Clone)]
  pub enum EventType {
    Error = 1,
    Warning = 2,
    Info = 4,
  }

  impl EventType {
    pub fn from_i32(v: i32) -> Option<Self> {
      match v {
        1 => Some(EventType::Error),
        2 => Some(EventType::Warning),
        4 => Some(EventType::Info),
        _ => None,
      }
    }
  }

  /**< Error event type */
  pub const EVT_ERROR: EventType = EventType::Error;
  /**< Warning event type */
  pub const EVT_WARNING: EventType = EventType::Warning;
  /**< Debug event type */
  pub const EVT_INFO: EventType = EventType::Info;
}

/// Enum representing JPIP box types.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum JpipBoxType {
  #[default]
  None,
  /// Codestream index
  Cidx,
  /// Codestream Finder Box
  Cptr,
  /// Manifest Box
  Manf,
  /// Fragment array Index box
  Faix,
  /// Main Header Index Table
  Mhix,
  /// Tile-part Index Table box
  Tpix,
  /// Tile header Index Table box
  Thix,
  /// Precinct Packet Index Table box
  Ppix,
  /// Packet Header index Table
  Phix,
  /// File Index
  Fidx,
  /// File Finder
  Fptr,
  /// Proxy boxes
  Prxy,
  /// Index finder box
  Iptr,
  /// Place holder
  Phld,
  /// Unknown box type
  Unknown(u32),
}

impl From<u32> for JpipBoxType {
  fn from(value: u32) -> Self {
    match value {
      0x63696478 => JpipBoxType::Cidx,
      0x63707472 => JpipBoxType::Cptr,
      0x6d616e66 => JpipBoxType::Manf,
      0x66616978 => JpipBoxType::Faix,
      0x6d686978 => JpipBoxType::Mhix,
      0x74706978 => JpipBoxType::Tpix,
      0x74686978 => JpipBoxType::Thix,
      0x70706978 => JpipBoxType::Ppix,
      0x70686978 => JpipBoxType::Phix,
      0x66696478 => JpipBoxType::Fidx,
      0x66707472 => JpipBoxType::Fptr,
      0x70727879 => JpipBoxType::Prxy,
      0x69707472 => JpipBoxType::Iptr,
      0x70686c64 => JpipBoxType::Phld,
      _ => JpipBoxType::Unknown(value),
    }
  }
}

impl JpipBoxType {
  pub fn to_u32(&self) -> u32 {
    match self {
      JpipBoxType::None => 0x00000000,
      JpipBoxType::Cidx => 0x63696478,
      JpipBoxType::Cptr => 0x63707472,
      JpipBoxType::Manf => 0x6d616e66,
      JpipBoxType::Faix => 0x66616978,
      JpipBoxType::Mhix => 0x6d686978,
      JpipBoxType::Tpix => 0x74706978,
      JpipBoxType::Thix => 0x74686978,
      JpipBoxType::Ppix => 0x70706978,
      JpipBoxType::Phix => 0x70686978,
      JpipBoxType::Fidx => 0x66696478,
      JpipBoxType::Fptr => 0x66707472,
      JpipBoxType::Prxy => 0x70727879,
      JpipBoxType::Iptr => 0x69707472,
      JpipBoxType::Phld => 0x70686c64,
      JpipBoxType::Unknown(value) => *value,
    }
  }
}

pub mod j2k {
  pub const J2K_CP_CSTY_PRT: u32 = 0x01;
  pub const J2K_CP_CSTY_SOP: u32 = 0x02;
  pub const J2K_CP_CSTY_EPH: u32 = 0x04;
  pub const J2K_CCP_CSTY_PRT: u32 = 0x01;
  /**< Selective arithmetic coding bypass */
  pub const J2K_CCP_CBLKSTY_LAZY: u32 = 0x01;
  /**< Reset context probabilities on coding pass boundaries */
  pub const J2K_CCP_CBLKSTY_RESET: u32 = 0x02;
  /**< Termination on each coding pass */
  pub const J2K_CCP_CBLKSTY_TERMALL: u32 = 0x04;
  /**< Vertically stripe causal context */
  pub const J2K_CCP_CBLKSTY_VSC: u32 = 0x08;
  /**< Predictable termination */
  pub const J2K_CCP_CBLKSTY_PTERM: u32 = 0x10;
  /**< Segmentation symbols are used */
  pub const J2K_CCP_CBLKSTY_SEGSYM: u32 = 0x20;
  /**< (high throughput) HT codeblocks */
  pub const J2K_CCP_CBLKSTY_HT: u32 = 0x40;
  /**< MIXED mode HT codeblocks */
  pub const J2K_CCP_CBLKSTY_HTMIXED: u32 = 0x80;
  pub const J2K_CCP_QNTSTY_NOQNT: u32 = 0;
  pub const J2K_CCP_QNTSTY_SIQNT: u32 = 1;
  pub const J2K_CCP_QNTSTY_SEQNT: u32 = 2;
  /**< SOC marker value */
  pub const J2K_MS_SOC: u32 = 0xff4f;
  /**< SOT marker value */
  pub const J2K_MS_SOT: u32 = 0xff90;
  /**< SOD marker value */
  pub const J2K_MS_SOD: u32 = 0xff93;
  /**< EOC marker value */
  pub const J2K_MS_EOC: u32 = 0xffd9;
  /**< CAP marker value */
  pub const J2K_MS_CAP: u32 = 0xff50;
  /**< SIZ marker value */
  pub const J2K_MS_SIZ: u32 = 0xff51;
  /**< COD marker value */
  pub const J2K_MS_COD: u32 = 0xff52;
  /**< COC marker value */
  pub const J2K_MS_COC: u32 = 0xff53;
  /**< CPF marker value */
  pub const J2K_MS_CPF: u32 = 0xff59;
  /**< RGN marker value */
  pub const J2K_MS_RGN: u32 = 0xff5e;
  /**< QCD marker value */
  pub const J2K_MS_QCD: u32 = 0xff5c;
  /**< QCC marker value */
  pub const J2K_MS_QCC: u32 = 0xff5d;
  /**< POC marker value */
  pub const J2K_MS_POC: u32 = 0xff5f;
  /**< TLM marker value */
  pub const J2K_MS_TLM: u32 = 0xff55;
  /**< PLM marker value */
  pub const J2K_MS_PLM: u32 = 0xff57;
  /**< PLT marker value */
  pub const J2K_MS_PLT: u32 = 0xff58;
  /**< PPM marker value */
  pub const J2K_MS_PPM: u32 = 0xff60;
  /**< PPT marker value */
  pub const J2K_MS_PPT: u32 = 0xff61;
  /**< SOP marker value */
  pub const J2K_MS_SOP: u32 = 0xff91;
  /**< EPH marker value */
  pub const J2K_MS_EPH: u32 = 0xff92;
  /**< CRG marker value */
  pub const J2K_MS_CRG: u32 = 0xff63;
  /**< COM marker value */
  pub const J2K_MS_COM: u32 = 0xff64;
  /**< CBD marker value */
  pub const J2K_MS_CBD: u32 = 0xff78;
  /**< MCC marker value */
  pub const J2K_MS_MCC: u32 = 0xff75;
  /**< MCT marker value */
  pub const J2K_MS_MCT: u32 = 0xff74;
  /**< MCO marker value */
  pub const J2K_MS_MCO: u32 = 0xff77;
  /**< UNKNOWN marker value */
  pub const J2K_MS_UNK: u32 = 0;
  /**< EPC marker value (Part 11: JPEG 2000 for Wireless) */
  pub const J2K_MS_EPC: u32 = 0xff68;
  /**< EPB marker value (Part 11: JPEG 2000 for Wireless) */
  pub const J2K_MS_EPB: u32 = 0xff66;
  /**< ESD marker value (Part 11: JPEG 2000 for Wireless) */
  pub const J2K_MS_ESD: u32 = 0xff67;
  /**< RED marker value (Part 11: JPEG 2000 for Wireless) */
  pub const J2K_MS_RED: u32 = 0xff69;
  /**< SEC marker value (Part 8: Secure JPEG 2000) */
  pub const J2K_MS_SEC: u32 = 0xff65;
  /**< INSEC marker value (Part 8: Secure JPEG 2000) */
  pub const J2K_MS_INSEC: u32 = 0xff94;
  /**< Maximum number of POCs */
  pub const J2K_MAX_POCS: u32 = 32;

  pub const J2K_TCD_MATRIX_MAX_LAYER_COUNT: i32 = 10;
  pub const J2K_TCD_MATRIX_MAX_RESOLUTION_COUNT: i32 = 10;
}

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub enum Jp2BoxType {
  /// No box type
  #[default]
  None,
  /// JPEG 2000 signature box
  JPIP,
  /// File type box
  JP,
  /// JP2 header box (super-box)
  FTYP,
  /// Image header box
  JP2H,
  /// Colour specification box
  IHDR,
  /// Contiguous codestream box
  COLR,
  /// Data entry URL box
  JP2C,
  /// Palette box
  URL,
  /// Component Mapping box
  PCLR,
  /// Channel Definition box
  CMAP,
  /// Data Reference box
  CDEF,
  /// Bits per component box
  DTBL,
  /// File type fields
  BPCC,
  /// Resolution box (super-box)
  JP2,
  /// Intellectual property box
  RES,
  /// XML box
  JP2I,
  /// UUID box
  XML,
  /// UUID info box (super-box)
  UUID,
  /// UUID list box
  UINF,
  /// Unknown box type
  Unknown(u32),
}

impl From<u32> for Jp2BoxType {
  fn from(value: u32) -> Self {
    match value {
      0x6a706970 => Jp2BoxType::JPIP,
      0x6a502020 => Jp2BoxType::JP,
      0x66747970 => Jp2BoxType::FTYP,
      0x6a703268 => Jp2BoxType::JP2H,
      0x69686472 => Jp2BoxType::IHDR,
      0x636f6c72 => Jp2BoxType::COLR,
      0x6a703263 => Jp2BoxType::JP2C,
      0x75726c20 => Jp2BoxType::URL,
      0x70636c72 => Jp2BoxType::PCLR,
      0x636d6170 => Jp2BoxType::CMAP,
      0x63646566 => Jp2BoxType::CDEF,
      0x6474626c => Jp2BoxType::DTBL,
      0x62706363 => Jp2BoxType::BPCC,
      0x6a703220 => Jp2BoxType::JP2,
      0x72657320 => Jp2BoxType::RES,
      0x6a703269 => Jp2BoxType::JP2I,
      0x786d6c20 => Jp2BoxType::XML,
      0x75756994 => Jp2BoxType::UUID,
      0x75696e66 => Jp2BoxType::UINF,
      _ => Jp2BoxType::Unknown(value),
    }
  }
}

impl Jp2BoxType {
  pub fn is_file_header(&self) -> bool {
    match self {
      Jp2BoxType::JP => true,
      Jp2BoxType::FTYP => true,
      Jp2BoxType::JP2H => true,
      _ => false,
    }
  }

  pub fn is_image_header(&self) -> bool {
    match self {
      Jp2BoxType::IHDR => true,
      Jp2BoxType::COLR => true,
      Jp2BoxType::BPCC => true,
      Jp2BoxType::PCLR => true,
      Jp2BoxType::CMAP => true,
      Jp2BoxType::CDEF => true,
      _ => false,
    }
  }

  /// Converts the enum variant back into a u32 value.
  pub fn to_u32(&self) -> Option<u32> {
    match self {
      Jp2BoxType::None => None,
      Jp2BoxType::JPIP => Some(0x6a706970),
      Jp2BoxType::JP => Some(0x6a502020),
      Jp2BoxType::FTYP => Some(0x66747970),
      Jp2BoxType::JP2H => Some(0x6a703268),
      Jp2BoxType::IHDR => Some(0x69686472),
      Jp2BoxType::COLR => Some(0x636f6c72),
      Jp2BoxType::JP2C => Some(0x6a703263),
      Jp2BoxType::URL => Some(0x75726c20),
      Jp2BoxType::PCLR => Some(0x70636c72),
      Jp2BoxType::CMAP => Some(0x636d6170),
      Jp2BoxType::CDEF => Some(0x63646566),
      Jp2BoxType::DTBL => Some(0x6474626c),
      Jp2BoxType::BPCC => Some(0x62706363),
      Jp2BoxType::JP2 => Some(0x6a703220),
      Jp2BoxType::RES => Some(0x72657320),
      Jp2BoxType::JP2I => Some(0x6a703269),
      Jp2BoxType::XML => Some(0x786d6c20),
      Jp2BoxType::UUID => Some(0x75756994),
      Jp2BoxType::UINF => Some(0x75696e66),
      Jp2BoxType::Unknown(value) => Some(*value),
    }
  }
}

pub mod mqc {
  pub const MQC_NUMCTXS: usize = 19;
  pub const BYPASS_CT_INIT: u32 = 0xDEADBEEF;
}

pub mod opj {
  pub const OPJ_TRUE: u32 = 1;
  pub const OPJ_FALSE: u32 = 0;
  /**< Maximum allowed size for filenames */
  pub const OPJ_PATH_LEN: u32 = 4096;
  /**< Number of maximum resolution level authorized */
  pub const OPJ_J2K_MAXRLVLS: u32 = 33;
  /**< Number of maximum sub-band linked to number of resolution level */
  pub const OPJ_J2K_MAXBANDS: u32 = 3 * OPJ_J2K_MAXRLVLS - 2;
  pub const OPJ_J2K_DEFAULT_NB_SEGS: u32 = 10;
  /** 1 mega by default */
  pub const OPJ_J2K_STREAM_CHUNK_SIZE: u32 = 0x100000;
  pub const OPJ_J2K_DEFAULT_HEADER_SIZE: u32 = 1000;
  pub const OPJ_J2K_MCC_DEFAULT_NB_RECORDS: u32 = 10;
  pub const OPJ_J2K_MCT_DEFAULT_NB_RECORDS: u32 = 10;
  /**< Maximum number of tile parts expected by JPWL: increase at your will */
  pub const JPWL_MAX_NO_TILESPECS: u32 = 16;
  /**< Maximum number of packet parts expected by JPWL: increase at your will */
  pub const JPWL_MAX_NO_PACKSPECS: u32 = 16;
  /**< Maximum number of JPWL markers: increase at your will */
  pub const JPWL_MAX_NO_MARKERS: u32 = 512;
  /**< index file name used when JPWL is on */
  pub const JPWL_PRIVATEINDEX_NAME: &str = "jpwl_index_privatefilename";
  /**< Expect this number of components, so you'll find better the first EPB */
  pub const JPWL_EXPECTED_COMPONENTS: u32 = 3;
  /**< Expect this maximum number of tiles, to avoid some crashes */
  pub const JPWL_MAXIMUM_TILES: u32 = 8192;
  /**< Expect this maximum number of bit errors in marker id's */
  pub const JPWL_MAXIMUM_HAMMING: u32 = 2;
  /**< Expect this maximum number of bytes for composition of EPBs */
  pub const JPWL_MAXIMUM_EPB_ROOM: u32 = 65450;
  /**< Basic image information provided to the user */
  pub const OPJ_IMG_INFO: u32 = 1;
  /**< Codestream information based only on the main header */
  pub const OPJ_J2K_MH_INFO: u32 = 2;
  /**< Tile information based on the current tile header */
  pub const OPJ_J2K_TH_INFO: u32 = 4;
  /**< Tile/Component information of all tiles */
  pub const OPJ_J2K_TCH_INFO: u32 = 8;
  /**< Codestream index based only on the main header */
  pub const OPJ_J2K_MH_IND: u32 = 16;
  /**< Tile index based on the current tile */
  pub const OPJ_J2K_TH_IND: u32 = 32;
  /*FIXME pub const OPJ_J2K_CSTR_IND: u32     = 48;*/
  /**< JP2 file information */
  pub const OPJ_JP2_INFO: u32 = 128;
  /**< JP2 file index */
  pub const OPJ_JP2_IND: u32 = 256;
  /** no profile, conform to 15444-1 */
  pub const OPJ_PROFILE_NONE: u32 = 0x0000;
  /** Profile 0 as described in 15444-1,Table A.45 */
  pub const OPJ_PROFILE_0: u32 = 0x0001;
  /** Profile 1 as described in 15444-1,Table A.45 */
  pub const OPJ_PROFILE_1: u32 = 0x0002;
  /** At least 1 extension defined in 15444-2 (Part-2) */
  pub const OPJ_PROFILE_PART2: u32 = 0x8000;
  /** 2K cinema profile defined in 15444-1 AMD1 */
  pub const OPJ_PROFILE_CINEMA_2K: u32 = 0x0003;
  /** 4K cinema profile defined in 15444-1 AMD1 */
  pub const OPJ_PROFILE_CINEMA_4K: u32 = 0x0004;
  /** Scalable 2K cinema profile defined in 15444-1 AMD2 */
  pub const OPJ_PROFILE_CINEMA_S2K: u32 = 0x0005;
  /** Scalable 4K cinema profile defined in 15444-1 AMD2 */
  pub const OPJ_PROFILE_CINEMA_S4K: u32 = 0x0006;
  /** Long term storage cinema profile defined in 15444-1 AMD2 */
  pub const OPJ_PROFILE_CINEMA_LTS: u32 = 0x0007;
  /** Single Tile Broadcast profile defined in 15444-1 AMD3 */
  pub const OPJ_PROFILE_BC_SINGLE: u32 = 0x0100;
  /** Multi Tile Broadcast profile defined in 15444-1 AMD3 */
  pub const OPJ_PROFILE_BC_MULTI: u32 = 0x0200;
  /** Multi Tile Reversible Broadcast profile defined in 15444-1 AMD3 */
  pub const OPJ_PROFILE_BC_MULTI_R: u32 = 0x0300;
  /** 2K Single Tile Lossy IMF profile defined in 15444-1 AMD 8 */
  pub const OPJ_PROFILE_IMF_2K: u32 = 0x0400;
  /** 4K Single Tile Lossy IMF profile defined in 15444-1 AMD 8 */
  pub const OPJ_PROFILE_IMF_4K: u32 = 0x0500;
  /** 8K Single Tile Lossy IMF profile defined in 15444-1 AMD 8 */
  pub const OPJ_PROFILE_IMF_8K: u32 = 0x0600;
  /** 2K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8 */
  pub const OPJ_PROFILE_IMF_2K_R: u32 = 0x0700;
  /** 4K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8 */
  pub const OPJ_PROFILE_IMF_4K_R: u32 = 0x0800;
  /** 8K Single/Multi Tile Reversible IMF profile defined in 15444-1 AMD 8 */
  pub const OPJ_PROFILE_IMF_8K_R: u32 = 0x0900;
  /** No Part-2 extension */
  pub const OPJ_EXTENSION_NONE: u32 = 0x0000;
  /** Custom MCT support */
  pub const OPJ_EXTENSION_MCT: u32 = 0x0100;

  pub fn opj_is_cinema(v: u32) -> bool {
    (OPJ_PROFILE_CINEMA_2K..=OPJ_PROFILE_CINEMA_S4K).contains(&v)
  }

  pub fn opj_is_storage(v: u32) -> bool {
    v == OPJ_PROFILE_CINEMA_LTS
  }

  pub fn opj_is_broadcast(v: u32) -> bool {
    (OPJ_PROFILE_BC_SINGLE..=(OPJ_PROFILE_BC_MULTI_R | 0x000b)).contains(&v)
  }

  pub fn opj_is_imf(v: u32) -> bool {
    (OPJ_PROFILE_IMF_2K..=(OPJ_PROFILE_IMF_8K_R | 0x009b)).contains(&v)
  }

  pub fn opj_is_part2(v: u32) -> bool {
    v & OPJ_PROFILE_PART2 != 0
  }

  /** Extract IMF profile without mainlevel/sublevel */
  pub fn opj_get_imf_profile(v: u32) -> u32 {
    v & 0xff00
  }

  /** Extract IMF main level */
  pub fn opj_get_imf_mainlevel(v: u32) -> u32 {
    v & 0xf
  }

  /** Extract IMF sub level */
  pub fn opj_get_imf_sublevel(v: u32) -> u32 {
    (v >> 4) & 0xf
  }

  /** Maximum main level */
  pub const OPJ_IMF_MAINLEVEL_MAX: u32 = 11;
  /** MSamples/sec for IMF main level 1 */
  pub const OPJ_IMF_MAINLEVEL_1_MSAMPLESEC: u32 = 65;
  /** MSamples/sec for IMF main level 2 */
  pub const OPJ_IMF_MAINLEVEL_2_MSAMPLESEC: u32 = 130;
  /** MSamples/sec for IMF main level 3 */
  pub const OPJ_IMF_MAINLEVEL_3_MSAMPLESEC: u32 = 195;
  /** MSamples/sec for IMF main level 4 */
  pub const OPJ_IMF_MAINLEVEL_4_MSAMPLESEC: u32 = 260;
  /** MSamples/sec for IMF main level 5 */
  pub const OPJ_IMF_MAINLEVEL_5_MSAMPLESEC: u32 = 520;
  /** MSamples/sec for IMF main level 6 */
  pub const OPJ_IMF_MAINLEVEL_6_MSAMPLESEC: u32 = 1200;
  /** MSamples/sec for IMF main level 7 */
  pub const OPJ_IMF_MAINLEVEL_7_MSAMPLESEC: u32 = 2400;
  /** MSamples/sec for IMF main level 8 */
  pub const OPJ_IMF_MAINLEVEL_8_MSAMPLESEC: u32 = 4800;
  /** MSamples/sec for IMF main level 9 */
  pub const OPJ_IMF_MAINLEVEL_9_MSAMPLESEC: u32 = 9600;
  /** MSamples/sec for IMF main level 10 */
  pub const OPJ_IMF_MAINLEVEL_10_MSAMPLESEC: u32 = 19200;
  /** MSamples/sec for IMF main level 11 */
  pub const OPJ_IMF_MAINLEVEL_11_MSAMPLESEC: u32 = 38400;
  /** Mbits/s for IMF sub level 1 */
  pub const OPJ_IMF_SUBLEVEL_1_MBITSSEC: u32 = 200;
  /** Mbits/s for IMF sub level 2 */
  pub const OPJ_IMF_SUBLEVEL_2_MBITSSEC: u32 = 400;
  /** Mbits/s for IMF sub level 3 */
  pub const OPJ_IMF_SUBLEVEL_3_MBITSSEC: u32 = 800;
  /** Mbits/s for IMF sub level 4 */
  pub const OPJ_IMF_SUBLEVEL_4_MBITSSEC: u32 = 1600;
  /** Mbits/s for IMF sub level 5 */
  pub const OPJ_IMF_SUBLEVEL_5_MBITSSEC: u32 = 3200;
  /** Mbits/s for IMF sub level 6 */
  pub const OPJ_IMF_SUBLEVEL_6_MBITSSEC: u32 = 6400;
  /** Mbits/s for IMF sub level 7 */
  pub const OPJ_IMF_SUBLEVEL_7_MBITSSEC: u32 = 12800;
  /** Mbits/s for IMF sub level 8 */
  pub const OPJ_IMF_SUBLEVEL_8_MBITSSEC: u32 = 25600;
  /** Mbits/s for IMF sub level 9 */
  pub const OPJ_IMF_SUBLEVEL_9_MBITSSEC: u32 = 51200;
  /** Maximum codestream length for 24fps */
  pub const OPJ_CINEMA_24_CS: u32 = 1302083;
  /** Maximum codestream length for 48fps */
  pub const OPJ_CINEMA_48_CS: u32 = 651041;
  /** Maximum size per color component for 2K & 4K @ 24fps */
  pub const OPJ_CINEMA_24_COMP: u32 = 1041666;
  /** Maximum size per color component for 2K @ 48fps */
  pub const OPJ_CINEMA_48_COMP: u32 = 520833;
  pub const OPJ_DPARAMETERS_IGNORE_PCLR_CMAP_CDEF_FLAG: u32 = 0x0001;
  pub const OPJ_DPARAMETERS_DUMP_FLAG: u32 = 0x0002;
  pub const OPJ_STREAM_READ: u32 = OPJ_TRUE;
  pub const OPJ_STREAM_WRITE: u32 = OPJ_FALSE;
}

pub mod common {
  /**< Margin for a fake FFFF marker */
  pub const OPJ_COMMON_CBLK_DATA_EXTRA: u32 = 2;
  pub const OPJ_COMP_PARAM_DEFAULT_CBLOCKW: u32 = 64;
  pub const OPJ_COMP_PARAM_DEFAULT_CBLOCKH: u32 = 64;
  pub const OPJ_COMP_PARAM_DEFAULT_PROG_ORDER: i32 = crate::openjpeg::OPJ_LRCP;
  pub const OPJ_COMP_PARAM_DEFAULT_NUMRESOLUTION: u32 = 6;
}

pub mod t1 {
  pub const T1_NMSEDEC_BITS: u32 = 7;
  pub const T1_NUMCTXS_ZC: u8 = 9;
  pub const T1_NUMCTXS_SC: u8 = 5;
  pub const T1_NUMCTXS_MAG: u8 = 3;
  pub const T1_NUMCTXS_AGG: u8 = 1;
  pub const T1_NUMCTXS_UNI: u8 = 1;
  pub const T1_CTXNO_ZC: u8 = 0;
  pub const T1_CTXNO_SC: u8 = T1_CTXNO_ZC + T1_NUMCTXS_ZC;
  pub const T1_CTXNO_MAG: u8 = T1_CTXNO_SC + T1_NUMCTXS_SC;
  pub const T1_CTXNO_AGG: u8 = T1_CTXNO_MAG + T1_NUMCTXS_MAG;
  pub const T1_CTXNO_UNI: u8 = T1_CTXNO_AGG + T1_NUMCTXS_AGG;
  pub const T1_NUMCTXS: u8 = T1_CTXNO_UNI + T1_NUMCTXS_UNI;
  pub const T1_NMSEDEC_FRACBITS: i32 = (T1_NMSEDEC_BITS - 1) as i32;
  /**< Normal coding using entropy coder */
  pub const T1_TYPE_MQ: u8 = 0;
  /**< No encoding the information is store under raw format in codestream (mode switch RAW)*/
  pub const T1_TYPE_RAW: u8 = 1;
  pub const T1_SIGMA_0: u32 = 1u32 << 0;
  pub const T1_SIGMA_1: u32 = 1u32 << 1;
  pub const T1_SIGMA_2: u32 = 1u32 << 2;
  pub const T1_SIGMA_3: u32 = 1u32 << 3;
  pub const T1_SIGMA_4: u32 = 1u32 << 4;
  pub const T1_SIGMA_5: u32 = 1u32 << 5;
  pub const T1_SIGMA_6: u32 = 1u32 << 6;
  pub const T1_SIGMA_7: u32 = 1u32 << 7;
  pub const T1_SIGMA_8: u32 = 1u32 << 8;
  pub const T1_SIGMA_9: u32 = 1u32 << 9;
  pub const T1_SIGMA_10: u32 = 1u32 << 10;
  pub const T1_SIGMA_11: u32 = 1u32 << 11;
  pub const T1_SIGMA_12: u32 = 1u32 << 12;
  pub const T1_SIGMA_13: u32 = 1u32 << 13;
  pub const T1_SIGMA_14: u32 = 1u32 << 14;
  pub const T1_SIGMA_15: u32 = 1u32 << 15;
  pub const T1_SIGMA_16: u32 = 1u32 << 16;
  pub const T1_SIGMA_17: u32 = 1u32 << 17;
  pub const T1_CHI_0: u32 = 1u32 << 18;
  pub const T1_CHI_0_I: u32 = 18;
  pub const T1_CHI_1: u32 = 1u32 << 19;
  pub const T1_CHI_1_I: u32 = 19;
  pub const T1_MU_0: u32 = 1u32 << 20;
  pub const T1_PI_0: u32 = 1u32 << 21;
  pub const T1_CHI_2: u32 = 1u32 << 22;
  pub const T1_CHI_2_I: u32 = 22;
  pub const T1_MU_1: u32 = 1u32 << 23;
  pub const T1_PI_1: u32 = 1u32 << 24;
  pub const T1_CHI_3: u32 = 1u32 << 25;
  pub const T1_MU_2: u32 = 1u32 << 26;
  pub const T1_PI_2: u32 = 1u32 << 27;
  pub const T1_CHI_4: u32 = 1u32 << 28;
  pub const T1_MU_3: u32 = 1u32 << 29;
  pub const T1_PI_3: u32 = 1u32 << 30;
  pub const T1_CHI_5: u32 = 1u32 << 31;
  pub const T1_CHI_5_I: u32 = 31;
  pub const T1_SIGMA_NW: u32 = T1_SIGMA_0;
  pub const T1_SIGMA_N: u32 = T1_SIGMA_1;
  pub const T1_SIGMA_NE: u32 = T1_SIGMA_2;
  pub const T1_SIGMA_W: u32 = T1_SIGMA_3;
  pub const T1_SIGMA_THIS: u32 = T1_SIGMA_4;
  pub const T1_SIGMA_E: u32 = T1_SIGMA_5;
  pub const T1_SIGMA_SW: u32 = T1_SIGMA_6;
  pub const T1_SIGMA_S: u32 = T1_SIGMA_7;
  pub const T1_SIGMA_SE: u32 = T1_SIGMA_8;
  pub const T1_SIGMA_NEIGHBOURS: u32 = T1_SIGMA_NW
    | T1_SIGMA_N
    | T1_SIGMA_NE
    | T1_SIGMA_W
    | T1_SIGMA_E
    | T1_SIGMA_SW
    | T1_SIGMA_S
    | T1_SIGMA_SE;
  pub const T1_CHI_THIS: u32 = T1_CHI_1;
  pub const T1_CHI_THIS_I: u32 = T1_CHI_1_I;
  pub const T1_MU_THIS: u32 = T1_MU_0;
  pub const T1_PI_THIS: u32 = T1_PI_0;
  pub const T1_CHI_S: u32 = T1_CHI_2;
  pub const T1_LUT_SGN_W: u32 = 1u32 << 0;
  pub const T1_LUT_SIG_N: u32 = 1u32 << 1;
  pub const T1_LUT_SGN_E: u32 = 1u32 << 2;
  pub const T1_LUT_SIG_W: u32 = 1u32 << 3;
  pub const T1_LUT_SGN_N: u32 = 1u32 << 4;
  pub const T1_LUT_SIG_E: u32 = 1u32 << 5;
  pub const T1_LUT_SGN_S: u32 = 1u32 << 6;
  pub const T1_LUT_SIG_S: u32 = 1u32 << 7;
}

pub use common::*;
pub use event::*;
pub use j2k::*;
pub use mqc::*;
pub use opj::*;
pub use t1::*;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_jpip_box_type_roundtrip() {
    let values = [
      JpipBoxType::Cidx,
      JpipBoxType::Cptr,
      JpipBoxType::Manf,
      JpipBoxType::Faix,
      JpipBoxType::Mhix,
      JpipBoxType::Tpix,
      JpipBoxType::Thix,
      JpipBoxType::Ppix,
      JpipBoxType::Phix,
      JpipBoxType::Fidx,
      JpipBoxType::Fptr,
      JpipBoxType::Prxy,
      JpipBoxType::Iptr,
      JpipBoxType::Phld,
      JpipBoxType::Unknown(123),
    ];

    for value in values.iter() {
      let u32_value = value.to_u32();
      let converted_value = JpipBoxType::from(u32_value);
      assert_eq!(*value, converted_value);
    }
  }

  #[test]
  fn test_jp2_box_type_roundtrip() {
    let variants = [
      Jp2BoxType::JPIP,
      Jp2BoxType::JP,
      Jp2BoxType::FTYP,
      Jp2BoxType::JP2H,
      Jp2BoxType::IHDR,
      Jp2BoxType::COLR,
      Jp2BoxType::JP2C,
      Jp2BoxType::URL,
      Jp2BoxType::PCLR,
      Jp2BoxType::CMAP,
      Jp2BoxType::CDEF,
      Jp2BoxType::DTBL,
      Jp2BoxType::BPCC,
      Jp2BoxType::JP2,
      Jp2BoxType::RES,
      Jp2BoxType::JP2I,
      Jp2BoxType::XML,
      Jp2BoxType::UUID,
      Jp2BoxType::UINF,
      Jp2BoxType::Unknown(123),
    ];

    for variant in &variants {
      let value = variant.to_u32().unwrap();
      let roundtrip_variant = Jp2BoxType::from(value);
      assert_eq!(*variant, roundtrip_variant);
    }
  }
}

use super::bio::*;
use super::event::*;
use super::math::*;
use super::openjpeg::*;
use super::pi::*;
use super::tcd::*;
use super::tgt::*;

use super::malloc::*;

extern "C" {
  fn memset(_: *mut core::ffi::c_void, _: core::ffi::c_int, _: usize) -> *mut core::ffi::c_void;

  fn memcpy(
    _: *mut core::ffi::c_void,
    _: *const core::ffi::c_void,
    _: usize,
  ) -> *mut core::ffi::c_void;
}

#[repr(C)]
#[derive(Copy, Clone)]
pub(crate) struct opj_t2 {
  pub image: *mut opj_image_t,
  pub cp: *mut opj_cp_t,
}
pub(crate) type opj_t2_t = opj_t2;
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

/* * @defgroup T2 T2 - Implementation of a tier-2 coding */

/* ----------------------------------------------------------------------- */
fn opj_t2_putcommacode(mut bio: *mut opj_bio_t, mut n: OPJ_INT32) {
  for _ in 0..n {
    opj_bio_putbit(bio, 1);
  }
  opj_bio_putbit(bio, 0);
}

fn opj_t2_getcommacode(mut bio: *mut opj_bio_t) -> OPJ_UINT32 {
  let mut n = 0 as OPJ_UINT32;
  while opj_bio_read(bio, 1 as OPJ_UINT32) != 0 {
    n += 1;
  }
  n
}

/* *
Variable length code for signalling delta Zil (truncation point)
@param bio  Bit Input/Output component
@param n    delta Zil
*/
fn opj_t2_putnumpasses(mut bio: *mut opj_bio_t, mut n: OPJ_UINT32) {
  if n == 1u32 {
    opj_bio_putbit(bio, 0);
  } else if n == 2u32 {
    opj_bio_write(bio, 2 as OPJ_UINT32, 2 as OPJ_UINT32);
  } else if n <= 5u32 {
    opj_bio_write(bio, 0xcu32 | n.wrapping_sub(3u32), 4 as OPJ_UINT32);
  } else if n <= 36u32 {
    opj_bio_write(bio, 0x1e0u32 | n.wrapping_sub(6u32), 9 as OPJ_UINT32);
  } else if n <= 164u32 {
    opj_bio_write(bio, 0xff80u32 | n.wrapping_sub(37u32), 16 as OPJ_UINT32);
  };
}

fn opj_t2_getnumpasses(mut bio: *mut opj_bio_t) -> OPJ_UINT32 {
  let mut n: OPJ_UINT32 = 0;
  if opj_bio_read(bio, 1 as OPJ_UINT32) == 0 {
    return 1 as OPJ_UINT32;
  }
  if opj_bio_read(bio, 1 as OPJ_UINT32) == 0 {
    return 2 as OPJ_UINT32;
  }
  n = opj_bio_read(bio, 2 as OPJ_UINT32);
  if n != 3u32 {
    return (3u32).wrapping_add(n);
  }
  n = opj_bio_read(bio, 5 as OPJ_UINT32);
  if n != 31u32 {
    return (6u32).wrapping_add(n);
  }
  (37u32).wrapping_add(opj_bio_read(bio, 7 as OPJ_UINT32))
}

/* ----------------------------------------------------------------------- */

pub(crate) fn opj_t2_encode_packets(
  mut p_t2: *mut opj_t2_t,
  mut p_tile_no: OPJ_UINT32,
  mut p_tile: *mut opj_tcd_tile_t,
  mut p_maxlayers: OPJ_UINT32,
  mut p_dest: *mut OPJ_BYTE,
  mut p_data_written: *mut OPJ_UINT32,
  mut p_max_len: OPJ_UINT32,
  mut cstr_info: *mut opj_codestream_info_t,
  mut p_marker_info: *mut opj_tcd_marker_info_t,
  mut p_tp_num: OPJ_UINT32,
  mut p_tp_pos: OPJ_INT32,
  mut p_pino: OPJ_UINT32,
  mut p_t2_mode: J2K_T2_MODE,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_current_data = p_dest; /* t2_mode == FINAL_PASS  */
    let mut l_nb_bytes = 0 as OPJ_UINT32;
    let mut compno: OPJ_UINT32 = 0;
    let mut poc: OPJ_UINT32 = 0;
    let mut l_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut l_current_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut l_image = (*p_t2).image;
    let mut l_cp = (*p_t2).cp;
    let mut l_tcp: *mut opj_tcp_t = &mut *(*l_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    let mut pocno = if (*l_cp).rsiz as core::ffi::c_int == 0x4i32 {
      2i32
    } else {
      1i32
    } as OPJ_UINT32;
    let mut l_max_comp = if (*l_cp).m_specific_param.m_enc.m_max_comp_size > 0u32 {
      (*l_image).numcomps
    } else {
      1u32
    };
    let mut l_nb_pocs = (*l_tcp).numpocs.wrapping_add(1u32);
    l_pi = opj_pi_initialise_encode(l_image, l_cp, p_tile_no, p_t2_mode, p_manager);
    if l_pi.is_null() {
      return 0i32;
    }
    *p_data_written = 0 as OPJ_UINT32;
    if p_t2_mode as core::ffi::c_uint == THRESH_CALC as core::ffi::c_uint {
      /* Calculating threshold */
      l_current_pi = l_pi;
      compno = 0 as OPJ_UINT32;
      while compno < l_max_comp {
        let mut l_comp_len = 0 as OPJ_UINT32;
        l_current_pi = l_pi;
        poc = 0 as OPJ_UINT32;
        while poc < pocno {
          let mut l_tp_num = compno;
          /* TODO MSD : check why this function cannot fail (cf. v1) */
          opj_pi_create_encode(l_pi, l_cp, p_tile_no, poc, l_tp_num, p_tp_pos, p_t2_mode);
          if (*l_current_pi).poc.prg as core::ffi::c_int == OPJ_PROG_UNKNOWN as core::ffi::c_int {
            /* TODO ADE : add an error */
            opj_pi_destroy(l_pi, l_nb_pocs);
            return 0i32;
          }
          while opj_pi_next(l_current_pi) != 0 {
            if (*l_current_pi).layno < p_maxlayers {
              l_nb_bytes = 0 as OPJ_UINT32;
              if opj_t2_encode_packet(
                p_tile_no,
                p_tile,
                l_tcp,
                l_current_pi,
                l_current_data,
                &mut l_nb_bytes,
                p_max_len,
                cstr_info,
                p_t2_mode,
                p_manager,
              ) == 0
              {
                opj_pi_destroy(l_pi, l_nb_pocs);
                return 0i32;
              }
              l_comp_len = (l_comp_len as core::ffi::c_uint).wrapping_add(l_nb_bytes) as OPJ_UINT32;
              l_current_data = l_current_data.offset(l_nb_bytes as isize);
              p_max_len = (p_max_len as core::ffi::c_uint).wrapping_sub(l_nb_bytes) as OPJ_UINT32;
              *p_data_written =
                (*p_data_written as core::ffi::c_uint).wrapping_add(l_nb_bytes) as OPJ_UINT32
            }
          }
          if (*l_cp).m_specific_param.m_enc.m_max_comp_size != 0
            && l_comp_len > (*l_cp).m_specific_param.m_enc.m_max_comp_size
          {
            opj_pi_destroy(l_pi, l_nb_pocs);
            return 0i32;
          }
          l_current_pi = l_current_pi.offset(1);
          poc += 1;
        }
        compno += 1;
      }
    } else {
      opj_pi_create_encode(l_pi, l_cp, p_tile_no, p_pino, p_tp_num, p_tp_pos, p_t2_mode);
      l_current_pi = &mut *l_pi.offset(p_pino as isize) as *mut opj_pi_iterator_t;
      if (*l_current_pi).poc.prg as core::ffi::c_int == OPJ_PROG_UNKNOWN as core::ffi::c_int {
        /* TODO ADE : add an error */
        opj_pi_destroy(l_pi, l_nb_pocs);
        return 0i32;
      }
      if !p_marker_info.is_null() && (*p_marker_info).need_PLT != 0 {
        /* One time use intended */

        assert!((*p_marker_info).packet_count == 0u32);
        assert!((*p_marker_info).p_packet_size.is_null());
        (*p_marker_info).p_packet_size = opj_malloc(
          (opj_get_encoding_packet_count(l_image, l_cp, p_tile_no) as usize)
            .wrapping_mul(core::mem::size_of::<OPJ_UINT32>()),
        ) as *mut OPJ_UINT32;
        if (*p_marker_info).p_packet_size.is_null() {
          opj_pi_destroy(l_pi, l_nb_pocs);
          return 0i32;
        }
      }
      while opj_pi_next(l_current_pi) != 0 {
        if (*l_current_pi).layno < p_maxlayers {
          l_nb_bytes = 0 as OPJ_UINT32;
          if opj_t2_encode_packet(
            p_tile_no,
            p_tile,
            l_tcp,
            l_current_pi,
            l_current_data,
            &mut l_nb_bytes,
            p_max_len,
            cstr_info,
            p_t2_mode,
            p_manager,
          ) == 0
          {
            opj_pi_destroy(l_pi, l_nb_pocs);
            return 0i32;
          }
          l_current_data = l_current_data.offset(l_nb_bytes as isize);
          p_max_len = (p_max_len as core::ffi::c_uint).wrapping_sub(l_nb_bytes) as OPJ_UINT32;
          *p_data_written =
            (*p_data_written as core::ffi::c_uint).wrapping_add(l_nb_bytes) as OPJ_UINT32;
          if !p_marker_info.is_null() && (*p_marker_info).need_PLT != 0 {
            *(*p_marker_info)
              .p_packet_size
              .offset((*p_marker_info).packet_count as isize) = l_nb_bytes;
            (*p_marker_info).packet_count = (*p_marker_info).packet_count.wrapping_add(1)
          }
          /* INDEX >> */
          if !cstr_info.is_null() {
            if (*cstr_info).index_write != 0 {
              let mut info_TL: *mut opj_tile_info_t =
                &mut *(*cstr_info).tile.offset(p_tile_no as isize) as *mut opj_tile_info_t;
              let mut info_PK: *mut opj_packet_info_t =
                &mut *(*info_TL).packet.offset((*cstr_info).packno as isize)
                  as *mut opj_packet_info_t;
              if (*cstr_info).packno == 0 {
                (*info_PK).start_pos = ((*info_TL).end_header + 1i32) as OPJ_OFF_T
              } else {
                (*info_PK).start_pos = if ((*l_cp).m_specific_param.m_enc.m_tp_on || (*l_tcp).POC)
                  && (*info_PK).start_pos != 0
                {
                  (*info_PK).start_pos
                } else {
                  ((*(*info_TL)
                    .packet
                    .offset(((*cstr_info).packno - 1i32) as isize))
                  .end_pos)
                    + 1i64
                }
              }
              (*info_PK).end_pos = (*info_PK).start_pos + l_nb_bytes as i64 - 1;
              (*info_PK).end_ph_pos += (*info_PK).start_pos - 1i64
              /* End of packet header which now only represents the distance
              to start of packet is incremented by value of start of packet*/
            }
            (*cstr_info).packno += 1
          }
          /* << INDEX */
          (*p_tile).packno = (*p_tile).packno.wrapping_add(1)
        }
      }
    }
    opj_pi_destroy(l_pi, l_nb_pocs);
    1i32
  }
}

pub(crate) fn opj_t2_decode_packets(
  mut tcd: &mut opj_tcd,
  mut p_t2: *mut opj_t2_t,
  mut p_tile_no: OPJ_UINT32,
  mut p_src: *mut OPJ_BYTE,
  mut p_data_read: *mut OPJ_UINT32,
  mut p_max_len: OPJ_UINT32,
  mut _p_cstr_index: *mut opj_codestream_index_t,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_current_data = p_src;
    let mut l_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut pino: OPJ_UINT32 = 0;
    let mut l_image = (*p_t2).image;
    let mut l_cp = (*p_t2).cp;
    let mut l_tcp: *mut opj_tcp_t =
      &mut *(*(*p_t2).cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    let mut l_nb_bytes_read: OPJ_UINT32 = 0;
    let mut l_nb_pocs = (*l_tcp).numpocs.wrapping_add(1u32);
    let mut l_current_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut l_pack_info = core::ptr::null_mut::<opj_packet_info_t>();
    let mut l_img_comp = core::ptr::null_mut::<opj_image_comp_t>();
    /* create a packet iterator */
    l_pi = opj_pi_create_decode(l_image, l_cp, p_tile_no, p_manager);
    if l_pi.is_null() {
      return 0i32;
    }
    l_current_pi = l_pi;
    pino = 0 as OPJ_UINT32;
    while pino <= (*l_tcp).numpocs {
      /* if the resolution needed is too low, one dim of the tilec could be equal to zero
       * and no packets are used to decode this resolution and
       * l_current_pi->resno is always >= p_tile->comps[l_current_pi->compno].minimum_num_resolutions
       * and no l_img_comp->resno_decoded are computed
       */
      let mut first_pass_failed = core::ptr::null_mut::<OPJ_BOOL>();
      if (*l_current_pi).poc.prg as core::ffi::c_int == OPJ_PROG_UNKNOWN as core::ffi::c_int {
        /* TODO ADE : add an error */
        opj_pi_destroy(l_pi, l_nb_pocs);
        return 0i32;
      }
      first_pass_failed =
        opj_malloc(((*l_image).numcomps as usize).wrapping_mul(core::mem::size_of::<OPJ_BOOL>()))
          as *mut OPJ_BOOL;
      if first_pass_failed.is_null() {
        opj_pi_destroy(l_pi, l_nb_pocs);
        return 0i32;
      }
      memset(
        first_pass_failed as *mut core::ffi::c_void,
        1i32,
        ((*l_image).numcomps as usize).wrapping_mul(core::mem::size_of::<OPJ_BOOL>()),
      );
      while opj_pi_next(l_current_pi) != 0 {
        let mut skip_packet = 0i32;
        log::debug!(
          "packet offset=00000166 prg={} cmptno={:02} rlvlno={:02} prcno={:03} lyrno={:02}",
          (*l_current_pi).poc.prg1 as core::ffi::c_int,
          (*l_current_pi).compno,
          (*l_current_pi).resno,
          (*l_current_pi).precno,
          (*l_current_pi).layno,
        );
        /* INDEX >> */
        /* << INDEX */
        if (*l_current_pi).layno >= (*l_tcp).num_layers_to_decode {
          skip_packet = 1i32
        } else if (*l_current_pi).resno
          >= (*tcd
            .tcd_image
            .tiles
            .comps
            .offset((*l_current_pi).compno as isize))
          .minimum_num_resolutions
        {
          skip_packet = 1i32
        } else {
          /* If the packet layer is greater or equal than the maximum */
          /* number of layers, skip the packet */
          /* If the packet resolution number is greater than the minimum */
          /* number of resolution allowed, skip the packet */
          /* If no precincts of any band intersects the area of interest, */
          /* skip the packet */
          let mut bandno: OPJ_UINT32 = 0;
          let mut tilec: *mut opj_tcd_tilecomp_t = &mut *tcd
            .tcd_image
            .tiles
            .comps
            .offset((*l_current_pi).compno as isize)
            as *mut opj_tcd_tilecomp_t;
          let mut res: *mut opj_tcd_resolution_t =
            &mut *(*tilec).resolutions.offset((*l_current_pi).resno as isize)
              as *mut opj_tcd_resolution_t;
          skip_packet = 1i32;
          bandno = 0 as OPJ_UINT32;
          while bandno < (*res).numbands {
            let mut band: *mut opj_tcd_band_t =
              &mut *(*res).bands.as_mut_ptr().offset(bandno as isize) as *mut opj_tcd_band_t;
            let mut prec: *mut opj_tcd_precinct_t =
              &mut *(*band).precincts.offset((*l_current_pi).precno as isize)
                as *mut opj_tcd_precinct_t;
            if opj_tcd_is_subband_area_of_interest(
              tcd,
              (*l_current_pi).compno,
              (*l_current_pi).resno,
              (*band).bandno,
              (*prec).x0 as OPJ_UINT32,
              (*prec).y0 as OPJ_UINT32,
              (*prec).x1 as OPJ_UINT32,
              (*prec).y1 as OPJ_UINT32,
            ) != 0
            {
              skip_packet = 0i32;
              break;
            } else {
              bandno += 1;
            }
          }
        }
        if skip_packet == 0 {
          l_nb_bytes_read = 0 as OPJ_UINT32;
          *first_pass_failed.offset((*l_current_pi).compno as isize) = 0i32;
          if opj_t2_decode_packet(
            p_t2,
            &mut tcd.tcd_image.tiles,
            l_tcp,
            l_current_pi,
            l_current_data,
            &mut l_nb_bytes_read,
            p_max_len,
            l_pack_info,
            p_manager,
          ) == 0
          {
            opj_pi_destroy(l_pi, l_nb_pocs);
            opj_free(first_pass_failed as *mut core::ffi::c_void);
            return 0i32;
          }
          l_img_comp =
            &mut *(*l_image).comps.offset((*l_current_pi).compno as isize) as *mut opj_image_comp_t;
          (*l_img_comp).resno_decoded =
            opj_uint_max((*l_current_pi).resno, (*l_img_comp).resno_decoded)
        } else {
          l_nb_bytes_read = 0 as OPJ_UINT32;
          if opj_t2_skip_packet(
            p_t2,
            &mut tcd.tcd_image.tiles,
            l_tcp,
            l_current_pi,
            l_current_data,
            &mut l_nb_bytes_read,
            p_max_len,
            l_pack_info,
            p_manager,
          ) == 0
          {
            opj_pi_destroy(l_pi, l_nb_pocs);
            opj_free(first_pass_failed as *mut core::ffi::c_void);
            return 0i32;
          }
        }
        if *first_pass_failed.offset((*l_current_pi).compno as isize) != 0 {
          l_img_comp =
            &mut *(*l_image).comps.offset((*l_current_pi).compno as isize) as *mut opj_image_comp_t;
          if (*l_img_comp).resno_decoded == 0u32 {
            (*l_img_comp).resno_decoded = (*tcd
              .tcd_image
              .tiles
              .comps
              .offset((*l_current_pi).compno as isize))
            .minimum_num_resolutions
            .wrapping_sub(1u32)
          }
        }
        l_current_data = l_current_data.offset(l_nb_bytes_read as isize);
        p_max_len = (p_max_len as core::ffi::c_uint).wrapping_sub(l_nb_bytes_read) as OPJ_UINT32
      }
      l_current_pi = l_current_pi.offset(1);
      opj_free(first_pass_failed as *mut core::ffi::c_void);
      pino += 1;
    }
    /* INDEX >> */
    /* << INDEX */
    /* don't forget to release pi */
    opj_pi_destroy(l_pi, l_nb_pocs);
    *p_data_read = l_current_data.offset_from(p_src) as OPJ_UINT32;
    1i32
  }
}
/* ----------------------------------------------------------------------- */
/* *
 * Creates a Tier 2 handle
 *
 * @param       p_image         Source or destination image
 * @param       p_cp            Image coding parameters.
 * @return              a new T2 handle if successful, NULL otherwise.
*/

pub(crate) fn opj_t2_create(
  mut p_image: *mut opj_image_t,
  mut p_cp: *mut opj_cp_t,
) -> *mut opj_t2_t {
  unsafe {
    /* create the t2 structure */
    let mut l_t2 = opj_calloc(1i32 as size_t, core::mem::size_of::<opj_t2_t>()) as *mut opj_t2_t;
    if l_t2.is_null() {
      return core::ptr::null_mut::<opj_t2_t>();
    }
    (*l_t2).image = p_image;
    (*l_t2).cp = p_cp;
    l_t2
  }
}

pub(crate) fn opj_t2_destroy(mut t2: *mut opj_t2_t) {
  if !t2.is_null() {
    opj_free(t2 as *mut core::ffi::c_void);
  };
}
/* *
Decode a packet of a tile from a source buffer
@param t2 T2 handle
@param tile Tile for which to write the packets
@param tcp Tile coding parameters
@param pi Packet identity
@param src Source buffer
@param data_read   FIXME DOC
@param max_length  FIXME DOC
@param pack_info Packet information
@param p_manager the user event manager

@return  FIXME DOC
*/
fn opj_t2_decode_packet(
  mut p_t2: *mut opj_t2_t,
  mut p_tile: *mut opj_tcd_tile_t,
  mut p_tcp: *mut opj_tcp_t,
  mut p_pi: *mut opj_pi_iterator_t,
  mut p_src: *mut OPJ_BYTE,
  mut p_data_read: *mut OPJ_UINT32,
  mut p_max_length: OPJ_UINT32,
  mut p_pack_info: *mut opj_packet_info_t,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_read_data: OPJ_BOOL = 0;
    let mut l_nb_bytes_read = 0 as OPJ_UINT32;
    let mut l_nb_total_bytes_read = 0 as OPJ_UINT32;
    *p_data_read = 0 as OPJ_UINT32;
    if opj_t2_read_packet_header(
      p_t2,
      p_tile,
      p_tcp,
      p_pi,
      &mut l_read_data,
      p_src,
      &mut l_nb_bytes_read,
      p_max_length,
      p_pack_info,
      p_manager,
    ) == 0
    {
      return 0i32;
    }
    p_src = p_src.offset(l_nb_bytes_read as isize);
    l_nb_total_bytes_read =
      (l_nb_total_bytes_read as core::ffi::c_uint).wrapping_add(l_nb_bytes_read) as OPJ_UINT32;
    p_max_length = (p_max_length as core::ffi::c_uint).wrapping_sub(l_nb_bytes_read) as OPJ_UINT32;
    /* we should read data for the packet */
    if l_read_data != 0 {
      l_nb_bytes_read = 0 as OPJ_UINT32;
      if opj_t2_read_packet_data(
        p_t2,
        p_tile,
        p_pi,
        p_src,
        &mut l_nb_bytes_read,
        p_max_length,
        p_pack_info,
        p_manager,
      ) == 0
      {
        return 0i32;
      }
      l_nb_total_bytes_read =
        (l_nb_total_bytes_read as core::ffi::c_uint).wrapping_add(l_nb_bytes_read) as OPJ_UINT32
    }
    *p_data_read = l_nb_total_bytes_read;
    1i32
  }
}
/* *
Encode a packet of a tile to a destination buffer
@param tileno Number of the tile encoded
@param tile Tile for which to write the packets
@param tcp Tile coding parameters
@param pi Packet identity
@param dest Destination buffer
@param p_data_written   FIXME DOC
@param len Length of the destination buffer
@param cstr_info Codestream information structure
@param p_t2_mode If == THRESH_CALC In Threshold calculation ,If == FINAL_PASS Final pass
@param p_manager the user event manager
@return
*/
fn opj_t2_encode_packet(
  mut tileno: OPJ_UINT32,
  mut tile: *mut opj_tcd_tile_t,
  mut tcp: *mut opj_tcp_t,
  mut pi: *mut opj_pi_iterator_t,
  mut dest: *mut OPJ_BYTE,
  mut p_data_written: *mut OPJ_UINT32,
  mut length: OPJ_UINT32,
  mut cstr_info: *mut opj_codestream_info_t,
  mut p_t2_mode: J2K_T2_MODE,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut bandno: OPJ_UINT32 = 0; /* component value */
    let mut cblkno: OPJ_UINT32 = 0; /* resolution level value */
    let mut c = dest; /* precinct value */
    let mut l_nb_bytes: OPJ_UINT32 = 0; /* quality layer value */
    let mut compno = (*pi).compno; /* BIO component */
    let mut resno = (*pi).resno;
    let mut precno = (*pi).precno;
    let mut layno = (*pi).layno;
    let mut l_nb_blocks: OPJ_UINT32 = 0;
    let mut band = core::ptr::null_mut::<opj_tcd_band_t>();
    let mut cblk = core::ptr::null_mut::<opj_tcd_cblk_enc_t>();
    let mut pass = core::ptr::null_mut::<opj_tcd_pass_t>();
    let mut tilec: *mut opj_tcd_tilecomp_t =
      &mut *(*tile).comps.offset(compno as isize) as *mut opj_tcd_tilecomp_t;
    let mut res: *mut opj_tcd_resolution_t =
      &mut *(*tilec).resolutions.offset(resno as isize) as *mut opj_tcd_resolution_t;
    let mut bio = core::ptr::null_mut::<opj_bio_t>();
    let mut packet_empty = 0i32;
    /* <SOP 0xff91> */
    if (*tcp).csty & 0x2u32 != 0 {
      if length < 6u32 {
        if p_t2_mode as core::ffi::c_uint == FINAL_PASS as core::ffi::c_uint {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "opj_t2_encode_packet(): only %u bytes remaining in output buffer. %u needed.\n",
            length,
            6i32,
          ); /* packno is uint32_t */
        }
        return 0i32;
      }
      *c.offset(0) = 255 as OPJ_BYTE;
      *c.offset(1) = 145 as OPJ_BYTE;
      *c.offset(2) = 0 as OPJ_BYTE;
      *c.offset(3) = 4 as OPJ_BYTE;
      *c.offset(4) = ((*tile).packno >> 8i32 & 0xffu32) as OPJ_BYTE;
      *c.offset(5) = ((*tile).packno & 0xffu32) as OPJ_BYTE;
      c = c.offset(6);
      length = (length as core::ffi::c_uint).wrapping_sub(6u32) as OPJ_UINT32 as OPJ_UINT32
    }
    /* </SOP> */
    if layno == 0 {
      band = (*res).bands.as_mut_ptr();
      bandno = 0 as OPJ_UINT32;
      while bandno < (*res).numbands {
        let mut prc = core::ptr::null_mut::<opj_tcd_precinct_t>();
        /* Skip empty bands */
        if opj_tcd_is_band_empty(band) == 0 {
          /* Avoid out of bounds access of https://github.com/uclouvain/openjpeg/issues/1294 */
          /* but likely not a proper fix. */
          if precno >= (*res).pw.wrapping_mul((*res).ph) {
            event_msg!(
              p_manager,
              EVT_ERROR,
              "opj_t2_encode_packet(): accessing precno=%u >= %u\n",
              precno,
              (*res).pw.wrapping_mul((*res).ph),
            );
            return 0i32;
          }
          prc = &mut *(*band).precincts.offset(precno as isize) as *mut opj_tcd_precinct_t;
          opj_tgt_reset((*prc).incltree);
          opj_tgt_reset((*prc).imsbtree);
          l_nb_blocks = (*prc).cw.wrapping_mul((*prc).ch);
          cblkno = 0 as OPJ_UINT32;
          while cblkno < l_nb_blocks {
            cblk = &mut *(*prc).cblks.enc.offset(cblkno as isize) as *mut opj_tcd_cblk_enc_t;
            (*cblk).numpasses = 0 as OPJ_UINT32;
            opj_tgt_setvalue(
              (*prc).imsbtree,
              cblkno,
              (*band).numbps - (*cblk).numbps as OPJ_INT32,
            );
            cblkno += 1;
          }
        }
        bandno = bandno.wrapping_add(1);
        band = band.offset(1)
      }
    }
    bio = opj_bio_create();
    if bio.is_null() {
      /* FIXME event manager error callback */
      return 0i32;
    }
    opj_bio_init_enc(bio, c, length);
    /* Empty header bit */
    opj_bio_putbit(bio, if packet_empty != 0 { 0 } else { 1 });

    /* Writing Packet header */
    band = (*res).bands.as_mut_ptr();
    bandno = 0 as OPJ_UINT32;
    while packet_empty == 0 && bandno < (*res).numbands {
      let mut prc_0 = core::ptr::null_mut::<opj_tcd_precinct_t>();
      /* Skip empty bands */
      if opj_tcd_is_band_empty(band) == 0 {
        /* Avoid out of bounds access of https://github.com/uclouvain/openjpeg/issues/1297 */
        /* but likely not a proper fix. */
        if precno >= (*res).pw.wrapping_mul((*res).ph) {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "opj_t2_encode_packet(): accessing precno=%u >= %u\n",
            precno,
            (*res).pw.wrapping_mul((*res).ph),
          );
          return 0i32;
        }
        prc_0 = &mut *(*band).precincts.offset(precno as isize) as *mut opj_tcd_precinct_t;
        l_nb_blocks = (*prc_0).cw.wrapping_mul((*prc_0).ch);
        cblk = (*prc_0).cblks.enc;
        cblkno = 0 as OPJ_UINT32;
        while cblkno < l_nb_blocks {
          let mut layer: *mut opj_tcd_layer_t =
            &mut *(*cblk).layers.offset(layno as isize) as *mut opj_tcd_layer_t;
          if (*cblk).numpasses == 0 && (*layer).numpasses != 0 {
            opj_tgt_setvalue((*prc_0).incltree, cblkno, layno as OPJ_INT32);
          }
          cblk = cblk.offset(1);
          cblkno += 1;
        }
        cblk = (*prc_0).cblks.enc;
        cblkno = 0 as OPJ_UINT32;
        while cblkno < l_nb_blocks {
          let mut layer_0: *mut opj_tcd_layer_t =
            &mut *(*cblk).layers.offset(layno as isize) as *mut opj_tcd_layer_t;
          let mut increment = 0 as OPJ_UINT32;
          let mut nump = 0 as OPJ_UINT32;
          let mut len = 0 as OPJ_UINT32;
          let mut passno: OPJ_UINT32 = 0;
          let mut l_nb_passes: OPJ_UINT32 = 0;
          /* cblk inclusion bits */
          if (*cblk).numpasses == 0 {
            opj_tgt_encode(
              bio,
              (*prc_0).incltree,
              cblkno,
              layno.wrapping_add(1u32) as OPJ_INT32,
            );
          } else {
            opj_bio_putbit(bio, ((*layer_0).numpasses != 0u32) as u32);
          }
          /* if cblk not included, go to the next cblk  */
          if (*layer_0).numpasses == 0 {
            cblk = cblk.offset(1)
          } else {
            /* if first instance of cblk --> zero bit-planes information */
            if (*cblk).numpasses == 0 {
              (*cblk).numlenbits = 3 as OPJ_UINT32;
              opj_tgt_encode(bio, (*prc_0).imsbtree, cblkno, 999i32);
            }
            /* number of coding passes included */
            opj_t2_putnumpasses(bio, (*layer_0).numpasses);
            l_nb_passes = (*cblk).numpasses.wrapping_add((*layer_0).numpasses);
            pass = (*cblk).passes.offset((*cblk).numpasses as isize);
            /* computation of the increase of the length indicator and insertion in the header     */
            passno = (*cblk).numpasses;
            while passno < l_nb_passes {
              nump = nump.wrapping_add(1);
              len = (len as core::ffi::c_uint).wrapping_add((*pass).len) as OPJ_UINT32;
              if (*pass).term
                || passno
                  == (*cblk)
                    .numpasses
                    .wrapping_add((*layer_0).numpasses)
                    .wrapping_sub(1u32)
              {
                increment = opj_int_max(
                  increment as OPJ_INT32,
                  opj_int_floorlog2(len as OPJ_INT32) + 1i32
                    - ((*cblk).numlenbits as OPJ_INT32 + opj_int_floorlog2(nump as OPJ_INT32)),
                ) as OPJ_UINT32;
                len = 0 as OPJ_UINT32;
                nump = 0 as OPJ_UINT32
              }
              pass = pass.offset(1);
              passno += 1;
            }
            opj_t2_putcommacode(bio, increment as OPJ_INT32);
            /* computation of the new Length indicator */
            (*cblk).numlenbits =
              ((*cblk).numlenbits as core::ffi::c_uint).wrapping_add(increment) as OPJ_UINT32;
            pass = (*cblk).passes.offset((*cblk).numpasses as isize);
            /* insertion of the codeword segment length */
            passno = (*cblk).numpasses;
            while passno < l_nb_passes {
              nump = nump.wrapping_add(1);
              len = (len as core::ffi::c_uint).wrapping_add((*pass).len) as OPJ_UINT32;
              if (*pass).term
                || passno
                  == (*cblk)
                    .numpasses
                    .wrapping_add((*layer_0).numpasses)
                    .wrapping_sub(1u32)
              {
                opj_bio_write(
                  bio,
                  len,
                  (*cblk)
                    .numlenbits
                    .wrapping_add(opj_int_floorlog2(nump as OPJ_INT32) as OPJ_UINT32),
                );
                len = 0 as OPJ_UINT32;
                nump = 0 as OPJ_UINT32
              }
              pass = pass.offset(1);
              passno += 1;
            }
            cblk = cblk.offset(1)
          }
          cblkno += 1;
        }
      }
      bandno = bandno.wrapping_add(1);
      band = band.offset(1)
    }
    if opj_bio_flush(bio) == 0 {
      opj_bio_destroy(bio);
      return 0i32;
      /* modified to eliminate longjmp !! */
    }
    l_nb_bytes = opj_bio_numbytes(bio) as OPJ_UINT32;
    c = c.offset(l_nb_bytes as isize);
    length = (length as core::ffi::c_uint).wrapping_sub(l_nb_bytes) as OPJ_UINT32;
    opj_bio_destroy(bio);
    /* <EPH 0xff92> */
    if (*tcp).csty & 0x4u32 != 0 {
      if length < 2u32 {
        if p_t2_mode as core::ffi::c_uint == FINAL_PASS as core::ffi::c_uint {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "opj_t2_encode_packet(): only %u bytes remaining in output buffer. %u needed.\n",
            length,
            2i32,
          );
        }
        return 0i32;
      }
      *c.offset(0) = 255 as OPJ_BYTE;
      *c.offset(1) = 146 as OPJ_BYTE;
      c = c.offset(2);
      length = (length as core::ffi::c_uint).wrapping_sub(2u32) as OPJ_UINT32 as OPJ_UINT32
    }
    /* </EPH> */
    /* << INDEX */
    /* End of packet header position. Currently only represents the distance to start of packet
    Will be updated later by incrementing with packet start value*/
    if !cstr_info.is_null() && (*cstr_info).index_write != 0 {
      let mut info_PK: *mut opj_packet_info_t = &mut *(*(*cstr_info).tile.offset(tileno as isize))
        .packet
        .offset((*cstr_info).packno as isize)
        as *mut opj_packet_info_t;
      (*info_PK).end_ph_pos = c.offset_from(dest) as OPJ_OFF_T
    }
    /* INDEX >> */
    /* Writing the packet body */
    band = (*res).bands.as_mut_ptr();
    bandno = 0 as OPJ_UINT32;
    while packet_empty == 0 && bandno < (*res).numbands {
      let mut prc_1 = core::ptr::null_mut::<opj_tcd_precinct_t>();
      /* Skip empty bands */
      if opj_tcd_is_band_empty(band) == 0 {
        prc_1 = &mut *(*band).precincts.offset(precno as isize) as *mut opj_tcd_precinct_t;
        l_nb_blocks = (*prc_1).cw.wrapping_mul((*prc_1).ch);
        cblk = (*prc_1).cblks.enc;
        cblkno = 0 as OPJ_UINT32;
        while cblkno < l_nb_blocks {
          let mut layer_1: *mut opj_tcd_layer_t =
            &mut *(*cblk).layers.offset(layno as isize) as *mut opj_tcd_layer_t;
          if (*layer_1).numpasses == 0 {
            cblk = cblk.offset(1)
          } else {
            if (*layer_1).len > length {
              if p_t2_mode as core::ffi::c_uint == FINAL_PASS as core::ffi::c_uint {
                event_msg!(
                  p_manager,
                  EVT_ERROR,
                  "opj_t2_encode_packet(): only %u bytes remaining in output buffer. %u needed.\n",
                  length,
                  (*layer_1).len
                );
              }
              return 0i32;
            }

            if p_t2_mode == FINAL_PASS {
              memcpy(
                c as *mut core::ffi::c_void,
                (*layer_1).data as *const core::ffi::c_void,
                (*layer_1).len as usize,
              );
            }
            (*cblk).numpasses = ((*cblk).numpasses as core::ffi::c_uint)
              .wrapping_add((*layer_1).numpasses) as OPJ_UINT32;
            c = c.offset((*layer_1).len as isize);
            length = (length as core::ffi::c_uint).wrapping_sub((*layer_1).len) as OPJ_UINT32;
            /* INDEX >> */
            /* << INDEX */
            if !cstr_info.is_null() && (*cstr_info).index_write != 0 {
              let mut info_PK_0: *mut opj_packet_info_t =
                &mut *(*(*cstr_info).tile.offset(tileno as isize))
                  .packet
                  .offset((*cstr_info).packno as isize) as *mut opj_packet_info_t;
              (*info_PK_0).disto += (*layer_1).disto;
              if (*cstr_info).D_max < (*info_PK_0).disto {
                (*cstr_info).D_max = (*info_PK_0).disto
              }
            }
            cblk = cblk.offset(1)
          }
          cblkno += 1;
        }
      }
      bandno = bandno.wrapping_add(1);
      band = band.offset(1)
    }
    assert!(c >= dest);
    *p_data_written = (*p_data_written as core::ffi::c_uint)
      .wrapping_add(c.offset_from(dest) as OPJ_UINT32) as OPJ_UINT32;
    1i32
  }
}
fn opj_t2_skip_packet(
  mut p_t2: *mut opj_t2_t,
  mut p_tile: *mut opj_tcd_tile_t,
  mut p_tcp: *mut opj_tcp_t,
  mut p_pi: *mut opj_pi_iterator_t,
  mut p_src: *mut OPJ_BYTE,
  mut p_data_read: *mut OPJ_UINT32,
  mut p_max_length: OPJ_UINT32,
  mut p_pack_info: *mut opj_packet_info_t,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut l_read_data: OPJ_BOOL = 0;
    let mut l_nb_bytes_read = 0 as OPJ_UINT32;
    let mut l_nb_total_bytes_read = 0 as OPJ_UINT32;
    *p_data_read = 0 as OPJ_UINT32;
    if opj_t2_read_packet_header(
      p_t2,
      p_tile,
      p_tcp,
      p_pi,
      &mut l_read_data,
      p_src,
      &mut l_nb_bytes_read,
      p_max_length,
      p_pack_info,
      p_manager,
    ) == 0
    {
      return 0i32;
    }
    p_src = p_src.offset(l_nb_bytes_read as isize);
    l_nb_total_bytes_read =
      (l_nb_total_bytes_read as core::ffi::c_uint).wrapping_add(l_nb_bytes_read) as OPJ_UINT32;
    p_max_length = (p_max_length as core::ffi::c_uint).wrapping_sub(l_nb_bytes_read) as OPJ_UINT32;
    /* we should read data for the packet */
    if l_read_data != 0 {
      l_nb_bytes_read = 0 as OPJ_UINT32;
      if opj_t2_skip_packet_data(
        p_t2,
        p_tile,
        p_pi,
        &mut l_nb_bytes_read,
        p_max_length,
        p_pack_info,
        p_manager,
      ) == 0
      {
        return 0i32;
      }
      l_nb_total_bytes_read =
        (l_nb_total_bytes_read as core::ffi::c_uint).wrapping_add(l_nb_bytes_read) as OPJ_UINT32
    }
    *p_data_read = l_nb_total_bytes_read;
    1i32
  }
}

fn opj_t2_read_packet_header(
  mut p_t2: *mut opj_t2_t,
  mut p_tile: *mut opj_tcd_tile_t,
  mut p_tcp: *mut opj_tcp_t,
  mut p_pi: *mut opj_pi_iterator_t,
  mut p_is_data_present: *mut OPJ_BOOL,
  mut p_src_data: *mut OPJ_BYTE,
  mut p_data_read: *mut OPJ_UINT32,
  mut p_max_length: OPJ_UINT32,
  mut p_pack_info: *mut opj_packet_info_t,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    /* loop */
    let mut bandno: OPJ_UINT32 = 0; /* BIO component */
    let mut cblkno: OPJ_UINT32 = 0;
    let mut l_nb_code_blocks: OPJ_UINT32 = 0;
    let mut l_remaining_length: OPJ_UINT32 = 0;
    let mut l_header_length: OPJ_UINT32 = 0;
    let mut l_modified_length_ptr = core::ptr::null_mut::<OPJ_UINT32>();
    let mut l_current_data = p_src_data;
    let mut l_cp = (*p_t2).cp;
    let mut l_bio = core::ptr::null_mut::<opj_bio_t>();
    let mut l_band = core::ptr::null_mut::<opj_tcd_band_t>();
    let mut l_cblk = core::ptr::null_mut::<opj_tcd_cblk_dec_t>();
    let mut l_res: *mut opj_tcd_resolution_t =
      &mut *(*(*p_tile).comps.offset((*p_pi).compno as isize))
        .resolutions
        .offset((*p_pi).resno as isize) as *mut opj_tcd_resolution_t;
    let mut l_header_data = core::ptr::null_mut::<OPJ_BYTE>();
    let mut l_header_data_start = core::ptr::null_mut::<*mut OPJ_BYTE>();
    let mut l_present: OPJ_UINT32 = 0;
    if (*p_pi).layno == 0u32 {
      l_band = (*l_res).bands.as_mut_ptr();
      /* reset tagtrees */
      bandno = 0 as OPJ_UINT32;
      while bandno < (*l_res).numbands {
        if opj_tcd_is_band_empty(l_band) == 0 {
          let mut l_prc: *mut opj_tcd_precinct_t =
            &mut *(*l_band).precincts.offset((*p_pi).precno as isize) as *mut opj_tcd_precinct_t;
          if ((*p_pi).precno as usize)
            >= ((*l_band).precincts_data_size as usize)
              .wrapping_div(core::mem::size_of::<opj_tcd_precinct_t>())
          {
            event_msg!(p_manager, EVT_ERROR, "Invalid precinct\n",);
            return 0i32;
          }
          opj_tgt_reset((*l_prc).incltree);
          opj_tgt_reset((*l_prc).imsbtree);
          l_cblk = (*l_prc).cblks.dec;
          l_nb_code_blocks = (*l_prc).cw.wrapping_mul((*l_prc).ch);
          cblkno = 0 as OPJ_UINT32;
          while cblkno < l_nb_code_blocks {
            (*l_cblk).numsegs = 0 as OPJ_UINT32;
            (*l_cblk).real_num_segs = 0 as OPJ_UINT32;
            l_cblk = l_cblk.offset(1);
            cblkno += 1;
          }
        }
        l_band = l_band.offset(1);
        bandno += 1;
      }
    }

    /* SOP markers */
    if (*p_tcp).csty & 0x2u32 != 0 {
      if p_max_length < 6u32 {
        event_msg!(
          p_manager,
          EVT_WARNING,
          "Not enough space for expected SOP marker\n",
        );
      } else if *l_current_data as core::ffi::c_int != 0xffi32
        || *l_current_data.offset(1) as core::ffi::c_int != 0x91i32
      {
        event_msg!(p_manager, EVT_WARNING, "Expected SOP marker\n",);
      } else {
        l_current_data = l_current_data.offset(6)
      }
      /* * TODO : check the Nsop value */
    }

    /*
    When the marker PPT/PPM is used the packet header are store in PPT/PPM marker
    This part deal with this characteristic
    step 1: Read packet header in the saved structure
    step 2: Return to codestream for decoding
    */
    l_bio = opj_bio_create();
    if l_bio.is_null() {
      return 0i32;
    }

    if (*l_cp).ppm {
      /* PPM */
      l_header_data_start = &mut (*l_cp).ppm_data; /* Normal Case */
      l_header_data = *l_header_data_start;
      l_modified_length_ptr = &mut (*l_cp).ppm_len
    } else if (*p_tcp).ppt {
      /* PPT */
      l_header_data_start = &mut (*p_tcp).ppt_data;
      l_header_data = *l_header_data_start;
      l_modified_length_ptr = &mut (*p_tcp).ppt_len
    } else {
      l_header_data_start = &mut l_current_data;
      l_header_data = *l_header_data_start;
      l_remaining_length = p_src_data
        .offset(p_max_length as isize)
        .offset_from(l_header_data) as OPJ_UINT32;
      l_modified_length_ptr = &mut l_remaining_length
    }

    opj_bio_init_dec(l_bio, l_header_data, *l_modified_length_ptr);

    l_present = opj_bio_read(l_bio, 1 as OPJ_UINT32);
    log::debug!("present={}", l_present);
    if l_present == 0 {
      /* TODO MSD: no test to control the output of this function*/
      opj_bio_inalign(l_bio);
      l_header_data = l_header_data.offset(opj_bio_numbytes(l_bio));
      opj_bio_destroy(l_bio);
      /* EPH markers */
      if (*p_tcp).csty & 0x4u32 != 0 {
        if (*l_modified_length_ptr)
          .wrapping_sub(l_header_data.offset_from(*l_header_data_start) as OPJ_UINT32)
          < 2u32
        {
          event_msg!(
            p_manager,
            EVT_WARNING,
            "Not enough space for expected EPH marker\n",
          );
        } else if *l_header_data as core::ffi::c_int != 0xffi32
          || *l_header_data.offset(1) as core::ffi::c_int != 0x92i32
        {
          event_msg!(p_manager, EVT_WARNING, "Expected EPH marker\n",);
        } else {
          l_header_data = l_header_data.offset(2)
        }
      }
      l_header_length = l_header_data.offset_from(*l_header_data_start) as OPJ_UINT32;
      *l_modified_length_ptr =
        (*l_modified_length_ptr as core::ffi::c_uint).wrapping_sub(l_header_length) as OPJ_UINT32;
      *l_header_data_start = (*l_header_data_start).offset(l_header_length as isize);
      /* << INDEX */
      /* End of packet header position. Currently only represents the distance to start of packet
      Will be updated later by incrementing with packet start value */
      if !p_pack_info.is_null() {
        (*p_pack_info).end_ph_pos = l_current_data.offset_from(p_src_data) as OPJ_OFF_T
      }
      /* INDEX >> */
      *p_is_data_present = 0i32;
      *p_data_read = l_current_data.offset_from(p_src_data) as OPJ_UINT32;
      return 1i32;
    }

    l_band = (*l_res).bands.as_mut_ptr();
    bandno = 0 as OPJ_UINT32;
    while bandno < (*l_res).numbands {
      let mut l_prc_0: *mut opj_tcd_precinct_t =
        &mut *(*l_band).precincts.offset((*p_pi).precno as isize) as *mut opj_tcd_precinct_t;
      if opj_tcd_is_band_empty(l_band) == 0 {
        l_nb_code_blocks = (*l_prc_0).cw.wrapping_mul((*l_prc_0).ch);
        l_cblk = (*l_prc_0).cblks.dec;
        cblkno = 0 as OPJ_UINT32;
        while cblkno < l_nb_code_blocks {
          let mut l_included: OPJ_UINT32 = 0;
          let mut l_increment: OPJ_UINT32 = 0;
          let mut l_segno: OPJ_UINT32 = 0;
          let mut n: OPJ_INT32 = 0;

          /* if cblk not yet included before --> inclusion tagtree */
          if (*l_cblk).numsegs == 0 {
            l_included = opj_tgt_decode(
              l_bio,
              (*l_prc_0).incltree,
              cblkno,
              (*p_pi).layno.wrapping_add(1u32) as OPJ_INT32,
            )
          /* else one bit */
          } else {
            l_included = opj_bio_read(l_bio, 1 as OPJ_UINT32)
          }

          /* if cblk not included */
          if l_included == 0 {
            (*l_cblk).numnewpasses = 0 as OPJ_UINT32;
            l_cblk = l_cblk.offset(1);
            log::debug!("included={}", l_included);
          } else {
            /* if cblk not yet included --> zero-bitplane tagtree */
            if (*l_cblk).numsegs == 0 {
              let mut i = 0 as OPJ_UINT32;
              while opj_tgt_decode(l_bio, (*l_prc_0).imsbtree, cblkno, i as OPJ_INT32) == 0 {
                i += 1;
              }
              (*l_cblk).Mb = (*l_band).numbps as OPJ_UINT32;
              if (*l_band).numbps as u32 + 1 < i {
                /* Not totally sure what we should do in that situation,
                 * but that avoids the integer overflow of
                 * https://github.com/uclouvain/openjpeg/pull/1488
                 * while keeping the regression test suite happy.
                 */
                (*l_cblk).numbps = ((*l_band).numbps + 1 - i as i32) as u32;
              } else {
                (*l_cblk).numbps = (*l_band).numbps as u32 + 1 - i;
              }
              (*l_cblk).numlenbits = 3 as OPJ_UINT32
            }
            /* number of coding passes */
            (*l_cblk).numnewpasses = opj_t2_getnumpasses(l_bio);
            l_increment = opj_t2_getcommacode(l_bio);
            /* length indicator increment */
            (*l_cblk).numlenbits =
              ((*l_cblk).numlenbits as core::ffi::c_uint).wrapping_add(l_increment) as OPJ_UINT32;
            l_segno = 0 as OPJ_UINT32;
            if (*l_cblk).numsegs == 0 {
              if opj_t2_init_seg(
                l_cblk,
                l_segno,
                (*(*p_tcp).tccps.offset((*p_pi).compno as isize)).cblksty,
                1 as OPJ_UINT32,
              ) == 0
              {
                opj_bio_destroy(l_bio);
                return 0i32;
              }
            } else {
              l_segno = (*l_cblk).numsegs.wrapping_sub(1u32);
              if (*(*l_cblk).segs.offset(l_segno as isize)).numpasses
                == (*(*l_cblk).segs.offset(l_segno as isize)).maxpasses
              {
                l_segno = l_segno.wrapping_add(1);
                if opj_t2_init_seg(
                  l_cblk,
                  l_segno,
                  (*(*p_tcp).tccps.offset((*p_pi).compno as isize)).cblksty,
                  0 as OPJ_UINT32,
                ) == 0
                {
                  opj_bio_destroy(l_bio);
                  return 0i32;
                }
              }
            }
            n = (*l_cblk).numnewpasses as OPJ_INT32;
            if (*(*p_tcp).tccps.offset((*p_pi).compno as isize)).cblksty & 0x40u32 != 0u32 {
              loop {
                let mut bit_number: OPJ_UINT32 = 0;
                (*(*l_cblk).segs.offset(l_segno as isize)).numnewpasses = if l_segno == 0u32 {
                  1u32
                } else {
                  n as OPJ_UINT32
                };
                bit_number = (*l_cblk).numlenbits.wrapping_add(opj_uint_floorlog2(
                  (*(*l_cblk).segs.offset(l_segno as isize)).numnewpasses,
                ));
                if bit_number > 32u32 {
                  event_msg!(
                    p_manager,
                    EVT_ERROR,
                    "Invalid bit number %d in opj_t2_read_packet_header()\n",
                    bit_number,
                  );
                  opj_bio_destroy(l_bio);
                  return 0i32;
                }
                (*(*l_cblk).segs.offset(l_segno as isize)).newlen = opj_bio_read(l_bio, bit_number);
                log::debug!(
                  "included={} numnewpasses={} increment={} len={}",
                  l_included,
                  (*(*l_cblk).segs.offset(l_segno as isize)).numnewpasses,
                  l_increment,
                  (*(*l_cblk).segs.offset(l_segno as isize)).newlen,
                );
                n -= (*(*l_cblk).segs.offset(l_segno as isize)).numnewpasses as OPJ_INT32;
                if n > 0i32 {
                  l_segno = l_segno.wrapping_add(1);
                  if opj_t2_init_seg(
                    l_cblk,
                    l_segno,
                    (*(*p_tcp).tccps.offset((*p_pi).compno as isize)).cblksty,
                    0 as OPJ_UINT32,
                  ) == 0
                  {
                    opj_bio_destroy(l_bio);
                    return 0i32;
                  }
                }
                if n <= 0i32 {
                  break;
                }
              }
            } else {
              loop {
                let mut bit_number_0: OPJ_UINT32 = 0;
                (*(*l_cblk).segs.offset(l_segno as isize)).numnewpasses = opj_int_min(
                  (*(*l_cblk).segs.offset(l_segno as isize))
                    .maxpasses
                    .wrapping_sub((*(*l_cblk).segs.offset(l_segno as isize)).numpasses)
                    as OPJ_INT32,
                  n,
                )
                  as OPJ_UINT32;
                bit_number_0 = (*l_cblk).numlenbits.wrapping_add(opj_uint_floorlog2(
                  (*(*l_cblk).segs.offset(l_segno as isize)).numnewpasses,
                ));
                if bit_number_0 > 32u32 {
                  event_msg!(
                    p_manager,
                    EVT_ERROR,
                    "Invalid bit number %d in opj_t2_read_packet_header()\n",
                    bit_number_0,
                  );
                  opj_bio_destroy(l_bio);
                  return 0i32;
                }
                (*(*l_cblk).segs.offset(l_segno as isize)).newlen =
                  opj_bio_read(l_bio, bit_number_0);
                log::debug!(
                  "included={} numnewpasses={} increment={} len={}",
                  l_included,
                  (*(*l_cblk).segs.offset(l_segno as isize)).numnewpasses,
                  l_increment,
                  (*(*l_cblk).segs.offset(l_segno as isize)).newlen,
                );
                n -= (*(*l_cblk).segs.offset(l_segno as isize)).numnewpasses as OPJ_INT32;
                if n > 0i32 {
                  l_segno = l_segno.wrapping_add(1);
                  if opj_t2_init_seg(
                    l_cblk,
                    l_segno,
                    (*(*p_tcp).tccps.offset((*p_pi).compno as isize)).cblksty,
                    0 as OPJ_UINT32,
                  ) == 0
                  {
                    opj_bio_destroy(l_bio);
                    return 0i32;
                  }
                }
                if n <= 0i32 {
                  break;
                }
              }
            }
            l_cblk = l_cblk.offset(1)
          }
          cblkno += 1;
        }
      }
      bandno = bandno.wrapping_add(1);
      l_band = l_band.offset(1)
    }
    if opj_bio_inalign(l_bio) == 0 {
      opj_bio_destroy(l_bio);
      return 0i32;
    }
    l_header_data = l_header_data.offset(opj_bio_numbytes(l_bio));
    opj_bio_destroy(l_bio);
    /* EPH markers */
    if (*p_tcp).csty & 0x4u32 != 0 {
      if (*l_modified_length_ptr)
        .wrapping_sub(l_header_data.offset_from(*l_header_data_start) as OPJ_UINT32)
        < 2u32
      {
        event_msg!(
          p_manager,
          EVT_WARNING,
          "Not enough space for expected EPH marker\n",
        );
      } else if *l_header_data as core::ffi::c_int != 0xffi32
        || *l_header_data.offset(1) as core::ffi::c_int != 0x92i32
      {
        event_msg!(p_manager, EVT_WARNING, "Expected EPH marker\n",);
      } else {
        l_header_data = l_header_data.offset(2)
      }
    }
    l_header_length = l_header_data.offset_from(*l_header_data_start) as OPJ_UINT32;
    log::debug!("hdrlen={}", l_header_length);
    log::debug!("packet body");
    *l_modified_length_ptr =
      (*l_modified_length_ptr as core::ffi::c_uint).wrapping_sub(l_header_length) as OPJ_UINT32;
    *l_header_data_start = (*l_header_data_start).offset(l_header_length as isize);
    /* << INDEX */
    /* End of packet header position. Currently only represents the distance to start of packet
    Will be updated later by incrementing with packet start value */
    if !p_pack_info.is_null() {
      (*p_pack_info).end_ph_pos = l_current_data.offset_from(p_src_data) as OPJ_OFF_T
    }
    /* INDEX >> */
    *p_is_data_present = 1i32; /* next code_block */
    *p_data_read = l_current_data.offset_from(p_src_data) as OPJ_UINT32;
    1i32
  }
}
fn opj_t2_read_packet_data(
  mut p_t2: *mut opj_t2_t,
  mut p_tile: *mut opj_tcd_tile_t,
  mut p_pi: *mut opj_pi_iterator_t,
  mut p_src_data: *mut OPJ_BYTE,
  mut p_data_read: *mut OPJ_UINT32,
  mut p_max_length: OPJ_UINT32,
  mut _pack_info: *mut opj_packet_info_t,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut bandno: OPJ_UINT32 = 0;
    let mut cblkno: OPJ_UINT32 = 0;
    let mut l_nb_code_blocks: OPJ_UINT32 = 0;
    let mut l_current_data = p_src_data;
    let mut l_band = core::ptr::null_mut::<opj_tcd_band_t>();
    let mut l_cblk = core::ptr::null_mut::<opj_tcd_cblk_dec_t>();
    let mut l_res: *mut opj_tcd_resolution_t =
      &mut *(*(*p_tile).comps.offset((*p_pi).compno as isize))
        .resolutions
        .offset((*p_pi).resno as isize) as *mut opj_tcd_resolution_t;
    let mut partial_buffer = 0i32;
    l_band = (*l_res).bands.as_mut_ptr();
    bandno = 0 as OPJ_UINT32;
    while bandno < (*l_res).numbands {
      let mut l_prc: *mut opj_tcd_precinct_t =
        &mut *(*l_band).precincts.offset((*p_pi).precno as isize) as *mut opj_tcd_precinct_t;
      if (*l_band).x1 - (*l_band).x0 == 0i32 || (*l_band).y1 - (*l_band).y0 == 0i32 {
        l_band = l_band.offset(1)
      } else {
        l_nb_code_blocks = (*l_prc).cw.wrapping_mul((*l_prc).ch);
        l_cblk = (*l_prc).cblks.dec;
        cblkno = 0 as OPJ_UINT32;
        while cblkno < l_nb_code_blocks {
          let mut l_seg = core::ptr::null_mut::<opj_tcd_seg_t>();
          // if we have a partial data stream, set numchunks to zero
          // since we have no data to actually decode.
          if partial_buffer != 0 {
            (*l_cblk).numchunks = 0 as OPJ_UINT32
          }
          if (*l_cblk).numnewpasses == 0 {
            /* nothing to do */
            l_cblk = l_cblk.offset(1)
          } else {
            if (*l_cblk).numsegs == 0 {
              l_seg = (*l_cblk).segs;
              (*l_cblk).numsegs = (*l_cblk).numsegs.wrapping_add(1)
            } else {
              l_seg = &mut *(*l_cblk)
                .segs
                .offset((*l_cblk).numsegs.wrapping_sub(1u32) as isize)
                as *mut opj_tcd_seg_t;
              if (*l_seg).numpasses == (*l_seg).maxpasses {
                l_seg = l_seg.offset(1);
                (*l_cblk).numsegs = (*l_cblk).numsegs.wrapping_add(1)
              }
            }
            loop
            /* Check possible overflow (on l_current_data only, assumes input args already checked) then size */
            {
              if (l_current_data as OPJ_SIZE_T).wrapping_add((*l_seg).newlen as OPJ_SIZE_T)
                < l_current_data as OPJ_SIZE_T
                || l_current_data.offset((*l_seg).newlen as isize)
                  > p_src_data.offset(p_max_length as isize)
                || partial_buffer != 0
              {
                if (*(*p_t2).cp).strict != 0 {
                  event_msg!(p_manager, EVT_ERROR,
                                                "read: segment too long (%d) with max (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
                                                (*l_seg).newlen, p_max_length,
                                                cblkno, (*p_pi).precno, bandno,
                                                (*p_pi).resno, (*p_pi).compno);
                  return 0i32;
                } else {
                  event_msg!(p_manager, EVT_WARNING,
                                                "read: segment too long (%d) with max (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
                                                (*l_seg).newlen, p_max_length,
                                                cblkno, (*p_pi).precno, bandno,
                                                (*p_pi).resno, (*p_pi).compno);
                  // skip this codeblock since it is a partial read
                  partial_buffer = 1i32;
                  (*l_cblk).numchunks = 0 as OPJ_UINT32;
                  (*l_seg).numpasses = ((*l_seg).numpasses as core::ffi::c_uint)
                    .wrapping_add((*l_seg).numnewpasses)
                    as OPJ_UINT32;
                  (*l_cblk).numnewpasses = ((*l_cblk).numnewpasses as core::ffi::c_uint)
                    .wrapping_sub((*l_seg).numnewpasses)
                    as OPJ_UINT32;
                  if (*l_cblk).numnewpasses > 0u32 {
                    l_seg = l_seg.offset(1);
                    (*l_cblk).numsegs = (*l_cblk).numsegs.wrapping_add(1);
                    break;
                  }
                }
              } else {
                /* USE_JPWL */
                if (*l_cblk).numchunks == (*l_cblk).numchunksalloc {
                  let mut l_numchunksalloc = (*l_cblk)
                    .numchunksalloc
                    .wrapping_mul(2u32)
                    .wrapping_add(1u32);
                  let mut l_chunks = opj_realloc(
                    (*l_cblk).chunks as *mut core::ffi::c_void,
                    (l_numchunksalloc as usize)
                      .wrapping_mul(core::mem::size_of::<opj_tcd_seg_data_chunk_t>()),
                  ) as *mut opj_tcd_seg_data_chunk_t;
                  if l_chunks.is_null() {
                    event_msg!(
                      p_manager,
                      EVT_ERROR,
                      "cannot allocate opj_tcd_seg_data_chunk_t* array",
                    );
                    return 0i32;
                  }
                  (*l_cblk).chunks = l_chunks;
                  (*l_cblk).numchunksalloc = l_numchunksalloc
                }
                let fresh0 = &mut (*(*l_cblk).chunks.offset((*l_cblk).numchunks as isize)).data;
                *fresh0 = l_current_data;
                (*(*l_cblk).chunks.offset((*l_cblk).numchunks as isize)).len = (*l_seg).newlen;
                (*l_cblk).numchunks = (*l_cblk).numchunks.wrapping_add(1);
                l_current_data = l_current_data.offset((*l_seg).newlen as isize);
                (*l_seg).len =
                  ((*l_seg).len as core::ffi::c_uint).wrapping_add((*l_seg).newlen) as OPJ_UINT32;
                (*l_seg).numpasses = ((*l_seg).numpasses as core::ffi::c_uint)
                  .wrapping_add((*l_seg).numnewpasses)
                  as OPJ_UINT32;
                (*l_cblk).numnewpasses = ((*l_cblk).numnewpasses as core::ffi::c_uint)
                  .wrapping_sub((*l_seg).numnewpasses)
                  as OPJ_UINT32;
                (*l_seg).real_num_passes = (*l_seg).numpasses;
                if (*l_cblk).numnewpasses > 0u32 {
                  l_seg = l_seg.offset(1);
                  (*l_cblk).numsegs = (*l_cblk).numsegs.wrapping_add(1)
                }
              }
              if (*l_cblk).numnewpasses <= 0u32 {
                break;
              }
            }
            (*l_cblk).real_num_segs = (*l_cblk).numsegs;
            l_cblk = l_cblk.offset(1)
          }
          cblkno += 1;
        }
        l_band = l_band.offset(1)
      }
      bandno += 1;
    }
    // return the number of bytes read
    if partial_buffer != 0 {
      *p_data_read = p_max_length
    } else {
      *p_data_read = l_current_data.offset_from(p_src_data) as OPJ_UINT32
    }
    1i32
  }
}

fn opj_t2_skip_packet_data(
  mut p_t2: *mut opj_t2_t,
  mut p_tile: *mut opj_tcd_tile_t,
  mut p_pi: *mut opj_pi_iterator_t,
  mut p_data_read: *mut OPJ_UINT32,
  mut p_max_length: OPJ_UINT32,
  mut _pack_info: *mut opj_packet_info_t,
  mut p_manager: &mut opj_event_mgr,
) -> OPJ_BOOL {
  unsafe {
    let mut bandno: OPJ_UINT32 = 0;
    let mut cblkno: OPJ_UINT32 = 0;
    let mut l_nb_code_blocks: OPJ_UINT32 = 0;
    let mut l_band = core::ptr::null_mut::<opj_tcd_band_t>();
    let mut l_cblk = core::ptr::null_mut::<opj_tcd_cblk_dec_t>();
    let mut l_res: *mut opj_tcd_resolution_t =
      &mut *(*(*p_tile).comps.offset((*p_pi).compno as isize))
        .resolutions
        .offset((*p_pi).resno as isize) as *mut opj_tcd_resolution_t;
    *p_data_read = 0 as OPJ_UINT32;
    l_band = (*l_res).bands.as_mut_ptr();
    bandno = 0 as OPJ_UINT32;
    while bandno < (*l_res).numbands {
      let mut l_prc: *mut opj_tcd_precinct_t =
        &mut *(*l_band).precincts.offset((*p_pi).precno as isize) as *mut opj_tcd_precinct_t;
      if (*l_band).x1 - (*l_band).x0 == 0i32 || (*l_band).y1 - (*l_band).y0 == 0i32 {
        l_band = l_band.offset(1)
      } else {
        l_nb_code_blocks = (*l_prc).cw.wrapping_mul((*l_prc).ch);
        l_cblk = (*l_prc).cblks.dec;
        cblkno = 0 as OPJ_UINT32;
        while cblkno < l_nb_code_blocks {
          let mut l_seg = core::ptr::null_mut::<opj_tcd_seg_t>();
          if (*l_cblk).numnewpasses == 0 {
            /* nothing to do */
            l_cblk = l_cblk.offset(1)
          } else {
            if (*l_cblk).numsegs == 0 {
              l_seg = (*l_cblk).segs;
              (*l_cblk).numsegs = (*l_cblk).numsegs.wrapping_add(1)
            } else {
              l_seg = &mut *(*l_cblk)
                .segs
                .offset((*l_cblk).numsegs.wrapping_sub(1u32) as isize)
                as *mut opj_tcd_seg_t;
              if (*l_seg).numpasses == (*l_seg).maxpasses {
                l_seg = l_seg.offset(1);
                (*l_cblk).numsegs = (*l_cblk).numsegs.wrapping_add(1)
              }
            }
            loop {
              /* Check possible overflow then size */
              if (*p_data_read).wrapping_add((*l_seg).newlen) < *p_data_read
                || (*p_data_read).wrapping_add((*l_seg).newlen) > p_max_length
              {
                if (*(*p_t2).cp).strict != 0 {
                  event_msg!(p_manager, EVT_ERROR,
                                                "skip: segment too long (%d) with max (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
                                                (*l_seg).newlen, p_max_length,
                                                cblkno, (*p_pi).precno, bandno,
                                                (*p_pi).resno, (*p_pi).compno);
                  return 0i32;
                } else {
                  event_msg!(p_manager, EVT_WARNING,
                                                "skip: segment too long (%d) with max (%d) for codeblock %d (p=%d, b=%d, r=%d, c=%d)\n",
                                                (*l_seg).newlen, p_max_length,
                                                cblkno, (*p_pi).precno, bandno,
                                                (*p_pi).resno, (*p_pi).compno);
                  return 1i32;
                }
              }
              /* USE_JPWL */
              log::debug!(
                "p_data_read ({}) newlen ({})",
                *p_data_read,
                (*l_seg).newlen
              );
              *p_data_read =
                (*p_data_read as core::ffi::c_uint).wrapping_add((*l_seg).newlen) as OPJ_UINT32;
              (*l_seg).numpasses = ((*l_seg).numpasses as core::ffi::c_uint)
                .wrapping_add((*l_seg).numnewpasses)
                as OPJ_UINT32 as OPJ_UINT32;
              (*l_cblk).numnewpasses = ((*l_cblk).numnewpasses as core::ffi::c_uint)
                .wrapping_sub((*l_seg).numnewpasses)
                as OPJ_UINT32;
              if (*l_cblk).numnewpasses > 0u32 {
                l_seg = l_seg.offset(1);
                (*l_cblk).numsegs = (*l_cblk).numsegs.wrapping_add(1)
              }
              if (*l_cblk).numnewpasses <= 0u32 {
                break;
              }
            }
            l_cblk = l_cblk.offset(1)
          }
          cblkno += 1;
        }
        l_band = l_band.offset(1)
      }
      bandno += 1;
    }
    1i32
  }
}
/* *
@param cblk
@param index
@param cblksty
@param first
*/
fn opj_t2_init_seg(
  mut cblk: *mut opj_tcd_cblk_dec_t,
  mut index: OPJ_UINT32,
  mut cblksty: OPJ_UINT32,
  mut first: OPJ_UINT32,
) -> OPJ_BOOL {
  unsafe {
    let mut seg = core::ptr::null_mut::<opj_tcd_seg_t>();
    let mut l_nb_segs = index.wrapping_add(1u32);
    if l_nb_segs > (*cblk).m_current_max_segs {
      let mut new_segs = core::ptr::null_mut::<opj_tcd_seg_t>();
      let mut l_m_current_max_segs = (*cblk).m_current_max_segs.wrapping_add(10u32);
      new_segs = opj_realloc(
        (*cblk).segs as *mut core::ffi::c_void,
        (l_m_current_max_segs as usize).wrapping_mul(core::mem::size_of::<opj_tcd_seg_t>()),
      ) as *mut opj_tcd_seg_t;
      if new_segs.is_null() {
        /* event_msg!(p_manager, EVT_ERROR, "Not enough memory to initialize segment %d\n", l_nb_segs); */
        return 0i32;
      }
      (*cblk).segs = new_segs;
      memset(
        new_segs.offset((*cblk).m_current_max_segs as isize) as *mut core::ffi::c_void,
        0i32,
        10 * core::mem::size_of::<opj_tcd_seg_t>(),
      );
      (*cblk).m_current_max_segs = l_m_current_max_segs
    }
    seg = &mut *(*cblk).segs.offset(index as isize) as *mut opj_tcd_seg_t;
    opj_tcd_reinit_segment(seg);
    if cblksty & 0x4u32 != 0 {
      (*seg).maxpasses = 1 as OPJ_UINT32
    } else if cblksty & 0x1u32 != 0 {
      if first != 0 {
        (*seg).maxpasses = 10 as OPJ_UINT32
      } else {
        (*seg).maxpasses =
          if (*seg.offset(-1)).maxpasses == 1u32 || (*seg.offset(-1)).maxpasses == 10u32 {
            2i32
          } else {
            1i32
          } as OPJ_UINT32
      }
    } else {
      /* See paragraph "B.10.6 Number of coding passes" of the standard.
       * Probably that 109 must be interpreted a (Mb-1)*3 + 1 with Mb=37,
       * Mb being the maximum number of bit-planes available for the
       * representation of coefficients in the sub-band */
      (*seg).maxpasses = 109 as OPJ_UINT32
    }
    1i32
  }
}

use super::event::*;
use super::j2k::*;
use super::math::*;
use super::openjpeg::*;

use super::malloc::*;

pub type T2_MODE = core::ffi::c_uint;
pub const FINAL_PASS: T2_MODE = 1;
pub const THRESH_CALC: T2_MODE = 0;
pub type J2K_T2_MODE = T2_MODE;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_pi_resolution {
  pub pdx: OPJ_UINT32,
  pub pdy: OPJ_UINT32,
  pub pw: OPJ_UINT32,
  pub ph: OPJ_UINT32,
}
pub type opj_pi_resolution_t = opj_pi_resolution;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct opj_pi_comp {
  pub dx: OPJ_UINT32,
  pub dy: OPJ_UINT32,
  pub numresolutions: OPJ_UINT32,
  pub resolutions: *mut opj_pi_resolution_t,
}
pub type opj_pi_comp_t = opj_pi_comp;

#[repr(C)]
#[derive(Clone)]
pub struct opj_pi_iterator {
  pub tp_on: OPJ_BYTE,
  pub include: *mut OPJ_INT16,
  pub include_size: OPJ_UINT32,
  pub step_l: OPJ_UINT32,
  pub step_r: OPJ_UINT32,
  pub step_c: OPJ_UINT32,
  pub step_p: OPJ_UINT32,
  pub compno: OPJ_UINT32,
  pub resno: OPJ_UINT32,
  pub precno: OPJ_UINT32,
  pub layno: OPJ_UINT32,
  pub first: OPJ_BOOL,
  pub poc: opj_poc_t,
  pub numcomps: OPJ_UINT32,
  pub comps: *mut opj_pi_comp_t,
  pub tx0: OPJ_UINT32,
  pub ty0: OPJ_UINT32,
  pub tx1: OPJ_UINT32,
  pub ty1: OPJ_UINT32,
  pub x: OPJ_UINT32,
  pub y: OPJ_UINT32,
  pub dx: OPJ_UINT32,
  pub dy: OPJ_UINT32,
  pub manager: opj_event_mgr,
}
pub type opj_pi_iterator_t = opj_pi_iterator;

#[inline]
fn opj_uint_ceildivpow2(mut a: OPJ_UINT32, mut b: OPJ_UINT32) -> OPJ_UINT32 {
  ((a as u64)
    .wrapping_add((1 as OPJ_UINT64) << b)
    .wrapping_sub(1u64)
    >> b) as u32
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
 * Copyright (c) 2006-2007, Parvatha Elangovan
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
/* * @defgroup PI PI - Implementation of a packet iterator */
/*@{*/
/* * @name Local static functions */
/*@{*/
/* *
Get next packet in layer-resolution-component-precinct order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
/*@}*/
/*@}*/
/*
==========================================================
   local functions
==========================================================
*/
fn opj_pi_next_lrcp(mut pi: *mut opj_pi_iterator_t) -> OPJ_BOOL {
  unsafe {
    let mut current_block: u64;
    let mut comp = core::ptr::null_mut::<opj_pi_comp_t>();
    let mut res = core::ptr::null_mut::<opj_pi_resolution_t>();
    let mut index = 0 as OPJ_UINT32;
    if (*pi).poc.compno0 >= (*pi).numcomps || (*pi).poc.compno1 >= (*pi).numcomps.wrapping_add(1u32)
    {
      event_msg!(
        (*pi).manager,
        EVT_ERROR,
        "opj_pi_next_lrcp(): invalid compno0/compno1\n",
      );
      return 0i32;
    }
    if (*pi).first == 0 {
      comp = &mut *(*pi).comps.offset((*pi).compno as isize) as *mut opj_pi_comp_t;
      res = &mut *(*comp).resolutions.offset((*pi).resno as isize) as *mut opj_pi_resolution_t;
      current_block = 5634871135123216486;
    } else {
      (*pi).first = 0i32;
      (*pi).layno = (*pi).poc.layno0;
      current_block = 2868539653012386629;
    }
    loop {
      match current_block {
        5634871135123216486 => {
          (*pi).precno = (*pi).precno.wrapping_add(1);
          current_block = 1109700713171191020;
        }
        _ => {
          if (*pi).layno >= (*pi).poc.layno1 {
            break;
          }
          (*pi).resno = (*pi).poc.resno0;
          current_block = 1856101646708284338;
        }
      }
      loop {
        match current_block {
          1856101646708284338 => {
            if (*pi).resno < (*pi).poc.resno1 {
              (*pi).compno = (*pi).poc.compno0
            } else {
              (*pi).layno = (*pi).layno.wrapping_add(1);
              current_block = 2868539653012386629;
              break;
            }
            current_block = 10048703153582371463;
          }
          _ => {
            if (*pi).precno < (*pi).poc.precno1 {
              index = (*pi)
                .layno
                .wrapping_mul((*pi).step_l)
                .wrapping_add((*pi).resno.wrapping_mul((*pi).step_r))
                .wrapping_add((*pi).compno.wrapping_mul((*pi).step_c))
                .wrapping_add((*pi).precno.wrapping_mul((*pi).step_p));
              /* Avoids index out of bounds access with */
              /* id_000098,sig_11,src_005411,op_havoc,rep_2 of */
              /* https://github.com/uclouvain/openjpeg/issues/938 */
              /* Not sure if this is the most clever fix. Perhaps */
              /* include should be resized when a POC arises, or */
              /* the POC should be rejected */
              if index >= (*pi).include_size {
                event_msg!((*pi).manager, EVT_ERROR, "Invalid access to pi->include",);
                return 0i32;
              }
              if *(*pi).include.offset(index as isize) == 0 {
                *(*pi).include.offset(index as isize) = 1 as OPJ_INT16;
                return 1i32;
              }
              current_block = 5634871135123216486;
              break;
            } else {
              current_block = 17860125682698302841;
            }
          }
        }
        loop {
          match current_block {
            17860125682698302841 => {
              (*pi).compno = (*pi).compno.wrapping_add(1);
              current_block = 10048703153582371463;
            }
            _ => {
              if (*pi).compno < (*pi).poc.compno1 {
                comp = &mut *(*pi).comps.offset((*pi).compno as isize) as *mut opj_pi_comp_t;
                if (*pi).resno >= (*comp).numresolutions {
                  current_block = 17860125682698302841;
                  continue;
                }
                res = &mut *(*comp).resolutions.offset((*pi).resno as isize)
                  as *mut opj_pi_resolution_t;
                if (*pi).tp_on == 0 {
                  (*pi).poc.precno1 = (*res).pw.wrapping_mul((*res).ph)
                }
                (*pi).precno = (*pi).poc.precno0;
                current_block = 1109700713171191020;
                break;
              } else {
                (*pi).resno = (*pi).resno.wrapping_add(1);
                current_block = 1856101646708284338;
                break;
              }
            }
          }
        }
      }
    }
    0i32
  }
}
/* *
Get next packet in resolution-layer-component-precinct order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
fn opj_pi_next_rlcp(mut pi: *mut opj_pi_iterator_t) -> OPJ_BOOL {
  unsafe {
    let mut current_block: u64;
    let mut comp = core::ptr::null_mut::<opj_pi_comp_t>();
    let mut res = core::ptr::null_mut::<opj_pi_resolution_t>();
    let mut index = 0 as OPJ_UINT32;
    if (*pi).poc.compno0 >= (*pi).numcomps || (*pi).poc.compno1 >= (*pi).numcomps.wrapping_add(1u32)
    {
      event_msg!(
        (*pi).manager,
        EVT_ERROR,
        "opj_pi_next_rlcp(): invalid compno0/compno1\n",
      );
      return 0i32;
    }
    if (*pi).first == 0 {
      comp = &mut *(*pi).comps.offset((*pi).compno as isize) as *mut opj_pi_comp_t;
      res = &mut *(*comp).resolutions.offset((*pi).resno as isize) as *mut opj_pi_resolution_t;
      current_block = 5634871135123216486;
    } else {
      (*pi).first = 0i32;
      (*pi).resno = (*pi).poc.resno0;
      current_block = 2868539653012386629;
    }
    loop {
      match current_block {
        5634871135123216486 => {
          (*pi).precno = (*pi).precno.wrapping_add(1);
          current_block = 1109700713171191020;
        }
        _ => {
          if (*pi).resno >= (*pi).poc.resno1 {
            break;
          }
          (*pi).layno = (*pi).poc.layno0;
          current_block = 1856101646708284338;
        }
      }
      loop {
        match current_block {
          1856101646708284338 => {
            if (*pi).layno < (*pi).poc.layno1 {
              (*pi).compno = (*pi).poc.compno0
            } else {
              (*pi).resno = (*pi).resno.wrapping_add(1);
              current_block = 2868539653012386629;
              break;
            }
            current_block = 10048703153582371463;
          }
          _ => {
            if (*pi).precno < (*pi).poc.precno1 {
              index = (*pi)
                .layno
                .wrapping_mul((*pi).step_l)
                .wrapping_add((*pi).resno.wrapping_mul((*pi).step_r))
                .wrapping_add((*pi).compno.wrapping_mul((*pi).step_c))
                .wrapping_add((*pi).precno.wrapping_mul((*pi).step_p));
              if index >= (*pi).include_size {
                event_msg!((*pi).manager, EVT_ERROR, "Invalid access to pi->include",);
                return 0i32;
              }
              if *(*pi).include.offset(index as isize) == 0 {
                *(*pi).include.offset(index as isize) = 1 as OPJ_INT16;
                return 1i32;
              }
              current_block = 5634871135123216486;
              break;
            } else {
              current_block = 17860125682698302841;
            }
          }
        }
        loop {
          match current_block {
            17860125682698302841 => {
              (*pi).compno = (*pi).compno.wrapping_add(1);
              current_block = 10048703153582371463;
            }
            _ => {
              if (*pi).compno < (*pi).poc.compno1 {
                comp = &mut *(*pi).comps.offset((*pi).compno as isize) as *mut opj_pi_comp_t;
                if (*pi).resno >= (*comp).numresolutions {
                  current_block = 17860125682698302841;
                  continue;
                }
                res = &mut *(*comp).resolutions.offset((*pi).resno as isize)
                  as *mut opj_pi_resolution_t;
                if (*pi).tp_on == 0 {
                  (*pi).poc.precno1 = (*res).pw.wrapping_mul((*res).ph)
                }
                (*pi).precno = (*pi).poc.precno0;
                current_block = 1109700713171191020;
                break;
              } else {
                (*pi).layno = (*pi).layno.wrapping_add(1);
                current_block = 1856101646708284338;
                break;
              }
            }
          }
        }
      }
    }
    0i32
  }
}
/* *
Get next packet in resolution-precinct-component-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
fn opj_pi_next_rpcl(mut pi: *mut opj_pi_iterator_t) -> OPJ_BOOL {
  unsafe {
    let mut levelno: OPJ_UINT32 = 0;
    let mut trx0: OPJ_UINT32 = 0;
    let mut try0: OPJ_UINT32 = 0;
    let mut trx1: OPJ_UINT32 = 0;
    let mut try1: OPJ_UINT32 = 0;
    let mut rpx: OPJ_UINT32 = 0;
    let mut rpy: OPJ_UINT32 = 0;
    let mut prci: OPJ_UINT32 = 0;
    let mut prcj: OPJ_UINT32 = 0;
    let mut current_block: u64;
    let mut comp = core::ptr::null_mut::<opj_pi_comp_t>();
    let mut res = core::ptr::null_mut::<opj_pi_resolution_t>();
    let mut index = 0 as OPJ_UINT32;
    if (*pi).poc.compno0 >= (*pi).numcomps || (*pi).poc.compno1 >= (*pi).numcomps.wrapping_add(1u32)
    {
      event_msg!(
        (*pi).manager,
        EVT_ERROR,
        "opj_pi_next_rpcl(): invalid compno0/compno1\n",
      );
      return 0i32;
    }
    if (*pi).first == 0 {
      current_block = 129780949503461575;
    } else {
      let mut compno: OPJ_UINT32 = 0;
      let mut resno: OPJ_UINT32 = 0;
      (*pi).first = 0i32;
      (*pi).dx = 0 as OPJ_UINT32;
      (*pi).dy = 0 as OPJ_UINT32;
      compno = 0 as OPJ_UINT32;
      while compno < (*pi).numcomps {
        comp = &mut *(*pi).comps.offset(compno as isize) as *mut opj_pi_comp_t;
        resno = 0 as OPJ_UINT32;
        while resno < (*comp).numresolutions {
          let mut dx: OPJ_UINT32 = 0;
          let mut dy: OPJ_UINT32 = 0;
          res = &mut *(*comp).resolutions.offset(resno as isize) as *mut opj_pi_resolution_t;
          if (*res)
            .pdx
            .wrapping_add((*comp).numresolutions)
            .wrapping_sub(1u32)
            .wrapping_sub(resno)
            < 32u32
            && (*comp).dx
              <= (2147483647u32)
                .wrapping_mul(2u32)
                .wrapping_add(1u32)
                .wrapping_div(
                  (1u32)
                    << (*res)
                      .pdx
                      .wrapping_add((*comp).numresolutions)
                      .wrapping_sub(1u32)
                      .wrapping_sub(resno),
                )
          {
            dx = (*comp).dx.wrapping_mul(
              (1u32)
                << (*res)
                  .pdx
                  .wrapping_add((*comp).numresolutions)
                  .wrapping_sub(1u32)
                  .wrapping_sub(resno),
            );
            (*pi).dx = if (*pi).dx == 0 {
              dx
            } else {
              opj_uint_min((*pi).dx, dx)
            }
          }
          if (*res)
            .pdy
            .wrapping_add((*comp).numresolutions)
            .wrapping_sub(1u32)
            .wrapping_sub(resno)
            < 32u32
            && (*comp).dy
              <= (2147483647u32)
                .wrapping_mul(2u32)
                .wrapping_add(1u32)
                .wrapping_div(
                  (1u32)
                    << (*res)
                      .pdy
                      .wrapping_add((*comp).numresolutions)
                      .wrapping_sub(1u32)
                      .wrapping_sub(resno),
                )
          {
            dy = (*comp).dy.wrapping_mul(
              (1u32)
                << (*res)
                  .pdy
                  .wrapping_add((*comp).numresolutions)
                  .wrapping_sub(1u32)
                  .wrapping_sub(resno),
            );
            (*pi).dy = if (*pi).dy == 0 {
              dy
            } else {
              opj_uint_min((*pi).dy, dy)
            }
          }
          resno += 1;
        }
        compno += 1;
      }
      if (*pi).dx == 0u32 || (*pi).dy == 0u32 {
        return 0i32;
      }
      if (*pi).tp_on == 0 {
        (*pi).poc.ty0 = (*pi).ty0;
        (*pi).poc.tx0 = (*pi).tx0;
        (*pi).poc.ty1 = (*pi).ty1;
        (*pi).poc.tx1 = (*pi).tx1
      }
      (*pi).resno = (*pi).poc.resno0;
      current_block = 11385396242402735691;
    }
    loop {
      match current_block {
        129780949503461575 => {
          (*pi).layno = (*pi).layno.wrapping_add(1);
          current_block = 2606304779496145856;
        }
        _ => {
          if (*pi).resno >= (*pi).poc.resno1 {
            break;
          }
          (*pi).y = (*pi).poc.ty0;
          current_block = 6450636197030046351;
        }
      }
      loop {
        match current_block {
          6450636197030046351 => {
            if (*pi).y < (*pi).poc.ty1 {
              (*pi).x = (*pi).poc.tx0
            } else {
              (*pi).resno = (*pi).resno.wrapping_add(1);
              current_block = 11385396242402735691;
              break;
            }
            current_block = 3123434771885419771;
          }
          _ => {
            if (*pi).layno < (*pi).poc.layno1 {
              index = (*pi)
                .layno
                .wrapping_mul((*pi).step_l)
                .wrapping_add((*pi).resno.wrapping_mul((*pi).step_r))
                .wrapping_add((*pi).compno.wrapping_mul((*pi).step_c))
                .wrapping_add((*pi).precno.wrapping_mul((*pi).step_p));
              if index >= (*pi).include_size {
                event_msg!((*pi).manager, EVT_ERROR, "Invalid access to pi->include",);
                return 0i32;
              }
              if *(*pi).include.offset(index as isize) == 0 {
                *(*pi).include.offset(index as isize) = 1 as OPJ_INT16;
                return 1i32;
              }
              current_block = 129780949503461575;
              break;
            } else {
              current_block = 10891380440665537214;
            }
          }
        }
        loop {
          match current_block {
            10891380440665537214 => (*pi).compno = (*pi).compno.wrapping_add(1),
            _ => {
              if (*pi).x < (*pi).poc.tx1 {
                (*pi).compno = (*pi).poc.compno0
              } else {
                (*pi).y = ((*pi).y as core::ffi::c_uint)
                  .wrapping_add((*pi).dy.wrapping_sub((*pi).y.wrapping_rem((*pi).dy)))
                  as OPJ_UINT32;
                current_block = 6450636197030046351;
                break;
              }
            }
          }
          if (*pi).compno < (*pi).poc.compno1 {
            levelno = 0;
            trx0 = 0;
            try0 = 0;
            trx1 = 0;
            try1 = 0;
            rpx = 0;
            rpy = 0;
            prci = 0;
            prcj = 0;
            comp = &mut *(*pi).comps.offset((*pi).compno as isize) as *mut opj_pi_comp_t;
            if (*pi).resno >= (*comp).numresolutions {
              current_block = 10891380440665537214;
              continue;
            }
            res =
              &mut *(*comp).resolutions.offset((*pi).resno as isize) as *mut opj_pi_resolution_t;
            levelno = (*comp).numresolutions - 1 - (*pi).resno;
            let dx = (*comp).dx as u64;
            let dy = (*comp).dy as u64;

            if ((dx << levelno as u64) >> levelno) != dx
              || ((dy << levelno as u64) >> levelno) != dy
            {
              current_block = 10891380440665537214;
              continue;
            }

            trx0 = opj_uint64_ceildiv_res_uint32((*pi).tx0 as u64, dx << levelno);
            try0 = opj_uint64_ceildiv_res_uint32((*pi).ty0 as u64, dy << levelno);
            trx1 = opj_uint64_ceildiv_res_uint32((*pi).tx1 as u64, dx << levelno);
            try1 = opj_uint64_ceildiv_res_uint32((*pi).ty1 as u64, dy << levelno);
            rpx = (*res).pdx.wrapping_add(levelno);
            rpy = (*res).pdy.wrapping_add(levelno);

            if ((dx << rpx) >> rpx) != dx || ((dy << rpy) >> rpy) != dy {
              current_block = 10891380440665537214;
              continue;
            }

            /* See ISO-15441. B.12.1.3 Resolution level-position-component-layer progression */
            if !(((*pi).y as u64 % (dy << rpy)) == 0
              || ((*pi).y == (*pi).ty0 && (((try0 as u64) << levelno) % (1u64 << rpy)) != 0))
            {
              current_block = 10891380440665537214;
              continue;
            }
            if !(((*pi).x as u64 % (dx << rpx)) == 0
              || ((*pi).x == (*pi).tx0 && (((trx0 as u64) << levelno) % (1u64 << rpx)) != 0))
            {
              current_block = 10891380440665537214;
              continue;
            }
            if (*res).pw == 0u32 || (*res).ph == 0u32 {
              current_block = 10891380440665537214;
              continue;
            }
            if trx0 == trx1 || try0 == try1 {
              current_block = 10891380440665537214;
              continue;
            }
            prci = opj_uint_floordivpow2(
              opj_uint64_ceildiv_res_uint32((*pi).x as u64, dx << levelno),
              (*res).pdx,
            ) - opj_uint_floordivpow2(trx0, (*res).pdx);
            prcj = opj_uint_floordivpow2(
              opj_uint64_ceildiv_res_uint32((*pi).y as u64, dy << levelno),
              (*res).pdy,
            ) - opj_uint_floordivpow2(try0, (*res).pdy);
            (*pi).precno = prci.wrapping_add(prcj.wrapping_mul((*res).pw));
            (*pi).layno = (*pi).poc.layno0;
            current_block = 2606304779496145856;
            break;
          } else {
            (*pi).x = ((*pi).x as core::ffi::c_uint)
              .wrapping_add((*pi).dx.wrapping_sub((*pi).x.wrapping_rem((*pi).dx)))
              as OPJ_UINT32;
            current_block = 3123434771885419771;
          }
        }
      }
    }
    0i32
  }
}
/* *
Get next packet in precinct-component-resolution-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
fn opj_pi_next_pcrl(mut pi: *mut opj_pi_iterator_t) -> OPJ_BOOL {
  unsafe {
    let mut levelno: OPJ_UINT32 = 0;
    let mut trx0: OPJ_UINT32 = 0;
    let mut try0: OPJ_UINT32 = 0;
    let mut trx1: OPJ_UINT32 = 0;
    let mut try1: OPJ_UINT32 = 0;
    let mut rpx: OPJ_UINT32 = 0;
    let mut rpy: OPJ_UINT32 = 0;
    let mut prci: OPJ_UINT32 = 0;
    let mut prcj: OPJ_UINT32 = 0;
    let mut current_block: u64;
    let mut comp = core::ptr::null_mut::<opj_pi_comp_t>();
    let mut res = core::ptr::null_mut::<opj_pi_resolution_t>();
    let mut index = 0 as OPJ_UINT32;
    if (*pi).poc.compno0 >= (*pi).numcomps || (*pi).poc.compno1 >= (*pi).numcomps.wrapping_add(1u32)
    {
      event_msg!(
        (*pi).manager,
        EVT_ERROR,
        "opj_pi_next_pcrl(): invalid compno0/compno1\n",
      );
      return 0i32;
    }
    if (*pi).first == 0 {
      comp = &mut *(*pi).comps.offset((*pi).compno as isize) as *mut opj_pi_comp_t;
      current_block = 10853015579903106591;
    } else {
      let mut compno: OPJ_UINT32 = 0;
      let mut resno: OPJ_UINT32 = 0;
      (*pi).first = 0i32;
      (*pi).dx = 0 as OPJ_UINT32;
      (*pi).dy = 0 as OPJ_UINT32;
      compno = 0 as OPJ_UINT32;
      while compno < (*pi).numcomps {
        comp = &mut *(*pi).comps.offset(compno as isize) as *mut opj_pi_comp_t;
        resno = 0 as OPJ_UINT32;
        while resno < (*comp).numresolutions {
          let mut dx: OPJ_UINT32 = 0;
          let mut dy: OPJ_UINT32 = 0;
          res = &mut *(*comp).resolutions.offset(resno as isize) as *mut opj_pi_resolution_t;
          if (*res)
            .pdx
            .wrapping_add((*comp).numresolutions)
            .wrapping_sub(1u32)
            .wrapping_sub(resno)
            < 32u32
            && (*comp).dx
              <= (2147483647u32)
                .wrapping_mul(2u32)
                .wrapping_add(1u32)
                .wrapping_div(
                  (1u32)
                    << (*res)
                      .pdx
                      .wrapping_add((*comp).numresolutions)
                      .wrapping_sub(1u32)
                      .wrapping_sub(resno),
                )
          {
            dx = (*comp).dx.wrapping_mul(
              (1u32)
                << (*res)
                  .pdx
                  .wrapping_add((*comp).numresolutions)
                  .wrapping_sub(1u32)
                  .wrapping_sub(resno),
            );
            (*pi).dx = if (*pi).dx == 0 {
              dx
            } else {
              opj_uint_min((*pi).dx, dx)
            }
          }
          if (*res)
            .pdy
            .wrapping_add((*comp).numresolutions)
            .wrapping_sub(1u32)
            .wrapping_sub(resno)
            < 32u32
            && (*comp).dy
              <= (2147483647u32)
                .wrapping_mul(2u32)
                .wrapping_add(1u32)
                .wrapping_div(
                  (1u32)
                    << (*res)
                      .pdy
                      .wrapping_add((*comp).numresolutions)
                      .wrapping_sub(1u32)
                      .wrapping_sub(resno),
                )
          {
            dy = (*comp).dy.wrapping_mul(
              (1u32)
                << (*res)
                  .pdy
                  .wrapping_add((*comp).numresolutions)
                  .wrapping_sub(1u32)
                  .wrapping_sub(resno),
            );
            (*pi).dy = if (*pi).dy == 0 {
              dy
            } else {
              opj_uint_min((*pi).dy, dy)
            }
          }
          resno += 1;
        }
        compno += 1;
      }
      if (*pi).dx == 0u32 || (*pi).dy == 0u32 {
        return 0i32;
      }
      if (*pi).tp_on == 0 {
        (*pi).poc.ty0 = (*pi).ty0;
        (*pi).poc.tx0 = (*pi).tx0;
        (*pi).poc.ty1 = (*pi).ty1;
        (*pi).poc.tx1 = (*pi).tx1
      }
      (*pi).y = (*pi).poc.ty0;
      current_block = 7245201122033322888;
    }
    loop {
      match current_block {
        10853015579903106591 => {
          (*pi).layno = (*pi).layno.wrapping_add(1);
          current_block = 6281126495347172768;
        }
        _ => {
          if (*pi).y >= (*pi).poc.ty1 {
            break;
          }
          (*pi).x = (*pi).poc.tx0;
          current_block = 8845338526596852646;
        }
      }
      loop {
        match current_block {
          8845338526596852646 => {
            if (*pi).x < (*pi).poc.tx1 {
              (*pi).compno = (*pi).poc.compno0
            } else {
              (*pi).y = ((*pi).y as core::ffi::c_uint)
                .wrapping_add((*pi).dy.wrapping_sub((*pi).y.wrapping_rem((*pi).dy)))
                as OPJ_UINT32;
              current_block = 7245201122033322888;
              break;
            }
            current_block = 980989089337379490;
          }
          _ => {
            if (*pi).layno < (*pi).poc.layno1 {
              index = (*pi)
                .layno
                .wrapping_mul((*pi).step_l)
                .wrapping_add((*pi).resno.wrapping_mul((*pi).step_r))
                .wrapping_add((*pi).compno.wrapping_mul((*pi).step_c))
                .wrapping_add((*pi).precno.wrapping_mul((*pi).step_p));
              if index >= (*pi).include_size {
                event_msg!((*pi).manager, EVT_ERROR, "Invalid access to pi->include",);
                return 0i32;
              }
              if *(*pi).include.offset(index as isize) == 0 {
                *(*pi).include.offset(index as isize) = 1 as OPJ_INT16;
                return 1i32;
              }
              current_block = 10853015579903106591;
              break;
            } else {
              current_block = 15512526488502093901;
            }
          }
        }
        loop {
          match current_block {
            15512526488502093901 => (*pi).resno = (*pi).resno.wrapping_add(1),
            _ => {
              if (*pi).compno < (*pi).poc.compno1 {
                comp = &mut *(*pi).comps.offset((*pi).compno as isize) as *mut opj_pi_comp_t;
                (*pi).resno = (*pi).poc.resno0
              } else {
                (*pi).x = ((*pi).x as core::ffi::c_uint)
                  .wrapping_add((*pi).dx.wrapping_sub((*pi).x.wrapping_rem((*pi).dx)))
                  as OPJ_UINT32;
                current_block = 8845338526596852646;
                break;
              }
            }
          }
          if (*pi).resno < opj_uint_min((*pi).poc.resno1, (*comp).numresolutions) {
            levelno = 0;
            trx0 = 0;
            try0 = 0;
            trx1 = 0;
            try1 = 0;
            rpx = 0;
            rpy = 0;
            prci = 0;
            prcj = 0;
            res =
              &mut *(*comp).resolutions.offset((*pi).resno as isize) as *mut opj_pi_resolution_t;
            levelno = (*comp).numresolutions - 1 - (*pi).resno;

            if (((*comp).dx << levelno as u64) >> levelno) as u32 != (*comp).dx
              || (((*comp).dy << levelno as u64) >> levelno) as u32 != (*comp).dy
            {
              current_block = 15512526488502093901;
              continue;
            }

            trx0 = opj_uint64_ceildiv_res_uint32((*pi).tx0 as u64, ((*comp).dx as u64) << levelno);
            try0 = opj_uint64_ceildiv_res_uint32((*pi).ty0 as u64, ((*comp).dy as u64) << levelno);
            trx1 = opj_uint64_ceildiv_res_uint32((*pi).tx1 as u64, ((*comp).dx as u64) << levelno);
            try1 = opj_uint64_ceildiv_res_uint32((*pi).ty1 as u64, ((*comp).dy as u64) << levelno);
            rpx = (*res).pdx.wrapping_add(levelno);
            rpy = (*res).pdy.wrapping_add(levelno);

            if (((*comp).dx << rpx) as u64 >> rpx) as u32 != (*comp).dx
              || (((*comp).dy << rpy) as u64 >> rpy) as u32 != (*comp).dy
            {
              current_block = 15512526488502093901;
              continue;
            }
            /* See ISO-15441. B.12.1.4 Position-component-resolution level-layer progression */
            if !(((*pi).y as u64 % (((*comp).dy as u64) << rpy)) == 0
              || ((*pi).y == (*pi).ty0 && (((try0 as u64) << levelno) % (1u64 << rpy)) != 0))
            {
              current_block = 15512526488502093901;
              continue;
            }
            if !(((*pi).x as u64 % (((*comp).dx as u64) << rpx)) == 0
              || ((*pi).x == (*pi).tx0 && (((trx0 as u64) << levelno) % (1u64 << rpx)) != 0))
            {
              current_block = 15512526488502093901;
              continue;
            }
            if (*res).pw == 0u32 || (*res).ph == 0u32 {
              current_block = 15512526488502093901;
              continue;
            }
            if trx0 == trx1 || try0 == try1 {
              current_block = 15512526488502093901;
              continue;
            }
            prci = opj_uint_floordivpow2(
              opj_uint64_ceildiv_res_uint32((*pi).x as u64, ((*comp).dx as u64) << levelno),
              (*res).pdx,
            ) - opj_uint_floordivpow2(trx0, (*res).pdx);
            prcj = opj_uint_floordivpow2(
              opj_uint64_ceildiv_res_uint32((*pi).y as u64, ((*comp).dy as u64) << levelno),
              (*res).pdy,
            ) - opj_uint_floordivpow2(try0, (*res).pdy);
            (*pi).precno = prci.wrapping_add(prcj.wrapping_mul((*res).pw));
            (*pi).layno = (*pi).poc.layno0;
            current_block = 6281126495347172768;
            break;
          } else {
            (*pi).compno = (*pi).compno.wrapping_add(1);
            current_block = 980989089337379490;
          }
        }
      }
    }
    0i32
  }
}
/* *
Get next packet in component-precinct-resolution-layer order.
@param pi packet iterator to modify
@return returns false if pi pointed to the last packet or else returns true
*/
fn opj_pi_next_cprl(mut pi: *mut opj_pi_iterator_t) -> OPJ_BOOL {
  unsafe {
    let mut resno: OPJ_UINT32 = 0;
    let mut levelno: OPJ_UINT32 = 0;
    let mut trx0: OPJ_UINT32 = 0;
    let mut try0: OPJ_UINT32 = 0;
    let mut trx1: OPJ_UINT32 = 0;
    let mut try1: OPJ_UINT32 = 0;
    let mut rpx: OPJ_UINT32 = 0;
    let mut rpy: OPJ_UINT32 = 0;
    let mut prci: OPJ_UINT32 = 0;
    let mut prcj: OPJ_UINT32 = 0;
    let mut current_block: u64;
    let mut comp = core::ptr::null_mut::<opj_pi_comp_t>();
    let mut res = core::ptr::null_mut::<opj_pi_resolution_t>();
    let mut index = 0 as OPJ_UINT32;
    if (*pi).poc.compno0 >= (*pi).numcomps || (*pi).poc.compno1 >= (*pi).numcomps.wrapping_add(1u32)
    {
      event_msg!(
        (*pi).manager,
        EVT_ERROR,
        "opj_pi_next_cprl(): invalid compno0/compno1\n",
      );
      return 0i32;
    }
    if (*pi).first == 0 {
      comp = &mut *(*pi).comps.offset((*pi).compno as isize) as *mut opj_pi_comp_t;
      current_block = 13707613154239713890;
    } else {
      (*pi).first = 0i32;
      (*pi).compno = (*pi).poc.compno0;
      current_block = 3640593987805443782;
    }
    loop {
      match current_block {
        13707613154239713890 => {
          (*pi).layno = (*pi).layno.wrapping_add(1);
          current_block = 15594839951440953787;
        }
        _ => {
          if (*pi).compno >= (*pi).poc.compno1 {
            break;
          }
          resno = 0;
          comp = &mut *(*pi).comps.offset((*pi).compno as isize) as *mut opj_pi_comp_t;
          (*pi).dx = 0 as OPJ_UINT32;
          (*pi).dy = 0 as OPJ_UINT32;
          resno = 0 as OPJ_UINT32;
          while resno < (*comp).numresolutions {
            let mut dx: OPJ_UINT32 = 0;
            let mut dy: OPJ_UINT32 = 0;
            res = &mut *(*comp).resolutions.offset(resno as isize) as *mut opj_pi_resolution_t;
            if (*res)
              .pdx
              .wrapping_add((*comp).numresolutions)
              .wrapping_sub(1u32)
              .wrapping_sub(resno)
              < 32u32
              && (*comp).dx
                <= (2147483647u32)
                  .wrapping_mul(2u32)
                  .wrapping_add(1u32)
                  .wrapping_div(
                    (1u32)
                      << (*res)
                        .pdx
                        .wrapping_add((*comp).numresolutions)
                        .wrapping_sub(1u32)
                        .wrapping_sub(resno),
                  )
            {
              dx = (*comp).dx.wrapping_mul(
                (1u32)
                  << (*res)
                    .pdx
                    .wrapping_add((*comp).numresolutions)
                    .wrapping_sub(1u32)
                    .wrapping_sub(resno),
              );
              (*pi).dx = if (*pi).dx == 0 {
                dx
              } else {
                opj_uint_min((*pi).dx, dx)
              }
            }
            if (*res)
              .pdy
              .wrapping_add((*comp).numresolutions)
              .wrapping_sub(1u32)
              .wrapping_sub(resno)
              < 32u32
              && (*comp).dy
                <= (2147483647u32)
                  .wrapping_mul(2u32)
                  .wrapping_add(1u32)
                  .wrapping_div(
                    (1u32)
                      << (*res)
                        .pdy
                        .wrapping_add((*comp).numresolutions)
                        .wrapping_sub(1u32)
                        .wrapping_sub(resno),
                  )
            {
              dy = (*comp).dy.wrapping_mul(
                (1u32)
                  << (*res)
                    .pdy
                    .wrapping_add((*comp).numresolutions)
                    .wrapping_sub(1u32)
                    .wrapping_sub(resno),
              );
              (*pi).dy = if (*pi).dy == 0 {
                dy
              } else {
                opj_uint_min((*pi).dy, dy)
              }
            }
            resno += 1;
          }
          if (*pi).dx == 0u32 || (*pi).dy == 0u32 {
            return 0i32;
          }
          if (*pi).tp_on == 0 {
            (*pi).poc.ty0 = (*pi).ty0;
            (*pi).poc.tx0 = (*pi).tx0;
            (*pi).poc.ty1 = (*pi).ty1;
            (*pi).poc.tx1 = (*pi).tx1
          }
          (*pi).y = (*pi).poc.ty0;
          current_block = 18153031941552419006;
        }
      }
      loop {
        match current_block {
          18153031941552419006 => {
            if (*pi).y < (*pi).poc.ty1 {
              (*pi).x = (*pi).poc.tx0
            } else {
              (*pi).compno = (*pi).compno.wrapping_add(1);
              current_block = 3640593987805443782;
              break;
            }
            current_block = 10692455896603418738;
          }
          _ => {
            if (*pi).layno < (*pi).poc.layno1 {
              index = (*pi)
                .layno
                .wrapping_mul((*pi).step_l)
                .wrapping_add((*pi).resno.wrapping_mul((*pi).step_r))
                .wrapping_add((*pi).compno.wrapping_mul((*pi).step_c))
                .wrapping_add((*pi).precno.wrapping_mul((*pi).step_p));
              if index >= (*pi).include_size {
                event_msg!((*pi).manager, EVT_ERROR, "Invalid access to pi->include",);
                return 0i32;
              }
              if *(*pi).include.offset(index as isize) == 0 {
                *(*pi).include.offset(index as isize) = 1 as OPJ_INT16;
                return 1i32;
              }
              current_block = 13707613154239713890;
              break;
            } else {
              current_block = 3123434771885419771;
            }
          }
        }
        loop {
          match current_block {
            3123434771885419771 => (*pi).resno = (*pi).resno.wrapping_add(1),
            _ => {
              if (*pi).x < (*pi).poc.tx1 {
                (*pi).resno = (*pi).poc.resno0
              } else {
                (*pi).y = ((*pi).y as core::ffi::c_uint)
                  .wrapping_add((*pi).dy.wrapping_sub((*pi).y.wrapping_rem((*pi).dy)))
                  as OPJ_UINT32;
                current_block = 18153031941552419006;
                break;
              }
            }
          }
          if (*pi).resno < opj_uint_min((*pi).poc.resno1, (*comp).numresolutions) {
            levelno = 0;
            trx0 = 0;
            try0 = 0;
            trx1 = 0;
            try1 = 0;
            rpx = 0;
            rpy = 0;
            prci = 0;
            prcj = 0;
            res =
              &mut *(*comp).resolutions.offset((*pi).resno as isize) as *mut opj_pi_resolution_t;
            levelno = (*comp).numresolutions - 1 - (*pi).resno;

            if (((*comp).dx << levelno as u64) >> levelno) as u32 != (*comp).dx
              || (((*comp).dy << levelno as u64) >> levelno) as u32 != (*comp).dy
            {
              current_block = 3123434771885419771;
              continue;
            }

            trx0 = opj_uint64_ceildiv_res_uint32((*pi).tx0 as u64, ((*comp).dx as u64) << levelno);
            try0 = opj_uint64_ceildiv_res_uint32((*pi).ty0 as u64, ((*comp).dy as u64) << levelno);
            trx1 = opj_uint64_ceildiv_res_uint32((*pi).tx1 as u64, ((*comp).dx as u64) << levelno);
            try1 = opj_uint64_ceildiv_res_uint32((*pi).ty1 as u64, ((*comp).dy as u64) << levelno);
            rpx = (*res).pdx.wrapping_add(levelno);
            rpy = (*res).pdy.wrapping_add(levelno);

            if (((*comp).dx << rpx) as u64 >> rpx) as u32 != (*comp).dx
              || (((*comp).dy << rpy) as u64 >> rpy) as u32 != (*comp).dy
            {
              current_block = 3123434771885419771;
              continue;
            }
            /* See ISO-15441. B.12.1.5 Component-position-resolution level-layer progression */
            if !(((*pi).y as u64 % (((*comp).dy as u64) << rpy)) == 0
              || ((*pi).y == (*pi).ty0 && (((try0 as u64) << levelno) % (1u64 << rpy)) != 0))
            {
              current_block = 3123434771885419771;
              continue;
            }
            if !(((*pi).x as u64 % (((*comp).dx as u64) << rpx)) == 0
              || ((*pi).x == (*pi).tx0 && (((trx0 as u64) << levelno) % (1u64 << rpx)) != 0))
            {
              current_block = 3123434771885419771;
              continue;
            }
            if (*res).pw == 0u32 || (*res).ph == 0u32 {
              current_block = 3123434771885419771;
              continue;
            }
            if trx0 == trx1 || try0 == try1 {
              current_block = 3123434771885419771;
              continue;
            }
            prci = opj_uint_floordivpow2(
              opj_uint64_ceildiv_res_uint32((*pi).x as u64, ((*comp).dx as u64) << levelno),
              (*res).pdx,
            ) - opj_uint_floordivpow2(trx0, (*res).pdx);
            prcj = opj_uint_floordivpow2(
              opj_uint64_ceildiv_res_uint32((*pi).y as u64, ((*comp).dy as u64) << levelno),
              (*res).pdy,
            ) - opj_uint_floordivpow2(try0, (*res).pdy);
            (*pi).precno = prci.wrapping_add(prcj.wrapping_mul((*res).pw));
            (*pi).layno = (*pi).poc.layno0;
            current_block = 15594839951440953787;
            break;
          } else {
            (*pi).x = ((*pi).x as core::ffi::c_uint)
              .wrapping_add((*pi).dx.wrapping_sub((*pi).x.wrapping_rem((*pi).dx)))
              as OPJ_UINT32;
            current_block = 10692455896603418738;
          }
        }
      }
    }
    0i32
  }
}

/* *
 * Gets the encoding parameters needed to update the coding parameters and all the pocs.
 *
 * @param   p_image         the image being encoded.
 * @param   p_cp            the coding parameters.
 * @param   tileno          the tile index of the tile being encoded.
 * @param   p_tx0           pointer that will hold the X0 parameter for the tile
 * @param   p_tx1           pointer that will hold the X1 parameter for the tile
 * @param   p_ty0           pointer that will hold the Y0 parameter for the tile
 * @param   p_ty1           pointer that will hold the Y1 parameter for the tile
 * @param   p_max_prec      pointer that will hold the maximum precision for all the bands of the tile
 * @param   p_max_res       pointer that will hold the maximum number of resolutions for all the poc inside the tile.
 * @param   p_dx_min            pointer that will hold the minimum dx of all the components of all the resolutions for the tile.
 * @param   p_dy_min            pointer that will hold the minimum dy of all the components of all the resolutions for the tile.
 */
pub(crate) fn opj_get_encoding_parameters(
  mut p_image: *const opj_image_t,
  mut p_cp: *const opj_cp_t,
  mut p_tileno: OPJ_UINT32,
  mut p_tx0: *mut OPJ_UINT32,
  mut p_tx1: *mut OPJ_UINT32,
  mut p_ty0: *mut OPJ_UINT32,
  mut p_ty1: *mut OPJ_UINT32,
  mut p_dx_min: *mut OPJ_UINT32,
  mut p_dy_min: *mut OPJ_UINT32,
  mut p_max_prec: *mut OPJ_UINT32,
  mut p_max_res: *mut OPJ_UINT32,
) {
  unsafe {
    /* loop */
    let mut compno: OPJ_UINT32 = 0;
    let mut resno: OPJ_UINT32 = 0;
    /* pointers */
    let mut l_tcp = core::ptr::null::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null::<opj_tccp_t>();
    let mut l_img_comp = core::ptr::null::<opj_image_comp_t>();
    /* position in x and y of tile */
    let mut p: OPJ_UINT32 = 0;
    let mut q: OPJ_UINT32 = 0;
    /* non-corrected (in regard to image offset) tile offset */
    let mut l_tx0: OPJ_UINT32 = 0;
    let mut l_ty0: OPJ_UINT32 = 0;
    /* preconditions */

    assert!(!p_cp.is_null());
    assert!(!p_image.is_null());
    assert!(p_tileno < (*p_cp).tw.wrapping_mul((*p_cp).th));
    /* initializations */
    l_tcp = &mut *(*p_cp).tcps.offset(p_tileno as isize) as *mut opj_tcp_t;
    l_img_comp = (*p_image).comps;
    l_tccp = (*l_tcp).tccps;
    /* here calculation of tx0, tx1, ty0, ty1, maxprec, dx and dy */
    p = p_tileno.wrapping_rem((*p_cp).tw);
    q = p_tileno.wrapping_div((*p_cp).tw);
    /* find extent of tile */
    l_tx0 = (*p_cp).tx0.wrapping_add(p.wrapping_mul((*p_cp).tdx)); /* can't be greater than p_image->x1 so won't overflow */
    *p_tx0 = opj_uint_max(l_tx0, (*p_image).x0); /* can't be greater than p_image->y1 so won't overflow */
    *p_tx1 = opj_uint_min(opj_uint_adds(l_tx0, (*p_cp).tdx), (*p_image).x1);
    l_ty0 = (*p_cp).ty0.wrapping_add(q.wrapping_mul((*p_cp).tdy));
    *p_ty0 = opj_uint_max(l_ty0, (*p_image).y0);
    *p_ty1 = opj_uint_min(opj_uint_adds(l_ty0, (*p_cp).tdy), (*p_image).y1);
    /* max precision is 0 (can only grow) */
    *p_max_prec = 0 as OPJ_UINT32;
    *p_max_res = 0 as OPJ_UINT32;
    /* take the largest value for dx_min and dy_min */
    *p_dx_min = 0x7fffffff as OPJ_UINT32;
    *p_dy_min = 0x7fffffff as OPJ_UINT32;
    compno = 0 as OPJ_UINT32;
    while compno < (*p_image).numcomps {
      /* arithmetic variables to calculate */
      let mut l_level_no: OPJ_UINT32 = 0;
      let mut l_rx0: OPJ_UINT32 = 0;
      let mut l_ry0: OPJ_UINT32 = 0;
      let mut l_rx1: OPJ_UINT32 = 0;
      let mut l_ry1: OPJ_UINT32 = 0;
      let mut l_px0: OPJ_UINT32 = 0;
      let mut l_py0: OPJ_UINT32 = 0;
      let mut l_px1: OPJ_UINT32 = 0;
      let mut py1: OPJ_UINT32 = 0;
      let mut l_pdx: OPJ_UINT32 = 0;
      let mut l_pdy: OPJ_UINT32 = 0;
      let mut l_pw: OPJ_UINT32 = 0;
      let mut l_ph: OPJ_UINT32 = 0;
      let mut l_product: OPJ_UINT32 = 0;
      let mut l_tcx0: OPJ_UINT32 = 0;
      let mut l_tcy0: OPJ_UINT32 = 0;
      let mut l_tcx1: OPJ_UINT32 = 0;
      let mut l_tcy1: OPJ_UINT32 = 0;
      l_tcx0 = opj_uint_ceildiv(*p_tx0, (*l_img_comp).dx);
      l_tcy0 = opj_uint_ceildiv(*p_ty0, (*l_img_comp).dy);
      l_tcx1 = opj_uint_ceildiv(*p_tx1, (*l_img_comp).dx);
      l_tcy1 = opj_uint_ceildiv(*p_ty1, (*l_img_comp).dy);
      if (*l_tccp).numresolutions > *p_max_res {
        *p_max_res = (*l_tccp).numresolutions
      }
      /* use custom size for precincts */
      resno = 0 as OPJ_UINT32;
      while resno < (*l_tccp).numresolutions {
        let mut l_dx: OPJ_UINT64 = 0;
        let mut l_dy: OPJ_UINT64 = 0;

        /* precinct width and height */
        l_pdx = (*l_tccp).prcw[resno as usize];
        l_pdy = (*l_tccp).prch[resno as usize];

        l_dx = (*l_img_comp).dx as u64 * ((1u64) << (l_pdx + (*l_tccp).numresolutions - 1 - resno));
        l_dy =
          (*l_img_comp).dy as u64 * ((1u64) << (l_pdy + (*l_tccp).numresolutions - 1u32 - resno));
        /* take the minimum size for dx for each comp and resolution */
        if l_dx < u32::MAX as u64 {
          *p_dx_min = opj_uint_min(*p_dx_min, l_dx as u32);
        }
        if l_dy < u32::MAX as u64 {
          *p_dy_min = opj_uint_min(*p_dy_min, l_dy as u32);
        }
        /* various calculations of extents */
        l_level_no = (*l_tccp)
          .numresolutions
          .wrapping_sub(1u32)
          .wrapping_sub(resno);
        l_rx0 = opj_uint_ceildivpow2(l_tcx0, l_level_no);
        l_ry0 = opj_uint_ceildivpow2(l_tcy0, l_level_no);
        l_rx1 = opj_uint_ceildivpow2(l_tcx1, l_level_no);
        l_ry1 = opj_uint_ceildivpow2(l_tcy1, l_level_no);
        l_px0 = opj_uint_floordivpow2(l_rx0, l_pdx) << l_pdx;
        l_py0 = opj_uint_floordivpow2(l_ry0, l_pdy) << l_pdy;
        l_px1 = opj_uint_ceildivpow2(l_rx1, l_pdx) << l_pdx;
        py1 = opj_uint_ceildivpow2(l_ry1, l_pdy) << l_pdy;
        l_pw = if l_rx0 == l_rx1 {
          0u32
        } else {
          (l_px1.wrapping_sub(l_px0)) >> l_pdx
        };
        l_ph = if l_ry0 == l_ry1 {
          0u32
        } else {
          (py1.wrapping_sub(l_py0)) >> l_pdy
        };
        l_product = l_pw.wrapping_mul(l_ph);
        /* update precision */
        if l_product > *p_max_prec {
          *p_max_prec = l_product
        }
        resno += 1;
      }
      l_img_comp = l_img_comp.offset(1);
      l_tccp = l_tccp.offset(1);
      compno += 1;
    }
  }
}
/* *
 * Gets the encoding parameters needed to update the coding parameters and all the pocs.
 * The precinct widths, heights, dx and dy for each component at each resolution will be stored as well.
 * the last parameter of the function should be an array of pointers of size nb components, each pointer leading
 * to an area of size 4 * max_res. The data is stored inside this area with the following pattern :
 * dx_compi_res0 , dy_compi_res0 , w_compi_res0, h_compi_res0 , dx_compi_res1 , dy_compi_res1 , w_compi_res1, h_compi_res1 , ...
 *
 * @param   p_image         the image being encoded.
 * @param   p_cp            the coding parameters.
 * @param   tileno          the tile index of the tile being encoded.
 * @param   p_tx0           pointer that will hold the X0 parameter for the tile
 * @param   p_tx1           pointer that will hold the X1 parameter for the tile
 * @param   p_ty0           pointer that will hold the Y0 parameter for the tile
 * @param   p_ty1           pointer that will hold the Y1 parameter for the tile
 * @param   p_max_prec      pointer that will hold the maximum precision for all the bands of the tile
 * @param   p_max_res       pointer that will hold the maximum number of resolutions for all the poc inside the tile.
 * @param   p_dx_min        pointer that will hold the minimum dx of all the components of all the resolutions for the tile.
 * @param   p_dy_min        pointer that will hold the minimum dy of all the components of all the resolutions for the tile.
 * @param   p_resolutions   pointer to an area corresponding to the one described above.
 */
pub(crate) fn opj_get_all_encoding_parameters(
  mut p_image: *const opj_image_t,
  mut p_cp: *const opj_cp_t,
  mut tileno: OPJ_UINT32,
  mut p_tx0: *mut OPJ_UINT32,
  mut p_tx1: *mut OPJ_UINT32,
  mut p_ty0: *mut OPJ_UINT32,
  mut p_ty1: *mut OPJ_UINT32,
  mut p_dx_min: *mut OPJ_UINT32,
  mut p_dy_min: *mut OPJ_UINT32,
  mut p_max_prec: *mut OPJ_UINT32,
  mut p_max_res: *mut OPJ_UINT32,
  mut p_resolutions: *mut *mut OPJ_UINT32,
) {
  unsafe {
    /* loop*/
    let mut compno: OPJ_UINT32 = 0;
    let mut resno: OPJ_UINT32 = 0;
    /* pointers*/
    let mut tcp = core::ptr::null::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null::<opj_tccp_t>();
    let mut l_img_comp = core::ptr::null::<opj_image_comp_t>();
    /* to store l_dx, l_dy, w and h for each resolution and component.*/
    let mut lResolutionPtr = core::ptr::null_mut::<OPJ_UINT32>();
    /* position in x and y of tile*/
    let mut p: OPJ_UINT32 = 0;
    let mut q: OPJ_UINT32 = 0;
    /* non-corrected (in regard to image offset) tile offset */
    let mut l_tx0: OPJ_UINT32 = 0;
    let mut l_ty0: OPJ_UINT32 = 0;
    /* preconditions in debug*/

    assert!(!p_cp.is_null());
    assert!(!p_image.is_null());
    assert!(tileno < (*p_cp).tw.wrapping_mul((*p_cp).th));
    /* initializations*/
    tcp = &mut *(*p_cp).tcps.offset(tileno as isize) as *mut opj_tcp_t;
    l_tccp = (*tcp).tccps;
    l_img_comp = (*p_image).comps;
    /* position in x and y of tile*/
    p = tileno.wrapping_rem((*p_cp).tw);
    q = tileno.wrapping_div((*p_cp).tw);
    /* here calculation of tx0, tx1, ty0, ty1, maxprec, l_dx and l_dy */
    l_tx0 = (*p_cp).tx0.wrapping_add(p.wrapping_mul((*p_cp).tdx)); /* can't be greater than p_image->x1 so won't overflow */
    *p_tx0 = opj_uint_max(l_tx0, (*p_image).x0); /* can't be greater than p_image->y1 so won't overflow */
    *p_tx1 = opj_uint_min(opj_uint_adds(l_tx0, (*p_cp).tdx), (*p_image).x1);
    l_ty0 = (*p_cp).ty0.wrapping_add(q.wrapping_mul((*p_cp).tdy));
    *p_ty0 = opj_uint_max(l_ty0, (*p_image).y0);
    *p_ty1 = opj_uint_min(opj_uint_adds(l_ty0, (*p_cp).tdy), (*p_image).y1);
    /* max precision and resolution is 0 (can only grow)*/
    *p_max_prec = 0 as OPJ_UINT32;
    *p_max_res = 0 as OPJ_UINT32;
    /* take the largest value for dx_min and dy_min*/
    *p_dx_min = 0x7fffffff as OPJ_UINT32;
    *p_dy_min = 0x7fffffff as OPJ_UINT32;
    compno = 0 as OPJ_UINT32;
    while compno < (*p_image).numcomps {
      /* arithmetic variables to calculate*/
      let mut l_level_no: OPJ_UINT32 = 0;
      let mut l_rx0: OPJ_UINT32 = 0;
      let mut l_ry0: OPJ_UINT32 = 0;
      let mut l_rx1: OPJ_UINT32 = 0;
      let mut l_ry1: OPJ_UINT32 = 0;
      let mut l_px0: OPJ_UINT32 = 0;
      let mut l_py0: OPJ_UINT32 = 0;
      let mut l_px1: OPJ_UINT32 = 0;
      let mut py1: OPJ_UINT32 = 0;
      let mut l_product: OPJ_UINT32 = 0;
      let mut l_tcx0: OPJ_UINT32 = 0;
      let mut l_tcy0: OPJ_UINT32 = 0;
      let mut l_tcx1: OPJ_UINT32 = 0;
      let mut l_tcy1: OPJ_UINT32 = 0;
      let mut l_pdx: OPJ_UINT32 = 0;
      let mut l_pdy: OPJ_UINT32 = 0;
      let mut l_pw: OPJ_UINT32 = 0;
      let mut l_ph: OPJ_UINT32 = 0;
      lResolutionPtr = if !p_resolutions.is_null() {
        *p_resolutions.offset(compno as isize)
      } else {
        core::ptr::null_mut::<OPJ_UINT32>()
      };
      l_tcx0 = opj_uint_ceildiv(*p_tx0, (*l_img_comp).dx);
      l_tcy0 = opj_uint_ceildiv(*p_ty0, (*l_img_comp).dy);
      l_tcx1 = opj_uint_ceildiv(*p_tx1, (*l_img_comp).dx);
      l_tcy1 = opj_uint_ceildiv(*p_ty1, (*l_img_comp).dy);
      if (*l_tccp).numresolutions > *p_max_res {
        *p_max_res = (*l_tccp).numresolutions
      }
      /* use custom size for precincts*/
      l_level_no = (*l_tccp).numresolutions;
      resno = 0 as OPJ_UINT32;
      while resno < (*l_tccp).numresolutions {
        let mut l_dx: OPJ_UINT32 = 0;
        let mut l_dy: OPJ_UINT32 = 0;
        l_level_no = l_level_no.wrapping_sub(1);
        /* precinct width and height*/
        l_pdx = (*l_tccp).prcw[resno as usize];
        l_pdy = (*l_tccp).prch[resno as usize];
        if !lResolutionPtr.is_null() {
          let fresh0 = lResolutionPtr;
          lResolutionPtr = lResolutionPtr.offset(1);
          *fresh0 = l_pdx;
          let fresh1 = lResolutionPtr;
          lResolutionPtr = lResolutionPtr.offset(1);
          *fresh1 = l_pdy
        }
        if l_pdx.wrapping_add(l_level_no) < 32u32
          && (*l_img_comp).dx
            <= (2147483647u32)
              .wrapping_mul(2u32)
              .wrapping_add(1u32)
              .wrapping_div((1u32) << l_pdx.wrapping_add(l_level_no))
        {
          l_dx = (*l_img_comp)
            .dx
            .wrapping_mul((1u32) << l_pdx.wrapping_add(l_level_no));
          /* take the minimum size for l_dx for each comp and resolution*/
          *p_dx_min = opj_uint_min(*p_dx_min, l_dx)
        }
        if l_pdy.wrapping_add(l_level_no) < 32u32
          && (*l_img_comp).dy
            <= (2147483647u32)
              .wrapping_mul(2u32)
              .wrapping_add(1u32)
              .wrapping_div((1u32) << l_pdy.wrapping_add(l_level_no))
        {
          l_dy = (*l_img_comp)
            .dy
            .wrapping_mul((1u32) << l_pdy.wrapping_add(l_level_no));
          *p_dy_min = opj_uint_min(*p_dy_min, l_dy)
        }
        /* various calculations of extents*/
        l_rx0 = opj_uint_ceildivpow2(l_tcx0, l_level_no);
        l_ry0 = opj_uint_ceildivpow2(l_tcy0, l_level_no);
        l_rx1 = opj_uint_ceildivpow2(l_tcx1, l_level_no);
        l_ry1 = opj_uint_ceildivpow2(l_tcy1, l_level_no);
        l_px0 = opj_uint_floordivpow2(l_rx0, l_pdx) << l_pdx;
        l_py0 = opj_uint_floordivpow2(l_ry0, l_pdy) << l_pdy;
        l_px1 = opj_uint_ceildivpow2(l_rx1, l_pdx) << l_pdx;
        py1 = opj_uint_ceildivpow2(l_ry1, l_pdy) << l_pdy;
        l_pw = if l_rx0 == l_rx1 {
          0u32
        } else {
          (l_px1.wrapping_sub(l_px0)) >> l_pdx
        };
        l_ph = if l_ry0 == l_ry1 {
          0u32
        } else {
          (py1.wrapping_sub(l_py0)) >> l_pdy
        };
        if !lResolutionPtr.is_null() {
          let fresh2 = lResolutionPtr;
          lResolutionPtr = lResolutionPtr.offset(1);
          *fresh2 = l_pw;
          let fresh3 = lResolutionPtr;
          lResolutionPtr = lResolutionPtr.offset(1);
          *fresh3 = l_ph
        }
        l_product = l_pw.wrapping_mul(l_ph);
        /* update precision*/
        if l_product > *p_max_prec {
          *p_max_prec = l_product
        }
        resno += 1;
      }
      l_tccp = l_tccp.offset(1);
      l_img_comp = l_img_comp.offset(1);
      compno += 1;
    }
  }
}
/* *
 * Allocates memory for a packet iterator. Data and data sizes are set by this operation.
 * No other data is set. The include section of the packet  iterator is not allocated.
 *
 * @param   p_image     the image used to initialize the packet iterator (in fact only the number of components is relevant.
 * @param   p_cp        the coding parameters.
 * @param   tileno  the index of the tile from which creating the packet iterator.
 * @param   manager Event manager
 */
pub(crate) fn opj_pi_create(
  mut image: *const opj_image_t,
  mut cp: *const opj_cp_t,
  mut tileno: OPJ_UINT32,
  mut manager: &mut opj_event_mgr,
) -> *mut opj_pi_iterator_t {
  unsafe {
    /* loop*/
    let mut pino: OPJ_UINT32 = 0;
    let mut compno: OPJ_UINT32 = 0;
    /* number of poc in the p_pi*/
    let mut l_poc_bound: OPJ_UINT32 = 0;
    /* pointers to tile coding parameters and components.*/
    let mut l_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut tccp = core::ptr::null::<opj_tccp_t>();
    /* current packet iterator being allocated*/
    let mut l_current_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    /* preconditions in debug*/

    assert!(!cp.is_null());
    assert!(!image.is_null());
    assert!(tileno < (*cp).tw.wrapping_mul((*cp).th));
    /* initializations*/
    tcp = &mut *(*cp).tcps.offset(tileno as isize) as *mut opj_tcp_t;
    l_poc_bound = (*tcp).numpocs.wrapping_add(1u32);
    /* memory allocations*/
    l_pi = opj_calloc(
      l_poc_bound as size_t,
      core::mem::size_of::<opj_pi_iterator_t>(),
    ) as *mut opj_pi_iterator_t;
    if l_pi.is_null() {
      return core::ptr::null_mut::<opj_pi_iterator_t>();
    }
    l_current_pi = l_pi;
    pino = 0 as OPJ_UINT32;
    while pino < l_poc_bound {
      (*l_current_pi).manager = *manager;
      (*l_current_pi).comps = opj_calloc(
        (*image).numcomps as size_t,
        core::mem::size_of::<opj_pi_comp_t>(),
      ) as *mut opj_pi_comp_t;
      if (*l_current_pi).comps.is_null() {
        opj_pi_destroy(l_pi, l_poc_bound);
        return core::ptr::null_mut::<opj_pi_iterator_t>();
      }
      (*l_current_pi).numcomps = (*image).numcomps;
      compno = 0 as OPJ_UINT32;
      while compno < (*image).numcomps {
        let mut comp: *mut opj_pi_comp_t =
          &mut *(*l_current_pi).comps.offset(compno as isize) as *mut opj_pi_comp_t;
        tccp = &mut *(*tcp).tccps.offset(compno as isize) as *mut opj_tccp_t;
        (*comp).resolutions = opj_calloc(
          (*tccp).numresolutions as size_t,
          core::mem::size_of::<opj_pi_resolution_t>(),
        ) as *mut opj_pi_resolution_t;
        if (*comp).resolutions.is_null() {
          opj_pi_destroy(l_pi, l_poc_bound);
          return core::ptr::null_mut::<opj_pi_iterator_t>();
        }
        (*comp).numresolutions = (*tccp).numresolutions;
        compno += 1;
      }
      l_current_pi = l_current_pi.offset(1);
      pino += 1;
    }
    l_pi
  }
}
/* *
 * Updates the coding parameters if the encoding is used with Progression order changes and final (or cinema parameters are used).
 *
 * @param   p_cp        the coding parameters to modify
 * @param   p_tileno    the tile index being concerned.
 * @param   p_tx0       X0 parameter for the tile
 * @param   p_tx1       X1 parameter for the tile
 * @param   p_ty0       Y0 parameter for the tile
 * @param   p_ty1       Y1 parameter for the tile
 * @param   p_max_prec  the maximum precision for all the bands of the tile
 * @param   _p_max_res  the maximum number of resolutions for all the poc inside the tile.
 * @param   p_dx_min        the minimum dx of all the components of all the resolutions for the tile.
 * @param   p_dy_min        the minimum dy of all the components of all the resolutions for the tile.
 */
pub(crate) fn opj_pi_update_encode_poc_and_final(
  mut p_cp: *mut opj_cp_t,
  mut p_tileno: OPJ_UINT32,
  mut p_tx0: OPJ_UINT32,
  mut p_tx1: OPJ_UINT32,
  mut p_ty0: OPJ_UINT32,
  mut p_ty1: OPJ_UINT32,
  mut p_max_prec: OPJ_UINT32,
  mut _p_max_res: OPJ_UINT32,
  mut p_dx_min: OPJ_UINT32,
  mut p_dy_min: OPJ_UINT32,
) {
  unsafe {
    /* loop*/
    let mut pino: OPJ_UINT32 = 0;
    /* tile coding parameter*/
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    /* current poc being updated*/
    let mut l_current_poc = core::ptr::null_mut::<opj_poc_t>();
    /* number of pocs*/
    let mut l_poc_bound: OPJ_UINT32 = 0;
    /* preconditions in debug*/

    assert!(!p_cp.is_null());
    assert!(p_tileno < (*p_cp).tw.wrapping_mul((*p_cp).th));
    /* initializations*/
    l_tcp = &mut *(*p_cp).tcps.offset(p_tileno as isize) as *mut opj_tcp_t;
    /* number of iterations in the loop */
    l_poc_bound = (*l_tcp).numpocs.wrapping_add(1u32);
    /* start at first element, and to make sure the compiler will not make a calculation each time in the loop
    store a pointer to the current element to modify rather than l_tcp->pocs[i]*/
    l_current_poc = (*l_tcp).pocs.as_mut_ptr();
    (*l_current_poc).compS = (*l_current_poc).compno0;
    (*l_current_poc).compE = (*l_current_poc).compno1;
    (*l_current_poc).resS = (*l_current_poc).resno0;
    (*l_current_poc).resE = (*l_current_poc).resno1;
    (*l_current_poc).layE = (*l_current_poc).layno1;
    /* special treatment for the first element*/
    (*l_current_poc).layS = 0 as OPJ_UINT32;
    (*l_current_poc).prg = (*l_current_poc).prg1;
    (*l_current_poc).prcS = 0 as OPJ_UINT32;
    (*l_current_poc).prcE = p_max_prec;
    (*l_current_poc).txS = p_tx0;
    (*l_current_poc).txE = p_tx1;
    (*l_current_poc).tyS = p_ty0;
    (*l_current_poc).tyE = p_ty1;
    (*l_current_poc).dx = p_dx_min;
    (*l_current_poc).dy = p_dy_min;
    l_current_poc = l_current_poc.offset(1);
    pino = 1 as OPJ_UINT32;
    while pino < l_poc_bound {
      (*l_current_poc).compS = (*l_current_poc).compno0;
      (*l_current_poc).compE = (*l_current_poc).compno1;
      (*l_current_poc).resS = (*l_current_poc).resno0;
      (*l_current_poc).resE = (*l_current_poc).resno1;
      (*l_current_poc).layE = (*l_current_poc).layno1;
      (*l_current_poc).prg = (*l_current_poc).prg1;
      (*l_current_poc).prcS = 0 as OPJ_UINT32;
      /* special treatment here different from the first element*/
      (*l_current_poc).layS = if (*l_current_poc).layE > (*l_current_poc.offset(-1)).layE {
        (*l_current_poc).layE
      } else {
        0u32
      };
      (*l_current_poc).prcE = p_max_prec;
      (*l_current_poc).txS = p_tx0;
      (*l_current_poc).txE = p_tx1;
      (*l_current_poc).tyS = p_ty0;
      (*l_current_poc).tyE = p_ty1;
      (*l_current_poc).dx = p_dx_min;
      (*l_current_poc).dy = p_dy_min;
      l_current_poc = l_current_poc.offset(1);
      pino += 1;
    }
  }
}
/* *
 * Updates the coding parameters if the encoding is not used with Progression order changes and final (and cinema parameters are used).
 *
 * @param   p_cp        the coding parameters to modify
 * @param   p_num_comps     the number of components
 * @param   p_tileno    the tile index being concerned.
 * @param   p_tx0       X0 parameter for the tile
 * @param   p_tx1       X1 parameter for the tile
 * @param   p_ty0       Y0 parameter for the tile
 * @param   p_ty1       Y1 parameter for the tile
 * @param   p_max_prec  the maximum precision for all the bands of the tile
 * @param   p_max_res   the maximum number of resolutions for all the poc inside the tile.
 * @param   p_dx_min        the minimum dx of all the components of all the resolutions for the tile.
 * @param   p_dy_min        the minimum dy of all the components of all the resolutions for the tile.
 */
pub(crate) fn opj_pi_update_encode_not_poc(
  mut p_cp: *mut opj_cp_t,
  mut p_num_comps: OPJ_UINT32,
  mut p_tileno: OPJ_UINT32,
  mut p_tx0: OPJ_UINT32,
  mut p_tx1: OPJ_UINT32,
  mut p_ty0: OPJ_UINT32,
  mut p_ty1: OPJ_UINT32,
  mut p_max_prec: OPJ_UINT32,
  mut p_max_res: OPJ_UINT32,
  mut p_dx_min: OPJ_UINT32,
  mut p_dy_min: OPJ_UINT32,
) {
  unsafe {
    /* loop*/
    let mut pino: OPJ_UINT32 = 0;
    /* tile coding parameter*/
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    /* current poc being updated*/
    let mut l_current_poc = core::ptr::null_mut::<opj_poc_t>();
    /* number of pocs*/
    let mut l_poc_bound: OPJ_UINT32 = 0;
    /* preconditions in debug*/

    assert!(!p_cp.is_null());
    assert!(p_tileno < (*p_cp).tw.wrapping_mul((*p_cp).th));
    /* initializations*/
    l_tcp = &mut *(*p_cp).tcps.offset(p_tileno as isize) as *mut opj_tcp_t;
    /* number of iterations in the loop */
    l_poc_bound = (*l_tcp).numpocs.wrapping_add(1u32);
    /* start at first element, and to make sure the compiler will not make a calculation each time in the loop
    store a pointer to the current element to modify rather than l_tcp->pocs[i]*/
    l_current_poc = (*l_tcp).pocs.as_mut_ptr(); /*p_image->numcomps;*/
    pino = 0 as OPJ_UINT32;
    while pino < l_poc_bound {
      (*l_current_poc).compS = 0 as OPJ_UINT32;
      (*l_current_poc).compE = p_num_comps;
      (*l_current_poc).resS = 0 as OPJ_UINT32;
      (*l_current_poc).resE = p_max_res;
      (*l_current_poc).layS = 0 as OPJ_UINT32;
      (*l_current_poc).layE = (*l_tcp).numlayers;
      (*l_current_poc).prg = (*l_tcp).prg;
      (*l_current_poc).prcS = 0 as OPJ_UINT32;
      (*l_current_poc).prcE = p_max_prec;
      (*l_current_poc).txS = p_tx0;
      (*l_current_poc).txE = p_tx1;
      (*l_current_poc).tyS = p_ty0;
      (*l_current_poc).tyE = p_ty1;
      (*l_current_poc).dx = p_dx_min;
      (*l_current_poc).dy = p_dy_min;
      l_current_poc = l_current_poc.offset(1);
      pino += 1;
    }
  }
}
/* *
 * FIXME DOC
 */
pub(crate) fn opj_pi_update_decode_poc(
  mut p_pi: *mut opj_pi_iterator_t,
  mut p_tcp: *mut opj_tcp_t,
  mut p_max_precision: OPJ_UINT32,
  mut _p_max_res: OPJ_UINT32,
) {
  unsafe {
    /* loop*/
    let mut pino: OPJ_UINT32 = 0;
    /* encoding parameters to set*/
    let mut l_bound: OPJ_UINT32 = 0;
    let mut l_current_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut l_current_poc = core::ptr::null_mut::<opj_poc_t>();
    /* preconditions in debug*/

    assert!(!p_pi.is_null());
    assert!(!p_tcp.is_null());
    /* initializations*/
    l_bound = (*p_tcp).numpocs.wrapping_add(1u32); /* Progression Order #0 */
    l_current_pi = p_pi; /* Resolution Level Index #0 (Start) */
    l_current_poc = (*p_tcp).pocs.as_mut_ptr(); /* Component Index #0 (Start) */
    pino = 0 as OPJ_UINT32; /* Resolution Level Index #0 (End) */
    while pino < l_bound {
      (*l_current_pi).poc.prg = (*l_current_poc).prg; /* Component Index #0 (End) */
      (*l_current_pi).first = 1i32; /* Layer Index #0 (End) */
      (*l_current_pi).poc.resno0 = (*l_current_poc).resno0;
      (*l_current_pi).poc.compno0 = (*l_current_poc).compno0;
      (*l_current_pi).poc.layno0 = 0 as OPJ_UINT32;
      (*l_current_pi).poc.precno0 = 0 as OPJ_UINT32;
      (*l_current_pi).poc.resno1 = (*l_current_poc).resno1;
      (*l_current_pi).poc.compno1 = (*l_current_poc).compno1;
      (*l_current_pi).poc.layno1 = opj_uint_min((*l_current_poc).layno1, (*p_tcp).numlayers);
      (*l_current_pi).poc.precno1 = p_max_precision;
      l_current_pi = l_current_pi.offset(1);
      l_current_poc = l_current_poc.offset(1);
      pino += 1;
    }
  }
}
/* *
 * FIXME DOC
 */
pub(crate) fn opj_pi_update_decode_not_poc(
  mut p_pi: *mut opj_pi_iterator_t,
  mut p_tcp: *mut opj_tcp_t,
  mut p_max_precision: OPJ_UINT32,
  mut p_max_res: OPJ_UINT32,
) {
  unsafe {
    /* loop*/
    let mut pino: OPJ_UINT32 = 0;
    /* encoding parameters to set*/
    let mut l_bound: OPJ_UINT32 = 0;
    let mut l_current_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    /* preconditions in debug*/

    assert!(!p_tcp.is_null());
    assert!(!p_pi.is_null());
    /* initializations*/
    l_bound = (*p_tcp).numpocs.wrapping_add(1u32);
    l_current_pi = p_pi;
    pino = 0 as OPJ_UINT32;
    while pino < l_bound {
      (*l_current_pi).poc.prg = (*p_tcp).prg;
      (*l_current_pi).first = 1i32;
      (*l_current_pi).poc.resno0 = 0 as OPJ_UINT32;
      (*l_current_pi).poc.compno0 = 0 as OPJ_UINT32;
      (*l_current_pi).poc.layno0 = 0 as OPJ_UINT32;
      (*l_current_pi).poc.precno0 = 0 as OPJ_UINT32;
      (*l_current_pi).poc.resno1 = p_max_res;
      (*l_current_pi).poc.compno1 = (*l_current_pi).numcomps;
      (*l_current_pi).poc.layno1 = (*p_tcp).numlayers;
      (*l_current_pi).poc.precno1 = p_max_precision;
      l_current_pi = l_current_pi.offset(1);
      pino += 1;
    }
  }
}
/* *
 * FIXME DOC
 */
fn opj_pi_check_next_level(
  mut pos: OPJ_INT32,
  mut cp: *mut opj_cp_t,
  mut tileno: OPJ_UINT32,
  mut pino: OPJ_UINT32,
  mut prog: ProgressionOrder,
) -> OPJ_BOOL {
  unsafe {
    let mut i: OPJ_INT32 = 0; /*end if*/
    let mut tcps: *mut opj_tcp_t = &mut *(*cp).tcps.offset(tileno as isize) as *mut opj_tcp_t;
    let mut tcp: *mut opj_poc_t =
      &mut *(*tcps).pocs.as_mut_ptr().offset(pino as isize) as *mut opj_poc_t;
    if pos >= 0i32 {
      i = pos;
      while i >= 0i32 {
        match prog.get_step(i) {
          ProgressionStep::Resolution => {
            if (*tcp).res_t == (*tcp).resE {
              if opj_pi_check_next_level(pos - 1i32, cp, tileno, pino, prog) != 0 {
                return 1i32;
              } else {
                return 0i32;
              }
            } else {
              return 1i32;
            }
            /*end case P*/
          }
          ProgressionStep::Component => {
            if (*tcp).comp_t == (*tcp).compE {
              if opj_pi_check_next_level(pos - 1i32, cp, tileno, pino, prog) != 0 {
                return 1i32;
              } else {
                return 0i32;
              }
            } else {
              return 1i32;
            }
          }
          ProgressionStep::Layer => {
            if (*tcp).lay_t == (*tcp).layE {
              if opj_pi_check_next_level(pos - 1i32, cp, tileno, pino, prog) != 0 {
                return 1i32;
              } else {
                return 0i32;
              }
            } else {
              return 1i32;
            }
          }
          ProgressionStep::Precinct => {
            match (*tcp).prg as core::ffi::c_int {
              0 | 1 => {
                /* fall through */
                if (*tcp).prc_t == (*tcp).prcE {
                  if opj_pi_check_next_level(i - 1i32, cp, tileno, pino, prog) != 0 {
                    return 1i32;
                  } else {
                    return 0i32;
                  }
                } else {
                  return 1i32;
                }
              }
              _ => {
                if (*tcp).tx0_t == (*tcp).txE {
                  /*TY*/
                  if (*tcp).ty0_t == (*tcp).tyE {
                    if opj_pi_check_next_level(i - 1i32, cp, tileno, pino, prog) != 0 {
                      return 1i32;
                    } else {
                      return 0i32;
                    }
                  } else {
                    return 1i32;
                  }
                /*TY*/
                } else {
                  return 1i32;
                }
              }
            }
          }
          _ => {}
        }
        i -= 1
        /*end switch*/
      }
      /*end for*/
    }
    0i32
  }
}
/*
==========================================================
   Packet iterator interface
==========================================================
*/

pub(crate) fn opj_pi_create_decode(
  mut p_image: *mut opj_image_t,
  mut p_cp: *mut opj_cp_t,
  mut p_tile_no: OPJ_UINT32,
  mut manager: &mut opj_event_mgr,
) -> *mut opj_pi_iterator_t {
  unsafe {
    let mut numcomps = (*p_image).numcomps;
    /* loop */
    let mut pino: OPJ_UINT32 = 0;
    let mut compno: OPJ_UINT32 = 0;
    let mut resno: OPJ_UINT32 = 0;
    /* to store w, h, dx and dy for all components and resolutions */
    let mut l_tmp_data = core::ptr::null_mut::<OPJ_UINT32>();
    let mut l_tmp_ptr = core::ptr::null_mut::<*mut OPJ_UINT32>();
    /* encoding parameters to set */
    let mut l_max_res: OPJ_UINT32 = 0;
    let mut l_max_prec: OPJ_UINT32 = 0;
    let mut l_tx0: OPJ_UINT32 = 0;
    let mut l_tx1: OPJ_UINT32 = 0;
    let mut l_ty0: OPJ_UINT32 = 0;
    let mut l_ty1: OPJ_UINT32 = 0;
    let mut l_dx_min: OPJ_UINT32 = 0;
    let mut l_dy_min: OPJ_UINT32 = 0;
    let mut l_bound: OPJ_UINT32 = 0;
    let mut l_step_p: OPJ_UINT32 = 0;
    let mut l_step_c: OPJ_UINT32 = 0;
    let mut l_step_r: OPJ_UINT32 = 0;
    let mut l_step_l: OPJ_UINT32 = 0;
    let mut l_data_stride: OPJ_UINT32 = 0;
    /* pointers */
    let mut l_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null::<opj_tccp_t>();
    let mut l_current_comp = core::ptr::null_mut::<opj_pi_comp_t>();
    let mut l_img_comp = core::ptr::null_mut::<opj_image_comp_t>();
    let mut l_current_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut l_encoding_value_ptr = core::ptr::null_mut::<OPJ_UINT32>();
    /* preconditions in debug */

    assert!(!p_cp.is_null());
    assert!(!p_image.is_null());
    assert!(p_tile_no < (*p_cp).tw.wrapping_mul((*p_cp).th));
    /* initializations */
    l_tcp = &mut *(*p_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    l_bound = (*l_tcp).numpocs.wrapping_add(1u32);
    l_data_stride = (4i32 * 33i32) as OPJ_UINT32;
    l_tmp_data = opj_malloc(
      (l_data_stride.wrapping_mul(numcomps) as usize)
        .wrapping_mul(core::mem::size_of::<OPJ_UINT32>()),
    ) as *mut OPJ_UINT32;
    if l_tmp_data.is_null() {
      return core::ptr::null_mut::<opj_pi_iterator_t>();
    }
    l_tmp_ptr =
      opj_malloc((numcomps as usize).wrapping_mul(core::mem::size_of::<*mut OPJ_UINT32>()))
        as *mut *mut OPJ_UINT32;
    if l_tmp_ptr.is_null() {
      opj_free(l_tmp_data as *mut core::ffi::c_void);
      return core::ptr::null_mut::<opj_pi_iterator_t>();
    }
    /* memory allocation for pi */
    l_pi = opj_pi_create(p_image, p_cp, p_tile_no, manager);
    if l_pi.is_null() {
      opj_free(l_tmp_data as *mut core::ffi::c_void);
      opj_free(l_tmp_ptr as *mut core::ffi::c_void);
      return core::ptr::null_mut::<opj_pi_iterator_t>();
    }
    l_encoding_value_ptr = l_tmp_data;
    /* update pointer array */
    compno = 0 as OPJ_UINT32;
    while compno < numcomps {
      let fresh4 = &mut (*l_tmp_ptr.offset(compno as isize));
      *fresh4 = l_encoding_value_ptr;
      l_encoding_value_ptr = l_encoding_value_ptr.offset(l_data_stride as isize);
      compno += 1;
    }
    /* get encoding parameters */
    opj_get_all_encoding_parameters(
      p_image,
      p_cp,
      p_tile_no,
      &mut l_tx0,
      &mut l_tx1,
      &mut l_ty0,
      &mut l_ty1,
      &mut l_dx_min,
      &mut l_dy_min,
      &mut l_max_prec,
      &mut l_max_res,
      l_tmp_ptr,
    );
    /* step calculations */
    l_step_p = 1 as OPJ_UINT32;
    l_step_c = l_max_prec.wrapping_mul(l_step_p);
    l_step_r = numcomps.wrapping_mul(l_step_c);
    l_step_l = l_max_res.wrapping_mul(l_step_r);
    /* set values for first packet iterator */
    l_current_pi = l_pi;
    /* memory allocation for include */
    /* prevent an integer overflow issue */
    /* 0 < l_tcp->numlayers < 65536 c.f. opj_j2k_read_cod in j2k.c */
    (*l_current_pi).include = core::ptr::null_mut::<OPJ_INT16>();
    if l_step_l
      <= (2147483647u32)
        .wrapping_mul(2u32)
        .wrapping_add(1u32)
        .wrapping_div((*l_tcp).numlayers.wrapping_add(1u32))
    {
      (*l_current_pi).include_size = (*l_tcp).numlayers.wrapping_add(1u32).wrapping_mul(l_step_l);
      (*l_current_pi).include = opj_calloc(
        (*l_current_pi).include_size as size_t,
        core::mem::size_of::<OPJ_INT16>(),
      ) as *mut OPJ_INT16
    }
    if (*l_current_pi).include.is_null() {
      opj_free(l_tmp_data as *mut core::ffi::c_void);
      opj_free(l_tmp_ptr as *mut core::ffi::c_void);
      opj_pi_destroy(l_pi, l_bound);
      return core::ptr::null_mut::<opj_pi_iterator_t>();
    }
    /* special treatment for the first packet iterator */
    l_current_comp = (*l_current_pi).comps;
    l_img_comp = (*p_image).comps;
    l_tccp = (*l_tcp).tccps;
    (*l_current_pi).tx0 = l_tx0;
    (*l_current_pi).ty0 = l_ty0;
    (*l_current_pi).tx1 = l_tx1;
    (*l_current_pi).ty1 = l_ty1;
    /*l_current_pi->dx = l_img_comp->dx;*/
    /*l_current_pi->dy = l_img_comp->dy;*/
    (*l_current_pi).step_p = l_step_p;
    (*l_current_pi).step_c = l_step_c;
    (*l_current_pi).step_r = l_step_r;
    (*l_current_pi).step_l = l_step_l;
    /* allocation for components and number of components has already been calculated by opj_pi_create */
    compno = 0 as OPJ_UINT32;
    while compno < numcomps {
      let mut l_res = (*l_current_comp).resolutions;
      l_encoding_value_ptr = *l_tmp_ptr.offset(compno as isize);
      (*l_current_comp).dx = (*l_img_comp).dx;
      (*l_current_comp).dy = (*l_img_comp).dy;
      /* resolutions have already been initialized */
      resno = 0 as OPJ_UINT32;
      while resno < (*l_current_comp).numresolutions {
        let fresh5 = l_encoding_value_ptr;
        l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
        (*l_res).pdx = *fresh5;
        let fresh6 = l_encoding_value_ptr;
        l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
        (*l_res).pdy = *fresh6;
        let fresh7 = l_encoding_value_ptr;
        l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
        (*l_res).pw = *fresh7;
        let fresh8 = l_encoding_value_ptr;
        l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
        (*l_res).ph = *fresh8;
        l_res = l_res.offset(1);
        resno += 1;
      }
      l_current_comp = l_current_comp.offset(1);
      l_img_comp = l_img_comp.offset(1);
      l_tccp = l_tccp.offset(1);
      compno += 1;
    }
    l_current_pi = l_current_pi.offset(1);
    pino = 1 as OPJ_UINT32;
    while pino < l_bound {
      l_current_comp = (*l_current_pi).comps;
      l_img_comp = (*p_image).comps;
      l_tccp = (*l_tcp).tccps;
      (*l_current_pi).tx0 = l_tx0;
      (*l_current_pi).ty0 = l_ty0;
      (*l_current_pi).tx1 = l_tx1;
      (*l_current_pi).ty1 = l_ty1;
      /*l_current_pi->dx = l_dx_min;*/
      /*l_current_pi->dy = l_dy_min;*/
      (*l_current_pi).step_p = l_step_p;
      (*l_current_pi).step_c = l_step_c;
      (*l_current_pi).step_r = l_step_r;
      (*l_current_pi).step_l = l_step_l;
      /* allocation for components and number of components has already been calculated by opj_pi_create */
      compno = 0 as OPJ_UINT32;
      while compno < numcomps {
        let mut l_res_0 = (*l_current_comp).resolutions;
        l_encoding_value_ptr = *l_tmp_ptr.offset(compno as isize);
        (*l_current_comp).dx = (*l_img_comp).dx;
        (*l_current_comp).dy = (*l_img_comp).dy;
        /* resolutions have already been initialized */
        resno = 0 as OPJ_UINT32;
        while resno < (*l_current_comp).numresolutions {
          let fresh9 = l_encoding_value_ptr;
          l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
          (*l_res_0).pdx = *fresh9;
          let fresh10 = l_encoding_value_ptr;
          l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
          (*l_res_0).pdy = *fresh10;
          let fresh11 = l_encoding_value_ptr;
          l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
          (*l_res_0).pw = *fresh11;
          let fresh12 = l_encoding_value_ptr;
          l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
          (*l_res_0).ph = *fresh12;
          l_res_0 = l_res_0.offset(1);
          resno += 1;
        }
        l_current_comp = l_current_comp.offset(1);
        l_img_comp = l_img_comp.offset(1);
        l_tccp = l_tccp.offset(1);
        compno += 1;
      }
      /* special treatment*/
      (*l_current_pi).include = (*l_current_pi.offset(-1)).include;
      (*l_current_pi).include_size = (*l_current_pi.offset(-1)).include_size;
      l_current_pi = l_current_pi.offset(1);
      pino += 1;
    }
    opj_free(l_tmp_data as *mut core::ffi::c_void);
    l_tmp_data = core::ptr::null_mut::<OPJ_UINT32>();
    opj_free(l_tmp_ptr as *mut core::ffi::c_void);
    l_tmp_ptr = core::ptr::null_mut::<*mut OPJ_UINT32>();
    if (*l_tcp).POC {
      opj_pi_update_decode_poc(l_pi, l_tcp, l_max_prec, l_max_res);
    } else {
      opj_pi_update_decode_not_poc(l_pi, l_tcp, l_max_prec, l_max_res);
    }
    l_pi
  }
}

pub(crate) fn opj_get_encoding_packet_count(
  mut p_image: *const opj_image_t,
  mut p_cp: *const opj_cp_t,
  mut p_tile_no: OPJ_UINT32,
) -> OPJ_UINT32 {
  unsafe {
    let mut l_max_res: OPJ_UINT32 = 0;
    let mut l_max_prec: OPJ_UINT32 = 0;
    let mut l_tx0: OPJ_UINT32 = 0;
    let mut l_tx1: OPJ_UINT32 = 0;
    let mut l_ty0: OPJ_UINT32 = 0;
    let mut l_ty1: OPJ_UINT32 = 0;
    let mut l_dx_min: OPJ_UINT32 = 0;
    let mut l_dy_min: OPJ_UINT32 = 0;
    /* preconditions in debug*/

    assert!(!p_cp.is_null());
    assert!(!p_image.is_null());
    assert!(p_tile_no < (*p_cp).tw.wrapping_mul((*p_cp).th));
    /* get encoding parameters*/
    opj_get_all_encoding_parameters(
      p_image,
      p_cp,
      p_tile_no,
      &mut l_tx0,
      &mut l_tx1,
      &mut l_ty0,
      &mut l_ty1,
      &mut l_dx_min,
      &mut l_dy_min,
      &mut l_max_prec,
      &mut l_max_res,
      core::ptr::null_mut::<*mut OPJ_UINT32>(),
    );
    (*(*p_cp).tcps.offset(p_tile_no as isize))
      .numlayers
      .wrapping_mul(l_max_prec)
      .wrapping_mul((*p_image).numcomps)
      .wrapping_mul(l_max_res)
  }
}

pub(crate) fn opj_pi_initialise_encode(
  mut p_image: *const opj_image_t,
  mut p_cp: *mut opj_cp_t,
  mut p_tile_no: OPJ_UINT32,
  mut p_t2_mode: J2K_T2_MODE,
  mut manager: &mut opj_event_mgr,
) -> *mut opj_pi_iterator_t {
  unsafe {
    let mut numcomps = (*p_image).numcomps;
    /* loop*/
    let mut pino: OPJ_UINT32 = 0;
    let mut compno: OPJ_UINT32 = 0;
    let mut resno: OPJ_UINT32 = 0;
    /* to store w, h, dx and dy for all components and resolutions*/
    let mut l_tmp_data = core::ptr::null_mut::<OPJ_UINT32>();
    let mut l_tmp_ptr = core::ptr::null_mut::<*mut OPJ_UINT32>();
    /* encoding parameters to set*/
    let mut l_max_res: OPJ_UINT32 = 0;
    let mut l_max_prec: OPJ_UINT32 = 0;
    let mut l_tx0: OPJ_UINT32 = 0;
    let mut l_tx1: OPJ_UINT32 = 0;
    let mut l_ty0: OPJ_UINT32 = 0;
    let mut l_ty1: OPJ_UINT32 = 0;
    let mut l_dx_min: OPJ_UINT32 = 0;
    let mut l_dy_min: OPJ_UINT32 = 0;
    let mut l_bound: OPJ_UINT32 = 0;
    let mut l_step_p: OPJ_UINT32 = 0;
    let mut l_step_c: OPJ_UINT32 = 0;
    let mut l_step_r: OPJ_UINT32 = 0;
    let mut l_step_l: OPJ_UINT32 = 0;
    let mut l_data_stride: OPJ_UINT32 = 0;
    /* pointers*/
    let mut l_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    let mut l_tccp = core::ptr::null::<opj_tccp_t>();
    let mut l_current_comp = core::ptr::null_mut::<opj_pi_comp_t>();
    let mut l_img_comp = core::ptr::null_mut::<opj_image_comp_t>();
    let mut l_current_pi = core::ptr::null_mut::<opj_pi_iterator_t>();
    let mut l_encoding_value_ptr = core::ptr::null_mut::<OPJ_UINT32>();
    /* preconditions in debug*/

    assert!(!p_cp.is_null());
    assert!(!p_image.is_null());
    assert!(p_tile_no < (*p_cp).tw.wrapping_mul((*p_cp).th));
    /* initializations*/
    l_tcp = &mut *(*p_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    l_bound = (*l_tcp).numpocs.wrapping_add(1u32);
    l_data_stride = (4i32 * 33i32) as OPJ_UINT32;
    l_tmp_data = opj_malloc(
      (l_data_stride.wrapping_mul(numcomps) as usize)
        .wrapping_mul(core::mem::size_of::<OPJ_UINT32>()),
    ) as *mut OPJ_UINT32;
    if l_tmp_data.is_null() {
      return core::ptr::null_mut::<opj_pi_iterator_t>();
    }
    l_tmp_ptr =
      opj_malloc((numcomps as usize).wrapping_mul(core::mem::size_of::<*mut OPJ_UINT32>()))
        as *mut *mut OPJ_UINT32;
    if l_tmp_ptr.is_null() {
      opj_free(l_tmp_data as *mut core::ffi::c_void);
      return core::ptr::null_mut::<opj_pi_iterator_t>();
    }
    /* memory allocation for pi*/
    l_pi = opj_pi_create(p_image, p_cp, p_tile_no, manager);
    if l_pi.is_null() {
      opj_free(l_tmp_data as *mut core::ffi::c_void);
      opj_free(l_tmp_ptr as *mut core::ffi::c_void);
      return core::ptr::null_mut::<opj_pi_iterator_t>();
    }
    l_encoding_value_ptr = l_tmp_data;
    /* update pointer array*/
    compno = 0 as OPJ_UINT32;
    while compno < numcomps {
      let fresh13 = &mut (*l_tmp_ptr.offset(compno as isize));
      *fresh13 = l_encoding_value_ptr;
      l_encoding_value_ptr = l_encoding_value_ptr.offset(l_data_stride as isize);
      compno += 1;
    }
    /* get encoding parameters*/
    opj_get_all_encoding_parameters(
      p_image,
      p_cp,
      p_tile_no,
      &mut l_tx0,
      &mut l_tx1,
      &mut l_ty0,
      &mut l_ty1,
      &mut l_dx_min,
      &mut l_dy_min,
      &mut l_max_prec,
      &mut l_max_res,
      l_tmp_ptr,
    );
    /* step calculations*/
    l_step_p = 1 as OPJ_UINT32;
    l_step_c = l_max_prec.wrapping_mul(l_step_p);
    l_step_r = numcomps.wrapping_mul(l_step_c);
    l_step_l = l_max_res.wrapping_mul(l_step_r);
    /* set values for first packet iterator*/
    (*l_pi).tp_on = (*p_cp).m_specific_param.m_enc.m_tp_on as OPJ_BYTE;
    l_current_pi = l_pi;
    /* memory allocation for include*/
    (*l_current_pi).include_size = (*l_tcp).numlayers.wrapping_mul(l_step_l);
    (*l_current_pi).include = opj_calloc(
      (*l_current_pi).include_size as size_t,
      core::mem::size_of::<OPJ_INT16>(),
    ) as *mut OPJ_INT16;
    if (*l_current_pi).include.is_null() {
      opj_free(l_tmp_data as *mut core::ffi::c_void);
      opj_free(l_tmp_ptr as *mut core::ffi::c_void);
      opj_pi_destroy(l_pi, l_bound);
      return core::ptr::null_mut::<opj_pi_iterator_t>();
    }
    /* special treatment for the first packet iterator*/
    l_current_comp = (*l_current_pi).comps;
    l_img_comp = (*p_image).comps;
    l_tccp = (*l_tcp).tccps;
    (*l_current_pi).tx0 = l_tx0;
    (*l_current_pi).ty0 = l_ty0;
    (*l_current_pi).tx1 = l_tx1;
    (*l_current_pi).ty1 = l_ty1;
    (*l_current_pi).dx = l_dx_min;
    (*l_current_pi).dy = l_dy_min;
    (*l_current_pi).step_p = l_step_p;
    (*l_current_pi).step_c = l_step_c;
    (*l_current_pi).step_r = l_step_r;
    (*l_current_pi).step_l = l_step_l;
    /* allocation for components and number of components has already been calculated by opj_pi_create */
    compno = 0 as OPJ_UINT32;
    while compno < numcomps {
      let mut l_res = (*l_current_comp).resolutions;
      l_encoding_value_ptr = *l_tmp_ptr.offset(compno as isize);
      (*l_current_comp).dx = (*l_img_comp).dx;
      (*l_current_comp).dy = (*l_img_comp).dy;
      /* resolutions have already been initialized */
      resno = 0 as OPJ_UINT32;
      while resno < (*l_current_comp).numresolutions {
        let fresh14 = l_encoding_value_ptr;
        l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
        (*l_res).pdx = *fresh14;
        let fresh15 = l_encoding_value_ptr;
        l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
        (*l_res).pdy = *fresh15;
        let fresh16 = l_encoding_value_ptr;
        l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
        (*l_res).pw = *fresh16;
        let fresh17 = l_encoding_value_ptr;
        l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
        (*l_res).ph = *fresh17;
        l_res = l_res.offset(1);
        resno += 1;
      }
      l_current_comp = l_current_comp.offset(1);
      l_img_comp = l_img_comp.offset(1);
      l_tccp = l_tccp.offset(1);
      compno += 1;
    }
    l_current_pi = l_current_pi.offset(1);
    pino = 1 as OPJ_UINT32;
    while pino < l_bound {
      l_current_comp = (*l_current_pi).comps;
      l_img_comp = (*p_image).comps;
      l_tccp = (*l_tcp).tccps;
      (*l_current_pi).tx0 = l_tx0;
      (*l_current_pi).ty0 = l_ty0;
      (*l_current_pi).tx1 = l_tx1;
      (*l_current_pi).ty1 = l_ty1;
      (*l_current_pi).dx = l_dx_min;
      (*l_current_pi).dy = l_dy_min;
      (*l_current_pi).step_p = l_step_p;
      (*l_current_pi).step_c = l_step_c;
      (*l_current_pi).step_r = l_step_r;
      (*l_current_pi).step_l = l_step_l;
      /* allocation for components and number of components has already been calculated by opj_pi_create */
      compno = 0 as OPJ_UINT32;
      while compno < numcomps {
        let mut l_res_0 = (*l_current_comp).resolutions;
        l_encoding_value_ptr = *l_tmp_ptr.offset(compno as isize);
        (*l_current_comp).dx = (*l_img_comp).dx;
        (*l_current_comp).dy = (*l_img_comp).dy;
        /* resolutions have already been initialized */
        resno = 0 as OPJ_UINT32;
        while resno < (*l_current_comp).numresolutions {
          let fresh18 = l_encoding_value_ptr;
          l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
          (*l_res_0).pdx = *fresh18;
          let fresh19 = l_encoding_value_ptr;
          l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
          (*l_res_0).pdy = *fresh19;
          let fresh20 = l_encoding_value_ptr;
          l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
          (*l_res_0).pw = *fresh20;
          let fresh21 = l_encoding_value_ptr;
          l_encoding_value_ptr = l_encoding_value_ptr.offset(1);
          (*l_res_0).ph = *fresh21;
          l_res_0 = l_res_0.offset(1);
          resno += 1;
        }
        l_current_comp = l_current_comp.offset(1);
        l_img_comp = l_img_comp.offset(1);
        l_tccp = l_tccp.offset(1);
        compno += 1;
      }
      /* special treatment*/
      (*l_current_pi).include = (*l_current_pi.offset(-1)).include;
      (*l_current_pi).include_size = (*l_current_pi.offset(-1)).include_size;
      l_current_pi = l_current_pi.offset(1);
      pino += 1;
    }
    opj_free(l_tmp_data as *mut core::ffi::c_void);
    l_tmp_data = core::ptr::null_mut::<OPJ_UINT32>();
    opj_free(l_tmp_ptr as *mut core::ffi::c_void);
    l_tmp_ptr = core::ptr::null_mut::<*mut OPJ_UINT32>();
    if (*l_tcp).POC
      && ((*p_cp).rsiz as core::ffi::c_int >= 0x3i32 && (*p_cp).rsiz as core::ffi::c_int <= 0x6i32
        || p_t2_mode as core::ffi::c_uint == FINAL_PASS as core::ffi::c_uint)
    {
      opj_pi_update_encode_poc_and_final(
        p_cp, p_tile_no, l_tx0, l_tx1, l_ty0, l_ty1, l_max_prec, l_max_res, l_dx_min, l_dy_min,
      );
    } else {
      opj_pi_update_encode_not_poc(
        p_cp, numcomps, p_tile_no, l_tx0, l_tx1, l_ty0, l_ty1, l_max_prec, l_max_res, l_dx_min,
        l_dy_min,
      );
    }
    l_pi
  }
}

pub(crate) fn opj_pi_create_encode(
  mut pi: *mut opj_pi_iterator_t,
  mut cp: *mut opj_cp_t,
  mut tileno: OPJ_UINT32,
  mut pino: OPJ_UINT32,
  mut tpnum: OPJ_UINT32,
  mut tppos: OPJ_INT32,
  mut t2_mode: J2K_T2_MODE,
) {
  unsafe {
    let mut i: OPJ_INT32 = 0;
    let mut incr_top = 1 as OPJ_UINT32;
    let mut resetX = 0 as OPJ_UINT32;
    let mut tcps: *mut opj_tcp_t = &mut *(*cp).tcps.offset(tileno as isize) as *mut opj_tcp_t;
    let mut tcp: *mut opj_poc_t =
      &mut *(*tcps).pocs.as_mut_ptr().offset(pino as isize) as *mut opj_poc_t;
    let prog = opj_j2k_convert_progression_order((*tcp).prg);
    (*pi.offset(pino as isize)).first = 1i32;
    (*pi.offset(pino as isize)).poc.prg = (*tcp).prg;
    if !((*cp).m_specific_param.m_enc.m_tp_on
      && (!((*cp).rsiz as core::ffi::c_int >= 0x3i32 && (*cp).rsiz as core::ffi::c_int <= 0x6i32)
        && !((*cp).rsiz as core::ffi::c_int >= 0x400i32
          && (*cp).rsiz as core::ffi::c_int <= 0x900i32 | 0x9bi32)
        && t2_mode as core::ffi::c_uint == FINAL_PASS as core::ffi::c_uint
        || (*cp).rsiz as core::ffi::c_int >= 0x3i32 && (*cp).rsiz as core::ffi::c_int <= 0x6i32
        || (*cp).rsiz as core::ffi::c_int >= 0x400i32
          && (*cp).rsiz as core::ffi::c_int <= 0x900i32 | 0x9bi32))
    {
      (*pi.offset(pino as isize)).poc.resno0 = (*tcp).resS;
      (*pi.offset(pino as isize)).poc.resno1 = (*tcp).resE;
      (*pi.offset(pino as isize)).poc.compno0 = (*tcp).compS;
      (*pi.offset(pino as isize)).poc.compno1 = (*tcp).compE;
      (*pi.offset(pino as isize)).poc.layno0 = (*tcp).layS;
      (*pi.offset(pino as isize)).poc.layno1 = (*tcp).layE;
      (*pi.offset(pino as isize)).poc.precno0 = (*tcp).prcS;
      (*pi.offset(pino as isize)).poc.precno1 = (*tcp).prcE;
      (*pi.offset(pino as isize)).poc.tx0 = (*tcp).txS;
      (*pi.offset(pino as isize)).poc.ty0 = (*tcp).tyS;
      (*pi.offset(pino as isize)).poc.tx1 = (*tcp).txE;
      (*pi.offset(pino as isize)).poc.ty1 = (*tcp).tyE
    } else {
      i = tppos + 1i32;
      while i < 4i32 {
        match prog.get_step(i) {
          ProgressionStep::Resolution => {
            (*pi.offset(pino as isize)).poc.resno0 = (*tcp).resS;
            (*pi.offset(pino as isize)).poc.resno1 = (*tcp).resE
          }
          ProgressionStep::Component => {
            (*pi.offset(pino as isize)).poc.compno0 = (*tcp).compS;
            (*pi.offset(pino as isize)).poc.compno1 = (*tcp).compE
          }
          ProgressionStep::Layer => {
            (*pi.offset(pino as isize)).poc.layno0 = (*tcp).layS;
            (*pi.offset(pino as isize)).poc.layno1 = (*tcp).layE
          }
          ProgressionStep::Precinct => match (*tcp).prg as core::ffi::c_int {
            0 | 1 => {
              (*pi.offset(pino as isize)).poc.precno0 = (*tcp).prcS;
              (*pi.offset(pino as isize)).poc.precno1 = (*tcp).prcE
            }
            _ => {
              (*pi.offset(pino as isize)).poc.tx0 = (*tcp).txS;
              (*pi.offset(pino as isize)).poc.ty0 = (*tcp).tyS;
              (*pi.offset(pino as isize)).poc.tx1 = (*tcp).txE;
              (*pi.offset(pino as isize)).poc.ty1 = (*tcp).tyE
            }
          },
          _ => {}
        }
        i += 1
      }
      if tpnum == 0u32 {
        i = tppos;
        while i >= 0i32 {
          match prog.get_step(i) {
            ProgressionStep::Component => {
              (*tcp).comp_t = (*tcp).compS;
              (*pi.offset(pino as isize)).poc.compno0 = (*tcp).comp_t;
              (*pi.offset(pino as isize)).poc.compno1 = (*tcp).comp_t.wrapping_add(1u32);
              (*tcp).comp_t = ((*tcp).comp_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32
            }
            ProgressionStep::Resolution => {
              (*tcp).res_t = (*tcp).resS;
              (*pi.offset(pino as isize)).poc.resno0 = (*tcp).res_t;
              (*pi.offset(pino as isize)).poc.resno1 = (*tcp).res_t.wrapping_add(1u32);
              (*tcp).res_t = ((*tcp).res_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32
            }
            ProgressionStep::Layer => {
              (*tcp).lay_t = (*tcp).layS;
              (*pi.offset(pino as isize)).poc.layno0 = (*tcp).lay_t;
              (*pi.offset(pino as isize)).poc.layno1 = (*tcp).lay_t.wrapping_add(1u32);
              (*tcp).lay_t = ((*tcp).lay_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32
            }
            ProgressionStep::Precinct => match (*tcp).prg as core::ffi::c_int {
              0 | 1 => {
                (*tcp).prc_t = (*tcp).prcS;
                (*pi.offset(pino as isize)).poc.precno0 = (*tcp).prc_t;
                (*pi.offset(pino as isize)).poc.precno1 = (*tcp).prc_t.wrapping_add(1u32);
                (*tcp).prc_t = ((*tcp).prc_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32
              }
              _ => {
                (*tcp).tx0_t = (*tcp).txS;
                (*tcp).ty0_t = (*tcp).tyS;
                (*pi.offset(pino as isize)).poc.tx0 = (*tcp).tx0_t;
                (*pi.offset(pino as isize)).poc.tx1 = (*tcp)
                  .tx0_t
                  .wrapping_add((*tcp).dx)
                  .wrapping_sub((*tcp).tx0_t.wrapping_rem((*tcp).dx));
                (*pi.offset(pino as isize)).poc.ty0 = (*tcp).ty0_t;
                (*pi.offset(pino as isize)).poc.ty1 = (*tcp)
                  .ty0_t
                  .wrapping_add((*tcp).dy)
                  .wrapping_sub((*tcp).ty0_t.wrapping_rem((*tcp).dy));
                (*tcp).tx0_t = (*pi.offset(pino as isize)).poc.tx1;
                (*tcp).ty0_t = (*pi.offset(pino as isize)).poc.ty1
              }
            },
            _ => {}
          }
          i -= 1
        }
        incr_top = 1 as OPJ_UINT32
      } else {
        i = tppos;
        while i >= 0i32 {
          match prog.get_step(i) {
            ProgressionStep::Component => {
              (*pi.offset(pino as isize)).poc.compno0 = (*tcp).comp_t.wrapping_sub(1u32);
              (*pi.offset(pino as isize)).poc.compno1 = (*tcp).comp_t
            }
            ProgressionStep::Resolution => {
              (*pi.offset(pino as isize)).poc.resno0 = (*tcp).res_t.wrapping_sub(1u32);
              (*pi.offset(pino as isize)).poc.resno1 = (*tcp).res_t
            }
            ProgressionStep::Layer => {
              (*pi.offset(pino as isize)).poc.layno0 = (*tcp).lay_t.wrapping_sub(1u32);
              (*pi.offset(pino as isize)).poc.layno1 = (*tcp).lay_t
            }
            ProgressionStep::Precinct => match (*tcp).prg as core::ffi::c_int {
              0 | 1 => {
                (*pi.offset(pino as isize)).poc.precno0 = (*tcp).prc_t.wrapping_sub(1u32);
                (*pi.offset(pino as isize)).poc.precno1 = (*tcp).prc_t
              }
              _ => {
                (*pi.offset(pino as isize)).poc.tx0 = (*tcp)
                  .tx0_t
                  .wrapping_sub((*tcp).dx)
                  .wrapping_sub((*tcp).tx0_t.wrapping_rem((*tcp).dx));
                (*pi.offset(pino as isize)).poc.tx1 = (*tcp).tx0_t;
                (*pi.offset(pino as isize)).poc.ty0 = (*tcp)
                  .ty0_t
                  .wrapping_sub((*tcp).dy)
                  .wrapping_sub((*tcp).ty0_t.wrapping_rem((*tcp).dy));
                (*pi.offset(pino as isize)).poc.ty1 = (*tcp).ty0_t
              }
            },
            _ => {}
          }
          if incr_top == 1u32 {
            match prog.get_step(i) {
              ProgressionStep::Resolution => {
                if (*tcp).res_t == (*tcp).resE {
                  if opj_pi_check_next_level(i - 1i32, cp, tileno, pino, prog) != 0 {
                    (*tcp).res_t = (*tcp).resS;
                    (*pi.offset(pino as isize)).poc.resno0 = (*tcp).res_t;
                    (*pi.offset(pino as isize)).poc.resno1 = (*tcp).res_t.wrapping_add(1u32);
                    (*tcp).res_t =
                      ((*tcp).res_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32;
                    incr_top = 1 as OPJ_UINT32
                  } else {
                    incr_top = 0 as OPJ_UINT32
                  }
                } else {
                  (*pi.offset(pino as isize)).poc.resno0 = (*tcp).res_t;
                  (*pi.offset(pino as isize)).poc.resno1 = (*tcp).res_t.wrapping_add(1u32);
                  (*tcp).res_t =
                    ((*tcp).res_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32;
                  incr_top = 0 as OPJ_UINT32
                }
              }
              ProgressionStep::Component => {
                if (*tcp).comp_t == (*tcp).compE {
                  if opj_pi_check_next_level(i - 1i32, cp, tileno, pino, prog) != 0 {
                    (*tcp).comp_t = (*tcp).compS;
                    (*pi.offset(pino as isize)).poc.compno0 = (*tcp).comp_t;
                    (*pi.offset(pino as isize)).poc.compno1 = (*tcp).comp_t.wrapping_add(1u32);
                    (*tcp).comp_t =
                      ((*tcp).comp_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32;
                    incr_top = 1 as OPJ_UINT32
                  } else {
                    incr_top = 0 as OPJ_UINT32
                  }
                } else {
                  (*pi.offset(pino as isize)).poc.compno0 = (*tcp).comp_t;
                  (*pi.offset(pino as isize)).poc.compno1 = (*tcp).comp_t.wrapping_add(1u32);
                  (*tcp).comp_t =
                    ((*tcp).comp_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32;
                  incr_top = 0 as OPJ_UINT32
                }
              }
              ProgressionStep::Layer => {
                if (*tcp).lay_t == (*tcp).layE {
                  if opj_pi_check_next_level(i - 1i32, cp, tileno, pino, prog) != 0 {
                    (*tcp).lay_t = (*tcp).layS;
                    (*pi.offset(pino as isize)).poc.layno0 = (*tcp).lay_t;
                    (*pi.offset(pino as isize)).poc.layno1 = (*tcp).lay_t.wrapping_add(1u32);
                    (*tcp).lay_t =
                      ((*tcp).lay_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32;
                    incr_top = 1 as OPJ_UINT32
                  } else {
                    incr_top = 0 as OPJ_UINT32
                  }
                } else {
                  (*pi.offset(pino as isize)).poc.layno0 = (*tcp).lay_t;
                  (*pi.offset(pino as isize)).poc.layno1 = (*tcp).lay_t.wrapping_add(1u32);
                  (*tcp).lay_t =
                    ((*tcp).lay_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32;
                  incr_top = 0 as OPJ_UINT32
                }
              }
              ProgressionStep::Precinct => match (*tcp).prg as core::ffi::c_int {
                0 | 1 => {
                  if (*tcp).prc_t == (*tcp).prcE {
                    if opj_pi_check_next_level(i - 1i32, cp, tileno, pino, prog) != 0 {
                      (*tcp).prc_t = (*tcp).prcS;
                      (*pi.offset(pino as isize)).poc.precno0 = (*tcp).prc_t;
                      (*pi.offset(pino as isize)).poc.precno1 = (*tcp).prc_t.wrapping_add(1u32);
                      (*tcp).prc_t =
                        ((*tcp).prc_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32;
                      incr_top = 1 as OPJ_UINT32
                    } else {
                      incr_top = 0 as OPJ_UINT32
                    }
                  } else {
                    (*pi.offset(pino as isize)).poc.precno0 = (*tcp).prc_t;
                    (*pi.offset(pino as isize)).poc.precno1 = (*tcp).prc_t.wrapping_add(1u32);
                    (*tcp).prc_t =
                      ((*tcp).prc_t as core::ffi::c_uint).wrapping_add(1u32) as OPJ_UINT32;
                    incr_top = 0 as OPJ_UINT32
                  }
                }
                _ => {
                  if (*tcp).tx0_t >= (*tcp).txE {
                    if (*tcp).ty0_t >= (*tcp).tyE {
                      if opj_pi_check_next_level(i - 1i32, cp, tileno, pino, prog) != 0 {
                        (*tcp).ty0_t = (*tcp).tyS;
                        (*pi.offset(pino as isize)).poc.ty0 = (*tcp).ty0_t;
                        (*pi.offset(pino as isize)).poc.ty1 = (*tcp)
                          .ty0_t
                          .wrapping_add((*tcp).dy)
                          .wrapping_sub((*tcp).ty0_t.wrapping_rem((*tcp).dy));
                        (*tcp).ty0_t = (*pi.offset(pino as isize)).poc.ty1;
                        incr_top = 1 as OPJ_UINT32;
                        resetX = 1 as OPJ_UINT32
                      } else {
                        incr_top = 0 as OPJ_UINT32;
                        resetX = 0 as OPJ_UINT32
                      }
                    } else {
                      (*pi.offset(pino as isize)).poc.ty0 = (*tcp).ty0_t;
                      (*pi.offset(pino as isize)).poc.ty1 = (*tcp)
                        .ty0_t
                        .wrapping_add((*tcp).dy)
                        .wrapping_sub((*tcp).ty0_t.wrapping_rem((*tcp).dy));
                      (*tcp).ty0_t = (*pi.offset(pino as isize)).poc.ty1;
                      incr_top = 0 as OPJ_UINT32;
                      resetX = 1 as OPJ_UINT32
                    }
                    if resetX == 1u32 {
                      (*tcp).tx0_t = (*tcp).txS;
                      (*pi.offset(pino as isize)).poc.tx0 = (*tcp).tx0_t;
                      (*pi.offset(pino as isize)).poc.tx1 = (*tcp)
                        .tx0_t
                        .wrapping_add((*tcp).dx)
                        .wrapping_sub((*tcp).tx0_t.wrapping_rem((*tcp).dx));
                      (*tcp).tx0_t = (*pi.offset(pino as isize)).poc.tx1
                    }
                  } else {
                    (*pi.offset(pino as isize)).poc.tx0 = (*tcp).tx0_t;
                    (*pi.offset(pino as isize)).poc.tx1 = (*tcp)
                      .tx0_t
                      .wrapping_add((*tcp).dx)
                      .wrapping_sub((*tcp).tx0_t.wrapping_rem((*tcp).dx));
                    (*tcp).tx0_t = (*pi.offset(pino as isize)).poc.tx1;
                    incr_top = 0 as OPJ_UINT32
                  }
                }
              },
              _ => {}
            }
          }
          i -= 1
        }
      }
    };
  }
}

pub(crate) fn opj_pi_destroy(mut p_pi: *mut opj_pi_iterator_t, mut p_nb_elements: OPJ_UINT32) {
  unsafe {
    let mut compno: OPJ_UINT32 = 0;
    let mut pino: OPJ_UINT32 = 0;
    let mut l_current_pi = p_pi;
    if !p_pi.is_null() {
      if !(*p_pi).include.is_null() {
        opj_free((*p_pi).include as *mut core::ffi::c_void);
        (*p_pi).include = core::ptr::null_mut::<OPJ_INT16>()
      }
      pino = 0 as OPJ_UINT32;
      while pino < p_nb_elements {
        if !(*l_current_pi).comps.is_null() {
          let mut l_current_component = (*l_current_pi).comps;
          compno = 0 as OPJ_UINT32;
          while compno < (*l_current_pi).numcomps {
            if !(*l_current_component).resolutions.is_null() {
              opj_free((*l_current_component).resolutions as *mut core::ffi::c_void);
              (*l_current_component).resolutions = core::ptr::null_mut::<opj_pi_resolution_t>()
            }
            l_current_component = l_current_component.offset(1);
            compno += 1;
          }
          opj_free((*l_current_pi).comps as *mut core::ffi::c_void);
          (*l_current_pi).comps = core::ptr::null_mut::<opj_pi_comp_t>()
        }
        l_current_pi = l_current_pi.offset(1);
        pino += 1;
      }
      opj_free(p_pi as *mut core::ffi::c_void);
    };
  }
}

pub(crate) fn opj_pi_update_encoding_parameters(
  mut p_image: *const opj_image_t,
  mut p_cp: *mut opj_cp_t,
  mut p_tile_no: OPJ_UINT32,
) {
  unsafe {
    /* encoding parameters to set */
    let mut l_max_res: OPJ_UINT32 = 0;
    let mut l_max_prec: OPJ_UINT32 = 0;
    let mut l_tx0: OPJ_UINT32 = 0;
    let mut l_tx1: OPJ_UINT32 = 0;
    let mut l_ty0: OPJ_UINT32 = 0;
    let mut l_ty1: OPJ_UINT32 = 0;
    let mut l_dx_min: OPJ_UINT32 = 0;
    let mut l_dy_min: OPJ_UINT32 = 0;
    /* pointers */
    let mut l_tcp = core::ptr::null_mut::<opj_tcp_t>();
    /* preconditions */

    assert!(!p_cp.is_null());
    assert!(!p_image.is_null());
    assert!(p_tile_no < (*p_cp).tw.wrapping_mul((*p_cp).th));
    l_tcp = &mut *(*p_cp).tcps.offset(p_tile_no as isize) as *mut opj_tcp_t;
    /* get encoding parameters */
    opj_get_encoding_parameters(
      p_image,
      p_cp,
      p_tile_no,
      &mut l_tx0,
      &mut l_tx1,
      &mut l_ty0,
      &mut l_ty1,
      &mut l_dx_min,
      &mut l_dy_min,
      &mut l_max_prec,
      &mut l_max_res,
    );
    if (*l_tcp).POC {
      opj_pi_update_encode_poc_and_final(
        p_cp, p_tile_no, l_tx0, l_tx1, l_ty0, l_ty1, l_max_prec, l_max_res, l_dx_min, l_dy_min,
      );
    } else {
      opj_pi_update_encode_not_poc(
        p_cp,
        (*p_image).numcomps,
        p_tile_no,
        l_tx0,
        l_tx1,
        l_ty0,
        l_ty1,
        l_max_prec,
        l_max_res,
        l_dx_min,
        l_dy_min,
      );
    };
  }
}

pub(crate) fn opj_pi_next(mut pi: *mut opj_pi_iterator_t) -> OPJ_BOOL {
  unsafe {
    match (*pi).poc.prg as core::ffi::c_int {
      0 => return opj_pi_next_lrcp(pi),
      1 => return opj_pi_next_rlcp(pi),
      2 => return opj_pi_next_rpcl(pi),
      3 => return opj_pi_next_pcrl(pi),
      4 => return opj_pi_next_cprl(pi),
      -1 => return 0i32,
      _ => {}
    }
    0i32
  }
}

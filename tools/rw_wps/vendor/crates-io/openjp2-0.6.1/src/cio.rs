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

use super::openjpeg::*;

pub(crate) fn opj_write_bytes(
  mut p_buffer: *mut OPJ_BYTE,
  mut p_value: OPJ_UINT32,
  mut p_nb_bytes: OPJ_UINT32,
) {
  let nb = p_nb_bytes as usize;
  const SIZE: usize = core::mem::size_of::<OPJ_UINT32>();
  assert!(nb > 0 && nb <= SIZE);
  let value = p_value as u32;
  let dest = unsafe { core::slice::from_raw_parts_mut(p_buffer, nb) };
  let offset = SIZE - nb;
  let buf = value.to_be_bytes();
  dest.copy_from_slice(&buf[offset..]);
}

pub(crate) fn opj_read_bytes(
  mut p_buffer: *const OPJ_BYTE,
  mut p_value: *mut OPJ_UINT32,
  mut p_nb_bytes: OPJ_UINT32,
) {
  let nb = p_nb_bytes as usize;
  const SIZE: usize = core::mem::size_of::<OPJ_UINT32>();
  assert!(nb > 0 && nb <= SIZE);
  let (src, value) = unsafe { (core::slice::from_raw_parts(p_buffer, nb), &mut *p_value) };
  let offset = SIZE - nb;
  let mut buf = [0u8; SIZE];
  buf[offset..].copy_from_slice(src);
  *value = u32::from_be_bytes(buf);
}

pub(crate) fn opj_write_double(mut p_buffer: *mut OPJ_BYTE, mut p_value: OPJ_FLOAT64) {
  let value = p_value as f64;
  let dest =
    unsafe { core::slice::from_raw_parts_mut(p_buffer, core::mem::size_of::<OPJ_FLOAT64>()) };
  dest.copy_from_slice(&value.to_be_bytes());
}

pub(crate) fn opj_read_double(mut p_buffer: *const OPJ_BYTE, mut p_value: *mut OPJ_FLOAT64) {
  let (src, value) = unsafe {
    (
      core::slice::from_raw_parts(p_buffer, core::mem::size_of::<OPJ_FLOAT64>()),
      &mut *p_value,
    )
  };
  *value = f64::from_be_bytes([
    src[0], src[1], src[2], src[3], src[4], src[5], src[6], src[7],
  ]);
}

pub(crate) fn opj_write_float(p_buffer: *mut OPJ_BYTE, p_value: OPJ_FLOAT32) {
  let value = p_value as f32;
  let dest =
    unsafe { core::slice::from_raw_parts_mut(p_buffer, core::mem::size_of::<OPJ_FLOAT32>()) };
  dest.copy_from_slice(&value.to_be_bytes());
}

pub(crate) fn opj_read_float(mut p_buffer: *const OPJ_BYTE, mut p_value: *mut OPJ_FLOAT32) {
  let (src, value) = unsafe {
    (
      core::slice::from_raw_parts(p_buffer, core::mem::size_of::<OPJ_FLOAT32>()),
      &mut *p_value,
    )
  };
  *value = f32::from_be_bytes([src[0], src[1], src[2], src[3]]);
}

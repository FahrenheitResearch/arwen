/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
 * Copyright (c) 2017, IntoPix SA <contact@intopix.com>
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

use crate::math::*;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

extern "C" {
  fn memset(_: *mut core::ffi::c_void, _: core::ffi::c_int, _: usize) -> *mut core::ffi::c_void;
}

#[derive(Clone)]
pub struct SparseArray {
  pub width: u32,
  pub height: u32,
  pub block_width: u32,
  pub block_height: u32,
  pub block_count_hor: u32,
  pub block_count_ver: u32,
  pub blocks: Vec<Option<Vec<i32>>>,
}

impl SparseArray {
  pub fn new(width: u32, height: u32, block_width: u32, block_height: u32) -> Option<Self> {
    if width == 0 || height == 0 || block_width == 0 || block_height == 0 {
      return None;
    }

    let block_count_hor = opj_uint_ceildiv(width, block_width);
    let block_count_ver = opj_uint_ceildiv(height, block_height);

    if block_count_hor > u32::MAX / block_count_ver {
      return None;
    }

    let total_blocks = (block_count_hor * block_count_ver) as usize;
    Some(Self {
      width,
      height,
      block_width,
      block_height,
      block_count_hor,
      block_count_ver,
      blocks: vec![None; total_blocks],
    })
  }

  fn block_index(&self, x: u32, y: u32) -> usize {
    (y * self.block_count_hor + x) as usize
  }

  pub fn data_block(&self, x: u32, y: u32) -> Option<&[i32]> {
    self.blocks.get(self.block_index(x, y))?.as_deref()
  }

  pub fn data_block_mut(&mut self, x: u32, y: u32) -> Option<&mut [i32]> {
    let index = self.block_index(x, y);
    self.blocks.get_mut(index)?.as_deref_mut()
  }

  pub fn set_data_block(&mut self, x: u32, y: u32, value: Vec<i32>) {
    let index = self.block_index(x, y);
    self.blocks[index] = Some(value);
  }
}

pub(crate) fn is_region_valid(sa: &SparseArray, x0: u32, y0: u32, x1: u32, y1: u32) -> bool {
  !(x0 >= sa.width || x1 <= x0 || x1 > sa.width || y0 >= sa.height || y1 <= y0 || y1 > sa.height)
}

pub(crate) fn sparse_array_read(
  sa: &SparseArray,
  x0: u32,
  y0: u32,
  x1: u32,
  y1: u32,
  dest: *mut i32,
  dest_col_stride: u32,
  dest_line_stride: u32,
  forgiving: bool,
) -> bool {
  if !is_region_valid(sa, x0, y0, x1, y1) {
    return forgiving;
  }

  let mut y = y0;
  let mut block_y = y0 / sa.block_height;
  while y < y1 {
    let mut x: u32 = 0;
    let mut block_x: u32 = 0;
    let mut x_incr = 0 as u32;
    let mut block_y_offset: u32 = 0;
    let y_incr = if y == y0 {
      sa.block_height
        .wrapping_sub(y0.wrapping_rem(sa.block_height))
    } else {
      sa.block_height
    };
    block_y_offset = sa.block_height.wrapping_sub(y_incr);
    let y_incr = y_incr.min(y1.wrapping_sub(y));
    block_x = x0.wrapping_div(sa.block_width);
    x = x0;
    while x < x1 {
      let mut block_x_offset = 0;
      x_incr = if x == x0 {
        sa.block_width.wrapping_sub(x0.wrapping_rem(sa.block_width))
      } else {
        sa.block_width
      };
      block_x_offset = sa.block_width.wrapping_sub(x_incr);
      x_incr = x_incr.min(x1.wrapping_sub(x));
      if let Some(src_block) = sa.data_block(block_x, block_y) {
        let mut src = src_block[((block_y_offset * sa.block_width) + block_x_offset) as usize..]
          .chunks(sa.block_width as usize);
        if dest_col_stride == 1 {
          let mut dest_ptr = unsafe {
            dest
              .add((y.wrapping_sub(y0) as usize).wrapping_mul(dest_line_stride as usize))
              .offset(x.wrapping_sub(x0).wrapping_mul(dest_col_stride) as isize)
          };
          for _ in 0..y_incr {
            let src_ptr = src.next().unwrap();
            unsafe {
              for k in 0..x_incr {
                *dest_ptr.offset(k as isize) = src_ptr[k as usize];
              }
              dest_ptr = dest_ptr.offset(dest_line_stride as isize);
            }
          }
        } else {
          let mut dest_ptr = unsafe {
            dest
              .add((y.wrapping_sub(y0) as usize).wrapping_mul(dest_line_stride as usize))
              .offset(x.wrapping_sub(x0).wrapping_mul(dest_col_stride) as isize)
          };
          for _ in 0..y_incr {
            let src_ptr = src.next().unwrap();
            for k in 0..x_incr {
              unsafe {
                *dest_ptr.offset(k.wrapping_mul(dest_col_stride) as isize) = src_ptr[k as usize];
              }
            }
            unsafe {
              dest_ptr = dest_ptr.offset(dest_line_stride as isize);
            }
          }
        }
      } else {
        if dest_col_stride == 1 {
          let mut dest_ptr = unsafe {
            dest
              .add((y.wrapping_sub(y0) as usize).wrapping_mul(dest_line_stride as usize))
              .offset(x.wrapping_sub(x0).wrapping_mul(dest_col_stride) as isize)
          };
          for _ in 0..y_incr {
            unsafe {
              memset(
                dest_ptr as *mut core::ffi::c_void,
                0i32,
                core::mem::size_of::<i32>().wrapping_mul(x_incr as usize),
              );
              dest_ptr = dest_ptr.offset(dest_line_stride as isize);
            }
          }
        } else {
          let mut dest_ptr = unsafe {
            dest
              .add((y.wrapping_sub(y0) as usize).wrapping_mul(dest_line_stride as usize))
              .offset(x.wrapping_sub(x0).wrapping_mul(dest_col_stride) as isize)
          };
          for _ in 0..y_incr {
            for k in 0..x_incr {
              unsafe {
                *dest_ptr.offset(k.wrapping_mul(dest_col_stride) as isize) = 0i32;
              }
            }
            unsafe {
              dest_ptr = dest_ptr.offset(dest_line_stride as isize);
            }
          }
        }
      }
      block_x = block_x.wrapping_add(1);
      x = x.wrapping_add(x_incr)
    }
    block_y = block_y.wrapping_add(1);
    y = y.wrapping_add(y_incr)
  }
  true
}

pub(crate) fn sparse_array_write(
  sa: &mut SparseArray,
  x0: u32,
  y0: u32,
  x1: u32,
  y1: u32,
  src: *const i32,
  src_col_stride: u32,
  src_line_stride: u32,
  forgiving: bool,
) -> bool {
  if !is_region_valid(sa, x0, y0, x1, y1) {
    return forgiving;
  }

  let mut y = y0;
  let mut block_y = y0 / sa.block_height;
  while y < y1 {
    let mut x: u32 = 0;
    let mut block_x: u32 = 0;
    let mut x_incr = 0 as u32;
    let mut block_y_offset: u32 = 0;
    let y_incr = if y == y0 {
      sa.block_height
        .wrapping_sub(y0.wrapping_rem(sa.block_height))
    } else {
      sa.block_height
    };
    block_y_offset = sa.block_height.wrapping_sub(y_incr);
    let y_incr = y_incr.min(y1.wrapping_sub(y));
    block_x = x0.wrapping_div(sa.block_width);
    x = x0;
    while x < x1 {
      let mut block_x_offset: u32 = 0;
      x_incr = if x == x0 {
        sa.block_width.wrapping_sub(x0.wrapping_rem(sa.block_width))
      } else {
        sa.block_width
      };
      block_x_offset = sa.block_width.wrapping_sub(x_incr);
      x_incr = x_incr.min(x1.wrapping_sub(x));
      let index = sa.block_index(block_x, block_y);
      if sa.blocks[index].is_none() {
        sa.blocks[index] = Some(vec![0i32; (sa.block_width * sa.block_height) as usize]);
      }
      if let Some(ref mut dest_block) = sa.blocks[index] {
        let mut dest = dest_block[((block_y_offset * sa.block_width) + block_x_offset) as usize..]
          .chunks_mut(sa.block_width as usize);
        if src_col_stride == 1 {
          let mut src_ptr = unsafe {
            src
              .add((y.wrapping_sub(y0) as usize).wrapping_mul(src_line_stride as usize))
              .offset(x.wrapping_sub(x0).wrapping_mul(src_col_stride) as isize)
          };
          for _ in 0..y_incr {
            let dest_ptr = dest.next().unwrap();
            unsafe {
              for k in 0..x_incr {
                dest_ptr[k as usize] = *src_ptr.offset(k as isize);
              }
              src_ptr = src_ptr.offset(src_line_stride as isize);
            }
          }
        } else {
          let mut src_ptr = unsafe {
            src
              .add((y.wrapping_sub(y0) as usize).wrapping_mul(src_line_stride as usize))
              .offset(x.wrapping_sub(x0).wrapping_mul(src_col_stride) as isize)
          };
          for _ in 0..y_incr {
            let dest_ptr = dest.next().unwrap();
            for k in 0..x_incr {
              unsafe {
                dest_ptr[k as usize] = *src_ptr.offset(k.wrapping_mul(src_col_stride) as isize);
              }
            }
            unsafe {
              src_ptr = src_ptr.offset(src_line_stride as isize);
            }
          }
        }
      }
      block_x = block_x.wrapping_add(1);
      x = x.wrapping_add(x_incr)
    }
    block_y = block_y.wrapping_add(1);
    y = y.wrapping_add(y_incr)
  }
  true
}

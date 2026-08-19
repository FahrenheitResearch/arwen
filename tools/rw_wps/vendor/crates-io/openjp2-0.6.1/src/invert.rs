use smallvec::SmallVec;

/*
==========================================================
   Matric inversion interface
==========================================================
*/
/* *
 * Matrix inversion.
 */
pub(crate) fn opj_matrix_inversion_f(
  mut pSrcMatrix: &mut [f32],
  mut pDestMatrix: &mut [f32],
  mut nb_compo: usize,
) -> bool {
  let mut lPermutations = SmallVec::<[u32; 0]>::with_capacity(nb_compo);
  let mut src_tmp = SmallVec::<[f32; 0]>::with_capacity(nb_compo);
  let mut dest_tmp = SmallVec::<[f32; 0]>::with_capacity(nb_compo);
  let mut swap_area = SmallVec::<[f32; 0]>::with_capacity(nb_compo);
  lPermutations.resize(nb_compo, 0);
  src_tmp.resize(nb_compo, 0.0);
  dest_tmp.resize(nb_compo, 0.0);
  swap_area.resize(nb_compo, 0.0);
  if !opj_lupDecompose(
    pSrcMatrix,
    lPermutations.as_mut_slice(),
    swap_area.as_mut_slice(),
    nb_compo,
  ) {
    return false;
  }
  opj_lupInvert(
    pSrcMatrix,
    pDestMatrix,
    nb_compo,
    lPermutations.as_slice(),
    src_tmp.as_mut_slice(),
    dest_tmp.as_mut_slice(),
    swap_area.as_mut_slice(),
  );
  true
}
/*
 * The copyright in this software is being made available under the 2-clauses
 * BSD License, included below. This software may be subject to other third
 * party and contributor rights, including patent rights, and no such rights
 * are granted under this license.
 *
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
/* *
 * LUP decomposition
 */
/*
==========================================================
   Local functions
==========================================================
*/
fn opj_lupDecompose(
  matrix: &mut [f32],
  permutations: &mut [u32],
  p_swap_area: &mut [f32],
  nb_compo: usize,
) -> bool {
  let mut tmp_permutations = 0;
  let mut k2 = 0;
  let mut temp = 0.0;
  let l_last_column = nb_compo - 1;
  let l_swap_size = nb_compo * core::mem::size_of::<f32>();
  let mut l_tmp_matrix = 0;
  let mut l_column_matrix = 0;
  let mut l_dest_matrix = 0;
  let mut offset = 1;
  let mut l_stride = nb_compo - 1;
  // initialize permutations
  for i in 0..nb_compo {
    permutations[i] = i as u32;
  }
  // now make a pivot with column switch
  for k in 0..l_last_column {
    // take the middle element
    l_column_matrix = l_tmp_matrix + k;
    // make permutation with the biggest value in the column
    let mut p = 0.0;
    for i in k..nb_compo {
      temp = matrix[l_column_matrix].abs();
      if temp > p {
        p = temp;
        k2 = i;
      }
      // next line
      l_column_matrix += nb_compo;
    }
    // a whole rest of 0 -> non singular
    if p == 0.0 {
      return false;
    }
    // should we permute ?
    if k2 != k {
      // exchange of line
      // k2 > k
      let dst_p = tmp_permutations + k2 - k;
      // swap indices
      permutations.swap(tmp_permutations, dst_p);
      // and swap entire line.
      l_column_matrix = l_tmp_matrix + (k2 - k) * nb_compo;
      p_swap_area.copy_from_slice(&matrix[l_column_matrix..l_column_matrix + l_swap_size]);
      matrix.copy_within(l_tmp_matrix..l_tmp_matrix + l_swap_size, l_column_matrix);
      matrix[l_tmp_matrix..l_tmp_matrix + l_swap_size].copy_from_slice(&p_swap_area[..l_swap_size]);
    }
    // now update data in the rest of the line and line after
    l_dest_matrix = l_tmp_matrix + k;
    l_column_matrix = l_dest_matrix + nb_compo;
    // take the middle element
    let temp = matrix[l_dest_matrix];
    l_dest_matrix += 1;
    // now compute up data (i.e. coeff up of the diagonal).
    for _ in offset..nb_compo {
      // divide the lower column elements by the diagonal value
      // matrix[i][k] /= matrix[k][k];
      // p = matrix[i][k]
      let p = matrix[l_column_matrix] / temp;
      l_column_matrix += 1;
      matrix[l_column_matrix] = p;

      for j in offset..nb_compo {
        matrix[l_dest_matrix + j] -= p * matrix[l_dest_matrix + j - 1];
      }
      // come back to the k+1th element
      l_dest_matrix -= l_stride;
      // go to kth element of the next line
      l_column_matrix += k;
    }
    // offset is now k+2
    offset += 1;
    // 1 element less for stride
    l_stride -= 1;
    // next line
    l_tmp_matrix += nb_compo;
  }
  true
}
/* *
 * LUP solving
 */
fn opj_lupSolve(
  p_result: &mut [f32],
  p_matrix: &[f32],
  p_vector: &[f32],
  p_permutations: &[u32],
  nb_compo: usize,
  p_intermediate_data: &mut [f32],
) {
  let mut k: i32 = 0;
  let mut i: usize = 0;
  let mut j: usize = 0;
  let mut sum: f32 = 0.;
  let mut u: f32 = 0.;
  let l_stride = nb_compo + 1;
  let mut l_current_ptr: usize;
  let mut l_intermediate_ptr: usize;
  let mut l_dest_ptr: usize;
  let mut l_tmp_matrix: usize;
  let mut l_line_matrix = 0;
  let mut l_begin_ptr = p_result.len() - 1;
  let mut l_generated_data = 0;
  let mut l_current_permutation_ptr = 0;
  l_intermediate_ptr = 0;
  l_generated_data = nb_compo - 1;
  i = 0;
  while i < nb_compo {
    sum = 0.0;
    l_current_ptr = 0;
    l_tmp_matrix = l_line_matrix;
    j = 1;
    while j <= i {
      sum += p_matrix[l_tmp_matrix] * p_intermediate_data[l_current_ptr];
      l_tmp_matrix += 1;
      l_current_ptr += 1;
      j += 1;
    }
    p_intermediate_data[l_intermediate_ptr] =
      p_vector[p_permutations[l_current_permutation_ptr] as usize] - sum;
    l_line_matrix += nb_compo;
    l_intermediate_ptr += 1;
    l_current_permutation_ptr += 1;
    i += 1;
  }
  l_line_matrix = p_matrix.len() - nb_compo - 1;
  l_dest_ptr = nb_compo;
  assert!(nb_compo != 0);
  k = nb_compo as i32 - 1;
  while k != -1 {
    sum = 0.0;
    l_tmp_matrix = l_line_matrix;
    u = p_matrix[l_tmp_matrix];
    l_line_matrix -= l_stride;
    l_current_ptr = l_dest_ptr;
    j = (k + 1) as usize;
    while j < nb_compo {
      sum += p_matrix[l_tmp_matrix] * p_result[l_current_ptr];
      l_tmp_matrix += 1;
      l_current_ptr += 1;
      j += 1;
    }
    p_result[l_begin_ptr] = (p_intermediate_data[l_generated_data] - sum) / u;
    l_generated_data -= 1;
    l_begin_ptr -= 1;
    l_dest_ptr -= 1;
    k -= 1;
  }
}
/* *
 *LUP inversion (call with the result of lupDecompose)
 */
fn opj_lupInvert(
  pSrcMatrix: &mut [f32],
  pDestMatrix: &mut [f32],
  nb_compo: usize,
  pPermutations: &[u32],
  p_src_temp: &mut [f32],
  p_dest_temp: &mut [f32],
  p_swap_area: &mut [f32],
) {
  let nb_compo_usize = nb_compo;
  let mut lLineMatrix = pDestMatrix;
  for j in 0..nb_compo {
    let lCurrentPtr = &mut lLineMatrix[j + 1..];
    p_src_temp.fill(0.0);
    p_src_temp[j] = 1.0;
    let p_src_temp_slice = &mut p_src_temp[..nb_compo_usize];
    let p_dest_temp_slice = &mut p_dest_temp[..nb_compo_usize];
    let pSrcMatrix_slice = &pSrcMatrix[..nb_compo_usize * nb_compo_usize];
    let pPermutations_slice = &pPermutations[..nb_compo_usize];
    let p_swap_area_slice = &mut p_swap_area[..nb_compo_usize];
    opj_lupSolve(
      p_dest_temp_slice,
      pSrcMatrix_slice,
      p_src_temp_slice,
      pPermutations_slice,
      nb_compo_usize,
      p_swap_area_slice,
    );
    for i in 0..nb_compo_usize {
      lCurrentPtr[i * nb_compo_usize] = p_dest_temp[i];
    }
  }
}

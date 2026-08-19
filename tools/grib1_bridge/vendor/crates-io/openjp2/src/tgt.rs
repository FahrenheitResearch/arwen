use super::bio::*;
use super::event::*;
use super::openjpeg::*;

use super::malloc::*;

extern "C" {
  fn memset(_: *mut core::ffi::c_void, _: core::ffi::c_int, _: usize) -> *mut core::ffi::c_void;
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
/*
==========================================================
   Tag-tree coder interface
==========================================================
*/

pub(crate) fn opj_tgt_create(
  numleafsh: OPJ_UINT32,
  numleafsv: OPJ_UINT32,
  p_manager: &mut opj_event_mgr,
) -> *mut opj_tgt_tree_t {
  unsafe {
    let mut nplh: [OPJ_INT32; 32] = [0; 32];
    let mut nplv: [OPJ_INT32; 32] = [0; 32];
    let mut node = core::ptr::null_mut::<opj_tgt_node_t>();
    let mut l_parent_node = core::ptr::null_mut::<opj_tgt_node_t>();
    let mut l_parent_node0 = core::ptr::null_mut::<opj_tgt_node_t>();
    let mut tree = core::ptr::null_mut::<opj_tgt_tree_t>();
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_INT32 = 0;
    let mut k: OPJ_INT32 = 0;
    let mut numlvls: OPJ_UINT32 = 0;
    let mut n: OPJ_UINT32 = 0;
    tree =
      opj_calloc(1i32 as size_t, core::mem::size_of::<opj_tgt_tree_t>()) as *mut opj_tgt_tree_t;
    if tree.is_null() {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to create Tag-tree\n",
      );
      return core::ptr::null_mut::<opj_tgt_tree_t>();
    }
    (*tree).numleafsh = numleafsh;
    (*tree).numleafsv = numleafsv;
    numlvls = 0 as OPJ_UINT32;
    nplh[0_usize] = numleafsh as OPJ_INT32;
    nplv[0_usize] = numleafsv as OPJ_INT32;
    (*tree).numnodes = 0 as OPJ_UINT32;
    loop {
      n = (nplh[numlvls as usize] * nplv[numlvls as usize]) as OPJ_UINT32;
      nplh[numlvls.wrapping_add(1u32) as usize] = (nplh[numlvls as usize] + 1i32) / 2i32;
      nplv[numlvls.wrapping_add(1u32) as usize] = (nplv[numlvls as usize] + 1i32) / 2i32;
      (*tree).numnodes = ((*tree).numnodes as core::ffi::c_uint).wrapping_add(n) as OPJ_UINT32;
      numlvls = numlvls.wrapping_add(1);
      if n <= 1u32 {
        break;
      }
    }
    /* ADD */
    if (*tree).numnodes == 0u32 {
      opj_free(tree as *mut core::ffi::c_void);
      return core::ptr::null_mut::<opj_tgt_tree_t>();
    }
    (*tree).nodes = opj_calloc(
      (*tree).numnodes as size_t,
      core::mem::size_of::<opj_tgt_node_t>(),
    ) as *mut opj_tgt_node_t;
    if (*tree).nodes.is_null() {
      event_msg!(
        p_manager,
        EVT_ERROR,
        "Not enough memory to create Tag-tree nodes\n",
      );
      opj_free(tree as *mut core::ffi::c_void);
      return core::ptr::null_mut::<opj_tgt_tree_t>();
    }
    (*tree).nodes_size = (*tree)
      .numnodes
      .wrapping_mul(core::mem::size_of::<opj_tgt_node_t>() as OPJ_UINT32);
    node = (*tree).nodes;
    l_parent_node = &mut *(*tree)
      .nodes
      .offset((*tree).numleafsh.wrapping_mul((*tree).numleafsv) as isize)
      as *mut opj_tgt_node_t;
    l_parent_node0 = l_parent_node;
    i = 0 as OPJ_UINT32;
    while i < numlvls.wrapping_sub(1u32) {
      j = 0i32;
      while j < nplv[i as usize] {
        k = nplh[i as usize];
        loop {
          k -= 1;
          if k < 0i32 {
            break;
          }
          (*node).parent = l_parent_node;
          node = node.offset(1);
          k -= 1;
          if k >= 0i32 {
            (*node).parent = l_parent_node;
            node = node.offset(1)
          }
          l_parent_node = l_parent_node.offset(1)
        }
        if j & 1i32 != 0 || j == nplv[i as usize] - 1i32 {
          l_parent_node0 = l_parent_node
        } else {
          l_parent_node = l_parent_node0;
          l_parent_node0 = l_parent_node0.offset(nplh[i as usize] as isize)
        }
        j += 1
      }
      i += 1;
    }
    (*node).parent = core::ptr::null_mut::<opj_tgt_node>();
    opj_tgt_reset(tree);
    tree
  }
}
/* *
 * Reinitialises a tag-tree from an existing one.
 *
 * @param       p_tree                          the tree to reinitialize.
 * @param       p_num_leafs_h           the width of the array of leafs of the tree
 * @param       p_num_leafs_v           the height of the array of leafs of the tree
 * @return      a new tag-tree if successful, NULL otherwise
*/

pub(crate) fn opj_tgt_init(
  p_tree: *mut opj_tgt_tree_t,
  p_num_leafs_h: OPJ_UINT32,
  p_num_leafs_v: OPJ_UINT32,
  p_manager: &mut opj_event_mgr,
) -> *mut opj_tgt_tree_t {
  unsafe {
    let mut l_nplh: [OPJ_INT32; 32] = [0; 32];
    let mut l_nplv: [OPJ_INT32; 32] = [0; 32];
    let mut l_node = core::ptr::null_mut::<opj_tgt_node_t>();
    let mut l_parent_node = core::ptr::null_mut::<opj_tgt_node_t>();
    let mut l_parent_node0 = core::ptr::null_mut::<opj_tgt_node_t>();
    let mut i: OPJ_UINT32 = 0;
    let mut j: OPJ_INT32 = 0;
    let mut k: OPJ_INT32 = 0;
    let mut l_num_levels: OPJ_UINT32 = 0;
    let mut n: OPJ_UINT32 = 0;
    let mut l_node_size: OPJ_UINT32 = 0;
    if p_tree.is_null() {
      return core::ptr::null_mut::<opj_tgt_tree_t>();
    }
    if (*p_tree).numleafsh != p_num_leafs_h || (*p_tree).numleafsv != p_num_leafs_v {
      (*p_tree).numleafsh = p_num_leafs_h;
      (*p_tree).numleafsv = p_num_leafs_v;
      l_num_levels = 0 as OPJ_UINT32;
      l_nplh[0_usize] = p_num_leafs_h as OPJ_INT32;
      l_nplv[0_usize] = p_num_leafs_v as OPJ_INT32;
      (*p_tree).numnodes = 0 as OPJ_UINT32;
      loop {
        n = (l_nplh[l_num_levels as usize] * l_nplv[l_num_levels as usize]) as OPJ_UINT32;
        l_nplh[l_num_levels.wrapping_add(1u32) as usize] =
          (l_nplh[l_num_levels as usize] + 1i32) / 2i32;
        l_nplv[l_num_levels.wrapping_add(1u32) as usize] =
          (l_nplv[l_num_levels as usize] + 1i32) / 2i32;
        (*p_tree).numnodes =
          ((*p_tree).numnodes as core::ffi::c_uint).wrapping_add(n) as OPJ_UINT32;
        l_num_levels = l_num_levels.wrapping_add(1);
        if n <= 1u32 {
          break;
        }
      }
      /* ADD */
      if (*p_tree).numnodes == 0u32 {
        opj_tgt_destroy(p_tree);
        return core::ptr::null_mut::<opj_tgt_tree_t>();
      }
      l_node_size = (*p_tree)
        .numnodes
        .wrapping_mul(core::mem::size_of::<opj_tgt_node_t>() as OPJ_UINT32);
      if l_node_size > (*p_tree).nodes_size {
        let mut new_nodes = opj_realloc(
          (*p_tree).nodes as *mut core::ffi::c_void,
          l_node_size as size_t,
        ) as *mut opj_tgt_node_t;
        if new_nodes.is_null() {
          event_msg!(
            p_manager,
            EVT_ERROR,
            "Not enough memory to reinitialize the tag tree\n",
          );
          opj_tgt_destroy(p_tree);
          return core::ptr::null_mut::<opj_tgt_tree_t>();
        }
        (*p_tree).nodes = new_nodes;
        memset(
          ((*p_tree).nodes as *mut core::ffi::c_char).offset((*p_tree).nodes_size as isize)
            as *mut core::ffi::c_void,
          0i32,
          l_node_size.wrapping_sub((*p_tree).nodes_size) as usize,
        );
        (*p_tree).nodes_size = l_node_size
      }
      l_node = (*p_tree).nodes;
      l_parent_node = &mut *(*p_tree)
        .nodes
        .offset((*p_tree).numleafsh.wrapping_mul((*p_tree).numleafsv) as isize)
        as *mut opj_tgt_node_t;
      l_parent_node0 = l_parent_node;
      i = 0 as OPJ_UINT32;
      while i < l_num_levels.wrapping_sub(1u32) {
        j = 0i32;
        while j < l_nplv[i as usize] {
          k = l_nplh[i as usize];
          loop {
            k -= 1;
            if k < 0i32 {
              break;
            }
            (*l_node).parent = l_parent_node;
            l_node = l_node.offset(1);
            k -= 1;
            if k >= 0i32 {
              (*l_node).parent = l_parent_node;
              l_node = l_node.offset(1)
            }
            l_parent_node = l_parent_node.offset(1)
          }
          if j & 1i32 != 0 || j == l_nplv[i as usize] - 1i32 {
            l_parent_node0 = l_parent_node
          } else {
            l_parent_node = l_parent_node0;
            l_parent_node0 = l_parent_node0.offset(l_nplh[i as usize] as isize)
          }
          j += 1
        }
        i += 1;
      }
      (*l_node).parent = core::ptr::null_mut::<opj_tgt_node>()
    }
    opj_tgt_reset(p_tree);
    p_tree
  }
}

pub(crate) fn opj_tgt_destroy(mut p_tree: *mut opj_tgt_tree_t) {
  unsafe {
    if p_tree.is_null() {
      return;
    }
    if !(*p_tree).nodes.is_null() {
      opj_free((*p_tree).nodes as *mut core::ffi::c_void);
      (*p_tree).nodes = core::ptr::null_mut::<opj_tgt_node_t>()
    }
    opj_free(p_tree as *mut core::ffi::c_void);
  }
}

pub(crate) fn opj_tgt_reset(mut p_tree: *mut opj_tgt_tree_t) {
  unsafe {
    let mut i: OPJ_UINT32 = 0;
    let mut l_current_node = core::ptr::null_mut::<opj_tgt_node_t>();
    if p_tree.is_null() {
      return;
    }
    l_current_node = (*p_tree).nodes;
    i = 0 as OPJ_UINT32;
    while i < (*p_tree).numnodes {
      (*l_current_node).value = 999i32;
      (*l_current_node).low = 0i32;
      (*l_current_node).known = 0 as OPJ_UINT32;
      l_current_node = l_current_node.offset(1);
      i += 1;
    }
  }
}

pub(crate) fn opj_tgt_setvalue(tree: *mut opj_tgt_tree_t, leafno: OPJ_UINT32, value: OPJ_INT32) {
  unsafe {
    let mut node = core::ptr::null_mut::<opj_tgt_node_t>();
    node = &mut *(*tree).nodes.offset(leafno as isize) as *mut opj_tgt_node_t;
    while !node.is_null() && (*node).value > value {
      (*node).value = value;
      node = (*node).parent
    }
  }
}

pub(crate) fn opj_tgt_encode(
  bio: *mut opj_bio_t,
  tree: *mut opj_tgt_tree_t,
  leafno: OPJ_UINT32,
  threshold: OPJ_INT32,
) {
  unsafe {
    let mut stk: [*mut opj_tgt_node_t; 31] = [core::ptr::null_mut::<opj_tgt_node_t>(); 31];
    let mut stkptr = core::ptr::null_mut::<*mut opj_tgt_node_t>();
    let mut node = core::ptr::null_mut::<opj_tgt_node_t>();
    let mut low: OPJ_INT32 = 0;
    stkptr = stk.as_mut_ptr();
    node = &mut *(*tree).nodes.offset(leafno as isize) as *mut opj_tgt_node_t;
    while !(*node).parent.is_null() {
      let fresh0 = stkptr;
      stkptr = stkptr.offset(1);
      *fresh0 = node;
      node = (*node).parent
    }
    low = 0i32;
    loop {
      if low > (*node).low {
        (*node).low = low
      } else {
        low = (*node).low
      }
      while low < threshold {
        if low >= (*node).value {
          if (*node).known == 0 {
            opj_bio_putbit(bio, 1);
            (*node).known = 1 as OPJ_UINT32
          }
          break;
        } else {
          opj_bio_putbit(bio, 0);
          low += 1
        }
      }
      (*node).low = low;
      if stkptr == stk.as_mut_ptr() {
        break;
      }
      stkptr = stkptr.offset(-1);
      node = *stkptr
    }
  }
}

pub(crate) fn opj_tgt_decode(
  bio: *mut opj_bio_t,
  tree: *mut opj_tgt_tree_t,
  leafno: OPJ_UINT32,
  threshold: OPJ_INT32,
) -> OPJ_UINT32 {
  unsafe {
    let mut stk: [*mut opj_tgt_node_t; 31] = [core::ptr::null_mut::<opj_tgt_node_t>(); 31];
    let mut stkptr = core::ptr::null_mut::<*mut opj_tgt_node_t>();
    let mut node = core::ptr::null_mut::<opj_tgt_node_t>();
    let mut low: OPJ_INT32 = 0;
    stkptr = stk.as_mut_ptr();
    node = &mut *(*tree).nodes.offset(leafno as isize) as *mut opj_tgt_node_t;
    while !(*node).parent.is_null() {
      let fresh1 = stkptr;
      stkptr = stkptr.offset(1);
      *fresh1 = node;
      node = (*node).parent
    }
    low = 0i32;
    loop {
      if low > (*node).low {
        (*node).low = low
      } else {
        low = (*node).low
      }
      while low < threshold && low < (*node).value {
        if opj_bio_read(bio, 1 as OPJ_UINT32) != 0 {
          (*node).value = low
        } else {
          low += 1
        }
      }
      (*node).low = low;
      if stkptr == stk.as_mut_ptr() {
        break;
      }
      stkptr = stkptr.offset(-1);
      node = *stkptr
    }
    (if (*node).value < threshold {
      1i32
    } else {
      0i32
    }) as OPJ_UINT32
  }
}

// SPDX-License-Identifier: Apache-2.0
//
// WOOF post-processor: the ONE kernel set (specification 3.5, D30).
//
// One translation unit holds every group's kernels, compiled once per
// architecture with --fmad=false --prec-div=true --prec-sqrt=true
// --ftz=false into woof_post_sm*.cubin and woof_post.ptx, described by
// woof_post.manifest.json and loaded by src/gpu.rs.
//
// - woof_math.cuh: the shared maths library (twin of src/math.rs).
// - post_group_a.cu: Group A, the shared column state (ST_*, SI_* slots)
//   every other group consumes, and the shelter and surface fields.
// - severe_c.cu, group_e.cu, group_d.cu, group_b.cu: Groups C, E, D and B, each in its own
//   namespace so their device helpers cannot collide.  Kernel entry points
//   are extern "C", so their names are unchanged.  Group B comes last
//   because its constants are plain macros.

#include "woof_math.cuh"
#include "post_group_a.cu"

namespace woof_c {
#include "severe_c.cu"
}

namespace woof_e {
#include "group_e.cu"
}

namespace woof_d {
#include "group_d.cu"
}

namespace woof_b {
#include "group_b.cu"
}

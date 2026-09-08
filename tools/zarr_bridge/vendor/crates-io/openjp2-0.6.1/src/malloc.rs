use super::openjpeg::*;

extern "C" {

  fn malloc(_: usize) -> *mut core::ffi::c_void;

  fn calloc(_: usize, _: usize) -> *mut core::ffi::c_void;

  fn realloc(_: *mut core::ffi::c_void, _: usize) -> *mut core::ffi::c_void;

  fn free(_: *mut core::ffi::c_void);

  fn memcpy(
    _: *mut core::ffi::c_void,
    _: *const core::ffi::c_void,
    _: usize,
  ) -> *mut core::ffi::c_void;
}

pub(crate) fn opj_malloc(mut size: size_t) -> *mut core::ffi::c_void {
  unsafe {
    if size == 0 {
      /* prevent implementation defined behavior of realloc */
      return core::ptr::null_mut::<core::ffi::c_void>();
    }
    malloc(size)
  }
}

pub(crate) fn opj_calloc(mut num: size_t, mut size: size_t) -> *mut core::ffi::c_void {
  unsafe {
    if num == 0 || size == 0 {
      /* prevent implementation defined behavior of realloc */
      return core::ptr::null_mut::<core::ffi::c_void>();
    }
    calloc(num, size)
  }
}

pub(crate) fn opj_realloc(
  mut ptr: *mut core::ffi::c_void,
  mut new_size: size_t,
) -> *mut core::ffi::c_void {
  unsafe {
    if new_size == 0 {
      /* prevent implementation defined behavior of realloc */
      return core::ptr::null_mut::<core::ffi::c_void>();
    }
    realloc(ptr, new_size)
  }
}

pub(crate) fn opj_free(mut ptr: *mut core::ffi::c_void) {
  unsafe {
    free(ptr);
  }
}

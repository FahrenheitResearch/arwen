macro_rules! fprintf {
  ($file:expr, $fmt:expr, $($arg:expr),* $(,)?) => {
    let s = ::sprintf::sprintf!($fmt, $($arg),*);
    match &s {
      Ok(s) => {
        let bytes = s.as_bytes();
        let len = bytes.len();
        let nb = libc::fwrite(bytes.as_ptr() as *const libc::c_void, 1, len, $file);
        nb
      },
      Err(err) => {
        log::error!("sprintf failed: {err:?}");
        0
      }
    }
  };
}

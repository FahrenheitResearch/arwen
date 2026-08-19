use std::ffi::*;
use std::fs::*;
use std::io::*;

use openjp2::openjpeg::*;

extern "C" fn log_info(msg: *const c_char, _data: *mut c_void) {
  unsafe {
    eprintln!("[INFO] {:?}", CStr::from_ptr(msg).to_string_lossy());
  }
}

extern "C" fn log_warn(msg: *const c_char, _data: *mut c_void) {
  unsafe {
    eprintln!("[WARN] {:?}", CStr::from_ptr(msg).to_string_lossy());
  }
}

extern "C" fn log_error(msg: *const c_char, _data: *mut c_void) {
  unsafe {
    eprintln!("[ERROR] {:?}", CStr::from_ptr(msg).to_string_lossy());
  }
}

struct WrappedSlice<'a> {
  offset: usize,
  buf: &'a [u8],
}

impl<'a> WrappedSlice<'a> {
  fn new(buf: &'a [u8]) -> Box<Self> {
    Box::new(Self { offset: 0, buf })
  }

  fn remaining(&self) -> usize {
    self.buf.len() - self.offset
  }

  fn seek(&mut self, new_offset: usize) -> usize {
    // Make sure `new_offset <= buf.len()`
    self.offset = std::cmp::min(self.buf.len(), new_offset);
    self.offset
  }

  fn consume(&mut self, n_bytes: usize) -> usize {
    let offset = self.offset.saturating_add(n_bytes);
    // Make sure `offset <= buf.len()`
    self.offset = self.buf.len().min(offset);
    self.offset
  }

  fn read_into(&mut self, out_buffer: &mut [u8]) -> Option<usize> {
    // Get number of remaining bytes.
    let remaining = self.remaining();
    if remaining == 0 {
      // No more bytes.
      return None;
    }

    // Try to fill the output buffer.
    let n_read = std::cmp::min(remaining, out_buffer.len());
    let offset = self.offset;
    let end_off = self.consume(n_read);
    out_buffer[0..n_read].copy_from_slice(&self.buf[offset..end_off]);

    Some(n_read)
  }
}

extern "C" fn buf_read_stream_free_fn(p_data: *mut c_void) {
  let ptr = p_data as *mut WrappedSlice;
  drop(unsafe { Box::from_raw(ptr) })
}

extern "C" fn buf_read_stream_read_fn(
  p_buffer: *mut c_void,
  nb_bytes: usize,
  p_data: *mut c_void,
) -> usize {
  if p_buffer.is_null() || nb_bytes == 0 {
    return usize::MAX;
  }

  let slice = unsafe { &mut *(p_data as *mut WrappedSlice) };
  let out_buf = unsafe { std::slice::from_raw_parts_mut(p_buffer as *mut u8, nb_bytes) };
  slice.read_into(out_buf).unwrap_or(usize::MAX)
}

extern "C" fn buf_read_stream_skip_fn(nb_bytes: i64, p_data: *mut c_void) -> i64 {
  let slice = unsafe { &mut *(p_data as *mut WrappedSlice) };
  slice.consume(nb_bytes as usize) as i64
}

extern "C" fn buf_read_stream_seek_fn(nb_bytes: i64, p_data: *mut c_void) -> i32 {
  let slice = unsafe { &mut *(p_data as *mut WrappedSlice) };
  let seek_offset = nb_bytes as usize;
  let new_offset = slice.seek(seek_offset);

  // Return true if the seek worked.
  if seek_offset == new_offset {
    1
  } else {
    0
  }
}

#[test]
fn decode_byte_stream() {
  let mut params = opj_dparameters_t::default();

  // Read file.
  let mut file = File::open("samples/Hadley_Crater.jp2").expect("Open file");
  let mut bytes = Vec::new();
  file.read_to_end(&mut bytes).expect("Read file");
  let len = bytes.len();
  let data = WrappedSlice::new(bytes.as_slice());
  let stream = unsafe {
    let p_data = Box::into_raw(data) as *mut c_void;
    let stream = opj_stream_default_create(1);
    opj_stream_set_read_function(stream, Some(buf_read_stream_read_fn));
    opj_stream_set_skip_function(stream, Some(buf_read_stream_skip_fn));
    opj_stream_set_seek_function(stream, Some(buf_read_stream_seek_fn));
    opj_stream_set_user_data_length(stream, len as u64);
    opj_stream_set_user_data(stream, p_data, Some(buf_read_stream_free_fn));

    stream
  };

  // Create decoder.
  let codec = opj_create_decompress(OPJ_CODEC_JP2);

  unsafe {
    let null = core::ptr::null_mut();
    opj_set_info_handler(codec, Some(log_info), null);
    opj_set_warning_handler(codec, Some(log_warn), null);
    opj_set_error_handler(codec, Some(log_error), null);
  }

  // setup decoder.
  let ret = unsafe { opj_setup_decoder(codec, &mut params) };
  assert!(ret != 0);

  // read header
  let mut image = std::ptr::null_mut() as *mut opj_image_t;
  let ret = unsafe { opj_read_header(stream, codec, &mut image) };
  assert!(ret != 0);

  // Decode the image.
  let ret = unsafe { opj_decode(codec, stream, image) };
  assert!(ret != 0);

  // End Decode.
  let ret = unsafe { opj_end_decompress(codec, stream) };
  assert!(ret != 0);

  if !codec.is_null() {
    unsafe {
      opj_destroy_codec(codec);
    }
  }

  if !stream.is_null() {
    unsafe {
      opj_stream_destroy(stream);
    }
  }

  if !image.is_null() {
    opj_image_destroy(image);
  }
}

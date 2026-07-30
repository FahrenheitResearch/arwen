use std::ffi::*;

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

#[test]
fn decode_sample_file_hadley_crater() {
  let mut params = opj_dparameters_t::default();

  // Open file stream.
  let filename = b"samples/Hadley_Crater.jp2\x00";
  let stream = unsafe { opj_stream_create_default_file_stream(filename.as_ptr() as _, 1) };

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

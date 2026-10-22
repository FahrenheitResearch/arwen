use snappy_src::{
    snappy_compress, snappy_max_compressed_length, snappy_status_SNAPPY_OK, snappy_uncompress,
    snappy_uncompressed_length,
};

fn compress(src: &[u8]) -> Vec<u8> {
    let mut len = unsafe { snappy_max_compressed_length(src.len()) };
    let mut dst = Vec::<u8>::with_capacity(len);
    unsafe {
        assert_eq!(
            snappy_status_SNAPPY_OK,
            snappy_compress(
                src.as_ptr().cast(),
                src.len(),
                dst.as_mut_ptr().cast(),
                &mut len,
            )
        );
        dst.set_len(len as usize);
    }
    dst
}

fn uncompress(src: &[u8]) -> Vec<u8> {
    let mut len = 0;
    assert_eq!(snappy_status_SNAPPY_OK, unsafe {
        snappy_uncompressed_length(src.as_ptr().cast(), src.len(), &mut len)
    });
    let mut dst = Vec::<u8>::with_capacity(len);
    unsafe {
        assert_eq!(
            snappy_status_SNAPPY_OK,
            snappy_uncompress(
                src.as_ptr().cast(),
                src.len(),
                dst.as_mut_ptr().cast(),
                &mut len,
            )
        );
        dst.set_len(len as usize);
    }
    dst
}

#[cfg(test)]
#[test]
fn round_trip() {
    let src = (0..100).collect::<Vec<u8>>();
    let dst = compress(&src);
    let copy = uncompress(&dst);
    assert_eq!(src, copy);
}

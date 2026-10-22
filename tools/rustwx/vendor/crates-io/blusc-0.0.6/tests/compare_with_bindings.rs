use blosc2_src::{
    blosc2_init as bound_blosc2_init,
    blosc2_compress as bound_blosc2_compress,
    blosc2_decompress as bound_blosc2_decompress,
    blosc2_create_cctx as bound_blosc2_create_cctx,
    blosc2_compress_ctx as bound_blosc2_compress_ctx,
    blosc2_create_dctx as bound_blosc2_create_dctx,
    blosc2_decompress_ctx as bound_blosc2_decompress_ctx,
    blosc2_cbuffer_sizes as bound_blosc2_cbuffer_sizes,
    blosc2_destroy as bound_blosc2_destroy,
    BLOSC_NOSHUFFLE as BOUND_BLOSC_NOSHUFFLE,
    BLOSC2_MAX_OVERHEAD as BOUND_BLOSC2_MAX_OVERHEAD,
    BLOSC2_CPARAMS_DEFAULTS as BOUND_BLOSC2_CPARAMS_DEFAULTS,
    BLOSC2_DPARAMS_DEFAULTS as BOUND_BLOSC2_DPARAMS_DEFAULTS,
};
use blusc::api::{
    blosc2_compress as blusc_blosc2_compress,
    blosc2_decompress as blusc_blosc2_decompress,
    blosc2_create_cctx as blusc_blosc2_create_cctx,
    blosc2_compress_ctx as blusc_blosc2_compress_ctx,
    blosc2_create_dctx as blusc_blosc2_create_dctx,
    blosc2_decompress_ctx as blusc_blosc2_decompress_ctx,
    BLOSC2_CPARAMS_DEFAULTS as BLUSC_BLOSC2_CPARAMS_DEFAULTS,
    BLOSC2_DPARAMS_DEFAULTS as BLUSC_BLOSC2_DPARAMS_DEFAULTS,
};

use ctor::{ctor, dtor};

#[ctor]
fn blosc2_init() {
    unsafe {
        bound_blosc2_init();
    }
}

#[dtor]
fn blosc2_cleanup() {
    unsafe {
        bound_blosc2_destroy();
    }
}

#[test]
fn compare_compressed_bytes() {
    unsafe {

        let text =
            "I am here writing some very cool and novel words which I will compress and check";

        let bytes = text.as_bytes();

        let mut compressed_bound = vec![0; bytes.len() * 2];
        let stat_bound = bound_blosc2_compress(
            0,
            BOUND_BLOSC_NOSHUFFLE as _,
            std::mem::size_of::<u8>() as i32,
            bytes.as_ptr().cast(),
            bytes.len() as i32,
            compressed_bound.as_mut_ptr().cast(),
            compressed_bound.len() as i32,
        );
        assert!(stat_bound > 0);
        compressed_bound.truncate(stat_bound as usize);

        let mut compressed_blusc = vec![0; bytes.len() * 2];
        let stat_blusc: i32 = blusc_blosc2_compress(
            0,
            BOUND_BLOSC_NOSHUFFLE as _,
            std::mem::size_of::<u8>(),
            bytes,
            &mut compressed_blusc,
        );
        assert!(stat_blusc > 0);
        compressed_blusc.truncate(stat_blusc as usize);

        assert_eq!(compressed_bound, compressed_blusc);

    }
}

#[test]
fn roundtrip_blosc_compress_then_blusc_decompress() {
    unsafe {

        let text =
            "I am here writing some very cool and novel words which I will compress and decompress";

        let bytes = text.as_bytes();

        let mut compressed = vec![0; bytes.len() * 2];

        let stat = bound_blosc2_compress(
            0,
            BOUND_BLOSC_NOSHUFFLE as _,
            std::mem::size_of::<u8>() as i32,
            bytes.as_ptr().cast(),
            bytes.len() as i32,
            compressed.as_mut_ptr().cast(),
            compressed.len() as i32,
        );
        assert!(stat > 0);

        let mut outtext = vec![0_u8; bytes.len()];
        let stat = blusc_blosc2_decompress(
            &compressed,
            &mut outtext,
        );
        assert!(stat > 0);

        assert_eq!(text, std::str::from_utf8(&outtext).unwrap());

    }
}

#[test]
fn roundtrip_blusc_compress_then_blosc_decompress() {
    unsafe {

        let text =
            "I am here writing some very cool and novel words which I will compress and decompress";

        let bytes = text.as_bytes();

        let mut compressed = vec![0; bytes.len() * 2 + BOUND_BLOSC2_MAX_OVERHEAD as usize];

        let stat = blusc_blosc2_compress(
            5,
            BOUND_BLOSC_NOSHUFFLE as _,
            std::mem::size_of::<u8>(),
            bytes,
            &mut compressed,
        );
        assert!(stat > 0);

        let mut outtext = vec![0_u8; bytes.len()];
        let stat = bound_blosc2_decompress(
            compressed.as_ptr().cast(),
            stat,
            outtext.as_mut_ptr().cast(),
            outtext.len() as i32,
        );
        assert!(stat > 0);

        assert_eq!(text, std::str::from_utf8(&outtext).unwrap());

    }
}

#[test]
fn floats_roundtrip_blosc_compress_then_blusc_decompress() {
    // generate numerical data
    let src: Vec<f32> = (0..10000)
        .map(|num| ((num * 8923) % 100) as f32 / 2f32) // multiply by big prime number
        .collect();

    // compress
    let dest: Vec<u8> = {
        let typesize = std::mem::size_of::<f32>();
        let src_size = src.len() * typesize;
        let dest_size = src_size + BOUND_BLOSC2_MAX_OVERHEAD as usize;
        let mut dest = vec![0; dest_size];

        let rsize = unsafe {
            let mut cparams = BOUND_BLOSC2_CPARAMS_DEFAULTS;
            cparams.clevel = 0;
            cparams.typesize = typesize as i32;
            let context = bound_blosc2_create_cctx(cparams);

            bound_blosc2_compress_ctx(
                context,
                src.as_ptr().cast(),
                src_size as i32,
                dest.as_mut_ptr().cast(),
                dest_size as i32,
            )
        };

        assert!(rsize > 0);
        dest.drain(rsize as usize..);
        dest
    };

    // make sure it actually compresses

    // decompress
    let result = {
        let mut nbytes: i32 = 0;
        let mut _cbytes: i32 = 0;
        let mut _blocksize: i32 = 0;
        unsafe {
            bound_blosc2_cbuffer_sizes(
                dest.as_ptr().cast(),
                &mut nbytes,
                &mut _cbytes,
                &mut _blocksize,
            )
        };
        assert!(nbytes != 0);
        let dest_size = nbytes / std::mem::size_of::<f32>() as i32;
        let mut result = vec![0f32; dest_size as usize];
        let error = unsafe {
            let dparams = BLUSC_BLOSC2_DPARAMS_DEFAULTS;
            let context = blusc_blosc2_create_dctx(dparams);
            let result_bytes = std::slice::from_raw_parts_mut(
                result.as_mut_ptr() as *mut u8,
                result.len() * std::mem::size_of::<f32>(),
            );
            blusc_blosc2_decompress_ctx(
                &context,
                &dest,
                result_bytes,
            )
        };
        assert!(error >= 1);
        result
    };

    // check if the values in both arrays are equal
    assert_eq!(src, result);
}


#[test]
fn floats_roundtrip_blusc_compress_then_blosc_decompress() {
    // generate numerical data
    let src: Vec<f32> = (0..10000)
        .map(|num| ((num * 8923) % 100) as f32 / 2f32) // multiply by big prime number
        .collect();

    // compress
    let dest: Vec<u8> = {
        let typesize = std::mem::size_of::<f32>();
        let src_size = src.len() * typesize;
        let dest_size = src_size + BOUND_BLOSC2_MAX_OVERHEAD as usize;
        let mut dest = vec![0; dest_size];

        let rsize = {
            let mut cparams = BLUSC_BLOSC2_CPARAMS_DEFAULTS;
            cparams.clevel = 5;
            cparams.typesize = typesize as i32;
            let context = blusc_blosc2_create_cctx(cparams);
            
            let src_bytes = unsafe {
                std::slice::from_raw_parts(
                    src.as_ptr() as *const u8,
                    src_size
                )
            };

            blusc_blosc2_compress_ctx(
                &context,
                src_bytes,
                &mut dest,
            )
        };

        assert!(rsize > 0);
        println!("Rust compressed to {} bytes", rsize);
        println!("Header: {:02x?}", &dest[0..16]);
        println!("First 32 bytes of compressed data: {:02x?}", &dest[16..48.min(rsize as usize)]);

        /*
        // DEBUG: Compress with Blosc (C) to compare
        unsafe {
            let mut cparams = BOUND_BLOSC2_CPARAMS_DEFAULTS;
            cparams.clevel = 5;
            cparams.typesize = typesize as i32;
            cparams.compcode = 0; // BLOSCLZ
            let context = bound_blosc2_create_cctx(cparams);
            
            let mut c_dest = vec![0u8; dest_size];
            let c_rsize = bound_blosc2_compress_ctx(
                context,
                src.as_ptr() as *const std::ffi::c_void,
                src_size as i32,
                c_dest.as_mut_ptr() as *mut std::ffi::c_void,
                c_dest.len() as i32,
            );
            println!("C compressed to {} bytes", c_rsize);
            if c_rsize > 0 {
                println!("C Header: {:02x?}", &c_dest[0..16]);
                println!("C First 32 bytes of compressed data: {:02x?}", &c_dest[16..48.min(c_rsize as usize)]);
                
                // Compare headers
                if dest[0..32] != c_dest[0..32] {
                    println!("Headers differ!");
                    println!("Rust: {:02x?}", &dest[0..32]);
                    println!("C:    {:02x?}", &c_dest[0..32]);
                }
                
                // Compare bstarts
                if dest[32..36] != c_dest[32..36] {
                    println!("Bstarts differ!");
                    println!("Rust: {:02x?}", &dest[32..36]);
                    println!("C:    {:02x?}", &c_dest[32..36]);
                }
                
                // Compare compressed size
                if dest[36..40] != c_dest[36..40] {
                    println!("Compressed size field differs!");
                    println!("Rust: {:02x?}", &dest[36..40]);
                    println!("C:    {:02x?}", &c_dest[36..40]);
                }
                
                // Compare data
                // Note: data might differ if compression algorithm is slightly different
            }
        }
        */

        dest.into_iter().take(rsize as usize).collect()
    };

    // decompress
    let result = {
        let mut nbytes: i32 = 0;
        let mut _cbytes: i32 = 0;
        let mut _blocksize: i32 = 0;
        unsafe {
            bound_blosc2_cbuffer_sizes(
                dest.as_ptr().cast(),
                &mut nbytes,
                &mut _cbytes,
                &mut _blocksize,
            )
        };
        assert!(nbytes != 0);
        let dest_size = nbytes / std::mem::size_of::<f32>() as i32;
        let mut result = vec![0f32; dest_size as usize];
        
        let error = unsafe {
            let dparams = BOUND_BLOSC2_DPARAMS_DEFAULTS;
            let context = bound_blosc2_create_dctx(dparams);
            
            bound_blosc2_decompress_ctx(
                context,
                dest.as_ptr().cast(),
                dest.len() as i32,
                result.as_mut_ptr().cast(),
                result.len() as i32 * std::mem::size_of::<f32>() as i32,
            )
        };
        println!("error code: {}", error);
        assert!(error >= 1);
        result
    };

    // check if the values in both arrays are equal
    assert_eq!(src, result);
}

#[cfg(test)]
mod tests {
    use blosc2_src::{
        blosc2_compress as bound_blosc2_compress,
        blosc2_destroy as bound_blosc2_destroy, blosc2_init as bound_blosc2_init,
    };
    use blusc::api::blosc2_compress as blusc_blosc2_compress;
    use blusc::BLOSC2_MAX_OVERHEAD;

    #[test]
    fn debug_clevel0() {
        unsafe { bound_blosc2_init(); }
        
        for &(ts, ne, cl, ds) in &[
            (1usize, 8000usize, 0i32, 1i32),
            (1, 8000, 0, 0),
            (4, 8000, 0, 1),
            (1, 8000, 1, 1),
            (1, 8000, 9, 1),
        ] {
            let buffer_size = ts * ne;
            let dest_size = buffer_size + BLOSC2_MAX_OVERHEAD;
            let mut original = vec![0u8; buffer_size];
            for (k, byte) in original.iter_mut().enumerate() {
                *byte = k as u8;
            }
            
            let mut cb = vec![0u8; dest_size];
            let cs_b = blusc_blosc2_compress(cl, ds, ts, &original, &mut cb);
            
            let mut cc = vec![0u8; dest_size];
            let cs_c = unsafe {
                bound_blosc2_compress(cl, ds, ts as i32,
                    original.as_ptr().cast(), original.len() as i32,
                    cc.as_mut_ptr().cast(), cc.len() as i32)
            };
            
            let matches = if cs_b as i32 == cs_c {
                let len = cs_b as usize;
                cb[..len] == cc[..len]
            } else { false };
            
            eprintln!("ts={} ne={} cl={} ds={}: blusc_size={} c_size={} match={}",
                ts, ne, cl, ds, cs_b, cs_c, matches);
            
            if !matches {
                let len = (cs_b as usize).min(cs_c as usize).min(64);
                eprintln!("  blusc header: {:02x?}", &cb[..len]);
                eprintln!("  bound header: {:02x?}", &cc[..len]);
                for i in 0..(cs_b as usize).min(cs_c as usize) {
                    if cb[i] != cc[i] {
                        eprintln!("  first diff at byte {}: blusc={:#04x} c={:#04x}", i, cb[i], cc[i]);
                        break;
                    }
                }
            }
        }
        
        unsafe { bound_blosc2_destroy(); }
    }
}


#[cfg(test)]
mod tests {
    use blusc::filters::bitshuffle;

    #[test]
    fn debug_bitshuffle_large() {
        let size = 200000;
        let typesize = 8;
        let blocksize = size * typesize;
        let mut src = vec![0u8; blocksize];
        
        // Initialize with f64 values 0.0 .. 199999.0
        for i in 0..size {
            let val = i as f64;
            let bytes = val.to_le_bytes();
            for j in 0..8 {
                src[i * 8 + j] = bytes[j];
            }
        }

        let mut dest = vec![0u8; blocksize];
        bitshuffle(typesize, blocksize, &src, &mut dest).unwrap();

        // Compress Split 6
        let split_idx = 6;
        let start = split_idx * size;
        let end = start + size;
        let chunk = &dest[start..end];
        
        let mut compressed = vec![0u8; size + size/2]; // ample space
        let csize = blusc::codecs::blosclz::compress(1, chunk, &mut compressed);
        println!("Split 6 compressed size: {}", csize);
        
        // Compress Split 7
        let split_idx = 7;
        let start = split_idx * size;
        let end = start + size;
        let chunk = &dest[start..end];
        let csize = blusc::codecs::blosclz::compress(1, chunk, &mut compressed);
        println!("Split 7 compressed size: {}", csize);
        
        // Compress Split 0 (zeros)
        let split_idx = 0;
        let start = split_idx * size;
        let end = start + size;
        let chunk = &dest[start..end];
        let csize = blusc::codecs::blosclz::compress(1, chunk, &mut compressed);
        println!("Split 0 compressed size: {}", csize);
    }
}

use blusc::codecs::blosclz;

#[test]
fn test_blosclz_compression_ratio() {
    // Create data with lots of runs (should compress well)
    let mut data = vec![0u8; 10000];
    for i in 0..10000 {
        data[i] = (i / 100) as u8; // 100 identical bytes at a time
    }
    
    let mut compressed = vec![0u8; 20000];
    let csize = blosclz::compress(5, &data, &mut compressed);
    
    println!("Original size: {}", data.len());
    println!("Compressed size: {}", csize);
    println!("Compression ratio: {:.2}", data.len() as f64 / csize as f64);
    
    assert!(csize < data.len() / 2, "Compression should be better than 2:1 for this data");
}

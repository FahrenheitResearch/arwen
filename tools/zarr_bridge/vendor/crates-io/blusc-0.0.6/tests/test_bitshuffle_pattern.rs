use blusc::filters::bitshuffle;

#[test]
fn test_bitshuffle_pattern() {
    let type_size = 8;
    let num_elements = 10000;
    let buffer_size = type_size * num_elements;
    let mut original = vec![0u8; buffer_size];
    
    // Fill with sequential data
    for j in 0..buffer_size {
        original[j] = (j % 255) as u8;
    }

    let mut dest = vec![0u8; buffer_size];
    let result = bitshuffle(type_size, buffer_size, &original, &mut dest);
    assert!(result.is_ok());
    
    // Print first 100 bytes of bitshuffled output
    println!("First 100 bytes of bitshuffled data:");
    for i in 0..10 {
        println!("{:3}: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} {:02x}",
            i*10,
            dest[i*10], dest[i*10+1], dest[i*10+2], dest[i*10+3], dest[i*10+4],
            dest[i*10+5], dest[i*10+6], dest[i*10+7], dest[i*10+8], dest[i*10+9]);
    }
    
    // Check for runs of identical bytes (good for compression)
    let mut run_count = 0;
    let mut prev = dest[0];
    for &b in &dest[1..1000] {
        if b == prev {
            run_count += 1;
        }
        prev = b;
    }
    println!("\nRuns of identical bytes in first 1000: {}", run_count);
}

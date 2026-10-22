use blusc::filters::{bitshuffle, bitunshuffle};

#[test]
fn test_bitshuffle_large() {
    let size = 10000;
    let typesize = 8;
    let blocksize = size * typesize;
    let mut src = vec![0u8; blocksize];
    for i in 0..blocksize {
        src[i] = (i % 255) as u8;
    }
    let mut dest = vec![0u8; blocksize];
    let mut recovered = vec![0u8; blocksize];

    let result = bitshuffle(typesize, blocksize, &src, &mut dest);
    println!("Bitshuffle result: {:?}", result);
    assert!(result.is_ok(), "Bitshuffle failed: {:?}", result);
    
    let result2 = bitunshuffle(typesize, blocksize, &dest, &mut recovered);
    println!("Bitunshuffle result: {:?}", result2);
    assert!(result2.is_ok(), "Bitunshuffle failed: {:?}", result2);

    assert_eq!(src, recovered, "Bitshuffle roundtrip failed");
}

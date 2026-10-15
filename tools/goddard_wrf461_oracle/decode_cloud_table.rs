//! Decode the pinned WRF direct-access cloud table without numerical changes.
//! Input storage is property, SW/LW, band, species, radius, radius contiguous.
use std::{env,fs,io::{self,Write}};
const WORDS: usize = 3*2*11*6*109;
fn decode(bytes: &[u8]) -> Result<Vec<u8>, String> {
    if bytes.len()!=WORDS*4+24 {return Err(format!("cloud table: expected {} bytes, found {}",WORDS*4+24,bytes.len()));}
    // WRF direct reads consume three records and ignore the final 24 bytes.
    let mut output=Vec::with_capacity(WORDS*4);
    for (index, word) in bytes[..WORDS*4].chunks_exact(4).enumerate() {
        let bits=u32::from_be_bytes(word.try_into().unwrap());
        if !f32::from_bits(bits).is_finite() {return Err(format!("cloud table: non-finite word {index}"));}
        output.extend_from_slice(&bits.to_le_bytes());
    }
    Ok(output)
}
fn main() -> Result<(),Box<dyn std::error::Error>> {
    let path=env::args().nth(1).ok_or("usage: rw_lookup_table CLOUD_TABLE")?;
    let output=decode(&fs::read(path)?).map_err(io::Error::other)?;
    io::stdout().lock().write_all(&output)?;
    Ok(())
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn rejects_truncated_record() {assert!(decode(&[0;12]).unwrap_err().contains("expected"));}
    #[test] fn rejects_nan_word() {
        let mut words=vec![0;WORDS*4+24];words[40..44].copy_from_slice(&f32::NAN.to_bits().to_be_bytes());
        assert!(decode(&words).unwrap_err().contains("word 10"));
    }
    #[test] fn preserves_smallest_subnormal_word() {
        let mut words=vec![0;WORDS*4+24];words[..4].copy_from_slice(&1u32.to_be_bytes());
        assert_eq!(&decode(&words).unwrap()[..4],&1u32.to_le_bytes());
    }
}

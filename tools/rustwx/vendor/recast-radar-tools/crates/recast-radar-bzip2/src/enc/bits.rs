//! MSB-first bit writer appending to a `Vec<u8>`.

pub(crate) struct BitWriter<'a> {
    out: &'a mut Vec<u8>,
    /// Length of `out` when the writer was created.
    start: usize,
    /// Pending bits in the low `n` bits (higher bits are stale).
    acc: u64,
    n: u32,
}

impl<'a> BitWriter<'a> {
    pub(crate) fn new(out: &'a mut Vec<u8>) -> Self {
        let start = out.len();
        BitWriter {
            out,
            start,
            acc: 0,
            n: 0,
        }
    }

    /// Bits written so far.
    pub(crate) fn bit_len(&self) -> u64 {
        (self.out.len() - self.start) as u64 * 8 + u64::from(self.n)
    }

    /// Append the low `nbits` bits of `v` (`nbits <= 32`, higher bits of `v`
    /// zero).
    #[inline(always)]
    pub(crate) fn put(&mut self, nbits: u32, v: u32) {
        debug_assert!(nbits <= 32 && (nbits == 32 || v >> nbits == 0));
        self.acc = (self.acc << nbits) | u64::from(v);
        self.n += nbits;
        if self.n >= 32 {
            self.n -= 32;
            let w = (self.acc >> self.n) as u32;
            self.out.extend_from_slice(&w.to_be_bytes());
        }
    }

    /// Append the 48-bit magic, most significant bit first.
    pub(crate) fn put48(&mut self, v: u64) {
        self.put(24, (v >> 24) as u32 & 0xff_ffff);
        self.put(24, v as u32 & 0xff_ffff);
    }

    /// Pad with zero bits to a byte boundary and write out what is left.
    pub(crate) fn finish(mut self) {
        let pad = (8 - self.n % 8) % 8;
        self.acc <<= pad;
        self.n += pad;
        while self.n >= 8 {
            self.n -= 8;
            self.out.push((self.acc >> self.n) as u8);
        }
    }
}

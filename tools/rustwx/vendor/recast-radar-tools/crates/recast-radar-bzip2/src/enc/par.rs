//! Parallel compression of independent inputs (feature `rayon`).

use std::sync::{Mutex, PoisonError};

use rayon::prelude::*;

use super::{Encoder, Level};

/// Compress every input into its own bzip2 stream, in parallel on the
/// current rayon pool. `result[i]` is exactly what
/// [`Encoder::encode_into`] appends for `inputs[i]`.
///
/// This is the shape of NEXRAD Level II LDM records: one bzip2 stream per
/// record. The call runs on a fresh [`EncoderPool`], so it creates at most
/// one encoder per worker thread, each with its own work buffers (about
/// 20 MB at level 9) reused for every input it compresses, and frees them on
/// return. To compress batch after batch (volume after volume, or real-time
/// chunks as they fill), keep an [`EncoderPool`] and call
/// [`EncoderPool::encode_many`] so the buffers are allocated once.
pub fn encode_many<T: AsRef<[u8]> + Sync>(level: Level, inputs: &[T]) -> Vec<Vec<u8>> {
    EncoderPool::new(level).encode_many(inputs)
}

/// Encoders shared by the threads of a rayon pool, for compressing batches
/// of independent inputs in parallel (feature `rayon`).
///
/// A worker takes an idle encoder, or creates one when none is idle, for
/// each stretch of inputs rayon hands it, and puts it back when the stretch
/// is done. The pool therefore holds at most one encoder per thread that
/// has worked for it, each with its level's work buffers (about 22 bytes per
/// byte of block capacity, 20 MB at level 9), and reuses them across
/// inputs and across calls. Drop the pool to free them.
///
/// ```
/// use recast_radar_bzip2::{EncoderPool, Level};
///
/// /// The LDM records of each volume, compressed in parallel with the same
/// /// work buffers for every volume.
/// fn compress_volumes(volumes: &[Vec<Vec<u8>>]) -> Vec<Vec<Vec<u8>>> {
///     let pool = EncoderPool::new(Level::BEST);
///     volumes.iter().map(|records| pool.encode_many(records)).collect()
/// }
/// ```
pub struct EncoderPool {
    level: Level,
    idle: Mutex<Vec<Encoder>>,
}

impl EncoderPool {
    /// An empty pool for `level`: encoders are created as workers need them.
    pub fn new(level: Level) -> EncoderPool {
        EncoderPool {
            level,
            idle: Mutex::new(Vec::new()),
        }
    }

    /// The block size the pool's encoders write.
    pub fn level(&self) -> Level {
        self.level
    }

    /// Compress every input into its own bzip2 stream, in parallel on the
    /// current rayon pool. `result[i]` is exactly what
    /// [`Encoder::encode_into`] appends for `inputs[i]`.
    pub fn encode_many<T: AsRef<[u8]> + Sync>(&self, inputs: &[T]) -> Vec<Vec<u8>> {
        inputs
            .par_iter()
            .map_init(
                // Called once per stretch of inputs a worker takes (rayon
                // splits the work into more stretches than threads), so the
                // encoder comes from the pool rather than being created here.
                || Lease::take(self),
                |lease, input| {
                    let mut out = Vec::new();
                    lease.encoder.encode_into(input.as_ref(), &mut out);
                    out
                },
            )
            .collect()
    }

    /// Encoders the pool holds, each with its level's work buffers once it
    /// has compressed an input. Between calls this is every encoder the
    /// pool has created: at most the largest number of threads that have
    /// run one of its calls.
    pub fn idle_encoders(&self) -> usize {
        self.lock().len()
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Encoder>> {
        // The lock is held only to push or pop, which cannot panic, so a
        // poisoned lock still guards a valid list.
        self.idle.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl core::fmt::Debug for EncoderPool {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EncoderPool")
            .field("level", &self.level)
            .field("idle_encoders", &self.idle_encoders())
            .finish()
    }
}

/// An encoder taken from a pool for one stretch of inputs, returned to the
/// pool when the stretch is done.
struct Lease<'a> {
    pool: &'a EncoderPool,
    encoder: Encoder,
}

impl<'a> Lease<'a> {
    fn take(pool: &'a EncoderPool) -> Lease<'a> {
        let encoder = pool
            .lock()
            .pop()
            .unwrap_or_else(|| Encoder::new(pool.level));
        Lease { pool, encoder }
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        // `Encoder::new` allocates nothing: its buffers come on first use.
        let encoder = core::mem::replace(&mut self.encoder, Encoder::new(self.pool.level));
        self.pool.lock().push(encoder);
    }
}

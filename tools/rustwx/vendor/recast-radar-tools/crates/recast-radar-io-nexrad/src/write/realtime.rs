//! Real-time chunks: a volume written as the `S`, `I` and `E` files of the
//! NEXRAD real-time chunks bucket (`unidata-nexrad-level2-chunks`).
//!
//! The start chunk holds the 24-byte volume header and the LDM record of
//! the metadata messages; every later chunk holds one LDM record of radials
//! (at most [`WriteOptions::radials_per_record`], 120 by default, running on
//! across cuts or all of one cut by [`WriteOptions::record_layout`]; every
//! NOAA chunk seen holds 120 radials of one cut, its cuts being multiples of
//! 120), the last one being the end chunk, whose control word is negated and
//! whose last radial ends the volume.
//!
//! - [`write_realtime_chunks`] cuts a complete volume into chunks.
//!   Concatenated in order they are byte for byte the
//!   [`Compression::Bzip2LdmRecords`] file the writer produces for the same
//!   volume (as NOAA's archive files are the concatenation of their chunks).
//! - [`ChunkWriter`] sends the chunks while the volume is still being
//!   collected: the start chunk with the first sweep, the records of each
//!   sweep as it arrives, the end chunk when the volume is finished.

use chrono::{DateTime, Utc};
use recast_radar_core::model::{CycleTracker, Volume, collection_order};

use super::encode::{self, RadialContext, RadialRecord};
use super::plan::{self, PinnedCoding};
use super::{
    Compression, RecordLayout, SourceMetadata, WriteError, WriteOptions, WriteSummary, compress,
};

/// Kind of a real-time chunk (the letter at the end of its object name).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum ChunkKind {
    /// `S`: volume header and metadata record.
    Start,
    /// `I`: one record of radials.
    Intermediate,
    /// `E`: the last record of radials.
    End,
}

impl ChunkKind {
    /// The object-name letter: `S`, `I` or `E`.
    pub fn letter(self) -> char {
        match self {
            ChunkKind::Start => 'S',
            ChunkKind::Intermediate => 'I',
            ChunkKind::End => 'E',
        }
    }
}

/// One real-time chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct Chunk {
    /// Chunk kind.
    pub kind: ChunkKind,
    /// Chunk number in the volume, from 1.
    pub number: u16,
    /// The chunk file's bytes.
    pub bytes: Vec<u8>,
}

/// Object key of `chunk` in the chunks bucket layout:
/// `SITE/VOLUME/YYYYMMDD-HHMMSS-NNN-K`, for example
/// `KIWA/307/20260917-003629-001-S`.
pub fn chunk_key(
    icao: &str,
    volume_number: u16,
    volume_time: DateTime<Utc>,
    chunk: &Chunk,
) -> String {
    format!(
        "{icao}/{volume_number}/{}-{:03}-{}",
        volume_time.format("%Y%m%d-%H%M%S"),
        chunk.number,
        chunk.kind.letter()
    )
}

/// A volume written as real-time chunks.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkedVolume {
    /// Site identifier (the first path component of the chunk keys).
    pub icao: String,
    /// Volume number (the second path component, and the header's
    /// `AR2V0006.NNN` extension).
    pub volume_number: u16,
    /// Volume header time (the time in the chunk names).
    pub volume_time: DateTime<Utc>,
    /// The chunks in order.
    pub chunks: Vec<Chunk>,
    /// What was written.
    pub summary: WriteSummary,
}

impl ChunkedVolume {
    /// Object key of `chunk` in the chunks bucket layout ([`chunk_key`]).
    pub fn chunk_key(&self, chunk: &Chunk) -> String {
        chunk_key(&self.icao, self.volume_number, self.volume_time, chunk)
    }

    /// The chunks concatenated: the complete Archive II file.
    pub fn concatenated(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(self.chunks.iter().map(|chunk| chunk.bytes.len()).sum());
        for chunk in &self.chunks {
            bytes.extend_from_slice(&chunk.bytes);
        }
        bytes
    }
}

/// Refuse options that real-time chunks cannot have.
fn check_chunk_options(options: &WriteOptions) -> Result<(), WriteError> {
    if options.compression != Compression::Bzip2LdmRecords || options.gzip {
        return Err(WriteError::InvalidOption(
            "real-time chunks are LDM bzip2 records: use Compression::Bzip2LdmRecords without gzip"
                .to_owned(),
        ));
    }
    Ok(())
}

/// Write `volume` as real-time chunks. `options.compression` must be
/// [`Compression::Bzip2LdmRecords`] and `options.gzip` `false`: chunks are
/// always LDM bzip2 records.
pub fn write_realtime_chunks(
    volume: &Volume,
    options: &WriteOptions,
) -> Result<ChunkedVolume, WriteError> {
    write_realtime_chunks_with_source(volume, SourceMetadata::default(), options)
}

/// [`write_realtime_chunks`] reusing Level II context from the source.
pub fn write_realtime_chunks_with_source(
    volume: &Volume,
    source: SourceMetadata<'_>,
    options: &WriteOptions,
) -> Result<ChunkedVolume, WriteError> {
    check_chunk_options(options)?;
    let archive = super::build_archive(volume, source, options)?;
    let count = archive.records.len();
    if count > usize::from(u16::MAX) {
        return Err(WriteError::LimitExceeded(format!(
            "{count} records exceed the chunk numbering"
        )));
    }
    let mut chunks = Vec::with_capacity(count);
    for (index, record) in archive.records.into_iter().enumerate() {
        let (kind, bytes) = if index == 0 {
            let mut bytes = Vec::with_capacity(archive.header.len() + record.len());
            bytes.extend_from_slice(&archive.header);
            bytes.extend_from_slice(&record);
            (ChunkKind::Start, bytes)
        } else if index + 1 == count {
            (ChunkKind::End, record)
        } else {
            (ChunkKind::Intermediate, record)
        };
        chunks.push(Chunk {
            kind,
            number: (index + 1) as u16,
            bytes,
        });
    }
    let mut summary = archive.summary;
    summary.bytes = chunks.iter().map(|chunk| chunk.bytes.len()).sum();
    Ok(ChunkedVolume {
        icao: summary.icao.clone(),
        volume_number: archive.volume_number,
        volume_time: summary.volume_time.unwrap_or(DateTime::<Utc>::UNIX_EPOCH),
        chunks,
        summary,
    })
}

/// Real-time chunks of a volume written while its sweeps are still being
/// collected, as the RDA sends them.
///
/// The planned volume fixes what the start chunk must say before the sweeps
/// arrive: its sweeps are the volume's elevation cuts, in order, which the
/// start chunk's Message 5 lists, and each moment's coding is chosen from
/// their fields (every sweep of a moment must share one coding: Py-ART
/// decodes a moment with its first sweep's scale and offset). The previous
/// volume of the same radar, or the first complete one, serves. The site,
/// VCP and volume number are fixed from it and `options` too. A pushed
/// sweep with a moment the planned volume lacks is refused
/// ([`WriteError::UnplannedMoment`]): its coding would differ from sweep to
/// sweep. Plan with a volume that has every moment the radar sends.
///
/// Each [`ChunkWriter::push`] takes the next sweeps (a volume holding one or
/// more of them, in cut order: one decoded ODIM scan file, say) and returns
/// the chunks they complete: the start chunk with the first push, then one
/// chunk per complete record. The last record so far is held back until the
/// next push or [`ChunkWriter::finish`], because the last record of the
/// volume must end it (radial status 4, negated control word); under
/// [`RecordLayout::Continuous`] the next push first fills it up to
/// [`WriteOptions::radials_per_record`] radials, so a cut whose radials are
/// not a multiple of that goes out with the next cut's first radials, while
/// under [`RecordLayout::WithinCuts`] each cut's records are complete when
/// it is pushed. A pushed sweep with a value outside its moment's fixed
/// coding is refused ([`WriteError::ValueOutsideCoding`]), nothing sent: the
/// writer never clips a value. Plan with a volume whose values span the
/// radar's.
///
/// Pushing all the planned volume's sweeps, one by one, gives the chunks
/// [`write_realtime_chunks`] gives for it. A volume finished early (fewer
/// sweeps than planned) ends where it stops.
pub struct ChunkWriter<'a> {
    planned: &'a Volume,
    options: WriteOptions,
    codings: Vec<PinnedCoding>,
    planned_cuts: usize,
    last_cut: u8,
    icao: String,
    volume_number: u16,
    volume_time: Option<DateTime<Utc>>,
    /// Sweeps written so far.
    written_sweeps: usize,
    /// Sweeps of the parts pushed so far, empty ones included.
    pushed_sweeps: usize,
    /// The scan cycle the pushed sweeps make up.
    cycle: CycleTracker,
    sequence: u32,
    chunks: usize,
    /// The last record so far, held back: it becomes the end chunk if the
    /// volume ends there.
    held: Option<RadialRecord>,
    summary: WriteSummary,
    /// The encoders of every chunk of this volume.
    compressor: compress::LdmCompressor,
}

impl<'a> ChunkWriter<'a> {
    /// Start a volume planned as `planned` (see [`ChunkWriter`]).
    /// `options.compression` must be [`Compression::Bzip2LdmRecords`] and
    /// `options.gzip` `false`.
    pub fn new(planned: &'a Volume, options: &WriteOptions) -> Result<Self, WriteError> {
        check_chunk_options(options)?;
        let plan = plan::plan(planned, SourceMetadata::default(), options)?;
        let mut fixed = options.clone();
        fixed.icao = Some(plan.summary.icao.clone());
        fixed.vcp = Some(plan.vcp);
        fixed.volume_number = Some(plan.volume_number);
        let summary = WriteSummary {
            icao: plan.summary.icao.clone(),
            ..WriteSummary::default()
        };
        Ok(Self {
            planned,
            codings: plan.codings(),
            planned_cuts: plan.sweeps.len(),
            last_cut: plan.sweeps.last().map_or(0, |sweep| sweep.elevation_number),
            icao: plan.summary.icao.clone(),
            volume_number: plan.volume_number,
            options: fixed,
            volume_time: None,
            written_sweeps: 0,
            pushed_sweeps: 0,
            cycle: CycleTracker::new(),
            sequence: encode::FIRST_RADIAL_SEQUENCE,
            chunks: 0,
            held: None,
            summary,
            compressor: compress::LdmCompressor::new(),
        })
    }

    /// Site identifier (the first path component of the chunk keys).
    pub fn icao(&self) -> &str {
        &self.icao
    }

    /// Volume number (the second path component of the chunk keys).
    pub fn volume_number(&self) -> u16 {
        self.volume_number
    }

    /// Volume header time: the first pushed part's earliest radial, `None`
    /// before the first push.
    pub fn volume_time(&self) -> Option<DateTime<Utc>> {
        self.volume_time
    }

    /// Object key of `chunk` ([`chunk_key`]); `None` before the first push.
    pub fn chunk_key(&self, chunk: &Chunk) -> Option<String> {
        Some(chunk_key(
            &self.icao,
            self.volume_number,
            self.volume_time?,
            chunk,
        ))
    }

    /// Add the next sweeps of the volume (`part`'s sweeps with rays, in cut
    /// order) and return the chunks they complete. Refused, with nothing
    /// sent, when the part cannot be written, has a moment the planned
    /// volume lacks or a value outside its moment's planned coding, or would
    /// take the volume past its planned cuts.
    pub fn push(&mut self, part: &Volume) -> Result<Vec<Chunk>, WriteError> {
        let mut plan = plan::plan_with_codings(
            part,
            SourceMetadata::default(),
            &self.options,
            Some(&self.codings),
        )?;
        let count = self.written_sweeps + plan.sweeps.len();
        if count > self.planned_cuts {
            return Err(WriteError::TooManySweeps {
                count,
                max: self.planned_cuts,
            });
        }
        // The pushed sweeps continue the scan cycle of the ones before
        // (numbered across the parts pushed).
        let mut cycle = self.cycle.clone();
        for index in collection_order(part) {
            let label = self.pushed_sweeps + index;
            if let Some(begins) = cycle.check(part, index, label) {
                return Err(WriteError::MixedScanCycles { begins });
            }
            cycle.add(part, index, label);
        }
        for (offset, sweep) in plan.sweeps.iter_mut().enumerate() {
            sweep.elevation_number =
                u8::try_from(self.written_sweeps + offset + 1).unwrap_or(u8::MAX);
        }
        let mut stored = Vec::new();
        if self.volume_time.is_none() {
            // The start chunk: the volume header at this part's first radial
            // and the metadata record of the planned volume.
            let mut planned = plan::plan(self.planned, SourceMetadata::default(), &self.options)?;
            planned.header_time = plan.header_time;
            let mut bytes = encode::volume_header(&planned).to_vec();
            bytes.extend_from_slice(
                &self
                    .compressor
                    .record(&encode::metadata_record_bytes(&planned), false)?,
            );
            stored.push((ChunkKind::Start, bytes));
        }
        let context = RadialContext {
            opens_volume: self.written_sweeps == 0,
            last_cut: self.last_cut,
            ends_volume: false,
        };
        // Under the continuous layout a held record that is not full takes
        // the part's first radials.
        let per_record = self.options.radials_per_record;
        let continues = self.options.record_layout == RecordLayout::Continuous
            && self
                .held
                .as_ref()
                .is_some_and(|held| held.radials < per_record);
        let room = match &self.held {
            Some(held) if continues => per_record - held.radials,
            _ => per_record,
        };
        let mut sequence = self.sequence;
        let mut records = encode::radial_records(&plan, &context, &mut sequence, room)?.into_iter();
        let mut held = self.held.clone();
        if continues && let (Some(held), Some(first)) = (held.as_mut(), records.next()) {
            let base = held.bytes.len();
            held.bytes
                .try_reserve(first.bytes.len())
                .map_err(|err| WriteError::LimitExceeded(format!("record buffer: {err}")))?;
            held.bytes.extend_from_slice(&first.bytes);
            held.radials += first.radials;
            held.last_status = base + first.last_status;
        }
        let mut complete = Vec::new();
        for record in records {
            if let Some(done) = held.replace(record) {
                complete.push(done);
            }
        }
        if held.is_none() {
            return Err(WriteError::EmptyVolume);
        }
        for record in &complete {
            stored.push((
                ChunkKind::Intermediate,
                self.compressor.record(&record.bytes, false)?,
            ));
        }
        if self.chunks + stored.len() + 1 > usize::from(u16::MAX) {
            return Err(WriteError::LimitExceeded(
                "the volume's chunks exceed the chunk numbering".to_owned(),
            ));
        }

        // Nothing below fails: commit the part.
        if self.volume_time.is_none() {
            self.volume_time = plan.summary.volume_time;
            self.summary.volume_time = plan.summary.volume_time;
        }
        self.held = held;
        self.sequence = sequence;
        self.cycle = cycle;
        self.written_sweeps = count;
        self.absorb(plan.summary, part.sweeps.len());
        let chunks: Vec<Chunk> = stored
            .into_iter()
            .map(|(kind, bytes)| {
                self.chunks += 1;
                Chunk {
                    kind,
                    number: self.chunks as u16,
                    bytes,
                }
            })
            .collect();
        self.summary.records += chunks.len();
        self.summary.bytes += chunks.iter().map(|chunk| chunk.bytes.len()).sum::<usize>();
        Ok(chunks)
    }

    /// End the volume: the held record, its last radial now ending the
    /// volume, as the end chunk; and what was written. Refused when nothing
    /// was pushed.
    pub fn finish(mut self) -> Result<(Chunk, WriteSummary), WriteError> {
        let Some(mut held) = self.held.take() else {
            return Err(WriteError::EmptyVolume);
        };
        // The last radial ended its elevation (2); it now ends the volume
        // (4). A one-radial cut keeps its opening status, as in a volume
        // written whole.
        if let Some(status) = held.bytes.get_mut(held.last_status)
            && *status == 2
        {
            *status = 4;
        }
        let bytes = self.compressor.record(&held.bytes, true)?;
        self.chunks += 1;
        let chunk = Chunk {
            kind: ChunkKind::End,
            number: self.chunks as u16,
            bytes,
        };
        self.summary.records += 1;
        self.summary.bytes += chunk.bytes.len();
        Ok((chunk, self.summary))
    }

    /// Add a part's summary, its sweep indices counted after the sweeps of
    /// the parts before it.
    fn absorb(&mut self, part: WriteSummary, part_sweeps: usize) {
        let offset = self.pushed_sweeps;
        let summary = &mut self.summary;
        summary.sweeps += part.sweeps;
        summary.radials += part.radials;
        summary.max_range_error_m = summary.max_range_error_m.max(part.max_range_error_m);
        summary
            .moments
            .extend(part.moments.into_iter().map(|mut report| {
                report.sweep += offset;
                report
            }));
        summary
            .skipped_fields
            .extend(part.skipped_fields.into_iter().map(|mut skipped| {
                skipped.sweep += offset;
                skipped
            }));
        summary
            .skipped_sweeps
            .extend(part.skipped_sweeps.into_iter().map(|index| index + offset));
        summary
            .written_rays
            .extend(part.written_rays.into_iter().map(|mut rays| {
                rays.sweep += offset;
                rays
            }));
        summary
            .written_sweeps
            .extend(part.written_sweeps.into_iter().map(|index| index + offset));
        summary.notes.extend(part.notes);
        self.pushed_sweeps += part_sweeps;
    }
}

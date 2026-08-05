//! The real-time Level-II chunk feed: the same radars, published while the
//! antenna is still turning.
//!
//! The archive route this crate opened with cannot be fresher than the scan
//! it reports: `unidata-nexrad-level2` only gains a volume file when the
//! volume *ends*, so a poll at a random instant meets an object whose newest
//! radial is on average half a volume period old and at worst a whole one --
//! 3.4 to 6.8 minutes at a site running a 410 s VCP.  The chunk feed
//! publishes the same bytes in ~110 pieces as they are collected, 1-4 s
//! behind the antenna.
//!
//! Nothing here decodes anything.  A chunk is an LDM record -- a 4-byte
//! big-endian length followed by a bzip2 block -- and a volume is those
//! records concatenated in sequence order behind the 24-byte volume header
//! that the first chunk carries.  So assembly is `Vec::extend_from_slice`
//! and the decoder is the one that was already here: `wx_radar`'s
//! `Level2File`, reached through [`crate::pack`], reads an assembled file
//! byte-for-byte as it reads an archived one.  Proven rather than asserted:
//! concatenating all 106 chunks of one site's volume on 2026-08-05 produced
//! 16,952,558 bytes with the same md5 as the archive object for the same
//! volume.
//!
//! **A partial volume is a first-class product here.**  An assembly that
//! stops mid-scan ends on an LDM block boundary, which is the one place the
//! framing validator is happy to stop, and the decoder reads the radials
//! that arrived.  Measured on the same day: a 3-chunk assembly (header plus
//! two data chunks) decoded to 240 radials of one 0.5-degree cut 391 s
//! before the volume file existed.  It is not faked and it is not implied:
//! [`LiveVolume::complete`] says which one this is, the assembled file is
//! named `_P{NNN}` when it is partial so it cannot be mistaken for the
//! archive object, and a caller that will not accept a partial says so.
//!
//! Discovery is S3 `ListObjectsV2` polling and nothing else.  There is no
//! push transport for this feed: no SNS subscription an anonymous caller can
//! make, no LDM, no WebSocket.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Datelike, NaiveDateTime, TimeZone, Timelike, Utc};

use rw_store::atomic::atomic_write_bytes;

use crate::s3::{
    boxed_error, cached_volume_path, download_object, hex_sha256, iso8601, list_s3,
    parse_s3_timestamp, publish_volume, ListRequest, S3Object,
};

/// How many chunk GETs run at once.
///
/// A cold complete volume is ~110 objects; serially that is ~110 round
/// trips, which spends more time than the feed's whole latency advantage.
/// Eight is what BowEcho's own live consumer uses.
const CHUNK_DOWNLOAD_CONCURRENCY: usize = 8;

/// The oldest a *discovered* live volume may be before discovery is refused.
///
/// The volume-id space is a counter that wraps, so identifying the live
/// volume means finding where the counter's arc ends (see [`id_runs`]).  If
/// retention ever outruns the wrap the arc closes, there is no end to find,
/// and the run analysis would answer with whichever id happens to sort last
/// -- an id whose newest volume is hours old, returned as though it were
/// live.  Thirty minutes is four to nine volume periods: far past any real
/// scan, far short of the ~48 h the bucket retains.  A caller who wants an
/// older volume names it with `--volume-id`, which is an instruction and not
/// a guess.
const DISCOVERY_MAX_AGE_SECONDS: i64 = 1800;

/// How many volume-id runs discovery will probe before refusing.
///
/// A rotating counter with a retention window makes one run, or two across
/// the wrap.  More than that means holes in the id space, and each hole is
/// another candidate for "newest".  Probing a handful is cheap; probing an
/// unbounded number is a listing storm dressed up as discovery.
const MAX_ID_RUN_PROBES: usize = 8;

/// Where a chunk sits in its volume: the header chunk, a data chunk, or the
/// last one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    /// `-S`: carries the 24-byte `AR2V…` volume header and the metadata
    /// messages.  Exactly one per volume, always sequence 1.
    Start,
    /// `-I`: a bare LDM record of radials.
    Intermediate,
    /// `-E`: the last record of the volume.  Its LDM length prefix is
    /// negative, which is the format's own end-of-volume marker.
    End,
}

impl ChunkKind {
    fn code(self) -> &'static str {
        match self {
            ChunkKind::Start => "S",
            ChunkKind::Intermediate => "I",
            ChunkKind::End => "E",
        }
    }

    fn parse(code: &str) -> Option<Self> {
        match code {
            "S" => Some(ChunkKind::Start),
            "I" => Some(ChunkKind::Intermediate),
            "E" => Some(ChunkKind::End),
            _ => None,
        }
    }
}

/// What [`parse_chunk_key`] read out of a chunk object key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkKey {
    pub site: String,
    pub volume_id: u32,
    /// The volume *start* time, which is the same stamp the archived
    /// volume file carries -- so a live assembly and the archive object it
    /// will become agree on their identity in time without being compared.
    pub volume_time: DateTime<Utc>,
    pub sequence: u32,
    pub kind: ChunkKind,
}

/// One listed chunk object.
#[derive(Debug, Clone)]
pub struct LiveChunk {
    pub object: S3Object,
    pub sequence: u32,
    pub kind: ChunkKind,
}

impl LiveChunk {
    pub fn last_modified(&self) -> Option<DateTime<Utc>> {
        parse_s3_timestamp(&self.object.last_modified)
    }
}

/// One volume of the live feed, as far as it has been published.
#[derive(Debug, Clone)]
pub struct LiveVolume {
    pub site: String,
    pub volume_id: u32,
    pub volume_time: DateTime<Utc>,
    /// The validated contiguous prefix, sequence 1 first.  Never a subset
    /// with a hole in it: see [`validated_prefix`].
    pub chunks: Vec<LiveChunk>,
    /// True when the prefix runs to the volume's `-E` chunk and nothing was
    /// dropped getting there.
    pub complete: bool,
    /// How many chunks the listing held for this volume, which is larger
    /// than `chunks.len()` exactly when the prefix was truncated.
    pub listed_chunks: usize,
    /// Why the prefix stopped short of the listing, when it did.
    pub truncation: Option<String>,
}

impl LiveVolume {
    pub fn total_bytes(&self) -> u64 {
        self.chunks.iter().map(|chunk| chunk.object.size_bytes).sum()
    }

    /// The `LastModified` of the newest chunk in the prefix: the instant the
    /// freshest byte this volume can offer became readable.
    pub fn newest_last_modified(&self) -> Option<DateTime<Utc>> {
        self.chunks.iter().filter_map(LiveChunk::last_modified).max()
    }

    /// Seconds between the newest chunk landing in the bucket and
    /// `observed_at` -- the feed lag, measured, against the clock of the
    /// machine that stamped the object.
    pub fn lag_seconds(&self, observed_at: Option<DateTime<Utc>>) -> Option<f64> {
        let newest = self.newest_last_modified()?;
        let now = observed_at?;
        Some((now - newest).num_milliseconds() as f64 / 1000.0)
    }

    pub fn keys(&self) -> Vec<String> {
        self.chunks
            .iter()
            .map(|chunk| chunk.object.key.clone())
            .collect()
    }
}

/// A live volume concatenated into one Archive-II file on disk.
#[derive(Debug, Clone)]
pub struct AssembledVolume {
    pub filename: String,
    pub cache_path: PathBuf,
    pub path: PathBuf,
    pub bytes: u64,
    pub sha256: String,
    pub chunk_cache_hits: usize,
}

/// Parse `{SITE}/{VOLUME_ID}/{YYYYMMDD}-{HHMMSS}-{NNN}-{S|I|E}`.
///
/// Returns `None` for anything else.  A sibling object under the site
/// prefix is skipped rather than fataled, exactly as an unrecognised
/// archive key is -- but it is never counted as a chunk either, and a
/// listing that yields no chunks at all is an empty volume rather than a
/// silently short one.
pub fn parse_chunk_key(key: &str) -> Option<ChunkKey> {
    let mut segments = key.split('/');
    let site = segments.next()?;
    let id = segments.next()?;
    let name = segments.next()?;
    if segments.next().is_some() {
        return None;
    }
    if site.len() != 4 || !site.chars().all(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let volume_id: u32 = id.parse().ok()?;
    // `20260805-073454-002-I`
    let mut fields = name.split('-');
    let day = fields.next()?;
    let clock = fields.next()?;
    let sequence = fields.next()?;
    let code = fields.next()?;
    if fields.next().is_some() {
        return None;
    }
    if day.len() != 8 || !day.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if clock.len() != 6 || !clock.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    if sequence.len() != 3 || !sequence.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let kind = ChunkKind::parse(code)?;
    let naive = NaiveDateTime::parse_from_str(&format!("{day}{clock}"), "%Y%m%d%H%M%S").ok()?;
    Some(ChunkKey {
        site: site.to_string(),
        volume_id,
        volume_time: Utc.from_utc_datetime(&naive),
        sequence: sequence.parse().ok()?,
        kind,
    })
}

/// The longest contiguous chunk prefix that can be concatenated into a
/// readable volume, and whether it is the whole volume.
///
/// **This is the fail-closed heart of the live route.**  Concatenation
/// cannot detect its own holes: chunks 1,2,4 glued together are a
/// well-formed LDM stream that decodes without complaint into radials whose
/// azimuths jump.  So the sequence is checked before any byte is fetched,
/// and the rules are:
///
/// * sequence 1 must be present and must be the `-S` chunk.  It carries the
///   volume header; without it there is no volume to assemble, and starting
///   at chunk 2 would produce a file whose first four bytes are read as a
///   volume header.  A refusal, not a truncation;
/// * a duplicate sequence number is a refusal.  Two objects claiming one
///   slot is two answers to one question, and picking either is a decision
///   nobody made;
/// * a gap **truncates** the prefix at the last chunk before it.  What
///   arrived is still a valid partial volume, and [`LiveVolume::truncation`]
///   records the sentence explaining why it stops there;
/// * only sequence 1 may be `-S`, and an `-E` must be last.  Either fault
///   truncates: past that point the sequence is not describing one volume.
pub fn validated_prefix(
    mut chunks: Vec<LiveChunk>,
) -> Result<(Vec<LiveChunk>, bool, Option<String>), Box<dyn Error>> {
    chunks.sort_by_key(|chunk| chunk.sequence);
    let listed = chunks.len();
    let Some(first) = chunks.first() else {
        return Err(boxed_error("the volume listed no chunks"));
    };
    if first.sequence != 1 {
        return Err(boxed_error(format!(
            "the volume's first listed chunk is sequence {} ({}), not 1; the \
             sequence-1 chunk carries the Archive-II volume header, so an \
             assembly that starts here would have its first LDM record read \
             as that header. The chunk has most likely expired out of the \
             bucket's ~48 h retention",
            first.sequence, first.object.key
        )));
    }
    if first.kind != ChunkKind::Start {
        return Err(boxed_error(format!(
            "the volume's sequence-1 chunk is type {} ({}), not S; a volume \
             whose header chunk is not a start-of-volume chunk is not a \
             volume this client will concatenate",
            first.kind.code(),
            first.object.key
        )));
    }
    for pair in chunks.windows(2) {
        if pair[0].sequence == pair[1].sequence {
            return Err(boxed_error(format!(
                "two objects claim chunk sequence {} of this volume ({} and \
                 {}); one of them is wrong and nothing in the listing says \
                 which, so no assembly made from either is trustworthy",
                pair[0].sequence, pair[0].object.key, pair[1].object.key
            )));
        }
    }

    let mut kept: Vec<LiveChunk> = Vec::with_capacity(chunks.len());
    let mut truncation: Option<String> = None;
    for chunk in chunks {
        let expected = kept.len() as u32 + 1;
        if chunk.sequence != expected {
            truncation = Some(format!(
                "chunk {expected} has not been published yet (the listing \
                 jumps to {}), so the assembly stops at chunk {}",
                chunk.sequence,
                kept.len()
            ));
            break;
        }
        if expected > 1 && chunk.kind == ChunkKind::Start {
            truncation = Some(format!(
                "chunk {expected} ({}) is a second start-of-volume chunk, so \
                 the assembly stops at chunk {}",
                chunk.object.key,
                kept.len()
            ));
            break;
        }
        let ended = kept
            .last()
            .is_some_and(|last| last.kind == ChunkKind::End);
        if ended {
            truncation = Some(format!(
                "chunk {} ({}) follows the end-of-volume chunk, so the \
                 assembly stops at chunk {}",
                chunk.sequence,
                chunk.object.key,
                kept.len()
            ));
            break;
        }
        kept.push(chunk);
    }

    let ended = kept
        .last()
        .is_some_and(|last| last.kind == ChunkKind::End);
    let complete = ended && kept.len() == listed && truncation.is_none();
    Ok((kept, complete, truncation))
}

/// Contiguous runs in a sorted set of volume ids, as inclusive `(first,
/// last)` pairs.
///
/// The live volume's id is the *end* of a run: the counter has not created
/// its successor yet.  Across the counter's wrap the retained ids form two
/// runs -- `1..=571` and `957..=999` on one site on 2026-08-05 -- and only
/// one of those two ends is live; the other is the tail of the previous
/// cycle, two days old and expiring.  Which is which is settled by reading
/// the volume start time out of each candidate's own keys, not by arithmetic
/// on the counter, because the counter's modulus is a property of the feed
/// that this client would then be asserting rather than observing.
pub fn id_runs(ids: &[u32]) -> Vec<(u32, u32)> {
    let mut sorted: Vec<u32> = ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs = Vec::new();
    let mut iter = sorted.into_iter();
    let Some(mut start) = iter.next() else {
        return runs;
    };
    let mut previous = start;
    for id in iter {
        if id == previous + 1 {
            previous = id;
            continue;
        }
        runs.push((start, previous));
        start = id;
        previous = id;
    }
    runs.push((start, previous));
    runs
}

/// The volume-id directories the bucket currently holds for `site`, plus the
/// bucket's own clock at the instant it answered.
pub fn list_volume_ids(
    agent: &ureq::Agent,
    bucket: &str,
    site: &str,
) -> Result<(Vec<u32>, Option<DateTime<Utc>>), Box<dyn Error>> {
    let prefix = format!("{site}/");
    let listing = list_s3(agent, ListRequest::new(bucket, &prefix).delimiter("/"))?;
    let mut ids = Vec::with_capacity(listing.common_prefixes.len());
    for rolled in &listing.common_prefixes {
        let Some(tail) = rolled.strip_prefix(&prefix) else {
            continue;
        };
        let id = tail.trim_end_matches('/');
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if let Ok(value) = id.parse::<u32>() {
            ids.push(value);
        }
    }
    ids.sort_unstable();
    ids.dedup();
    Ok((ids, listing.server_date))
}

/// Every volume published under one volume-id directory, newest start time
/// first.
///
/// There is usually one.  There are two when the id counter has wrapped
/// back onto a directory whose previous cycle has not yet expired -- the
/// bucket retains ~48 h and one site's counter wraps in ~2.3 days, so the
/// overlap is real and was observed.  Grouping by the volume start time in
/// the key is what keeps a fresh volume from being assembled out of a
/// two-day-old one's chunks.
pub fn list_volumes_under_id(
    agent: &ureq::Agent,
    bucket: &str,
    site: &str,
    volume_id: u32,
) -> Result<IdListing, Box<dyn Error>> {
    let prefix = format!("{site}/{volume_id}/");
    let listing = list_s3(agent, ListRequest::new(bucket, &prefix))?;
    let mut groups: BTreeMap<DateTime<Utc>, Vec<LiveChunk>> = BTreeMap::new();
    for object in listing.objects {
        let Some(parsed) = parse_chunk_key(&object.key) else {
            continue;
        };
        if parsed.site != site || parsed.volume_id != volume_id {
            continue;
        }
        groups.entry(parsed.volume_time).or_default().push(LiveChunk {
            object,
            sequence: parsed.sequence,
            kind: parsed.kind,
        });
    }
    Ok(group_volumes(site, volume_id, groups))
}

/// What one volume-id directory held: the volumes that validate, newest
/// first, and one sentence per volume that did not.
///
/// **A directory can legitimately hold an unassemblable volume.**  The
/// bucket expires objects by age, not by volume, so the oldest volume in
/// the retained window is routinely cut through the middle -- its
/// sequence-1 chunk is gone while its later chunks remain, and
/// [`validated_prefix`] refuses it, correctly.  Propagating that refusal
/// would take the LIVE volume down with the expiring one whenever the two
/// share an id, which across the counter's wrap they do daily.  So a
/// refused volume is recorded and stepped over, and only a directory with
/// nothing left in it is an error.
#[derive(Debug, Clone)]
pub struct IdListing {
    pub volumes: Vec<LiveVolume>,
    pub refused: Vec<String>,
}

fn group_volumes(
    site: &str,
    volume_id: u32,
    groups: BTreeMap<DateTime<Utc>, Vec<LiveChunk>>,
) -> IdListing {
    let mut volumes = Vec::with_capacity(groups.len());
    let mut refused = Vec::new();
    for (volume_time, chunks) in groups {
        let listed_chunks = chunks.len();
        match validated_prefix(chunks) {
            Ok((kept, complete, truncation)) => volumes.push(LiveVolume {
                site: site.to_string(),
                volume_id,
                volume_time,
                chunks: kept,
                complete,
                listed_chunks,
                truncation,
            }),
            Err(problem) => refused.push(format!(
                "volume {volume_id} at {} ({listed_chunks} chunks listed): \
                 {problem}",
                iso8601(volume_time)
            )),
        }
    }
    volumes.sort_by(|a, b| b.volume_time.cmp(&a.volume_time));
    IdListing { volumes, refused }
}

/// The newest assemblable volume under one id, or a refusal that names
/// every volume the directory held and why each was unusable.
fn newest_under_id(
    agent: &ureq::Agent,
    bucket: &str,
    site: &str,
    volume_id: u32,
) -> Result<LiveVolume, Box<dyn Error>> {
    let listing = list_volumes_under_id(agent, bucket, site, volume_id)?;
    match listing.volumes.into_iter().next() {
        Some(volume) => Ok(volume),
        None if listing.refused.is_empty() => Err(boxed_error(format!(
            "s3://{bucket}/{site}/{volume_id}/ holds no real-time chunks \
             for {site}"
        ))),
        None => Err(boxed_error(format!(
            "s3://{bucket}/{site}/{volume_id}/ holds no volume this client \
             can assemble: {}",
            listing.refused.join("; ")
        ))),
    }
}

/// What one discovery call found, including the ids it had to probe to
/// find it.
#[derive(Debug, Clone)]
pub struct Discovery {
    pub volume: LiveVolume,
    pub observed_at: Option<DateTime<Utc>>,
    /// Every volume-id directory the bucket still holds, which the
    /// look-back walk needs and a receipt reports the size of.
    pub ids: Vec<u32>,
    pub probed_ids: Vec<u32>,
    pub id_runs: Vec<(u32, u32)>,
}

/// Find the volume the radar is filling right now.
pub fn discover(
    agent: &ureq::Agent,
    bucket: &str,
    site: &str,
    volume_id: Option<u32>,
) -> Result<Discovery, Box<dyn Error>> {
    let (ids, observed_at) = list_volume_ids(agent, bucket, site)?;
    if ids.is_empty() {
        return Err(boxed_error(format!(
            "s3://{bucket}/{site}/ holds no real-time volume directories. \
             Either {site} is not publishing to the chunk feed or the site id \
             is not one this feed carries"
        )));
    }
    let runs = id_runs(&ids);
    if let Some(named) = volume_id {
        let volume = newest_under_id(agent, bucket, site, named)?;
        return Ok(Discovery {
            volume,
            observed_at,
            ids,
            probed_ids: vec![named],
            id_runs: runs,
        });
    }
    if runs.len() > MAX_ID_RUN_PROBES {
        return Err(boxed_error(format!(
            "s3://{bucket}/{site}/ holds {} retained volume ids in {} \
             disjoint runs; the live volume is the end of one of them and \
             probing more than {MAX_ID_RUN_PROBES} is a listing storm rather \
             than discovery. Pass --volume-id to name the volume",
            ids.len(),
            runs.len()
        )));
    }
    let mut probed = Vec::with_capacity(runs.len());
    let mut best: Option<LiveVolume> = None;
    let mut unusable: Vec<String> = Vec::new();
    for &(_, end) in &runs {
        probed.push(end);
        // One candidate failing is normal, not fatal.  The run that ends
        // at the expiring side of the wrap holds the volume the bucket is
        // currently cutting through the middle, and refusing the whole
        // discovery because the OLDEST retained volume is unassemblable
        // would take the live one down with it.
        match newest_under_id(agent, bucket, site, end) {
            Ok(candidate) => {
                let newer = best
                    .as_ref()
                    .is_none_or(|current| candidate.volume_time > current.volume_time);
                if newer {
                    best = Some(candidate);
                }
            }
            Err(problem) => unusable.push(problem.to_string()),
        }
    }
    let volume = best.ok_or_else(|| {
        boxed_error(format!(
            "no volume-id run end under s3://{bucket}/{site}/ held a volume \
             this client can assemble ({} probed): {}",
            probed.len(),
            unusable.join(" | ")
        ))
    })?;
    if let (Some(now), Some(newest)) = (observed_at, volume.newest_last_modified()) {
        let age = (now - newest).num_seconds();
        if age > DISCOVERY_MAX_AGE_SECONDS {
            return Err(boxed_error(format!(
                "the newest real-time chunk discovery could find for {site} is \
                 {age} s old ({}), past the {DISCOVERY_MAX_AGE_SECONDS} s \
                 ceiling. Either the site stopped publishing, or its retained \
                 id space has no gap for discovery to find the counter's end \
                 in -- both of which make 'newest' a guess. Pass --volume-id \
                 to name the volume, or use the archive route",
                volume.chunks.last().map_or("", |chunk| chunk.object.key.as_str())
            )));
        }
    }
    Ok(Discovery {
        volume,
        observed_at,
        ids,
        probed_ids: probed,
        id_runs: runs,
    })
}

/// Walk back from `newest` through the ids the bucket still holds, taking
/// volumes whose start times keep decreasing.
///
/// The walk stops rather than wrapping onto the previous counter cycle: an
/// id whose newest volume is *not* older than the one before it is the
/// two-day-old remnant, and returning it as "the volume before last" would
/// put a stale scan in the middle of a freshness-ordered list.
pub fn preceding_volumes(
    agent: &ureq::Agent,
    bucket: &str,
    site: &str,
    ids: &[u32],
    newest: &LiveVolume,
    count: usize,
) -> Result<Vec<LiveVolume>, Box<dyn Error>> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let sorted: Vec<u32> = {
        let mut copy = ids.to_vec();
        copy.sort_unstable();
        copy.dedup();
        copy
    };
    let Some(position) = sorted.iter().position(|&id| id == newest.volume_id) else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    let mut previous_time = newest.volume_time;
    let mut index = position;
    for _ in 0..count {
        index = if index == 0 { sorted.len() - 1 } else { index - 1 };
        if index == position {
            break;
        }
        let candidate = newest_under_id(agent, bucket, site, sorted[index])?;
        if candidate.volume_time >= previous_time {
            break;
        }
        previous_time = candidate.volume_time;
        out.push(candidate);
    }
    Ok(out)
}

/// The Archive-II file name an assembly is published under.
///
/// A complete assembly gets exactly the name the archive object will have --
/// it is the same bytes, so giving it a different name would make one volume
/// look like two.  A partial gets `_P{NNN}` with the chunk count, which
/// [`crate::s3::parse_volume_key`] refuses, so a partial can never be picked
/// up by an archive listing or mistaken for a finished volume by anything
/// that reads names.
pub fn assembled_filename(volume: &LiveVolume, format: &str) -> String {
    let stamp = format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}",
        volume.volume_time.year(),
        volume.volume_time.month(),
        volume.volume_time.day(),
        volume.volume_time.hour(),
        volume.volume_time.minute(),
        volume.volume_time.second()
    );
    if volume.complete {
        format!("{}{stamp}_{format}", volume.site)
    } else {
        format!(
            "{}{stamp}_{format}_P{:03}",
            volume.site,
            volume.chunks.len()
        )
    }
}

/// Read the Archive-II build out of the assembled header: `AR2V0006` -> `V06`.
///
/// The format token goes in the published file name, so it is read from the
/// bytes rather than assumed.  A header this cannot read is a refusal: a
/// file named `_V06` whose header says something else is a name that lies.
fn archive2_format(raw: &[u8]) -> Result<String, Box<dyn Error>> {
    if raw.len() < 8 {
        return Err(boxed_error(format!(
            "the assembled volume is {} bytes, too short to carry the 8-byte \
             Archive-II version header",
            raw.len()
        )));
    }
    let magic = String::from_utf8_lossy(&raw[..8]).to_string();
    let build = magic.strip_prefix("AR2V").filter(|rest| {
        rest.len() == 4 && rest.chars().all(|c| c.is_ascii_digit())
    });
    match build {
        Some(digits) => Ok(format!("V{}", &digits[2..])),
        None => Err(boxed_error(format!(
            "the assembled volume opens {magic:?}, not AR2V followed by four \
             digits; the sequence-1 chunk did not carry an Archive-II volume \
             header and the assembly is not a Level-II volume"
        ))),
    }
}

/// Fetch every chunk of the prefix, `CHUNK_DOWNLOAD_CONCURRENCY` at a time.
fn fetch_chunks(
    agent: &ureq::Agent,
    bucket: &str,
    cache_dir: &Path,
    chunks: &[LiveChunk],
    use_cache: bool,
) -> Result<Vec<crate::s3::CachedObject>, Box<dyn Error>> {
    let workers = CHUNK_DOWNLOAD_CONCURRENCY.min(chunks.len().max(1));
    let mut slots: Vec<Option<crate::s3::CachedObject>> = vec![None; chunks.len()];
    let mut failures: Vec<String> = Vec::new();
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        for worker in 0..workers {
            let agent = agent.clone();
            handles.push(scope.spawn(move || {
                let mut done: Vec<(usize, Result<crate::s3::CachedObject, String>)> = Vec::new();
                let mut index = worker;
                while index < chunks.len() {
                    let outcome =
                        download_object(&agent, bucket, cache_dir, &chunks[index].object, use_cache)
                            .map_err(|error| {
                                format!("{}: {error}", chunks[index].object.key)
                            });
                    done.push((index, outcome));
                    index += workers;
                }
                done
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(done) => {
                    for (index, outcome) in done {
                        match outcome {
                            Ok(object) => slots[index] = Some(object),
                            Err(problem) => failures.push(problem),
                        }
                    }
                }
                Err(_) => failures.push("a chunk download worker panicked".to_string()),
            }
        }
    });
    if !failures.is_empty() {
        failures.sort();
        return Err(boxed_error(format!(
            "{} of {} chunk downloads failed, so the assembly would have a \
             hole in it: {}",
            failures.len(),
            chunks.len(),
            failures.join("; ")
        )));
    }
    let mut out = Vec::with_capacity(slots.len());
    for (index, slot) in slots.into_iter().enumerate() {
        out.push(slot.ok_or_else(|| {
            boxed_error(format!(
                "chunk {} ({}) was never downloaded",
                index + 1,
                chunks[index].object.key
            ))
        })?);
    }
    Ok(out)
}

/// Download the validated prefix and concatenate it into one Archive-II
/// file under the cache, then publish it into `out_dir` when one is given.
pub fn assemble(
    agent: &ureq::Agent,
    bucket: &str,
    cache_dir: &Path,
    out_dir: Option<&Path>,
    volume: &LiveVolume,
    use_cache: bool,
) -> Result<AssembledVolume, Box<dyn Error>> {
    if volume.chunks.is_empty() {
        return Err(boxed_error(format!(
            "{} volume {} at {} has no validated chunks to assemble",
            volume.site,
            volume.volume_id,
            iso8601(volume.volume_time)
        )));
    }
    let fetched = fetch_chunks(agent, bucket, cache_dir, &volume.chunks, use_cache)?;
    let chunk_cache_hits = fetched.iter().filter(|object| object.cache_hit).count();
    let total: usize = fetched.iter().map(|object| object.bytes.len()).sum();
    let mut raw = Vec::with_capacity(total);
    for (chunk, object) in volume.chunks.iter().zip(&fetched) {
        // The listed size was already enforced by `download_object`; this
        // states the same fact about the bytes that are actually being
        // concatenated, because a cache entry is read from disk and the
        // disk is not the listing.
        if chunk.object.size_bytes > 0 && object.bytes.len() as u64 != chunk.object.size_bytes {
            return Err(boxed_error(format!(
                "chunk {} ({}) holds {} bytes on disk, the listing said {}; a \
                 concatenation cannot see its own short chunk afterwards",
                chunk.sequence,
                chunk.object.key,
                object.bytes.len(),
                chunk.object.size_bytes
            )));
        }
        raw.extend_from_slice(&object.bytes);
    }
    let format = archive2_format(&raw)?;
    let filename = assembled_filename(volume, &format);
    let cache_key = format!("{}/{}/{filename}", volume.site, volume.volume_id);
    let cache_path = cached_volume_path(cache_dir, bucket, &cache_key)?;
    atomic_write_bytes(&cache_path, &raw)?;
    let path = match out_dir {
        Some(out) => publish_volume(&cache_path, out, &cache_key)?,
        None => cache_path.clone(),
    };
    Ok(AssembledVolume {
        filename,
        cache_path,
        path,
        bytes: raw.len() as u64,
        sha256: hex_sha256(&raw),
        chunk_cache_hits,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn object(key: &str, size: u64, last_modified: &str) -> S3Object {
        S3Object {
            key: key.to_string(),
            size_bytes: size,
            last_modified: last_modified.to_string(),
        }
    }

    fn chunk(sequence: u32, kind: ChunkKind) -> LiveChunk {
        LiveChunk {
            object: object(
                &format!("TEST/42/20260805-073454-{sequence:03}-{}", kind.code()),
                1000,
                "2026-08-05T07:35:00.000Z",
            ),
            sequence,
            kind,
        }
    }

    // -- key parsing -------------------------------------------------------

    #[test]
    fn a_chunk_key_parses_into_its_five_fields() {
        let parsed = parse_chunk_key("TEST/571/20260805-073454-002-I").expect("parses");
        assert_eq!(parsed.site, "TEST");
        assert_eq!(parsed.volume_id, 571);
        assert_eq!(parsed.sequence, 2);
        assert_eq!(parsed.kind, ChunkKind::Intermediate);
        assert_eq!(iso8601(parsed.volume_time), "2026-08-05T07:34:54Z");
    }

    #[test]
    fn the_three_chunk_types_all_parse() {
        for (code, kind) in [
            ("S", ChunkKind::Start),
            ("I", ChunkKind::Intermediate),
            ("E", ChunkKind::End),
        ] {
            let key = format!("TEST/1/20260805-073454-001-{code}");
            assert_eq!(parse_chunk_key(&key).expect("parses").kind, kind);
        }
    }

    #[test]
    fn keys_that_are_not_chunks_are_skipped_rather_than_guessed_at() {
        for key in [
            // an archive volume name, which lives in the other bucket
            "TEST/1/TEST20260805_073454_V06",
            // the wrong number of path segments
            "TEST/20260805-073454-001-S",
            "TEST/1/2/20260805-073454-001-S",
            // a chunk type this client does not know
            "TEST/1/20260805-073454-001-X",
            // a non-numeric volume id
            "TEST/abc/20260805-073454-001-S",
            // a two-digit sequence
            "TEST/1/20260805-073454-01-S",
            // an impossible clock
            "TEST/1/20260805-993454-001-S",
            // a five-character site
            "TESTS/1/20260805-073454-001-S",
        ] {
            assert!(parse_chunk_key(key).is_none(), "{key} should not parse");
        }
    }

    // -- prefix validation -------------------------------------------------

    #[test]
    fn a_finished_volume_validates_whole_and_complete() {
        let chunks = vec![
            chunk(1, ChunkKind::Start),
            chunk(2, ChunkKind::Intermediate),
            chunk(3, ChunkKind::End),
        ];
        let (kept, complete, truncation) = validated_prefix(chunks).expect("validates");
        assert_eq!(kept.len(), 3);
        assert!(complete);
        assert!(truncation.is_none());
    }

    #[test]
    fn a_mid_scan_volume_validates_partial_rather_than_being_refused() {
        let chunks = vec![
            chunk(1, ChunkKind::Start),
            chunk(2, ChunkKind::Intermediate),
        ];
        let (kept, complete, truncation) = validated_prefix(chunks).expect("validates");
        assert_eq!(kept.len(), 2);
        assert!(!complete, "no end-of-volume chunk means not complete");
        assert!(truncation.is_none(), "reaching the end of what exists is not a truncation");
    }

    #[test]
    fn a_gap_truncates_the_prefix_and_says_where() {
        let chunks = vec![
            chunk(1, ChunkKind::Start),
            chunk(2, ChunkKind::Intermediate),
            chunk(4, ChunkKind::Intermediate),
        ];
        let (kept, complete, truncation) = validated_prefix(chunks).expect("validates");
        assert_eq!(kept.len(), 2, "chunk 4 is unusable while 3 is missing");
        assert!(!complete);
        let note = truncation.expect("the gap is recorded");
        assert!(note.contains("chunk 3 has not been published"), "{note}");
    }

    #[test]
    fn a_gap_before_the_end_chunk_does_not_report_completeness() {
        let chunks = vec![
            chunk(1, ChunkKind::Start),
            chunk(3, ChunkKind::End),
        ];
        let (kept, complete, truncation) = validated_prefix(chunks).expect("validates");
        assert_eq!(kept.len(), 1);
        assert!(!complete, "an end chunk reached over a hole is not a complete volume");
        assert!(truncation.is_some());
    }

    #[test]
    fn a_missing_header_chunk_is_refused_not_truncated() {
        let chunks = vec![
            chunk(2, ChunkKind::Intermediate),
            chunk(3, ChunkKind::End),
        ];
        let error = validated_prefix(chunks).expect_err("refuses");
        assert!(error.to_string().contains("volume header"), "{error}");
    }

    #[test]
    fn a_sequence_one_that_is_not_a_start_chunk_is_refused() {
        let chunks = vec![chunk(1, ChunkKind::Intermediate)];
        let error = validated_prefix(chunks).expect_err("refuses");
        assert!(error.to_string().contains("not S"), "{error}");
    }

    #[test]
    fn two_objects_claiming_one_sequence_refuse_the_whole_volume() {
        let mut duplicate = chunk(2, ChunkKind::Intermediate);
        duplicate.object.key = "TEST/42/20260805-073454-002-E".to_string();
        let chunks = vec![chunk(1, ChunkKind::Start), chunk(2, ChunkKind::Intermediate), duplicate];
        let error = validated_prefix(chunks).expect_err("refuses");
        assert!(error.to_string().contains("claim chunk sequence 2"), "{error}");
    }

    #[test]
    fn chunks_after_the_end_of_volume_chunk_truncate_the_prefix() {
        let chunks = vec![
            chunk(1, ChunkKind::Start),
            chunk(2, ChunkKind::End),
            chunk(3, ChunkKind::Intermediate),
        ];
        let (kept, complete, truncation) = validated_prefix(chunks).expect("validates");
        assert_eq!(kept.len(), 2);
        assert!(!complete, "a volume with data past its own end is not a clean volume");
        assert!(truncation.expect("recorded").contains("follows the end-of-volume chunk"));
    }

    #[test]
    fn an_empty_listing_is_refused_rather_than_assembled_as_zero_bytes() {
        let error = validated_prefix(Vec::new()).expect_err("refuses");
        assert!(error.to_string().contains("no chunks"), "{error}");
    }

    // -- one id directory, two counter cycles ------------------------------

    fn stamped(volume_time: &str, sequence: u32, kind: ChunkKind) -> LiveChunk {
        LiveChunk {
            object: object(
                &format!("TEST/42/{volume_time}-{sequence:03}-{}", kind.code()),
                1000,
                "2026-08-05T07:35:00.000Z",
            ),
            sequence,
            kind,
        }
    }

    fn grouped(entries: Vec<(&str, Vec<LiveChunk>)>) -> IdListing {
        let mut groups: BTreeMap<DateTime<Utc>, Vec<LiveChunk>> = BTreeMap::new();
        for (stamp, chunks) in entries {
            let naive =
                NaiveDateTime::parse_from_str(stamp, "%Y%m%d-%H%M%S").expect("stamp");
            groups.insert(Utc.from_utc_datetime(&naive), chunks);
        }
        group_volumes("TEST", 42, groups)
    }

    #[test]
    fn an_expiring_volume_does_not_take_the_live_one_down_with_it() {
        // The bucket expires OBJECTS by age, so the oldest retained volume
        // is routinely cut through the middle -- its sequence-1 chunk is
        // gone.  When the id counter has wrapped back onto that directory,
        // the fresh volume and the half-expired one share it, and refusing
        // the directory would refuse the live scan.
        let listing = grouped(vec![
            (
                "20260803-045257",
                vec![
                    stamped("20260803-045257", 28, ChunkKind::Intermediate),
                    stamped("20260803-045257", 29, ChunkKind::End),
                ],
            ),
            (
                "20260805-073454",
                vec![
                    stamped("20260805-073454", 1, ChunkKind::Start),
                    stamped("20260805-073454", 2, ChunkKind::Intermediate),
                ],
            ),
        ]);
        assert_eq!(listing.volumes.len(), 1, "the live volume survives");
        assert_eq!(
            iso8601(listing.volumes[0].volume_time),
            "2026-08-05T07:34:54Z"
        );
        assert_eq!(listing.refused.len(), 1, "and the refusal is recorded");
        assert!(listing.refused[0].contains("volume header"), "{:?}", listing.refused);
    }

    #[test]
    fn two_cycles_under_one_id_are_ordered_newest_first_and_never_spliced() {
        let listing = grouped(vec![
            (
                "20260803-045257",
                vec![
                    stamped("20260803-045257", 1, ChunkKind::Start),
                    stamped("20260803-045257", 2, ChunkKind::End),
                ],
            ),
            (
                "20260805-073454",
                vec![stamped("20260805-073454", 1, ChunkKind::Start)],
            ),
        ]);
        assert_eq!(listing.volumes.len(), 2);
        assert_eq!(
            iso8601(listing.volumes[0].volume_time),
            "2026-08-05T07:34:54Z"
        );
        assert_eq!(listing.volumes[0].chunks.len(), 1, "no chunk crosses cycles");
        assert_eq!(listing.volumes[1].chunks.len(), 2);
        assert!(listing.refused.is_empty());
    }

    // -- volume-id runs ----------------------------------------------------

    #[test]
    fn a_contiguous_id_space_is_one_run() {
        assert_eq!(id_runs(&[3, 4, 5, 6]), vec![(3, 6)]);
    }

    #[test]
    fn a_wrapped_counter_shows_as_two_runs_and_two_candidates() {
        // The shape one site actually held on 2026-08-05: 1..=571 live and
        // 957..=999 expiring from the previous cycle.
        let mut ids: Vec<u32> = (1..=571).collect();
        ids.extend(957..=999);
        assert_eq!(id_runs(&ids), vec![(1, 571), (957, 999)]);
    }

    #[test]
    fn runs_are_computed_from_an_unsorted_set_with_duplicates() {
        assert_eq!(id_runs(&[9, 2, 3, 9, 1]), vec![(1, 3), (9, 9)]);
    }

    #[test]
    fn an_empty_id_set_has_no_runs() {
        assert!(id_runs(&[]).is_empty());
    }

    // -- naming ------------------------------------------------------------

    fn volume(chunks: Vec<LiveChunk>, complete: bool) -> LiveVolume {
        LiveVolume {
            site: "TEST".to_string(),
            volume_id: 42,
            volume_time: Utc.with_ymd_and_hms(2026, 8, 5, 7, 34, 54).unwrap(),
            listed_chunks: chunks.len(),
            chunks,
            complete,
            truncation: None,
        }
    }

    #[test]
    fn a_complete_assembly_is_named_exactly_as_the_archive_object_will_be() {
        let assembled = volume(vec![chunk(1, ChunkKind::Start)], true);
        assert_eq!(assembled_filename(&assembled, "V06"), "TEST20260805_073454_V06");
    }

    #[test]
    fn a_partial_assembly_carries_its_chunk_count_and_is_not_a_volume_name() {
        let chunks = vec![chunk(1, ChunkKind::Start), chunk(2, ChunkKind::Intermediate)];
        let assembled = volume(chunks, false);
        let name = assembled_filename(&assembled, "V06");
        assert_eq!(name, "TEST20260805_073454_V06_P002");
        assert!(
            crate::s3::parse_volume_key(&name).is_none(),
            "a partial must never be picked up by an archive listing"
        );
    }

    #[test]
    fn the_archive2_build_is_read_out_of_the_header_rather_than_assumed() {
        assert_eq!(archive2_format(b"AR2V0006.571").expect("reads"), "V06");
        assert_eq!(archive2_format(b"AR2V0008.001").expect("reads"), "V08");
    }

    #[test]
    fn a_header_that_is_not_archive2_refuses_the_assembly() {
        let error = archive2_format(b"BZh91AY&SY").expect_err("refuses");
        assert!(error.to_string().contains("not AR2V"), "{error}");
        let short = archive2_format(b"AR2V").expect_err("refuses");
        assert!(short.to_string().contains("too short"), "{short}");
    }

    // -- lag ---------------------------------------------------------------

    #[test]
    fn the_lag_is_measured_from_the_newest_chunk_against_the_bucket_clock() {
        let mut first = chunk(1, ChunkKind::Start);
        first.object.last_modified = "2026-08-05T07:34:56.000Z".to_string();
        let mut second = chunk(2, ChunkKind::Intermediate);
        second.object.last_modified = "2026-08-05T07:35:01.500Z".to_string();
        let assembled = volume(vec![first, second], false);
        let observed = Utc.with_ymd_and_hms(2026, 8, 5, 7, 35, 4).unwrap();
        assert_eq!(assembled.lag_seconds(Some(observed)), Some(2.5));
    }

    #[test]
    fn an_unreadable_bucket_clock_leaves_the_lag_unstated_rather_than_guessed() {
        let assembled = volume(vec![chunk(1, ChunkKind::Start)], false);
        assert_eq!(assembled.lag_seconds(None), None);
    }
}

//! The networked half of `rw_fetch`.
//!
//! Every request in this module goes through `wx_core`'s
//! [`DownloadClient`], and that is load-bearing rather than merely
//! convenient.  The client carries four mechanisms this tool exists to
//! reuse, and reaching around it to `agent()` would silently drop all
//! four:
//!
//! 1. **`get_bytes_parallel_whole`** (`wx-core client.rs:723`) --
//!    probes range support, splits the object into 16 MiB chunks
//!    (`FULL_FILE_RANGE_CHUNK_BYTES`) and pulls them through rayon.
//! 2. **`get_ranges`** (`client.rs:828`) -- parallel range GETs for
//!    everything except NOMADS, which it fetches **serially** on
//!    purpose (`client.rs:842`).
//! 3. **The cross-process NOMADS rate governor** (`client.rs:178-232`):
//!    a lock file plus `%TEMP%\rustwx_nomads_rate_limit.state` holding
//!    a 2.5 s minimum inter-request gap and a 15-minute node-wide
//!    cooldown when Akamai's malformed over-rate-limit 3xx is
//!    fingerprinted.  It is enforced inside `with_retry`, so it only
//!    applies to requests made through the client's own methods.
//! 4. **The two-tier disk cache** (`download/cache.rs`), keyed by URL
//!    and byte range, so the probe's tail read is free to the transfer
//!    that follows it.
//!
//! One consequence shapes the probe: the client deliberately exposes no
//! response headers, so there is no public way to read a
//! `Content-Length` or a `Content-Range` total.  Coverage is therefore
//! proven with a **bounded tail range GET** instead -- see
//! [`probe_object`] -- which is both exact and paced.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use wx_core::download::{
    byte_ranges, find_entries, parse_idx, CacheDedup, DownloadClient, DownloadConfig,
};

use crate::plan::{coalesce_ranges, grib2_message_length, validate_idx, IdxRow, ProbeFacts};
use crate::record::RangeRecord;

/// Bytes of Section 0 needed to read a GRIB2 message's total length.
const GRIB2_SECTION0_BYTES: u64 = 16;

/// Marks a `read_idx` failure as "the index never arrived", as opposed
/// to "it arrived and did not validate".
///
/// The probe records those as different facts and `plan::decide` prints
/// different reasons for them, so the distinction has to survive the
/// trip through a `String`.  It used to be recovered by sniffing for
/// `"HTTP request failed"` at the *start* of the message -- which no
/// message ever begins with, because `wx_core`'s error type renders
/// that variant as `"HTTP error: HTTP request failed for ..."`.  Every
/// 404 therefore landed in the else-branch and was recorded as
/// fetched-but-malformed: an absent index reported as a broken one.
/// The transport half of `read_idx` now tags its own failures instead
/// of leaving the caller to guess from prose.
const IDX_TRANSPORT_PREFIX: &str = "could not fetch the index: ";

/// Did this `read_idx` error mean the index never arrived?
pub fn idx_was_unfetchable(error: &str) -> bool {
    error.starts_with(IDX_TRANSPORT_PREFIX)
}

pub struct Fetcher {
    client: DownloadClient,
}

/// What one successful `.idx` read produced.
pub struct IdxPayload {
    pub text: String,
    pub sha256: String,
    pub bytes: u64,
    pub rows: Vec<IdxRow>,
}

impl Fetcher {
    pub fn new(cache_dir: Option<&Path>) -> Result<Self, String> {
        let client = match cache_dir {
            Some(dir) => DownloadClient::new_with_cache(Some(
                dir.to_str()
                    .ok_or_else(|| format!("--cache-dir is not valid UTF-8: {}", dir.display()))?,
            )),
            None => DownloadClient::new_with_config(DownloadConfig::default()),
        }
        .map_err(|error| format!("could not build the download client: {error}"))?;
        Ok(Self { client })
    }

    /// What this run's cache wrote versus what it already held.
    ///
    /// `None` when no `--cache-dir` was given: nothing was cached, which is
    /// a different statement from "nothing was deduplicated".  One full-file
    /// object reaches the cache under two key shapes, so the receipt has to
    /// be able to say that the second one cost a reference and not a copy.
    pub fn cache_dedup(&self) -> Option<CacheDedup> {
        self.client.cache().map(|cache| cache.dedup())
    }

    /// Fetch and strictly validate the `.idx` for one object.
    pub fn read_idx(&self, idx_url: &str) -> Result<IdxPayload, String> {
        let text = self
            .client
            .get_text(idx_url)
            .map_err(|error| format!("{IDX_TRANSPORT_PREFIX}{error}"))?;
        let rows = validate_idx(&text, None)?;
        // Belt and braces: the lenient upstream parser and the strict
        // one must agree on how many messages this index describes.  A
        // disagreement means one of them is skipping a line, and a
        // subset built on the difference would be quietly wrong.
        let lenient = parse_idx(&text);
        if lenient.len() != rows.len() {
            return Err(format!(
                "index parsers disagree: wx-core parse_idx found {} records, \
                 strict validation found {}",
                lenient.len(),
                rows.len()
            ));
        }
        Ok(IdxPayload {
            sha256: hex_digest(text.as_bytes()),
            bytes: text.len() as u64,
            rows,
            text,
        })
    }

    /// Gather every fact [`crate::plan::decide`] needs about one object.
    ///
    /// The coverage proof is the interesting part.  Rather than trust a
    /// declared object size, this reads the last indexed message's own
    /// Section 0 to learn its length, then asks for exactly one byte
    /// **more** than that message occupies.  If the origin returns the
    /// message and nothing else, the index provably ends where the
    /// object ends.  If it returns one extra byte, the object carries
    /// records the index never mentions -- a mid-publish `.idx`, and
    /// the case Drew's rule exists for.  Both answers cost one bounded
    /// range GET whose bytes the cache keeps for the transfer that
    /// follows.
    pub fn probe_object(
        &self,
        grib_url: &str,
        idx_url: Option<&str>,
        consult_idx: bool,
    ) -> (ProbeFacts, Option<IdxPayload>) {
        let mut facts = ProbeFacts {
            idx_declared: idx_url.is_some(),
            ..ProbeFacts::default()
        };
        let mut payload: Option<IdxPayload> = None;

        if consult_idx {
            if let Some(url) = idx_url {
                match self.read_idx(url) {
                    Ok(found) => {
                        facts.idx_fetched = true;
                        facts.idx_rows = found.rows.len();
                        let last = found.rows[found.rows.len() - 1].offset;
                        facts.idx_last_offset = Some(last);
                        let (message_bytes, covers, object_bytes) =
                            self.probe_index_coverage(grib_url, last);
                        facts.idx_last_message_bytes = message_bytes;
                        facts.idx_covers_object = covers;
                        facts.object_bytes = object_bytes;
                        if covers.is_some() {
                            facts.object_present = true;
                        }
                        payload = Some(found);
                    }
                    Err(error) => {
                        // "could not fetch" and "fetched but malformed"
                        // are different facts and the record keeps both.
                        facts.idx_fetched = !idx_was_unfetchable(&error);
                        facts.idx_error = Some(error);
                    }
                }
            }
        }

        if !facts.object_present {
            facts.object_present = self.client.head_ok(grib_url);
        }
        (facts, payload)
    }

    fn probe_index_coverage(
        &self,
        grib_url: &str,
        last_offset: u64,
    ) -> (Option<u64>, Option<bool>, Option<u64>) {
        let head = match self.client.get_range(
            grib_url,
            last_offset,
            last_offset + GRIB2_SECTION0_BYTES - 1,
        ) {
            Ok(bytes) => bytes,
            Err(_) => return (None, None, None),
        };
        let Some(message_bytes) = grib2_message_length(&head) else {
            return (None, None, None);
        };
        // One byte past the message: present => the index is short.
        let tail = match self
            .client
            .get_range(grib_url, last_offset, last_offset + message_bytes)
        {
            Ok(bytes) => bytes,
            Err(_) => return (Some(message_bytes), None, None),
        };
        let observed = tail.len() as u64;
        if observed == message_bytes {
            (
                Some(message_bytes),
                Some(true),
                Some(last_offset + message_bytes),
            )
        } else if observed == message_bytes + 1 {
            (Some(message_bytes), Some(false), None)
        } else {
            (Some(message_bytes), None, None)
        }
    }

    /// Pull the whole object through 16 MiB parallel range GETs.
    pub fn get_full_file(&self, grib_url: &str) -> Result<Vec<u8>, String> {
        self.client
            .get_bytes_parallel_whole(grib_url)
            .map_err(|error| format!("{error}"))
    }

    /// Pull only the selected messages, as coalesced range GETs.
    pub fn get_idx_subset(
        &self,
        grib_url: &str,
        payload: &IdxPayload,
        selection: &[usize],
    ) -> Result<(Vec<u8>, Vec<RangeRecord>), String> {
        // Range algebra stays upstream's: parse_idx -> byte_ranges,
        // then coalesced so a 561-record selection becomes a handful of
        // GETs instead of 561.
        let entries = parse_idx(&payload.text);
        let chosen: Vec<_> = selection.iter().map(|index| &entries[*index]).collect();
        let raw = byte_ranges(&entries, &chosen);
        let merged = coalesce_ranges(&raw);

        let bytes = self
            .client
            .get_ranges(grib_url, &merged)
            .map_err(|error| format!("{error}"))?;

        // Attribute each merged range back to the index rows it spans,
        // so the Python side can author the per-range receipt.  The
        // last selected message has no successor offset, so upstream's
        // `byte_ranges` gives it an open end (`u64::MAX`); its true
        // length is whatever is left over once the finite ranges are
        // accounted for.
        let finite: u64 = merged
            .iter()
            .filter(|(_, end)| *end != u64::MAX)
            .map(|(start, end)| end - start + 1)
            .sum();
        let total = bytes.len() as u64;
        let mut records = Vec::with_capacity(merged.len());
        for (start, end) in &merged {
            let spanned: Vec<u32> = payload
                .rows
                .iter()
                .filter(|row| row.offset >= *start && (*end == u64::MAX || row.offset <= *end))
                .map(|row| row.sequence)
                .collect();
            let length = if *end == u64::MAX {
                total.saturating_sub(finite)
            } else {
                end - start + 1
            };
            records.push(RangeRecord {
                start: *start,
                end: start + length.saturating_sub(1),
                bytes: length,
                first_index_row: spanned.first().copied().unwrap_or_default(),
                last_index_row: spanned.last().copied().unwrap_or_default(),
            });
        }
        if records.iter().map(|record| record.bytes).sum::<u64>() != total {
            return Err(format!(
                "range accounting does not add up: {} range bytes vs {total} received",
                records.iter().map(|record| record.bytes).sum::<u64>()
            ));
        }
        Ok((bytes, records))
    }
}

/// Select index rows by exact `VAR:LEVEL` patterns.
///
/// Exact on both columns, deliberately unlike `wx_core`'s
/// `find_entries`, whose level match is a substring test -- there,
/// `"PRES:1 hybrid level"` also matches levels 11, 21, 31 and 41, which
/// is exactly the sort of over-selection a 561-record bar exists to
/// catch.  A pattern with no colon matches the variable alone.  Every
/// pattern must match at least one row: a pattern that matches nothing
/// is a fail-closed error, never a quietly smaller download.
///
/// The variable half may be an alternation, `A|B:LEVEL`, matching a row
/// whose variable is *either* name.  Providers do not always agree on a
/// field's short name for the same physical record: NOMADS spells HRRR
/// hybrid cloud water `CLWMR` where AWS S3 spells it `CLMR`.  That is a
/// naming difference, not an inventory difference, and an alternation
/// says so explicitly -- one selector, one record, either spelling.
/// The "matched nothing" rule applies to the alternation as a whole, so
/// a genuinely absent field is still a hard error.
///
/// `exclusions` are substrings tested against the whole raw index line.
/// HRRR publishes accumulated variants of several surface fields --
/// `WEASD:surface:0-1 hr acc fcst` sits beside the instantaneous
/// `WEASD:surface` -- and the native bridge selects only the
/// instantaneous one.  Without the exclusion an exact `VAR:LEVEL`
/// selector would take both and blow the record bar.
pub fn select_exact(
    rows: &[IdxRow],
    patterns: &[String],
    exclusions: &[String],
) -> Result<Vec<usize>, String> {
    let mut chosen: Vec<usize> = Vec::new();
    let mut unmatched: Vec<&str> = Vec::new();
    for pattern in patterns {
        let (variable, level) = match pattern.split_once(':') {
            Some((variable, level)) => (variable, Some(level)),
            None => (pattern.as_str(), None),
        };
        let accepted: Vec<&str> = variable.split('|').map(str::trim).collect();
        let mut hit = false;
        for (index, row) in rows.iter().enumerate() {
            if !accepted.iter().any(|name| *name == row.variable) {
                continue;
            }
            if let Some(level) = level {
                if row.level != level {
                    continue;
                }
            }
            if exclusions
                .iter()
                .any(|excluded| row.raw.contains(excluded.as_str()))
            {
                continue;
            }
            hit = true;
            chosen.push(index);
        }
        if !hit {
            unmatched.push(pattern);
        }
    }
    if !unmatched.is_empty() {
        let shown: Vec<&str> = unmatched.iter().take(8).copied().collect();
        return Err(format!(
            "{} of {} --var-pattern selectors matched no index record (first: {}); \
             refusing to download a silently smaller subset",
            unmatched.len(),
            patterns.len(),
            shown.join(", ")
        ));
    }
    chosen.sort_unstable();
    chosen.dedup();
    Ok(chosen)
}

/// Select index rows with `wx_core`'s substring level semantics.
pub fn select_contains(text: &str, patterns: &[String]) -> Result<Vec<usize>, String> {
    let entries = parse_idx(text);
    let mut chosen: Vec<usize> = Vec::new();
    let mut unmatched: Vec<&str> = Vec::new();
    for pattern in patterns {
        let hits = find_entries(&entries, pattern);
        if hits.is_empty() {
            unmatched.push(pattern);
            continue;
        }
        for hit in hits {
            if let Some(index) = entries
                .iter()
                .position(|entry| entry.byte_offset == hit.byte_offset)
            {
                chosen.push(index);
            }
        }
    }
    if !unmatched.is_empty() {
        return Err(format!(
            "{} of {} --var-pattern-contains selectors matched no index record (first: {})",
            unmatched.len(),
            patterns.len(),
            unmatched[0]
        ));
    }
    chosen.sort_unstable();
    chosen.dedup();
    Ok(chosen)
}

/// Is this a complete GRIB2 stream: `GRIB` in, `7777` out?
pub fn grib_framed(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[0..4] == b"GRIB" && &bytes[bytes.len() - 4..] == b"7777"
}

pub fn hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Write `bytes` to `path` through a `.part` file and one `rename`.
///
/// Never overwrites: an existing destination is the caller's to resolve
/// (Python's resume-with-identity-guard owns that decision, and it
/// quarantines rather than deletes).
pub fn publish(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        return Err(format!(
            "refusing to replace an existing file: {}",
            path.display()
        ));
    }
    let partial: PathBuf = path.with_extension(format!(
        "{}.part",
        path.extension().and_then(|e| e.to_str()).unwrap_or("out")
    ));
    {
        let mut handle = fs::File::create(&partial)
            .map_err(|error| format!("could not create {}: {error}", partial.display()))?;
        handle
            .write_all(bytes)
            .map_err(|error| format!("could not write {}: {error}", partial.display()))?;
        handle
            .sync_all()
            .map_err(|error| format!("could not flush {}: {error}", partial.display()))?;
    }
    fs::rename(&partial, path)
        .map_err(|error| format!("could not publish {}: {error}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows() -> Vec<IdxRow> {
        [
            ("PRES", "1 hybrid level", 0u64, "anl"),
            ("PRES", "11 hybrid level", 100, "anl"),
            ("PRES", "surface", 200, "anl"),
            ("TMP", "2 m above ground", 300, "anl"),
            ("WEASD", "surface", 400, "anl"),
            ("WEASD", "surface", 500, "0-1 hr acc fcst"),
        ]
        .into_iter()
        .enumerate()
        .map(|(index, (variable, level, offset, forecast))| IdxRow {
            sequence: index as u32 + 1,
            offset,
            variable: variable.to_string(),
            level: level.to_string(),
            raw: format!(
                "{}:{offset}:d=2026072812:{variable}:{level}:{forecast}:",
                index + 1
            ),
        })
        .collect()
    }

    #[test]
    fn exact_selection_does_not_let_level_1_swallow_level_11() {
        let chosen = select_exact(&rows(), &["PRES:1 hybrid level".to_string()], &[]).unwrap();
        assert_eq!(chosen, vec![0]);
    }

    #[test]
    fn exact_selection_dedupes_and_keeps_index_order() {
        let chosen = select_exact(
            &rows(),
            &[
                "TMP:2 m above ground".to_string(),
                "PRES:surface".to_string(),
                "TMP:2 m above ground".to_string(),
            ],
            &[],
        )
        .unwrap();
        assert_eq!(chosen, vec![2, 3]);
    }

    #[test]
    fn exclusions_drop_the_accumulated_twin_of_a_surface_field() {
        // Without the exclusion both WEASD:surface rows are selected and
        // a 561-record bar reads 562.
        let both = select_exact(&rows(), &["WEASD:surface".to_string()], &[]).unwrap();
        assert_eq!(both, vec![4, 5]);
        let instantaneous = select_exact(
            &rows(),
            &["WEASD:surface".to_string()],
            &["acc fcst".to_string()],
        )
        .unwrap();
        assert_eq!(instantaneous, vec![4]);
    }

    #[test]
    fn an_exclusion_that_removes_every_match_is_a_hard_error() {
        let error = select_exact(
            &rows(),
            &["TMP:2 m above ground".to_string()],
            &["anl".to_string()],
        )
        .unwrap_err();
        assert!(error.contains("matched no index record"), "{error}");
    }

    #[test]
    fn a_pattern_matching_nothing_is_a_hard_error() {
        let error = select_exact(&rows(), &["SPFH:2 m above ground".to_string()], &[]).unwrap_err();
        assert!(error.contains("matched no index record"), "{error}");
        assert!(error.contains("silently smaller"), "{error}");
    }

    #[test]
    fn a_bare_variable_pattern_matches_every_level() {
        let chosen = select_exact(&rows(), &["PRES".to_string()], &[]).unwrap();
        assert_eq!(chosen, vec![0, 1, 2]);
    }

    #[test]
    fn an_alternation_matches_either_provider_spelling() {
        // NOMADS says CLWMR where AWS says CLMR for the same hybrid
        // cloud-water record; one selector has to serve both indexes.
        let aws = vec![IdxRow {
            sequence: 1,
            offset: 0,
            variable: "CLMR".to_string(),
            level: "1 hybrid level".to_string(),
            raw: "1:0:d=2026072812:CLMR:1 hybrid level:anl:".to_string(),
        }];
        let nomads = vec![IdxRow {
            variable: "CLWMR".to_string(),
            raw: "1:0:d=2026072812:CLWMR:1 hybrid level:anl:".to_string(),
            ..aws[0].clone()
        }];
        let selector = ["CLMR|CLWMR:1 hybrid level".to_string()];
        assert_eq!(select_exact(&aws, &selector, &[]).unwrap(), vec![0]);
        assert_eq!(select_exact(&nomads, &selector, &[]).unwrap(), vec![0]);
    }

    #[test]
    fn an_alternation_still_fails_closed_when_no_spelling_is_present() {
        let error = select_exact(
            &rows(),
            &["CLMR|CLWMR:1 hybrid level".to_string()],
            &[],
        )
        .unwrap_err();
        assert!(error.contains("matched no index record"), "{error}");
    }

    #[test]
    fn an_alternation_does_not_double_count_when_both_spellings_appear() {
        // Defensive: were a provider ever to publish both names, the
        // selector must still yield one record per level, not two.
        let mut both = rows();
        both.push(IdxRow {
            sequence: 7,
            offset: 600,
            variable: "CLMR".to_string(),
            level: "1 hybrid level".to_string(),
            raw: "7:600:d=2026072812:CLMR:1 hybrid level:anl:".to_string(),
        });
        both.push(IdxRow {
            sequence: 8,
            offset: 700,
            variable: "CLWMR".to_string(),
            level: "1 hybrid level".to_string(),
            raw: "8:700:d=2026072812:CLWMR:1 hybrid level:anl:".to_string(),
        });
        let chosen =
            select_exact(&both, &["CLMR|CLWMR:1 hybrid level".to_string()], &[]).unwrap();
        // Two records really are present, so two are selected and the
        // record bar downstream is what says so out loud.
        assert_eq!(chosen, vec![6, 7]);
    }

    #[test]
    fn grib_framing_needs_both_ends() {
        assert!(grib_framed(b"GRIB....7777"));
        assert!(!grib_framed(b"GRIB....7778"));
        assert!(!grib_framed(b"GRUB....7777"));
        assert!(!grib_framed(b"GRIB"));
    }

    #[test]
    fn hex_digest_matches_the_standard_known_answer() {
        assert_eq!(
            hex_digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn a_missing_index_is_classified_absent_not_malformed() {
        // Exactly the string read_idx builds.  A 404 reaches it as
        // wx_core's RustmetError::Http, whose Display prepends
        // "HTTP error: " -- which is why the old prefix match on the
        // inner text never fired, and every absent .idx was recorded
        // as one that arrived and failed validation.
        let transport = wx_core::RustmetError::Http(
            "HTTP request failed for https://example.invalid/x.idx: 404".to_string(),
        );
        let rendered = format!("{IDX_TRANSPORT_PREFIX}{transport}");
        assert!(
            rendered.contains("HTTP error: HTTP request failed"),
            "the double prefix is the whole bug: {rendered}"
        );
        assert!(idx_was_unfetchable(&rendered));

        // A body read that dies mid-transfer is equally "never arrived".
        let truncated = wx_core::RustmetError::Http(
            "failed to read https://example.invalid/x.idx: eof".to_string(),
        );
        assert!(idx_was_unfetchable(&format!(
            "{IDX_TRANSPORT_PREFIX}{truncated}"
        )));

        // ...while an index that DID arrive and then failed strict
        // validation stays classified as fetched, because that is a
        // different fact and the receipt keeps both.
        for malformed in [
            "index parsers disagree: wx-core parse_idx found 3 records,              strict validation found 4",
            "offsets are not strictly increasing",
            "index row 2 has no offset field",
        ] {
            assert!(!idx_was_unfetchable(malformed), "{malformed}");
        }
    }
}

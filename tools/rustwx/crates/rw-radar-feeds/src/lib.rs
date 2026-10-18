//! National radar-volume feeds as dated URL rows.
//!
//! The catalog adapters are BowEcho's ([`data_source::international`]): one
//! [`IntlProvider`] per national feed plus EUMETNET ORD for every other OPERA
//! country. They answer "what is the newest frame" and "the last N frames",
//! which is what a viewer needs. A data-assimilation cycle needs one more
//! question answered, "every frame of this site inside [start, end]", for a
//! window that may be live or years old. This crate answers it with no
//! per-network code path:
//!
//! * a provider that has dated archive lookup ([`ArchiveFrames`]) answers it
//!   from the archive;
//! * a provider with only a rolling window answers it from [`IntlProvider::recent`];
//! * either way each frame gets a nominal time ([`frame_stamp`]) read from its
//!   identity or file name, and the window is cut on that.
//!
//! Two providers gain the archive lookup their upstreams already offer and
//! BowEcho's adapters do not use: FMI (the AWS bucket keeps every volume from
//! 2007 under the same `{yyyy}/{mm}/{dd}/{site}/` layout the live adapter
//! reads) and DMI (the radardata STAC answers a `datetime` interval). They
//! wrap BowEcho's own providers, so their live behaviour is BowEcho's.
//!
//! Adding a network is a row: a new BowEcho provider appears here through
//! [`intl_providers`] with no change to this crate.

use chrono::{DateTime, Datelike, NaiveDateTime, TimeZone, Utc};
use data_source::international::{self as intl, FramePlan, IntlProvider, PlanPart};

#[doc(hidden)]
pub use data_source::international as __intl;

pub mod archive;

/// The decode format of a provider's frames. Everything BowEcho lists is
/// ODIM_H5 except JMA's multi-station polar GRIB2 tar.
pub fn frame_format(provider_id: &str) -> &'static str {
    match provider_id {
        "jma" => "JMA_POLAR_GRIB2_TAR",
        _ => "ODIM_H5",
    }
}

/// Every feed: BowEcho's registry, with FMI and DMI given their archives.
pub fn feeds() -> Vec<Box<dyn IntlProvider>> {
    intl::intl_providers()
        .into_iter()
        .map(|provider| -> Box<dyn IntlProvider> {
            match provider.id() {
                "fmi" => Box::new(archive::FmiArchive::new(provider)),
                "dmi" => Box::new(archive::DmiArchive::new(provider)),
                _ => provider,
            }
        })
        .collect()
}

/// Re-export so a consumer does not need its own `data_source` dependency.
pub fn intl_providers() -> Vec<Box<dyn IntlProvider>> {
    intl::intl_providers()
}

/// The feed with this id.
pub fn feed(id: &str) -> Result<Box<dyn IntlProvider>, String> {
    let all = feeds();
    let names: Vec<&'static str> = all.iter().map(|p| p.id()).collect();
    all.into_iter()
        .find(|p| p.id() == id)
        .ok_or_else(|| format!("no radar feed {id:?}; the feeds are {}", names.join(", ")))
}

/// The nominal time of a frame, read from its identity or, failing that, the
/// file name of its first part.
///
/// Every national feed names a frame by its time, in one of a few
/// spellings: `202606120635` (FMI, DMI), `20260612T1455` (ORD),
/// `2026061206455700` (DWD sweep start, centiseconds), `20260612_063500`
/// (NCI). All are an 8-digit date followed, after an optional `T` or `_`,
/// by HHMM. The first such run whose fields form a real date and time wins.
/// `None` when nothing in the names is a time: such a frame cannot be placed
/// in a window, and [`window`] refuses to guess where it falls.
///
/// A split frame (per-sweep files, `merge`) is stamped by its EARLIEST part:
/// DWD names the frame after its last sweep (`ess_20261016081402`, the
/// 08:14 terminal sweep of the volume that began 08:10:57), and a volume
/// belongs to the window its scan started in.
pub fn frame_stamp(plan: &FramePlan) -> Option<DateTime<Utc>> {
    if plan.merge
        && let Some(first) = plan
            .parts
            .iter()
            .filter_map(|part| part.url.rsplit('/').next().and_then(stamp_in))
            .min()
    {
        return Some(first);
    }
    if let Some(stamp) = stamp_in(&plan.identity) {
        return Some(stamp);
    }
    plan.parts
        .first()
        .and_then(|part| part.url.rsplit('/').next())
        .and_then(stamp_in)
}

fn stamp_in(text: &str) -> Option<DateTime<Utc>> {
    let bytes = text.as_bytes();
    let digit = |i: usize| bytes.get(i).is_some_and(u8::is_ascii_digit);
    let mut start = 0;
    while start + 12 <= bytes.len() {
        let run_start = start == 0 || !bytes[start - 1].is_ascii_digit();
        if run_start && (0..8).all(|k| digit(start + k)) {
            let mut time_at = start + 8;
            if matches!(bytes.get(time_at), Some(b'T') | Some(b'_')) {
                time_at += 1;
            }
            if (0..4).all(|k| digit(time_at + k)) {
                let date = &text[start..start + 8];
                let hhmm = &text[time_at..time_at + 4];
                if let Ok(naive) =
                    NaiveDateTime::parse_from_str(&format!("{date}{hhmm}"), "%Y%m%d%H%M")
                    && (1990..=2100).contains(&naive.year())
                {
                    return Some(Utc.from_utc_datetime(&naive));
                }
            }
        }
        start += 1;
    }
    None
}

/// One frame as a row: its identity, nominal time and the URLs that make it.
#[derive(Debug, Clone)]
pub struct FrameRow {
    pub provider: String,
    pub site: String,
    pub format: &'static str,
    pub identity: String,
    pub stamp: Option<DateTime<Utc>>,
    /// True when the parts are partial volumes of one scan (per-sweep or
    /// per-moment files) that belong together.
    pub merge: bool,
    pub urls: Vec<String>,
}

impl FrameRow {
    pub fn new(provider: &str, site: &str, plan: FramePlan) -> Self {
        let stamp = frame_stamp(&plan);
        FrameRow {
            provider: provider.to_string(),
            site: site.to_string(),
            format: frame_format(provider),
            identity: plan.identity,
            stamp,
            merge: plan.merge,
            urls: plan.parts.into_iter().map(|PlanPart { url }| url).collect(),
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "provider": self.provider,
            "site": self.site,
            "format": self.format,
            "identity": self.identity,
            "stamp": self.stamp.map(|s| s.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
            "merge": self.merge,
            "parts": self.urls.iter().map(|url| serde_json::json!({
                "url": url,
                "filename": url.rsplit('/').next().unwrap_or(url),
            })).collect::<Vec<_>>(),
        })
    }
}

/// How a window was answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowRoute {
    /// The provider's dated archive.
    Archive,
    /// The provider's rolling recent window (no archive exists for it).
    Recent,
}

impl WindowRoute {
    pub fn name(self) -> &'static str {
        match self {
            WindowRoute::Archive => "archive",
            WindowRoute::Recent => "recent",
        }
    }
}

/// Every frame of `site` whose nominal time is inside `[start, end]`,
/// oldest first.
///
/// Frames without a readable time are refused by name rather than kept or
/// dropped silently: kept, they could be from any hour; dropped, a window
/// would look thinner than the feed.
pub fn window(
    provider: &dyn IntlProvider,
    site: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    max: usize,
) -> Result<(WindowRoute, Vec<FrameRow>), String> {
    if end < start {
        return Err(format!("window end {end} precedes start {start}"));
    }
    let (route, plans) = match provider.archive_source() {
        Some(archive) => (
            WindowRoute::Archive,
            archive.window_plans(site, start, end, usize::MAX)?,
        ),
        None => (WindowRoute::Recent, provider.recent(site, max.max(1))?),
    };
    let mut rows = Vec::new();
    let mut unstamped = Vec::new();
    for plan in plans {
        let row = FrameRow::new(provider.id(), site, plan);
        match row.stamp {
            Some(stamp) if stamp >= start && stamp <= end => rows.push(row),
            Some(_) => {}
            None => unstamped.push(row.identity),
        }
    }
    if !unstamped.is_empty() {
        return Err(format!(
            "{} frame(s) of {}:{site} carry no readable time ({}); a frame that cannot \
             be placed in time cannot be placed in a cycle window",
            unstamped.len(),
            provider.id(),
            unstamped.join(", ")
        ));
    }
    rows.sort_by(|a, b| a.stamp.cmp(&b.stamp).then(a.identity.cmp(&b.identity)));
    rows.dedup_by(|a, b| a.identity == b.identity);
    let skip = rows.len().saturating_sub(max);
    Ok((route, rows.split_off(skip)))
}

/// Delegate every [`IntlProvider`] method to an inner BowEcho provider.
/// `archive_source` is the one each wrapper overrides.
macro_rules! delegate_provider {
    ($ty:ty) => {
        impl $crate::__intl::IntlProvider for $ty {
            fn id(&self) -> &'static str {
                self.inner.id()
            }
            fn label(&self) -> &'static str {
                self.inner.label()
            }
            fn country(&self) -> &'static str {
                self.inner.country()
            }
            fn list_sites(&self) -> Result<Vec<$crate::__intl::IntlSite>, String> {
                self.inner.list_sites()
            }
            fn latest(&self, site_id: &str) -> Result<$crate::__intl::FramePlan, String> {
                self.inner.latest(site_id)
            }
            fn recent_source(&self) -> Option<&dyn $crate::__intl::RecentFrames> {
                self.inner.recent_source()
            }
            fn archive_source(&self) -> Option<&dyn $crate::__intl::ArchiveFrames> {
                Some(self)
            }
            fn static_sites(&self) -> Vec<$crate::__intl::IntlSite> {
                self.inner.static_sites()
            }
        }
    };
}
pub(crate) use delegate_provider;

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(identity: &str, url: &str) -> FramePlan {
        FramePlan {
            identity: identity.into(),
            parts: vec![PlanPart { url: url.into() }],
            merge: false,
        }
    }

    fn at(y: i32, mo: u32, d: u32, h: u32, mi: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(y, mo, d, h, mi, 0).unwrap()
    }

    #[test]
    fn every_national_spelling_of_a_frame_time_is_read() {
        // FMI, DMI, ORD, DWD (sweep start with centiseconds), NCI.
        let cases = [
            ("202606120635_fianj_PVOL.h5", at(2026, 6, 12, 6, 35)),
            ("dkste_202605010030.vol.h5", at(2026, 5, 1, 0, 30)),
            ("nlhrw_20260612T1455_p1_h", at(2026, 6, 12, 14, 55)),
            (
                "ras07-vol5minng01_sweeph5onem_vradh_00-2026061206455700-asb-10103-hd5",
                at(2026, 6, 12, 6, 45),
            ),
            ("2_20240101_123000.pvol.h5", at(2024, 1, 1, 12, 30)),
        ];
        for (name, expected) in cases {
            assert_eq!(stamp_in(name), Some(expected), "{name}");
        }
    }

    #[test]
    fn a_name_with_no_time_has_no_stamp_and_a_station_number_is_not_one() {
        assert_eq!(stamp_in("ras07-asb-10103-hd5"), None);
        assert_eq!(stamp_in("06177"), None);
        // 8 digits that are not a date are skipped, the real stamp after them is read.
        assert_eq!(
            stamp_in("99999999_202606120635"),
            Some(at(2026, 6, 12, 6, 35))
        );
    }

    #[test]
    fn a_split_frame_is_stamped_by_its_earliest_part() {
        let p = FramePlan {
            identity: "ess_20261016081402_p50_h".into(),
            parts: vec![
                PlanPart { url: "https://x/ras07-vol5minng01_sweeph5onem_vradh_09-2026101608140200-ess-10410-hd5".into() },
                PlanPart { url: "https://x/ras07-vol5minng01_sweeph5onem_dbzh_00-2026101608105700-ess-10410-hd5".into() },
            ],
            merge: true,
        };
        assert_eq!(frame_stamp(&p), Some(at(2026, 10, 16, 8, 10)));
    }

    #[test]
    fn the_stamp_falls_back_to_the_first_part_file_name() {
        let p = plan("abc", "https://x/y/202606120635_fianj_PVOL.h5");
        assert_eq!(frame_stamp(&p), Some(at(2026, 6, 12, 6, 35)));
    }

    #[test]
    fn fmi_and_dmi_gain_an_archive_and_every_other_feed_is_bowechos() {
        let all = feeds();
        let bowecho = intl_providers();
        assert_eq!(all.len(), bowecho.len());
        for (ours, theirs) in all.iter().zip(&bowecho) {
            assert_eq!(ours.id(), theirs.id());
            if matches!(ours.id(), "fmi" | "dmi") {
                assert!(ours.supports_archive(), "{}", ours.id());
            } else {
                assert_eq!(ours.supports_archive(), theirs.supports_archive());
            }
        }
    }
}

# B5 addendum — the archive is 5418 objects, and why the pull says 5369

2026-08-04, beside `B5-OBS-ARCHIVE-PULL.md`, which is **not edited**: its
numbers are a true record of the pass it describes. This note reconciles them
with the manifests, which describe the archive as it now stands.

## The arithmetic

| | objects | bytes |
|---|---|---|
| the whole-day pass (`B5-OBS-ARCHIVE-PULL.md` section 1) | 5369 | 9.252 GB |
| day+1 radar supplements, seven case days x 7 frames | +49 | +0.096 GB |
| **the archive, as manifested** | **5418** | **9.348 GB** |

The manifests carrying the second row are
`OBS-ARCHIVE-MANIFEST.json` and `manifests/obs-YYYYMMDD.json`, re-issued at
`2a27ab61`.

## Why there are 49 more objects than were pulled

The radar pull took whole UTC days, `D 00:00:00 .. D 23:59:59`. The scored
window does not end at midnight: every case initialises at 12 UTC and is
scored F02-F18, so its last seven hourly valid times fall on `D+1` and their
frames live under the next day's S3 prefix. The pack builds fetched exactly
those seven frames per case day, by valid time, and nothing else from that
prefix — the day+1 part of the archive is hourly, not the 2-minute cadence the
day-of part carries.

Per case day the radar count therefore reads:

| case day | day-of | day+1 | total |
|---|---|---|---|
| 2021-12-10 | 718 | 7 | 725 |
| 2023-03-31 | 708 | 7 | 715 |
| 2024-04-27 | 720 | 7 | 727 |
| 2024-05-21 | 720 | 7 | 727 |
| 2024-07-15 | 720 | 7 | 727 |
| 2025-03-14 | 720 | 7 | 727 |
| 2025-05-16 | 720 | 7 | 727 |

The two short day-of counts are the archive's own gaps, unchanged from the
pull receipt's section 2. Stage-IV is unchanged at 46 objects a case day.

Each case's manifest now names both pulls in `mrms.windows`, so a count and
the window that produced it stay legible; `mrms.window` still carries the
whole-day pull it always described.

## What else moved in the same re-issue

The ASOS blocks of six case days carry new digests. Those station tables were
frozen at 09:49 from provisional case boxes, twenty-four minutes before the
wizard-derived centres were committed, and the two disagreed — by as much as
2.35 degrees of a box edge on B-07. Every one is re-frozen from its case's
committed box with the screen thresholds identical to the shakedown case's.
The shakedown case 2024-05-21 is the one that did not move: its provisional
box was already its committed box.

| case day | stations frozen, before -> after |
|---|---|
| 2021-12-10 | 799 -> 830 |
| 2023-03-31 | 743 -> 816 |
| 2024-04-27 | 575 -> 599 |
| 2024-05-21 | 642 (unchanged) |
| 2024-07-15 | 698 -> 738 |
| 2025-03-14 | 775 -> 782 |
| 2025-05-16 | 802 -> 839 |

The superseded tables were moved, not deleted, to
`<cache>/obsbattery/battery/asos/superseded-provisional-box/<case-day>/`.

## Integrity

Every object the re-issued manifests name was re-hashed against the digest its
front door took at fetch: **5418 of 5418 match, none missing**, and every
per-case manifest digest agrees with the index that carries it.

No forecast was run for this document and it makes no claim about skill.

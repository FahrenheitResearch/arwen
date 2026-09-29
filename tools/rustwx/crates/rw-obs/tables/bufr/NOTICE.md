# WMO BUFR master tables, version 46

`element.table` (Table B, the element descriptors: code, abbreviation,
type, name, unit, scale, reference value, bit width) and `sequence.def`
(Table D, the sequence descriptors) as distributed by ECMWF's ecCodes
(`definitions/bufr/tables/0/wmo/46/`, Apache License 2.0, Copyright
2005- ECMWF), transcribing the WMO Manual on Codes (WMO-No. 306), FM 94
BUFR master table version 46.  Table B and D entries are stable across
versions (a published entry never changes its width, scale or reference;
later versions add entries), so this one table set decodes messages of
every earlier master table version the WIS2 global broker carries
(measured 2026-09-06: versions 22 to 38 across 29 centres).

`rw-obs/src/bufr.rs` reads these files at build time (`include_str!`);
a local descriptor (class 48 and above, or an entry these tables do not
carry) is refused by name unless the message skips it with operator 206.

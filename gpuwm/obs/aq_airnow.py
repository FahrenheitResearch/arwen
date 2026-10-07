"""AirNow orchestration over Rust-decoded neutral tables and companion records."""
from __future__ import annotations
import json
from pathlib import Path
from gpuwm.obs.frontdoor import AIRNOW
from gpuwm.obs.obspack import sha256_of
from gpuwm.obs.aq_table import VARIABLE_TABLE



def fetch_airnow(start, end, out, bbox=None):
    """Fetch bytes and decode them in Rust. Return the neutral table path.

    Times are inclusive UTC sample-start times. Existing matching digests
    resume unchanged; delete a fetch receipt explicitly to refresh revisions.
    """
    binary = AIRNOW.require()
    ok, detail = AIRNOW.probe(binary)
    if not ok:
        raise RuntimeError(detail)
    out = Path(out)
    window = ["--start", str(start), "--end", str(end)]
    AIRNOW.run("fetch", window + ["--out", str(out)],
               schema="gpuwm-obs.airnow-fetch.v1", binary=binary)
    table = out / "airnow.csv"
    arguments = window + ["--dir", str(out), "--out", str(table),
                           "--fetch-record", str(out / "fetch.json")]
    if bbox is not None:
        arguments += ["--bbox", ",".join(map(str, bbox))]
    AIRNOW.run("table", arguments, schema="gpuwm-obs.airnow-table.v1", binary=binary)
    return table


class AirNowStationSource:
    """StationObsSource over a neutral CSV and its Rust-written JSON index.

    Python never decodes the archive or the CSV. The companion index carries
    the same station/time/variable values Rust wrote into the table.
    """
    def __init__(self, table_path):
        self.path = Path(table_path)
        self.record = json.loads(self.path.with_suffix(".json").read_text())
        if self.record.get("schema") != "gpuwm-obs.airnow-table.v1":
            raise ValueError("not an AirNow table record")
        if self.record.get("table_schema") != "gpuwm-obs.table.v2":
            raise ValueError("not a neutral v2 table")
        if sha256_of(self.path) != self.record["sha256"]:
            raise ValueError("AirNow table digest mismatch")

    def observations(self, valid_times):
        from gpuwm.obs.sources import _contracts
        c = _contracts()
        wanted = {str(t) for t in valid_times}
        for t in wanted:
            c.parse_valid_time(t)
        stations = tuple(c.Station(**s) for s in self.record["stations"])
        reports = []
        for row in self.record["reports"]:
            if row["valid_time"] not in wanted:
                continue
            for name, value in row["values"].items():
                spec = VARIABLE_TABLE.get(name)
                if spec is None or not spec[1] <= value <= spec[2]:
                    raise ValueError(f"invalid AirNow value {name}")
            reports.append(c.StationReport(**row))
        provenance = c.ObsProvenance(
            source="airnow", product="HourlyData", uri=str(self.path.resolve()),
            sha256=self.record["sha256"], fetched_at=self.record["created_at"])
        return c.StationObsSet(stations=stations, reports=tuple(reports), provenance=provenance)

    def valid_times(self):
        return tuple(sorted({r["valid_time"] for r in self.record["reports"]}))

    def verify(self, provenance, *, root=None):
        from gpuwm.obs.sources import _GriddedSource
        return _GriddedSource.verify(self, provenance, root=root)


__all__ = ["AirNowStationSource", "fetch_airnow"]

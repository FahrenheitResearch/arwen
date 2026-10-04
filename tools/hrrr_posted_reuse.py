"""Reuse one pinned ordinary HRRR producer in posted ensemble members."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


class SharedPostedHrrr:
    """Carry the source's existing admission and seal into a member writer.

    The member consumes physical snapshots and performs its own native real
    initialization. It neither opens raw GRIB nor launches a decoder. Source
    segment markers retain the original posted and decoded byte authorities.
    """

    def __init__(self, context, *, cycle, source_forecast_hours):
        self.context = context.verify()
        self.head = context.prepared_head
        self.posted = self.head["basis"]["as_posted"]
        self.hours = tuple(source_forecast_hours)
        actual_cycle = context.trajectory.cycle.replace(tzinfo=None)
        if (context.trajectory.source != "hrrr" or actual_cycle != cycle
                or tuple(self.posted["forcing_leads"]) != self.hours):
            raise ValueError("shared HRRR preparation differs from the requested source window")
        self.waits = []
        self.writer = None
        self.certificate = None
        self.markers = {}
        self.decoded = {}

    def route_table_sha256(self):
        self.context.verify()
        return self.context.source_plan["route_table_sha256"]

    def acquire(self, hour):
        self.context.verify()
        if hour:
            self.context.require_interval(hour - 1)
        instant = self.context.physical_stream.times[hour]
        self.context.physical_stream.require(instant)
        return SimpleNamespace(valid_time=instant.replace(tzinfo=None))

    def seal(self, args):
        """Reuse the original verified bridge and complete lead record bodies."""
        from tools.prepare_hrrr_wrf import _link_file_create, _manifest_entries, _sha256
        self.certificate = self.context.capture_seal()
        artifacts = self.certificate["artifacts"]
        record = json.loads(artifacts["boundary-stream/posted-leads.json"])
        if record["route_table_sha256"] != self.route_table_sha256():
            raise ValueError("shared HRRR source seal changed its route table")
        self.markers = {int(key): value["marker"] for key, value in record["leads"].items()}
        self.decoded = {int(key): value["decoded"] for key, value in record["leads"].items()}
        documents = self.posted["documents"]
        bridge = self.context.prepared_root / documents["bridge"]["path"]
        # Only the ordinary source consumed these decoded payloads. Linking
        # its exact sealed files retains portable provenance without decoding
        # or mapping them again in every member.
        for name in _manifest_entries(bridge):
            _link_file_create(bridge.parent / name, Path(args.bridge) / name)
        for role, destination in (("bridge", Path(args.bridge) / "SHA256SUMS"),
                                  ("source_manifest", Path(args.source_manifest))):
            name = documents[role]["path"]
            origin = self.context.prepared_root / name
            if origin.read_bytes() != artifacts[name].encode("utf-8"):
                raise ValueError("shared HRRR source document changed after seal verification")
            _link_file_create(origin, destination)
        args.manifest_sha256 = _sha256(Path(args.bridge) / "SHA256SUMS")
        args.source_manifest_sha256 = _sha256(Path(args.source_manifest))
        self.context.verify()
        operation = "reused_posted_native_source"
        common = {"operation": operation,
                  "ordinary_source_head_sha256": self.head["head_sha256"],
                  "bridge_manifest_sha256": args.manifest_sha256,
                  "source_manifest_sha256": args.source_manifest_sha256}
        return ({**common, "decoder_invocations": 0,
                 "workers": {"requested": str(args.pipeline_workers).strip().lower(),
                             "selected": 0, "operation": operation}}, common)

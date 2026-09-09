"""Run with: python -m unittest discover -s examples -v"""
import json
from pathlib import Path
import tempfile
import unittest
from client import InterfaceError, RunReader


class DurableClientTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.manifest = self.root / "run-manifest.json"
        self.manifest.write_text(json.dumps({"schema": "gpuwm.run-manifest.v1", "run_id": "run-a",
                                            "events_path": str(self.root / "events.jsonl")}), encoding="utf-8")
        self.events = self.root / "events.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def event(self, sequence, **fields):
        return json.dumps({"schema_version": "gpuwm.run-plan.event.v1", "sequence": sequence,
                           "event": "output_committed", "emitted_unix_ms": 1700000000000 + sequence, **fields}).encode()

    def test_torn_tail_waits_then_reads_once(self):
        reader = RunReader(self.manifest)
        first, second = self.event(1), self.event(2)
        self.events.write_bytes(first + b"\n" + second[:15])
        self.assertEqual([event["sequence"] for event in reader.snapshot()["new_events"]], [1])
        self.assertEqual(reader.snapshot()["new_events"], [])
        with self.events.open("ab") as stream:
            stream.write(second[15:] + b"\n")
        self.assertEqual([event["sequence"] for event in reader.snapshot()["new_events"]], [2])

    def test_changed_manifest_refuses_reattach(self):
        reader = RunReader(self.manifest)
        self.manifest.write_text(self.manifest.read_text().replace("run-a", "run-b"), encoding="utf-8")
        with self.assertRaisesRegex(InterfaceError, "producer manifest changed"):
            reader.snapshot()

    def test_wrong_run_and_repeated_sequence_are_rejected(self):
        for payload in (self.event(1, run_id="run-b") + b"\n",
                        self.event(1) + b"\n" + self.event(1) + b"\n"):
            self.events.write_bytes(payload)
            with self.assertRaises(InterfaceError):
                RunReader(self.manifest).snapshot()

    def test_truncated_stream_is_rejected(self):
        reader = RunReader(self.manifest)
        self.events.write_bytes(self.event(1) + b"\n")
        reader.snapshot()
        self.events.write_bytes(b"")
        with self.assertRaisesRegex(InterfaceError, "truncated"):
            reader.snapshot()


    def test_progress_is_bound_to_the_manifest_run(self):
        progress = self.root / "run-progress.json"
        progress.write_text(json.dumps({"schema": "gpuwm.run-progress/v1", "run_id": "run-a", "status": "complete"}), encoding="utf-8")
        reader = RunReader(self.manifest)
        self.assertEqual(reader.snapshot()["progress"]["run_id"], "run-a")
        progress.write_text(json.dumps({"schema": "gpuwm.run-progress/v1", "run_id": "run-b"}), encoding="utf-8")
        with self.assertRaisesRegex(InterfaceError, "Progress belongs to another run"):
            reader.snapshot()

    def test_event_tag_and_timestamp_are_required(self):
        for fields in ({"event": None}, {"event": ""}, {"emitted_unix_ms": None}, {"emitted_unix_ms": True}):
            self.events.write_bytes(self.event(1, **fields) + b"\n")
            with self.assertRaisesRegex(InterfaceError, "event tag and emission timestamp"):
                RunReader(self.manifest).snapshot()

    def test_nonobject_documents_have_interface_errors(self):
        self.events.write_text("[]\n", encoding="utf-8")
        with self.assertRaisesRegex(InterfaceError, "JSON object"):
            RunReader(self.manifest).snapshot()
        self.events.write_text("", encoding="utf-8")
        (self.root / "run-progress.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(InterfaceError, "Unsupported run progress"):
            RunReader(self.manifest).snapshot()
        self.manifest.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(InterfaceError, "Unsupported run manifest"):
            RunReader(self.manifest)


if __name__ == "__main__":
    unittest.main()

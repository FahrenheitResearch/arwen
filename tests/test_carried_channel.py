"""Exact CPU transport tests, with disposable Git histories and no runtime imports."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools/carried_physics_channel.py"
spec = importlib.util.spec_from_file_location("channel_tested", SOURCE)
channel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(channel)
SCOPE = {"schema": channel.SCOPE_SCHEMA, "engine_root": "gpuwm", "units": [
    {"engine": "gpuwm/core/x.py", "carried": "core/x.py", "kind": "file"},
    {"engine": "gpuwm/data/tables", "carried": "data/tables", "kind": "tree"}]}
OLD = {"commit": "a" * 40, "tree": "b" * 40, "committed_utc": "2026-09-12T00:00:00+00:00", "version": "old"}
NEW = {"commit": "c" * 40, "tree": "d" * 40, "committed_utc": "2026-09-12T01:00:00+00:00", "version": "new"}


def release(old=b"old", new=b"new"):
    return channel.build_release(SCOPE, {"gpuwm/core/x.py": old},
                                 {"gpuwm/core/x.py": new}, OLD, NEW)


class TransportTests(unittest.TestCase):
    def test_presence_encoding_matrix(self):
        values = [None, b"", b"\n", b"x", b"x\n", b"x\r\n", b"x\r", b"\xff\x00", "snow \u2603\n".encode()]
        for old in values:
            for new in values:
                with self.subTest(old=old, new=new):
                    obj = json.loads(json.dumps(release(old, new)))
                    a, b = channel.unpack_release(obj)
                    self.assertEqual(a["gpuwm/core/x.py"], old)
                    self.assertEqual(b["gpuwm/core/x.py"], new)
                    self.assertEqual(obj["files"][0]["changed"], old != new)
                    for hunk in obj["files"][0]["release_hunks"]:
                        self.assertNotIn("engine_sha256", hunk)
                        i, j = hunk["new_bytes"]
                        self.assertEqual(hunk["new_sha256"], channel.digest((new or b"")[i:j]))

    def test_missing_is_not_absent(self):
        with self.assertRaisesRegex(channel.ChannelError, "explicit inventory"):
            channel.build_release(SCOPE, {"gpuwm/core/x.py": b"x"}, {}, OLD, NEW)
        with self.assertRaisesRegex(channel.ChannelError, "omit a mapped file"):
            channel.build_release(SCOPE, {}, {}, OLD, NEW)

    def test_raw_vs_pair_hash_namespaces(self):
        obj = release(b"a\nb\nc\n", b"a\nB\nc\n")
        hunk = obj["files"][0]["release_hunks"][0]
        self.assertEqual(hunk["new_bytes"], [2, 4])
        self.assertEqual(hunk["new_sha256"], channel.digest(b"B\n"))
        self.assertNotIn("carried_sha256", hunk)

    def test_untrusted_id_does_not_authenticate_its_own_contents(self):
        obj = release()
        obj["new"]["version"] = "forged"
        before, after = channel.unpack_release(release())
        altered = channel.build_release(SCOPE, before, after, OLD, {**NEW, "version": "forged"})
        with self.assertRaisesRegex(channel.ChannelError, "trusted release id"):
            channel.require_release(altered, release()["id"])
        with self.assertRaises(channel.ChannelError):
            channel.require_release(obj, obj["id"])

    def test_snapshot_metadata_needs_full_object_ids_and_aware_time(self):
        for override in ({"commit": "main"}, {"tree": "abcd"}, {"committed_utc": "2026-09-12"},
                         {"committed_utc": "yesterday"}, {"unexpected": "value"}):
            with self.subTest(override=override), self.assertRaises(channel.ChannelError):
                channel.build_release(SCOPE, {"gpuwm/core/x.py": b"a"},
                                      {"gpuwm/core/x.py": b"b"}, OLD, {**NEW, **override})

    def test_release_metadata_is_owned_not_aliased(self):
        scope = copy.deepcopy(SCOPE)
        old, new = dict(OLD), dict(NEW)
        obj = channel.build_release(scope, {"gpuwm/core/x.py": b"a"},
                                    {"gpuwm/core/x.py": b"b"}, old, new)
        frozen = copy.deepcopy(obj)
        scope["units"].clear(); old["version"] = "changed"; new["version"] = "changed"
        self.assertEqual(obj, frozen)
        channel.unpack_release(obj)

    def test_tampered_resealed_raw_blob(self):
        obj = release()
        obj["files"][0]["after"]["base64"] = "dGFtcGVyZWQ="
        with self.assertRaisesRegex(channel.ChannelError, "hash or size"):
            channel.unpack_release(channel.seal(obj))

    def test_resealed_bad_size_bool(self):
        obj = release(b"a", b"b")
        obj["files"][0]["after"]["size"] = True
        with self.assertRaises(channel.ChannelError):
            channel.unpack_release(channel.seal(obj))

    def test_noncanonical_base64_refused(self):
        obj = release(b"a", b"b")
        obj["files"][0]["after"]["base64"] = "Yh=="
        with self.assertRaisesRegex(channel.ChannelError, "noncanonical"):
            channel.unpack_release(channel.seal(obj))

    def test_omitted_explicit_file_refused(self):
        obj = release()
        obj["files"] = []
        with self.assertRaises(channel.ChannelError):
            channel.unpack_release(channel.seal(obj))

    def test_unknown_or_duplicate_inventory_refused(self):
        for mutation in (lambda o: o.update(extra=True),
                         lambda o: o["files"].append(copy.deepcopy(o["files"][0]))):
            obj = release(); mutation(obj)
            with self.assertRaises(channel.ChannelError):
                channel.unpack_release(channel.seal(obj))

    def test_path_boundary_validation(self):
        for name in ("/absolute", "../x", "gpuwm/../x", "gpuwm//x", "gpuwm/./x",
                     "C:x", "a\\b", "x\x00", "x\x7f", "x\ud800", "a/"):
            with self.subTest(name=name), self.assertRaises(channel.ChannelError):
                channel.safe_path(name)
        scope = copy.deepcopy(SCOPE)
        scope["units"].append({"engine": "gpuwm/data/tables/child", "carried": "data/child", "kind": "file"})
        with self.assertRaisesRegex(channel.ChannelError, "overlapping"):
            channel.validate_scope(scope)

    def test_outside_scope_not_exported(self):
        with self.assertRaises(channel.ChannelError):
            channel.build_release(SCOPE, {"gpuwm/core/x.py": b"x", "gpuwm/private/y": b"z"},
                {"gpuwm/core/x.py": b"x", "gpuwm/private/y": b"z"}, OLD, NEW)

    def test_duplicate_nonfinite_json_refused(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "input.json"
            for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
                p.write_text(text)
                with self.assertRaises(channel.ChannelError):
                    channel.read_json(p)

    def test_new_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as name:
            p = Path(name) / "out.json"
            channel.write_new(p, release()); before = p.read_bytes()
            with self.assertRaises(channel.ChannelError):
                channel.write_new(p, release(b"a", b"b"))
            self.assertEqual(p.read_bytes(), before)

    def test_default_output_allocates_distinct_generations(self):
        with tempfile.TemporaryDirectory() as name:
            program = ("import runpy,sys; c=runpy.run_path(sys.argv[1]); "
                       "a=c['_write_output'](None,'test',{}); "
                       "b=c['_write_output'](None,'test',{}); assert a != b; "
                       "assert a.is_file() and b.is_file()")
            subprocess.run([sys.executable, "-c", program, str(SOURCE)], cwd=name, check=True)

    def test_no_runtime_imports(self):
        program = ("import runpy,sys; runpy.run_path(sys.argv[1]); "
                   "assert not {'gpuwm','arwen_global','cupy'} & set(sys.modules)")
        subprocess.run([sys.executable, "-c", program, str(SOURCE)], check=True)


class GitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "-q"); self.git("config", "user.name", "Fixture")
        self.git("config", "user.email", "fixture@example.invalid")
        self.git("config", "core.autocrlf", "false")
        self.put("gpuwm/core/x.py", b"old\r\n")
        self.put("gpuwm/data/tables/removed.bin", b"\xff")
        self.put("private/unselected", b"not selected")
        self.commit("old"); self.old = self.git("rev-parse", "HEAD").strip()
        self.put("gpuwm/core/x.py", b"new\r\n")
        (self.root / "gpuwm/data/tables/removed.bin").unlink()
        self.put("gpuwm/data/tables/NEW.TBL", b"parameter=1\n")
        self.commit("new"); self.new = self.git("rev-parse", "HEAD").strip()

    def put(self, path, raw):
        p = self.root / path; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(raw)

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.root), *args], capture_output=True,
                              check=True, text=True).stdout

    def commit(self, message):
        self.git("add", "."); self.git("commit", "-qm", message)

    def put_index(self, path, raw, mode="100644"):
        """Build literal Git entries without platform filesystem restrictions."""
        blob = subprocess.run(["git", "-C", str(self.root), "hash-object", "-w", "--stdin"],
                              input=raw, capture_output=True, check=True).stdout.decode().strip()
        self.git("update-index", "--add", "--cacheinfo", mode, blob, path)

    def manifest(self):
        return channel.create_release(self.root, self.old, self.new, SCOPE, "old", "new")

    def test_union_reads_git_not_worktree(self):
        self.put("gpuwm/core/x.py", b"dirty")
        obj = self.manifest(); channel.verify_git(obj, self.root)
        before, after = channel.unpack_release(obj)
        self.assertEqual(before["gpuwm/core/x.py"], b"old\r\n")
        self.assertEqual(after["gpuwm/core/x.py"], b"new\r\n")
        self.assertIsNone(before["gpuwm/data/tables/NEW.TBL"])
        self.assertEqual(after["gpuwm/data/tables/NEW.TBL"], b"parameter=1\n")
        self.assertIsNone(after["gpuwm/data/tables/removed.bin"])
        self.assertNotIn("private/unselected", after)

    def test_omitted_engine_directory_child_detected_against_git(self):
        obj = self.manifest()
        obj["files"] = [f for f in obj["files"] if not f["path"].endswith("NEW.TBL")]
        obj = channel.seal(obj)
        # Internally consistent transport cannot prove inventory completeness.
        channel.unpack_release(obj)
        with self.assertRaisesRegex(channel.ChannelError, "named Git"):
            channel.verify_git(obj, self.root)
        with self.assertRaisesRegex(channel.ChannelError, "trusted release id"):
            channel.require_release(obj, self.manifest()["id"])

    def test_symlink_file_and_ancestor_refused(self):
        self.put_index("gpuwm/core/x.py", b"../../private/unselected", "120000")
        self.git("commit", "-qm", "link")
        with self.assertRaises(channel.ChannelError):
            channel.GitTree(self.root, "HEAD", "x").snapshot(SCOPE)
        self.git("update-index", "--force-remove", "gpuwm/core/x.py")
        self.put_index("gpuwm/core", b"../data", "120000")
        self.git("commit", "-qm", "ancestor")
        with self.assertRaisesRegex(channel.ChannelError, "ancestor"):
            channel.GitTree(self.root, "HEAD", "x").snapshot(SCOPE)

    def test_directory_file_type_change_refused(self):
        (self.root / "gpuwm/core/x.py").unlink()
        self.put("gpuwm/core/x.py/child", b"x"); self.commit("directory")
        with self.assertRaisesRegex(channel.ChannelError, "file is a directory"):
            channel.GitTree(self.root, "HEAD", "x").snapshot(SCOPE)

    def test_literal_git_paths(self):
        name = "gpuwm/core/[wild].py"
        self.put_index(name, b"literal"); self.git("commit", "-qm", "literal")
        scope = {"schema": channel.SCOPE_SCHEMA, "engine_root": "gpuwm", "units": [
            {"engine": name, "carried": "core/literal.py", "kind": "file"}]}
        self.assertEqual(channel.GitTree(self.root, "HEAD", "x").snapshot(scope), {name: b"literal"})

    def test_cli_emit_verify_and_collision(self):
        scope = self.root / "scope.json"; channel.write_new(scope, SCOPE)
        output = self.root / "release.json"
        command = [sys.executable, str(SOURCE), "emit", "--repo", str(self.root), "--scope", str(scope),
                   "--old", self.old, "--new", self.new, "--old-version", "old", "--new-version", "new",
                   "--out", str(output)]
        subprocess.run(command, check=True, capture_output=True)
        self.assertEqual(channel.read_json(output), self.manifest())
        proc = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("already exists", proc.stderr)
        subprocess.run([sys.executable, str(SOURCE), "verify-git", "--repo", str(self.root),
                        "--manifest", str(output)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()

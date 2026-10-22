"""Durable writer identities and consumer receipts for hourly raw retirement."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


class EnsembleHistoryLedger:
    CONTRACT = "gpuwm-ensemble-history-ledger.v1"
    RETIREMENT = "gpuwm-ensemble-history-retirement.v1"

    def __init__(self, root, *, member_order, resume=False):
        self.root = Path(root).resolve()
        self.member_order = tuple(member_order)
        self.path = self.root / "ensemble-history-ledger.json"
        self.retirements = self.root / "ensemble-history-retirements"
        self.records = {}
        self._lock = threading.RLock()
        if resume and self.path.is_file():
            from gpuwm.ensemble.restart_roster import resolve_file
            document = json.loads(self.path.read_text(encoding="utf-8"))
            if (document.get("schema") != self.CONTRACT
                    or document.get("member_order") != list(self.member_order)):
                raise ValueError("retained history ledger belongs to another ensemble roster")
            for row in document.get("records", ()):
                if (row.get("member_id") not in self.member_order
                        or not _digest(row.get("sha256"))
                        or type(row.get("bytes")) is not int or row["bytes"] < 0):
                    raise ValueError("retained ensemble history ledger has invalid writer identities")
                resolve_file(self.root, row["path"])
                if row["path"] in self.records:
                    raise ValueError("retained ensemble history ledger repeats a writer path")
                self.records[row["path"]] = dict(row)
            self.validate_retirements()

    def _write(self):
        temporary = self.path.with_suffix(".json.pending")
        self.root.mkdir(parents=True, exist_ok=True)
        with temporary.open("w", encoding="utf-8") as file:
            json.dump({"schema": self.CONTRACT, "member_order": list(self.member_order),
                "records": self.inventory()}, file, indent=2, sort_keys=True)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(self.path)

    def inventory(self):
        with self._lock:
            return [dict(row) for row in sorted(self.records.values(), key=lambda row:
                (self.member_order.index(row["member_id"]), row["path"]))]

    def register(self, proof, *, member_id, grid_id, episode, valid_time):
        """Record a process-local completed writer proof before raw consumers."""
        from gpuwm.output_identity import CompletedFileRecord
        if not isinstance(proof, CompletedFileRecord):
            raise TypeError("ensemble raw registration requires the completed writer proof")
        if member_id not in self.member_order or not _digest(proof.sha256):
            raise ValueError("ensemble history proof has invalid member or SHA-256 identity")
        path = Path(proof.path).resolve().relative_to(self.root).as_posix()
        domain = f"d{int(grid_id):02d}" + (f"-episode-{int(episode):03d}" if episode else "")
        valid = valid_time.strftime("%Y-%m-%d_%H:%M:%S") if hasattr(valid_time, "strftime") else str(valid_time)
        record = {"path": path, "bytes": int(proof.size), "sha256": proof.sha256,
            "member_id": int(member_id), "grid_id": int(grid_id), "episode": int(episode),
            "domain": domain, "valid_time": valid, "retirement_state": "registered"}
        with self._lock:
            previous = self.records.get(path)
            if previous is not None:
                identity = {key: previous[key] for key in record if key != "retirement_state"}
                if identity != {key: value for key, value in record.items() if key != "retirement_state"}:
                    raise ValueError("ensemble history path was registered with different writer bytes")
                return dict(previous)
            self.records[path] = record
            self._write()
            return dict(record)

    def validate_retirements(self):
        """Keep original identities when a verified consumer has removed raw."""
        with self._lock:
            for path, record in self.records.items():
                raw = self.root / path
                token = hashlib.sha256(path.encode("utf-8")).hexdigest()
                receipt_path = self.retirements / f"{token}.json"
                if not receipt_path.is_file():
                    if not raw.is_file():
                        raise RuntimeError(f"ensemble raw history is missing without its verified retirement receipt: {path}")
                    continue
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                products = receipt.get("products", ())
                if (receipt.get("schema") != self.RETIREMENT
                        or any(receipt.get(key) != record[key] for key in ("path", "sha256", "bytes"))
                        or not receipt.get("retired_at")
                        or not _digest(receipt.get("product_manifest_sha256"))
                        or not products
                        or any(not isinstance(row, dict) or not row.get("path")
                            or not _digest(row.get("sha256"))
                            or type(row.get("bytes")) is not int or row["bytes"] < 0 for row in products)):
                    raise RuntimeError(f"ensemble history retirement receipt does not verify its raw identity and products: {path}")
                record["retirement_state"] = "registered" if raw.is_file() else "retired"
                record["retirement_receipt"] = receipt_path.relative_to(self.root).as_posix()
            if self.records:
                self._write()
            return self.inventory()

    def retire_consumed(self, manifest, products, *, workers=4):
        """Commit consumer proofs before removing this ledger's own raw files."""
        from concurrent.futures import ThreadPoolExecutor
        from datetime import datetime, timezone
        from gpuwm.ensemble.restart_roster import resolve_file

        def digest(path):
            with path.open('rb') as stream:
                return hashlib.file_digest(stream, 'sha256').hexdigest()
        manifest = Path(manifest).resolve()
        manifest.relative_to(self.root)
        manifest_sha256 = digest(manifest)
        if not products:
            raise ValueError('raw retirement requires durable product artifacts')
        for product in products:
            path = resolve_file(self.root, product['path'])
            if path.stat().st_size != product['bytes'] or digest(path) != product['sha256']:
                raise RuntimeError('raw consumer product changed before history retirement')
        with self._lock:
            records = self.inventory()
            def verify(record):
                path = resolve_file(self.root, record['path'])
                if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
                    raise RuntimeError('raw history changed before cyclone consumer retirement')
                return record
            # Validate all writer identities before deleting any file.
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(verify, records))
            self.retirements.mkdir(exist_ok=True)
            deleted = []
            for record in records:
                token = hashlib.sha256(record['path'].encode()).hexdigest()
                path = self.retirements/f'{token}.json'
                receipt = dict(schema=self.RETIREMENT,
                    **{k: record[k] for k in ('path', 'sha256', 'bytes')},
                    retired_at=datetime.now(timezone.utc).isoformat(),
                    product_manifest_sha256=manifest_sha256,
                    product_manifest=manifest.relative_to(self.root).as_posix(), products=products)
                temporary = path.with_suffix('.json.pending')
                with temporary.open('w') as stream:
                    json.dump(receipt, stream, indent=2, sort_keys=True)
                    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
                temporary.replace(path)
                directory_fd = os.open(self.retirements, os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
                (self.root/record['path']).unlink()
                deleted.append(dict(path=record['path'], bytes=record['bytes']))
            self.validate_retirements()
            return deleted

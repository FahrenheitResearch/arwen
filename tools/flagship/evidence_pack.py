#!/usr/bin/env python3
"""Assemble a cryptographically bound Phase-5 close-out evidence pack."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from gpuwm.verify import nest_gates  # noqa: E402 (read-only authority import)


_AMENDMENT_RE = re.compile(r"\bF\d+[a-z]?\b", re.IGNORECASE)
_OPTIONAL_MILESTONES = frozenset({"P5B"})
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")


class EvidencePackError(ValueError):
    """Raised when evidence cannot support an unambiguous bound pack."""


@dataclass(frozen=True)
class FoundEvidence:
    milestone: str
    metric: str
    passed: bool
    pointer: str
    amendment_references: tuple[str, ...]
    priority: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_file(path: Path, *, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(block_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EvidencePackError(f"{label} must be an exact integer")
    return value


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidencePackError(f"malformed {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidencePackError(f"malformed {label} {path}: root must be an object")
    return payload


def _json_pointer(parts: Iterable[object]) -> str:
    encoded = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "" if not encoded else "/" + "/".join(encoded)


def _amendment_values(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        found.update(match.upper() for match in _AMENDMENT_RE.findall(value))
        if "amend" in value.lower() and not found:
            found.add(value.strip())
    elif isinstance(value, Mapping):
        for key, item in value.items():
            if "amend" in str(key).lower() and item is True:
                found.add(str(key))
            found.update(_amendment_values(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_amendment_values(item))
    return {item for item in found if item}


def _gate_row(row: object, *, default_metric: str | None,
              milestone: str, pointer: str, priority: int) -> FoundEvidence | None:
    if isinstance(row, bool) and default_metric is not None:
        return FoundEvidence(milestone, default_metric, row, pointer, (), priority)
    if not isinstance(row, Mapping):
        return None
    metric = row.get("metric", default_metric)
    passed = row.get("passed")
    if not isinstance(metric, str) or not metric or not isinstance(passed, bool):
        return None
    declared = row.get("milestone", row.get("registered_milestone", milestone))
    if not isinstance(declared, str) or declared.upper() != milestone:
        raise EvidencePackError(
            f"evidence at {pointer} declares milestone {declared!r}, expected {milestone}")
    return FoundEvidence(
        milestone, metric, passed, pointer,
        tuple(sorted(_amendment_values(row))), priority)


def _extract_report(payload: Mapping[str, object], *, milestone: str,
                    relative: str, verdict: bool) -> list[FoundEvidence]:
    declared = payload.get("rung", payload.get("milestone", milestone))
    if not isinstance(declared, str) or declared.upper() != milestone:
        raise EvidencePackError(
            f"evidence artifact {relative} does not declare rung {milestone}")
    priority = 1 if verdict else 0
    found: list[FoundEvidence] = []
    top_level = _gate_row(
        payload, default_metric=None, milestone=milestone,
        pointer=f"{relative}#", priority=priority)
    if top_level is not None:
        found.append(top_level)

    if "gates" in payload:
        gates = payload["gates"]
        if not isinstance(gates, list):
            raise EvidencePackError(f"malformed report {relative}: gates must be a list")
        for index, row in enumerate(gates):
            pointer = f"{relative}#{_json_pointer(('gates', index))}"
            item = _gate_row(row, default_metric=None, milestone=milestone,
                             pointer=pointer, priority=priority)
            if item is None:
                raise EvidencePackError(
                    f"malformed report {relative}: gates[{index}] needs "
                    "a string metric and bool passed")
            found.append(item)

    for container_name in ("results", "gate_results"):
        container = payload.get(container_name)
        if container is None:
            continue
        if not isinstance(container, Mapping):
            raise EvidencePackError(
                f"malformed report {relative}: {container_name} must be an object")
        for metric, row in container.items():
            pointer = f"{relative}#{_json_pointer((container_name, metric))}"
            item = _gate_row(row, default_metric=str(metric), milestone=milestone,
                             pointer=pointer, priority=priority)
            if item is None:
                raise EvidencePackError(
                    f"malformed report {relative}: {container_name}.{metric} "
                    "needs bool/object passed evidence")
            found.append(item)

    if verdict and top_level is None:
        ignored = {"schema", "rung", "milestone", "registered_milestone",
                   "generated_utc", "amendments", "metric", "passed"}
        for metric, row in payload.items():
            if metric in ignored:
                continue
            pointer = f"{relative}#{_json_pointer((metric,))}"
            item = _gate_row(row, default_metric=metric, milestone=milestone,
                             pointer=pointer, priority=priority)
            if item is None:
                raise EvidencePackError(
                    f"malformed verdict {relative}: {metric} has no bool passed value")
            found.append(item)
    return found


def _artifact_identity(path: Path, *, root: Path, role: str,
                       declared: Mapping[str, object] | None = None
                       ) -> dict[str, object]:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if not resolved.is_relative_to(root_resolved):
        raise EvidencePackError(f"artifact escapes evidence root: {path}")
    if not resolved.is_file():
        raise EvidencePackError(f"manifest artifact does not exist: {path}")
    actual_bytes = resolved.stat().st_size
    actual_sha = _sha256_file(resolved)
    if declared is not None:
        expected_bytes = _strict_int(declared.get("bytes"), f"artifact bytes for {path}")
        expected_sha = declared.get("sha256")
        if expected_bytes <= 0 or not isinstance(expected_sha, str) \
                or _HEX64_RE.fullmatch(expected_sha) is None:
            raise EvidencePackError(
                f"manifest artifact {path} needs positive bytes and lowercase SHA-256")
        if actual_bytes != expected_bytes or actual_sha != expected_sha:
            raise EvidencePackError(
                f"artifact byte/hash verification failed for {path}: "
                f"declared {expected_bytes}/{expected_sha}, actual {actual_bytes}/{actual_sha}")
    return {
        "relative_path": resolved.relative_to(root_resolved).as_posix(),
        "role": role,
        "bytes": actual_bytes,
        "sha256": actual_sha,
    }


def _manifest_pin(payload: Mapping[str, object], path: Path, milestone: str
                 ) -> dict[str, object]:
    if payload.get("schema") != 3 or payload.get("rung") != milestone:
        raise EvidencePackError(
            f"malformed rung manifest {path}: expected schema 3 and rung {milestone}")
    commit = payload.get("evaluator_commit")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise EvidencePackError(
            f"malformed rung manifest {path}: evaluator_commit must be a full Git hash")
    fingerprint = payload.get("experiment_fingerprint")
    config_digest = payload.get("config_digest", payload.get("configuration_digest"))
    if (not isinstance(fingerprint, str) or _HEX64_RE.fullmatch(fingerprint) is None) \
            and (not isinstance(config_digest, str)
                 or _HEX64_RE.fullmatch(config_digest) is None):
        raise EvidencePackError(
            f"malformed rung manifest {path}: requires a lowercase 64-hex "
            "experiment_fingerprint or config_digest")
    pin: dict[str, object] = {
        "rung": milestone,
        "evaluator_commit": commit,
        "experiment_fingerprint": fingerprint if isinstance(fingerprint, str) else None,
        "config_digest": config_digest if isinstance(config_digest, str) else None,
        "manifest_relative_path": path.relative_to(path.parents[1]).as_posix(),
    }
    if "config" in payload:
        pin["config"] = payload["config"]
    return pin


def _rung_inventory(rung_dir: Path) -> set[str]:
    if not rung_dir.is_dir():
        return set()
    return {
        path.relative_to(rung_dir).as_posix()
        for path in rung_dir.rglob("*")
        if (path.is_file() or path.is_symlink())
        and path != rung_dir / "manifest.json"
    }


def _load_manifest(root: Path, milestone: str
                  ) -> tuple[dict[str, object], list[dict[str, object]],
                             list[FoundEvidence], dict[str, object], list[str]]:
    rung_dir = root / milestone
    manifest_path = rung_dir / "manifest.json"
    payload = _load_json(manifest_path, "rung manifest")
    pin = _manifest_pin(payload, manifest_path, milestone)
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise EvidencePackError(
            f"malformed rung manifest {manifest_path}: artifacts must be a nonempty list")

    identities = [_artifact_identity(
        manifest_path, root=root, role="manifest")]
    evidence: list[FoundEvidence] = []
    seen: set[str] = set()
    for index, item in enumerate(artifacts):
        if not isinstance(item, Mapping):
            raise EvidencePackError(
                f"malformed rung manifest {manifest_path}: artifacts[{index}] is not an object")
        relative = item.get("relative_path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise EvidencePackError(
                f"malformed rung manifest {manifest_path}: artifacts[{index}] "
                "needs a portable relative_path")
        portable = Path(relative).as_posix()
        if portable == "manifest.json":
            raise EvidencePackError(
                f"malformed rung manifest {manifest_path}: the manifest must not "
                "list itself as an artifact")
        if portable in seen:
            raise EvidencePackError(
                f"malformed rung manifest {manifest_path}: duplicate artifact {portable}")
        seen.add(portable)
        target = (rung_dir / relative).resolve()
        if not target.is_relative_to(rung_dir.resolve()):
            raise EvidencePackError(
                f"manifest artifact escapes rung {milestone}: {relative}")
        if ".." in Path(relative).parts:
            raise EvidencePackError(
                f"malformed rung manifest {manifest_path}: artifacts[{index}] "
                "relative_path must name an exact rung path")
        role = ("report" if target.name.endswith("-report.json") else
                "verdict" if target.name.endswith("-verdicts.json") else "artifact")
        identities.append(_artifact_identity(
            target, root=root, role=role, declared=item))
        if role in {"report", "verdict"}:
            artifact_payload = _load_json(target, role)
            evidence.extend(_extract_report(
                artifact_payload, milestone=milestone,
                relative=target.relative_to(root).as_posix(),
                verdict=role == "verdict"))
    undeclared = sorted(_rung_inventory(rung_dir) - seen)
    return payload, identities, evidence, pin, undeclared


def _source_commit(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", str(path)],
            cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidencePackError(f"cannot resolve source revision for {path}: {exc}") from exc
    commit = result.stdout.strip()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise EvidencePackError(f"source {path} has no full Git revision")
    return commit


def _ledger_sources() -> list[dict[str, object]]:
    specs = (
        ("gate_ledger", REPOSITORY_ROOT / "gpuwm/verify/nest_gates.py"),
        ("architecture_doc", REPOSITORY_ROOT / nest_gates.ARCHITECTURE_DOC),
        ("plan_doc", REPOSITORY_ROOT / nest_gates.PLAN_DOC),
    )
    sources = []
    for role, path in specs:
        identity = _artifact_identity(path, root=REPOSITORY_ROOT, role=role)
        identity["source_commit"] = _source_commit(path.relative_to(REPOSITORY_ROOT))
        sources.append(identity)
    return sources


def _ledger_amendments(record) -> list[str]:
    text = " ".join((record.convention, record.anchor))
    return sorted({match.upper() for match in _AMENDMENT_RE.findall(text)})


def _escape_markdown(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown(pack: Mapping[str, object]) -> str:
    lines = [
        "# T18 close-out evidence pack",
        "",
        f"Generated: `{pack['generated_utc']}`",
        "",
        f"Mode: `{pack['mode']}`; close-out ready: `{pack['closeout_ready']}`",
        "",
        (f"Registered gates: {pack['counts']['registered']}; evidence found: "
         f"{pack['counts']['with_evidence']}; gaps: {pack['counts']['missing_evidence']}.") ,
        "",
        "## Rung inventory",
        "",
        f"Present: {', '.join(pack['present_rungs']) or 'none'}",
        "",
        f"Missing: {', '.join(pack['missing_rungs']) or 'none'}",
        "",
    ]
    if pack["blockers"]:
        lines.extend(("## Blockers", ""))
        lines.extend(f"- {_escape_markdown(item)}" for item in pack["blockers"])
        lines.append("")
    for milestone in nest_gates.MILESTONES:
        rows = [row for row in pack["gates"] if row["milestone"] == milestone]
        lines.extend((
            f"## {milestone}", "",
            "| Metric | Kind | Status | Evidence | Amendment refs |",
            "|---|---|---:|---|---|",
        ))
        for row in rows:
            status = ("MISSING" if not row["evidence_found"] else
                      "PASS" if row["passed"] else "FAIL")
            evidence = row["evidence_pointer"] or "--"
            amendments = ", ".join(row["amendment_references"]) or "--"
            lines.append(
                f"| `{_escape_markdown(row['metric'])}` | "
                f"`{_escape_markdown(row['kind'])}` | **{status}** | "
                f"`{_escape_markdown(evidence)}` | {_escape_markdown(amendments)} |")
        lines.append("")
    return "\n".join(lines)


def build_evidence_pack(rungs_root: str | Path, outdir: str | Path, *,
                        diagnostic_incomplete: bool = False) -> dict[str, object]:
    """Validate manifest-declared bytes and publish only an unambiguous pack."""
    root = Path(rungs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"rungs root does not exist: {root}")

    required = [item for item in nest_gates.MILESTONES
                if item not in _OPTIONAL_MILESTONES]
    present = [item for item in nest_gates.MILESTONES
               if (root / item / "manifest.json").is_file()]
    missing = [item for item in nest_gates.MILESTONES if item not in present]
    missing_required = [item for item in required if item not in present]
    blockers = ([] if not missing_required else [
        f"required rung manifests are missing: {', '.join(missing_required)}"])

    source_artifacts: list[dict[str, object]] = []
    evidence: list[FoundEvidence] = []
    manifest_provenance: list[dict[str, object]] = []
    undeclared: list[str] = []
    for milestone in present:
        _, identities, found, pin, rung_undeclared = _load_manifest(root, milestone)
        source_artifacts.extend(identities)
        evidence.extend(found)
        manifest_provenance.append(pin)
        undeclared.extend(f"{milestone}/{item}" for item in rung_undeclared)
    for milestone in missing:
        undeclared.extend(
            f"{milestone}/{item}" for item in sorted(_rung_inventory(root / milestone)))
    if undeclared:
        blockers.append("undeclared rung artifacts: " + ", ".join(sorted(undeclared)))

    ledger_keys = {(record.milestone, record.metric) for record in nest_gates.NEST_GATES}
    by_key: dict[tuple[str, str], list[FoundEvidence]] = {}
    unregistered = []
    for item in evidence:
        key = (item.milestone, item.metric)
        if key not in ledger_keys:
            unregistered.append(item.pointer)
            continue
        by_key.setdefault(key, []).append(item)
    if unregistered:
        raise EvidencePackError(
            "unregistered or ambiguous evidence pointers: " + ", ".join(sorted(unregistered)))

    rows: list[dict[str, object]] = []
    for record in nest_gates.NEST_GATES:
        matches = sorted(by_key.get((record.milestone, record.metric), []),
                         key=lambda item: (item.priority, item.pointer))
        passed_values = sorted({item.passed for item in matches})
        if len(passed_values) > 1:
            raise EvidencePackError(
                f"conflicting passed values for {record.milestone}/{record.metric}: "
                f"{passed_values}")
        primary = matches[0] if matches else None
        amendments = set(_ledger_amendments(record))
        for item in matches:
            amendments.update(item.amendment_references)
        blocking = record.kind != "diagnostic"
        closeout_required = blocking and record.milestone not in _OPTIONAL_MILESTONES
        if closeout_required and primary is None:
            blockers.append(f"missing blocking evidence: {record.milestone}/{record.metric}")
        elif closeout_required and primary is not None and not primary.passed:
            blockers.append(f"failed blocking evidence: {record.milestone}/{record.metric}")
        rows.append({
            "milestone": record.milestone,
            "metric": record.metric,
            "kind": record.kind,
            "threshold": record.threshold,
            "blocking": blocking,
            "closeout_required": closeout_required,
            "passed": primary.passed if primary is not None else None,
            "evidence_found": primary is not None,
            "evidence_pointer": primary.pointer if primary is not None else None,
            "evidence_pointers": [item.pointer for item in matches],
            "conflicting_passed_values": [],
            "amendment_references": sorted(amendments),
            "anchor": record.anchor,
            "convention": record.convention,
        })

    source_artifacts.sort(key=lambda item: (item["relative_path"], item["role"]))
    manifest_provenance.sort(key=lambda item: item["rung"])
    ledger_sources = _ledger_sources()
    inventory_hash = _stable_hash(source_artifacts)
    mode = "diagnostic-incomplete" if diagnostic_incomplete else "closeout"
    counts = {
        "registered": len(rows),
        "with_evidence": sum(bool(row["evidence_found"]) for row in rows),
        "missing_evidence": sum(not bool(row["evidence_found"]) for row in rows),
        "passed": sum(row["passed"] is True for row in rows),
        "failed": sum(row["passed"] is False for row in rows),
    }
    identity = {
        "mode": mode,
        "source_artifacts": source_artifacts,
        "manifest_provenance": manifest_provenance,
        "ledger_sources": ledger_sources,
        "gates": rows,
        "blockers": blockers,
    }
    pack: dict[str, object] = {
        "schema": 2,
        "generated_utc": _utc_now(),
        "mode": mode,
        "closeout_ready": not blockers,
        "rungs_root": {"relative_path": ".", "inventory_sha256": inventory_hash},
        "diagnostics": {"rungs_root_absolute": str(root.resolve())},
        "pack_fingerprint": _stable_hash(identity),
        "ledger": {
            "module": "gpuwm.verify.nest_gates",
            "sources": ledger_sources,
        },
        "source_artifacts": source_artifacts,
        "manifest_provenance": manifest_provenance,
        "present_rungs": present,
        "missing_rungs": missing,
        "missing_optional_rungs": [item for item in missing
                                   if item in _OPTIONAL_MILESTONES],
        "gates": rows,
        "blockers": blockers,
        "counts": counts,
    }

    if blockers and not diagnostic_incomplete:
        preview = "; ".join(blockers[:4])
        suffix = " ..." if len(blockers) > 4 else ""
        raise EvidencePackError(
            f"close-out evidence has {len(blockers)} blocker(s): {preview}{suffix}")

    output = Path(outdir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence-pack.json").write_text(
        json.dumps(pack, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (output / "evidence-pack.md").write_text(_markdown(pack), encoding="utf-8")
    return pack


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rungs-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--diagnostic-incomplete", action="store_true",
                        help="write an explicitly non-close-out incomplete-run diagnostic")
    args = parser.parse_args(argv)
    pack = build_evidence_pack(
        args.rungs_root, args.outdir,
        diagnostic_incomplete=args.diagnostic_incomplete)
    return 2 if pack["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

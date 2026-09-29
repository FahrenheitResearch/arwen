"""The history each publication job checks out, read from the workflow text.

A reused native binary is proved by ``git merge-base --is-ancestor`` between
the commit it was built at and the release commit
(``gpuwm.bridge_assets.native_input_difference``). A job that checks out at
the default ``fetch-depth`` of 1 holds only the release commit, so every reused
binary is refused there, and in ``publish.yml`` that happens after the tag and
the push to main. This module names every job that runs the stamp verifier and
the depth it checks out, so the test suite refuses a shallow one and the packet
preflight runs the verifier once from a clone of exactly that depth.

Standard library only: the packet preflight loads it from the public checkout
at the release commit, inside a runtime that carries no YAML parser. It ships
in the public tree because the test that holds the workflow to it does. ``tests/test_native_build_inputs.py`` checks that this
reading agrees with a YAML parse of the same file.
"""
from __future__ import annotations

import re
from pathlib import Path

#: Command text that reaches ``native_input_difference`` against the checkout.
VERIFIER_CALLS = (
    'tools/verify_release_artifacts',
    'tools.verify_release_artifacts',
    'build_bridge_bundle.py pin',
    'build_bridge_bundle pin',
    'promote_prepared_release.py verify',
    'promote_prepared_release.py smoke',
)

_JOB = re.compile(r'^  ([A-Za-z0-9_-]+):\s*$')
_DEPTH = re.compile(r'^\s+fetch-depth:\s*([0-9]+)\s*(?:#.*)?$')


def verifier_jobs(text: str) -> dict[str, int | None]:
    """{job: checkout depth} for every job whose commands run the verifier.

    The depth is 0 for full history, the declared number otherwise, 1 when a
    checkout declares none (the action's default), and None when the job runs
    the verifier with no checkout at all.
    """
    jobs: dict[str, list[str]] = {}
    current = None
    inside = False
    for line in text.splitlines():
        if line.startswith('jobs:'):
            inside = True
            continue
        if inside and line and not line.startswith((' ', '#')):
            inside = False
        if not inside:
            continue
        match = _JOB.match(line)
        if match:
            current = match.group(1)
            jobs[current] = []
        elif current is not None:
            jobs[current].append(line)
    found: dict[str, int | None] = {}
    for job, lines in jobs.items():
        body = '\n'.join(lines)
        if not any(call in body for call in VERIFIER_CALLS):
            continue
        if 'actions/checkout@' not in body:
            found[job] = None
            continue
        depths = [int(m.group(1)) for m in map(_DEPTH.match, lines) if m]
        found[job] = depths[0] if depths else 1
    return found


def shallowest(path: Path) -> tuple[str, int | None] | None:
    """The verifier job with the least history, or None when no job runs it."""
    jobs = verifier_jobs(Path(path).read_text(encoding='utf-8'))
    if not jobs:
        return None
    def rank(item):
        depth = item[1]
        return (0 if depth is None else 1, float('inf') if depth == 0 else depth)
    return min(jobs.items(), key=rank)

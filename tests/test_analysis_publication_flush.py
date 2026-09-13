"""Native file durability at the analysis decision boundary, without a GPU."""
import errno
import os

import pytest

from gpuwm.ensemble import analysis_commit


def _staged_decision(tmp_path):
    root = tmp_path / "cycle_001"
    pairs = []
    for member in range(2):
        directory = root / f"member_{member:03d}"
        directory.mkdir(parents=True)
        analysis = directory / "analysis.npz"
        staged = directory / "analysis.npz.staged"
        staged.write_bytes(b"staged analysis bytes\x00" + bytes([member]))
        pairs.append((member, staged, analysis))
    context = dict(leg_root=str(root.resolve()), members=[0, 1], cycle=1,
                   assets=[], positivity="none", moment_policy="mass-only",
                   moment_repair=False, method={"callable": "test.analysis"})
    receipt = dict(member_count=2, cycle=1, status="APPLIED",
                   positivity_policy="none", moment_policy="mass-only",
                   moment_repair=False, method={"callable": "test.analysis"},
                   receipts=[{"member": member} for member in range(2)])
    arguments = dict(context=context, receipt=receipt, pairs=pairs,
                     staged_suffix=".staged", analysis_name="analysis.npz")
    return root, arguments


def test_native_staged_flush_publishes_recoverable_decision(tmp_path):
    # This calls the real OS fsync. Windows CRT rejects the former read-only
    # staged descriptor with EBADF before any intent can be published.
    root, arguments = _staged_decision(tmp_path)
    before = {staged: staged.read_bytes() for _, staged, _ in arguments["pairs"]}
    intent = analysis_commit.begin(root, **arguments)
    assert analysis_commit.read_record(root / analysis_commit.INTENT_NAME) == intent
    assert not (root / analysis_commit.COMMIT_NAME).exists()
    for _, staged, analysis in arguments["pairs"]:
        assert staged.read_bytes() == before[staged]
        assert not analysis.exists()

    def publish(staged, analysis):
        assert (root / analysis_commit.INTENT_NAME).exists()
        staged.replace(analysis)

    recovered = analysis_commit.recover(root, publish=publish,
                                        context=arguments["context"])
    assert recovered == arguments["receipt"]
    assert analysis_commit.read_record(root / analysis_commit.COMMIT_NAME)["receipt"] == recovered
    for _, staged, analysis in arguments["pairs"]:
        assert not staged.exists()
        assert analysis.read_bytes() == before[staged]


def test_persistent_flush_failure_preserves_bytes_without_a_decision(tmp_path, monkeypatch):
    root, arguments = _staged_decision(tmp_path)
    prior = tmp_path / "prior-analysis.npz"
    prior.write_bytes(b"previously completed analysis")
    before = {staged: staged.read_bytes() for _, staged, _ in arguments["pairs"]}
    before[prior] = prior.read_bytes()
    attempts = []

    def failed_flush(descriptor):
        attempts.append(descriptor)
        raise OSError(errno.EIO, "persistent staged flush failure")

    monkeypatch.setattr(os, "fsync", failed_flush)
    for _ in range(2):
        with pytest.raises(OSError, match="persistent staged flush failure") as failure:
            analysis_commit.begin(root, **arguments)
        assert failure.value.errno == errno.EIO
        assert not (root / analysis_commit.INTENT_NAME).exists()
        assert not (root / analysis_commit.COMMIT_NAME).exists()
        assert all(not analysis.exists() for _, _, analysis in arguments["pairs"])
        assert all(path.read_bytes() == content for path, content in before.items())
    assert len(attempts) == 2

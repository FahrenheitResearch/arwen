"""Contradictory picture/output requests fail before any preparation."""

import pytest

from gpuwm import capabilities, prepared_domain_tree_forecast as runner


class ReachedPreparation(Exception):
    pass


@pytest.mark.parametrize("io_mode,products,refused", [
    ("none", "t2", True), ("none", "all", True),
    ("none", "none", False), ("history", "t2", False),
])
def test_picture_request_needs_history(tmp_path, monkeypatch, capsys,
                                      io_mode, products, refused):
    monkeypatch.setattr(capabilities, "require", lambda *a, **kw: None)
    def preparation(**kwargs):
        raise ReachedPreparation
    monkeypatch.setattr(runner, "preflight_prepared_tree", preparation)
    out = tmp_path / "out"
    argv = ["--prepared-root", str(tmp_path / "prepared"),
            "--preparation-receipt-sha256", "0" * 64,
            "--experiment-config", str(tmp_path / "config.toml"),
            "--experiment-config-sha256", "0" * 64,
            "--outdir", str(out), "--io-mode", io_mode,
            "--render-products", products]
    if refused:
        assert runner.main(argv) == 2
        assert not out.exists()
        error = capsys.readouterr().err
        assert "--io-mode history" in error
        assert "--render-products none" in error
        assert "Traceback" not in error
    else:
        with pytest.raises(ReachedPreparation):
            runner.main(argv)

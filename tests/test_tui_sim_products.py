"""Public prepared-run plot selection reaches both existing runner arms."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from gpuwm import stage_cli
from gpuwm.cli import build_parser


def bundle(tmp_path, layout):
    from gpuwm import prepared_single_domain_forecast as single
    root = tmp_path / "prepared"
    root.mkdir()
    schema = (single._PROOF_SCHEMA if layout == "single" else
              single._HIERARCHY_PROOF_SCHEMA)["gfs"]
    (root / "proof.json").write_text(json.dumps({
        "schema": schema, "status": "READY", "domain_count": 1 if layout == "single" else 2,
        "input_manifest_sha256": "11" * 32,
        "prepared_cache": {"content_sha256": "22" * 32},
    }))
    config = tmp_path / "e.toml"
    config.write_text("# only a command is being composed\n")
    wps = tmp_path / "namelist.wps"
    wps.write_text("&share\n/\n")
    return root, config, wps


@pytest.mark.parametrize("layout", ["single", "tree"])
@pytest.mark.parametrize("products", [None, "none", "all", "var:SNOWH,total_qpf"])
def test_public_sim_prints_exact_supported_runner_plot_arguments(
        tmp_path, capsys, layout, products):
    from gpuwm import prepared_single_domain_forecast as single
    from gpuwm import prepared_domain_tree_forecast as tree
    root, config, wps = bundle(tmp_path, layout)
    words = ["sim", str(root), "--experiment-config", str(config),
             "--wps-namelist", str(wps), "--outdir", str(tmp_path / "run"),
             "--print-command"]
    if products is not None:
        words += ["--render-products", products,
                  "--render-dir", str(tmp_path / "my pictures")]
    args = build_parser().parse_args(words)
    assert stage_cli.sim_main(args) == 0
    command = shlex.split(capsys.readouterr().out.strip())
    parser = single.build_parser() if layout == "single" else tree.build_parser()
    parsed = parser.parse_args(command[3:])
    assert parsed.render_products == products
    assert parsed.render_dir == (None if products is None else tmp_path / "my pictures")
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "my pictures").exists()


def test_tree_main_arms_only_the_requested_first_frame_and_keeps_host_ownership(
        tmp_path, monkeypatch, capsys):
    from gpuwm import capabilities, provenance_gate
    from gpuwm import prepared_domain_tree_forecast as tree
    from gpuwm.first_products import FirstProducts

    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda *a: None)
    monkeypatch.setattr(tree, "experimental_selection_sentence", lambda *a: None)
    monkeypatch.setattr(tree, "preflight_prepared_tree", lambda **kwargs:
                        SimpleNamespace(experiment=SimpleNamespace(domains=[])))
    captures = []

    def run(inputs, **kwargs):
        captures.append(kwargs)
        return {"status": "PASS", "readiness": "fixture", "wall_seconds": 0,
                "execution_plan": {"plan_id": "fixture", "domain_count": 2},
                "output": {"frame_count": 0}}

    monkeypatch.setattr(tree, "run_prepared_tree", run)
    for index, (products, hosted) in enumerate([(None, False), ("none", False),
                                               ("total_qpf", False), ("total_qpf", True)]):
        argv = ["--prepared-root", str(tmp_path / "prep"),
                "--preparation-receipt-sha256", "a" * 64,
                "--experiment-config", str(tmp_path / "e.toml"),
                "--experiment-config-sha256", "b" * 64,
                "--outdir", str(tmp_path / f"run{index}")]
        if products is not None:
            argv += ["--render-products", products]
        observer = SimpleNamespace(first_products=object()) if hosted else None
        assert tree.main(argv, observer=observer) == 0
        captured = captures[-1]
        assert captured["observer"] is observer
        armed = captured.get("first_products")
        if products == "total_qpf" and not hosted:
            assert isinstance(armed, FirstProducts)
            assert armed.render_products == products
            assert armed.render_dir == tmp_path / f"run{index}" / "png"
            assert not armed.dispatched
        else:
            assert armed is None
    assert "already armed" in capsys.readouterr().err


def test_sim_dispatch_forwards_selection_in_process_without_starting_a_plotter(
        tmp_path, monkeypatch, capsys):
    from gpuwm import prepared_domain_tree_forecast as tree
    root, config, wps = bundle(tmp_path, "tree")
    calls = []
    monkeypatch.setattr(tree, "main", lambda argv: calls.append(argv) or 0)
    args = build_parser().parse_args([
        "sim", str(root), "--experiment-config", str(config),
        "--outdir", str(tmp_path / "out"), "--render-products", "none"])
    assert stage_cli.sim_main(args) == 0
    assert tree.build_parser().parse_args(calls[0]).render_products == "none"
    assert "no fetch and no render" in capsys.readouterr().out


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize("join_fails", [False, True])
def test_tree_failure_joins_dispatched_plot_thread_and_preserves_primary_error(
        tmp_path, monkeypatch, capsys, failure_type, join_fails):
    import threading
    from gpuwm import capabilities, provenance_gate
    from gpuwm import prepared_domain_tree_forecast as tree
    from gpuwm.first_products import FirstProducts

    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.setattr(provenance_gate, "announce_for_main", lambda *a: None)
    monkeypatch.setattr(tree, "experimental_selection_sentence", lambda *a: None)
    monkeypatch.setattr(tree, "preflight_prepared_tree", lambda **kwargs:
                        SimpleNamespace(experiment=SimpleNamespace(domains=[])))
    worker_started, allow_finish, worker_finished = (
        threading.Event(), threading.Event(), threading.Event())
    plots = FirstProducts({"run": tmp_path, "render": tmp_path / "png",
                           "render_products": "total_qpf"},
                          report=lambda *a: None, warn=lambda *a, **k: None)

    def render(**kwargs):
        worker_started.set()
        if allow_finish.wait(5):
            worker_finished.set()

    monkeypatch.setattr(plots, "_render", render)
    original_wait = plots.wait
    joins = []

    def join():
        joins.append(True)
        assert worker_started.is_set()
        allow_finish.set()
        original_wait(timeout=2)
        assert worker_finished.is_set()
        assert not plots._thread.is_alive()
        if join_fails:
            raise RuntimeError("plot join fixture")

    monkeypatch.setattr(plots, "wait", join)
    monkeypatch.setattr(tree.prepared_single, "_route_owned_first_products",
                        lambda *a, **k: plots)
    primary = failure_type("forecast failed after first frame")
    receipts = []
    monkeypatch.setattr(tree, "_write_failed_run_receipt",
                        lambda outdir, error: receipts.append(error))

    def run(inputs, **kwargs):
        assert kwargs["first_products"] is plots
        assert plots.frame_committed(domain=1, valid_time="fixture",
                                     path=tmp_path / "first.nc")
        assert worker_started.wait(2)
        raise primary

    monkeypatch.setattr(tree, "run_prepared_tree", run)
    argv = ["--prepared-root", str(tmp_path / "prep"),
            "--preparation-receipt-sha256", "a" * 64,
            "--experiment-config", str(tmp_path / "e.toml"),
            "--experiment-config-sha256", "b" * 64,
            "--outdir", str(tmp_path / "run"),
            "--render-products", "total_qpf"]
    with pytest.raises(failure_type) as raised:
        tree.main(argv)
    assert raised.value is primary
    assert receipts == [primary]
    assert joins == [True] and worker_finished.is_set()
    assert ("plot join fixture" in capsys.readouterr().err) is join_fails

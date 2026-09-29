"""A section requested by a GUI draft reaches the native render command."""

import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from gpuwm import go_cli, machine_agent, runplan
from gpuwm.gui.api import ApiError, CreateMixin
from gpuwm.gui.machines import MachineError
from gpuwm.gui.remote_runs import request_render


LINE = "35,-100,36,-99"
PRODUCTS = "xsec:QCLOUD=0.01,0.1/wa"


class Drafts(CreateMixin):
    def _offered(self):
        return {"sources": [{"id": "gfs", "route": "prepared"}]}

    def availability_of(self, *args, **kwargs):
        return {"starts": "yes"}


def payload(**extra):
    return {"name": "section-run", "source": "gfs", "cycle": "2026-09-24T00",
            "card": "8gb", "lat": 35, "lon": -100, "products": PRODUCTS, **extra}


@pytest.mark.parametrize("following", [False, True])
def test_gui_section_draft_builds_a_plan_and_render_command(tmp_path, following):
    draft = Drafts().draft(payload(render_section=LINE), need_name=True)
    draft["following"] = following
    config = tmp_path / "cyclone.toml"
    config.write_text("[experiment]\n", encoding="utf-8")
    document = CreateMixin.plan_document(draft, tmp_path)
    assert document["run_options"]["render_section"] == LINE
    if following:
        document["config"]["path"] = str(config)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(document), encoding="utf-8")
    plan = runplan.load_plan(plan_file)
    render = runplan._chain_render_plan(plan, forecast_dir=tmp_path / "run", run_dir=tmp_path)
    command = go_cli.render_command(render, [tmp_path / "wrfout_d01_frame"])
    assert command[command.index("--products") + 1] == PRODUCTS
    assert "--section=" + LINE in command


@pytest.mark.parametrize("line", [None, "", "35,-100,35,-100", "91,0,35,1", "line.json"])
def test_gui_refuses_a_section_that_cannot_be_drawn(line):
    with pytest.raises(ApiError) as raised:
        Drafts().draft(payload(render_section=line), need_name=True)
    assert "section" in str(raised.value).lower()
    assert "draw" in str(raised.value).lower() or "render" in str(raised.value).lower()


def test_gui_plan_builder_does_not_make_an_unlined_section(tmp_path):
    draft = Drafts().draft(payload(products="all"), need_name=True)
    draft["products"] = PRODUCTS
    with pytest.raises(ApiError, match="line"):
        CreateMixin.plan_document(draft, tmp_path)


def test_gui_machine_render_request_keeps_the_line(tmp_path):
    request = request_render(tmp_path, "node", products=PRODUCTS, render_section=LINE)
    assert request["render_section"] == LINE
    assert request_render(tmp_path, "node", products=PRODUCTS, render_section=LINE) == request
    with pytest.raises(MachineError, match="old slice"):
        request_render(tmp_path, "node", products=PRODUCTS, render_section="36,-100,37,-99")


def test_gui_machine_render_worker_passes_the_line(tmp_path, monkeypatch):
    args = SimpleNamespace(workspace=str(tmp_path), job="section-job", python="python")
    monkeypatch.setattr(machine_agent, "spawn", lambda *args, **kwargs: 0)
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(
        {"run": "section-run", "products": PRODUCTS, "render_section": LINE})))
    machine_agent.cmd_render_start(args)
    frame = tmp_path / "wrfout_d01_frame"
    frame.write_bytes(b"frame")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps([str(frame)])))
    machine_agent.cmd_render_feed(args)
    commands = []

    def draw(argv, **kwargs):
        commands.append(argv)
        machine_agent.cmd_render_end(args)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(machine_agent.subprocess, "run", draw)
    assert machine_agent.cmd_render_loop(args) == 0
    assert len(commands) == 1
    assert commands[0][commands[0].index("--products") + 1] == PRODUCTS
    assert "--section=" + LINE in commands[0]


@pytest.mark.parametrize("empty", [None, "", "  "])
def test_gui_refuses_an_empty_named_list_instead_of_drawing_the_standard_set(empty):
    with pytest.raises(ApiError) as raised:
        Drafts().draft(payload(products=empty, products_named=True), need_name=True)
    message = str(raised.value)
    assert "no picture is named" in message and "standard set" in message
    # The standard set chosen as itself is not refused.
    assert Drafts().draft(payload(products=empty), need_name=True)["products"] is None

"""A pinned download host lives in the experiment's ``[fetch]`` table.

``gpuwm fetch --transport`` pins one host of a source's endpoint ladder,
and the experiment TOML refused the same key as unknown ("unknown key(s)
['transport'] in [fetch]"), so a run that needed a pinned host had to be
fetched, prepared and launched by hand.  The table now takes it with the
values and refusals the flag has, ``gpuwm go`` and ``gpuwm run-plan``
carry it to the fetch stage, and ``gpuwm go --transport`` wins over it,
with the plan saying which one it used.
"""

from __future__ import annotations

import re
import shutil
import tomllib
from pathlib import Path

import pytest

from gpuwm import fetch, go_cli, runplan
from gpuwm.cli import main as cli_main
from gpuwm.experiment import load_experiment

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEMO = "hrrr_native_quick_demo"
OUT_LINE = f'out = "data/{DEMO}"'


def _hrrr_config(tmp_path: Path, transport: str | None = None) -> Path:
    """The shipped HRRR demo and its companions, with ``transport`` set."""

    for companion in CONFIGS.glob(f"{DEMO}.*"):
        shutil.copy(companion, tmp_path / companion.name)
    config = tmp_path / f"{DEMO}.toml"
    if transport is not None:
        text = config.read_text(encoding="utf-8")
        assert text.count(OUT_LINE) == 1
        config.write_text(text.replace(
            OUT_LINE, f'{OUT_LINE}\ntransport = "{transport}"'), encoding="utf-8")
    return config


@pytest.mark.parametrize("source, transport", [
    ("hrrr", "s3"), ("hrrr", "nomads"), ("hrrr", "auto"),
    ("rap", "aws"), ("rap", "nomads")])
def test_the_table_takes_the_hosts_the_flag_takes(source, transport):
    fetch.validate_fetch_hints(
        {"source": source, "cycle": "2026-08-20T00", "hours": 1,
         "transport": transport}, source="case.toml")


def test_an_experiment_naming_a_host_loads(tmp_path):
    # Refused on the head: unknown key(s) ['transport'] in [fetch].
    load_experiment(_hrrr_config(tmp_path, "s3"))


@pytest.mark.parametrize("table, words", [
    ({"source": "hrrr", "transport": "ftp"},
     "transport = 'ftp' is not a host `gpuwm fetch --transport` takes"),
    ({"source": "hrrr", "transport": "aws"}, "unknown HRRR transport 'aws'"),
    ({"source": "rap", "transport": "s3"},
     "--transport s3: --source rap publishes on nomads, aws"),
    ({"source": "era5", "transport": "s3"}, "era5 has no host to choose between"),
    ({"source": "gfs", "transport": "s3"}, "--mode full-file"),
    ({"source": "hrrr", "transport": 3}, "transport in [fetch] of case.toml must be"),
])
def test_a_host_the_fetch_would_refuse_is_refused_by_name(table, words):
    with pytest.raises(ValueError, match=re.escape(words)) as caught:
        fetch.validate_fetch_hints(table, source="case.toml")
    assert "[fetch] of case.toml" in str(caught.value)


def test_go_carries_the_table_host_and_says_so(tmp_path, capsys):
    config = _hrrr_config(tmp_path, "s3")
    assert cli_main(["go", str(config), "--dry-run",
                     "--outdir", str(tmp_path / "runs")]) == 0
    printed = capsys.readouterr().out
    assert "go: the fetch pins host s3, from [fetch] transport" in printed


def test_the_go_flag_wins_over_the_table_and_the_plan_says_so(tmp_path, capsys):
    config = _hrrr_config(tmp_path, "s3")
    assert cli_main(["go", str(config), "--dry-run", "--transport", "nomads",
                     "--outdir", str(tmp_path / "runs")]) == 0
    printed = capsys.readouterr().out
    assert ("go: the fetch pins host nomads, from --transport "
            "(over [fetch] transport = 's3')") in printed
    run_line = next(line for line in printed.splitlines() if line.startswith("Run: "))
    assert "--transport nomads" in run_line


def test_a_go_flag_the_source_cannot_pin_is_refused_by_name(tmp_path, capsys):
    config = _hrrr_config(tmp_path)
    assert cli_main(["go", str(config), "--dry-run", "--transport", "aws",
                     "--outdir", str(tmp_path / "runs")]) == 2
    assert "--transport aws: unknown HRRR transport 'aws'" in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()


def _plan(tmp_path: Path, config: Path, **options) -> runplan.RunPlan:
    raw = {"schema": runplan.PLAN_SCHEMA, "name": "pinned", "route": "prepared",
           "config": {"path": str(config)}, "output_root": str(tmp_path / "out"),
           "run_options": options}
    return runplan.build_plan(raw, source="plan.json", base_dir=tmp_path,
                              sha256="0" * 64)


@pytest.mark.parametrize("option, host, basis", [
    (None, "s3", "[fetch] transport"),
    ("nomads", "nomads", "--transport")])
def test_run_plan_fetch_stage_asks_the_host_its_plan_records(
        tmp_path, option, host, basis):
    config = _hrrr_config(tmp_path, "s3")
    plan = _plan(tmp_path, config, **({} if option is None else {"transport": option}))
    resolution, _exp, _data = runplan.resolve_plan(plan, require_inputs=False)
    recorded = [row for row in resolution["automatic_resolutions"]
                if row.get("scope") == "fetch" and row.get("key") == "transport"]
    assert [(row["value"], row["basis"]) for row in recorded] == [(host, basis)]
    hints = tomllib.loads(config.read_text(encoding="utf-8"))["fetch"]
    arguments = runplan._fetch_arguments_from_hints(
        runplan._pinned_fetch_hints(plan, hints), out=tmp_path / "data")
    assert arguments[arguments.index("--transport") + 1] == host
    runplan._validate_fetch_arguments(arguments)


def test_run_plan_refuses_a_transport_option_that_is_no_host(tmp_path):
    config = _hrrr_config(tmp_path)
    with pytest.raises(runplan.PlanError, match=re.escape(
            "run_options.transport' = 'ftp' is not a host")):
        _plan(tmp_path, config, transport="ftp")


def _dry_run_downloads(monkeypatch, capsys, tmp_path: Path, config: Path,
                       *flags: str) -> tuple[set[Path], str]:
    """The download folders a ``gpuwm go --dry-run`` picks, and what it said."""

    picked: set[Path] = set()
    chosen = go_cli.managed_download_dir

    def spy(case_root, request):
        folder = chosen(case_root, request)
        picked.add(folder)
        return folder

    monkeypatch.setattr(go_cli, "managed_download_dir", spy)
    try:
        assert cli_main(["go", str(config), "--dry-run", *flags,
                         "--outdir", str(tmp_path / "runs")]) == 0
    finally:
        monkeypatch.setattr(go_cli, "managed_download_dir", chosen)
    return picked, capsys.readouterr().out


@pytest.mark.parametrize("table, flag", [("auto", None), ("s3", "auto")])
def test_go_reads_auto_as_no_pin_and_reuses_the_unpinned_download(
        tmp_path, monkeypatch, capsys, table, flag):
    """``auto`` is the unpinned default written out, in the table or the flag.

    It was read as a pinned host named "auto": the dry run said "the
    fetch pins host auto", and the managed download folder was keyed on
    it, so a config that spelled out the default fetched the whole cycle
    again into a second folder beside the one the same config without
    the key had already filled.  ``--transport auto`` over a table that
    pins a host unpins it: the flag wins, and it names no host.
    """

    unpinned = tmp_path / "unpinned"
    spelled = tmp_path / "spelled"
    unpinned.mkdir()
    spelled.mkdir()
    bare, _said = _dry_run_downloads(monkeypatch, capsys, tmp_path,
                                     _hrrr_config(unpinned))
    picked, said = _dry_run_downloads(
        monkeypatch, capsys, tmp_path, _hrrr_config(spelled, table),
        *(() if flag is None else ("--transport", flag)))
    assert len(bare) == 1
    assert picked == bare
    assert "pins host" not in said


def test_go_says_its_auto_flag_unpins_the_table_host(tmp_path, capsys):
    """``--transport auto`` over a table naming a host is named by the plan.

    The run asks hosts its config says it does not, and the plan printed
    nothing about it, although the plan names which setting it used for
    every other outcome.
    """

    config = _hrrr_config(tmp_path, "s3")
    assert cli_main(["go", str(config), "--dry-run", "--transport", "auto",
                     "--outdir", str(tmp_path / "runs")]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert [line for line in printed if line.startswith("go: the fetch ")] == [
        "go: the fetch walks the host ladder, from --transport auto, "
        "over [fetch] transport = 's3'"]


@pytest.mark.parametrize("table", [None, "auto"])
def test_go_says_nothing_when_auto_overrides_no_host(tmp_path, capsys, table):
    config = _hrrr_config(tmp_path, table)
    assert cli_main(["go", str(config), "--dry-run", "--transport", "auto",
                     "--outdir", str(tmp_path / "runs")]) == 0
    assert "go: the fetch " not in capsys.readouterr().out


def _case_data_config(tmp_path: Path, domains: int = 1) -> Path:
    from test_go_declared_inputs import case

    return case(tmp_path, domains=domains)


def test_go_takes_auto_on_a_run_with_no_fetch_stage(tmp_path, capsys):
    """``auto`` pins no host, so a route that fetches nothing takes it.

    The refusal is for a host nothing would ask; it read ``auto`` as one
    and turned away the default spelled out on every [case_data] run.
    """

    config = _case_data_config(tmp_path)
    argv = ["go", str(config), "--dry-run", "--products", "none",
            "--outdir", str(tmp_path / "runs")]
    assert cli_main([*argv, "--transport", "auto"]) == 0, capsys.readouterr().err
    printed = capsys.readouterr().out
    assert "go: the fetch " not in printed
    assert cli_main([*argv, "--transport", "s3"]) == 2
    assert "--transport s3 pins the host of a download" in capsys.readouterr().err


def test_go_and_run_plan_take_auto_beside_an_existing_bundle(tmp_path, capsys):
    from test_stage_seams import _tree_bundle

    config = _case_data_config(tmp_path, domains=2)
    prepared = _tree_bundle(tmp_path / "prepared", domains=2)
    assert cli_main(["go", str(config), "--prepared-root", str(prepared),
                     "--transport", "auto", "--products", "none",
                     "--outdir", str(tmp_path / "new"), "--dry-run"]) == 0, (
        capsys.readouterr().err)

    def plan(transport):
        return runplan.build_plan({
            "schema": runplan.PLAN_SCHEMA, "name": "bundle", "route": "prepared",
            "config": {"path": str(config)}, "output_root": str(tmp_path / transport),
            "run_options": {"render_products": "none", "prepared_root": str(prepared),
                            "transport": transport}},
            source="plan.json", base_dir=tmp_path, sha256="0" * 64)

    assert plan("auto").run_options["transport"] == "auto"
    with pytest.raises(runplan.PlanError, match="would pin a host nothing asks"):
        plan("s3")


@pytest.mark.parametrize("table, option", [("auto", None), ("s3", "auto")])
def test_run_plan_records_no_pin_for_auto_and_asks_no_host(
        tmp_path, table, option):
    config = _hrrr_config(tmp_path, table)
    plan = _plan(tmp_path, config, **({} if option is None else {"transport": option}))
    resolution, _exp, _data = runplan.resolve_plan(plan, require_inputs=False)
    assert not [row for row in resolution["automatic_resolutions"]
                if row.get("scope") == "fetch" and row.get("key") == "transport"]
    hints = tomllib.loads(config.read_text(encoding="utf-8"))["fetch"]
    pinned = runplan._pinned_fetch_hints(plan, hints)
    assert "transport" not in pinned
    bare = dict(hints)
    bare.pop("transport")
    assert go_cli.managed_download_key(pinned) == go_cli.managed_download_key(bare)
    if table == "auto":
        # The key itself reads the spelled-out default as no pin, so a
        # caller that hands it the table unpinned keys the same folder.
        assert go_cli.managed_download_key(hints) == go_cli.managed_download_key(bare)
    arguments = runplan._fetch_arguments_from_hints(pinned, out=tmp_path / "data")
    assert "--transport" not in arguments
    runplan._validate_fetch_arguments(arguments)


def test_the_go_fetch_command_spells_the_pinned_host(tmp_path):
    command = go_cli.fetch_command({
        "source": "gfs", "cycle": "2026-08-20T00", "hours": 6,
        "area": "25,-105,45,-85", "data": tmp_path, "transport": "s3"})
    assert command[command.index("--transport") + 1] == "s3"

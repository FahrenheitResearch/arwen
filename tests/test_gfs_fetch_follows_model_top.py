"""A GFS run's download reaches its own model top, with no flag typed.

The breakage these prevent: ``gpuwm go`` composed its fetch from the
config's ``[fetch]`` table alone, so a GFS config whose ladder tops out
at 50 hPa (WRF's own default top, and the 59-level storm-following
ladders) downloaded the certified 1000..100 hPa ladder, and preparation
then refused it with "source atmosphere stops at 10000 Pa but requested
p_top is 5000 Pa".  Only a hand-typed ``gpuwm fetch --p-top-pa 5000``
followed by ``gpuwm go --data-dir`` got a run through.  The same gap sat
under ``gpuwm run-plan`` and the desktop, which drive the same chain,
and under the download price, which counted the records of a request the
run did not make.

Every level question here is answered by the real inventory captured in
``tests/fixtures/gfs-inventory/``; no live request is issued.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest

import gpuwm.fetch as fetch
from gpuwm import domain_wizard, download_budget, go_cli
from gpuwm.cli import build_parser, main as cli_main
from tools import download_gfs_native_subset as gfs_transport

_INDEX = (Path(__file__).parent / "fixtures" / "gfs-inventory"
          / "gfs.t12z.pgrb2.0p25.f000.idx")
_BUDGET = Path(__file__).parent / "fixtures" / "download_budget" / "gfs-3km.toml"


def _grib2(messages: int) -> bytes:
    one = (b"GRIB" + b"\x00\x00" + b"\x00" + b"\x02"
           + (20).to_bytes(8, "big") + b"7777")
    return one * messages


def _levels_asked(url: str) -> set[float]:
    query = dict(parse_qsl(urlsplit(url).query, keep_blank_values=True))
    return {float(key[4:-3]) for key in query
            if key.startswith("lev_") and key.endswith("_mb")}


@pytest.fixture
def offline_gfs(monkeypatch):
    """The real fetch front door over the captured inventory.

    The index read answers from the fixture, and each NOMADS crop is a
    stand-in GRIB stream carrying exactly the records its URL selects, so
    the fetch's own record bar is what checks the ladder.  The cycle is
    held inside the crop host's window, and the whole-object transport
    refuses outright: a cycle judged too old for the crop host would
    otherwise download half a gigabyte per hour.  Returns the list the
    requested URLs land in.
    """

    urls: list[str] = []
    index = _INDEX.read_text(encoding="ascii")

    def download(url, destination, **_kw):
        urls.append(url)
        destination.write_bytes(_grib2(
            gfs_transport.record_count_for_levels(len(_levels_asked(url)))))

    def no_whole_objects(*_a, **_k):
        raise AssertionError("a test reached the whole-object transport")

    monkeypatch.setattr(fetch, "gfs_live_index", lambda *a, **k: index)
    monkeypatch.setattr(fetch, "require_published_cycle",
                        lambda *a, **k: None)
    monkeypatch.setattr(fetch, "archive_only_cycle", lambda *a, **k: False)
    monkeypatch.setattr(fetch, "fetch_gfs_fullfile", no_whole_objects)
    monkeypatch.setattr(gfs_transport, "_download", download)
    return urls


def _emit(tmp_path: Path, name: str = "default", *extra: str) -> Path:
    out = tmp_path / f"{name}.toml"
    assert cli_main(["domain", "--point=35.3,-97.5", "--card", "24gb",
                     "--source", "gfs", "--cycle", "2026-07-29T18",
                     "--hours", "6", "--out", str(out), *extra]) == 0
    return out


def _with_top(config: Path, p_top: float, name: str) -> Path:
    text = config.read_text(encoding="utf-8")
    shared = text.index("[shared]")
    start = text.index("\np_top = ", shared) + 1
    end = text.index("\n", start)
    edited = config.with_name(f"{name}.toml")
    edited.write_text(text[:start] + f"p_top = {p_top!r}" + text[end:],
                      encoding="utf-8")
    shutil.copyfile(config.with_suffix(".namelist.wps"),
                    edited.with_suffix(".namelist.wps"))
    return edited


def test_a_default_gfs_plan_fetches_the_70_and_50_hpa_levels(
        tmp_path, offline_gfs, capsys):
    """A bare wizard config through `gpuwm go`'s own fetch command."""

    config = _emit(tmp_path)
    capsys.readouterr()
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "go")
    command = go_cli.fetch_command(plan)
    assert command[3] == "fetch"

    assert cli_main(command[3:]) == 0

    assert offline_gfs, "the fetch asked NOMADS for nothing"
    for url in offline_gfs:
        assert {70.0, 50.0} <= _levels_asked(url), url
    manifest = json.loads(
        (Path(plan["data"]) / fetch.FETCH_MANIFEST_NAME).read_text())
    assert min(manifest["pressure_levels_hpa"]) == 50.0
    assert manifest["source_top_pressure_pa"] == 5000.0
    (bar,) = manifest["record_bars"]
    assert bar["expected"] == 134


def test_the_default_emission_and_its_printed_fetch_carry_50_hpa(
        tmp_path, capsys):
    """The pasted manual chain downloads what the config then prepares."""

    config = _emit(tmp_path, "default", "--explain")
    printed = capsys.readouterr().out
    assert domain_wizard.emitted_model_top_pa("gfs") == 5000.0
    assert "p_top = 5000.0" in config.read_text(encoding="utf-8")
    fetch_line = next(line for line in printed.splitlines()
                      if "gpuwm fetch " in line)
    assert "--p-top-pa 5000 " in fetch_line


@pytest.mark.parametrize("p_top,flag,top_hpa", [
    # The certified ladder already reaches it: the request is the one
    # this route has always made, byte for byte.
    (10000.0, None, 100.0),
    (20000.0, None, 100.0),
    (5000.0, "5000", 50.0),
    (2000.0, "2000", 20.0),
])
def test_the_fetch_follows_the_configs_own_top(
        tmp_path, offline_gfs, capsys, p_top, flag, top_hpa):
    config = _with_top(_emit(tmp_path), p_top, f"top{int(p_top)}")
    capsys.readouterr()
    plan = go_cli.plan_from_config(config, outdir=tmp_path / "go")
    command = go_cli.fetch_command(plan)
    if flag is None:
        assert "--p-top-pa" not in command
    else:
        assert command[command.index("--p-top-pa") + 1] == flag

    assert cli_main(command[3:]) == 0
    assert min(min(_levels_asked(url)) for url in offline_gfs) == top_hpa


def test_the_source_table_decides_not_the_source_name():
    from gpuwm.source_adapters import fetch_model_top_pa, get_source_adapter

    # GFS: the certified ladder stops at 100 hPa and the fetch extends it.
    assert fetch_model_top_pa("gfs", 5000.0) == 5000.0
    assert fetch_model_top_pa("gfs-0p25", 5000) == 5000.0
    assert fetch_model_top_pa("gfs", 10000.0) is None
    # A top above everything published is still asked for: the fetch
    # refuses it before any download, naming the deepest it can serve.
    assert fetch_model_top_pa("gfs", 0.5) == 0.5
    # GDAS fetches every published level already; HRRR has no isobaric
    # ladder; 20CRv3's PSL series stops at 100 hPa and its fetch takes
    # no top, so the preparation's coverage refusal names that instead.
    for source in ("gdas", "hrrr", "20crv3-cf"):
        assert fetch_model_top_pa(source, 5000.0) is None, source
    for nothing in (None, 0.0, -1.0, float("nan"), True):
        assert fetch_model_top_pa("gfs", nothing) is None
    assert fetch_model_top_pa("no-such-source", 5000.0) is None
    # The row's reach is the captured inventory's own top.
    assert get_source_adapter("gfs").extendable_source_top_pa == min(
        gfs_transport.CERTIFIED_AVAILABLE_LEVELS_HPA) * 100.0


def test_the_download_folder_is_keyed_on_the_top(tmp_path, capsys):
    """A folder fetched for 100 hPa holds fewer records; it is not reused."""

    import tomllib

    default = _emit(tmp_path)
    capsys.readouterr()
    shallow = _with_top(default, 10000.0, "shallow")
    deep_raw = tomllib.loads(default.read_text(encoding="utf-8"))
    shallow_raw = tomllib.loads(shallow.read_text(encoding="utf-8"))

    assert go_cli.config_fetch_request(deep_raw)["p_top_pa"] == 5000.0
    # A config that needs nothing more keeps the key it always had.
    assert go_cli.config_fetch_request(shallow_raw) == shallow_raw["fetch"]
    assert (go_cli.managed_download_key(go_cli.config_fetch_request(shallow_raw))
            == go_cli.managed_download_key(shallow_raw["fetch"]))
    assert (go_cli.managed_download_key(go_cli.config_fetch_request(deep_raw))
            != go_cli.managed_download_key(deep_raw["fetch"]))
    deep_plan = go_cli.plan_from_config(default, outdir=tmp_path / "go")
    shallow_plan = go_cli.plan_from_config(shallow, outdir=tmp_path / "go")
    assert deep_plan["data"] != shallow_plan["data"]


def test_the_download_price_counts_the_added_records():
    import tomllib

    hints = tomllib.loads(_BUDGET.read_text(encoding="utf-8"))["fetch"]
    certified = download_budget.download_estimate(dict(hints))
    deep = download_budget.download_estimate(dict(hints, p_top_pa=5000.0))
    assert deep["objects"] == certified["objects"] == 9
    assert deep["bytes"] == pytest.approx(certified["bytes"] * 134 / 124, abs=9)
    assert "134 records per file against the 124 measured" in deep["basis"]
    assert "records per file" not in certified["basis"]
    # The same request spelled as fetch flags is read back with its top.
    request = download_budget.request_from_arguments(
        ["--source", "gfs", "--cycle", "2026-09-26T06", "--hours", "24",
         "--area", "17.61,-116.54,53.24,-79.36", "--cadence", "3",
         "--p-top-pa", "5000", "--out", "data"])
    assert request["p_top_pa"] == 5000.0
    assert download_budget.download_estimate(request)["bytes"] == deep["bytes"]
    # GDAS was measured on its default whole ladder, and is priced on it.
    gdas = dict(hints, source="gdas")
    assert "records per file" not in download_budget.download_estimate(gdas)["basis"]


def test_the_preparation_memory_estimate_counts_the_levels_the_fetch_takes(
        tmp_path, capsys):
    """The ingest estimate sized every GFS forcing time at the 21-level
    ladder while a default 50 hPa run now decodes 23, so the phase was
    priced below what the run holds."""

    from gpuwm.core import preflight as pf

    deep_levels = len(fetch.container_subset_levels("gfs", top_pressure_pa=5000.0))
    assert deep_levels == 23
    assert pf.source_analysis_levels("gfs") == 21
    assert pf.source_analysis_levels("gfs", p_top_pa=10000.0) == 21
    assert pf.source_analysis_levels("gfs", p_top_pa=5000.0) == deep_levels
    # A top the fetch refuses is counted as every level it could take.
    assert pf.source_analysis_levels("gfs", p_top_pa=0.5) == len(
        gfs_transport.CERTIFIED_AVAILABLE_LEVELS_HPA)
    # ERA5 is fetched on its own full ladder whatever the top.
    assert pf.source_analysis_levels("era5", p_top_pa=5000.0) == 37
    assert (pf.source_analysis_fields_per_time("gfs", p_top_pa=5000.0)
            - pf.source_analysis_fields_per_time("gfs")) == 2 * 5

    default = _emit(tmp_path)
    capsys.readouterr()
    shallow = _with_top(default, 10000.0, "shallow")
    deep_exp = pf._load_experiment_any(default)
    shallow_exp = pf._load_experiment_any(shallow)
    deep = pf.estimate_ingest(deep_exp, source="gfs")
    certified = pf.estimate_ingest(shallow_exp, source="gfs")
    run = deep_exp.root.run
    mass = deep.items[0]
    assert mass.shape == (deep_levels, run.ny, run.nx), mass
    assert certified.items[0].shape == (21, run.ny, run.nx)
    added = (deep.category_bytes("analysis")
             - certified.category_bytes("analysis"))
    assert added == 4 * 2 * (3 * run.ny * run.nx + run.ny * (run.nx + 1)
                             + (run.ny + 1) * run.nx)
    assert deep.host_fields_per_time - certified.host_fields_per_time == 10


def test_run_plan_prices_the_download_its_chain_makes(tmp_path, capsys,
                                                       monkeypatch):
    from gpuwm.runplan import PLAN_SCHEMA, run_plan_main

    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    config = tmp_path / "gfs-3km.toml"
    config.write_text(_BUDGET.read_text(encoding="utf-8").replace(
        "p_top = 10000.0", "p_top = 5000.0"), encoding="utf-8")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "review", "route": "prepared",
        "config": {"path": str(config)},
        "output_root": str(tmp_path / "run")}), encoding="utf-8")
    assert run_plan_main(build_parser().parse_args(
        ["run-plan", "--estimate", str(plan)])) == 0
    download = json.loads(capsys.readouterr().out)["download"]
    assert "134 records per file" in download["basis"]


def test_an_interrupted_deep_fetch_resumes_on_its_own_ladder(
        tmp_path, offline_gfs, monkeypatch):
    calls = []

    def first_then_stop(url, destination, **_kw):
        calls.append(url)
        if len(calls) > 1:
            raise KeyboardInterrupt
        destination.write_bytes(_grib2(
            gfs_transport.record_count_for_levels(len(_levels_asked(url)))))

    monkeypatch.setattr(gfs_transport, "_download", first_then_stop)
    with pytest.raises(RuntimeError, match="resume exactly with") as caught:
        fetch.fetch_gfs(
            cycle=datetime(2026, 7, 28, 6), hours=(0, 3),
            area=fetch.parse_area("30,-100,40,-90"), out=tmp_path / "gfs",
            progress=lambda line: None, top_pressure_pa=5000.0,
            file_workers=1)
    resume = str(caught.value).split("resume exactly with: ", 1)[1].split()
    args = build_parser().parse_args(resume[1:])
    assert args.p_top_pa == 5000.0


def test_a_folder_fetched_for_another_top_is_refused_with_the_reason(
        tmp_path, offline_gfs, capsys):
    """`gpuwm go --data-dir` over a folder fetched by hand for 100 hPa.

    The run's fetch now asks for 50 hPa, so the crops already there are
    two levels short; the refusal says that is the reason rather than
    only the two record counts.
    """

    request = ["fetch", "--source", "gfs", "--cycle", "2026-07-28T06",
               "--hours", "3", "--area=30,-100,40,-90", "--cadence", "3",
               "--out", str(tmp_path / "gfs")]
    assert cli_main(request) == 0
    capsys.readouterr()

    assert cli_main([*request, "--p-top-pa", "5000"]) != 0
    err = capsys.readouterr().err
    assert "carries 124 GRIB2 messages, expected 134" in err
    assert ("takes 23 isobaric levels up to 50 hPa and the file holds 21, "
            "so it was fetched for another model top") in err


def test_a_folder_fetched_without_the_top_is_refused_with_the_fetch_flag():
    """Only a hand-fetched folder reaches preparation short of its top now,
    and its refusal names the fetch flag rather than only the eta ladder."""

    from types import SimpleNamespace

    from gpuwm.gfs_direct import _refuse_a_fetch_below_the_model_top
    from gpuwm.ingest.source_coverage import VerticalLadderRefusal

    def config(p_top):
        return SimpleNamespace(vertical=SimpleNamespace(p_top=p_top))

    with pytest.raises(VerticalLadderRefusal) as refused:
        _refuse_a_fetch_below_the_model_top(config(5000.0), 10000.0)
    text = str(refused.value)
    assert "stops at 10000 Pa" in text and "model top is 5000 Pa" in text
    assert "gpuwm fetch --source gfs --p-top-pa 5000" in refused.value.remedy
    # A folder that reaches the top, or goes past it, is not refused here.
    _refuse_a_fetch_below_the_model_top(config(10000.0), 10000.0)
    _refuse_a_fetch_below_the_model_top(config(10000.0), 5000.0)


def test_the_short_folder_refusal_reaches_the_reader_as_a_preparation_refusal(
        monkeypatch, tmp_path, capsys):
    """The door prints it as the refusal and its remedy at the family's
    exit status, as it printed the vertical contract's refusal for the
    same folder, not as one flat error line at status 2."""

    from types import SimpleNamespace

    from gpuwm import gfs_direct
    from gpuwm.ingest.source_coverage import PREPARATION_REFUSAL_EXIT_CODE

    def prepare(**_kwargs):
        gfs_direct._refuse_a_fetch_below_the_model_top(
            SimpleNamespace(vertical=SimpleNamespace(p_top=5000.0)), 10000.0)

    monkeypatch.setattr(gfs_direct, "prepare_gfs_wrf", prepare)
    rc = gfs_direct.main([
        "--series", "series.tsv", "--cycle", "2026-09-27_06:00:00",
        "--bridge", "bridge.exe", "--wps-namelist", "namelist.wps",
        "--experiment-config", "experiment.toml",
        "--input-manifest", "manifest.json",
        "--input-manifest-sha256", "a" * 64,
        "--output-root", str(tmp_path / "prep"),
    ])
    err = capsys.readouterr().err
    assert rc == PREPARATION_REFUSAL_EXIT_CODE
    lines = err.splitlines()
    assert lines[0].startswith("prep: REFUSED: GFS source atmosphere stops "
                               "at 10000 Pa")
    assert lines[1].startswith("remedy: gpuwm fetch --source gfs "
                               "--p-top-pa 5000")
    assert "rw-wps --source gfs:" not in err


_REPO = Path(__file__).resolve().parents[1]
_USER_DOCS = (*sorted((_REPO / "docs" / "public").glob("*.md")),
              _REPO / "docs" / "examples.md")


def _documented_commands(text: str) -> list[str]:
    """Every command in the page's fenced blocks, continuation lines joined."""

    commands: list[str] = []
    for block in re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S):
        joined = block.replace("\\" + "\n", " ")
        commands += [line.strip() for line in joined.splitlines()]
    return commands


def test_the_documented_gfs_downloads_reach_the_wizard_configs_top():
    """Every GFS config `gpuwm domain` writes carries a 50 hPa top, so a
    documented download line without --p-top-pa fetched the 100 hPa
    ladder and the chain a reader typed from the page ended in the
    preparation refusal for a folder short of its top."""

    from gpuwm.source_adapters import fetch_model_top_pa

    top = fetch_model_top_pa("gfs", domain_wizard.emitted_model_top_pa("gfs"))
    assert top == 5000.0
    pages = set()
    for page in _USER_DOCS:
        for command in _documented_commands(page.read_text(encoding="utf-8")):
            if not (command.startswith("gpuwm fetch ")
                    and re.search(r"--source[ =]gfs\b", command)
                    and "--cycle" in command):
                continue
            pages.add(page.name)
            assert f"--p-top-pa {top:g} " in command, (page.name, command)
    assert {"FIRST-LIGHT.md", "WITHOUT-A-GPU.md", "DOWNSCALE.md", "DATA.md",
            "examples.md"} <= pages


def _printed_fetch_line(printed: str) -> str | None:
    """The `gpuwm fetch` line of the wizard's closing next: block, if any."""

    for line in printed.split("next:")[-1].splitlines():
        text = re.sub(r"^\s*\d+\.\s*", "", line).strip()
        if text.startswith("gpuwm fetch "):
            return text
    return None


def _option(tokens: list[str], name: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == name and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


@pytest.mark.parametrize("page, fetch_out", [
    ("WITHOUT-A-GPU.md", "data/cpuwalk"), ("FIRST-LIGHT.md", "data/myarea"),
    ("DOWNSCALE.md", "gfs-raw"), ("DOWNSCALE.md", "era5-raw")])
def test_a_manual_walk_domain_command_prints_the_fetch_line_it_tells_you_to_paste(
        tmp_path, monkeypatch, capsys, page, fetch_out):
    """The manual walks run `gpuwm domain`, then paste the fetch line it
    printed.  The wizard prints that line only under --explain; without it
    the closing block is one `gpuwm go` line, so the documented domain
    command left the reader nothing to paste, and the line it did print
    under --explain downloaded into <config dir>/data/<name> rather than
    the folder the page's later steps read."""

    import shlex

    commands = _documented_commands(
        (_REPO / "docs" / "public" / page).read_text(encoding="utf-8"))
    fetch = next(index for index, command in enumerate(commands)
                 if command.startswith("gpuwm fetch ") and "--cycle" in command
                 and command.endswith(f"--out {fetch_out}"))
    documented_fetch = commands[fetch]
    domain = max(index for index, command in enumerate(commands[:fetch])
                 if command.startswith("gpuwm domain "))
    tokens = shlex.split(commands[domain].replace("\\", "/"))[2:]
    tokens = ["2026-07-29T18" if token == "latest" else token for token in tokens]
    if "--geog-root" in tokens:
        geog = tmp_path / "WPS_GEOG"
        geog.mkdir()
        tokens[tokens.index("--geog-root") + 1] = str(geog)
    monkeypatch.chdir(tmp_path)
    assert cli_main(["domain", *tokens]) == 0
    printed = _printed_fetch_line(capsys.readouterr().out)
    assert printed is not None, f"{page}: the documented domain command prints no fetch line"
    fetched = shlex.split(printed)
    wanted = shlex.split(documented_fetch)
    assert _option(fetched, "--source") == _option(wanted, "--source")
    assert _option(fetched, "--p-top-pa") == _option(wanted, "--p-top-pa")
    assert (Path(_option(fetched, "--out")).resolve()
            == (tmp_path / _option(wanted, "--out")).resolve()), (printed, documented_fetch)

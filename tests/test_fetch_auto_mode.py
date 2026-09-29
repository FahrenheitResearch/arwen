"""An explicit ``--mode auto`` takes the default transport instead of being refused.

Breakage this prevents: ``gpuwm fetch --source rap --mode auto`` (and every
other table-driven source) stopped with "--source rap has one byte
transport, so there is nothing to probe.  Drop --mode.", and GFS/GDAS
stopped with "'auto' has nothing to probe", although the product already
knows the automatic choice: the full-file route on a table source, and on
GFS/GDAS the same crop, cached-crop reuse and archive fallback an omitted
--mode takes.  Pricing a GFS request that named ``auto`` found no row and
left its download unpriced.

No bytes move: each transfer owner is replaced by a stop marker.
"""
from __future__ import annotations

import pytest

from gpuwm import download_budget, fetch, fetch_routes
from gpuwm.cli import build_parser

GFS_BOX = "30,-100,40,-90"


class TransferReached(Exception):
    """The fetch got past every check and handed the plan to a transfer."""


@pytest.mark.parametrize("source", fetch_routes.route_ids())
def test_auto_is_the_default_full_file_route_on_every_table_source(source):
    assert fetch_routes.resolve_mode(source, "auto") == "full-file"
    assert fetch_routes.resolve_mode(source, "auto") == fetch_routes.resolve_mode(source, None)


def test_a_table_source_fetch_with_auto_reaches_its_transfer(tmp_path, monkeypatch):
    monkeypatch.setenv("GPUWM_FETCH_LOCK_ROOT", str(tmp_path / "locks"))
    args = build_parser().parse_args([
        "fetch", "--source", "rap", "--cycle", "2026-09-27T00", "--hours", "1",
        "--mode", "auto", "--out", str(tmp_path / "fetch")])
    plans = []

    def transfer(plan, **kwargs):
        plans.append(plan)
        raise TransferReached

    monkeypatch.setattr(fetch_routes, "run_plan", transfer)
    with pytest.raises(TransferReached):
        fetch.fetch_main(args)
    assert plans[0].source_id == "rap" and plans[0].objects


def test_an_unsupported_record_subset_still_refuses_in_its_own_words():
    with pytest.raises(ValueError, match="idx-subset"):
        fetch_routes.resolve_mode("gefs", "idx-subset")


def _container_dispatch(tmp_path, monkeypatch, source, archive, mode):
    """Which transfer a GFS-container fetch reaches for one cycle age and mode."""
    argv = ["fetch", "--source", source, "--cycle", "2026-07-28T06", "--hours", "3",
            "--area", GFS_BOX, "--out", str(tmp_path / f"{source}-{mode}-{archive}")]
    if mode is not None:
        argv += ["--mode", mode]
    args = build_parser().parse_args(argv)
    monkeypatch.setenv("GPUWM_FETCH_LOCK_ROOT", str(tmp_path / "locks"))
    # A recent cycle whose crops are already on disk, or an old cycle the
    # crop host no longer keeps.
    monkeypatch.setattr(fetch, "cached_request_complete", lambda *a, **k: not archive)
    monkeypatch.setattr(fetch, "archive_only_cycle", lambda *a, **k: archive)
    monkeypatch.setattr(fetch, "require_published_cycle", lambda *a, **k: None)
    monkeypatch.setattr(fetch, "select_fetch_engine", lambda *a, **k: fetch.FetchEngineChoice(
        engine="rust", binary=None, selection="explicit"))
    reached = []

    def stop(route):
        def transfer(*a, **k):
            reached.append(route)
            raise TransferReached
        return transfer

    monkeypatch.setattr(fetch, "fetch_gfs_fullfile", stop("archive whole objects"))
    monkeypatch.setattr(fetch, "fetch_gfs", stop("grib-filter crop"))
    with pytest.raises(TransferReached):
        fetch.fetch_main(args)
    return reached


@pytest.mark.parametrize("source", fetch.GFS_CONTAINER_SOURCES)
@pytest.mark.parametrize("archive", [False, True])
def test_auto_on_gfs_and_gdas_takes_the_route_an_omitted_mode_takes(
        tmp_path, monkeypatch, source, archive):
    default = _container_dispatch(tmp_path, monkeypatch, source, archive, None)
    auto = _container_dispatch(tmp_path, monkeypatch, source, archive, "auto")
    assert default == ["archive whole objects" if archive else "grib-filter crop"]
    assert auto == default


def test_gfs_record_subsetting_still_refuses(tmp_path, capsys):
    from gpuwm import cli

    assert cli.main(["fetch", "--source", "gfs", "--cycle", "2026-07-28T06", "--hours", "3",
                     "--out", str(tmp_path / "out"), "--mode", "idx-subset"]) == 2
    assert "not a certified GFS route" in capsys.readouterr().err


@pytest.mark.parametrize("source", fetch.GFS_CONTAINER_SOURCES)
@pytest.mark.parametrize("archive", [False, True])
def test_auto_on_gfs_and_gdas_is_priced_as_the_default(monkeypatch, source, archive):
    monkeypatch.setattr(fetch, "archive_only_cycle", lambda *a, **k: archive)
    request = {"source": source, "cycle": "2026-09-26T00", "hours": 6, "cadence": 3,
               "area": GFS_BOX}
    default = download_budget.download_estimate(request)
    auto = download_budget.download_estimate(dict(request, mode="auto"))
    assert default["bytes"] is not None
    assert (auto["mode"], auto["bytes"]) == (default["mode"], default["bytes"])
    assert default["mode"] == ("full-file" if archive else "grib-filter")


def test_hrrr_keeps_its_own_auto_probe_rule_in_pricing():
    request = {"source": "hrrr", "cycle": "2026-09-26T00", "hours": 1,
               "area": GFS_BOX, "mode": "auto"}
    assert download_budget.download_estimate(request)["mode"] == "auto"


def test_the_help_no_longer_says_auto_refuses(capsys):
    from gpuwm import cli

    with pytest.raises(SystemExit):
        cli.main(["fetch", "--help"])
    # argparse wraps at hyphens too, so the fragments carry none.
    text = " ".join(capsys.readouterr().out.split())
    assert "'auto'/'idx" not in text
    assert "omitted or 'auto', the NOMADS" in text

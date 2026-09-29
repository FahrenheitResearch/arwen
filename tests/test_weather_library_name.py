"""The page ``gpuwm gui`` serves is named the Weather Library wherever a user reads it.

Its old name stays only where the allow-list keeps it (``KEPT`` in
tools/dev/rename_weather_library.py): file formats and file names saved runs
and the page store use, the assistant's tool names, the page's own module
names.  Its old addresses keep answering: ``/api/wiki/...`` answers as
``/api/library/...`` does, and ``#/wiki`` forwards to ``#/library``.  The
decisions are in docs/dev/WEATHER-LIBRARY-RENAME.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading

import pytest

from gpuwm.gui.server import build_server, serve_in_thread

from test_gui_server import FakeRunner, make_run, request
from test_gui_wiki import RECIPES, SEED, Era5Runner

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "gpuwm" / "gui" / "static" / "js"


def _tool():
    spec = importlib.util.spec_from_file_location("rename_weather_library",
                                                  ROOT / "tools" / "dev" / "rename_weather_library.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


# ---------------------------------------------------------------- the words

def test_no_user_facing_string_names_the_page_by_its_old_name():
    findings = TOOL.scan(ROOT)
    assert not findings, (
        "The page is the Weather Library. Run python tools/dev/rename_weather_library.py, or give the rule or "
        "the kept entry that covers each of these:\n" + "\n".join(str(f) for f in findings))


def test_the_rename_tool_would_change_nothing_and_refuse_nothing_on_this_tree():
    # A refusal here means a line an exact edit anchors on was rewritten: the tool would refuse at the cut.
    outcomes = TOOL.plan(ROOT)
    assert not [o.rel for o in outcomes if o.refusals], [r for o in outcomes for r in o.refusals]
    assert not [o.rel for o in outcomes if o.after != o.before], [(o.rel, o.lines) for o in outcomes]


def test_every_exact_edit_is_already_in_its_final_words():
    # An edit's result that another rule would rewrite again would make every run change the file.
    for edit in (*TOOL.EDITS, *TOOL.NOTES_EDITS):
        again, changes = TOOL.swap(edit.new, (*TOOL.ROUTES, *TOOL.PROSE))
        assert again == edit.new and not changes, (edit.path, changes)


def test_an_edit_that_keeps_its_old_text_inside_its_result_names_the_marks_of_its_result():
    # Without the marks, a run after someone changed the inserted lines would insert them a second time.
    for edit in TOOL.EDITS:
        if edit.old in edit.new:
            assert edit.made, edit.why
        for pattern in edit.made:
            assert re.search(pattern, edit.new, re.M) and not re.search(pattern, edit.old, re.M), (edit.why, pattern)
        for pattern in edit.stale:
            assert re.search(pattern, edit.old, re.M) and not re.search(pattern, edit.new, re.M), (edit.why, pattern)


def _rename(rel: str, text: str, edits=None):
    edits = tuple(e for e in TOOL.EDITS if e.path == rel) if edits is None else edits
    return TOOL.rename_text(text, TOOL.scope_of(rel), edits, TOOL.tree_version(ROOT), rel=rel)


def _text(rel: str) -> str:
    return (ROOT / rel).read_bytes().decode("utf-8")


def test_rewording_any_line_an_edit_anchors_on_refuses_nothing_while_no_old_name_is_left():
    # Each edit as if its lines were reworded after the rename: neither its old text nor its new text is left.
    for edit in TOOL.EDITS:
        text = _text(edit.path)
        reworded = dataclasses.replace(edit, old="\x00 the line as it read before", new="\x00 the line as it reads now")
        done = _rename(edit.path, text, (reworded,))
        assert not done.refusals and done.text == text and not done.lines, (edit.why, done.refusals)
        if not edit.made:  # an edit with marks is known to be made by its marks alone, and says nothing
            assert len(done.held) == 1 and edit.why in done.held[0]


def test_a_reworded_release_paragraph_counts_as_made_and_one_that_names_the_page_a_wiki_is_refused():
    text = _text("README.md")
    reworded = text.replace("ArWen's graphical interface in", "the main way to use ArWen, in")
    assert reworded != text
    done = _rename("README.md", reworded)
    assert not done.refusals and done.text == reworded and not done.lines
    assert done.held == ["edit (the release paragraph): its lines were reworded and hold no old name, so it counts "
                         "as made"]
    renamed = next(e for e in TOOL.EDITS if e.why == "the release paragraph").new
    stale = text.replace(renamed, renamed.replace("adds the Weather Library,", "adds a wiki,"))
    assert stale != text
    done = _rename("README.md", stale)
    assert len(done.refusals) == 1 and "still holds 'wiki' (the old name)" in done.refusals[0]


def test_the_weather_library_is_not_called_a_preview_beside_the_desktop_app():
    # The Weather Library is ArWen's GUI; ArWen Desktop keeps shipping beside it with its features unchanged.
    log = _text("CHANGELOG.md")
    region = TOOL.changelog_region(log, TOOL.tree_version(ROOT))
    places = {"README.md": _text("README.md"), "docs/public/GUI.md": _text("docs/public/GUI.md"),
              "CHANGELOG.md": log[region[0]:region[1]] if region else ""}
    for rel, text in places.items():
        found = re.search(r"(?i)preview\s+beside\s+the\s+desktop|\bis\s+a\s+preview\s+and\s+the\s+desktop\s+app\b", text)
        assert found is None, (rel, found.group(0))


def test_a_reworded_title_that_still_names_the_page_arwen_is_refused():
    text = _text("gpuwm/gui/static/index.html")
    done = _rename("gpuwm/gui/static/index.html", text.replace("<title>Weather Library</title>", "<title>ArWen GUI</title>"))
    assert any("still names the old page ('<title>ArWen')" in refusal for refusal in done.refusals), done.refusals
    done = _rename("gpuwm/gui/static/index.html", text.replace("<title>Weather Library</title>", "<title>Storms</title>"))
    assert not done.refusals and len(done.held) == 1


def test_lines_an_edit_inserted_are_not_inserted_again_after_someone_changes_them():
    router = _text("gpuwm/gui/static/js/router.js")
    more = router.replace('new Map([["wiki", "library"]])', 'new Map([["wiki", "library"], ["home", "library"]])')
    assert more != router
    done = _rename("gpuwm/gui/static/js/router.js", more)
    assert not done.refusals and done.text == more and not done.lines
    api = _text("gpuwm/gui/api.py")
    wider = api.replace('SECTIONS = {"library": "wiki"}', 'SECTIONS = {"library": "wiki", "storms": "wiki"}')
    assert wider != api
    done = _rename("gpuwm/gui/api.py", wider)
    assert not done.refusals and done.text == wider and not done.lines
    # Half of an insertion left is refused: the other half cannot be put back without a second copy of the first.
    half = router.replace("export function forwardedRoute(", "export function movedRoute(")
    done = _rename("gpuwm/gui/static/js/router.js", half)
    assert len(done.refusals) == 1 and "part of its result is here and part is not" in done.refusals[0]


def test_a_renamed_script_that_would_not_load_is_refused():
    assert "declares MOVED_ROUTES twice" in TOOL._valid(
        "router.js", "export const MOVED_ROUTES = 1;\nexport const MOVED_ROUTES = 2;\n")
    assert TOOL._valid("router.js", _text("gpuwm/gui/static/js/router.js")) is None


def test_the_scan_finds_the_old_name_in_a_reply_about_tabs():
    rel = "gpuwm/gui/server.py"
    found = TOOL.scan_text(rel, 'FIX = "Close a few ArWen tabs and reload."\n', TOOL.scope_of(rel), None)
    assert [f.text for f in found] == ["ArWen tabs"]


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "tree"
    files = {
        "pyproject.toml": '[project]\nname = "gpuwm"\nversion = "9.9.0"\n',
        "CHANGELOG.md": ("# Changelog\n\n## 9.9.0 (unreleased)\n\n- `gpuwm gui` opens ArWen in your browser; the storm "
                         "wiki has 17 events, each wiki event a best run (`GET /api/wiki/event/ID`).\n\n"
                         "## 9.8.0 (2026-01-01)\n\n- The storm wiki shipped.\n"),
        "docs/public/PAGE.md": ("# Pages\n\nThe sidebar opens a storm wiki. Ask the wiki.\n\n"
                                "```\ncurl /api/wiki/search?q=x\n```\n\nIts runs carry `wiki-run.json`.\n"),
        "gpuwm/gui/copy/words.json": json.dumps({"about": "The storm wiki's words.",
                                                 "nav": {"wiki_group": "Storm wiki", "wiki": "Main page"},
                                                 "none": "No event in the wiki matches."}, indent=2) + "\n",
        "gpuwm/gui/pages.py": ('"""The storm wiki\'s pages; a docstring keeps its words."""\n\n'
                               'WIKI_DIR = "wiki"\n'
                               'MISSING = ("No such wiki page.", "Go to the wiki\'s main page.")\n'
                               'LINK = "wiki-run.json"\n'
                               '# a comment about the storm wiki stays\n'),
        "gpuwm/gui/static/js/nav.js": ('// The wiki\'s links (#/wiki) and its API.\n'
                                       'export const HOME = "#/wiki";\n'
                                       'export const main = () => fetch(`/api/wiki/kind/${id}`);\n'),
        "gpuwm/gui/static/index.html": "<title>ArWen</title>\n",
        "tests/test_saved.py": 'SAVED = {"page": "Storm wiki", "path": "/api/wiki", "link": "wiki-run.json"}\n',
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(text.encode("utf-8"))
    return root


def _read(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in root.rglob("*") if p.is_file()}


#: The small tree's one exact edit, the real tab title's; the others name files and sentences this tree does not have.
TITLE = dataclasses.replace(next(e for e in TOOL.EDITS if e.old == "<title>ArWen</title>"), why="the tab title")


def test_the_tool_renames_words_and_addresses_once_and_leaves_what_it_keeps(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(TOOL, "EDITS", (TITLE,))
    root = _tree(tmp_path)
    before = _read(root)
    assert TOOL.main(["--root", str(root)]) == 0
    printed = capsys.readouterr().out
    after = _read(root)
    assert "docs/public/PAGE.md" in printed and "'the wiki' -> 'the Weather Library'" in printed
    assert "edit: the tab title" in printed and after["gpuwm/gui/static/index.html"] == "<title>Weather Library</title>\n"
    log = after["CHANGELOG.md"]
    unreleased, released = log.split("## 9.8.0")
    assert "opens the Weather Library in your browser; the Weather Library has 17 events, each Weather Library " \
           "event a best run (`GET /api/library/event/ID`)" in unreleased
    assert released == before["CHANGELOG.md"].split("## 9.8.0")[1]
    page = after["docs/public/PAGE.md"]
    assert "The sidebar opens the Weather Library. Ask the Weather Library." in page
    assert "curl /api/library/search?q=x" in page and "`wiki-run.json`" in page
    words = json.loads(after["gpuwm/gui/copy/words.json"])
    assert words["nav"] == {"wiki_group": "Weather Library", "wiki": "Main page"}
    assert words["none"] == "No event in the Weather Library matches." and words["about"] == "The Weather Library's words."
    code = after["gpuwm/gui/pages.py"]
    assert code.startswith('"""The storm wiki\'s pages') and "# a comment about the storm wiki stays" in code
    assert 'WIKI_DIR = "wiki"' in code and 'LINK = "wiki-run.json"' in code
    assert '("No such Weather Library page.", "Go to the Weather Library\'s main page.")' in code
    script = after["gpuwm/gui/static/js/nav.js"]
    assert 'HOME = "#/library"' in script and "`/api/library/kind/${id}`" in script
    assert after["tests/test_saved.py"] == before["tests/test_saved.py"]
    assert TOOL.scan(root) == []
    # A second run changes nothing.
    assert TOOL.main(["--root", str(root)]) == 0
    assert "Nothing to change" in capsys.readouterr().out
    assert _read(root) == after


def test_a_file_whose_edit_no_longer_matches_and_still_names_the_page_arwen_is_refused_and_nothing_is_written(
        tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(TOOL, "EDITS", (TITLE,))
    root = _tree(tmp_path)
    page = root / "gpuwm" / "gui" / "static" / "index.html"
    page.write_text("<title>ArWen, a page someone renamed by hand</title>\n", encoding="utf-8")
    before = _read(root)
    assert TOOL.main(["--root", str(root)]) == 2
    printed = capsys.readouterr().out
    assert "REFUSED gpuwm/gui/static/index.html" in printed and "still names the old page" in printed
    assert "Nothing was written" in printed
    assert _read(root) == before
    # Reworded with no old name left, the edit counts as made and the rest of the tree is renamed.
    page.write_text("<title>A page someone renamed by hand</title>\n", encoding="utf-8")
    assert TOOL.main(["--root", str(root)]) == 0
    printed = capsys.readouterr().out
    assert "gpuwm/gui/static/index.html: edit (the tab title): its lines were reworded and hold no old name" in printed
    assert page.read_text(encoding="utf-8") == "<title>A page someone renamed by hand</title>\n"
    assert TOOL.scan(root) == []


def test_release_notes_are_renamed_on_request(tmp_path, monkeypatch):
    monkeypatch.setattr(TOOL, "EDITS", (TITLE,))
    notes = tmp_path / "release-notes-9.9.0.md"
    notes.write_text("ArWen 9.9.0 is the first to ship ArWen in your browser as a preview. It is a local page "
                     "with the storm wiki, New forecast, My forecasts. Known limits: the web GUI is a preview.\n",
                     encoding="utf-8")
    root = _tree(tmp_path)
    assert TOOL.main(["--root", str(root), "--notes", str(notes)]) == 0
    text = notes.read_text(encoding="utf-8")
    assert text == ("ArWen 9.9.0 is the first to ship the Weather Library in your browser as a preview. It is a "
                    "local page with pages for past storms, New forecast, My forecasts. Known limits: the Weather "
                    "Library is a preview.\n")


# ---------------------------------------------------------------- the old addresses

@pytest.fixture()
def gui(tmp_path):
    server = build_server(tmp_path / "runs", port=0, runner=FakeRunner(), token="t" * 43)
    serve_in_thread(server)
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture()
def era5_gui(tmp_path):
    server = build_server(tmp_path / "runs", port=0, runner=Era5Runner(), token="t" * 43)
    serve_in_thread(server)
    yield server
    server.shutdown()
    server.server_close()


def test_each_old_api_path_answers_as_its_new_one(gui):
    event = SEED["events"][0]
    place = SEED["places"][0]
    for tail in ("", "/places", "/changes", f"/event/{event['id']}", "/kind/tornado", f"/place/{place['id']}",
                 f"/recipe/{event['id']}", f"/recipe/{event['id']}?card=8", "/search?q=tornado&sort=score"):
        new_response, new = request(gui, "GET", f"/api/library{tail}")
        old_response, old = request(gui, "GET", f"/api/wiki{tail}")
        assert new_response.status == 200, (tail, new)
        assert old_response.status == 200 and old == new, tail
    response, missing = request(gui, "GET", "/api/library/event/no-such-event")
    assert response.status == 404
    response, gone = request(gui, "GET", "/api/library/no-such-page")
    assert response.status == 404 and gone["message"] == "No such Weather Library page."
    assert gone["fix"] == "Go to the Weather Library's main page."
    _, old_gone = request(gui, "GET", "/api/wiki/no-such-page")
    assert old_gone == gone


def test_the_reply_to_too_many_live_views_names_the_weather_library_tabs(gui):
    make_run(gui.root, "busy", plan=True)
    gui.streams = threading.BoundedSemaphore(1)
    gui.streams.acquire()
    response, body = request(gui, "GET", "/api/runs/busy/events")
    assert response.status == 503
    assert body["fix"] == "Close a few Weather Library tabs and reload."


def test_the_old_simulate_path_starts_the_same_run_as_the_new_one(era5_gui, monkeypatch):
    from gpuwm.gui import api as api_module

    monkeypatch.setattr(api_module, "disk_free_gib", lambda path: 500.0)
    event = next(e for e in SEED["events"] if RECIPES[e["id"]]["source"] == "era5" and e["type"] == "tornado")
    body = {"event": event["id"], "card_gb": 16, "dry_run": True}
    response, new = request(era5_gui, "POST", "/api/library/simulate", body=body)
    assert response.status == 200 and new["dry_run"], new
    response, old = request(era5_gui, "POST", "/api/wiki/simulate", body=body)
    assert response.status == 200 and old == new


def test_the_page_asks_the_new_api_path_and_links_the_new_address():
    for path in sorted(JS.glob("*.js")):
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"/api/wiki\b", text), path.name
        assert not re.search(r"#/wiki\b", text), path.name
    registered = {m.group(1) for p in JS.glob("*.js") for m in re.finditer(r'register\("(\w+)"', p.read_text(
        encoding="utf-8"))}
    assert "library" in registered and "wiki" not in registered
    app = (JS / "app.js").read_text(encoding="utf-8")
    route = app[app.index("async function route()"):]
    # The forward comes first: nothing is drawn for an old address.
    assert route.index("forwardedRoute(") < route.index("frame(") and "location.replace(" in route


def test_every_page_the_assistant_opens_is_a_route_the_page_draws():
    from gpuwm.gui.assistant.agent import TOOLS

    registered = {m.group(1) for p in JS.glob("*.js") for m in re.finditer(r'register\("(\w+)"', p.read_text(
        encoding="utf-8"))}
    opener = next(t["function"] for t in TOOLS if t["function"]["name"] == "open_page")
    pages = opener["parameters"]["properties"]["page"]["enum"]
    assert "library" in pages and "wiki" not in pages
    assert set(pages) <= registered, set(pages) - registered


ROUTER_CHECK = r"""
import { forwardedRoute, MOVED_ROUTES } from "./router.mjs";
console.log(JSON.stringify({
  moved: [...MOVED_ROUTES],
  wiki: forwardedRoute("wiki"),
  deep: forwardedRoute("wiki/extra/part"),
  library: forwardedRoute("library"),
  empty: forwardedRoute(""),
  event: forwardedRoute("event/tc-2005236n23285"),
}));
"""


def test_the_old_page_address_forwards_to_the_new_one(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed")
    shutil.copyfile(JS / "router.js", tmp_path / "router.mjs")
    (tmp_path / "t.mjs").write_text(ROUTER_CHECK, encoding="utf-8")
    done = subprocess.run([node, "t.mjs"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout)
    assert answer["moved"] == [["wiki", "library"]]
    assert answer["wiki"] == "library" and answer["deep"] == "library/extra/part"
    assert answer["library"] is None and answer["empty"] is None and answer["event"] is None


# ---------------------------------------------------------------- the terminal

def _commands() -> dict[str, str]:
    from gpuwm.cli import build_parser

    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return {action.dest: action.help or "" for action in sub._choices_actions}


def test_the_help_names_the_weather_library(capsys):
    from gpuwm import cli

    helps = _commands()
    # gpuwm gui keeps its name: it names what it opens by kind, and no command or flag carried the old name.
    assert "gui" in helps and not [name for name in helps if "wiki" in name]
    assert helps["gui"].startswith("open the Weather Library in your browser")
    assert "the Weather Library's panel" in helps["assistant"]
    with pytest.raises(SystemExit):
        cli.main(["--help-all"])
    listed = capsys.readouterr().out
    assert re.search(r"^\s+gui\s+open the Weather Library in your browser", listed, re.M), listed[:2000]
    # The first-use guide lists it too, within its 30 lines.
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    short = capsys.readouterr().out
    assert re.search(r"^\s+gpuwm gui\s+Open the Weather Library in your browser$", short, re.M), short
    assert len(short.splitlines()) <= 30

"""Execute checkout installers with captured external commands, without installs.

Both shells run the actual shipped script through the standalone clone route.
Only git, pip, cargo and doctor are substitutes; they enforce the dependency
and executable prerequisites a clean checkout needs, and inject failures.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = r'''
import json, os, pathlib, sys
kind, *args = sys.argv[1:]
cwd = pathlib.Path.cwd()
with open(os.environ['GPUWM_INSTALL_TEST_LOG'], 'a', encoding='utf8') as stream:
    stream.write(json.dumps({'kind': kind, 'args': args, 'cwd': str(cwd)}) + '\n')
if kind == 'git':
    assert args[:1] == ['clone'], args
    checkout = cwd / args[-1]
    for name in ('gpuwm', 'gpuwm-data', 'tools/grib1_bridge', 'tools/rustwx',
                 'tools/arwen-tui', '.venv/bin', '.venv/Scripts'):
        (checkout / name).mkdir(parents=True, exist_ok=True)
    (checkout / 'pyproject.toml').write_text('# fixture checkout\n')
    for name in ('python', 'gpuwm'):
        program = checkout / '.venv/bin' / name
        program.write_text('#!/bin/sh\nexec "$GPUWM_INSTALL_TEST_PYTHON" '
                           '"$GPUWM_INSTALL_TEST_BACKEND" ' + name + ' "$@"\n')
        program.chmod(0o755)
        (checkout / '.venv/Scripts' / (name + '.exe')).touch()
elif kind == 'python' and args[:3] == ['-m', 'pip', 'install']:
    if 'gpuwm-data' in args:
        if os.environ.get('GPUWM_INSTALL_TEST_FAIL') == 'companion':
            sys.exit(17)
        assert (cwd / 'gpuwm-data').is_dir()
        (cwd / '.companion-installed').touch()
    elif any(arg.startswith('.[') for arg in args):
        if (os.environ.get('GPUWM_INSTALL_TEST_COMPANION_GATE', '1') == '1'
                and not (cwd / '.companion-installed').exists()):
            print('fixture pip: checkout gpuwm-data must satisfy the exact local pin first', file=sys.stderr)
            sys.exit(42)
elif kind == 'cargo':
    assert args == ['build', '--release', '--locked', '--offline'], args
    if cwd.name == 'arwen-tui':
        if os.environ.get('GPUWM_INSTALL_TEST_FAIL') == 'tui':
            sys.exit(19)
        output = cwd / 'target/release/arwen-tui'
        output.parent.mkdir(parents=True, exist_ok=True)
        output.touch()
        output.with_suffix('.exe').touch()
elif kind == 'gpuwm' and args == ['doctor']:
    if not (cwd / 'tools/arwen-tui/target/release/arwen-tui').is_file():
        print('fixture doctor: terminal workspace was never built', file=sys.stderr)
        sys.exit(43)
    sys.exit(int(os.environ.get('GPUWM_INSTALL_TEST_DOCTOR_EXIT', '0')))
'''


def _shell(platform):
    if platform == 'powershell':
        found = shutil.which('powershell') or shutil.which('pwsh')
    elif os.name == 'nt':
        found = next((str(path) for path in (
            Path('C:/Program Files/Git/bin/sh.exe'),
            Path('C:/Program Files/Git/usr/bin/sh.exe')) if path.is_file()), None)
    else:
        found = shutil.which('sh')
    if not found:
        pytest.skip(f'{platform} is not installed on this test host')
    return found


def _run(tmp_path, platform, *, no_render=False, failure='', companion_gate=True, doctor_exit=0):
    shell = _shell(platform)
    stage = tmp_path / 'new checkout with spaces'
    stage.mkdir()
    backend = tmp_path / 'commands.py'
    backend.write_text(BACKEND, encoding='utf8')
    log = tmp_path / 'commands.jsonl'
    env = dict(os.environ)
    for key in ('GPUWM_INSTALL_CUDA', 'GPUWM_INSTALL_NO_RENDER', 'GPUWM_INSTALL_YES',
                'GPUWM_INSTALL_NO_FETCH_TABLES', 'GPUWM_PYTHON'):
        env.pop(key, None)
    env.update({
        'GPUWM_REPO_URL': 'https://example.invalid/local-source-fixture',
        'GPUWM_INSTALL_TEST_PYTHON': Path(sys.executable).as_posix(),
        'GPUWM_INSTALL_TEST_BACKEND': backend.as_posix(),
        'GPUWM_INSTALL_TEST_LOG': str(log),
        'GPUWM_INSTALL_TEST_FAIL': failure,
        'GPUWM_INSTALL_TEST_COMPANION_GATE': '1' if companion_gate else '0',
        'GPUWM_INSTALL_TEST_DOCTOR_EXIT': str(doctor_exit),
        'GPUWM_INSTALL_TEST_SCRIPT': str(ROOT / ('install.ps1' if platform == 'powershell' else 'install.sh')),
    })
    if platform == 'powershell':
        harness = tmp_path / 'capture.ps1'
        harness.write_text(r'''
$ErrorActionPreference = 'Stop'
function global:git {
    & $env:GPUWM_INSTALL_TEST_PYTHON $env:GPUWM_INSTALL_TEST_BACKEND 'git' @args
    $global:LASTEXITCODE = $LASTEXITCODE
}
function global:cargo {
    & $env:GPUWM_INSTALL_TEST_PYTHON $env:GPUWM_INSTALL_TEST_BACKEND 'cargo' @args
    $global:LASTEXITCODE = $LASTEXITCODE
}
function global:.venv\Scripts\python.exe {
    & $env:GPUWM_INSTALL_TEST_PYTHON $env:GPUWM_INSTALL_TEST_BACKEND 'python' @args
    $global:LASTEXITCODE = $LASTEXITCODE
}
function global:.venv\Scripts\gpuwm.exe {
    & $env:GPUWM_INSTALL_TEST_PYTHON $env:GPUWM_INSTALL_TEST_BACKEND 'gpuwm' @args
    $global:LASTEXITCODE = $LASTEXITCODE
}
& $env:GPUWM_INSTALL_TEST_SCRIPT @args
exit $LASTEXITCODE
''', encoding='utf8')
        args = [shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(harness),
                '-Yes', '-NoFetchTables', '-Cuda', '13']
        if no_render:
            args.append('-NoRender')
    else:
        harness = tmp_path / 'capture.sh'
        harness.write_text('''#!/bin/sh
git() { "$GPUWM_INSTALL_TEST_PYTHON" "$GPUWM_INSTALL_TEST_BACKEND" git "$@"; }
cargo() { "$GPUWM_INSTALL_TEST_PYTHON" "$GPUWM_INSTALL_TEST_BACKEND" cargo "$@"; }
. "$GPUWM_INSTALL_TEST_SCRIPT"
''', encoding='utf8', newline='\n')
        env['GPUWM_INSTALL_TEST_SCRIPT'] = (ROOT / 'install.sh').as_posix()
        args = [shell, harness.as_posix(), '--yes', '--no-fetch-tables', '--cuda', '13']
        if no_render:
            args.append('--no-render')
    done = subprocess.run(args, cwd=stage, env=env, capture_output=True, text=True, timeout=60)
    rows = [json.loads(line) for line in log.read_text(encoding='utf8').splitlines()] if log.exists() else []
    return done, rows


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
@pytest.mark.parametrize('no_render', [False, True])
def test_clean_clone_installs_matching_companion_then_engine_and_builds_tui(tmp_path, platform, no_render):
    done, rows = _run(tmp_path, platform, no_render=no_render)
    assert done.returncode == 0, done.stdout + done.stderr
    installs = [row['args'] for row in rows if row['kind'] == 'python']
    assert installs == [
        ['-m', 'pip', 'install', '--upgrade', 'pip'],
        ['-m', 'pip', 'install', '-e', 'gpuwm-data'],
        ['-m', 'pip', 'install', '-e', '.[gpu-cu13,render]'],
    ]
    built = [Path(row['cwd']).name for row in rows if row['kind'] == 'cargo']
    assert built == ['grib1_bridge', *([] if no_render else ['rustwx']), 'arwen-tui']
    assert rows[0]['kind'] == 'git'
    assert rows[-1]['kind'] == 'gpuwm' and rows[-1]['args'] == ['doctor']


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_terminal_build_is_required_even_after_dependency_resolution_succeeds(tmp_path, platform):
    done, rows = _run(tmp_path, platform, companion_gate=False, no_render=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert any(row['kind'] == 'cargo' and Path(row['cwd']).name == 'arwen-tui' for row in rows)


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
@pytest.mark.parametrize('failure', ['companion', 'tui'])
def test_installer_stops_at_failed_owned_prerequisite(tmp_path, platform, failure):
    done, rows = _run(tmp_path, platform, failure=failure)
    assert done.returncode != 0
    assert not any(row['kind'] == 'gpuwm' for row in rows)
    if failure == 'companion':
        assert not any(any(arg.startswith('.[') for arg in row['args']) for row in rows)
        assert not any(row['kind'] == 'cargo' for row in rows)
    else:
        assert rows[-1]['kind'] == 'cargo' and Path(rows[-1]['cwd']).name == 'arwen-tui'


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_installer_preserves_doctor_exit_status(tmp_path, platform):
    done, rows = _run(tmp_path, platform, doctor_exit=23)
    assert done.returncode == 23, done.stdout + done.stderr
    assert rows[-1]['kind'] == 'gpuwm' and rows[-1]['args'] == ['doctor']

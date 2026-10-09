"""Execute checkout installers with captured external commands, without installs.

Both shells run the actual shipped script through the standalone clone route;
the PowerShell rows run on Windows, the only host install.ps1 installs on, and
elsewhere pwsh proves the script's refusal instead.
Only git, pip, cargo, doctor and the interpreters (python3.14t, python3/python,
uv and the uv installer) are substitutes; they enforce the dependency and
executable prerequisites a clean checkout needs, and inject failures.
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
    stream.write(json.dumps({'kind': kind, 'args': args, 'cwd': str(cwd),
                             'path': os.environ.get('PATH', ''),
                             'python_gil': os.environ.get('PYTHON_GIL'),
                             'uv_pref': os.environ.get('UV_PYTHON_PREFERENCE')}) + '\n')
def make_venv(venv, free_threaded):
    for name in ('bin', 'Scripts'):
        (venv / name).mkdir(parents=True, exist_ok=True)
    for name in ('python', 'gpuwm'):
        program = venv / 'bin' / name
        program.write_text('#!/bin/sh\nexec "$GPUWM_INSTALL_TEST_PYTHON" '
                           '"$GPUWM_INSTALL_TEST_BACKEND" ' + name + ' "$@"\n')
        program.chmod(0o755)
        (venv / 'Scripts' / (name + '.exe')).touch()
    # Whether the interpreter this .venv was made from is a free-threaded build.
    (venv / 'ft.txt').write_text('1' if free_threaded else '0')
    # A reused .venv that already holds CuPy builds: one line per registered distribution.
    (venv / 'cupy.txt').write_text(os.environ.get('GPUWM_INSTALL_TEST_CUPY', ''))


def uv_python():
    """The free-threaded interpreter `uv python find 3.14t` reports."""
    home = pathlib.Path(os.environ['GPUWM_INSTALL_TEST_UVHOME'])
    home.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        program = home / 'python3.14t.cmd'
        program.write_text('@"%GPUWM_INSTALL_TEST_PYTHON%" "%GPUWM_INSTALL_TEST_BACKEND%" py314t %*\r\n')
    else:
        program = home / 'python3.14t'
        program.write_text('#!/bin/sh\nexec "$GPUWM_INSTALL_TEST_PYTHON" '
                           '"$GPUWM_INSTALL_TEST_BACKEND" py314t "$@"\n')
        program.chmod(0o755)
    return program


def system_python():
    """A system python3.14t (deadsnakes' /usr/bin/python3.14t, say), which uv
    reports ahead of its own unless asked for a managed one."""
    home = pathlib.Path(os.environ['GPUWM_INSTALL_TEST_UVHOME']) / 'system'
    home.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        program = home / 'python3.14t.cmd'
        program.write_text('@set GPUWM_INSTALL_TEST_FROM_PATH=1\r\n'
                           '@"%GPUWM_INSTALL_TEST_PYTHON%" "%GPUWM_INSTALL_TEST_BACKEND%" py314t %*\r\n')
    else:
        program = home / 'python3.14t'
        program.write_text('#!/bin/sh\nGPUWM_INSTALL_TEST_FROM_PATH=1 exec "$GPUWM_INSTALL_TEST_PYTHON" '
                           '"$GPUWM_INSTALL_TEST_BACKEND" py314t "$@"\n')
        program.chmod(0o755)
    return program


if kind == 'git':
    assert args[:1] == ['clone'], args
    checkout = cwd / args[-1]
    for name in ('gpuwm', 'gpuwm-data', 'tools/grib1_bridge', 'tools/rustwx',
                 'tools/arwen-tui', 'tools/zarr_bridge', 'tools/rw_wps',
                 'tools/region_global_dealias'):
        (checkout / name).mkdir(parents=True, exist_ok=True)
    (checkout / 'pyproject.toml').write_text('# fixture checkout\n')
    if os.environ.get('GPUWM_INSTALL_TEST_FRESH') != '1':
        make_venv(checkout / '.venv', os.environ.get('GPUWM_INSTALL_TEST_FT') == '1')
if kind in ('py314t', 'py3'):
    if args[:1] == ['-c']:
        sys.exit(0 if kind == 'py314t' else 1)
    assert args[:2] == ['-m', 'venv'], args
    failure = os.environ.get('GPUWM_INSTALL_TEST_FAIL')
    if kind == 'py314t' and (failure == 'ft-venv' or (
            failure == 'path-ft-venv' and os.environ.get('GPUWM_INSTALL_TEST_FROM_PATH') == '1')):
        # As a distro python3.14t without its venv package (no ensurepip):
        # the interpreter is written into the new venv, then venv stops.
        make_venv(cwd / args[2], True)
        print('Error: Command [...] -m ensurepip [...] returned non-zero exit status 1.', file=sys.stderr)
        sys.exit(1)
    make_venv(cwd / args[2], kind == 'py314t')
if kind == 'python' and args[:1] == ['-c']:
    # The free-threading probe, answered for the interpreter the .venv was made from.
    sys.exit(0 if (cwd / '.venv/ft.txt').read_text() == '1' else 1)
if kind == 'uv':
    home = pathlib.Path(os.environ['GPUWM_INSTALL_TEST_UVHOME'])
    if args == ['python', 'find', '3.14t']:
        if (os.environ.get('GPUWM_INSTALL_TEST_UV_SEES_SYSTEM') == '1'
                and os.environ.get('UV_PYTHON_PREFERENCE') != 'only-managed'):
            print(system_python())
            sys.exit(0)
        if not (home / 'installed').exists():
            sys.exit(2)
        print(uv_python())
    elif args == ['python', 'install', '3.14t']:
        if os.environ.get('GPUWM_INSTALL_TEST_FAIL') == 'uv-install':
            sys.exit(3)
        # The installer keeps uv from dropping a python3.14t into the user's bin.
        assert os.environ.get('UV_PYTHON_INSTALL_BIN') == '0', 'UV_PYTHON_INSTALL_BIN'
        home.mkdir(parents=True, exist_ok=True)
        (home / 'installed').touch()
    else:
        raise AssertionError(args)
if kind == 'curl':
    # The uv installer, fetched to a file the installer then runs with sh.
    assert 'https://astral.sh/uv/install.sh' in args, args
    if os.environ.get('GPUWM_INSTALL_TEST_FAIL') == 'uv-download':
        sys.exit(6)
    target = pathlib.Path(args[args.index('-o') + 1])
    target.write_text('[ "$UV_NO_MODIFY_PATH" = 1 ] || exit 9\n'
                      'mkdir -p "$HOME/.local/bin"\n'
                      "printf '#!/bin/sh\\nexec \"$GPUWM_INSTALL_TEST_PYTHON\" "
                      "\"$GPUWM_INSTALL_TEST_BACKEND\" uv \"$@\"\\n' > \"$HOME/.local/bin/uv\"\n"
                      'chmod 755 "$HOME/.local/bin/uv"\n')
if kind == 'uv-installer':
    # install.ps1's child PowerShell running astral's install.ps1.
    assert os.environ.get('UV_NO_MODIFY_PATH') == '1'
    if os.environ.get('GPUWM_INSTALL_TEST_FAIL') == 'uv-download':
        sys.exit(6)
if kind == 'python' and args[:2] == ['-m', 'pip'] and os.environ.get('GPUWM_INSTALL_TEST_PIP_WARN'):
    # As real pip does beside a half-removed package (an uninstall that hit a locked DLL leaves ~upy_...).
    print('WARNING: Ignoring invalid distribution ~upy-cuda12x (fixture site-packages)', file=sys.stderr)
    sys.stderr.flush()
if kind == 'python' and args[:3] == ['-m', 'pip', 'list']:
    if os.environ.get('GPUWM_INSTALL_TEST_FAIL') == 'list':
        print('fixture pip: the environment could not be read', file=sys.stderr)
        sys.exit(2)
    for name in (cwd / '.venv/cupy.txt').read_text().split():
        print(f'{name}==14.2.0')
elif kind == 'python' and args[:4] == ['-m', 'pip', 'uninstall', '-y']:
    have = (cwd / '.venv/cupy.txt').read_text().split()
    (cwd / '.venv/cupy.txt').write_text(' '.join(n for n in have if n not in args[4:]))
elif kind == 'python' and args[:3] == ['-m', 'pip', 'install']:
    if 'gpuwm-data' in args:
        if os.environ.get('GPUWM_INSTALL_TEST_FAIL') == 'companion':
            sys.exit(17)
        assert (cwd / 'gpuwm-data').is_dir()
        (cwd / '.companion-installed').touch()
    elif any(arg.startswith('.[') for arg in args):
        if (os.environ.get('GPUWM_INSTALL_TEST_FAIL') == 'ft-pip'
                and (cwd / '.venv/ft.txt').read_text() == '1'):
            print('fixture pip: a dependency has no cp314t wheel and its sdist did not build', file=sys.stderr)
            sys.exit(31)
        if (os.environ.get('GPUWM_INSTALL_TEST_COMPANION_GATE', '1') == '1'
                and not (cwd / '.companion-installed').exists()):
            print('fixture pip: checkout gpuwm-data must satisfy the exact local pin first', file=sys.stderr)
            sys.exit(42)
        # As real pip does: the extra's CuPy is added beside any other major, never in its place.
        extra = next(arg for arg in args if arg.startswith('.['))
        wanted = 'cupy-cuda' + extra.split('gpu-cu', 1)[1][:2] + 'x'
        have = (cwd / '.venv/cupy.txt').read_text().split()
        if wanted not in have:
            (cwd / '.venv/cupy.txt').write_text(' '.join([*have, wanted]))
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


#: install.ps1 is the Windows installer (README: ``bash install.sh`` on Linux,
#: ``install.ps1`` on Windows).  Off Windows it refuses before touching anything
#: and names install.sh, which test_install_ps1_refuses_off_windows_and_names_install_sh
#: proves under pwsh.  Its install-flow rows assert Windows steps
#: (.venv\Scripts\python.exe, %USERPROFILE%\.cargo, ';'-joined Path); run under
#: the ubuntu-24.04 runner's pwsh they reported twelve failures of a route the
#: product does not offer there and now refuses by design.
POWERSHELL_FLOW_OFF_WINDOWS = (
    'install.ps1 refuses on a non-Windows host and points at install.sh; its Windows '
    'install-flow assertions (.venv\\Scripts, USERPROFILE, ;-joined Path) would fail here '
    'against a route the product refuses by design.  The refusal itself is covered by '
    'test_install_ps1_refuses_off_windows_and_names_install_sh')


def _shell(platform):
    if platform == 'powershell':
        if os.name != 'nt':
            pytest.skip(POWERSHELL_FLOW_OFF_WINDOWS)
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


def _run(tmp_path, platform, *, no_render=False, failure='', companion_gate=True, doctor_exit=0, cupy=(),
         cuda='13', pip_warns=False, fresh=False, venv_ft=False, has_314t=False, has_uv=False,
         uv_has_python=False, uv_sees_system=False, yes=True, extra=()):
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
        'GPUWM_INSTALL_TEST_CUPY': ' '.join(cupy),
        'GPUWM_INSTALL_TEST_PIP_WARN': '1' if pip_warns else '',
        'GPUWM_INSTALL_TEST_SCRIPT': str(ROOT / ('install.ps1' if platform == 'powershell' else 'install.sh')),
        'GPUWM_INSTALL_TEST_FRESH': '1' if fresh else '',
        'GPUWM_INSTALL_TEST_FT': '1' if venv_ft else '',
        'GPUWM_INSTALL_TEST_HAS_314T': '1' if has_314t else '',
        'GPUWM_INSTALL_TEST_HAS_UV': '1' if has_uv else '',
        'GPUWM_INSTALL_TEST_UVHOME': str(tmp_path / 'uv-pythons'),
        'GPUWM_INSTALL_TEST_UV_SEES_SYSTEM': '1' if uv_sees_system else '',
    })
    if uv_has_python:
        (tmp_path / 'uv-pythons').mkdir()
        (tmp_path / 'uv-pythons' / 'installed').touch()
    if fresh:
        # Hermetic interpreters: no uv or python3.14t of the test host's own may
        # answer, so the home and the search path hold only system tools.
        home = tmp_path / 'home'
        home.mkdir()
        env['HOME'] = str(home)
        env['USERPROFILE'] = str(home)
        if os.name == 'nt':
            system = os.environ.get('SystemRoot', r'C:\Windows')
            env['PATH'] = os.pathsep.join([system + r'\System32', system,
                                           system + r'\System32\WindowsPowerShell\v1.0'])
        else:
            env['PATH'] = '/usr/bin:/bin'
        env.pop('PYTHON_GIL', None)
    if has_314t and platform != 'powershell':
        # A file, not a shell function: dash refuses a function named python3.14t.
        found = tmp_path / 'free-threaded-bin'
        found.mkdir()
        program = found / 'python3.14t'
        program.write_text('#!/bin/sh\nGPUWM_INSTALL_TEST_FROM_PATH=1 exec "$GPUWM_INSTALL_TEST_PYTHON" '
                           '"$GPUWM_INSTALL_TEST_BACKEND" py314t "$@"\n',
                           encoding='utf8', newline='\n')
        program.chmod(0o755)
        env['PATH'] = str(found) + os.pathsep + env['PATH']
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
function global:Set-UvStub {
    function global:uv {
        & $env:GPUWM_INSTALL_TEST_PYTHON $env:GPUWM_INSTALL_TEST_BACKEND 'uv' @args
        $global:LASTEXITCODE = $LASTEXITCODE
    }
}
if ($env:GPUWM_INSTALL_TEST_FRESH -eq '1') {
    function global:python {
        & $env:GPUWM_INSTALL_TEST_PYTHON $env:GPUWM_INSTALL_TEST_BACKEND 'py3' @args
        $global:LASTEXITCODE = $LASTEXITCODE
    }
    function global:powershell {
        & $env:GPUWM_INSTALL_TEST_PYTHON $env:GPUWM_INSTALL_TEST_BACKEND 'uv-installer' @args
        $global:LASTEXITCODE = $LASTEXITCODE
        if ($LASTEXITCODE -eq 0) { Set-UvStub }
    }
}
if ($env:GPUWM_INSTALL_TEST_HAS_314T -eq '1') {
    function global:python3.14t {
        $env:GPUWM_INSTALL_TEST_FROM_PATH = '1'
        & $env:GPUWM_INSTALL_TEST_PYTHON $env:GPUWM_INSTALL_TEST_BACKEND 'py314t' @args
        $global:LASTEXITCODE = $LASTEXITCODE
        $env:GPUWM_INSTALL_TEST_FROM_PATH = $null
    }
}
if ($env:GPUWM_INSTALL_TEST_HAS_UV -eq '1') { Set-UvStub }
& $env:GPUWM_INSTALL_TEST_SCRIPT @args
$code = $LASTEXITCODE
# The caller's session Path as the script left it (the piped form runs in that session).
[IO.File]::WriteAllText($env:GPUWM_INSTALL_TEST_AFTER_PATH, $env:Path)
exit $code
''', encoding='utf8')
        args = [shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(harness),
                *(['-Yes'] if yes else []), '-NoFetchTables', '-Cuda', cuda, *extra]
        if no_render:
            args.append('-NoRender')
    else:
        harness = tmp_path / 'capture.sh'
        harness.write_text('''#!/bin/sh
git() { "$GPUWM_INSTALL_TEST_PYTHON" "$GPUWM_INSTALL_TEST_BACKEND" git "$@"; }
cargo() { "$GPUWM_INSTALL_TEST_PYTHON" "$GPUWM_INSTALL_TEST_BACKEND" cargo "$@"; }
curl() { "$GPUWM_INSTALL_TEST_PYTHON" "$GPUWM_INSTALL_TEST_BACKEND" curl "$@"; }
if [ "$GPUWM_INSTALL_TEST_FRESH" = 1 ]; then
    python3() { "$GPUWM_INSTALL_TEST_PYTHON" "$GPUWM_INSTALL_TEST_BACKEND" py3 "$@"; }
fi
if [ "$GPUWM_INSTALL_TEST_HAS_UV" = 1 ]; then
    uv() { "$GPUWM_INSTALL_TEST_PYTHON" "$GPUWM_INSTALL_TEST_BACKEND" uv "$@"; }
fi
. "$GPUWM_INSTALL_TEST_SCRIPT"
''', encoding='utf8', newline='\n')
        env['GPUWM_INSTALL_TEST_SCRIPT'] = (ROOT / 'install.sh').as_posix()
        args = [shell, harness.as_posix(), *(['--yes'] if yes else []), '--no-fetch-tables', '--cuda', cuda,
                *extra]
        if no_render:
            args.append('--no-render')
    after_path = tmp_path / 'path-after.txt'
    env['GPUWM_INSTALL_TEST_AFTER_PATH'] = str(after_path)
    # No terminal and no input: a consent prompt must read as "no", never block.
    done = subprocess.run(args, cwd=stage, env=env, capture_output=True, text=True, timeout=60,
                          stdin=subprocess.DEVNULL, start_new_session=os.name != 'nt')
    rows = [json.loads(line) for line in log.read_text(encoding='utf8').splitlines()] if log.exists() else []
    held = stage / 'gpuwm/.venv/cupy.txt'
    done.cupy = held.read_text().split() if held.exists() else None
    done.path_after = after_path.read_text(encoding='utf8') if after_path.exists() else None
    return done, rows


def _pip_rows(rows):
    """The .venv's pip invocations, without the free-threading probes."""
    return [row['args'] for row in rows if row['kind'] == 'python' and row['args'][:2] == ['-m', 'pip']]


def _required_workspaces(no_render=False):
    """Every cargo workspace that builds an artifact a bundle carries."""
    from gpuwm.bridge_assets import BUNDLED_ARTIFACTS
    required = {Path(artifact.crate).name for artifact in BUNDLED_ARTIFACTS}
    if no_render:
        required.discard('rustwx')
    return required


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
@pytest.mark.parametrize('no_render', [False, True])
def test_clean_clone_installs_matching_companion_then_engine_and_builds_tui(tmp_path, platform, no_render):
    done, rows = _run(tmp_path, platform, no_render=no_render)
    assert done.returncode == 0, done.stdout + done.stderr
    installs = _pip_rows(rows)
    assert installs == [
        ['-m', 'pip', 'install', '--upgrade', 'pip'],
        ['-m', 'pip', 'list', '--format=freeze', '--disable-pip-version-check'],
        ['-m', 'pip', 'install', '-e', 'gpuwm-data'],
        ['-m', 'pip', 'install', '-e', '.[gpu-cu13,render]'],
    ]
    built = [Path(row['cwd']).name for row in rows if row['kind'] == 'cargo']
    assert built == ['grib1_bridge', *([] if no_render else ['rustwx']), 'arwen-tui', 'zarr_bridge',
                     'rw_wps', 'region_global_dealias']
    # A workspace the bundle roster builds from and the installer skips leaves
    # doctor reporting its default route MISSING on every source install.
    assert _required_workspaces(no_render) <= set(built)
    assert rows[0]['kind'] == 'git'
    assert rows[-1]['kind'] == 'gpuwm' and rows[-1]['args'] == ['doctor']


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_installer_doctor_sees_the_environment_it_just_installed(tmp_path, platform):
    # Doctor's console-script check reported the installer's own .venv as off PATH.
    done, rows = _run(tmp_path, platform)
    assert done.returncode == 0, done.stdout + done.stderr
    doctor = rows[-1]
    assert doctor['kind'] == 'gpuwm' and doctor['args'] == ['doctor']
    scripts = Path(doctor['cwd']) / '.venv' / ('Scripts' if platform == 'powershell' else 'bin')
    entries = [entry for entry in doctor['path'].split(os.pathsep) if entry]
    assert any(Path(entry).resolve() == scripts.resolve() for entry in entries), entries[:3]
    if platform == 'powershell':
        # The piped form runs in the caller's session: its Path comes back as it was.
        assert done.path_after is not None
        left = [entry for entry in done.path_after.split(os.pathsep) if entry]
        assert not any(Path(entry).resolve() == scripts.resolve() for entry in left)


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


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
@pytest.mark.parametrize('had, cuda', [(['cupy-cuda12x'], '13'), (['cupy-cuda13x'], '12'),
                                       (['cupy-cuda12x', 'cupy-cuda13x'], '13')])
def test_switching_cuda_major_leaves_exactly_one_cupy(tmp_path, platform, had, cuda):
    done, rows = _run(tmp_path, platform, cupy=had, cuda=cuda)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.cupy == [f'cupy-cuda{cuda}x']
    uninstalls = [args[4:] for args in _pip_rows(rows) if args[2:3] == ['uninstall']]
    # Every CuPy goes, the chosen one too: the two shared one set of files, so neither is intact.
    assert uninstalls == [had]
    order = [args[2] for args in _pip_rows(rows)]
    assert order.index('uninstall') < order.index('install', 2)


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_rerunning_with_the_same_major_uninstalls_nothing(tmp_path, platform):
    done, rows = _run(tmp_path, platform, cupy=['cupy-cuda13x'], cuda='13')
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.cupy == ['cupy-cuda13x']
    assert not any(row['kind'] == 'python' and row['args'][2:3] == ['uninstall'] for row in rows)


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_a_pip_warning_on_stderr_still_leaves_exactly_one_cupy(tmp_path, platform):
    # Windows PowerShell 5.1 under 'Stop' threw on pip's first stderr line and read that as no CuPy.
    done, rows = _run(tmp_path, platform, cupy=['cupy-cuda12x', 'cupy-cuda13x'], cuda='13', pip_warns=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.cupy == ['cupy-cuda13x']
    uninstalls = [args[4:] for args in _pip_rows(rows) if args[2:3] == ['uninstall']]
    assert uninstalls == [['cupy-cuda12x', 'cupy-cuda13x']]


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_a_pip_that_cannot_list_the_venv_stops_the_install(tmp_path, platform):
    done, rows = _run(tmp_path, platform, cupy=['cupy-cuda12x'], cuda='13', failure='list')
    assert done.returncode != 0
    assert 'could not list the packages' in done.stdout + done.stderr
    # Nothing is installed over a CuPy nobody could see.
    assert not any(any(arg.startswith('.[') for arg in row['args']) for row in rows)
    assert done.cupy == ['cupy-cuda12x']


# ---------------------------------------------------------------------------
# The interpreter: free-threaded CPython 3.14t by default (2.8.8)
#
# THE BREAKAGE THESE GUARD: every [devices] rank of a multi-card forecast
# steps from its own Python thread, and under an interpreter lock the threads
# take turns (4 cards: 289 s per forecast hour with the lock, 130 s on 3.14t,
# measured 2026-10-03).  Through 2.8.7 both installers made .venv from
# whatever python3/python was on PATH, so a plain install ran multi-card
# forecasts about 2x slower than every published benchmark.
# ---------------------------------------------------------------------------

def _python_flag(platform, value):
    return ['-Python', value] if platform == 'powershell' else ['--python', value]


def _locked_name(platform):
    return 'python' if platform == 'powershell' else 'python3'


def _venv_makers(rows):
    """Which base interpreter made each .venv this run, in order."""
    return [row['kind'] for row in rows
            if row['kind'] in ('py314t', 'py3') and row['args'][:2] == ['-m', 'venv']]


def _engine_install(rows):
    return next(args for args in _pip_rows(rows) if any(arg.startswith('.[') for arg in args))


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_a_fresh_install_makes_its_venv_on_the_python314t_already_on_path(tmp_path, platform):
    done, rows = _run(tmp_path, platform, fresh=True, has_314t=True)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert _venv_makers(rows) == ['py314t']
    assert '.venv is free-threaded' in output
    assert 'WARNING' not in output
    # cftime 1.6.6 has no cp314t wheel; 1.6.5 does, and only --prefer-binary picks it.
    assert _engine_install(rows) == ['-m', 'pip', 'install', '--prefer-binary', '-e', '.[gpu-cu13,render]']
    # The engine re-runs itself with PYTHON_GIL=0; the installer never exports it,
    # because a Python with the lock refuses to start when it sees PYTHON_GIL=0.
    assert all(row.get('python_gil') is None for row in rows)
    assert not any(row['kind'] in ('uv', 'curl', 'uv-installer') for row in rows)


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
@pytest.mark.parametrize('uv_has_python', [True, False])
def test_without_python314t_on_path_uv_finds_or_installs_it(tmp_path, platform, uv_has_python):
    done, rows = _run(tmp_path, platform, fresh=True, has_uv=True, uv_has_python=uv_has_python)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    asked = [row['args'] for row in rows if row['kind'] == 'uv']
    if uv_has_python:
        assert asked == [['python', 'find', '3.14t']]
    else:
        assert asked == [['python', 'find', '3.14t'], ['python', 'install', '3.14t'],
                         ['python', 'find', '3.14t']]
    assert _venv_makers(rows) == ['py314t']
    assert 'WARNING' not in output
    assert not any(row['kind'] in ('curl', 'uv-installer') for row in rows)


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_with_consent_and_no_uv_the_installer_fetches_uv_then_python314t(tmp_path, platform):
    done, rows = _run(tmp_path, platform, fresh=True)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    kinds = [row['kind'] for row in rows]
    fetched = 'uv-installer' if platform == 'powershell' else 'curl'
    assert fetched in kinds
    assert kinds.index(fetched) < kinds.index('uv')
    assert [row['args'] for row in rows if row['kind'] == 'uv'][-1] == ['python', 'find', '3.14t']
    assert _venv_makers(rows) == ['py314t']
    assert 'WARNING' not in output


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_without_consent_the_install_goes_on_with_the_lock_and_names_the_cost(tmp_path, platform):
    done, rows = _run(tmp_path, platform, fresh=True, yes=False)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert not any(row['kind'] in ('curl', 'uv-installer', 'uv') for row in rows)
    assert _venv_makers(rows) == ['py3']
    # Gate law: the fallback names what it costs and how to get out of it.
    assert 'WARNING: .venv runs on ' + _locked_name(platform) in output
    assert 'consent to install uv was not given' in output
    assert 'about 2x slower' in output
    assert 'install uv' in output
    # A locked .venv installs exactly as it did before.
    assert _engine_install(rows) == ['-m', 'pip', 'install', '-e', '.[gpu-cu13,render]']


def test_a_failed_uv_download_falls_back_with_its_reason(tmp_path):
    done, rows = _run(tmp_path, 'posix', fresh=True, failure='uv-download')
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert _venv_makers(rows) == ['py3']
    assert 'could not be downloaded' in output
    assert 'about 2x slower' in output


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_a_failed_python314t_install_falls_back_with_its_reason(tmp_path, platform):
    done, rows = _run(tmp_path, platform, fresh=True, has_uv=True, failure='uv-install')
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert _venv_makers(rows) == ['py3']
    assert 'uv could not install Python 3.14t' in output
    assert 'about 2x slower' in output


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_an_interpreter_the_user_names_is_used_and_never_replaced(tmp_path, platform):
    done, rows = _run(tmp_path, platform, fresh=True, has_314t=True,
                      extra=_python_flag(platform, _locked_name(platform)))
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert _venv_makers(rows) == ['py3']
    assert 'GPUWM_PYTHON' in output and 'about 2x slower' in output


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_a_dependency_that_will_not_install_under_314t_remakes_the_venv_on_the_locked_python(tmp_path, platform):
    done, rows = _run(tmp_path, platform, fresh=True, has_314t=True, failure='ft-pip')
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    # Made on 3.14t, the engine install failed there, remade on the locked Python.
    assert _venv_makers(rows) == ['py314t', 'py3']
    engine = [args for args in _pip_rows(rows) if any(arg.startswith('.[') for arg in args)]
    assert engine == [['-m', 'pip', 'install', '--prefer-binary', '-e', '.[gpu-cu13,render]'],
                      ['-m', 'pip', 'install', '-e', '.[gpu-cu13,render]']]
    assert 'did not install under 3.14t' in output
    assert 'about 2x slower' in output
    assert rows[-1]['kind'] == 'gpuwm' and rows[-1]['args'] == ['doctor']


# A python3.14t that cannot make a venv (a distro build without its venv
# package, deadsnakes without python3.14-venv for one, has no ensurepip):
# under `set -eu` the venv step ended the whole install, and the half-made
# .venv it left was "reused" by the next run, which stopped at the pip
# upgrade.

@pytest.mark.parametrize('platform', ['posix', 'powershell'])
@pytest.mark.parametrize('uv_has_python', [True, False])
def test_a_python314t_on_path_that_cannot_make_a_venv_gives_way_to_the_one_uv_provides(
        tmp_path, platform, uv_has_python):
    # As uv does: asked plainly it answers with the system python3.14t (the
    # one that just failed), so the retry asks for a managed interpreter.
    done, rows = _run(tmp_path, platform, fresh=True, has_314t=True, has_uv=True, uv_has_python=uv_has_python,
                      uv_sees_system=True, failure='path-ft-venv')
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert _venv_makers(rows) == ['py314t', 'py314t']
    asked = [(row['args'], row['uv_pref']) for row in rows if row['kind'] == 'uv']
    managed = 'only-managed'
    if uv_has_python:
        assert asked == [(['python', 'find', '3.14t'], managed)]
    else:
        assert asked == [(['python', 'find', '3.14t'], managed), (['python', 'install', '3.14t'], managed),
                         (['python', 'find', '3.14t'], managed)]
    # The preference is scoped to the uv lookup: the venv, pip and builds never see it.
    assert all(row['uv_pref'] is None for row in rows
               if row['kind'] in ('python', 'gpuwm', 'cargo') or row['args'][:2] == ['-m', 'venv'])
    assert '.venv is free-threaded' in output
    assert 'WARNING' not in output
    assert _engine_install(rows) == ['-m', 'pip', 'install', '--prefer-binary', '-e', '.[gpu-cu13,render]']


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_a_python314t_that_cannot_make_a_venv_falls_back_with_the_reason_named(tmp_path, platform):
    done, rows = _run(tmp_path, platform, fresh=True, has_314t=True, yes=False, failure='ft-venv')
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    # The failed attempt, then python3/python; no uv without consent.
    assert _venv_makers(rows) == ['py314t', 'py3']
    assert not any(row['kind'] in ('curl', 'uv-installer', 'uv') for row in rows)
    assert 'WARNING: .venv runs on ' + _locked_name(platform) in output
    assert 'python3.14t could not create a venv' in output
    assert 'no uv was found and consent to install uv was not given' in output
    # A python3.14t was found; the reason must not say otherwise.
    assert 'no python3.14t or uv was found' not in output
    assert 'about 2x slower' in output
    # The remedy is the venv support that Python lacks, not "install a 3.14t".
    assert "that Python" in output.split('To fix:', 1)[1]
    # The locked .venv installs exactly as before and the install finishes.
    assert _engine_install(rows) == ['-m', 'pip', 'install', '-e', '.[gpu-cu13,render]']
    assert rows[-1]['kind'] == 'gpuwm' and rows[-1]['args'] == ['doctor']


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_when_every_python314t_fails_its_venv_the_install_still_finishes_on_the_locked_python(tmp_path, platform):
    done, rows = _run(tmp_path, platform, fresh=True, has_314t=True, failure='ft-venv')
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    # PATH python3.14t, then the 3.14t uv installs (with --yes consent), then python3/python.
    assert _venv_makers(rows) == ['py314t', 'py314t', 'py3']
    assert 'neither could' in output
    assert 'about 2x slower' in output
    assert _engine_install(rows) == ['-m', 'pip', 'install', '-e', '.[gpu-cu13,render]']


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_a_named_interpreter_that_cannot_make_a_venv_stops_and_leaves_no_half_venv(tmp_path, platform):
    done, rows = _run(tmp_path, platform, fresh=True, has_314t=True, failure='ft-venv',
                      extra=_python_flag(platform, 'python3.14t'))
    output = done.stdout + done.stderr
    assert done.returncode != 0, output
    assert _venv_makers(rows) == ['py314t']
    assert 'could not create .venv' in output
    # A half-made .venv would be "reused" by the next run and stop at its pip upgrade.
    assert not (tmp_path / 'new checkout with spaces' / 'gpuwm' / '.venv').exists()
    assert not _pip_rows(rows)


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_a_reused_locked_venv_is_kept_and_named(tmp_path, platform):
    done, rows = _run(tmp_path, platform)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert _venv_makers(rows) == []
    assert "WARNING: .venv runs on the existing .venv's Python" in output
    assert 'remove .venv' in output


@pytest.mark.parametrize('platform', ['posix', 'powershell'])
def test_a_reused_free_threaded_venv_installs_with_prefer_binary(tmp_path, platform):
    done, rows = _run(tmp_path, platform, venv_ft=True)
    output = done.stdout + done.stderr
    assert done.returncode == 0, output
    assert 'the existing .venv is free-threaded' in output
    assert 'WARNING' not in output
    assert _engine_install(rows) == ['-m', 'pip', 'install', '--prefer-binary', '-e', '.[gpu-cu13,render]']


@pytest.mark.parametrize('form', ['file', 'piped'])
def test_install_ps1_refuses_off_windows_and_names_install_sh(tmp_path, form):
    """Under pwsh on Linux install.ps1 ran its Windows steps: it cloned, sent
    pip to a .venv\\Scripts\\python.exe that xdg-open could not open, and died
    at Join-Path on the unset USERPROFILE.  Off Windows it now refuses first,
    in both the file form and the piped (iwr | iex) form, changes nothing,
    and names the installer that host has: install.sh."""
    if os.name == 'nt':
        pytest.skip('install.ps1 runs its install on Windows (the powershell rows above cover it); '
                    'asserting the off-Windows refusal here would fail on the host the script supports')
    pwsh = shutil.which('pwsh')
    if not pwsh:
        pytest.skip('pwsh is not installed on this non-Windows host, so install.ps1 cannot be '
                    'started here and there is no refusal to observe')
    stage = tmp_path / 'new checkout with spaces'
    stage.mkdir()
    # A git that records being called: the refusal must come before the clone,
    # and a broken guard must not reach the network.
    shims = tmp_path / 'shims'
    shims.mkdir()
    called = tmp_path / 'git-called'
    git = shims / 'git'
    git.write_text(f'#!/bin/sh\necho "$@" >> "{called.as_posix()}"\nexit 1\n', encoding='utf8')
    git.chmod(0o755)
    env = dict(os.environ)
    env.update({'PATH': str(shims) + os.pathsep + env.get('PATH', ''),
                'GPUWM_REPO_URL': 'https://example.invalid/local-source-fixture',
                'GPUWM_INSTALL_TEST_SCRIPT': str(ROOT / 'install.ps1')})
    if form == 'file':
        args = [pwsh, '-NoProfile', '-File', str(ROOT / 'install.ps1'), '-Yes', '-NoFetchTables']
    else:
        # The catch reports the thrown message unwrapped; status 3 proves the
        # piped form threw rather than calling `exit`.
        args = [pwsh, '-NoProfile', '-Command',
                'try { Get-Content -Raw -LiteralPath $env:GPUWM_INSTALL_TEST_SCRIPT | Invoke-Expression } '
                'catch { [Console]::Out.WriteLine($_.Exception.Message); exit 3 }; exit 0']
    done = subprocess.run(args, cwd=stage, env=env, capture_output=True, text=True, timeout=60)
    output = done.stdout + done.stderr
    if form == 'file':
        # A file invocation exits with the refusal's own status.
        assert done.returncode == 2, output
        message = done.stderr
    else:
        # The piped form throws instead: `exit` would close the caller's console.
        assert done.returncode == 3, output
        message = done.stdout
    assert 'is the Windows installer' in message, output
    assert 'bash install.sh' in message, output
    assert not called.exists(), called.read_text()
    assert list(stage.iterdir()) == []


# ---------------------------------------------------------------------------
# The documented manual installs build what the installers build
# ---------------------------------------------------------------------------

def _install_pages():
    """Pages a reader installs from.  The ``*-runbook.md`` files in ``docs/``
    record one campaign's box and are not held to the current install."""
    pages = [ROOT / 'README.md', ROOT / 'CONTRIBUTING.md']
    pages += sorted(page for page in (ROOT / 'docs').glob('*.md') if not page.name.endswith('-runbook.md'))
    pages += sorted((ROOT / 'docs' / 'public').glob('*.md'))
    pages += sorted((ROOT / 'docs' / 'manual').glob('*.md'))
    return [page for page in pages if page.is_file()]


def _page_units(text):
    """Fenced blocks and prose paragraphs, each as (kind, text)."""
    import re
    units = [('block', body) for body in re.findall(r'```[^\n]*\n(.*?)```', text, flags=re.S)]
    prose = re.sub(r'```.*?```', '\n\n', text, flags=re.S)
    units += [('prose', ' '.join(line.strip() for line in part.splitlines()))
              for part in re.split(r'\n\s*\n', prose) if part.strip()]
    return units


def _named_workspaces(text):
    import re
    return {name for name in re.findall(r'tools[/\\]([A-Za-z0-9_-]+)', text)}


_ENGINE_EDITABLE = r"pip install\s+(?:--?\S+\s+)*-e\s+['\"]?\.\["
_COMPANION_EDITABLE = r"pip install\s+(?:--?\S+\s+)*-e\s+['\"]?gpuwm-data\b"


def test_every_documented_source_install_puts_the_checkout_companion_first():
    """The engine requires the companion of its own version, which PyPI does
    not carry before that version is published: an editable engine install
    with no `pip install -e gpuwm-data` ahead of it cannot resolve on a
    checkout, as FIRST-LIGHT's manual steps and CONTRIBUTING's did.

    Each page installs the companion before its first engine install, and
    each block that creates a fresh environment does so on its own, because
    a reader follows the POSIX or the PowerShell block, not both.  A later
    `pip install -e '.[dev]'` into the environment the page already built
    resolves against the companion installed there."""
    import re
    wrong, seen = [], set()
    for page in _install_pages():
        name = page.relative_to(ROOT).as_posix()
        whole = page.read_text(encoding='utf8')
        engine = re.search(_ENGINE_EDITABLE, whole)
        if engine is None:
            continue
        seen.add(name)
        companion = re.search(_COMPANION_EDITABLE, whole)
        if companion is None or companion.start() > engine.start():
            wrong.append(f'{name}: first engine install at offset {engine.start()} has no companion ahead of it')
        for kind, text in _page_units(whole):
            engine = re.search(_ENGINE_EDITABLE, text)
            if engine is None or 'venv' not in text:
                continue
            companion = re.search(_COMPANION_EDITABLE, text)
            if companion is None or companion.start() > engine.start():
                wrong.append(f'{name} ({kind}): {text.strip()[:160]}')
    assert {'docs/install.md', 'docs/public/FIRST-LIGHT.md', 'CONTRIBUTING.md'} <= seen, seen
    assert not wrong, '\n'.join(wrong)


def test_every_documented_manual_build_covers_the_bundle_roster():
    """A manual install or rebuild block that skips a workspace the bundle
    roster builds from leaves doctor reporting that default route MISSING;
    FIRST-LIGHT's manual steps built two of the six and ended at "MISSING
    mapped decode engine" and "MISSING region-global dealiasing engine"."""
    import re
    roster = _required_workspaces()
    wrong, checked = [], 0
    for page in _install_pages():
        for kind, text in _page_units(page.read_text(encoding='utf8')):
            if kind != 'block':
                continue
            # A build of one named binary (`--bin`) serves its own page's
            # route; a whole-workspace build is an install or a rebuild.
            lines = text.replace('\\\n', ' ').replace('`\n', ' ').splitlines()
            if not any('cargo build' in line and '--bin' not in line for line in lines):
                continue
            named = _named_workspaces(text) & roster
            if not (re.search(_ENGINE_EDITABLE, text) or len(named) >= 2):
                continue
            checked += 1
            if roster - named:
                wrong.append(f'{page.relative_to(ROOT).as_posix()}: missing {sorted(roster - named)}')
    assert checked >= 4, checked
    assert not wrong, '\n'.join(wrong)


@pytest.mark.parametrize('page, anchor', [
    ('docs/install.md', 'perform the whole developer install'),
    ('docs/public/FIRST-LIGHT.md', 'One command does all of it'),
    ('docs/public/CLI-USER-MANUAL.md', "The scripts install the checkout's matching"),
    ('CONTRIBUTING.md', "install the checkout's companion first"),
])
def test_every_description_of_the_source_install_names_the_whole_roster(page, anchor):
    """The prose that says what a source install builds names every
    workspace the installers build.  CLI-USER-MANUAL said "the vendored
    GRIB, renderer, and terminal workspaces" and FIRST-LIGHT named two."""
    units = [text for kind, text in _page_units((ROOT / page).read_text(encoding='utf8'))
             if kind == 'prose' and anchor in text]
    assert len(units) == 1, f'{page}: the install description is gone or duplicated'
    missing = _required_workspaces() - _named_workspaces(units[0])
    assert not missing, f'{page} does not name {sorted(missing)}'


def _rust_minimum():
    """The newest `rust-version` any package in a roster workspace declares,
    vendored crates included: cargo refuses the build below any of them."""
    import re
    newest = (0,)
    for name in _required_workspaces():
        for directory, subdirs, files in os.walk(ROOT / 'tools' / name):
            subdirs[:] = [sub for sub in subdirs if sub not in ('target', '.git')]
            if 'Cargo.toml' not in files:
                continue
            text = Path(directory, 'Cargo.toml').read_text(encoding='utf8', errors='replace')
            for found in re.findall(r'^rust-version\s*=\s*"([0-9.]+)"', text, flags=re.M):
                newest = max(newest, tuple(int(part) for part in found.split('.')))
    return newest


def test_the_install_pages_state_the_rust_the_workspaces_require():
    """The manual steps named no Rust version, and a box whose `cargo` on
    PATH was its distribution's 1.93 built two workspaces and then stopped
    at the terminal: "rustc 1.93.1 is not supported by the following
    packages: arwen-tui@2.8.0 requires rustc 1.94".  Each install page
    states the minimum, and it is the one cargo enforces."""
    import re
    minimum = _rust_minimum()
    assert minimum >= (1, 94), minimum
    wanted = '.'.join(str(part) for part in minimum[:2])
    for page in ('docs/install.md', 'docs/public/FIRST-LIGHT.md'):
        text = ' '.join((ROOT / page).read_text(encoding='utf8').split())
        assert f'Rust {wanted} or newer' in text, page
    for page in _install_pages():
        text = ' '.join(page.read_text(encoding='utf8').split())
        for stated in re.findall(r'Rust (\d+\.\d+) or newer', text):
            assert stated == wanted, (page.relative_to(ROOT).as_posix(), stated, wanted)

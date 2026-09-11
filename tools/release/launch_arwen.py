"""Start the packaged ArWen GUI and terminal with their own Python runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

try:
    import termios
except ImportError:  # Windows has no POSIX terminal to put back.
    termios = None


def cache_directory(state: Path) -> Path:
    """Honor the GUI preference stored within this launcher's selected profile."""
    preference = state / 'appdata/ArWenCompanion/preferences.json'
    try:
        value = json.loads(preference.read_text(encoding='utf-8'))
        folder = value.get('data_folder') if isinstance(value, dict) else None
        if isinstance(folder, str) and folder.strip() and Path(folder).is_absolute():
            return Path(folder) / 'cache'
    except (OSError, ValueError):
        pass
    return state / 'cache'


def child_environment(state: Path, python: Path, cds_credentials: Path | None = None,
                      parent: dict[str, str] | None = None) -> dict[str, str]:
    """The environment every engine, terminal and GUI subprocess inherits.

    The package must stay byte-identical after use. Python writes a compiled
    ``__pycache__`` next to every module it imports unless told not to, and
    one ``doctor`` run from the packaged runtime wrote 554 such files
    (134 MB) into ``runtime/Lib/site-packages``; the launcher's own ``-B``
    covers only the launcher process, so the flag travels in the environment
    to every child instead.
    """
    source = os.environ if parent is None else parent
    env = {key: value for key, value in source.items()
           if not key.upper().startswith(('PYTHON', 'GPUWM_', 'ARWEN_')) and key.upper() != 'NO_COLOR'}
    env.update(PYTHONNOUSERSITE='1', PYTHONSAFEPATH='1', PYTHONUTF8='1',
               PYTHONDONTWRITEBYTECODE='1',
               TERM='xterm-256color', COLORTERM='truecolor',
               ARWEN_CACHE_DIR=str(cache_directory(state)), LOCALAPPDATA=str(state / 'appdata'))
    if cds_credentials is not None:
        if not cds_credentials.is_file():
            raise ValueError('The configured CDS credentials file is unavailable.')
        env['CDSAPI_RC'] = str(cds_credentials)
    env['PATH'] = str(python.parent) + os.pathsep + env.get('PATH', '')
    return env


# The bytes the controller itself writes on the way out, in its order: SGR,
# urxvt, any-event, button-event and normal mouse reporting off, bracketed
# paste off, leave the alternate screen, show the cursor.  Kept identical to
# ``TERMINAL_RESTORE`` in ``tools/arwen-tui/src/main.rs``.
TERMINAL_RESTORE = ('\x1b[?1006l\x1b[?1015l\x1b[?1003l\x1b[?1002l'
                    '\x1b[?1000l\x1b[?2004l\x1b[?1049l\x1b[?25h')


def terminal_mode():
    """The terminal's attributes before the controller takes it, or None.

    THE BREAKAGE THIS PREVENTS: the controller puts the terminal into raw
    mode, the alternate screen and mouse reporting, and undoes all three
    itself on every exit it can see -- but a SIGKILL or an OOM kill runs no
    code in the process at all.  A user whose controller was killed during a
    fetch was left with a shell that still reported mouse motion, so every
    mouse move became a run of ``ESC[<35;..M`` "command not found" lines, on
    the alternate screen, with no cursor.  The launcher outlives the
    controller, so it is the one process that can always put the terminal
    back.
    """
    if termios is None or os.name == 'nt' or not sys.stdout.isatty():
        return None
    try:
        return termios.tcgetattr(sys.stdout.fileno())
    except (OSError, ValueError, termios.error):
        return None


def restore_terminal(saved) -> None:
    """Undo raw mode, mouse reporting and the alternate screen after a kill."""
    if termios is None or os.name == 'nt' or not sys.stdout.isatty():
        return
    try:
        sys.stdout.write(TERMINAL_RESTORE)
        sys.stdout.flush()
    except (OSError, ValueError):
        pass
    if saved is None:
        return
    try:
        termios.tcsetattr(sys.stdout.fileno(), termios.TCSADRAIN, saved)
    except (OSError, ValueError, termios.error):
        pass


def controller_log(command: list[str], state: Path) -> Path:
    """The controller's own crash record, in the run directory it was given."""
    try:
        output = Path(command[command.index('--output') + 1])
    except (ValueError, IndexError):
        output = state / 'runs'
    return output / '.arwen-tui/controller.log'


def signal_report(status: int, command: list[str], state: Path) -> str:
    """One plain line: which signal ended the controller, and where its log is."""
    try:
        name = signal.Signals(-status).name
    except ValueError:
        name = 'signal ' + str(-status)
    return ('ArWen: the terminal controller ' + Path(command[0]).name + ' was stopped by '
            + name + '. Its log is ' + str(controller_log(command, state)))


def launch(arguments: list[str]) -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--verify-launcher', action='store_true')
    parser.add_argument('--tui-only', action='store_true')
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--cds-credentials', type=Path)
    options, forwarded = parser.parse_known_args(arguments)
    if any(value in ('--python', '--companion', '--open-companion') for value in forwarded):
        raise ValueError('The launcher selects the Python and applications included in this package.')
    manifest = json.loads((root / 'ARWEN-DESKTOP.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 'arwen.desktop-package.v1' or manifest.get('status') not in ('READY_FOR_ACCEPTANCE', 'READY'):
        raise ValueError('This package has not completed assembly.')
    python = (root / 'runtime/python.exe').resolve(strict=True)
    if Path(sys.executable).resolve() != python:
        raise ValueError('Use Start ArWen.cmd in this package.')
    rows = manifest.get('launch_files', [])
    required = {'arwen-tui.exe', 'arwen-companion.exe', 'runtime/python.exe', 'runtime/ARWEN-RUNTIME.json'}
    if not required <= {row.get('path') for row in rows}:
        raise ValueError('The package launch manifest is incomplete.')
    for row in rows:
        path = (root / row['path']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('A packaged component resolves outside this installation.')
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != row['sha256']:
            raise ValueError('A packaged component has changed: ' + row['path'])
    import gpuwm
    if Path(gpuwm.__file__).resolve().parent != root / 'runtime/Lib/site-packages/gpuwm':
        raise ValueError('Python selected an engine outside this package.')
    local = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    state = (options.state_dir or local / 'ArWenDesktop/v1').resolve()
    env = child_environment(state, python, options.cds_credentials)
    command = [str(root / 'arwen-tui.exe'), '--python', str(python),
               '--companion', str(root / 'arwen-companion.exe')]
    if not options.tui_only:
        command.append('--open-companion')
    if '--output' not in forwarded:
        command += ['--output', str(state / 'runs')]
    command += forwarded
    if options.verify_launcher:
        print(json.dumps({'status': 'PASS', 'source_revision': manifest['engine_source_revision'],
                          'python': str(python), 'command': command, 'state_directory': str(state),
                          'cache_directory': env['ARWEN_CACHE_DIR'],
                          'cds_credentials_configured': bool(env.get('CDSAPI_RC')),
                          'bytecode_writes_disabled': env.get('PYTHONDONTWRITEBYTECODE') == '1',
                          'checked_components': len(rows), 'application_opened': False,
                          'forecast_started': False}))
        return 0
    state.mkdir(parents=True, exist_ok=True)
    (state / 'appdata').mkdir(exist_ok=True)
    saved = terminal_mode()
    try:
        status = subprocess.run(command, cwd=state, env=env, check=False).returncode
    finally:
        restore_terminal(saved)
    if status < 0 and sys.stdout.isatty():
        print(signal_report(status, command, state))
    return status


if __name__ == '__main__':
    try:
        raise SystemExit(launch(sys.argv[1:]))
    except (ValueError, OSError, KeyError) as error:
        print('Cannot start ArWen: ' + str(error), file=sys.stderr)
        raise SystemExit(2)

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


def preference_folder(state: Path, key: str) -> Path | None:
    """One absolute folder from the GUI preferences within this launcher's profile.

    A missing file, a missing key, a malformed document or a relative path all
    read as no preference; a relative folder would land wherever this process
    started, so it is never honored.
    """
    preference = state / 'appdata/ArWenCompanion/preferences.json'
    try:
        value = json.loads(preference.read_text(encoding='utf-8'))
        folder = value.get(key) if isinstance(value, dict) else None
        if isinstance(folder, str) and folder.strip() and Path(folder).is_absolute():
            return Path(folder)
    except (OSError, ValueError):
        pass
    return None


def preference_folders(state: Path, key: str) -> list[Path]:
    """The absolute folders listed under one key of the GUI preferences, once each.

    A missing file, a missing key, a value that is not a list and an entry that
    is not an absolute path all read as nothing, the way ``preference_folder``
    reads a single folder.
    """
    preference = state / 'appdata/ArWenCompanion/preferences.json'
    folders: list[Path] = []
    try:
        value = json.loads(preference.read_text(encoding='utf-8'))
        entries = value.get(key) if isinstance(value, dict) else None
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, str) and entry.strip() and Path(entry).is_absolute():
                folder = Path(entry)
                if not any(_same_folder(folder, known) for known in folders):
                    folders.append(folder)
    except (OSError, ValueError):
        pass
    return folders


def _same_folder(left: Path, right: Path) -> bool:
    return os.path.normcase(os.path.normpath(str(left))) == os.path.normcase(os.path.normpath(str(right)))


def cache_directory(state: Path) -> Path:
    """Honor the GUI preference stored within this launcher's selected profile."""
    folder = preference_folder(state, 'data_folder')
    return folder / 'cache' if folder is not None else state / 'cache'


def output_folder(state: Path) -> tuple[Path, str | None]:
    """The controller's run folder, and the sentence to print when it is not the chosen one.

    The forecast output folder chosen in the GUI is created here, before the
    controller starts: a saved folder on a drive that is absent today would
    make the controller fail to create its session folder and exit before the
    GUI, and its Settings, could open. Then the profile's own ``runs`` folder
    is used and the sentence names the saved folder, the reason and the way
    out.
    """
    default = state / 'runs'
    chosen = preference_folder(state, 'forecast_output_folder')
    if chosen is None:
        return default, None
    try:
        chosen.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return default, ('The saved forecast output folder ' + str(chosen) + ' is unavailable: ' + str(error)
                         + '. New forecasts go to ' + str(default)
                         + ' until a folder that can be created is chosen in Settings, Forecast output folder.')
    return chosen, None


def output_arguments(state: Path) -> list[str]:
    """The controller's run folder: the forecast output folder chosen in the GUI.

    The profile's own ``runs`` folder and every forecast output folder chosen
    before stay named as saved-runs folders, once each and never the output
    folder itself, so the forecasts already in them remain listed in My
    forecasts after the folder moves again.
    """
    default = state / 'runs'
    output, _ = output_folder(state)
    arguments = ['--output', str(output)]
    named: list[Path] = []
    for folder in [default] + preference_folders(state, 'saved_run_folders'):
        if _same_folder(folder, output) or any(_same_folder(folder, known) for known in named):
            continue
        named.append(folder)
        arguments += ['--saved-runs', str(folder)]
    return arguments


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
        command += output_arguments(state)
        notice = output_folder(state)[1]
        if notice is not None and not options.verify_launcher:
            print(notice, file=sys.stderr)
    command += forwarded
    if options.verify_launcher:
        print(json.dumps({'status': 'PASS', 'source_revision': manifest['engine_source_revision'],
                          'python': str(python), 'command': command, 'state_directory': str(state),
                          'cache_directory': env['ARWEN_CACHE_DIR'],
                          'output_directory': str(controller_log(command, state).parent.parent),
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

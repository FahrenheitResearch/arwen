"""Start the packaged ArWen GUI and terminal with the Python this package selects.

One file serves every package this project ships.  A package that carries its
own runtime (``runtime/python.exe`` beside this file on Windows,
``runtime/bin/python3`` on Linux) selects that runtime and takes no ``--python``;
a package that carries none takes ``--python`` (or ``ARWEN_PYTHON``) and verifies
the engine installed in it against ``ENGINE-PYTHON.json``.  Everything else, the
folder preferences, the terminal restore, the notices and the ``--verify-launcher``
record, is the same code on both routes and on both platforms.

THE BREAKAGE THIS PREVENTS: through 2.7.3 this template was the bundled-runtime
variant only, so the Windows desktop and the Linux tarball each shipped a
launcher assembled outside the repository, one carried forward by hand and one
lifted hunk by hunk out of this file.  A repair committed here reached the
packages a release late, or not at all, and the three launchers disagreed about
the same package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

try:
    import termios
except ImportError:  # Windows has no POSIX terminal to put back.
    termios = None


# What each package platform calls its programs, its payload, its own runtime
# and its profile.  ``system`` is the ``sys.platform`` this package's programs
# run on, matched by prefix so a version suffix does not need a second row.  A
# new platform is a row here; nothing else in this file knows a platform name.
PACKAGE_PLATFORMS = {
    'windows-x86_64': {
        'system': 'win32',
        'computer': 'Windows',
        'controller': 'arwen-tui.exe',
        'companion': 'arwen-companion.exe',
        'payload': ('arwen-weather.exe', 'maplibre-native-c.dll', 'vcruntime140.dll'),
        'runtime_python': 'runtime/python.exe',
        'runtime_manifest': 'runtime/ARWEN-RUNTIME.json',
        'start_script': 'Start ArWen.cmd',
        'state_variable': 'LOCALAPPDATA',
        'state_home': 'AppData/Local',
    },
    'linux-x86_64': {
        'system': 'linux',
        'computer': 'Linux',
        'controller': 'arwen-tui',
        'companion': 'arwen-companion',
        'payload': ('arwen-weather', 'libmaplibre-native-c.so'),
        'runtime_python': 'runtime/bin/python3',
        'runtime_manifest': 'runtime/ARWEN-RUNTIME.json',
        'start_script': 'Start ArWen.sh',
        'state_variable': 'XDG_STATE_HOME',
        'state_home': '.local/state',
    },
}

# The document a package without its own runtime carries: the engine version,
# the Python file digests and the terminal digest the selected Python must hold.
PACKAGE_IDENTITY = 'ENGINE-PYTHON.json'


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def interpreter(value: str) -> Path:
    """The interpreter a caller named, as it was given.

    A virtual environment's own ``python`` is kept, symlink and all: resolving
    it would select the base interpreter and lose the environment that holds
    the installed engine.
    """
    expanded = os.path.expanduser(value)
    candidate = expanded if any(separator in expanded for separator in ('/', '\\')) else shutil.which(expanded)
    if not candidate or not Path(candidate).is_file():
        raise ValueError('Python was not found. Install Python 3.11 or newer and pass --python to the python '
                         'of the environment ArWen is installed into.')
    return Path(os.path.abspath(candidate))


def package_manifest(root: Path) -> dict:
    """This package's identity document, or an empty one when it cannot be read.

    It is read before the arguments are parsed so the help text names the
    engine version this package carries instead of a version written into this
    file.  ``launch`` refuses an unusable document after parsing, where the
    refusal is printed as a sentence rather than a traceback.
    """
    try:
        document = json.loads((root / 'ARWEN-DESKTOP.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def package_platform(manifest: dict) -> dict:
    """The row naming this package's programs, runtime and profile folder."""
    name = manifest.get('platform')
    row = PACKAGE_PLATFORMS.get(name if isinstance(name, str) else '')
    if row is None:
        raise ValueError('This package names the platform ' + str(name) + ', and this launcher has no program '
                         'names for it. Use a package built for ' + ' or '.join(sorted(PACKAGE_PLATFORMS)) + '.')
    if not sys.platform.startswith(row['system']):
        raise ValueError('This package holds ' + row['computer'] + ' programs, which this computer cannot run. '
                         'Download the package built for this computer.')
    return row


def bundled_runtime(root: Path, row: dict) -> Path | None:
    """The Python this package carries, or None when the package carries none."""
    python = root / row['runtime_python']
    return python if python.is_file() else None


def required_components(row: dict, bundled: bool) -> set[str]:
    """The manifest rows a package of this platform must record.

    The two programs this launcher starts, the payload they load once they are
    running, and the document naming the Python this package runs on: the
    runtime it carries, or the engine it expects to find installed.  A row that
    is absent is never digest-verified, so a package assembled without one
    fails later, in the interface, on a file nothing checked.  Every other row
    a package records is verified all the same.
    """
    required = {row['controller'], row['companion'], *row['payload']}
    return required | ({row['runtime_python'], row['runtime_manifest']} if bundled else {PACKAGE_IDENTITY})


def default_state(row: dict) -> Path:
    """The profile this platform keeps its settings, cache and runs in."""
    base = os.environ.get(row['state_variable'])
    return Path(base if base else Path.home() / row['state_home']) / 'ArWenDesktop/v1'


# Characters no folder this launcher creates may carry.  A wildcard or an
# embedded null passes ``Path.is_absolute`` on every platform and passes
# ``pathlib`` itself on Linux, and then Windows raises out of ``mkdir``:
# ``ValueError`` for the null, ``OSError`` for the wildcard.  The desktop
# bootstrap refuses the same set, so all three readers of the preferences
# document accept one definition of an absolute folder.
_REFUSED_IN_A_FOLDER = '\0?*"<>|'


def absolute_folder(value: object) -> Path | None:
    """The absolute folder a saved preference names, or None when it names none."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if any(character in text for character in _REFUSED_IN_A_FOLDER) or any(ord(character) < 32 for character in text):
        return None
    folder = Path(text)
    return folder if folder.is_absolute() else None


def preference_folder(state: Path, key: str) -> Path | None:
    """One absolute folder from the GUI preferences within this launcher's profile.

    A missing file, a missing key, a malformed document or a relative path all
    read as no preference; a relative folder would land wherever this process
    started, so it is never honored.
    """
    preference = state / 'appdata/ArWenCompanion/preferences.json'
    try:
        value = json.loads(preference.read_text(encoding='utf-8'))
        return absolute_folder(value.get(key)) if isinstance(value, dict) else None
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
            folder = absolute_folder(entry)
            if folder is not None and not any(_same_folder(folder, known) for known in folders):
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
    except (OSError, ValueError) as error:
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


def child_environment(state: Path, python: Path | str, cds_credentials: Path | None = None,
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
    # Host Python, gpuwm and ArWen settings never reach a package's children, and everything
    # else the caller set travels untouched, NO_COLOR included. NO_COLOR is an instruction
    # from the user, not a description of the window: the controller implements it (a
    # monochrome theme) and the terminal library it draws with honors it, so a launcher that
    # dropped it made a package disobey an instruction its own terminal obeys.
    env = {key: value for key, value in source.items()
           if not key.upper().startswith(('PYTHON', 'GPUWM_', 'ARWEN_'))}
    env.update(PYTHONNOUSERSITE='1', PYTHONSAFEPATH='1', PYTHONUTF8='1',
               PYTHONDONTWRITEBYTECODE='1',
               ARWEN_CACHE_DIR=str(cache_directory(state)), ARWEN_PYTHON=str(python),
               LOCALAPPDATA=str(state / 'appdata'))
    # A terminal the caller already stands in describes itself, and the controller is a
    # full-screen terminal interface: it reads TERM for keys and colours and COLORTERM for
    # whether 24-bit colour is real. Naming both outright told a 256-colour terminal it was
    # truecolor, so they are defaults for the window that describes nothing, which is what a
    # package started from the desktop application or from Explorer hands its children.
    for name, described in (('TERM', 'xterm-256color'), ('COLORTERM', 'truecolor')):
        if not env.get(name):
            env[name] = described
    if cds_credentials is not None:
        credentials = Path(cds_credentials).expanduser().absolute()
        if not credentials.is_file():
            raise ValueError('The configured CDS credentials file is unavailable.')
        env['CDSAPI_RC'] = str(credentials)
    env['PATH'] = str(Path(python).parent) + os.pathsep + env.get('PATH', '')
    return env


# The engine check a package without its own runtime runs inside the Python it
# was given: the engine version, every Python file digest and the terminal
# digest recorded in ENGINE-PYTHON.json, read in that interpreter's own process.
RUNTIME_PROBE = r'''
import hashlib, importlib.metadata, json, pathlib, sys
from gpuwm import tui_cli
import gpuwm
if sys.version_info < (3, 11):
    raise SystemExit('Python 3.11 or newer is required.')
root = pathlib.Path(gpuwm.__file__).resolve().parent.parent
expected = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8'))
version = importlib.metadata.version('gpuwm')
if version != expected['engine_version']:
    raise SystemExit('This desktop needs gpuwm ' + expected['engine_version'] + '; found ' + version)
checked = 0
for record in expected['python_files']:
    path = root / record['path']
    if not path.is_file():
        raise SystemExit('The installed engine is incomplete: ' + record['path'])
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != record['sha256']:
        raise SystemExit('The installed engine does not match this desktop release: ' + record['path'])
    checked += 1
tui = tui_cli.require_tui()
with tui.open('rb') as stream:
    tui_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
if tui_sha != expected['tui_sha256']:
    raise SystemExit('The installed terminal does not match this desktop release.')
print(json.dumps({'engine_version': version, 'engine_source_revision': expected['engine_source_revision'],
                  'python': sys.executable, 'python_version': sys.version.split()[0],
                  'python_files_verified': checked, 'installed_tui_sha256': tui_sha}))
'''


def installed_engine(python: Path, identity: Path, env: dict[str, str]) -> dict:
    """What the selected Python holds, checked against this package's identity document."""
    probe = subprocess.run([str(python), '-I', '-B', '-c', RUNTIME_PROBE, str(identity)],
                           env=env, text=True, capture_output=True, timeout=60)
    if probe.returncode:
        reason = probe.stderr.strip() or probe.stdout.strip() or 'The selected Python could not load ArWen.'
        raise ValueError(reason + '\nInstall the matched engine in a virtual environment as described in '
                         'INSTALL.md, then pass --python to this launcher.')
    return json.loads(probe.stdout)


def bundled_engine(root: Path, python: Path, manifest: dict, row: dict) -> dict:
    """What the package's own runtime holds: its engine, from inside the package.

    The runtime's files are verified by digest with the rest of the package, so
    what is left to establish is that this process is that runtime and that the
    engine it imports is the one inside the package rather than one on the
    machine.
    """
    if Path(sys.executable).resolve() != python.resolve():
        raise ValueError('This package runs on the Python it carries. Use ' + row['start_script']
                         + ' in this package.')
    try:
        import gpuwm
    except ImportError as error:
        raise ValueError("This package's own Python could not import its engine: " + str(error))
    engine = Path(gpuwm.__file__).resolve().parent
    if not engine.is_relative_to((root / 'runtime').resolve()):
        raise ValueError('Python selected an engine outside this package.')
    return {'engine_version': manifest.get('engine_version'),
            'engine_source_revision': manifest.get('engine_source_revision'),
            'python': str(python), 'python_version': sys.version.split()[0],
            'engine_directory': str(engine)}


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
    manifest = package_manifest(root)
    version = manifest.get('engine_version')
    engine = 'gpuwm ' + version if isinstance(version, str) and version.strip() else 'the engine this package names'
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--python', help='Python of the environment holding ' + engine + ', or ARWEN_PYTHON; '
                                         'a package that carries its own runtime uses that runtime instead')
    parser.add_argument('--state-dir', type=Path, help='Separate settings, cache and run directory')
    parser.add_argument('--cds-credentials', type=Path, help='Optional CDS configuration file')
    parser.add_argument('--tui-only', action='store_true', help='Open the terminal without opening the GUI')
    parser.add_argument('--verify-launcher', action='store_true',
                        help='Verify this package and its engine without opening either interface')
    options, forwarded = parser.parse_known_args(arguments)
    if any(value in ('--companion', '--open-companion') for value in forwarded):
        raise ValueError('The launcher selects the terminal and the application included in this package, so '
                         '--companion and --open-companion name programs it already knows. Start it again '
                         'without them, or with --tui-only to open the terminal by itself.')
    if not manifest:
        raise ValueError('This folder does not hold a readable ARWEN-DESKTOP.json, so there is nothing here '
                         'to verify or start. Extract the complete ArWen desktop package and start it again.')
    if (manifest.get('schema') != 'arwen.desktop-package.v1'
            or manifest.get('status') not in ('READY', 'READY_FOR_ACCEPTANCE')):
        raise ValueError('ARWEN-DESKTOP.json does not mark this package complete, so its programs may be '
                         'half-written. Use a released package.')
    row = package_platform(manifest)
    runtime_python = bundled_runtime(root, row)
    notices: list[str] = []
    if runtime_python is not None and options.python:
        raise ValueError('This package includes its own Python at ' + str(runtime_python) + ', so it takes no '
                         '--python and cannot start on another one. Start it again without --python.')
    if runtime_python is not None and os.environ.get('ARWEN_PYTHON'):
        notices.append('ArWen: ARWEN_PYTHON names ' + os.environ['ARWEN_PYTHON'] + '. This package includes its '
                       'own Python at ' + str(runtime_python) + ' and starts on that one.')
    rows = manifest.get('launch_files', [])
    missing = sorted(required_components(row, runtime_python is not None)
                     - {record.get('path') for record in rows})
    if missing:
        raise ValueError('The package launch manifest does not record ' + ', '.join(missing)
                         + '. Extract a complete package and start it again.')
    for record in rows:
        path = (root / record['path']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('A packaged component resolves outside this installation: ' + record['path'])
        if digest(path) != record['sha256']:
            raise ValueError('A packaged component has changed: ' + record['path'])
    python = runtime_python if runtime_python is not None else interpreter(
        options.python or os.environ.get('ARWEN_PYTHON') or sys.executable)
    state = (options.state_dir or default_state(row)).expanduser().resolve()
    env = child_environment(state, python, options.cds_credentials)
    if runtime_python is not None:
        runtime = bundled_engine(root, python, manifest, row)
    else:
        runtime = installed_engine(python, root / PACKAGE_IDENTITY, env)
        if runtime.get('engine_source_revision') != manifest.get('engine_source_revision'):
            raise ValueError('The package and engine identity documents disagree.')
    command = [str(root / row['controller']), '--python', str(python),
               '--companion', str(root / row['companion'])]
    if not options.tui_only:
        command.append('--open-companion')
    if '--output' not in forwarded:
        command += output_arguments(state)
        notice = output_folder(state)[1]
        if notice is not None:
            notices.append(notice)
    command += forwarded
    if not options.verify_launcher:
        for line in notices:
            print(line, file=sys.stderr)
    if options.verify_launcher:
        print(json.dumps({'status': 'PASS', 'platform': manifest['platform'],
                          'runtime_source': 'bundled' if runtime_python is not None else 'installed',
                          'source_revision': manifest['engine_source_revision'],
                          'engine_version': manifest.get('engine_version'), 'runtime': runtime,
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
    env['ARWEN_COMPANION'] = str(root / row['companion'])
    # The controller keeps the directory the caller stands in. It reads its own working
    # directory and starts every engine job, node operation and file browse from it, and
    # resolves a relative path a caller forwards against it, so naming one here would move
    # all of those. The other door starts the same controller for the same package in the
    # package folder, which is where the caller of this package's own start script stands.
    saved = terminal_mode()
    try:
        status = subprocess.run(command, env=env, check=False).returncode
    finally:
        restore_terminal(saved)
    if status < 0 and sys.stdout.isatty():
        print(signal_report(status, command, state))
    return status


if __name__ == '__main__':
    if sys.version_info < (3, 11):
        raise SystemExit('ArWen requires Python 3.11 or newer. Select a newer interpreter with --python '
                         'or ARWEN_PYTHON.')
    try:
        raise SystemExit(launch(sys.argv[1:]))
    except (ValueError, OSError, KeyError, subprocess.TimeoutExpired) as error:
        print('Cannot start ArWen: ' + str(error), file=sys.stderr)
        raise SystemExit(2)

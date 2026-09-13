#!/usr/bin/env sh
# Open the ArWen terminal from this package, without the application.
set -eu
task_directory=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
# A package that carries its own runtime starts on that runtime and takes no --python.
if [ -x "$task_directory/runtime/bin/python3" ]; then
    exec "$task_directory/runtime/bin/python3" -I -B "$task_directory/launch_arwen.py" --tui-only "$@"
fi
task_python=${ARWEN_PYTHON:-python3}
task_python_value=0
for task_argument in "$@"; do
    if [ "$task_python_value" -eq 1 ]; then
        task_python=$task_argument
        task_python_value=0
    else
        case "$task_argument" in
            --python) task_python_value=1 ;;
            --python=*) task_python=${task_argument#--python=} ;;
            --) break ;;
        esac
    fi
done
if [ "$task_python_value" -eq 1 ] || [ -z "$task_python" ]; then
    printf '%s\n' 'ArWen: --python needs an executable.' >&2
    exit 2
fi
if ! command -v "$task_python" >/dev/null 2>&1; then
    printf '%s\n' 'Cannot start ArWen: no Python was found. Install Python 3.11 or newer, or set ARWEN_PYTHON to the python of the environment ArWen is installed into.' >&2
    exit 2
fi
exec "$task_python" -I -B "$task_directory/launch_arwen.py" --tui-only "$@"

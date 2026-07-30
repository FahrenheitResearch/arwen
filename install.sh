#!/bin/sh
# One-command install for a gpuwm (ArWen) developer checkout (POSIX sh).
#
#   ./install.sh [--yes] [--no-render]      -- from a checkout root
#
# The standalone (curl | sh) form clones the public repository into
# ./gpuwm when run outside a checkout; GPUWM_REPO_URL overrides the
# clone source (fork or mirror).
#
# What it does, in order (every step is re-run safe):
#   1. finds the checkout (or clones $GPUWM_REPO_URL into ./gpuwm);
#   2. creates .venv if absent and installs -e '.[gpu,render]' into it;
#   3. stages the externalized Thompson tables with `gpuwm fetch-tables`
#      (downloads only what is absent -- ~243 MiB from a checkout --
#      SHA-256 verified before install; a no-op when already staged;
#      skip with --no-fetch-tables or GPUWM_INSTALL_NO_FETCH_TABLES=1);
#   4. offers to install rustup when `cargo` is missing (prompts first;
#      --yes or GPUWM_INSTALL_YES=1 consents non-interactively);
#   5. builds the vendored Rust GRIB bridges offline in
#      tools/grib1_bridge;
#   6. builds the vendored production render engine offline in
#      tools/rustwx (skip with --no-render or
#      GPUWM_INSTALL_NO_RENDER=1; `gpuwm render` falls back to
#      matplotlib until it is built);
#   7. finishes with `gpuwm doctor` and exits with doctor's status.
#
# Environment:
#   GPUWM_REPO_URL     clone source when run outside a checkout
#                      (default:
#                      https://github.com/FahrenheitResearch/arwen).
#   GPUWM_PYTHON       interpreter used to create .venv (default:
#                      python3, then python)
#   GPUWM_INSTALL_YES  "1" behaves like --yes
#   GPUWM_INSTALL_NO_RENDER  "1" behaves like --no-render
#   GPUWM_INSTALL_NO_FETCH_TABLES  "1" behaves like --no-fetch-tables

set -eu

YES="${GPUWM_INSTALL_YES:-0}"
NO_RENDER="${GPUWM_INSTALL_NO_RENDER:-0}"
NO_FETCH_TABLES="${GPUWM_INSTALL_NO_FETCH_TABLES:-0}"
for arg in "$@"; do
    case "$arg" in
        -y|--yes) YES=1 ;;
        --no-render) NO_RENDER=1 ;;
        --no-fetch-tables) NO_FETCH_TABLES=1 ;;
        -h|--help)
            sed -n '2,36p' "$0" 2>/dev/null || true
            exit 0 ;;
        *)
            echo "install.sh: unknown argument '$arg'" \
                 "(--yes, --no-render, and --no-fetch-tables)" >&2
            exit 2 ;;
    esac
done

say()  { printf 'install: %s\n' "$*"; }
fail() { printf 'install: ERROR: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- checkout
REPO_URL="${GPUWM_REPO_URL:-https://github.com/FahrenheitResearch/arwen}"
if [ -f pyproject.toml ] && [ -d gpuwm ] && [ -d tools/grib1_bridge ]; then
    say "using the existing checkout at $(pwd)"
elif [ -f gpuwm/pyproject.toml ] && [ -d gpuwm/gpuwm ]; then
    cd gpuwm
    say "using the existing checkout at $(pwd)"
else
    command -v git >/dev/null 2>&1 || fail "git is required to clone"
    say "cloning $REPO_URL into ./gpuwm"
    git clone "$REPO_URL" gpuwm
    cd gpuwm
fi

# -------------------------------------------------------------------- venv
if [ -n "${GPUWM_PYTHON:-}" ]; then
    PYTHON="$GPUWM_PYTHON"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
elif command -v python >/dev/null 2>&1; then
    PYTHON=python
else
    fail "no python3/python on PATH (Python 3.11+ is required)"
fi

if [ -x .venv/bin/python ]; then
    say "reusing the existing .venv"
else
    say "creating .venv with $PYTHON"
    "$PYTHON" -m venv .venv
fi
VENV_PY=.venv/bin/python
"$VENV_PY" -m pip install --upgrade pip
say "installing gpuwm with the [gpu,render] extras (editable)"
"$VENV_PY" -m pip install -e '.[gpu,render]'

# ------------------------------------------------------- externalized tables
# The two largest Thompson tables ship as GitHub release assets rather
# than in the wheel (freezeH2O.dat, 243 MiB, is not in git either);
# fetch-tables downloads only what is absent and verifies SHA-256
# against the packaged pins before installing.
if [ "$NO_FETCH_TABLES" = 1 ]; then
    say "skipping the externalized table fetch (--no-fetch-tables);"
    say "gpuwm doctor prints the exact fetch command while they are missing"
else
    say "staging the externalized Thompson tables (downloads only what"
    say "is absent -- ~243 MiB from a checkout; SHA-256 verified)"
    .venv/bin/gpuwm fetch-tables
fi

# ---------------------------------------------------------------- rust/cargo
# A rustup installed earlier in this same run (or a previous one) lives in
# ~/.cargo/bin before it reaches PATH, so look there too.
PATH="$HOME/.cargo/bin:$PATH"
if command -v cargo >/dev/null 2>&1; then
    say "cargo found: $(command -v cargo)"
else
    say "cargo was not found; the Rust GRIB bridges need a Rust toolchain."
    if [ "$YES" != 1 ]; then
        if [ -r /dev/tty ]; then
            printf 'install: install rustup (https://rustup.rs) now? [y/N] '
            read -r answer < /dev/tty || answer=""
            case "$answer" in
                y|Y|yes|YES) YES=1 ;;
            esac
        fi
    fi
    if [ "$YES" != 1 ]; then
        fail "cargo is missing and consent to install rustup was not \
given; re-run with --yes (or GPUWM_INSTALL_YES=1), or install a Rust \
toolchain yourself and re-run"
    fi
    command -v curl >/dev/null 2>&1 || fail "curl is required to fetch rustup"
    say "installing rustup (stable toolchain, PATH left unmodified)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path
    command -v cargo >/dev/null 2>&1 \
        || fail "rustup finished but cargo is still not on PATH"
fi

# ------------------------------------------------------- offline Rust build
say "building the vendored Rust GRIB bridges (offline, locked)"
( cd tools/grib1_bridge && cargo build --release --locked --offline )
if [ "$NO_RENDER" = 1 ]; then
    say "skipping the tools/rustwx render engine (--no-render);"
    say "gpuwm render uses the matplotlib fallback until it is built"
else
    say "building the vendored render engine in tools/rustwx (offline,"
    say "locked; the long pole of install -- skip with --no-render)"
    ( cd tools/rustwx && cargo build --release --locked --offline )
fi

# ------------------------------------------------------------------ doctor
say "running gpuwm doctor"
if .venv/bin/gpuwm doctor; then
    say "done -- doctor is clean.  Activate with: . .venv/bin/activate"
else
    status=$?
    say "install steps completed; doctor reports gaps above (each line"
    say "prints its own remedy).  Re-run ./install.sh any time."
    exit $status
fi

#!/bin/sh
# One-command install for a gpuwm (ArWen) developer checkout (POSIX sh).
#
#   ./install.sh [--yes] [--no-render] [--cuda 12|13]
#                                           -- from a checkout root
#
# The standalone (curl | sh) form clones the public repository into
# ./gpuwm when run outside a checkout; GPUWM_REPO_URL overrides the
# clone source (fork or mirror).
#
# What it does, in order (every step is re-run safe):
#   1. finds the checkout (or clones $GPUWM_REPO_URL into ./gpuwm);
#   2. creates .venv if absent and installs -e '.[gpu-cuNN,render]' into
#      it, where NN is the CUDA major this box's driver reports (CuPy
#      ships one wheel per major and the wrong one dies at its first
#      cuBLAS load); --cuda overrides the detection, and an
#      undetectable major is announced rather than defaulted quietly;
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
#   GPUWM_INSTALL_CUDA  "12" or "13" behaves like --cuda

set -eu

YES="${GPUWM_INSTALL_YES:-0}"
NO_RENDER="${GPUWM_INSTALL_NO_RENDER:-0}"
NO_FETCH_TABLES="${GPUWM_INSTALL_NO_FETCH_TABLES:-0}"
CUDA_MAJOR="${GPUWM_INSTALL_CUDA:-}"
while [ "$#" -gt 0 ]; do
    case "$1" in
        -y|--yes) YES=1 ;;
        --no-render) NO_RENDER=1 ;;
        --no-fetch-tables) NO_FETCH_TABLES=1 ;;
        --cuda)
            [ "$#" -ge 2 ] || {
                echo "install.sh: --cuda needs a value (12 or 13)" >&2
                exit 2; }
            CUDA_MAJOR="$2"; shift ;;
        --cuda=*) CUDA_MAJOR="${1#--cuda=}" ;;
        -h|--help)
            sed -n '2,41p' "$0" 2>/dev/null || true
            exit 0 ;;
        *)
            echo "install.sh: unknown argument '$1'" \
                 "(--yes, --no-render, --no-fetch-tables, and --cuda)" >&2
            exit 2 ;;
    esac
    shift
done
case "$CUDA_MAJOR" in
    ''|12|13) ;;
    *)  echo "install.sh: --cuda takes 12 or 13, not '$CUDA_MAJOR'" >&2
        exit 2 ;;
esac

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
# ------------------------------------------------------------ CUDA major
# CuPy ships ONE wheel per CUDA major and pip cannot detect the major, so
# the extra has to name it.  Through 1.8.0 this line pasted
# `.[gpu,render]` unconditionally -- the cu12 wheel -- so a CUDA-13-only
# box got a CuPy that imports cleanly, compiles kernels, and then dies at
# its first cuBLAS load, with nothing in the install saying so.  Read the
# major off the driver instead, and when it cannot be read, SAY that
# rather than defaulting in silence.
# The header label is not one string: Linux drivers print
# "CUDA Version: 12.4" and the Windows driver on the reference box
# prints "CUDA UMD Version: 13.3".  Matching only the first spelling
# read as "no NVIDIA driver" on a machine that plainly had one.
detect_cuda_major() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1
    detected=$(nvidia-smi 2>/dev/null |
        sed -n 's/.*CUDA[A-Za-z ]*Version: *\([0-9][0-9]*\).*/\1/p' |
        head -n 1)
    case "$detected" in
        ''|*[!0-9]*) return 1 ;;
        *) printf '%s\n' "$detected" ;;
    esac
}

if [ -n "$CUDA_MAJOR" ]; then
    say "CUDA major $CUDA_MAJOR was given on the command line"
else
    CUDA_MAJOR=$(detect_cuda_major || true)
    if [ -n "$CUDA_MAJOR" ]; then
        say "nvidia-smi reports CUDA $CUDA_MAJOR"
    fi
fi
case "$CUDA_MAJOR" in
    12|13) GPU_EXTRA="gpu-cu$CUDA_MAJOR" ;;
    *)
        GPU_EXTRA="gpu-cu12"
        say "the box's CUDA major could not be read (no nvidia-smi, or no"
        say "driver answered), so this install falls back to [gpu-cu12]."
        say "IF THIS BOX'S CUDA IS 13-ONLY THAT WHEEL IS WRONG: it will"
        say "import fine and fail at the first cuBLAS load.  Re-run with"
        say "--cuda 13 in that case; gpuwm doctor judges the pairing at"
        say "the end of this script either way."
        ;;
esac
say "installing gpuwm with the [$GPU_EXTRA,render] extras (editable)"
"$VENV_PY" -m pip install -e ".[$GPU_EXTRA,render]"

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

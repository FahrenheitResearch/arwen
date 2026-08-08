# One-command install for a gpuwm (ArWen) developer checkout (PowerShell).
#
#   .\install.ps1 [-Yes] [-NoRender] [-Cuda 12|13]
#                                           -- from a checkout root
#
# The standalone (iwr | iex) form clones the public repository into
# .\gpuwm when run outside a checkout; GPUWM_REPO_URL overrides the
# clone source (fork or mirror).
#
# What it does, in order (every step is re-run safe):
#   1. finds the checkout (or clones $env:GPUWM_REPO_URL into .\gpuwm);
#   2. creates .venv if absent and installs -e ".[gpu-cuNN,render]" into
#      it, where NN is the CUDA major this box's driver reports (CuPy
#      ships one wheel per major and the wrong one dies at its first
#      cuBLAS load); -Cuda overrides the detection, and an undetectable
#      major is announced rather than defaulted quietly;
#   3. stages the externalized Thompson tables with `gpuwm fetch-tables`
#      (downloads only what is absent -- ~243 MiB from a checkout --
#      SHA-256 verified before install; a no-op when already staged;
#      skip with -NoFetchTables or GPUWM_INSTALL_NO_FETCH_TABLES=1);
#   4. offers to install rustup when `cargo` is missing (prompts first;
#      -Yes or GPUWM_INSTALL_YES=1 consents non-interactively);
#   5. builds the vendored Rust GRIB bridges offline in
#      tools\grib1_bridge;
#   6. builds the vendored production render engine offline in
#      tools\rustwx (skip with -NoRender or GPUWM_INSTALL_NO_RENDER=1;
#      `gpuwm render` falls back to matplotlib until it is built);
#   7. finishes with `gpuwm doctor` and exits with doctor's status.
#
# Environment:
#   GPUWM_REPO_URL     clone source when run outside a checkout
#                      (default:
#                      https://github.com/FahrenheitResearch/arwen).
#   GPUWM_PYTHON       interpreter used to create .venv (default:
#                      python on PATH, then the `py -3` launcher)
#   GPUWM_INSTALL_YES  "1" behaves like -Yes
#   GPUWM_INSTALL_NO_RENDER  "1" behaves like -NoRender
#   GPUWM_INSTALL_NO_FETCH_TABLES  "1" behaves like -NoFetchTables
#   GPUWM_INSTALL_CUDA  "12" or "13" behaves like -Cuda
#
# No param() block: the script must also run when piped through iex,
# where param() is unavailable; flags arrive via $args or environment.

$ErrorActionPreference = 'Stop'

$Yes = ($env:GPUWM_INSTALL_YES -eq '1')
$NoRender = ($env:GPUWM_INSTALL_NO_RENDER -eq '1')
$NoFetchTables = ($env:GPUWM_INSTALL_NO_FETCH_TABLES -eq '1')
$CudaMajor = $env:GPUWM_INSTALL_CUDA
$scriptArgs = @()
if (Test-Path variable:args) { $scriptArgs = @($args) }
$wantCuda = $false
foreach ($arg in $scriptArgs) {
    if ($wantCuda) { $CudaMajor = "$arg"; $wantCuda = $false; continue }
    switch -Regex ($arg) {
        '^(-y|-Yes|--yes)$' { $Yes = $true }
        '^(-NoRender|--no-render)$' { $NoRender = $true }
        '^(-NoFetchTables|--no-fetch-tables)$' { $NoFetchTables = $true }
        '^(-Cuda|--cuda)$' { $wantCuda = $true }
        '^(-Cuda|--cuda)[:=](.+)$' { $CudaMajor = $Matches[2] }
        default {
            throw ("install.ps1: unknown argument '$arg' " +
                   "(-Yes, -NoRender, -NoFetchTables, and -Cuda)")
        }
    }
}
if ($wantCuda) { throw 'install.ps1: -Cuda needs a value (12 or 13)' }
if ($CudaMajor -and @('12', '13') -notcontains "$CudaMajor") {
    throw "install.ps1: -Cuda takes 12 or 13, not '$CudaMajor'"
}

function Say([string]$Message) { Write-Host "install: $Message" }
function Fail([string]$Message) { throw "install: ERROR: $Message" }
function Invoke-Step([string]$What, [scriptblock]$Step) {
    & $Step
    if ($LASTEXITCODE -ne 0) { Fail "$What failed (exit $LASTEXITCODE)" }
}

# ---------------------------------------------------------------- checkout
$repoUrl = if ($env:GPUWM_REPO_URL) { $env:GPUWM_REPO_URL }
           else { 'https://github.com/FahrenheitResearch/arwen' }
if ((Test-Path 'pyproject.toml') -and (Test-Path 'gpuwm') -and
        (Test-Path 'tools\grib1_bridge')) {
    Say "using the existing checkout at $(Get-Location)"
} elseif ((Test-Path 'gpuwm\pyproject.toml') -and
        (Test-Path 'gpuwm\gpuwm')) {
    Set-Location 'gpuwm'
    Say "using the existing checkout at $(Get-Location)"
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Fail 'git is required to clone'
    }
    Say "cloning $repoUrl into .\gpuwm"
    Invoke-Step 'git clone' { git clone $repoUrl gpuwm }
    Set-Location 'gpuwm'
}

# -------------------------------------------------------------------- venv
$venvPython = Join-Path '.venv' 'Scripts\python.exe'
if (Test-Path $venvPython) {
    Say 'reusing the existing .venv'
} else {
    if ($env:GPUWM_PYTHON) {
        Say "creating .venv with $env:GPUWM_PYTHON"
        Invoke-Step 'venv creation' { & $env:GPUWM_PYTHON -m venv .venv }
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        Say 'creating .venv with python'
        Invoke-Step 'venv creation' { python -m venv .venv }
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        Say 'creating .venv with the py -3 launcher'
        Invoke-Step 'venv creation' { py -3 -m venv .venv }
    } else {
        Fail 'no python/py on PATH (Python 3.11+ is required)'
    }
}
Invoke-Step 'pip upgrade' {
    & $venvPython -m pip install --upgrade pip
}
# ------------------------------------------------------------ CUDA major
# CuPy ships ONE wheel per CUDA major and pip cannot detect the major, so
# the extra has to name it.  Through 1.8.0 this line pasted
# ".[gpu,render]" unconditionally -- the cu12 wheel -- so a CUDA-13-only
# box got a CuPy that imports cleanly, compiles kernels, and then dies at
# its first cuBLAS load, with nothing in the install saying so.  Read the
# major off the driver instead, and when it cannot be read, SAY that
# rather than defaulting in silence.
function Get-CudaMajor {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $null
    }
    try { $smi = & nvidia-smi } catch { return $null }
    # The header label is not one string: Linux drivers print
    # "CUDA Version: 12.4" and the Windows driver on the reference box
    # prints "CUDA UMD Version: 13.3".  Matching only the first
    # spelling read as "no NVIDIA driver" on a machine that plainly
    # had one.
    foreach ($line in @($smi)) {
        if ("$line" -match 'CUDA[A-Za-z ]*Version:\s*(\d+)') {
            return $Matches[1]
        }
    }
    return $null
}

if ($CudaMajor) {
    Say "CUDA major $CudaMajor was given on the command line"
} else {
    $CudaMajor = Get-CudaMajor
    if ($CudaMajor) { Say "nvidia-smi reports CUDA $CudaMajor" }
}
if (@('12', '13') -contains "$CudaMajor") {
    $gpuExtra = "gpu-cu$CudaMajor"
} else {
    $gpuExtra = 'gpu-cu12'
    Say 'the box''s CUDA major could not be read (no nvidia-smi, or no'
    Say 'driver answered), so this install falls back to [gpu-cu12].'
    Say 'IF THIS BOX''S CUDA IS 13-ONLY THAT WHEEL IS WRONG: it will'
    Say 'import fine and fail at the first cuBLAS load.  Re-run with'
    Say '-Cuda 13 in that case; gpuwm doctor judges the pairing at the'
    Say 'end of this script either way.'
}
Say "installing gpuwm with the [$gpuExtra,render] extras (editable)"
Invoke-Step 'pip install' {
    & $venvPython -m pip install -e ".[$gpuExtra,render]"
}

# ------------------------------------------------------- externalized tables
# The two largest Thompson tables ship as GitHub release assets rather
# than in the wheel (freezeH2O.dat, 243 MiB, is not in git either);
# fetch-tables downloads only what is absent and verifies SHA-256
# against the packaged pins before installing.
if ($NoFetchTables) {
    Say 'skipping the externalized table fetch (-NoFetchTables);'
    Say 'gpuwm doctor prints the exact fetch command while they are missing'
} else {
    Say 'staging the externalized Thompson tables (downloads only what'
    Say 'is absent -- ~243 MiB from a checkout; SHA-256 verified)'
    Invoke-Step 'gpuwm fetch-tables' {
        & (Join-Path '.venv' 'Scripts\gpuwm.exe') fetch-tables
    }
}

# ---------------------------------------------------------------- rust/cargo
# A rustup installed earlier in this same run (or a previous one) lives in
# ~\.cargo\bin before it reaches PATH, so look there too.
$cargoBin = Join-Path $env:USERPROFILE '.cargo\bin'
if (Test-Path (Join-Path $cargoBin 'cargo.exe')) {
    $env:Path = "$cargoBin;$env:Path"
}
$cargo = Get-Command cargo -ErrorAction SilentlyContinue
if ($cargo) {
    Say "cargo found: $($cargo.Source)"
} else {
    Say 'cargo was not found; the Rust GRIB bridges need a Rust toolchain.'
    if (-not $Yes) {
        $answer = ''
        try { $answer = Read-Host 'install: install rustup (https://rustup.rs) now? [y/N]' }
        catch { $answer = '' }
        if ($answer -match '^(y|yes)$') { $Yes = $true }
    }
    if (-not $Yes) {
        Fail ('cargo is missing and consent to install rustup was not ' +
              'given; re-run with -Yes (or GPUWM_INSTALL_YES=1), or ' +
              'install a Rust toolchain yourself and re-run')
    }
    Say 'installing rustup (stable toolchain, PATH left unmodified)'
    $rustupInit = Join-Path $env:TEMP 'rustup-init.exe'
    Invoke-WebRequest -UseBasicParsing 'https://win.rustup.rs/x86_64' `
        -OutFile $rustupInit
    Invoke-Step 'rustup-init' {
        & $rustupInit -y --no-modify-path
    }
    Remove-Item $rustupInit -ErrorAction SilentlyContinue
    $env:Path = "$cargoBin;$env:Path"
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        Fail 'rustup finished but cargo is still not on PATH'
    }
}

# ------------------------------------------------------- offline Rust build
Say 'building the vendored Rust GRIB bridges (offline, locked)'
Push-Location 'tools\grib1_bridge'
try {
    Invoke-Step 'cargo build' {
        cargo build --release --locked --offline
    }
} finally {
    Pop-Location
}
if ($NoRender) {
    Say 'skipping the tools\rustwx render engine (-NoRender);'
    Say 'gpuwm render uses the matplotlib fallback until it is built'
} else {
    Say 'building the vendored render engine in tools\rustwx (offline,'
    Say 'locked; the long pole of install -- skip with -NoRender)'
    Push-Location 'tools\rustwx'
    try {
        Invoke-Step 'cargo build (rustwx)' {
            cargo build --release --locked --offline
        }
    } finally {
        Pop-Location
    }
}

# ------------------------------------------------------------------ doctor
Say 'running gpuwm doctor'
& (Join-Path '.venv' 'Scripts\gpuwm.exe') doctor
$doctorExit = $LASTEXITCODE
if ($doctorExit -eq 0) {
    Say 'done -- doctor is clean.  Activate with: .\.venv\Scripts\Activate.ps1'
} else {
    Say 'install steps completed; doctor reports gaps above (each line'
    Say 'prints its own remedy).  Re-run .\install.ps1 any time.'
}
# `exit` would close an interactive console when this script arrives via
# `iwr | iex`, so only a file invocation propagates doctor's exit code that
# way; the piped form signals a doctor gap through a terminating error.
if ($MyInvocation.MyCommand.Path) {
    exit $doctorExit
} elseif ($doctorExit -ne 0) {
    throw "gpuwm doctor exited $doctorExit"
}

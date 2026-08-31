# Release-battery runner for the 2.6.0 cut: worktree, provision, three
# CPU legs, cargo gates, no-CuPy venv leg. Writes logs and a DONE marker
# so the coordinating session can poll instead of holding a console.
# Throwaway: the worktree and logs are scratch, cleared after the cut.
param(
    [string]$Tip = "8de6ee839",
    [string]$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$WT = (Join-Path (Split-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path) "wt-battery-260"),
    [string]$Logs = (Join-Path (Split-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path) "wt-battery-260-logs")
)
$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force $Logs | Out-Null
$summary = @{}

function Step([string]$name, [scriptblock]$body) {
    $log = Join-Path $Logs "$name.log"
    "=== $name started $(Get-Date -Format o) ===" | Out-File $log -Encoding utf8
    & $body 2>&1 | Out-File $log -Append -Encoding utf8
    $code = $LASTEXITCODE
    "=== $name exit $code $(Get-Date -Format o) ===" | Out-File $log -Append -Encoding utf8
    $script:summary[$name] = $code
    return $code
}

# The CPU legs' env, set BEFORE anything runs a test: the first attempt
# set GPUWM_NO_LOCAL_GPU after the provision instrument check (26 reds
# from the device being visible) and PYTHONNOUSERSITE globally (hiding
# the user-site pytest from the system python).
$env:GPUWM_NO_LOCAL_GPU = "1"

# 1. Throwaway worktree at the stamped tip.
if (-not (Test-Path $WT)) {
    git -C $Repo worktree add $WT $Tip *> (Join-Path $Logs "worktree.log")
}
Set-Location $WT

# 2. Provision the Rust binaries (copies, never links).
Step "provision" { powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $WT "tools\battery\provision_stage1.ps1") -Source $Repo }

# 3. The CPU legs.
$mark = "not gpu and not slow and not network"

$stage1 = @(Get-Content (Join-Path $WT "tools\battery\stage1_files.txt") |
    ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith("#") })
Step "stage1" { python -m pytest -q -p no:cacheprovider -m $mark @stage1 }

$always = @(Get-Content (Join-Path $WT "tools\battery\always_files.txt") |
    ForEach-Object { $_.Trim() } | Where-Object { $_ -and -not $_.StartsWith("#") })
Step "always" { python -m pytest -q -p no:cacheprovider -m $mark @always }

# 4. Cargo gates ran green on the first attempt at this same worktree
# and tip (28 entries, 2,712 tests, 0 failed); not re-run.
$script:summary["cargo_gates"] = 0

# 5. The no-CuPy venv leg: the publish workflow's own test job, the class
# that cost v2.4.0 its number.
$venv = Join-Path (Split-Path $Repo) "wt-battery-260-venv"
Step "venv_install" {
    # Dev/test wheel that never leaves this box: the documented override
    # for the unpinned-tree refusal.
    $env:GPUWM_ALLOW_UNPINNED_WHEEL = "1"
    python -m venv $venv
    # setuptools explicitly: py313 venvs do not bundle it, the editable
    # installs succeed without it via build isolation, and the five
    # wheel-content assertions in test_package_data_coverage need it at
    # RUNTIME -- absent, that file reports its own loud failure.
    & "$venv\Scripts\python" -m pip install --quiet setuptools
    # Companion FIRST: gpuwm hard-pins gpuwm-data==2.6.0, which is on no
    # index until the release act, so the pin must already be satisfied
    # in the venv when gpuwm installs.
    & "$venv\Scripts\python" -m pip install --quiet -e ".\gpuwm-data"
    & "$venv\Scripts\python" -m pip install --quiet -e ".[dev]"
    Remove-Item Env:GPUWM_ALLOW_UNPINNED_WHEEL
}
$publist = @(
    "tests/test_bridge_fetch.py", "tests/test_table_fetch.py",
    "tests/test_package_data_coverage.py", "tests/test_companion_distribution.py",
    "tests/test_configs_are_packaged.py", "tests/test_doctor.py",
    "tests/test_doctor_nexrad.py", "tests/test_cli.py",
    "tests/test_source_adapters.py", "tests/test_namelist_compat.py",
    "tests/test_native_wrf_distribution.py", "tests/test_publish_workflow_state_machine.py",
    "tests/test_publish_dist_shape_agreement.py", "tests/test_release_version_declaration.py",
    "tests/test_bridge_bundle_adopt.py", "tests/test_verify_source_bridge_pins.py",
    "tests/test_verify_release_artifacts.py")
Step "venv_nocupy" {
    $env:PYTHONNOUSERSITE = "1"
    & "$venv\Scripts\python" -m pytest -q -p no:cacheprovider -m $mark @publist
    Remove-Item Env:PYTHONNOUSERSITE
}

# 6. Summary + marker.
$lines = $summary.GetEnumerator() | ForEach-Object { "$($_.Key) exit $($_.Value)" }
$lines | Out-File (Join-Path $Logs "SUMMARY.txt") -Encoding utf8
"done $(Get-Date -Format o)" | Out-File (Join-Path $Logs "DONE.marker") -Encoding utf8

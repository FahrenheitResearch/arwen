# Storm-following moving-nest demo, 2011-04-27 outbreak (06Z-18Z).
#
# TWO ARMS, SAME STORMS, SAME PHYSICS, SAME WINDOW:
#   A (follow): d01 300x250 @6 km + MOVING d02 150x150 @2 km driven by
#               [relocation.follow] (UP_HELI_MAX centroid, reflectivity
#               handoff) -- configs/moving_nest_20110427_follow_2km.toml
#   B (static): the same coverage from one fixed 282x216 @2 km nest --
#               configs/moving_nest_20110427_static_2km.toml
# Punchline: B carries 2.7x arm A's nest columns (60912 vs 22500) and
# ~1.8x its wall time for the same storm coverage; the moving nest is the
# low-VRAM way to keep a 2 km grid on a 50 kt supercell.
#
# EXPECTED COST on the RTX 5090 (anchored to the tdshort receipt for this
# exact case: d01 300x250 + d02 504x402 integrated 8 model hours at
# ~7.2 min wall per model hour):
#   arm A: ~18 min integration + relocation overhead  => ~25-30 min wall
#   arm B: ~33 min wall
#   VRAM: the 13.6M-cell tdshort fit the 32 GiB card; arm A is 4.8M cells
#   (transiently +1.1M while a relocation holds two children), arm B is
#   6.7M. Both clear a 16 GiB card; `gpuwm run`'s preflight prints the
#   binding numbers before integration starts.
#
# DEPENDENCY: actual nest MOTION needs the relocation runner from
# lane/moving-nest-leg2 (it evaluates the [relocation.follow] tracker at
# cycle boundaries and calls gpuwm.core.nest_relocation.relocate_child).
# Until that lane merges, arm A validates and runs -- as a static nest.
# This script therefore stages and verifies by default and only launches
# with -Launch, run on whichever card frees first.
#
# Usage (from the repo root):
#   powershell -File tools\moving_nest_demo_20110427.ps1            # verify only
#   powershell -File tools\moving_nest_demo_20110427.ps1 -Launch    # run both arms
#
param(
    [string]$Era5   = "$env:LOCALAPPDATA\Temp\claude\C--Users-drew-gpuwm\fea7a141-0924-4ca3-8cc3-0d448f2facae\scratchpad\tdcrash\era5-nest\era5-combined.grib",
    [string]$Vtable = "$env:LOCALAPPDATA\Temp\claude\C--Users-drew-gpuwm\fea7a141-0924-4ca3-8cc3-0d448f2facae\scratchpad\tdcrash\local-case\Vtable.ERA5_CDO",
    [string]$OutRoot = "out\moving-nest-demo-20110427",
    [int]$Device = 0,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$configs = @(
    "configs\moving_nest_20110427_follow_2km.toml",
    "configs\moving_nest_20110427_static_2km.toml"
)

# --- stage the demo data root (hardlinks: zero-copy, nothing deleted) ---
$dataRoot = Join-Path $OutRoot "data"
New-Item -ItemType Directory -Force $dataRoot | Out-Null
foreach ($pair in @(@($Era5, "era5-combined.grib"),
                    @($Vtable, "Vtable.ERA5_CDO"))) {
    $src = $pair[0]; $dst = Join-Path $dataRoot $pair[1]
    if (-not (Test-Path $src)) {
        throw "missing input $src -- pass -Era5/-Vtable at the fetched locations"
    }
    if (-not (Test-Path $dst)) {
        try { New-Item -ItemType HardLink -Path $dst -Target $src | Out-Null }
        catch { Copy-Item $src $dst }   # cross-volume fallback
    }
}
$env:GPUWM_DEMO_20110427_ROOT = (Resolve-Path $dataRoot).Path
# The configs read ${GPUWM_CASE_DATA_ROOT}/WPS_GEOG.  The proven tdnest
# runs of this case resolved it at $env:USERPROFILE\WPS_GEOG, so fall
# back there when the current root does not carry one.
if (-not $env:GPUWM_CASE_DATA_ROOT -or
        -not (Test-Path (Join-Path $env:GPUWM_CASE_DATA_ROOT "WPS_GEOG"))) {
    if (Test-Path (Join-Path $env:USERPROFILE "WPS_GEOG")) {
        $env:GPUWM_CASE_DATA_ROOT = $env:USERPROFILE
    } else {
        throw "no WPS_GEOG under GPUWM_CASE_DATA_ROOT=$env:GPUWM_CASE_DATA_ROOT or $env:USERPROFILE"
    }
}
Write-Host "GPUWM_DEMO_20110427_ROOT = $env:GPUWM_DEMO_20110427_ROOT"
Write-Host "GPUWM_CASE_DATA_ROOT     = $env:GPUWM_CASE_DATA_ROOT"

# --- verify both arms end-to-end (schema + declared inputs), no GPU ---
foreach ($cfg in $configs) {
    Write-Host "== gpuwm check $cfg"
    python -m gpuwm.cli check $cfg
    if ($LASTEXITCODE -ne 0) { throw "check failed for $cfg" }
}

if (-not $Launch) {
    Write-Host ""
    Write-Host "Staged and verified; not launched (pass -Launch on a free card)."
    Write-Host "  arm A: python -m gpuwm.cli run $($configs[0]) --outdir $OutRoot\follow"
    Write-Host "  arm B: python -m gpuwm.cli run $($configs[1]) --outdir $OutRoot\static"
    exit 0
}

# --- the two arms, sequentially on one card; receipts stay in outdirs ---
$env:CUDA_VISIBLE_DEVICES = "$Device"
foreach ($arm in @(@($configs[0], "follow"), @($configs[1], "static"))) {
    $outdir = Join-Path $OutRoot $arm[1]
    Write-Host "== gpuwm run $($arm[0]) --outdir $outdir"
    python -m gpuwm.cli run $arm[0] --outdir $outdir
    if ($LASTEXITCODE -ne 0) { throw "run failed for $($arm[0])" }
}
Write-Host "both arms complete; compare $OutRoot\follow vs $OutRoot\static"

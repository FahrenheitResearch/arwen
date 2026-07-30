Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

function Resolve-PayloadPath {
    param([string]$Relative)
    if ([string]::IsNullOrWhiteSpace($Relative) -or
            [System.IO.Path]::IsPathRooted($Relative) -or
            $Relative.Contains("\") -or $Relative.Contains(":")) {
        throw "Invalid SHA256SUMS path: '$Relative'"
    }
    $parts = $Relative.Split("/")
    if ($parts -contains "" -or $parts -contains "." -or $parts -contains "..") {
        throw "Unsafe SHA256SUMS path: '$Relative'"
    }
    $candidate = [System.IO.Path]::GetFullPath(
        [System.IO.Path]::Combine($Root, $Relative.Replace("/", "\")))
    $prefix = $Root.TrimEnd("\") + "\"
    if (-not $candidate.StartsWith(
            $prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "SHA256SUMS path escapes release root: '$Relative'"
    }
    return $candidate
}

$sumPath = Join-Path $Root "SHA256SUMS"
if (-not (Test-Path -LiteralPath $sumPath -PathType Leaf)) {
    throw "Distribution SHA256SUMS is missing: '$sumPath'"
}
$seen = @{}
foreach ($line in [System.IO.File]::ReadAllLines($sumPath)) {
    if ($line -notmatch '^([0-9a-f]{64})  (.+)$') {
        throw "Malformed SHA256SUMS line: '$line'"
    }
    $expected = $Matches[1]
    $relative = $Matches[2]
    if ($seen.ContainsKey($relative)) {
        throw "Duplicate SHA256SUMS path: '$relative'"
    }
    $seen[$relative] = $true
    $path = Resolve-PayloadPath $relative
    $item = Get-Item -LiteralPath $path -Force -ErrorAction Stop
    if ($item.PSIsContainer -or
            ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint)) {
        throw "Release payload is not a regular file: '$relative'"
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -cne $expected) {
        throw "Release payload hash mismatch: '$relative'"
    }
}
if ($seen.Count -lt 1) {
    throw "SHA256SUMS contains no payload records"
}

$runtimeText = if ($env:GPUWM_INSTALL_ROOT) {
    $env:GPUWM_INSTALL_ROOT
} else {
    Join-Path $Root "runtime"
}
$runtime = [System.IO.Path]::GetFullPath($runtimeText)
$pythonRecord = Join-Path $runtime ".gpuwm-python"
if (-not (Test-Path -LiteralPath (Join-Path $runtime "gpuwm") -PathType Container) -or
        -not (Test-Path -LiteralPath $pythonRecord -PathType Leaf)) {
    throw "RW-WPS runtime is not installed; run '$Root\install.ps1 -SkipGpu' first"
}
$installedPython = [System.IO.Path]::GetFullPath(
    [System.IO.File]::ReadAllText($pythonRecord).Trim())
$pythonText = if ($env:GPUWM_PYTHON) { $env:GPUWM_PYTHON } else { $installedPython }
$pythonCommand = Get-Command $pythonText -CommandType Application -ErrorAction Stop
$python = [System.IO.Path]::GetFullPath($pythonCommand.Source)
if (-not $python.Equals(
        $installedPython, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "GPUWM_PYTHON differs from the interpreter used at install"
}

for ($index = 0; $index -lt $args.Count; $index++) {
    $argument = [string]$args[$index]
    if ($argument -eq "--preprocess-backend=cuda" -or
            ($argument -eq "--preprocess-backend" -and
             $index + 1 -lt $args.Count -and $args[$index + 1] -eq "cuda")) {
        throw "The sealed Windows release does not include the CUDA backend; use cpu or auto"
    }
}

$oldPythonPath = $env:PYTHONPATH
$oldPythonNoUserSite = $env:PYTHONNOUSERSITE
$oldPythonDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
try {
    $env:PYTHONPATH = $runtime
    # These environment controls are intentionally inherited by every Python
    # child spawned by source_cli (preparation, benchmark, and worker jobs).
    # Command-line -B/-P flags protect only this immediate interpreter.
    $env:PYTHONNOUSERSITE = "1"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    & $python -B -P -m gpuwm.native_wrf_distribution `
        --bridge-dir (Join-Path $Root "libexec\bridges") --skip-gpu | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Runtime verification failed with exit code $LASTEXITCODE"
    }

    $env:GPUWM_PYTHON = $python
    $env:GPUWM_HRRR_DECODER = Join-Path $Root "libexec\bridges\hrrr_grib2_bridge.exe"
    $env:GPUWM_GRIB1_BRIDGE = Join-Path $Root "libexec\bridges\grib1_bridge.exe"
    $env:GPUWM_GRIB2_INVENTORY = Join-Path $Root "libexec\bridges\grib2_inventory.exe"
    $env:GPUWM_GRIB2_DUMP = Join-Path $Root "libexec\bridges\grib2_dump.exe"
    $env:GPUWM_GFS_GRIB2_BRIDGE = Join-Path $Root "libexec\bridges\gfs_grib2_bridge.exe"
    $env:GPUWM_CPU_PREPROCESS_BRIDGE = Join-Path $Root "libexec\bridges\gpuwm_preprocess_cpu.dll"
    $env:GPUWM_NATIVE_DISTRIBUTION_MANIFEST = Join-Path $Root "manifest.json"

    & $python -B -P -m gpuwm.source_cli @args
    exit $LASTEXITCODE
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONNOUSERSITE = $oldPythonNoUserSite
    $env:PYTHONDONTWRITEBYTECODE = $oldPythonDontWriteBytecode
}

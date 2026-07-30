[CmdletBinding()]
param(
    [switch]$SkipGpu
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $SkipGpu) {
    throw "The sealed Windows x86_64 release is CPU-only; pass -SkipGpu."
}

$Root = [System.IO.Path]::GetFullPath($PSScriptRoot)

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

function Test-ReleaseHashes {
    $sumPath = Join-Path $Root "SHA256SUMS"
    if (-not (Test-Path -LiteralPath $sumPath -PathType Leaf)) {
        throw "Distribution SHA256SUMS is missing: $sumPath"
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
}

Test-ReleaseHashes

$pythonText = if ($env:GPUWM_PYTHON) { $env:GPUWM_PYTHON } else { "python.exe" }
$pythonCommand = Get-Command $pythonText -CommandType Application -ErrorAction Stop
$python = [System.IO.Path]::GetFullPath($pythonCommand.Source)

$targetText = if ($env:GPUWM_INSTALL_ROOT) {
    $env:GPUWM_INSTALL_ROOT
} else {
    Join-Path $Root "runtime"
}
$target = [System.IO.Path]::GetFullPath($targetText)
$targetParent = [System.IO.Directory]::GetParent($target)
if ($null -eq $targetParent -or -not $targetParent.Exists) {
    throw "GPUWM_INSTALL_ROOT parent does not exist: '$targetText'"
}
if (Test-Path -LiteralPath $target) {
    throw "Refusing existing GPUWM_INSTALL_ROOT: '$target'"
}

$wheels = @(Get-ChildItem -LiteralPath (Join-Path $Root "wheel") `
    -Filter "rw_wps-*.whl" -File)
if ($wheels.Count -ne 1) {
    throw "Distribution must contain exactly one rw-wps wheel"
}

$nonce = ([System.Guid]::NewGuid()).ToString("N").Substring(0, 8)
$partial = Join-Path $targetParent.FullName (".rw-{0}-{1}" -f $PID, $nonce)
$partial = [System.IO.Path]::GetFullPath($partial)
$parentPrefix = $targetParent.FullName.TrimEnd("\") + "\"
$partialLeaf = [System.IO.Path]::GetFileName($partial)
if (-not $partial.StartsWith(
            $parentPrefix, [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $partialLeaf.StartsWith(
            ".rw-", [System.StringComparison]::Ordinal) -or
        $partialLeaf.Length -gt 32) {
    throw "Refusing unsafe temporary install path: '$partial'"
}

# Windows PowerShell 5.1 still encounters classic MAX_PATH behavior in deep
# extracted bundles.  Account for pip's generated __pycache__ names before
# creating a partial tree so an install fails cleanly instead of silently
# omitting the longest helper and then failing during verification/cleanup.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$cacheProjectionScript = @'
import importlib.util
import sys

source = "pkg/module.py"
optimizations = ("", "1", "2", str(sys.flags.optimize))
print(max(
    len(importlib.util.cache_from_source(source, optimization=value))
    - len(source)
    for value in optimizations
))
'@
$cacheGrowthText = $cacheProjectionScript | & $python -
if ($LASTEXITCODE -ne 0) {
    throw "Could not project Python bytecode cache paths"
}
$cacheGrowth = 0
if (-not [int]::TryParse(
        (($cacheGrowthText | Out-String).Trim()), [ref]$cacheGrowth) -or
        $cacheGrowth -lt 0 -or $cacheGrowth -gt 512) {
    throw "Python returned an invalid bytecode cache path projection"
}
$wheelArchive = [System.IO.Compression.ZipFile]::OpenRead($wheels[0].FullName)
$maxInstalledRelativeLength = 0
try {
    foreach ($entry in $wheelArchive.Entries) {
        $relative = $entry.FullName.Replace("/", "\")
        $projectedRelativeLength = $relative.Length
        if ($relative.EndsWith(
                ".py", [System.StringComparison]::OrdinalIgnoreCase)) {
            $projectedRelativeLength += $cacheGrowth
        }
        if ($projectedRelativeLength -gt $maxInstalledRelativeLength) {
            $maxInstalledRelativeLength = $projectedRelativeLength
        }
    }
} finally {
    $wheelArchive.Dispose()
}
$receiptTemporaryLength = (
    "native-wrf-runtime-receipt.json.partial-{0}" -f $PID).Length
$maxInstalledRelativeLength = [System.Math]::Max(
    $maxInstalledRelativeLength, $receiptTemporaryLength)
$installRootLength = [System.Math]::Max($partial.Length, $target.Length)
$projectedLongestPath = $installRootLength + 1 + $maxInstalledRelativeLength
if ($projectedLongestPath -ge 248) {
    throw (
        "GPUWM install path is too deep for Windows PowerShell 5.1: " +
        "projected path length $projectedLongestPath must be below 248; " +
        "extract or install from a shorter parent path")
}

New-Item -ItemType Directory -Path $partial -ErrorAction Stop | Out-Null
try {
    & $python -m pip install --disable-pip-version-check --no-index --no-deps `
        --target $partial $wheels[0].FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Offline rw-wps wheel install failed with exit code $LASTEXITCODE"
    }
    [System.IO.File]::WriteAllText(
        (Join-Path $partial ".gpuwm-python"), $python,
        [System.Text.UTF8Encoding]::new($false))

    $oldPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONPATH = $partial
        & $python -P -m gpuwm.native_wrf_distribution `
            --bridge-dir (Join-Path $Root "libexec\bridges") `
            --receipt (Join-Path $partial "native-wrf-runtime-receipt.json") `
            --skip-gpu
        if ($LASTEXITCODE -ne 0) {
            throw "Installed runtime verification failed with exit code $LASTEXITCODE"
        }
    } finally {
        $env:PYTHONPATH = $oldPythonPath
    }
    # Directory.Move is an atomic same-volume rename and fails if a racing
    # installer created the target.  Move-Item would instead nest $partial
    # inside that target and incorrectly report success.
    [System.IO.Directory]::Move($partial, $target)
    $partial = $null
} finally {
    if ($null -ne $partial -and (Test-Path -LiteralPath $partial)) {
        Remove-Item -LiteralPath $partial -Recurse -Force
    }
}

Write-Output "PASS rw_wps_install=$target python=$python platform=windows-x86_64-cpu"

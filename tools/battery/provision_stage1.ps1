<#
.SYNOPSIS
    Stage the built release binaries into a checkout so the stage-1 battery
    tests the RUST artifacts instead of skipping past them.

.DESCRIPTION
    WHAT THIS IS

      The release battery is run in a throwaway `git worktree` at the stamped
      tip, never in a shared checkout.  A fresh worktree carries every TRACKED
      file -- including `tools/rustwx/assets/basemap`, which the renderer needs
      -- but the Rust binaries under `tools/<crate>/target/release/` are build
      output and `.gitignore`d, so a fresh worktree has NONE of them.

    THE BREAKAGE THIS PREVENTS

      Every suite that drives a real binary is written to SKIP honestly when
      the binary is absent (`tests/test_render_rust.py` reads the product's own
      `renderer_refusal()` gate).  So an unprovisioned worktree does not go
      red.  It goes GREEN with the rust engine, the mapped engine, the GRIB
      bridges and the NetCDF writer silently untested -- a release certified
      on the Python halves of paths whose whole point is that they are Rust.
      That is a fake green, and it is invisible in a pass count.

      The second breakage is subtler and cost two runs.  The binaries must be
      COPIED to the target path, never symlinked or junctioned.
      `gpuwm.rustwx.find_renderer()` calls `.resolve()` on what it finds, and
      the renderer locates basemaps by walking the first six ANCESTORS of its
      own executable path looking for `assets/basemap`.  A link resolves back
      to the donor tree, so the renderer certifies the DONOR's basemaps and
      the DONOR's provenance, not the tip under test.

    WHY IT IS A COMMITTED SCRIPT

      It had never been one.  The only pointer to the step was a comment in
      `tools/battery/stage1_files.txt` naming a scratchpad file,
      `q_TEMPLATE_stage1.ps1`, that no longer exists on any machine -- the
      exact lost-scratchpad failure `stage1_files.txt` itself was created to
      close (task #122), reappearing one level up in the same battery.

.PARAMETER Target
    Tree to provision.  Defaults to the repository root that contains THIS
    script, which is what you want when the script is invoked from inside the
    gate worktree.

.PARAMETER Source
    Donor tree whose `tools/<crate>/target/release/` directories are already
    built.  Defaults to the main checkout that owns the worktree, discovered
    from git, so the common case takes no argument.

.PARAMETER SkipVerify
    Skip the `tests/test_render_rust.py` instrument check.  For staging a tree
    you are going to verify by other means; a gate run should never pass this.

.PARAMETER Python
    Interpreter used for the verify step.

.EXAMPLE
    pwsh -File tools/battery/provision_stage1.ps1

    Provision the worktree the script lives in from its main checkout and
    prove the provisioning took.

.NOTES
    Re-runnable.  Copies are unconditional and overwrite in place, so running
    it twice is a no-op with the same report.  Exit code 0 only when all
    required artifacts are present AND the instrument check passed.
#>
[CmdletBinding()]
param(
    [string]$Target,
    [string]$Source,
    [switch]$SkipVerify,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# The manifest.
#
# Repository-relative paths, one per built artifact.  This is a CURATED list
# in the same spirit as stage1_files.txt: an artifact that appears in a crate
# but not here is reported as an extra and still copied, while an artifact
# named here and missing from the donor is a hard failure.  That asymmetry is
# deliberate -- a new binary should not fail the gate on the day it is added,
# but a binary that stops being built must not silently drop out of coverage.
#
# Amendment discipline matches stage1_files.txt: an addition carries its
# reason inline, a removal says why the coverage is gone, in the commit.
# ---------------------------------------------------------------------------
$Manifest = @(
    # GRIB decode + the MET intermediate writer.  The `--engine rust` fetch
    # and prep paths; `gpuwm_preprocess_cpu.dll` is the regrid/transform half.
    'tools/grib1_bridge/target/release/gfs_grib2_bridge.exe'
    'tools/grib1_bridge/target/release/gpuwm_preprocess_cpu.dll'
    'tools/grib1_bridge/target/release/grib1_bridge.exe'
    'tools/grib1_bridge/target/release/grib2_dump.exe'
    'tools/grib1_bridge/target/release/grib2_inventory.exe'
    'tools/grib1_bridge/target/release/hrrr_grib2_bridge.exe'
    'tools/grib1_bridge/target/release/met_intermediate.exe'

    # Radial-velocity dealiasing.
    'tools/region_global_dealias/target/release/region_global_dealias.dll'

    # The rustwx family: renderer, obs ingest, NetCDF I/O, scoring.
    # `rw_wrfbatch.exe` is the one the render law names.
    'tools/rustwx/target/release/netcdf_writer.dll'
    'tools/rustwx/target/release/obs_regrid.dll'
    'tools/rustwx/target/release/rw_asos.exe'
    'tools/rustwx/target/release/rw_ensbatch.exe'
    'tools/rustwx/target/release/rw_fetch.exe'
    'tools/rustwx/target/release/rw_fieldcmp.exe'
    'tools/rustwx/target/release/rw_goes.exe'
    'tools/rustwx/target/release/rw_mpas_convert.exe'
    'tools/rustwx/target/release/rw_mpas_init.exe'
    'tools/rustwx/target/release/rw_mrms.exe'
    'tools/rustwx/target/release/rw_netcdf.exe'
    'tools/rustwx/target/release/rw_nexrad.exe'
    'tools/rustwx/target/release/rw_obsgrid.exe'
    'tools/rustwx/target/release/rw_odim.exe'
    'tools/rustwx/target/release/rw_opera.exe'
    'tools/rustwx/target/release/rw_runscore.exe'
    'tools/rustwx/target/release/rw_stage4.exe'
    'tools/rustwx/target/release/rw_wrfbatch.exe'
    'tools/rustwx/target/release/static_fields.dll'

    # WPS replacement + the mapped-source engine `gpuwm doctor` probes.
    'tools/rw_wps/target/release/gpuwm_mapped_engine.exe'
    'tools/rw_wps/target/release/rw-wps.exe'
)

function Resolve-RepoRoot {
    param([string]$Start)
    $probe = (Resolve-Path -LiteralPath $Start).Path
    while ($probe) {
        if (Test-Path -LiteralPath (Join-Path $probe '.git')) { return $probe }
        $parent = Split-Path -Parent $probe
        if ($parent -eq $probe) { break }
        $probe = $parent
    }
    throw "not inside a git checkout: $Start"
}

# --- Target -----------------------------------------------------------------
if (-not $Target) {
    # <root>/tools/battery/provision_stage1.ps1 -> <root>
    $Target = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
}
$Target = (Resolve-Path -LiteralPath $Target).Path
$Target = Resolve-RepoRoot -Start $Target

# --- Source -----------------------------------------------------------------
# A linked worktree's common git dir is `<main>/.git`, so the main checkout is
# its parent.  In the main checkout itself the common dir IS `<target>/.git`,
# which would make the tree its own donor -- caught below.
if (-not $Source) {
    $commonDir = & git -C $Target rev-parse --path-format=absolute --git-common-dir
    if ($LASTEXITCODE -ne 0) { throw "git rev-parse failed in $Target" }
    $Source = Split-Path -Parent ($commonDir.Trim())
}
$Source = (Resolve-Path -LiteralPath $Source).Path

if ($Source -eq $Target) {
    throw ("donor tree and target tree are the same path ($Target).  " +
           "Pass -Source <checkout that has tools/*/target/release built>.")
}

Write-Host "provision_stage1: target = $Target"
Write-Host "provision_stage1: source = $Source"

# --- Copy -------------------------------------------------------------------
# Every *.exe/*.dll sitting directly in a crate's release dir, so a binary the
# manifest has not caught up with is still staged.  Copy-Item, never a link:
# see THE BREAKAGE THIS PREVENTS above.
$copied = New-Object System.Collections.Generic.List[string]
$crateDirs = Get-ChildItem -LiteralPath (Join-Path $Source 'tools') -Directory -ErrorAction SilentlyContinue
foreach ($crate in $crateDirs) {
    $srcDir = Join-Path $crate.FullName 'target\release'
    if (-not (Test-Path -LiteralPath $srcDir)) { continue }
    $dstDir = Join-Path $Target ("tools\" + $crate.Name + "\target\release")
    $artifacts = Get-ChildItem -LiteralPath $srcDir -File | Where-Object { $_.Extension -in '.exe', '.dll' }
    if (-not $artifacts) { continue }
    if (-not (Test-Path -LiteralPath $dstDir)) {
        New-Item -ItemType Directory -Path $dstDir -Force | Out-Null
    }
    foreach ($artifact in $artifacts) {
        Copy-Item -LiteralPath $artifact.FullName -Destination (Join-Path $dstDir $artifact.Name) -Force
        $copied.Add(("tools/" + $crate.Name + "/target/release/" + $artifact.Name))
    }
}

Write-Host ("provision_stage1: copied {0} artifact(s)" -f $copied.Count)

# --- Manifest check ---------------------------------------------------------
$missing = New-Object System.Collections.Generic.List[string]
foreach ($rel in $Manifest) {
    $abs = Join-Path $Target ($rel -replace '/', '\')
    if (-not (Test-Path -LiteralPath $abs -PathType Leaf)) { $missing.Add($rel) }
}
$extra = @($copied | Where-Object { $Manifest -notcontains $_ })

if ($extra.Count -gt 0) {
    Write-Host ("provision_stage1: NOTE {0} artifact(s) staged that the manifest does not name:" -f $extra.Count)
    foreach ($e in $extra) { Write-Host "    $e" }
    Write-Host "    Add them with their reason, or say in the commit why they are not gate surface."
}

if ($missing.Count -gt 0) {
    Write-Host ""
    Write-Host ("provision_stage1: FAIL -- {0} required artifact(s) absent from the donor:" -f $missing.Count)
    foreach ($m in $missing) { Write-Host "    $m" }
    Write-Host ""
    Write-Host "The donor tree has not built them.  Build the crate (cargo build --release"
    Write-Host "in tools/<crate>) or point -Source at a tree that has, then re-run."
    exit 1
}
Write-Host ("provision_stage1: manifest OK -- {0}/{0} required artifacts present" -f $Manifest.Count)

# --- Instrument check -------------------------------------------------------
# The provisioning is only real if the suites STOP SKIPPING.  Asserting on
# file existence would re-certify the same assumption the copy just made;
# tests/test_render_rust.py answers the question that matters, because its
# gate is the product's own renderer_refusal() -- it skips a foreign or
# unresolvable engine even when a file sits at the path.  45 passed / 0
# skipped is what a provisioned tree produces.  Any skip means the binaries
# did not take effect and the green that follows is fake.
if ($SkipVerify) {
    Write-Host "provision_stage1: verify SKIPPED by request -- this tree is not gate-ready."
    exit 0
}

$expectPassed = 45
$env:PYTHONPATH = ($Target + ';' + (Join-Path $Target 'gpuwm-data'))
# Do NOT set PYTHONSAFEPATH: it drops the target from sys.path and the
# editable install's .pth then certifies whatever checkout it points at.
Remove-Item Env:\PYTHONSAFEPATH -ErrorAction SilentlyContinue
$env:GPUWM_NO_LOCAL_GPU = '1'

Write-Host ""
Write-Host "provision_stage1: instrument check -- tests/test_render_rust.py"
Push-Location $Target
try {
    $out = & $Python -m pytest tests/test_render_rust.py -q --no-header -p no:cacheprovider 2>&1
} finally {
    Pop-Location
}
$tail = ($out | Select-Object -Last 3) -join "`n"
Write-Host $tail

$text = ($out -join "`n")
$passed = 0
$skipped = 0
if ($text -match '(\d+)\s+passed') { $passed = [int]$Matches[1] }
if ($text -match '(\d+)\s+skipped') { $skipped = [int]$Matches[1] }

if ($passed -ne $expectPassed -or $skipped -ne 0) {
    Write-Host ""
    Write-Host ("provision_stage1: FAIL -- expected {0} passed / 0 skipped, got {1} passed / {2} skipped." -f $expectPassed, $passed, $skipped)
    Write-Host "A skip here means the renderer at the staged path is not the engine the"
    Write-Host "render path will accept -- a link instead of a copy, a stale build, or a"
    Write-Host "GPUWM_RENDERER/RUSTWX_ASSETS_DIR override pointing elsewhere.  Do not run"
    Write-Host "the battery on this tree: its green would not cover the rust engine."
    exit 1
}

Write-Host ""
Write-Host ("provision_stage1: OK -- {0} artifacts staged, instrument check {1} passed / 0 skipped." -f $copied.Count, $passed)
exit 0

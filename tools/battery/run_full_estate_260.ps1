# Full CPU estate sweep: every test under tests/ and tilestream/, CPU
# markers only. NOT A RELEASE LEG AND NEVER BECOMES ONE (owner ruling
# 2026-08-30: the curated stage-1 list is the gate, permanently -- running
# everything per cut cost hours). This is an occasional deliberate stray
# hunt for tests rotting outside the curated lists, run before a
# no-stragglers cut or after a large landing. Writes a log and a DONE
# marker.
$ErrorActionPreference = "Continue"
$Logs = Join-Path (Split-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path) "wt-estate-260-logs"
New-Item -ItemType Directory -Force $Logs | Out-Null
Set-Location (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$env:GPUWM_NO_LOCAL_GPU = "1"
"=== estate started $(Get-Date -Format o) ===" | Out-File "$Logs\estate.log" -Encoding utf8
python -m pytest -q -p no:cacheprovider -m "not gpu and not slow and not network" tests tilestream 2>&1 |
    Out-File "$Logs\estate.log" -Append -Encoding utf8
"=== estate exit $LASTEXITCODE $(Get-Date -Format o) ===" | Out-File "$Logs\estate.log" -Append -Encoding utf8
"done" | Out-File "$Logs\DONE.marker" -Encoding utf8

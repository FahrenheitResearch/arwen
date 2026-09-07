# Installed gpuwm 2.7 CLI and terminal acceptance

Run these checks once on Windows x86-64 and once on Linux x86-64. Use a new
acceptance directory outside every checkout and the exact candidate artifacts.
The native builds include `tools/arwen-tui` alongside the four engine/bridge
workspaces. All five builds use the candidate's full source revision stamp.

The commands below assume the candidate package output contains `pure/` and
`companion/`, its generated manifest is `release-assets.json`, and `bundles/`
contains both `gpuwm-bridges-v2.7.0-{win,linux}-x86_64.zip` archives. Replace
the example directories and `FULL_40_HEX_CANDIDATE_COMMIT` with the retained
candidate paths and revision. The pins path names the generated, pinned source
tree used to build these distributions.

## Windows PowerShell

```powershell
$candidateSource = 'C:\release\arwen-source'
$candidateArtifacts = 'C:\release\candidate-packages'
$candidateAcceptance = 'C:\release\accept-windows-new'
$candidateRevision = 'FULL_40_HEX_CANDIDATE_COMMIT'
New-Item -ItemType Directory -Path $candidateAcceptance -ErrorAction Stop
python -m venv (Join-Path $candidateAcceptance 'venv')
$candidatePython = Join-Path $candidateAcceptance 'venv\Scripts\python.exe'
Set-Location $candidateAcceptance
Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
& $candidatePython -I -m pip install --no-cache-dir (Join-Path $candidateArtifacts 'pure\gpuwm-2.7.0-py3-none-any.whl') (Join-Path $candidateArtifacts 'companion\gpuwm_data-2.7.0-py3-none-any.whl')
& $candidatePython -I -m pip check
& $candidatePython -I (Join-Path $candidateSource 'tools\verify_release_artifacts.py') --wheel (Join-Path $candidateArtifacts 'pure\gpuwm-2.7.0-py3-none-any.whl') --sdist (Join-Path $candidateArtifacts 'pure\gpuwm-2.7.0.tar.gz') --pins (Join-Path $candidateSource 'gpuwm\data\bridges\bridge-pins.json') --manifest (Join-Path $candidateArtifacts 'release-assets.json') --bundles (Join-Path $candidateArtifacts 'bundles') --release v2.7.0 --source-rev $candidateRevision --repo-root $candidateSource --stage (Join-Path $candidateAcceptance 'verified-native') --receipt (Join-Path $candidateAcceptance 'artifact-proof.json')
$env:GPUWM_TUI_BIN = Join-Path $candidateAcceptance 'verified-native\arwen-tui.exe'
& $candidatePython -I -m gpuwm.cli --help
& $candidatePython -I -m gpuwm.cli version
& $candidatePython -I -m gpuwm.cli tui --help
& $candidatePython -I -m gpuwm.cli tui --snapshot "Drew's terminal preview.html"
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath "Drew's terminal preview.html")) { throw 'Installed TUI snapshot failed' }
```

Every command must exit zero. The TUI override binds this subprocess to the
freshly verified bundle. With a platform wheel, repeat in another new venv using
`gpuwm-2.7.0-py3-none-win_amd64.whl`, remove `Env:GPUWM_TUI_BIN`, and run the same
snapshot command to exercise the executable inside `site-packages/gpuwm/libexec`.

## Linux shell

```bash
set -euo pipefail
candidate_source=/release/arwen-source
candidate_artifacts=/release/candidate-packages
candidate_acceptance=/release/accept-linux-new
candidate_revision=FULL_40_HEX_CANDIDATE_COMMIT
mkdir "$candidate_acceptance"
python3 -m venv "$candidate_acceptance/venv"
candidate_python="$candidate_acceptance/venv/bin/python"
cd "$candidate_acceptance"
unset PYTHONPATH
"$candidate_python" -I -m pip install --no-cache-dir "$candidate_artifacts/pure/gpuwm-2.7.0-py3-none-any.whl" "$candidate_artifacts/companion/gpuwm_data-2.7.0-py3-none-any.whl"
"$candidate_python" -I -m pip check
"$candidate_python" -I "$candidate_source/tools/verify_release_artifacts.py" --wheel "$candidate_artifacts/pure/gpuwm-2.7.0-py3-none-any.whl" --sdist "$candidate_artifacts/pure/gpuwm-2.7.0.tar.gz" --pins "$candidate_source/gpuwm/data/bridges/bridge-pins.json" --manifest "$candidate_artifacts/release-assets.json" --bundles "$candidate_artifacts/bundles" --release v2.7.0 --source-rev "$candidate_revision" --repo-root "$candidate_source" --stage "$candidate_acceptance/verified-native" --receipt "$candidate_acceptance/artifact-proof.json"
export GPUWM_TUI_BIN="$candidate_acceptance/verified-native/arwen-tui"
"$candidate_python" -I -m gpuwm.cli --help
"$candidate_python" -I -m gpuwm.cli version
"$candidate_python" -I -m gpuwm.cli tui --help
"$candidate_python" -I -m gpuwm.cli tui --snapshot "Drew's terminal preview.html"
test -s "Drew's terminal preview.html"
```

With a Linux platform wheel, repeat in another new venv using
`gpuwm-2.7.0-py3-none-manylinux_2_28_x86_64.whl`, unset `GPUWM_TUI_BIN`, and run
the same snapshot command to exercise the installed native payload. Acceptance
on one Linux host establishes that host's compatibility; the wheel tag alone
does not prove execution on older glibc versions.

## Companion and installation identity

Run this code using the acceptance interpreter's `-I -c` option on each OS:

```python
import hashlib
from importlib.metadata import version
from pathlib import Path
import sys
import gpuwm
import gpuwm_data
from gpuwm import data_assets

assert version("gpuwm") == version("gpuwm-data") == "2.7.0"
for module in (gpuwm, gpuwm_data):
    assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
table = data_assets.data_path("rrtmgp/rrtmgp-gas-lw-g256.nc")
assert table.resolve().is_relative_to(Path(gpuwm_data.__file__).resolve().parent)
assert hashlib.sha256(table.read_bytes()).hexdigest() == "4048360199d1917ed8f2ccaae2ec097d0f990da3bbad9830337b739b4fa01be7"
print(gpuwm.__file__, gpuwm_data.__file__, table)
```

The verifier checks both archives' membership, sizes, hashes and source stamps,
then stages and probes this host's actual native executables/libraries. Retain
its JSON receipt with the snapshot and installed package inventory. Open
`gpuwm tui` in an interactive terminal for the final keyboard/mouse check;
`--snapshot` verifies installed launch and rendering without starting a job.

# Install the forward-only git hooks on Windows.  Same arguments as
# install.sh, which does the work through Git for Windows' own sh (git
# runs hooks with that sh anyway):
#
#   powershell -File tools\git-hooks\install.ps1 [--all-worktrees] [--refuse-all-push] [repo ...]

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$execPath = (& git --exec-path).Trim()
# <git root>\mingw64\libexec\git-core -> <git root>
$gitRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $execPath))
$sh = Join-Path $gitRoot 'bin\sh.exe'
if (-not (Test-Path $sh)) { $sh = Join-Path $gitRoot 'usr\bin\sh.exe' }
if (-not (Test-Path $sh)) { throw "Git for Windows sh.exe not found under $gitRoot" }
$script = (Join-Path $here 'install.sh') -replace '\\', '/'
$fwd = @($args | ForEach-Object { $_ -replace '\\', '/' })
& $sh $script @fwd
exit $LASTEXITCODE

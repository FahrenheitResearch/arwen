param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string]$Source,
    [Parameter(Mandatory=$true)][string]$Workspace,
    [string]$Geography = '',
    [string]$Executable = ''
)
$ErrorActionPreference='Stop'
if(-not $Executable){$Executable=Join-Path $PSScriptRoot 'arwen-launchpad.exe'}
$launchRoot=[IO.Path]::GetFullPath($Workspace)
[IO.Directory]::CreateDirectory($launchRoot) | Out-Null
$launchId=[Guid]::NewGuid().ToString('N')
$launchOut=Join-Path $launchRoot "launchpad-$launchId.log"
$launchErr=Join-Path $launchRoot "launchpad-$launchId.err"
$launchArgs=@('--python',$Python,'--source',$Source,'--workspace',$launchRoot)
if($Geography){$launchArgs+=@('--geog-root',$Geography)}
# Windows paths cannot contain a quote; reject one rather than reinterpret it.
if($launchArgs | Where-Object { $_.Contains('"') }){throw 'A launch argument contains an invalid quote.'}
$launchQuoted=($launchArgs | ForEach-Object { '"'+[regex]::Replace($_,'(\\+)$','$1$1')+'"' }) -join ' '
$launchProcess=Start-Process -FilePath $Executable -ArgumentList $launchQuoted -WindowStyle Hidden -RedirectStandardOutput $launchOut -RedirectStandardError $launchErr -PassThru
$launchDeadline=[DateTime]::UtcNow.AddSeconds(30)
do {
    if(Test-Path -LiteralPath $launchOut){
        # PowerShell 5 keeps redirected output open for writing while the server runs.
        $launchStream=[IO.File]::Open($launchOut,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
        $launchReader=New-Object IO.StreamReader($launchStream)
        try {$launchText=$launchReader.ReadToEnd()} finally {$launchReader.Dispose()}
        if($launchText -match 'ArWen launchpad: (http://127\.0\.0\.1:\d+)'){
            Start-Process -FilePath $Matches[1]
            return
        }
    }
    if($launchProcess.HasExited){throw "Launchpad exited. See $launchErr"}
    Start-Sleep -Milliseconds 100
} while([DateTime]::UtcNow -lt $launchDeadline)
throw "Launchpad did not report its local address. See $launchOut and $launchErr"
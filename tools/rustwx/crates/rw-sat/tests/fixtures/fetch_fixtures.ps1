# Fetch the cloud-product test fixtures (see README.md) into this
# directory, verifying sha256. Anonymous HTTPS, ~8.5 MB total.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$fixtures = @(
    @{ Key = "ABI-L2-ACHAM/2026/216/18/OR_ABI-L2-ACHAM1-M6_G19_s20262161801249_e20262161801336_c20262161801594.nc"
       Sha = "433cbe0d5b1d454a895e33bac1f7340909801863f7504c2d12d5c5dd92a973c3" }
    @{ Key = "ABI-L2-ACTPC/2026/216/18/OR_ABI-L2-ACTPC-M6_G19_s20262161801170_e20262161803545_c20262161804390.nc"
       Sha = "73a1d853a57ec67339ee7c5bd0f726b21e0c19b3db1965a4861d431154c0fbb3" }
    @{ Key = "ABI-L2-CODC/2026/216/18/OR_ABI-L2-CODC-M6_G19_s20262161801170_e20262161803545_c20262161805324.nc"
       Sha = "cb39c3b459079b94dab231a3a3b72d71c8aa054c6454723613495b2c83660c52" }
    @{ Key = "ABI-L2-CPSC/2026/216/18/OR_ABI-L2-CPSC-M6_G19_s20262161801170_e20262161803545_c20262161805325.nc"
       Sha = "a5edb9293d49d3e032f5f41cc1b8b73dfe3c12135b6ac1ac87847016fd3b5d99" }
)
foreach ($f in $fixtures) {
    $name = ($f.Key -split "/")[-1]
    if (-not (Test-Path $name)) {
        Invoke-WebRequest -Uri "https://noaa-goes19.s3.amazonaws.com/$($f.Key)" -OutFile $name
    }
    $actual = (Get-FileHash -Algorithm SHA256 $name).Hash.ToLowerInvariant()
    if ($actual -ne $f.Sha) { throw "sha256 mismatch for ${name}: $actual" }
    Write-Host "ok $name"
}
Write-Host "fixtures ready"

[CmdletBinding()]
param(
    [string]$Config = "config.yaml",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
Push-Location -LiteralPath $ProjectRoot
try {
    & $Python -m repro.serving.transformers_launcher --config $Config
    if ($LASTEXITCODE -ne 0) { throw "Local model server exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

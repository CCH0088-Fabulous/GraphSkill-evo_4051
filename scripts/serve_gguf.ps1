[CmdletBinding()]
param(
    [string]$Config = "config.yaml",
    [string]$ModelPath,
    [string]$ServerPath,
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$EnvironmentPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $EnvironmentPython)) { throw "Run scripts/bootstrap.ps1 first." }
$ServerArguments = @("-m", "repro.serving.llama_launcher", "--config", $Config)
if ($ModelPath) { $ServerArguments += @("--model-path", $ModelPath) }
if ($ServerPath) { $ServerArguments += @("--server-path", $ServerPath) }
if ($Port -ne 0) { $ServerArguments += @("--port", "$Port") }
Push-Location -LiteralPath $ProjectRoot
try {
    & $EnvironmentPython @ServerArguments
    if ($LASTEXITCODE -ne 0) { throw "Local GGUF server exited with code $LASTEXITCODE" }
}
finally {
    Pop-Location
}

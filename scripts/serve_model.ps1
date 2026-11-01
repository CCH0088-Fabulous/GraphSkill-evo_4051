[CmdletBinding()]
param(
    [string]$Config = "config.yaml",
    [string]$ModelPath,
    [string]$ModelName,
    [int]$MaxModelLen,
    [int]$Port,
    [switch]$EnableThinking
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$EnvironmentPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $EnvironmentPython)) { throw "Run scripts/bootstrap.ps1 first." }
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$ServerArguments = @(
    "-m", "repro.serving.transformers_launcher", "--config", $Config
)
if ($ModelPath) { $ServerArguments += @("--model-path", $ModelPath) }
if ($ModelName) { $ServerArguments += @("--model-name", $ModelName) }
if ($PSBoundParameters.ContainsKey("MaxModelLen")) { $ServerArguments += @("--max-model-len", "$MaxModelLen") }
if ($PSBoundParameters.ContainsKey("Port")) { $ServerArguments += @("--port", "$Port") }
if ($EnableThinking) { $ServerArguments += "--enable-thinking" }
& $EnvironmentPython @ServerArguments
if ($LASTEXITCODE -ne 0) { throw "Local model server exited with code $LASTEXITCODE" }

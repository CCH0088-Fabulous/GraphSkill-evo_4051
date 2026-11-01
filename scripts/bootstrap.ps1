[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$RefreshLock,
    [switch]$SkipTorch
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$EnvironmentRoot = Join-Path $ProjectRoot ".venv"
$EnvironmentPython = Join-Path $EnvironmentRoot "Scripts\python.exe"
$UpstreamRoot = Join-Path $ProjectRoot "third_party\GraphSkillEvo"
$LockPath = Join-Path $ProjectRoot "requirements\lock.windows.txt"
$TorchVersion = "2.14.1+cu130"
$TorchIndex = "https://download.pytorch.org/whl/cu130"
$TorchConstraintsPath = Join-Path $EnvironmentRoot "constraints.cuda.txt"

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Executable failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path -LiteralPath (Join-Path $UpstreamRoot "pyproject.toml"))) {
    throw "Official source is missing. Restore the commit recorded in third_party/upstream.lock.json first."
}

Push-Location -LiteralPath $ProjectRoot
try {
    if (-not (Test-Path -LiteralPath $EnvironmentPython)) {
        Invoke-Checked -Executable $Python -Arguments @("-m", "venv", $EnvironmentRoot)
    }
    Invoke-Checked -Executable $EnvironmentPython -Arguments @("-c", "import sys; assert sys.version_info >= (3,11), 'Python 3.11+ required'")
    Invoke-Checked -Executable $EnvironmentPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
    [System.IO.File]::WriteAllText($TorchConstraintsPath, "torch==$TorchVersion`n", [System.Text.UTF8Encoding]::new($false))
    $ConstraintArguments = @()
    if (-not $SkipTorch) { $ConstraintArguments = @("--constraint", $TorchConstraintsPath) }
    if ((Test-Path -LiteralPath $LockPath) -and -not $RefreshLock) {
        Invoke-Checked -Executable $EnvironmentPython -Arguments (@(
            "-m", "pip", "install", "-r", $LockPath,
            "--extra-index-url", $TorchIndex
        ) + $ConstraintArguments)
    }
    if (-not $SkipTorch) {
        & $EnvironmentPython -c "import torch; assert torch.__version__ == '$TorchVersion', 'Pinned CUDA torch required'"
        if ($LASTEXITCODE -ne 0) {
            # Force replacement: an already installed CPU build otherwise satisfies an unpinned torch request.
            Invoke-Checked -Executable $EnvironmentPython -Arguments @(
                "-m", "pip", "install", "--force-reinstall", "torch==$TorchVersion",
                "--index-url", $TorchIndex
            )
        }
    }
    Invoke-Checked -Executable $EnvironmentPython -Arguments (@(
        "-m", "pip", "install", "-e", ".[data,dev,server]", "-e", $UpstreamRoot,
        "--extra-index-url", $TorchIndex
    ) + $ConstraintArguments)
    Invoke-Checked -Executable $EnvironmentPython -Arguments @("-m", "pip", "check")
    if (-not $SkipTorch) {
        Invoke-Checked -Executable $EnvironmentPython -Arguments @(
            "-c", "import torch; assert torch.cuda.is_available(), 'CUDA GPU is not available'; print(torch.__version__, torch.cuda.get_device_name(0))"
        )
    }
    $LockDirectory = Split-Path -Parent $LockPath
    New-Item -ItemType Directory -Path $LockDirectory -Force | Out-Null
    $FrozenPackages = & $EnvironmentPython -m pip freeze --exclude-editable
    if ($LASTEXITCODE -ne 0) { throw "pip freeze failed" }
    [System.IO.File]::WriteAllLines($LockPath, $FrozenPackages, [System.Text.UTF8Encoding]::new($false))
    Write-Host "Windows development environment ready: $EnvironmentPython"
    Write-Host "Resolved dependency lock: $LockPath"
    Write-Host "No model weights were downloaded. Native Windows inference is configured; WSL is not required."
}
finally {
    Pop-Location
}


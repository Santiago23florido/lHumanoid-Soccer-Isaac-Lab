param(
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$conda = Join-Path $HOME "miniconda3\Scripts\conda.exe"

if (-not (Test-Path $conda)) {
    throw "Conda was not found at $conda."
}

$config = if ($Smoke) {
    "configs/benchmark/sample_complexity_smoke.yaml"
} else {
    "configs/benchmark/sample_complexity.yaml"
}

Push-Location $repoRoot
try {
    & $conda run -n env_isaaclab python scripts/run_lqr_surrogate.py
    if ($LASTEXITCODE -ne 0) {
        throw "LQR surrogate failed."
    }
    & $conda run -n env_isaaclab python scripts/run_sample_complexity.py --config $config
    if ($LASTEXITCODE -ne 0) {
        throw "Sample-complexity sweep failed."
    }
} finally {
    Pop-Location
}

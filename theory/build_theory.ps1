$ErrorActionPreference = "Stop"
$theoryDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $theoryDir
$knownMiKTeX = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
$pythonCandidates = @()
if ($env:CONDA_PREFIX) {
    $pythonCandidates += Join-Path $env:CONDA_PREFIX "python.exe"
}
$pythonCandidates += Join-Path $HOME "miniconda3\envs\env_isaaclab\python.exe"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand -and $pythonCommand.Source -notlike "*\WindowsApps\*") {
    $pythonCandidates += $pythonCommand.Source
}
$pythonExe = $pythonCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1

if ($pdflatex) {
    $pdflatexExe = $pdflatex.Source
} elseif (Test-Path $knownMiKTeX) {
    $pdflatexExe = $knownMiKTeX
} else {
    throw "pdflatex was not found. Install MiKTeX and reopen PowerShell."
}

if (-not $pythonExe) {
    throw "python was not found. Activate the project Conda environment."
}

& $pythonExe (Join-Path $repoRoot "scripts\generate_theory_constants.py")
if ($LASTEXITCODE -ne 0) {
    throw "Generating executable theory constants failed."
}

& $pythonExe (Join-Path $repoRoot "scripts\run_lqr_surrogate.py")
if ($LASTEXITCODE -ne 0) {
    throw "Running the LQR surrogate failed."
}

Push-Location $theoryDir
try {
    1..2 | ForEach-Object {
        & $pdflatexExe -interaction=nonstopmode -halt-on-error dynamics_models.tex
        if ($LASTEXITCODE -ne 0) {
            throw "pdflatex failed on pass $_."
        }
    }
} finally {
    Pop-Location
}

Write-Host "Built $theoryDir\dynamics_models.pdf"

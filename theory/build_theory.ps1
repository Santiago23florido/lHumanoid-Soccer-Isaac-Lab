$ErrorActionPreference = "Stop"
$theoryDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$knownMiKTeX = Join-Path $env:LOCALAPPDATA "Programs\MiKTeX\miktex\bin\x64\pdflatex.exe"
$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue

if ($pdflatex) {
    $pdflatexExe = $pdflatex.Source
} elseif (Test-Path $knownMiKTeX) {
    $pdflatexExe = $knownMiKTeX
} else {
    throw "pdflatex was not found. Install MiKTeX and reopen PowerShell."
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

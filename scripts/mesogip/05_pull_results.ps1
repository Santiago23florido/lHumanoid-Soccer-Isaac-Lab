<#
.SYNOPSIS
    Pull completed ICML result JSONs and Mesogip provenance records back from Mesogip
    and rebuild the unified traceable dispatch ledger (transfer/results/cluster_dispatch_ledger.json).
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/mesogip/05_pull_results.ps1
#>
param(
    [string]$HostAlias = "mesogip",
    [string]$RemoteRepo = "~/proyectos/lHumanoid-Soccer-Isaac-Lab",
    [string]$PythonExe = "C:\Users\USER\miniconda3\envs\env_isaaclab\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Push-Location $RepoRoot

try {
    New-Item -ItemType Directory -Force -Path "logs\icml\mesogip_provenance" | Out-Null

    Write-Host "Pulling Mesogip provenance files..." -ForegroundColor Cyan
    $provList = ssh -o BatchMode=yes $HostAlias "ls $RemoteRepo/logs/icml/mesogip_provenance/*.json 2>/dev/null || true"
    if ($provList) {
        scp -o BatchMode=yes "${HostAlias}:${RemoteRepo}/logs/icml/mesogip_provenance/*.json" "logs/icml/mesogip_provenance/"
    }

    Write-Host "Pulling newly produced transfer/results/icml_*.json from Mesogip..." -ForegroundColor Cyan
    $resList = ssh -o BatchMode=yes $HostAlias "cd $RemoteRepo && ls transfer/results/icml_*.json 2>/dev/null || true"
    $pulled = 0
    foreach ($rel in $resList) {
        if (-not $rel) { continue }
        $localPath = Join-Path $RepoRoot ($rel -replace '/', '\')
        if (-not (Test-Path $localPath)) {
            scp -o BatchMode=yes "${HostAlias}:${RemoteRepo}/${rel}" $localPath
            if ($LASTEXITCODE -ne 0) { throw "Failed pulling $rel" }
            $pulled++
        }
    }
    Write-Host ("Pulled {0} new result JSON(s) from Mesogip." -f $pulled) -ForegroundColor Green

    Write-Host "Rebuilding unified traceable dispatch ledger..." -ForegroundColor Cyan
    & $PythonExe scripts/mesogip/update_dispatch_ledger.py
}
finally {
    Pop-Location
}

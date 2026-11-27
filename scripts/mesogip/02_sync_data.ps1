<#
.SYNOPSIS
    Sync ONLY the 22 frozen offline dataset files (8.4 GB, listed by transfer/scripts/icml_cluster_data.py)
    from Windows to Mesogip, keeping shared NFS $HOME minimal and clean.
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts/mesogip/02_sync_data.ps1
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
    Write-Host "Listing required offline files via transfer/scripts/icml_cluster_data.py..." -ForegroundColor Cyan
    $lines = & $PythonExe transfer/scripts/icml_cluster_data.py
    $files = $lines | Where-Object { $_ -match '^(logs/)' }
    if (-not $files -or $files.Count -eq 0) {
        throw "No offline files returned by icml_cluster_data.py"
    }
    Write-Host ("Found {0} required files." -f $files.Count) -ForegroundColor Cyan

    # Create all required parent directories on Mesogip in a single SSH call
    $dirs = ($files | ForEach-Object {
        $p = $_ -replace '/[^/]+$', ''
        "$RemoteRepo/$p"
    } | Select-Object -Unique) -join " "
    ssh -o BatchMode=yes $HostAlias "mkdir -p $dirs $RemoteRepo/logs/icml/models $RemoteRepo/logs/icml/mesogip $RemoteRepo/logs/icml/mesogip_provenance $RemoteRepo/transfer/results"

    # Check which files already exist with matching byte size on remote (idempotent resume)
    $checkScript = ($files | ForEach-Object {
        "if [ -f $RemoteRepo/$_ ]; then echo `"$_ `$(stat -c%s $RemoteRepo/$_)`"; else echo `"$_ -1`"; fi"
    }) -join "; "
    $remoteSizes = @{}
    (ssh -o BatchMode=yes $HostAlias $checkScript) | ForEach-Object {
        $parts = $_ -split '\s+'
        if ($parts.Count -eq 2) { $remoteSizes[$parts[0]] = [int64]$parts[1] }
    }

    $uploaded = 0
    $skipped = 0
    foreach ($rel in $files) {
        $localPath = Join-Path $RepoRoot ($rel -replace '/', '\')
        if (-not (Test-Path $localPath)) {
            throw "Missing required local file: $localPath"
        }
        $localSize = (Get-Item $localPath).Length
        if ($remoteSizes.ContainsKey($rel) -and $remoteSizes[$rel] -eq $localSize) {
            Write-Host ("[SKIP] {0} ({1:N1} MB, exact size match)" -f $rel, ($localSize / 1MB)) -ForegroundColor DarkGray
            $skipped++
            continue
        }
        Write-Host ("[SYNC] {0} ({1:N1} MB) -> {2}:{3}/{0}" -f $rel, ($localSize / 1MB), $HostAlias, $RemoteRepo) -ForegroundColor Green
        $remoteDir = "$RemoteRepo/" + ($rel -replace '/[^/]+$', '/')
        scp -o BatchMode=yes $localPath "${HostAlias}:${remoteDir}"
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $rel" }
        $uploaded++
    }

    # Final byte-exact verification across all 22 files
    $verifyOut = ssh -o BatchMode=yes $HostAlias $checkScript
    $mismatches = @()
    foreach ($line in $verifyOut) {
        $parts = $line -split '\s+'
        if ($parts.Count -ne 2) { continue }
        $rel = $parts[0]
        $rSize = [int64]$parts[1]
        $lSize = (Get-Item (Join-Path $RepoRoot ($rel -replace '/', '\'))).Length
        if ($rSize -ne $lSize) { $mismatches += "$rel (local=$lSize remote=$rSize)" }
    }
    if ($mismatches.Count -gt 0) {
        throw ("Size verification FAILED: " + ($mismatches -join ", "))
    }
    Write-Host ("Sync verified: {0} uploaded, {1} skipped, {2}/{2} exact byte match." -f $uploaded, $skipped, $files.Count) -ForegroundColor Cyan
}
finally {
    Pop-Location
}

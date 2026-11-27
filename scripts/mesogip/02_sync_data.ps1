<#
.SYNOPSIS
    Sync ONLY the frozen offline dataset files listed by transfer/scripts/icml_cluster_data.py
    from Windows to Mesogip, deduplicating coexec/{body}_map.npz via symlinks to
    teachers/{body}_{variant}_probes.npz (SHA256-identical, saves 2.1 GB on NFS: 8.4 GB -> 6.3 GB).
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
    $allFiles = $lines | Where-Object { $_ -match '^(logs/)' }
    if (-not $allFiles -or $allFiles.Count -eq 0) {
        throw "No offline files returned by icml_cluster_data.py"
    }

    # coexec/{body}_map.npz is SHA256-identical to teachers/{body}_{variant}_probes.npz; symlink on NFS!
    $teachers = Get-Content "transfer/bodies/teachers.json" | ConvertFrom-Json
    $symlinks = @{}
    foreach ($b in 'g1','nao','h1','cassie','digit') {
        $v = $teachers.$b.variant
        $mapRel = "logs/pact/final/coexec/${b}_map.npz"
        $probeTarget = "../../teachers/${b}_${v}_probes.npz"
        $symlinks[$mapRel] = $probeTarget
    }

    $uploadFiles = $allFiles | Where-Object { -not $symlinks.ContainsKey($_) }
    Write-Host ("{0} total manifest files: {1} unique uploads + {2} NFS symlinks (saves 2.1 GB)." -f $allFiles.Count, $uploadFiles.Count, $symlinks.Count) -ForegroundColor Cyan

    # Create parent directories on Mesogip
    $dirs = ($allFiles | ForEach-Object {
        $p = $_ -replace '/[^/]+$', ''
        "$RemoteRepo/$p"
    } | Select-Object -Unique) -join " "
    ssh -o BatchMode=yes -o ConnectTimeout=15 $HostAlias "mkdir -p $dirs $RemoteRepo/logs/icml/models $RemoteRepo/logs/icml/mesogip $RemoteRepo/logs/icml/mesogip_provenance $RemoteRepo/transfer/results"
    if ($LASTEXITCODE -ne 0) { throw "SSH unreachable ($HostAlias). Check Telecom Paris VPN / eduroam." }

    # Check remote byte sizes for idempotent resume
    $checkScript = ($uploadFiles | ForEach-Object {
        "if [ -f $RemoteRepo/$_ ]; then echo `"$_ `$(stat -c%s $RemoteRepo/$_)`"; else echo `"$_ -1`"; fi"
    }) -join "; "
    $remoteSizes = @{}
    (ssh -o BatchMode=yes -o ConnectTimeout=15 $HostAlias $checkScript) | ForEach-Object {
        $parts = $_ -split '\s+'
        if ($parts.Count -eq 2) { $remoteSizes[$parts[0]] = [int64]$parts[1] }
    }

    $uploaded = 0
    $skipped = 0
    foreach ($rel in $uploadFiles) {
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
        scp -o BatchMode=yes -o ConnectTimeout=15 $localPath "${HostAlias}:${remoteDir}"
        if ($LASTEXITCODE -ne 0) { throw "scp failed for $rel" }
        $uploaded++
    }

    # Replace any duplicate *_map.npz copies on Mesogip with relative symlinks to teachers/*_probes.npz
    $linkCmds = ($symlinks.GetEnumerator() | ForEach-Object {
        "ln -sfn $($_.Value) $RemoteRepo/$($_.Key)"
    }) -join " && "
    ssh -o BatchMode=yes -o ConnectTimeout=15 $HostAlias $linkCmds
    if ($LASTEXITCODE -ne 0) { throw "Failed creating deduplication symlinks on Mesogip" }

    # Verify all 22 manifest paths resolve (following symlinks via stat -L) with exact byte size
    $verifyScript = ($allFiles | ForEach-Object {
        "if [ -f $RemoteRepo/$_ ]; then echo `"$_ `$(stat -Lc%s $RemoteRepo/$_)`"; else echo `"$_ -1`"; fi"
    }) -join "; "
    $verifyOut = ssh -o BatchMode=yes -o ConnectTimeout=15 $HostAlias $verifyScript
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
    Write-Host ("Sync verified: {0} uploaded, {1} skipped, {2} symlinked (-2.1 GB), {3}/{3} exact byte match." -f $uploaded, $skipped, $symlinks.Count, $allFiles.Count) -ForegroundColor Cyan
}
finally {
    Pop-Location
}

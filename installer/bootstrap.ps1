$ErrorActionPreference = "Stop"

$RepoZipUrl = "https://github.com/cowbook/voiceui/archive/refs/heads/main.zip"
$TargetDir = if ($env:VOICEUI_HOME) { $env:VOICEUI_HOME } else { Join-Path $HOME ".openclaw/apps/voiceui" }

$tmpRoot = Join-Path $env:TEMP ("voiceui-" + [Guid]::NewGuid().ToString("N"))
$zipPath = Join-Path $tmpRoot "voiceui.zip"
$extractDir = Join-Path $tmpRoot "extract"

New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null
New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

Write-Host "[voiceui-bootstrap] Downloading latest source..."
Invoke-WebRequest -Uri $RepoZipUrl -OutFile $zipPath

Write-Host "[voiceui-bootstrap] Extracting..."
Expand-Archive -Path $zipPath -DestinationPath $extractDir -Force
$srcDir = Join-Path $extractDir "voiceui-main"

if (-not (Test-Path (Join-Path $srcDir "installer"))) {
  throw "[voiceui-bootstrap][error] installer directory not found in archive."
}

if (Test-Path $TargetDir) {
  $backupDir = "$TargetDir.bak.$(Get-Date -Format yyyyMMddHHmmss)"
  Write-Host "[voiceui-bootstrap] Existing install found. Backup -> $backupDir"
  Move-Item -Path $TargetDir -Destination $backupDir -Force
}

$parent = Split-Path -Parent $TargetDir
New-Item -ItemType Directory -Force -Path $parent | Out-Null
Move-Item -Path $srcDir -Destination $TargetDir

Write-Host "[voiceui-bootstrap] Installed to $TargetDir"

$bash = Get-Command bash -ErrorAction SilentlyContinue
if (-not $bash) {
  throw "[voiceui-bootstrap][error] bash not found. Install Git Bash or use WSL, then run installer/one-click.sh."
}

Push-Location (Join-Path $TargetDir "installer")
& $bash.Source "./one-click.sh"
Pop-Location

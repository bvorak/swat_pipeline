# Freeze the current environment into a lock file for reproducibility
param(
  [string]$Python="python"
)

$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
try {
  $repoRoot = (Resolve-Path "..\").Path
  Set-Location $repoRoot
  $out = Join-Path $repoRoot "requirements.lock.txt"
  Write-Host "Writing lock file to $out"
  & $Python -m pip freeze | Out-File -FilePath $out -Encoding ascii
  Write-Host "Done. Commit requirements.lock.txt for team installs."
} finally {
  Pop-Location
}


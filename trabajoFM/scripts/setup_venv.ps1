<#
Creates a local virtual environment, installs requirements, and registers a Jupyter kernel.
Pins to Python 3.12 when available (py -3.12 / python3.12), with fallback to default python.
#>
param(
  [string]$Python="",
  [string]$VenvPath=""
)

$ErrorActionPreference = "Stop"

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)
try {
  $repoRoot = (Resolve-Path "..\").Path
  Set-Location $repoRoot

  if ([string]::IsNullOrWhiteSpace($VenvPath)) {
    $venvPath = Join-Path $repoRoot ".venv"
    if ($repoRoot -like "*OneDrive*") {
      Write-Host "Note: repository is under OneDrive. If you hit permission issues, re-run with -VenvPath '$env:LOCALAPPDATA\trabajoFM\.venv'" -ForegroundColor Yellow
    }
  } else {
    $venvPath = $VenvPath
  }

  # Resolve a Python 3.12 interpreter if not explicitly provided
  $PythonExe = $null
  if (-not $Python) {
    try { $pyExe = & py -3.12 -c "import sys;print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $pyExe) { $PythonExe = $pyExe.Trim() } } catch {}
    if (-not $PythonExe) {
      try { $pyExe = & python3.12 -c "import sys;print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $pyExe) { $PythonExe = $pyExe.Trim() } } catch {}
    }
    if (-not $PythonExe) {
      try { $pyExe = & python -c "import sys;print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $pyExe) { $PythonExe = $pyExe.Trim() } } catch {}
    }
  } else {
    try { $pyExe = & $Python -c "import sys;print(sys.executable)" 2>$null; if ($LASTEXITCODE -eq 0 -and $pyExe) { $PythonExe = $pyExe.Trim() } } catch {}
  }
 
  if (-not $PythonExe) { throw "Could not find a working Python interpreter (tried 3.12 and default)." }
  Write-Host "Using Python: $PythonExe"

  Write-Host "Creating venv at $venvPath"
  $parentDir = Split-Path -Parent $venvPath
  if (-not (Test-Path -LiteralPath $parentDir)) { New-Item -ItemType Directory -Path $parentDir -Force | Out-Null }
  & $PythonExe -m venv "$venvPath"

  $venvPy = Join-Path $venvPath "Scripts\python.exe"
  Write-Host "Upgrading pip"; & $venvPy -m pip install --upgrade pip

  $req = Join-Path $repoRoot "requirements.lock.txt"
  if (Test-Path $req) {
    Write-Host "Installing from requirements.lock.txt"
    & $venvPy -m pip install -r "$req"
  } else {
    $req = Join-Path $repoRoot "requirements.txt"
    Write-Host "Installing from requirements.txt"
    & $venvPy -m pip install -r "$req"
  }

  # Removed swat-pytools auto-install; if needed, install separately.

  # Ensure ipykernel is available for Jupyter kernel registration
  Write-Host "Installing ipykernel"
  & $venvPy -m pip install --upgrade ipykernel

  # Write a .pth file into site-packages so python can import repo-local modules from anywhere
  $purelib = & $venvPy -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  if ($LASTEXITCODE -eq 0 -and $purelib) {
    $pth = Join-Path $purelib "trabajoFM_local.pth"
    $pkgPath = Join-Path $repoRoot "trabajoFM"
    Set-Content -LiteralPath $pth -Value $pkgPath -Encoding ascii
    Write-Host "Wrote site path hint: $pth -> $pkgPath"
  } else {
    Write-Warning "Could not determine site-packages to write .pth; imports may require PYTHONPATH."
  }

  # Register Jupyter kernel with the requested display name
  Write-Host "Registering Jupyter kernel: Python (TrabajoFM SWAT)"
  & $venvPy -m ipykernel install --user --name trabajofm-swat --display-name "Python (TrabajoFM SWAT)"

  Write-Host "Done. Activate with: `"$venvPath\Scripts\Activate.ps1`""
} finally {
  Pop-Location
}

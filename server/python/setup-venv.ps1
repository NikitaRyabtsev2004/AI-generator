# Creates server/python/.venv and installs TensorFlow (see requirements.txt).
# TensorFlow has no wheels for very new Python (e.g. 3.14); we pick 3.12..3.9 via the py launcher when possible.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$venvPy = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
  $created = $false
  if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($tag in @("3.12", "3.11", "3.10", "3.9")) {
      cmd /c "py -$tag -m venv .venv >nul 2>&1"
      if ($LASTEXITCODE -eq 0 -and (Test-Path $venvPy)) {
        $created = $true
        break
      }
      if (Test-Path (Join-Path $here ".venv")) {
        Remove-Item -Recurse -Force (Join-Path $here ".venv")
      }
    }
  }
  if (-not $created) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
      cmd /c "py -3 -m venv .venv >nul 2>&1"
    } else {
      cmd /c "python -m venv .venv >nul 2>&1"
    }
  }
}

& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r (Join-Path $here "requirements.txt")
Write-Host "OK: $venvPy"

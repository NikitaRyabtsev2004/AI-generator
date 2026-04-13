# Windows GPU setup for TensorFlow 2.10 stack.
# Creates server/python/.venv2 with Python 3.10 and installs GPU-compatible requirements.
# Keeps existing setup-venv.ps1 untouched.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

$venvName = ".venv2"
$venvPy = Join-Path $here "$venvName\Scripts\python.exe"
$requirementsPath = Join-Path $here "requirements.windows-gpu.txt"

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
  throw "Python launcher 'py' not found. Install Python 3.10 and try again."
}

if (Test-Path (Join-Path $here $venvName)) {
  Write-Host "Reusing existing $venvName"
} else {
  Write-Host "Creating $venvName with Python 3.10..."
  cmd /c "py -3.10 -m venv $venvName"
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPy)) {
    throw "Failed to create $venvName with Python 3.10. Install Python 3.10 and ensure 'py -3.10' works."
  }
}

& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install -r $requirementsPath

Write-Host ""
Write-Host "GPU venv ready: $venvPy"
Write-Host "Set this env var before starting server:"
Write-Host "  `$env:AI_GENERATOR_PYTHON = '$venvPy'"
Write-Host ""
Write-Host "Important for native CUDA on Windows (TF 2.10):"
Write-Host "  1) NVIDIA driver installed"
Write-Host "  2) CUDA 11.2"
Write-Host "  3) cuDNN 8.1 for CUDA 11.2"
Write-Host ""
Write-Host "Quick check:"
Write-Host "  & '$venvPy' -c `"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))`""

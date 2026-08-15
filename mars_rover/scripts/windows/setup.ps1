# One-shot environment setup for Windows.
#
# Run this from an "x64 Native Tools Command Prompt for VS 2022" (or after
# calling vcvars64.bat in a plain PowerShell), from anywhere - it locates the
# repo relative to this script's own path.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
#
# It creates/updates .venv, installs build + runtime dependencies, and
# compiles the native extension. Safe to re-run after pulling changes.

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

if (-not (Get-Command gcc.exe -ErrorAction SilentlyContinue)) {
    Write-Warning "gcc.exe not found on PATH. Install MinGW-w64 and add its bin directory to PATH before running this script, otherwise the native extension build will fail."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    py -m venv .venv
}

$Python = Join-Path ".venv" "Scripts\python.exe"

& $Python -m pip install --upgrade pip
& $Python -m pip install --upgrade setuptools wheel pybind11 cmake
& $Python -m pip install --upgrade numpy gymnasium PyYAML

Write-Host "Building the native extension (this rebuilds _mars_rover_cpp)..."
& $Python -m pip install -e . --no-build-isolation

Write-Host "Verifying the build..."
& $Python -c "import _mars_rover_cpp; print('native extension ok:', _mars_rover_cpp.__file__)"

Write-Host ""
Write-Host "Setup complete. Next steps:"
Write-Host "  .venv\Scripts\activate"
Write-Host "  python -m pytest tests -q"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts\windows\play.ps1"

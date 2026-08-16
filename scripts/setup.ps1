param(
    [ValidateSet("Debug", "Release", "RelWithDebInfo")]
    [string]$BuildType = "Release",
    [ValidateSet("Auto", "MSVC", "GCC")]
    [string]$Toolchain = "Auto",
    [string]$Python = "py",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Venv = Join-Path $ProjectRoot ".venv"
$EnvironmentRoot = Join-Path $ProjectRoot "environment"
$BaselinesRoot = Join-Path $ProjectRoot "baselines"

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE" }
}

if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    Remove-Item -LiteralPath $Venv -Recurse -Force
}
if (-not (Test-Path -LiteralPath $Venv)) {
    Invoke-Checked { & $Python -m venv $Venv }
}

$VenvPython = Join-Path $Venv "Scripts\python.exe"
Invoke-Checked { & $VenvPython -m pip install --upgrade pip setuptools wheel pybind11 cmake numpy gymnasium PyYAML }
$env:MARS_ROVER_BUILD_TYPE = $BuildType
$ActiveBank = Join-Path $ProjectRoot "artifacts\active_bank.json"
if (Test-Path -LiteralPath $ActiveBank) {
    $Bank = Get-Content -Raw $ActiveBank | ConvertFrom-Json
    $env:MARS_ROVER_BANK_MANIFEST = $Bank.manifest
    $env:MARS_ROVER_BANK_INCLUDE = $Bank.include
}

if ($Toolchain -eq "Auto") {
    $Toolchain = if (Get-Command cl.exe -ErrorAction SilentlyContinue) { "MSVC" } else { "GCC" }
}
if ($Toolchain -eq "GCC" -and -not (Get-Command g++.exe -ErrorAction SilentlyContinue)) {
    throw "g++.exe is not on PATH. Install MinGW-w64 or run from a Visual Studio developer shell."
}

# The repository root is intentionally not a Python package.  Install the two
# distributions separately so their editable sources and native extension land
# in this virtual environment.
Invoke-Checked { & $VenvPython -m pip install --editable $EnvironmentRoot --no-build-isolation }
Invoke-Checked { & $VenvPython -m pip install --editable $BaselinesRoot --no-build-isolation }
Invoke-Checked { & $VenvPython -m mars_rover_env.tools.doctor }

Write-Host "Ready ($Toolchain/$BuildType). Play: .\scripts\windows\play.ps1 -Debug"

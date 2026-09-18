






param(
    [ValidateSet("train", "test")]
    [string]$Split = "train",
    [int]$Count = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$Python = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "No .venv found. Run scripts\windows\setup.ps1 first."
}

& $Python -m mars_rover_env.tools.bootstrap_run --split $Split --count $Count

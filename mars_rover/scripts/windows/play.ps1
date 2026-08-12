# Launch the manual-control player using this repo's .venv.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\play.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\windows\play.ps1 -Seed 3 -Debug

param(
    [int]$Seed = 1,
    [switch]$Debug
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$Python = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "No .venv found. Run scripts\windows\setup.ps1 first."
}

$PlayArgs = @("python\mars_rover_env\tools\play.py", "--seed", $Seed)
if ($Debug) { $PlayArgs += "--debug" }

& $Python @PlayArgs

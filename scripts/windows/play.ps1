




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

$PlayArgs = @("-m", "mars_rover_env.tools.play", "--seed", $Seed)
if ($Debug) { $PlayArgs += "--debug" }

& $Python @PlayArgs

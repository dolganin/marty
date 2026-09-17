




param(
    [Nullable[int]]$Seed = $null,
    [switch]$Debug,
    [switch]$Fullscreen,
    [ValidateSet("train", "test")]
    [string]$Split = "train"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$Python = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "No .venv found. Run scripts\windows\setup.ps1 first."
}

$PlayArgs = @("-m", "mars_rover_env.tools.play")
if ($null -ne $Seed) { $PlayArgs += @("--seed", $Seed) }
if ($Debug) { $PlayArgs += "--debug" }
if ($Fullscreen) { $PlayArgs += "--fullscreen" }
$PlayArgs += @("--split", $Split)

& $Python @PlayArgs

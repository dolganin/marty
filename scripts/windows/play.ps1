




param(
    [Nullable[int]]$Seed = $null,
    [switch]$Debug,
    [switch]$Fullscreen,
    [string]$Candidate = "",
    [ValidateRange(1, 4)]
    [int]$CandidateSeed = 1,
    [switch]$ListCandidates,
    [string]$Biome = "",
    [switch]$ListBiomes,
    [ValidateRange(1, 3)]
    [int]$RenderScale = 1,
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
if ($Candidate) { $PlayArgs += @("--candidate", $Candidate, "--candidate-seed-index", $CandidateSeed) }
if ($ListCandidates) { $PlayArgs += "--list-candidates" }
if ($Biome) { $PlayArgs += @("--biome", $Biome) }
if ($ListBiomes) { $PlayArgs += "--list-biomes" }
if ($RenderScale -gt 1) { $PlayArgs += @("--render-scale", $RenderScale) }
$PlayArgs += @("--split", $Split)

& $Python @PlayArgs

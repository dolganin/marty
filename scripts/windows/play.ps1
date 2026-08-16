




param(
    [int]$Seed = 1,
    [switch]$Debug,
    [switch]$RefreshBank,
    [ValidateRange(1, 64)]
    [int]$BiomeCount = 8
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot

$Python = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "No .venv found. Run scripts\windows\setup.ps1 first."
}

if ($RefreshBank) {
    $GeneratorConfig = Join-Path $RepoRoot "environment\python\mars_rover_env\configs\biome_generator.local.yaml"
    if (-not (Test-Path $GeneratorConfig) -and -not $env:OPENAI_API_KEY) {
        throw "LLM credentials are missing. Create $GeneratorConfig or set OPENAI_API_KEY."
    }

    Write-Host "Refreshing the LLM biome bank (train + test) before manual play..."
    & $Python -m mars_rover_env.tools.bootstrap_run --count $BiomeCount --split train --skip-rebuild
    if ($LASTEXITCODE -ne 0) { throw "Train-bank generation failed (exit code $LASTEXITCODE)." }
    & $Python -m mars_rover_env.tools.bootstrap_run --count $BiomeCount --split test
    if ($LASTEXITCODE -ne 0) { throw "Test-bank generation or rebuild failed (exit code $LASTEXITCODE)." }
}

$PlayArgs = @("-m", "mars_rover_env.tools.play", "--seed", $Seed)
if ($Debug) { $PlayArgs += "--debug" }

& $Python @PlayArgs

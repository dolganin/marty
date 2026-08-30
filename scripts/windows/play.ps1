




param(
    [Nullable[int]]$Seed = $null,
    [switch]$Debug,
    [switch]$Fullscreen,
    [switch]$RefreshBank,
    [string]$ResumeBank,
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

if ($RefreshBank -and $ResumeBank) {
    throw "Use either -RefreshBank or -ResumeBank, not both."
}

if ($RefreshBank) {
    $GeneratorConfig = Join-Path $RepoRoot "environment\python\mars_rover_env\configs\biome_generator.local.yaml"
    if (-not (Test-Path $GeneratorConfig) -and -not $env:OPENAI_API_KEY) {
        throw "LLM credentials are missing. Create $GeneratorConfig or set OPENAI_API_KEY."
    }

    Write-Host "Refreshing the LLM biome bank (train + test) before manual play..."
    $Stamp = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
    $BankDir = Join-Path $RepoRoot ("artifacts\banks\manual_" + $Stamp)
    Write-Host "Run bank: $BankDir"
    & $Python -m mars_rover_env.tools.bootstrap_run --count $BiomeCount --split train --skip-rebuild --bank-dir $BankDir
    if ($LASTEXITCODE -ne 0) { throw "Train-bank generation failed (exit code $LASTEXITCODE)." }
    & $Python -m mars_rover_env.tools.bootstrap_run --count $BiomeCount --split test --bank-dir $BankDir
    if ($LASTEXITCODE -ne 0) { throw "Test-bank generation or rebuild failed (exit code $LASTEXITCODE)." }
} elseif ($ResumeBank) {
    $BankDir = Resolve-Path $ResumeBank
    Write-Host "Rebuilding existing run bank without LLM calls: $BankDir"
    & $Python -m mars_rover_env.tools.bootstrap_run --split test --rebuild-only --bank-dir $BankDir
    if ($LASTEXITCODE -ne 0) { throw "Run-bank rebuild failed (exit code $LASTEXITCODE)." }
}

$PlayArgs = @("-m", "mars_rover_env.tools.play")
if ($null -ne $Seed) { $PlayArgs += @("--seed", $Seed) }
if ($Debug) { $PlayArgs += "--debug" }
if ($Fullscreen) { $PlayArgs += "--fullscreen" }

& $Python @PlayArgs

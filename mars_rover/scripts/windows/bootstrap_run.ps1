# Refresh the LLM-generated biome slice and rebuild before a training/validation run.
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows\bootstrap_run.ps1 -Split train -Count 8
#   powershell -ExecutionPolicy Bypass -File scripts\windows\bootstrap_run.ps1 -Split test -Count 8
#
# Requires OPENAI_API_KEY (or openai.token in biome_generator.local.yaml) to be set.

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

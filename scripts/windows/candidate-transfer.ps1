param(
    [int]$Timesteps = 1048576,
    [int]$NumEnvs = 256,
    [double]$EvalSeconds = 300.0,
    [string]$InitModel = "",
    [int]$CommonEvalSeeds = 0,
    [string]$Output = "artifacts/candidate_transfer_v1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $RepoRoot
$Python = Join-Path ".venv" "Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "No .venv found. Run scripts\windows\setup.ps1 first."
}

$TransferArgs = @(
    "-m", "mars_rover_agents.candidate_transfer",
    "--timesteps", $Timesteps,
    "--num-envs", $NumEnvs,
    "--eval-seconds", $EvalSeconds,
    "--output", $Output
)
if ($InitModel) { $TransferArgs += @("--init-model", $InitModel) }
if ($CommonEvalSeeds -gt 0) { $TransferArgs += @("--common-eval-seeds", $CommonEvalSeeds) }
& $Python @TransferArgs

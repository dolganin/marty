










param(
    [string]$Tag = "pipeline_$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$AgentsRoot = Join-Path $RepoRoot "baselines"
Set-Location $RepoRoot

Write-Host "=== [0/4] environment setup ==="
& "$PSScriptRoot\setup.ps1"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
& $Python -m pip install -e "$AgentsRoot" --no-build-isolation -q

Write-Host "=== [1/4] refresh LLM-generated biome slice (train + test) ==="
$BankDir = Join-Path $RepoRoot ("artifacts\banks\" + $Tag)
& $Python -m mars_rover_env.tools.bootstrap_run --count 8 --split train --skip-rebuild --bank-dir $BankDir
& $Python -m mars_rover_env.tools.bootstrap_run --count 8 --split test --bank-dir $BankDir

$RunsRoot = if ($env:MARS_ROVER_RUNS_ROOT) { $env:MARS_ROVER_RUNS_ROOT } else { Join-Path $RepoRoot "runs" }
New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null

Write-Host "=== [2/4] training candidates (~100M env-steps each; long) ==="
Set-Location $AgentsRoot
$Candidates = @()
foreach ($job in @(
    @{Name="reptile_warm"; Algo="reptile"},
    @{Name="reptile_cold"; Algo="reptile"},
    @{Name="vanilla_warm"; Algo="vanilla"},
    @{Name="vanilla_cold"; Algo="vanilla"}
)) {
    $out = Join-Path $RunsRoot "$Tag`_$($job.Name)"
    Write-Host "--- $($job.Name) -> $out ---"
    if ($job.Algo -eq "reptile") {
        & $Python -m mars_rover_agents.train_reptile_ppo `
            --outer-iterations 1000 --inner-timesteps 100000 --num-envs 256 `
            --reptile-lr 0.3 --reptile-lr-final 0.03 --checkpoint-every 100 `
            --eval-episodes 4 --eval-max-steps 1200 --seed (Get-Random) `
            --output $out --run-name "$Tag-$($job.Name)"
    } else {
        & $Python -m mars_rover_agents.train_reptile_ppo `
            --outer-iterations 1 --inner-timesteps 100000000 --num-envs 256 `
            --reptile-lr 1.0 --reptile-lr-final 1.0 --checkpoint-every 1 `
            --eval-episodes 4 --eval-max-steps 1200 --seed (Get-Random) `
            --output $out --run-name "$Tag-$($job.Name)"
    }
    $Candidates += "--candidate"
    $Candidates += "$($job.Name)=$out\final\model.zip"
}

Write-Host "=== [3/4] adaptation-speed evaluation on 40 fresh trials ==="
$ResultsPath = Join-Path $RunsRoot "$Tag`_adaptation_speed.json"
& $Python -m mars_rover_agents.evaluate_adaptation_speed @Candidates `
    --trials 40 --seed-begin (Get-Random) --target-distance 60 --max-attempts 5 `
    --max-episode-steps 1200 --finetune-steps 8192 `
    --output $ResultsPath

Write-Host "=== [4/4] done; results: $ResultsPath ==="

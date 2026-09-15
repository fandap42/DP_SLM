# PowerShell execution script for local runs
param (
    [string]$Config = "experiments/configs/poc_quick.json"
)

Write-Host "Running SLM Toki Pona Experiments with config: $Config"
& ".\.venv\Scripts\python.exe" experiments/scripts/run_experiments.py --config $Config

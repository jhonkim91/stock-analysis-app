param(
    [string]$TaskName = "StockAnalysisAutoRun",
    [string]$Time = "18:30"
)

$ErrorActionPreference = "Stop"

$ScriptPath = Join-Path $PSScriptRoot "run_auto.ps1"
$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$ScriptPath`""
$Trigger = New-ScheduledTaskTrigger -Daily -At $Time

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Description "Run stock analysis predictions automatically." `
    -Force | Out-Null

Write-Host "Registered task '$TaskName' at $Time."

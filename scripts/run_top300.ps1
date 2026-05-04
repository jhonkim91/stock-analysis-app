$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python .\top300_run.py --rank-limit 300 --top 10 --period 5y

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python .\target300_run.py --rank-limit 300 --top 10 --sleep 1 --retries 2

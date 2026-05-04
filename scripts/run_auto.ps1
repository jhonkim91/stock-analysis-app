$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python .\auto_run.py --watchlist .\watchlist.csv --output-dir .\outputs --period 5y

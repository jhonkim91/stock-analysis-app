$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python -m streamlit run .\app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false

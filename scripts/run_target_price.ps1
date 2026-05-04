param(
    [Parameter(Mandatory = $true)]
    [string]$Ticker,
    [string]$Exchange = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Exchange) {
    python .\target_price.py --ticker $Ticker --exchange $Exchange
} else {
    python .\target_price.py --ticker $Ticker
}

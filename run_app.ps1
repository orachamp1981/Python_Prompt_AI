$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$oracleClient = "C:\instantclient_23_4"
$tnsAdmin = Join-Path $oracleClient "network\admin"

if (-not $env:ORACLE_CLIENT_LIB_DIR) {
    if (-not (Test-Path -LiteralPath $oracleClient)) {
        throw "Oracle Instant Client not found at $oracleClient."
    }
    $env:ORACLE_CLIENT_LIB_DIR = $oracleClient
}

if (-not $env:TNS_ADMIN) {
    if (-not (Test-Path -LiteralPath $tnsAdmin)) {
        throw "Oracle TNS configuration directory not found at $tnsAdmin."
    }
    $env:TNS_ADMIN = $tnsAdmin
}

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python environment not found at $python."
}

& $python -m streamlit run app.py --server.headless true --server.port 8765 --browser.gatherUsageStats false
exit $LASTEXITCODE

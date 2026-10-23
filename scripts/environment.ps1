$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $ProjectRoot
$PortableGo = Join-Path $ProjectRoot '.tools\go\bin'
$PortableBin = Join-Path $ProjectRoot '.tools\bin'
if (Test-Path -LiteralPath $PortableGo) { $env:PATH = "$PortableGo;$env:PATH" }
$env:PATH = "$PortableBin;$env:USERPROFILE\go\bin;$env:PATH"
$env:PYTHONPATH = Join-Path $ProjectRoot 'worker'
$env:PYTHONIOENCODING = 'utf-8'
if (-not $env:OPWEAVER_PYTHON) { $env:OPWEAVER_PYTHON = Join-Path $ProjectRoot '.venv\Scripts\python.exe' }
$env:OPWEAVER_HOME = $ProjectRoot

function Assert-Exit([string]$Step) { if ($LASTEXITCODE -ne 0) { throw "$Step failed (exit $LASTEXITCODE)" } }

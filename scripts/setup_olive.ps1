param([string]$EnvironmentPath = '', [string]$BasePython = '')
. "$PSScriptRoot\environment.ps1"
if (-not $EnvironmentPath) { $EnvironmentPath = Join-Path $ProjectRoot '.tools\olive-env' }
$OliveRoot = [System.IO.Path]::GetFullPath($EnvironmentPath)
$WorkerRoot = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot '.venv'))
if ($OliveRoot.TrimEnd('\') -ieq $WorkerRoot.TrimEnd('\')) { throw 'Olive must use a separate environment from the Worker.' }
if (-not $BasePython) { $BasePython = $env:OPWEAVER_PYTHON }
if (-not (Test-Path -LiteralPath $OliveRoot)) {
    & $BasePython -m venv $OliveRoot
    Assert-Exit 'Create independent Olive environment'
}
if (-not (Test-Path -LiteralPath (Join-Path $OliveRoot 'pyvenv.cfg'))) { throw 'Choose a Python virtual environment directory.' }
$OlivePython = Join-Path $OliveRoot 'Scripts\python.exe'
# This script is an explicit install action. The application never calls it.
& $OlivePython -m pip install 'torch==2.11.0+cpu' --index-url 'https://download.pytorch.org/whl/cpu'
Assert-Exit 'Install CPU PyTorch in Olive environment'
& $OlivePython -m pip install -r (Join-Path $ProjectRoot 'requirements-olive.txt')
Assert-Exit 'Install Olive dependencies'
& $OlivePython -m pip check
Assert-Exit 'Check Olive dependencies'
& $OlivePython -m pip freeze | Set-Content -LiteralPath (Join-Path $OliveRoot 'installed-requirements.txt') -Encoding utf8
& $env:OPWEAVER_PYTHON -c 'import json,sys; from opweaver.optimization_environment import probe; result=probe({"python_path":sys.argv[1]}, force=True); print(json.dumps(result, indent=2)); sys.exit(0 if result["available"] else 1)' $OlivePython
Assert-Exit 'Probe installed Olive API'
Write-Host "In Optimization Lab select Microsoft Olive and configure: $OlivePython"

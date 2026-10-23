param([switch]$DownloadGo, [switch]$UseSystemTorch)
. "$PSScriptRoot\environment.ps1"
if (-not (Get-Command go -ErrorAction SilentlyContinue)) {
    if (-not $DownloadGo) { throw 'Go 1.25+ is required. Install Go or rerun with -DownloadGo for a local toolchain.' }
    New-Item -ItemType Directory -Force '.tools' | Out-Null
    $ReleaseList = Invoke-RestMethod 'https://go.dev/dl/?mode=json&include=all'
    $GoAsset = ($ReleaseList | Where-Object version -eq 'go1.27.3').files | Where-Object { $_.os -eq 'windows' -and $_.arch -eq 'amd64' -and $_.kind -eq 'archive' }
    if (-not $GoAsset) { throw 'Pinned Go release not found in the official index' }
    Invoke-WebRequest ("https://go.dev/dl/" + $GoAsset.filename) -OutFile '.tools\go.zip'
    if ((Get-FileHash '.tools\go.zip' -Algorithm SHA256).Hash.ToLower() -ne $GoAsset.sha256) { throw 'Go checksum mismatch' }
    Expand-Archive -LiteralPath '.tools\go.zip' -DestinationPath '.tools' -Force
    $env:PATH = "$PortableGo;$env:PATH"
}
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    if ($UseSystemTorch) { python -m venv --system-site-packages .venv } else { python -m venv .venv }
    Assert-Exit 'Create Python environment'
}
& $env:OPWEAVER_PYTHON -c 'import torch'
if ($LASTEXITCODE -ne 0) {
    & $env:OPWEAVER_PYTHON -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
    Assert-Exit 'Install CPU PyTorch'
}
& $env:OPWEAVER_PYTHON -m pip install -e '.[test]'
Assert-Exit 'Install worker dependencies'
$env:GOBIN = $PortableBin
go install github.com/wailsapp/wails/v2/cmd/wails@v2.17.0
Assert-Exit 'Install Wails'
go mod download
Assert-Exit 'Download Go dependencies'
Push-Location frontend
try { npm ci --registry=https://registry.npmjs.org; Assert-Exit 'Install frontend dependencies' } finally { Pop-Location }
Write-Host 'Ready. Run .\scripts\dev.ps1 or .\scripts\build.ps1'

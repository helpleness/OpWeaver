. "$PSScriptRoot\environment.ps1"
wails build
Assert-Exit 'Desktop build'
$OutputWorker = Join-Path $ProjectRoot 'build\bin\worker\opweaver'
New-Item -ItemType Directory -Force $OutputWorker | Out-Null
Get-ChildItem -LiteralPath 'worker\opweaver' -Filter '*.py' | Copy-Item -Destination $OutputWorker
Copy-Item -LiteralPath 'examples' -Destination 'build\bin' -Recurse -Force
Copy-Item -LiteralPath 'pyproject.toml','LICENSE' -Destination 'build\bin' -Force
Write-Host 'Built build\bin\OpWeaverStudio.exe. Python/PyTorch are external runtime dependencies.'

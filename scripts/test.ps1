. "$PSScriptRoot\environment.ps1"
& $env:OPWEAVER_PYTHON -m pytest -q
Assert-Exit 'Python tests'
& $env:OPWEAVER_PYTHON -m ruff check worker scripts
Assert-Exit 'Python lint'
& $env:OPWEAVER_PYTHON -m ruff format --check worker scripts
Assert-Exit 'Python formatting'
Push-Location frontend
try {
    npm test; Assert-Exit 'Frontend tests'
    npm run build; Assert-Exit 'Frontend build'
} finally { Pop-Location }
$env:OPWEAVER_INTEGRATION = '1'
go test -count=1 ./...
Assert-Exit 'Go tests including real Worker integration'
go vet ./...
Assert-Exit 'Go vet'

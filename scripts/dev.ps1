param([ValidateSet('desktop', 'server')][string]$Mode = 'desktop')
. "$PSScriptRoot\environment.ps1"
if ($Mode -eq 'server') {
    Push-Location frontend
    try { npm run build; Assert-Exit 'Build frontend' } finally { Pop-Location }
    go run . --serve
} else {
    wails dev
}
Assert-Exit 'Development host'

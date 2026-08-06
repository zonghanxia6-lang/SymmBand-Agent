$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

if (Test-Path -LiteralPath ".env.agent") {
    throw "Refusing to stage .env.agent because it may contain an API key. Keep only .env.agent.example."
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not installed."
}
& git lfs version | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Git LFS is required before staging epoch699.ckpt."
}

if (-not (Test-Path -LiteralPath ".git")) {
    & git init -b main
}
& git lfs install --local
& git lfs track "*.ckpt" "macemodel/*.model"
& git add .gitattributes
& git add .

Write-Host ""
Write-Host "Repository staged. Verify before committing:"
Write-Host "  git status --short"
Write-Host "  git lfs ls-files"
Write-Host "  git diff --cached --stat"
Write-Host ""
Write-Host "Then commit and publish:"
Write-Host "  git commit -m 'Initial SymmBand-Agent monorepo'"
Write-Host "  gh repo create SymmBand-Agent --source=. --private --push"


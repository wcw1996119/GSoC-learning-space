# Compile all TikZ figures + table to PDF (and PNG via pdftoppm or magick).
# Usage: pwsh ./compile_all.ps1
# Requires: pdflatex (MiKTeX or TeXLive), optional pdftoppm for PNG preview.

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$files = @(
    "figure1_architecture.tex",
    "figure2_training_deployment.tex",
    "table1_parameters.tex"
)

foreach ($tex in $files) {
    Write-Host "[compile_all] pdflatex $tex ..." -ForegroundColor Cyan
    pdflatex -interaction=nonstopmode -halt-on-error $tex
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[compile_all] FAILED on $tex" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# Optional: PNG preview via Playwright (already installed) or pdftoppm
$pdftoppm = (Get-Command pdftoppm -ErrorAction SilentlyContinue)
if ($pdftoppm) {
    foreach ($tex in $files) {
        $pdf = $tex -replace '\.tex$', '.pdf'
        $png = $tex -replace '\.tex$', ''
        Write-Host "[compile_all] pdftoppm $pdf -> ${png}.png ..." -ForegroundColor Cyan
        & pdftoppm -r 200 -png $pdf $png
    }
} else {
    Write-Host "[compile_all] (pdftoppm not found; skipping PNG preview)" -ForegroundColor Yellow
}

# Cleanup intermediate files
Remove-Item -ErrorAction SilentlyContinue *.aux, *.log, *.out, *.toc

Write-Host "[compile_all] DONE." -ForegroundColor Green
Get-ChildItem *.pdf, *.png 2>$null | Format-Table Name, Length

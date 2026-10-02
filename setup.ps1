# One-time setup on Windows (PowerShell): creates .venv and installs everything.
# Usage:  .\setup.ps1        then every time:  .\.venv\Scripts\Activate.ps1
Set-Location $PSScriptRoot
if (Test-Path .venv) { Remove-Item -Recurse -Force .venv }
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host ""
Write-Host "Done. Activate with:   .\.venv\Scripts\Activate.ps1"
Write-Host "Then run:              python gui.py"

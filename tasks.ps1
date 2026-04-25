param(
  [Parameter(Position = 0)]
  [ValidateSet('help','install','test','run','run-dev','docker-build','docker-run')]
  [string]$Task = 'help',

  [Parameter()]
  [int]$Port = 7860,

  [Parameter()]
  [string]$Image = 'dataclean-env'
)

$ErrorActionPreference = 'Stop'

function Assert-Command([string]$Name) {
  if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
    throw "Missing required command: $Name"
  }
}

function Help {
  Write-Host "DataClean-Env tasks"
  Write-Host ""
  Write-Host "  .\tasks.ps1 install          Install dependencies"
  Write-Host "  .\tasks.ps1 test             Run pytest"
  Write-Host "  .\tasks.ps1 run              Run FastAPI + Gradio (http://localhost:$Port/)"
  Write-Host "  .\tasks.ps1 run-dev          Run with auto-reload"
  Write-Host "  .\tasks.ps1 docker-build     Build Docker image ($Image)"
  Write-Host "  .\tasks.ps1 docker-run       Run Docker image on :$Port"
  Write-Host ""
  Write-Host "Options:"
  Write-Host "  -Port <int>   (default: 7860)"
  Write-Host "  -Image <str>  (default: dataclean-env)"
}

function Install {
  Assert-Command python
  python -m pip install -r requirements.txt
}

function Test {
  Assert-Command python
  python -m pytest -q
}

function Run {
  Assert-Command python
  python -m uvicorn server:app --host 0.0.0.0 --port $Port
}

function Run-Dev {
  Assert-Command python
  python -m uvicorn server:app --host 0.0.0.0 --port $Port --reload
}

function Docker-Build {
  Assert-Command docker
  docker build -t $Image .
}

function Docker-Run {
  Assert-Command docker
  docker run --rm -p "$Port`:7860" $Image
}

switch ($Task) {
  'help'        { Help }
  'install'     { Install }
  'test'        { Test }
  'run'         { Run }
  'run-dev'     { Run-Dev }
  'docker-build'{ Docker-Build }
  'docker-run'  { Docker-Run }
}


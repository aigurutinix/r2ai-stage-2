[CmdletBinding()]
param(
    [ValidateSet("up", "down", "restart", "logs", "status", "doctor", "files")]
    [string]$Action = "up",
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ComposeFile = Join-Path $ProjectRoot "compose.yaml"
$EnvFile = Join-Path $ProjectRoot ".env.docker"
$ExpectedReplaySha = "E19E4748DDAFFAD28112292EAE17E9296F85BBC99265C9D52EEB27E8F8539C85"

function Test-KingproFiles {
    $required = @(
        "build/catalog.jsonl",
        "build/bm25/params.index.json",
        "build/bm25/vocab.index.json",
        "build/tables",
        "data/code_stock.csv",
        "sub_v297_scope2/submission.json",
        "private_evidence",
        "config",
        "configs",
        "outputs"
    )
    $missing = @()
    foreach ($relative in $required) {
        $path = Join-Path $ProjectRoot $relative
        if (-not (Test-Path -LiteralPath $path)) { $missing += $relative }
    }
    if ($missing.Count -gt 0) {
        throw "Missing Docker runtime mounts: $($missing -join ', ')"
    }
    $replay = Join-Path $ProjectRoot "sub_v297_scope2/submission.json"
    $actual = (Get-FileHash -LiteralPath $replay -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actual -ne $ExpectedReplaySha) {
        throw "Sai SHA-256 v297 replay: expected=$ExpectedReplaySha actual=$actual"
    }
    Write-Host "[PASS] Runtime files complete; v297 replay SHA-256 matches." -ForegroundColor Green
}

function Get-ComposePrefix {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        throw "Docker CLI not found. Install/start Docker Desktop, then retry."
    }
    & docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Engine is not running. Start Docker Desktop, then retry."
    }
    $prefix = @("compose", "--project-directory", $ProjectRoot, "-f", $ComposeFile)
    if (Test-Path -LiteralPath $EnvFile) {
        $prefix += @("--env-file", $EnvFile)
    }
    return $prefix
}

Set-Location $ProjectRoot
Test-KingproFiles
if ($Action -eq "files") { exit 0 }

$compose = Get-ComposePrefix
switch ($Action) {
    "up" {
        $args = $compose + @("up", "-d", "--wait")
        if ($Rebuild) { $args += "--build" }
        & docker @args
        if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
        Write-Host "KINGPRO: http://127.0.0.1:3000" -ForegroundColor Cyan
        Write-Host "Backend health: http://127.0.0.1:8080/health" -ForegroundColor Cyan
    }
    "down" { & docker @compose down }
    "restart" { & docker @compose restart; & docker @compose ps }
    "logs" { & docker @compose logs --follow --tail 200 }
    "status" { & docker @compose ps }
    "doctor" {
        & docker @compose config --quiet
        if ($LASTEXITCODE -ne 0) { throw "compose config is invalid" }
        & docker @compose ps
    }
}

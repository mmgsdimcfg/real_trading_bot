<#
g041_pull_daily_backfill.ps1 - copy the g040 daily backfill (Raspberry Pi) to this notebook's runtime.

  powershell -ExecutionPolicy Bypass -File g041_pull_daily_backfill.ps1 -PiHost 192.168.0.10
  powershell -ExecutionPolicy Bypass -File g041_pull_daily_backfill.ps1 -PiHost mypi -StatusOnly

Steps: (1) progress JSON from the Pi (g040 --status), (2) the Pi packs <runtime>/data/daily_backfill into one tar.gz in /tmp
(3,600 small files copy slowly one by one), (3) scp to %TEMP%, (4) unpack into <local runtime>\data\ (overwrites the CSVs;
a half-finished code is simply replaced by the newer copy), (5) the archive on the Pi and the local copy are removed.
Uses the Windows OpenSSH client (ssh/scp/tar.exe); the key or password is the same one AutoBotH uses for the Pi.
Never touches runtime\data\daily_history (the bot's file).

Update log:
- [2026-11-17] type=feat owner=claude
    summary: 신규 - g040 일봉 백필을 파이에서 노트북으로 가져오기 (사용자 요청).
#>
param(
  [Parameter(Mandatory = $true)][string]$PiHost,
  [string]$PiUser = "xgraph",
  [string]$RemoteRuntime = "/data/srv/autobot/runtime",
  [string]$RemoteRepo = "/data/srv/autobot/real_trading_bot",
  [string]$RemotePython = "/home/xgraph/venv/bin/python",
  [string]$LocalRuntime = (Join-Path $PSScriptRoot "..\..\..\runtime"),
  [switch]$StatusOnly
)
$ErrorActionPreference = "Stop"
$target = "$PiUser@$PiHost"
$LocalRuntime = (Resolve-Path $LocalRuntime).Path

Write-Host "== Pi progress"
ssh $target "cd $RemoteRepo/AutoBot && AUTO_TRADING_RUNTIME=$RemoteRuntime $RemotePython data_simulation/g040_daily_backfill.py --status 2>/dev/null | tail -1"
if ($LASTEXITCODE -ne 0) { throw "ssh status failed (rc $LASTEXITCODE)" }
if ($StatusOnly) { return }

$remoteTar = "/tmp/daily_backfill_$(Get-Date -Format yyyyMMddHHmmss).tgz"
Write-Host "== pack on the Pi -> $remoteTar"
ssh $target "tar czf $remoteTar -C $RemoteRuntime/data --exclude=.g040.lock --exclude='*.tmp' daily_backfill"
if ($LASTEXITCODE -ne 0) { throw "remote tar failed (rc $LASTEXITCODE)" }

$localTar = Join-Path $env:TEMP (Split-Path $remoteTar -Leaf)
try {
  Write-Host "== copy -> $localTar"
  scp "${target}:$remoteTar" $localTar
  if ($LASTEXITCODE -ne 0) { throw "scp failed (rc $LASTEXITCODE)" }
  $dataDir = Join-Path $LocalRuntime "data"
  New-Item -ItemType Directory -Force $dataDir | Out-Null
  Write-Host "== unpack -> $dataDir\daily_backfill"
  tar -xzf $localTar -C $dataDir
  if ($LASTEXITCODE -ne 0) { throw "tar -x failed (rc $LASTEXITCODE)" }
  $n = (Get-ChildItem (Join-Path $dataDir "daily_backfill") -Filter *.csv).Count
  Write-Host "== done: $n csv files in $dataDir\daily_backfill"
}
finally {
  ssh $target "rm -f $remoteTar" | Out-Null
  if (Test-Path $localTar) { Remove-Item $localTar -Force }
}

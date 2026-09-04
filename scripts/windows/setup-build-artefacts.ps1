param(
  [string]$ArtefactsRoot = "D:\Build Artefacts",
  [string]$ProjectName,
  [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$PSNativeCommandErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($env:OS -ne "Windows_NT") {
  throw "This script must run on Windows -- it creates NTFS junctions."
}

# Redirects this checkout's large, gitignored build-output folders out to
# $ArtefactsRoot\$ProjectName\<same relative path>, via NTFS junctions. Two
# problems this solves at once:
#
# 1. A checkout that lives inside a cloud-sync folder (OneDrive, Dropbox,
#    Google Drive...) gets these folders synced regardless of .gitignore --
#    sync tools don't read it, they sync whatever's physically on disk. A
#    Rust `target/` alone is routinely tens of thousands of files; a bundled
#    Python runtime under `dist/` drags in every dependency's full source,
#    including things that look alarming out of context (yt-dlp's per-site
#    extractor modules, one of which is literally named after an adult
#    site -- a real, legitimate part of yt-dlp, not a compromise).
# 2. Even without cloud sync, keeping build output on the same physical
#    location as the checkout is just extra churn for backup/indexing tools
#    generally.
#
# Junctions (not symlinks) deliberately: no admin rights or Developer Mode
# required unlike symlinks, and OneDrive skips them outright rather than
# trying to sync through them.
#
# Idempotent -- safe to re-run any time: an already-correct junction is left
# alone, and a folder that doesn't exist yet gets pre-provisioned (so
# whatever creates it later -- cargo, npm, uv, make-portable.ps1 -- writes
# there directly and never touches the synced checkout, even transiently).

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $ProjectName) {
  # Defaults to this checkout's own folder name, so the same script works
  # unmodified for a differently-named clone or fork.
  $ProjectName = Split-Path -Leaf $Root
}
$ArtefactsBase = Join-Path $ArtefactsRoot $ProjectName

# The generated-output folders large/numerous enough to matter -- thousands
# of files or multiple GB -- out of everything .gitignore excludes. Smaller
# tool caches (.ruff_cache, .pytest_cache, .mypy_cache) are left where they
# are: real, but nowhere near this scale.
$Targets = @(
  "dist",
  ".venv",
  "desktop\src-tauri\target",
  "desktop\src-tauri\gen",
  "desktop\node_modules"
)

function Test-IsJunction([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return $false }
  $item = Get-Item -LiteralPath $Path -Force
  return [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

foreach ($rel in $Targets) {
  $source = Join-Path $Root $rel
  $dest = Join-Path $ArtefactsBase $rel

  Write-Host "== $rel ==" -ForegroundColor Cyan

  if (Test-IsJunction $source) {
    $existingTarget = (Get-Item -LiteralPath $source -Force).Target
    if ($existingTarget -and ($existingTarget[0] -ieq $dest)) {
      Write-Host "  already linked -> $dest (skipping)"
      continue
    }
    # Points somewhere else -- don't silently reparent someone's deliberate
    # choice. Same "never silently provision" instinct as the WSL2 setup
    # design (docs/plan.md).
    Write-Warning "  '$source' is already a reparse point, but points elsewhere ($existingTarget). Leaving it alone -- resolve by hand."
    continue
  }

  if (Test-Path -LiteralPath $source) {
    # Real folder still sitting here: move its content out first, then link.
    if (Test-Path -LiteralPath $dest) {
      $existing = Get-ChildItem -LiteralPath $dest -Force -ErrorAction SilentlyContinue
      if ($existing) {
        Write-Warning "  '$dest' already exists and is non-empty, and '$source' is also real. Not touching either -- resolve by hand (pick which copy to keep)."
        continue
      }
    }
    if ($WhatIf) {
      Write-Host "  [WhatIf] would move '$source' -> '$dest', then link"
      continue
    }
    $destParent = Split-Path -Parent $dest
    New-Item -ItemType Directory -Force -Path $destParent | Out-Null
    Write-Host "  moving existing content -> $dest ..."
    Move-Item -LiteralPath $source -Destination $dest
    New-Item -ItemType Junction -Path $source -Target $dest | Out-Null
    if (-not (Test-IsJunction $source)) {
      throw "New-Item reported success but '$source' isn't a junction afterward -- aborting rather than leaving content moved with nothing linking back to it."
    }
    Write-Host "  linked (content preserved)" -ForegroundColor Green
  } else {
    # Nothing here yet -- pre-provision the destination and link.
    if ($WhatIf) {
      Write-Host "  [WhatIf] would pre-create '$dest' and link (nothing exists here yet)"
      continue
    }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    $sourceParent = Split-Path -Parent $source
    New-Item -ItemType Directory -Force -Path $sourceParent | Out-Null
    New-Item -ItemType Junction -Path $source -Target $dest | Out-Null
    if (-not (Test-IsJunction $source)) {
      throw "New-Item reported success but '$source' isn't a junction afterward -- aborting."
    }
    Write-Host "  pre-linked (nothing existed yet)" -ForegroundColor Green
  }
}

Write-Host "`nDone. Everything above now writes to $ArtefactsBase instead of this checkout." -ForegroundColor Cyan

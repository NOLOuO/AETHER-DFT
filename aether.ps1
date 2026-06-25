#!/usr/bin/env pwsh
# AETHER-DFT launcher
# - Double-click aether.cmd or run .\aether.ps1 to enter the chat UI.
# - First launch creates a project-local .venv and installs AETHER-DFT.
# - Uses this computer's existing Python 3.12 or 3.13 to create the project .venv.

[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"
$Stamp = Join-Path $Venv ".aether_ready"
$RequiredMajor = 3
$SupportedMinors = @(12, 13)
$env:PYTHONIOENCODING = "utf-8"
try {
    $utf8 = [System.Text.UTF8Encoding]::new($false)
    [Console]::OutputEncoding = $utf8
    $OutputEncoding = $utf8
} catch {
    # Console encoding is best-effort; Python also reconfigures stdio on entry.
}

function Write-Step([string]$Message) {
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Info([string]$Message) {
    Write-Host $Message -ForegroundColor DarkGray
}

function Fail([string]$Message) {
    Write-Host ""
    Write-Host "AETHER-DFT 启动失败" -ForegroundColor Red
    Write-Host $Message -ForegroundColor Yellow
    Write-Host ""
    if (-not [Console]::IsInputRedirected) {
        Write-Host "按任意键退出..." -ForegroundColor DarkGray
        try { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") } catch { $null = $_ }
    }
    exit 1
}

function Test-PythonVersion([string]$PythonExe, [string[]]$PythonArgs) {
    try {
        $probe = "import json, sys; print(json.dumps({'major': sys.version_info[0], 'minor': sys.version_info[1], 'executable': sys.executable}))"
        $verArgs = $PythonArgs + @("-c", $probe)
        $out = & $PythonExe @verArgs 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $false }
        $info = $out | Select-Object -First 1 | ConvertFrom-Json
        if (Test-UnsupportedPythonPath ([string]$info.executable)) { return $false }
        $maj = [int]$info.major
        $min = [int]$info.minor
        return ($maj -eq $RequiredMajor -and ($SupportedMinors -contains $min))
    } catch {
        return $false
    }
}

function Test-UnsupportedPythonPath([string]$PythonExe) {
    if (-not $PythonExe) { return $false }
    return ($PythonExe -match "(?i)[\\/]uv[\\/]python|[\\/]Astral[\\/]")
}

function Test-CondaPythonPath([string]$PythonExe) {
    if (-not $PythonExe) { return $false }
    return ($PythonExe -match "(?i)[\\/]miniconda|[\\/]anaconda|[\\/]conda")
}

function Test-VenvUsesUnsupportedBase {
    $cfg = Join-Path $Venv "pyvenv.cfg"
    if (-not (Test-Path $cfg)) { return $false }
    $text = Get-Content $cfg -Raw -ErrorAction SilentlyContinue
    return ($text -match "(?i)[\\/]uv[\\/]python|Astral")
}

function Test-VenvUsesSharedSitePackages {
    $cfg = Join-Path $Venv "pyvenv.cfg"
    if (-not (Test-Path $cfg)) { return $false }
    $text = Get-Content $cfg -Raw -ErrorAction SilentlyContinue
    return ($text -match "(?im)^\s*include-system-site-packages\s*=\s*true\s*$")
}

function Find-CondaEnvPythons {
    $roots = @()
    if ($env:CONDA_PREFIX) {
        $roots += $env:CONDA_PREFIX
    }
    $conda = Get-Command "conda" -ErrorAction SilentlyContinue
    if ($conda) {
        $condaPath = $conda.Source
        $root = Split-Path -Parent (Split-Path -Parent $condaPath)
        if ($root) { $roots += $root }
    }
    foreach ($path in @(
        "$env:USERPROFILE\miniconda3",
        "$env:USERPROFILE\anaconda3",
        "D:\miniconda3",
        "C:\miniconda3",
        "C:\ProgramData\miniconda3",
        "C:\ProgramData\anaconda3"
    )) {
        if ($path) { $roots += $path }
    }
    foreach ($root in ($roots | Where-Object { $_ } | Select-Object -Unique)) {
        $basePy = Join-Path $root "python.exe"
        if (Test-Path $basePy) { $basePy }
        $envRoot = Join-Path $root "envs"
        if (Test-Path $envRoot) {
            Get-ChildItem $envRoot -Directory -ErrorAction SilentlyContinue |
                ForEach-Object {
                    $py = Join-Path $_.FullName "python.exe"
                    if (Test-Path $py) { $py }
                }
        }
    }
}

function Reset-ProjectVenv {
    if (-not (Test-Path $Venv)) { return }
    try {
        Remove-Item -LiteralPath $Venv -Recurse -Force
        return
    } catch {
        $deleteError = $_.Exception.Message
        $suffix = Get-Date -Format "yyyyMMddHHmmss"
        $backup = Join-Path $Root ".venv.old-$suffix"
        Write-Host "旧 .venv 暂时无法完整删除，改名隔离为 $backup" -ForegroundColor Yellow
        try {
            Rename-Item -LiteralPath $Venv -NewName (Split-Path -Leaf $backup) -Force
            return
        } catch {
            Fail "旧 .venv 当前被占用或拒绝访问，无法重建隔离环境。请关闭正在使用 AETHER/Python 的终端或编辑器后重试；也可以手动删除 $Venv。删除失败原因：$deleteError；改名失败原因：$($_.Exception.Message)"
        }
    }
}

function Find-BasePython {
    if ($env:AETHER_PYTHON) {
        if (Test-Path $env:AETHER_PYTHON) {
            if (Test-PythonVersion $env:AETHER_PYTHON @()) {
                return @{
                    Source = $env:AETHER_PYTHON
                    Args = [string[]]@()
                    IsConda = (Test-CondaPythonPath $env:AETHER_PYTHON)
                }
            }
        }
    }
    $candidates = @(
        @{ exe = "python"; args = @() },
        @{ exe = "py"; args = @("-3.12") },
        @{ exe = "py"; args = @("-3.13") },
        @{ exe = "python3"; args = @() }
    )
    foreach ($candidate in $candidates) {
        $cmd = Get-Command $candidate.exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        if (Test-UnsupportedPythonPath $cmd.Source) { continue }
        if (Test-PythonVersion $cmd.Source $candidate.args) {
            return @{
                Source = $cmd.Source
                Args = [string[]]$candidate.args
                IsConda = (Test-CondaPythonPath $cmd.Source)
            }
        }
    }
    foreach ($condaPython in (Find-CondaEnvPythons)) {
        if (Test-PythonVersion $condaPython @()) {
            return @{ Source = $condaPython; Args = [string[]]@(); IsConda = $true }
        }
    }
    return $null
}

function Test-AetherInstall {
    if (-not (Test-Path $VenvPy)) { return $false }
    if (Test-VenvUsesUnsupportedBase) { return $false }
    if (Test-VenvUsesSharedSitePackages) { return $false }
    & $VenvPy -c "import aether_dft.cli; import ase.io; import openai; import pymatgen" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Register-GlobalAether {
    try {
        $markerStart = "# >>> AETHER-DFT launcher >>>"
        $markerEnd = "# <<< AETHER-DFT launcher <<<"
        $launcher = Join-Path $Root "aether.ps1"
        $escapedLauncher = $launcher.Replace("'", "''")
        $line = "function aether { & '$escapedLauncher' @args }"
        $block = "$markerStart`n$line`n$markerEnd"
        if (-not (Test-Path $PROFILE)) {
            New-Item -ItemType File -Path $PROFILE -Force | Out-Null
        }
        $content = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
        if ($content -and $content.Contains($markerStart)) { return $false }
        Add-Content -Path $PROFILE -Value "`n$block`n"
        return $true
    } catch {
        Write-Host "全局 aether 注册失败（不影响本目录启动）：$($_.Exception.Message)" -ForegroundColor Yellow
        return $false
    }
}

function Repair-Pip {
    Write-Host "pip 状态异常，正在用 ensurepip 修复..." -ForegroundColor Yellow
    & $VenvPy -m ensurepip --upgrade
    if ($LASTEXITCODE -ne 0) { return $false }
    & $VenvPy -m pip install --upgrade --force-reinstall pip
    return ($LASTEXITCODE -eq 0)
}

function Bootstrap-Aether {
    if ((Test-Path $Stamp) -and (Test-AetherInstall)) { return }

    if (Test-AetherInstall) {
        New-Item -ItemType File -Path $Stamp -Force | Out-Null
        return
    }

    Write-Step "首次启动：正在配置 AETHER-DFT 运行环境（仅此一次，约 3-5 分钟）..."
    $base = Find-BasePython
    if (-not $base) {
        Fail "未找到本机 Python 3.12 或 3.13。请安装 Python 3.12/3.13 后重新运行 aether.cmd；AETHER 会把依赖安装到项目 .venv，不污染原 Python 环境。"
    }

    if ((Test-Path $VenvPy) -and ((Test-VenvUsesUnsupportedBase) -or (Test-VenvUsesSharedSitePackages) -or -not (Test-PythonVersion $VenvPy @()))) {
        Write-Info "检测到现有 .venv 不符合隔离要求（Python 3.12/3.13 / 不共享外部 site-packages），正在重建..."
        Reset-ProjectVenv
    }

    if (-not (Test-Path $VenvPy)) {
        Write-Info "创建项目内虚拟环境：.venv"
        $venvArgs = @()
        $venvArgs += $base.Args
        $venvArgs += @("-m", "venv", $Venv)
        & $base.Source @venvArgs
        if ($LASTEXITCODE -ne 0) { Fail "venv 创建失败。请确认 Python venv 模块可用。" }
    }

    Write-Info "升级 pip..."
    & $VenvPy -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -ne 0) {
        if (-not (Repair-Pip)) {
            Fail "pip 升级失败，且自动修复未成功。请检查网络或删除 .venv 后重新运行 aether.cmd。"
        }
    }

    Write-Info "安装 AETHER-DFT 运行依赖到项目 .venv（pymatgen 等较大，请耐心；安装日志会直接显示）..."
    & $VenvPy -m pip install `
        "pydantic>=2.12" `
        "pydantic-settings>=2.13" `
        "typer>=0.24" `
        "rich>=14.3" `
        "rapidfuzz>=3.14" `
        "tenacity>=9.1" `
        "jinja2>=3.1" `
        "openai>=1.57.4" `
        "ase>=3.23" `
        "pymatgen>=2025.10" `
        "rdkit>=2025.3" `
        "mp-api>=0.46"
    if ($LASTEXITCODE -ne 0) {
        Fail "依赖安装失败。请检查网络；修复后重新双击 aether.cmd 即可继续。"
    }

    Write-Info "安装 AETHER-DFT 项目本体..."
    & $VenvPy -m pip install -e $Root --no-deps
    if ($LASTEXITCODE -ne 0) {
        Fail "AETHER-DFT 项目安装失败。请修复后重新双击 aether.cmd。"
    }

    Write-Info "验证交互入口和关键科学依赖..."
    & $VenvPy -c "import aether_dft.cli; import ase.io; import openai; import pymatgen; print('AETHER import smoke OK')"
    if ($LASTEXITCODE -ne 0) {
        Fail "依赖验证失败：AETHER 或 ase/openai/pymatgen 未能正常导入。请重新运行 aether.cmd。"
    }

    New-Item -ItemType File -Path $Stamp -Force | Out-Null
    Write-Host "✓ 环境就绪：$Venv" -ForegroundColor Green
    if (Register-GlobalAether) {
        Write-Host "✓ 已注册全局命令 aether；重开终端后任何目录都可输入 aether。" -ForegroundColor Green
    }
}

Set-Location $Root

if ($env:AETHER_LAUNCHER_SELFTEST -eq "1") {
    Write-Host "AETHER launcher self-test OK"
    Write-Host "Root=$Root"
    Write-Host "Venv=$Venv"
    exit 0
}

Bootstrap-Aether

if ($Args.Count -eq 0) {
    & $VenvPy -m aether_dft.cli chat --resume
} elseif ($Args.Count -eq 1 -and $Args[0] -eq "--new") {
    & $VenvPy -m aether_dft.cli chat
} else {
    & $VenvPy -m aether_dft.cli @Args
}

$code = $LASTEXITCODE
$isPlainConsole = ($Host.Name -match "ConsoleHost")
if ($env:WT_SESSION -or $env:TERM_PROGRAM) {
    $isPlainConsole = $false
}
if ($isPlainConsole -and $Args.Count -eq 0 -and -not [Console]::IsInputRedirected) {
    Write-Host ""
    Write-Host "AETHER 已退出。按任意键关闭窗口..." -ForegroundColor DarkGray
    try { $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") } catch { $null = $_ }
}
exit $code

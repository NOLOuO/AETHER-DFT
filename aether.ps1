#!/usr/bin/env pwsh
# AETHER-DFT launcher
# - Double-click aether.cmd or run .\aether.ps1 to enter the chat UI.
# - First launch creates a project-local .venv and installs AETHER-DFT.
# - No Conda path is required for end users.

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
$MinMajor = 3
$MinMinor = 12

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
        $verArgs = $PythonArgs + @("-c", "import sys; print('%d %d' % sys.version_info[:2])")
        $out = & $PythonExe @verArgs 2>$null
        if ($LASTEXITCODE -ne 0 -or -not $out) { return $false }
        $parts = ($out.Trim() -split "\s+")
        $maj = [int]$parts[0]
        $min = [int]$parts[1]
        return ($maj -gt $MinMajor -or ($maj -eq $MinMajor -and $min -ge $MinMinor))
    } catch {
        return $false
    }
}

function Find-BasePython {
    $candidates = @(
        @{ exe = "py"; args = @("-3.13") },
        @{ exe = "py"; args = @("-3.12") },
        @{ exe = "python"; args = @() },
        @{ exe = "python3"; args = @() }
    )
    foreach ($candidate in $candidates) {
        $cmd = Get-Command $candidate.exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        if (Test-PythonVersion $cmd.Source $candidate.args) {
            return @{ Source = $cmd.Source; Args = [string[]]$candidate.args }
        }
    }
    return $null
}

function Test-AetherInstall {
    if (-not (Test-Path $VenvPy)) { return $false }
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

function Bootstrap-Aether {
    if (Test-Path $Stamp) { return }

    if (Test-AetherInstall) {
        New-Item -ItemType File -Path $Stamp -Force | Out-Null
        return
    }

    Write-Step "首次启动：正在配置 AETHER-DFT 运行环境（仅此一次，约 3-5 分钟）..."
    $base = Find-BasePython
    if (-not $base) {
        Fail "未找到 Python 3.12 或更高版本。请安装 Python 3.12+ 后重试：https://www.python.org/downloads/"
    }

    if ((Test-Path $VenvPy) -and -not (Test-PythonVersion $VenvPy @())) {
        Write-Info "检测到现有 .venv 的 Python 版本低于 3.12，正在重建项目虚拟环境..."
        Remove-Item -LiteralPath $Venv -Recurse -Force
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
    if ($LASTEXITCODE -ne 0) { Fail "pip 升级失败。请检查网络或 Python 安装。" }

    Write-Info "安装 AETHER-DFT 及依赖（pymatgen 等较大，请耐心；安装日志会直接显示）..."
    & $VenvPy -m pip install -e $Root
    if ($LASTEXITCODE -ne 0) {
        Fail "依赖安装失败。请检查网络；修复后重新双击 aether.cmd 即可继续。"
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

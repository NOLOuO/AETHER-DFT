param(
    [string]$RunRoot = "F:\AETHER-DFT\runs\task_0a4a1ddd\run_a295c506"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

cmd /c "D:\miniconda3\Scripts\activate.bat && conda activate p312env && cd /d $ProjectRoot && python -m aether_dft.cli demo --run-root `"$RunRoot`""

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path

& "D:\miniconda3\Scripts\activate.bat" | Out-Null
conda activate p312env
Set-Location $ProjectRoot
if ($Args.Count -eq 0) {
    python -m aether_dft.cli chat --resume
} else {
    python -m aether_dft.cli @Args
}

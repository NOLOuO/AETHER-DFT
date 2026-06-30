#!/usr/bin/env pwsh
# Optional convenience wrapper for the built-in demo command.
# Uses the project-local launcher and never activates Conda or hard-coded paths.
[CmdletBinding()]
param(
    [string]$RunRoot = ""
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Launcher = Join-Path $Root "aether.cmd"
if ($RunRoot) {
    & $Launcher demo --run-root $RunRoot
} else {
    & $Launcher demo
}
exit $LASTEXITCODE

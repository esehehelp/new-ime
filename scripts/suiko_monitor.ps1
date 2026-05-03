param(
    [string]$Run = "Suiko-v1.10-small-10k-earlykd",
    [int]$IntervalSec = 300,
    [int]$TargetStep = 10000,
    [double]$TargetEm = 0.15,
    [string]$Out = "",
    [switch]$Once
)

$ErrorActionPreference = "Stop"

$Repo = "D:\Dev\new-ime"
if (-not (Test-Path $Repo)) {
    $Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$Log = Join-Path $Repo "models\checkpoints\$Run\train.log"

function Get-Evals {
    if (-not (Test-Path $Log)) { return @() }
    $rows = @()
    foreach ($line in Get-Content -Encoding utf8 $Log) {
        if ($line -match '^\[eval (?<step>\d+)\]\s+loss=(?<loss>[0-9.]+)\s+EM=(?<em>[0-9.]+)\s+CharAcc=(?<char>[0-9.]+)') {
            $rows += [pscustomobject]@{
                Step = [int]$Matches.step
                Loss = [double]$Matches.loss
                EM = [double]$Matches.em
                CharAcc = [double]$Matches.char
            }
        }
    }
    return $rows
}

function Get-LastStep {
    if (-not (Test-Path $Log)) { return $null }
    $last = Select-String -Path $Log -Pattern '^\[step (?<step>\d+)\]' | Select-Object -Last 1
    if ($null -eq $last) { return $null }
    if ($last.Line -match '^\[step (?<step>\d+)\]') { return [int]$Matches.step }
    return $null
}

function Show-Status {
    $now = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $evals = Get-Evals
    $latestEval = $evals | Select-Object -Last 1
    $targetEval = $evals | Where-Object { $_.Step -eq $TargetStep } | Select-Object -Last 1
    $best = $evals | Where-Object { $_.Step -le $TargetStep } | Sort-Object EM -Descending | Select-Object -First 1
    $step = Get-LastStep
    $gpu = (& nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>$null) -join " "
    $py = Get-Process python -ErrorAction SilentlyContinue | Sort-Object CPU -Descending | Select-Object -First 1

    $stepText = if ($null -eq $step) { "-" } else { "$step" }
    $lines = @()
    $lines += "[$now] run=$Run step=$stepText gpu=[$gpu]"
    if ($null -ne $latestEval) {
        $lines += ("  latest eval: step={0} EM={1:N4} CharAcc={2:N4} loss={3:N4}" -f $latestEval.Step, $latestEval.EM, $latestEval.CharAcc, $latestEval.Loss)
    } else {
        $lines += "  latest eval: -"
    }
    if ($null -ne $best) {
        $lines += ("  best<=${TargetStep}: step={0} EM={1:N4}" -f $best.Step, $best.EM)
    }
    if ($null -ne $targetEval) {
        $status = if ($targetEval.EM -gt $TargetEm) { "PASS" } else { "FAIL" }
        $lines += ("  target: EM@{0}={1:N4} threshold>{2:N4} {3}" -f $TargetStep, $targetEval.EM, $TargetEm, $status)
    }
    if ($null -ne $py) {
        $lines += ("  top python: pid={0} cpu={1:N1} start={2}" -f $py.Id, $py.CPU, $py.StartTime)
    } else {
        $lines += "  top python: -"
    }

    foreach ($line in $lines) {
        Write-Host $line
    }
    if ($Out -ne "") {
        $lines | Add-Content -Encoding utf8 -Path $Out
    }
}

do {
    Show-Status
    if ($Once) { break }
    Start-Sleep -Seconds $IntervalSec
} while ($true)

param(
    [int]$MaxSteps = 300000,
    [string]$RunName = "Suiko-v1.12-small-long"
)

$ErrorActionPreference = "Stop"

$Repo = "D:\Dev\new-ime"
if (-not (Test-Path $Repo)) {
    $Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$LegacyPython = Join-Path $Repo "legacy\python"
$Python = Join-Path $LegacyPython ".venv\Scripts\python.exe"
$RunDir = Join-Path $Repo "models\checkpoints\$RunName"
$Train = Join-Path $Repo "datasets\mixes\student-v1_7-500m.kkc"
$Dev = Join-Path $Repo "datasets\eval\general\dev.jsonl"
$Probe = Join-Path $Repo "datasets\eval\probe\probe.json"
$Tokenizer = Join-Path $Repo "datasets\tokenizers\char-5k.json"
$Teacher = Join-Path $Repo "models\checkpoints\ctc-nat-41m-maskctc-student-wp\checkpoint_step_100000.pt"
$Log = Join-Path $RunDir "train.log"

foreach ($path in @($Python, $Train, $Dev, $Probe, $Tokenizer, $Teacher)) {
    if (-not (Test-Path $path)) { throw "missing required path: $path" }
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $RunDir "STOP")
"[launch] $(Get-Date -Format s) -- $RunName long-run final-EM target" | Set-Content -Encoding utf8 $Log

$Args = @(
    "-u", "-m", "models.src.training.train_ctc_nat",
    "--train", $Train,
    "--dev", $Dev,
    "--output", $RunDir,
    "--preset", "phase3_30m",
    "--tokenizer-path", $Tokenizer,
    "--batch-size", "64",
    "--eval-batch-size", "64",
    "--grad-accum", "2",
    "--max-steps", "$MaxSteps",
    "--max-seq-len", "128",
    "--max-train-samples", "0",
    "--max-dev-samples", "2000",
    "--max-context", "32",
    "--lr", "4e-4",
    "--warmup-steps", "2000",
    "--lr-schedule", "cosine",
    "--lr-min-ratio", "0.10",
    "--weight-decay", "0.01",
    "--grad-clip", "1.0",
    "--fp16",
    "--compile",
    "--num-workers", "0",
    "--log-every", "500",
    "--eval-every", "1000",
    "--checkpoint-every", "10000",
    "--keep-last-k", "5",
    "--print-samples", "5",
    "--probe-eval-path", $Probe,
    "--probe-eval-every", "10000",
    "--probe-eval-limit", "0",
    "--refine-loss-weight", "0.7",
    "--refine-warmup-steps", "5000",
    "--refine-mask-ratio-min", "0.15",
    "--refine-mask-ratio-max", "0.35",
    "--kd-teacher-type", "ctc",
    "--kd-teacher-path", $Teacher,
    "--kd-alpha", "0.05",
    "--kd-alpha-final", "0.01",
    "--kd-start-step", "1000",
    "--kd-warmup-steps", "4000",
    "--kd-alpha-decay-start", "30000",
    "--kd-alpha-decay-steps", "70000",
    "--kd-every", "4",
    "--kd-gate-mode", "all",
    "--seed", "52"
)

Push-Location $LegacyPython
try {
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @Args 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Output $line
        Add-Content -Encoding utf8 -Path $Log -Value $line
    }
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = $previousErrorActionPreference
    if ($exitCode -ne 0) { throw "$RunName exited with code $exitCode" }
} finally {
    if ($null -ne $previousErrorActionPreference) {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    Pop-Location
}

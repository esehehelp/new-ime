param(
    [double]$TargetEm = 0.15,
    [string]$StartVariant = "v1.9"
)

$ErrorActionPreference = "Stop"

$Repo = "D:\Dev\new-ime"
if (-not (Test-Path $Repo)) {
    $Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$Train = Join-Path $Repo "datasets\mixes\student-v1_7-500m.kkc"
$Dev = Join-Path $Repo "datasets\eval\general\dev.jsonl"
$Tokenizer = Join-Path $Repo "datasets\tokenizers\char-5k.json"
$Teacher = Join-Path $Repo "models\checkpoints\ctc-nat-41m-maskctc-student-wp\checkpoint_step_100000.pt"
$Root = Join-Path $Repo "models\checkpoints"
$LegacyPython = Join-Path $Repo "legacy\python"
$VenvPython = Join-Path $LegacyPython ".venv\Scripts\python.exe"

if (-not (Test-Path $Train)) { throw "missing train shard: $Train" }
if (-not (Test-Path $Dev)) { throw "missing dev set: $Dev" }
if (-not (Test-Path $VenvPython)) { throw "missing venv python: $VenvPython" }

function Get-EmAt10k {
    param([string]$LogPath)
    if (-not (Test-Path $LogPath)) { return $null }
    $line = Select-String -Path $LogPath -Pattern '^\[eval 10000\].*\bEM=([0-9.]+)' | Select-Object -Last 1
    if ($null -eq $line) { return $null }
    return [double]$line.Matches[0].Groups[1].Value
}

$VariantStarted = $false

function Invoke-Variant {
    param(
        [string]$Name,
        [string[]]$ExtraArgs
    )

    if (-not $script:VariantStarted) {
        if ($Name -notlike "*$StartVariant*") {
            "[skip] $Name before StartVariant=$StartVariant"
            return
        }
        $script:VariantStarted = $true
    }

    $Out = Join-Path $Root $Name
    $Log = Join-Path $Out "train.log"
    New-Item -ItemType Directory -Force -Path $Out | Out-Null
    Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Out "STOP")
    "[launch] $(Get-Date -Format s) -- $Name target_em>$TargetEm@10k" | Set-Content -Encoding utf8 $Log

    $Args = @(
        "-u", "-m", "models.src.training.train_ctc_nat",
        "--train", $Train,
        "--dev", $Dev,
        "--output", $Out,
        "--preset", "phase3_30m",
        "--tokenizer-path", $Tokenizer,
        "--batch-size", "64",
        "--eval-batch-size", "64",
        "--grad-accum", "2",
        "--max-steps", "10000",
        "--max-seq-len", "128",
        "--max-train-samples", "0",
        "--max-dev-samples", "2000",
        "--max-context", "32",
        "--fp16",
        "--compile",
        "--num-workers", "0",
        "--log-every", "500",
        "--eval-every", "1000",
        "--checkpoint-every", "10000",
        "--keep-last-k", "2",
        "--print-samples", "5",
        "--seed", "52"
    ) + $ExtraArgs

    Push-Location $LegacyPython
    try {
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        & $VenvPython @Args 2>&1 | ForEach-Object {
            $line = "$_"
            Write-Output $line
            Add-Content -Encoding utf8 -Path $Log -Value $line
        }
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorActionPreference
        if ($exitCode -ne 0) {
            throw "$Name exited with code $exitCode"
        }
    } finally {
        if ($null -ne $previousErrorActionPreference) {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        Pop-Location
    }

    $Em = Get-EmAt10k $Log
    "[result] $Name eval10000_em=$Em" | Tee-Object -FilePath $Log -Append
    if ($null -ne $Em -and $Em -gt $TargetEm) {
        "[success] $Name reached target: EM=$Em > $TargetEm" | Tee-Object -FilePath $Log -Append
        exit 0
    }
}

Invoke-Variant "Suiko-v1.9-small-10k-fastlr" @(
    "--lr", "5e-4", "--warmup-steps", "1000", "--weight-decay", "0.01", "--grad-clip", "1.0",
    "--warmup-short-sample-steps", "1000", "--warmup-short-sample-max-chars", "24",
    "--refine-loss-weight", "1.0", "--refine-warmup-steps", "2000",
    "--refine-mask-ratio-min", "0.15", "--refine-mask-ratio-max", "0.35",
    "--kd-alpha", "0.0"
)

Invoke-Variant "Suiko-v1.10-small-10k-earlykd" @(
    "--lr", "5e-4", "--warmup-steps", "1000", "--weight-decay", "0.01", "--grad-clip", "1.0",
    "--warmup-short-sample-steps", "1000", "--warmup-short-sample-max-chars", "24",
    "--refine-loss-weight", "1.0", "--refine-warmup-steps", "2000",
    "--refine-mask-ratio-min", "0.15", "--refine-mask-ratio-max", "0.35",
    "--kd-teacher-type", "ctc",
    "--kd-teacher-path", $Teacher,
    "--kd-alpha", "0.05", "--kd-alpha-final", "0.02",
    "--kd-start-step", "1000", "--kd-warmup-steps", "2000",
    "--kd-alpha-decay-start", "6000", "--kd-alpha-decay-steps", "4000",
    "--kd-every", "4", "--kd-gate-mode", "all"
)

Invoke-Variant "Suiko-v1.11-small-10k-noshort" @(
    "--lr", "5e-4", "--warmup-steps", "1000", "--weight-decay", "0.01", "--grad-clip", "1.0",
    "--refine-loss-weight", "1.0", "--refine-warmup-steps", "2000",
    "--refine-mask-ratio-min", "0.15", "--refine-mask-ratio-max", "0.35",
    "--kd-teacher-type", "ctc",
    "--kd-teacher-path", $Teacher,
    "--kd-alpha", "0.05", "--kd-alpha-final", "0.02",
    "--kd-start-step", "1000", "--kd-warmup-steps", "2000",
    "--kd-alpha-decay-start", "6000", "--kd-alpha-decay-steps", "4000",
    "--kd-every", "4", "--kd-gate-mode", "all"
)

throw "no variant reached target EM>$TargetEm @ 10k"

[CmdletBinding()]
param(
    [string]$Device = "auto",
    [int]$NumWorkers = 2,
    [int[]]$Seeds = @(42, 123, 456),
    [string]$DataDir = "data/raw/dcase2020/fan/fan",
    [string]$OutputRoot = "outputs/dcase2020_fan_reproducible"
)

$ErrorActionPreference = "Stop"

foreach ($seed in $Seeds) {
    $oneClassOutput = Join-Path $OutputRoot "full20/egfn_seed$seed"
    $hybridOutput = Join-Path $OutputRoot "dual_memory/egfn_seed$seed"
    $oneClassCheckpoint = Join-Path $oneClassOutput "checkpoint.pt"

    if (-not (Test-Path $oneClassCheckpoint)) {
        python scripts/train_mimii_one_class.py `
            --model egfn `
            --training-mode one_class `
            --dataset-format dcase2020 `
            --dcase-dir $DataDir `
            --device $Device `
            --epochs 20 `
            --batch-size 8 `
            --num-workers $NumWorkers `
            --evaluation-windows 5 `
            --memory-size 512 `
            --gate-mode macro `
            --seed $seed `
            --output-dir $oneClassOutput
    }

    python scripts/train_mimii_one_class.py `
        --model egfn `
        --training-mode hybrid `
        --dataset-format dcase2020 `
        --dcase-dir $DataDir `
        --device $Device `
        --epochs 5 `
        --batch-size 8 `
        --num-workers $NumWorkers `
        --evaluation-windows 5 `
        --memory-size 512 `
        --gate-mode macro `
        --ranking-weight 0.5 `
        --hybrid-max-validation-fpr 0.1 `
        --head-warmup-epochs 5 `
        --pretrained-one-class-checkpoint $oneClassCheckpoint `
        --seed $seed `
        --output-dir $hybridOutput
}

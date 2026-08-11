# Experiment Log

## Speech Commands: Conv1D vs EGFN

Dataset: Google Speech Commands v0.02  
Labels: `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, `go`  
Split: official train/validation/test split  
Train/validation/test records: `30769 / 3703 / 4074`

| Model | Output directory | Best validation accuracy | Test accuracy | Test gate mean |
| --- | --- | ---: | ---: | ---: |
| Conv1D baseline | `outputs/speech_commands_conv1d` | 0.902 | 0.892 | 0.000 |
| EGFN temporal | `outputs/speech_commands_egfn_temporal` | 0.856 | 0.849 | 0.547 |
| EGFN temporal wide | `outputs/speech_commands_egfn_temporal_wide` | 0.893 | 0.886 | 0.444 |

Key result:

```text
EGFN temporal wide nearly matches the Conv1D baseline on Speech Commands:
0.886 test accuracy versus 0.892 for Conv1D.
```

Interpretation:

The wider EGFN model closes most of the gap to the Conv1D baseline while
retaining an interpretable frequency-gated frontend. The lower gate mean in
the wide model suggests more selective use of frequency bands than the earlier
EGFN temporal run.

Next analysis:

1. Plot gate activations by class.
2. Compare class-level confusion matrices.
3. Run robustness tests under controlled SNR.
4. Transfer the same experiment framework to industrial anomalous sound
   detection.

## Industrial Application Setup: MIMII

Target dataset: MIMII Dataset for malfunctioning industrial machine sounds.  
Initial task: supervised binary classification, `normal` vs `abnormal`.  
Initial machine recommendation: `fan`, then `pump`, then `valve` and `slider`.

Downloaded subset:

- `data/raw/mimii/6_dB_valve.zip`
- extracted to `data/raw/mimii/valve`
- records: `4170`
- split from smoke test: `train=2922`, `val=624`, `test=624`

Added code:

- `src/data/mimii_dataset.py`
- `scripts/inspect_mimii.py`
- `scripts/train_mimii.py`

Initial command plan:

```powershell
python scripts/inspect_mimii.py --data-dir data/raw/mimii --machine-type valve
python scripts/train_mimii.py --data-dir data/raw/mimii --machine-type valve --model conv1d --augment --balanced-loss --num-workers 2 --epochs 40 --output-dir outputs/mimii_valve_conv1d
python scripts/train_mimii.py --data-dir data/raw/mimii --machine-type valve --model egfn_temporal --filter-bank fine --learnable-filters --gate-mode independent --frontend-channels 32 --temporal-channels 64,128 --dropout 0.25 --label-smoothing 0.05 --scheduler cosine --augment --balanced-loss --num-workers 2 --epochs 40 --patience 10 --output-dir outputs/mimii_valve_egfn_temporal_wide
```

## MIMII Valve Results and Threshold Calibration

Default test threshold: `0.50`. Threshold selection used validation F1 only;
the test split remained untouched until final evaluation.

| Model | Threshold | Accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conv1D | 0.50 | 0.955 | 0.765 | 0.873 | 0.816 | 0.982 |
| Conv1D calibrated | 0.65 | 0.960 | 0.883 | 0.746 | 0.809 | 0.982 |
| EGFN temporal wide | 0.50 | 0.929 | 0.639 | 0.873 | 0.738 | 0.984 |
| EGFN temporal wide calibrated | 0.78 | 0.965 | 0.930 | 0.746 | 0.828 | 0.984 |

At its validation-selected threshold, EGFN reduced test false positives from
35 to 4 and increased false negatives from 9 to 18. This makes `0.78` suitable
when false alarms are expensive, while `0.50` remains preferable when anomaly
recall is the main requirement.

Interpretability findings:

- mean gates were similar between normal and abnormal examples;
- the largest initialized-band energy changes occurred around `1500-2500 Hz`
  and above `4000 Hz`;
- unconstrained learned filters moved away from several initialized frequency
  ranges and sometimes became multiband;
- the current evidence supports interpretation of the complete frequency
  frontend, but not the claim that gate values alone explain predictions.

## EGFN V2 Controlled Protocol

Research question:

```text
Does EGFN improve because of its frequency-energy-gate frontend, or because
the temporal model is larger than the original Conv1D baseline?
```

The controlled study holds the projection, temporal head, classifier,
augmentation, weighted loss, label smoothing, scheduler, and paired data split
constant. Only the frontend changes:

| Experiment | Frontend | Parameters |
| --- | --- | ---: |
| `conv1d_matched` | free 8-channel waveform convolution | 188,754 |
| `egfn_free` | free FIR EGFN V1 | 188,770 |
| `egfn_sinc` | cutoff-parameterized Sinc EGFN V2 | 187,978 |

The matched Conv and free EGFN differ by only 16 parameters. EGFN-Sinc learns
two cutoff parameters per filter and guarantees valid ordered cutoffs without
hard clamping. Seed 42 is the screening experiment; seeds 123 and 456 are run
only if V2 remains close enough to justify the complete comparison.

# Energy-Gated Frequency Neuron

 project exploring a physically inspired neural block
for audio classification.

## Core Idea

The project does not treat filters as classical activation functions. Instead,
it builds a neural frontend that combines:

```text
raw audio -> filter bank -> activation -> band energy -> learned gate -> classifier
```

For each frequency band `k`, the first version computes:

```text
u_k[n] = (h_k * x)[n]
a_k[n] = phi(u_k[n] + b_k)
E_k = mean(u_k[n]^2)
g_k = sigmoid(alpha_k * log(1 + E_k) + beta_k)
y_k[n] = g_k * a_k[n]
```

The project supports two gate modes:

```text
independent:
g_k = sigmoid(alpha_k * log(1 + E_k) + beta_k)

contextual:
g = sigmoid(MLP([log(1 + E_1), ..., log(1 + E_K)]))
```

The contextual gate lets each band decision depend on the full energy pattern
across bands while preserving interpretable per-band gates.

The goal is to test whether an interpretable, lightweight frequency-gated
frontend can compete with simple audio baselines and improve robustness under
controlled noise.

## Project Structure

```text
frequency_gated_nn/
|-- data/
|   |-- raw/
|   |-- processed/
|-- notebooks/
|-- outputs/
|   |-- figures/
|   |-- models/
|-- scripts/
|   |-- check_env.py
|   |-- test_neuron.py
|   |-- train_synthetic.py
|-- src/
|   |-- blocks/
|   |-- models/
|   |-- utils/
|-- tests/
|-- requirements.txt
|-- README.md
```

## Initial Experimental Plan

1. Validate the block on synthetic signals with known frequency structure.
2. Train a small classifier on synthetic frequency classes.
3. Move to a small real audio dataset such as Free Spoken Digit Dataset.
4. Compare against baselines:
   - MFCC + MLP
   - Conv1D + ReLU
   - fixed filter bank + ReLU
   - fixed filter bank + energy gate
5. Evaluate robustness with controlled SNR levels.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
python scripts/test_neuron.py
python scripts/train_synthetic.py
```

GPU check:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Training scripts use `--device auto` by default, so they use CUDA when PyTorch
can see an NVIDIA GPU. You can force a device with `--device cuda` or
`--device cpu`.

Harder synthetic experiment:

```powershell
python scripts/train_synthetic.py --difficulty medium --filter-bank fine --snr-db 10 --epochs 80
python scripts/train_synthetic.py --difficulty hard --filter-bank fine --snr-db 10 --epochs 100
python scripts/train_synthetic.py --difficulty hard --filter-bank fine --learnable-filters --gate-mode contextual --snr-db 10 --epochs 100 --output-dir outputs/hard_contextual
python scripts/evaluate_snr_sweep.py --difficulty hard --filter-bank fine --learnable-filters --gate-mode contextual --model-path outputs/hard_contextual/models/frequency_gated_synthetic_best.pt
```

V2 selective sparse gate experiment:

```powershell
python scripts/train_synthetic.py --difficulty hard --filter-bank fine --learnable-filters --gate-mode independent --gate-l1 0.005 --snr-db 10 --epochs 100 --output-dir outputs/hard_sparse_gate
python scripts/evaluate_snr_sweep.py --difficulty hard --filter-bank fine --learnable-filters --gate-mode independent --model-path outputs/hard_sparse_gate/models/frequency_gated_synthetic_best.pt --output-dir outputs/hard_sparse_gate
```

`--gate-l1` encourages the model to use fewer active bands. Good runs should
preserve accuracy while reducing `val_gate_mean` and `val_active_bands`.

## First Real Audio Experiment: FSDD

Download Free Spoken Digit Dataset:

```powershell
python scripts/download_fsdd.py
```

Train a small waveform CNN baseline:

```powershell
python scripts/train_fsdd.py --model conv1d --epochs 50 --output-dir outputs/fsdd_conv1d
```

Train the current EGFN model:

```powershell
python scripts/train_fsdd.py --model egfn --filter-bank fine --learnable-filters --gate-mode independent --epochs 50 --output-dir outputs/fsdd_egfn
```

Train EGFN with a temporal convolution head:

```powershell
python scripts/train_fsdd.py --model egfn_temporal --filter-bank fine --learnable-filters --gate-mode independent --epochs 50 --output-dir outputs/fsdd_egfn_temporal
```

Train EGFN temporal with audio augmentation and early stopping:

```powershell
python scripts/train_fsdd.py --model egfn_temporal --filter-bank fine --learnable-filters --gate-mode independent --augment --epochs 80 --patience 12 --output-dir outputs/fsdd_egfn_temporal_aug
```

By default, FSDD is split by speaker so validation/test use speakers not seen
during training. This is harder and more honest than a random split.

## Larger Real Audio Experiment: Speech Commands

Google Speech Commands is a better next dataset after FSDD because it has more
speakers, more variation, and official train/validation/test splits. The
default command subset uses 10 labels:

```text
yes, no, up, down, left, right, on, off, stop, go
```

Download and train a Conv1D baseline:

```powershell
python scripts/train_speech_commands.py --download --model conv1d --epochs 30 --augment --output-dir outputs/speech_commands_conv1d
```

Train the EGFN temporal model on the same label subset:

```powershell
python scripts/train_speech_commands.py --model egfn_temporal --filter-bank fine --learnable-filters --gate-mode independent --augment --epochs 30 --patience 8 --output-dir outputs/speech_commands_egfn_temporal
```

Train a wider EGFN temporal variant:

```powershell
python scripts/train_speech_commands.py --model egfn_temporal --filter-bank fine --learnable-filters --gate-mode independent --frontend-channels 32 --temporal-channels 64,128 --dropout 0.25 --label-smoothing 0.05 --scheduler cosine --augment --num-workers 2 --epochs 40 --patience 10 --output-dir outputs/speech_commands_egfn_temporal_wide
```

If CUDA is available, the script will use it automatically. You can force GPU
execution with:

```powershell
python scripts/train_speech_commands.py --model egfn_temporal --filter-bank fine --learnable-filters --augment --device cuda --epochs 30 --output-dir outputs/speech_commands_egfn_temporal_cuda
```

This experiment is more useful than FSDD for evaluating whether EGFN can
generalize beyond a tiny speaker set.

Current Speech Commands results:

| Model | Best validation accuracy | Test accuracy | Test gate mean |
| --- | ---: | ---: | ---: |
| Conv1D baseline | 0.902 | 0.892 | 0.000 |
| EGFN temporal | 0.856 | 0.849 | 0.547 |
| EGFN temporal wide | 0.893 | 0.886 | 0.444 |

The wide EGFN temporal model nearly matches the Conv1D baseline while retaining
an interpretable frequency-gated frontend.

## Industrial Audio Experiment: MIMII

MIMII contains normal and anomalous sounds from industrial machines such as
fans, pumps, valves, and sliders. Download it manually from Zenodo and place
the extracted files under:

```text
data/raw/mimii/
```

Reference:

```text
https://zenodo.org/record/3384388
```

Inspect the local dataset:

```powershell
python scripts/inspect_mimii.py --data-dir data/raw/mimii
```

This repo currently uses `6_dB_valve.zip` as the first industrial subset.
Inspect the local valve data:

```powershell
python scripts/inspect_mimii.py --data-dir data/raw/mimii --machine-type valve
```

Train a Conv1D baseline:

```powershell
python scripts/train_mimii.py --data-dir data/raw/mimii --machine-type valve --model conv1d --augment --balanced-loss --num-workers 2 --epochs 40 --output-dir outputs/mimii_valve_conv1d
```

Train the EGFN temporal wide model:

```powershell
python scripts/train_mimii.py --data-dir data/raw/mimii --machine-type valve --model egfn_temporal --filter-bank fine --learnable-filters --gate-mode independent --frontend-channels 32 --temporal-channels 64,128 --dropout 0.25 --label-smoothing 0.05 --scheduler cosine --augment --balanced-loss --num-workers 2 --epochs 40 --patience 10 --output-dir outputs/mimii_valve_egfn_temporal_wide
```

If Windows multiprocessing gives a permission error, retry with
`--num-workers 0`.

The industrial question is whether EGFN can detect anomalous machine sounds
while showing which frequency bands become important for abnormal operation.

Current valve results at the default `0.50` anomaly threshold:

| Model | Test accuracy | Precision | Recall | F1 | AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conv1D baseline | 0.955 | 0.765 | 0.873 | 0.816 | 0.982 |
| EGFN temporal wide | 0.929 | 0.639 | 0.873 | 0.738 | **0.984** |

Selecting the threshold on validation F1 chooses `0.78` for EGFN. On the
untouched test split this gives `0.965` accuracy, `0.930` precision, `0.746`
recall, and `0.828` F1. This operating point reduces false alarms from 35 to 4
while increasing missed anomalies from 9 to 18. The appropriate threshold is
therefore an operational decision, not a universal model constant.

Calibrate the anomaly threshold on validation data and evaluate it on the
untouched test split:

```powershell
python scripts/analyze_mimii_results.py --run-dir outputs/mimii_valve_conv1d --num-workers 2
python scripts/analyze_mimii_results.py --run-dir outputs/mimii_valve_egfn_temporal_wide --num-workers 2
```

The analysis writes a validation threshold sweep, default-vs-calibrated test
metrics, per-record test predictions, and calibration figures to each run's
`analysis/` directory. For EGFN runs it also exports per-band gate and energy
statistics for normal versus abnormal audio, plus the learned filter responses.
Band names refer to filter initialization; the response plot shows where each
learnable filter actually moved during training. The test set is never used to
select the threshold.

The current independent gates have similar class-level means. Most visible
normal-versus-abnormal separation appears in band energy and the temporal head.
Also, unconstrained learnable kernels can become multiband and move away from
their initialized ranges. Both observations are limitations to report rather
than evidence that the gate alone explains every prediction.

### Multi-seed comparison

Run the matched three-seed experiment for both models:

```powershell
python scripts/run_mimii_multiseed.py --device cuda --num-workers 2
```

The runner uses seeds `42,123,456`, reuses the completed seed-42 runs, resumes
completed outputs automatically, and trains only missing runs. It calibrates
every model using its own validation split and writes per-seed results plus
mean and sample standard deviation to:

```text
outputs/mimii_valve_multiseed/multiseed_runs.csv
outputs/mimii_valve_multiseed/multiseed_summary.csv
outputs/mimii_valve_multiseed/multiseed_summary.json
```

Rerun the same command after an interruption. Use `--force` only when every
completed run should be trained again.

### EGFN V2: capacity-matched frontend study

V2 separates frontend effects from temporal-head capacity. All three controlled
architectures use eight frontend channels followed by the same `8 -> 32`
projection, `64,128` temporal head, classifier, augmentation, loss, and
regularization:

| Experiment | Frontend | Parameters |
| --- | --- | ---: |
| `conv1d_matched` | unconstrained 8-channel Conv1D | 188,754 |
| `egfn_free` | V1 free FIR filters, energy, and gates | 188,770 |
| `egfn_sinc` | constrained Sinc filters, energy, and gates | 187,978 |

The Sinc frontend learns lower and upper cutoff frequencies while guaranteeing
positive frequencies, at least 50 Hz bandwidth, and an upper cutoff below
Nyquist. Its kernels remain symmetric physical band-pass filters throughout
training.

Run the first controlled comparison with seed 42:

```powershell
python scripts/run_mimii_v2.py --device cuda --num-workers 2
```

This reuses the existing `egfn_free` seed-42 result and trains only the matched
Conv1D and Sinc EGFN. If the Sinc model remains competitive, run all three
paired seeds:

```powershell
python scripts/run_mimii_v2.py --seeds 42,123,456 --device cuda --num-workers 2
```

Completed runs are skipped automatically. Results are written to
`outputs/mimii_v2_controlled/v2_runs.csv` and `v2_summary.csv`. If Windows
multiprocessing fails, use `--num-workers 0`.

### Gating ablation

The capacity-matched V2 result does not by itself identify whether energy
gating helps beyond the filter bank. `gate_mode="none"` now replaces the gate
with exact identity modulation (`g = 1`) and removes the learned gate
parameters, while preserving the same filters, projection, temporal head,
training protocol, and diagnostic interface.

Run the paired free-filter and Sinc ablations with:

```powershell
python scripts/run_mimii_v2.py --include-gating-ablation --seeds 42,123,456 --device cuda --num-workers 2 --output-dir outputs/mimii_v2_gating_ablation
```

This adds `filterbank_free_nogate`, `egfn_free_gated`, and
`filterbank_sinc_nogate` to the existing matched Conv1D and gated Sinc
experiments. Until these runs are complete, the project makes no claim that
gating improves prediction over an otherwise identical filter bank.

## Portfolio Focus

This project is meant to show:

- signal processing intuition,
- custom PyTorch module design,
- controlled experiments,
- baseline comparisons,
- interpretable frequency-band analysis,
- honest limitations and ablation studies.

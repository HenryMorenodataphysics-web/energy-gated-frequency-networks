# Energy-Gated Frequency Networks (EGFN)

Interpretable acoustic anomaly detection with a spectral, condition-aware
frontend and evidence-based diagnostics.

The project started as a small frequency-gated neuron operating directly on
waveform features. It has since become a compact anomaly detector whose
decision can be inspected through energy changes, frequency gates, normal
profiles, and nearest-reference memories.

## What changed in the architecture

The transition was incremental and each stage remains testable:

```text
V1 raw waveform gate
  -> fixed/learnable frequency responses
  -> one-level energy gating

V2 spectral hierarchy
  -> log-power STFT
  -> 3 macro frequency bands and fixed subbands
  -> temporal transforms and local subband context
  -> macro/subband gates

Current EGFN
  -> condition-aware normal spectral profile
  -> sustained pooling + soft event pooling
  -> compact embedding
  -> normal feature memory + anomalous reference memory
  -> calibrated hybrid anomaly score
```

The current frontend detects rapid energy changes through
`abs(log_energy[t] - log_energy[t-1])`, while temporal convolutions model local
sequences. Event pooling prevents a brief impact from disappearing inside a
global mean.

## Current decision flow

For a new recording, the system:

1. Computes a log-power STFT.
2. Aggregates energy into macro bands and subbands.
3. Produces interpretable temporal gates.
4. Compares descriptors with the normal profile for the machine/condition.
5. Builds a sustained/event embedding.
6. Measures distance to normal and anomalous feature memories.
7. Combines reference evidence with the supervised head using validation-only
   calibration.
8. Returns anomaly score, threshold, condition, and reference evidence.

The numerical EGFN score is authoritative. A future LLM layer may explain the
score and summarize retrieved evidence, but it must not override the detector.

## Evidence from the dual-memory experiment

The same DCASE 2020 fan protocol was evaluated with three independent seeds.
The anomalous memory was fitted only from the training anomaly partition.

| Seed | ROC AUC | F1 |
| ---: | ---: | ---: |
| 42 | 0.7701 | 0.5642 |
| 123 | 0.6958 | 0.4424 |
| 456 | 0.7862 | 0.6281 |
| **Mean** | **0.7507** | **0.5449** |

The mean clears the current prototype gate of 0.75, but the spread matters:
seed 123 is below 0.75. This is evidence for a promising prototype, not a
claim of production-grade robustness.

## Reproduce the model

Create an environment and install the pinned project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/check_env.py
```

The canonical DCASE command is:

```powershell
python scripts/train_mimii_one_class.py `
  --model egfn `
  --training-mode hybrid `
  --dataset-format dcase2020 `
  --dcase-dir data/raw/dcase2020/fan/fan `
  --device cuda `
  --epochs 5 `
  --batch-size 8 `
  --num-workers 2 `
  --evaluation-windows 5 `
  --memory-size 512 `
  --gate-mode macro `
  --ranking-weight 0.5 `
  --hybrid-max-validation-fpr 0.1 `
  --head-warmup-epochs 5 `
  --pretrained-one-class-checkpoint outputs/dcase2020_fan_full20/egfn_seed42/checkpoint.pt `
  --seed 42 `
  --output-dir outputs/dcase2020_fan_dual_memory/egfn_seed42
```

For a complete three-seed run, use:

```powershell
.\scripts\reproduce_dcase_dual_memory.ps1 -Device cuda -NumWorkers 2
```

The script first trains the normal representation and then fits the small
supervised head and dual memories. Training and evaluation outputs stay local
and are intentionally excluded from Git.

## Planned diagnostic application

The next product layer is a small web app/API where a user uploads an audio
file and receives:

- normal/anomalous decision and calibrated score;
- machine/condition profile used;
- distance to the normal memory;
- similarity to known anomalous references;
- detected energy-change windows and active frequency bands;
- a concise explanation generated from structured EGFN evidence.

The app will be an experimental diagnostic aid. It will not claim a machine
failure, replace an engineer, or allow an LLM to make an uncalibrated decision.
See [`docs/architecture_and_app.md`](docs/architecture_and_app.md) for the
system boundary and implementation roadmap.

## Limitations and scope

- Current multiseed evidence uses DCASE 2020 `fan`; other machine families and
  SNR levels still need evaluation.
- MIMII and DCASE are adapters and test sources, not assumptions built into
  the neural layers. Generic normal/anomalous folders are supported.
- The anomalous memory can only retrieve anomaly types represented in its
  training partition; the normal memory remains necessary for unseen failures.
- The first temporal derivative is absolute, so upward and downward changes
  are currently indistinguishable.
- The current temporal context is local; long-duration event modelling and
  multi-resolution STFT remain future work.
- A mean AUC above 0.75 on three seeds is an internal prototype gate, not a
  safety or industrial certification threshold.
- High sample-rate claims require recordings captured at the target rate;
  resampling 16 kHz data does not create missing high-frequency information.

## Repository map

```text
src/blocks/       spectral frontend, gates, harmonic context
src/models/       EGFN encoder, pooling, baselines
src/anomaly/      profiles, memories, scoring and objectives
src/data/         generic, MIMII and DCASE adapters
scripts/          canonical training, evaluation and reproducibility commands
tests/            unit and protocol tests
docs/             architecture and diagnostic-app design
```

Raw audio, checkpoints, caches and generated outputs are not versioned.

# EGFN architecture and diagnostic application

## 1. Scientific core

The detector maps a waveform (x[n]) to an interpretable score:

```text
x[n]
  -> STFT power P(f,t)
  -> log-energy descriptors by macro band/subband
  -> temporal change |delta log E|
  -> learned gates g(f,t)
  -> gated feature map H
  -> sustained mean and soft-event pooling
  -> embedding e
```

The normal profile is conditioned on machine and operating condition. A
condition-specific fallback is used when the machine is unknown.

Two bounded memories use the same normal-standardized feature space:

```text
M_N = normal local feature memory
M_A = known anomalous local feature memory
d_N = nearest-reference distance to M_N
d_A = nearest-reference distance to M_A
s_reference = d_N / (d_N + d_A + epsilon)
```

The final score is calibrated on validation data only. The current hybrid
configuration combines reference evidence and the supervised head. The test
set is never used to choose the threshold or the mixture weights.

## 2. Application boundary

```text
Browser upload
    -> API validation (format, duration, sample rate)
    -> EGFN inference service
    -> structured evidence object
    -> deterministic decision and thresholding
    -> optional LLM explanation
    -> browser report
```

The evidence object should contain at least:

- `is_anomaly`, `score`, `threshold`, and `condition`;
- `normal_distance`, `anomalous_distance`, and `reference_score`;
- top active macro bands and high-change time windows;
- identifiers and distances of retrieved reference examples;
- model version, profile version, and calibration version.

The LLM receives only this structured evidence. It can explain the result in
plain language, but it cannot change `is_anomaly`, threshold, or score.

## 3. Reproducibility and governance

Every inference artifact must record:

1. model checkpoint hash;
2. training dataset/protocol name;
3. condition profile version;
4. memory-bank version;
5. calibration seed and validation policy;
6. audio preprocessing parameters.

Raw user audio should not be sent to an external LLM by default. The initial
application should keep inference local and send only numerical evidence to an
optional explanation provider.

## 4. Implementation phases

### Phase A — offline evidence (current)

- dual feature memory;
- calibrated reference score;
- multiseed DCASE evaluation;
- CSV/JSON evidence artifacts.

### Phase B — local inference service

- reusable `predict_file()` API around the trained model;
- safe audio validation and bounded duration;
- JSON response matching the evidence contract;
- tests for malformed audio and unknown conditions.

### Phase C — upload UI

- file upload and progress state;
- score, threshold and confidence context;
- energy-change timeline and active-band visualization;
- nearest normal/anomalous reference summaries.

### Phase D — explanation layer

- deterministic explanation fallback;
- optional LLM wording over the evidence JSON;
- explicit “evidence unavailable” and “unknown condition” states;
- prompt/evaluation tests that prevent score changes.

### Phase E — broader validation

- fan, pump, slider and valve;
- additional SNR conditions;
- unseen-machine fallback;
- three-seed confidence intervals and drift monitoring.

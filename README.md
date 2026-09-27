# Quantized verifiers in Best-of-N

Code for the study of weight-quantized reward models / verifiers in Best-of-N
mathematical answer selection, the memory-budgeted frontier, and
Contrastive-Hessian GPTQ (CH-GPTQ). Design: `plan-neurocomputing.md`.

## Layout

| Module | Role |
|---|---|
| `vq.grading` | final-answer extraction and correctness (fixed before scoring) |
| `vq.data` | datasets and hashed, frozen manifests (`data/manifest_*.json`) |
| `vq.generate` | candidate pools (vLLM) and annotation |
| `vq.verifiers` | Skywork-Reward-V2 ORMs and Qwen2.5-Math-PRM scoring, batched |
| `vq.quant.*` | fake-quant: RTN, NF4, GPTQ, CH-GPTQ (contrastive Hessian), AWQ-style; sequential application; packed GPTQ export |
| `vq.calibration` | label-free calibration recipes: generic, random, DPQ-adapted, CH pairs, shuffled pairs |
| `vq.selection`, `vq.metrics` | Best-of-N, weighted/majority vote, H_N/B_N, winner change, efficiency |
| `vq.controls` | permutation (free / stratified) and Gaussian controls |
| `vq.diagnostic` | C3 residual-model prediction of winner change |
| `vq.stats` | paired bootstrap by question, Holm, TOST |
| `vq.runner` | one scoring run end-to-end (resumable shards, audit) |
| `vq.report` | paper tables (main, controls, C3, frontier, confirmatory) |
| `vq.pod`, `scripts/pod.py` | RunPod launcher (dry run by default) |
| `configs/experiment.yaml` | the full grid — single source of truth |

## Local (CPU)

```bash
uv venv && uv pip install -e ".[dev]"
pytest                      # unit + integration tests (tiny random models)
pytest -m network           # real-checkpoint checks on CPU (downloads Skywork 0.6B)
python -m vq.cli plan       # campaigns and cost forecast
```

`.env` must define `HF_TOKEN` and `RUNPOD_API_KEY` (never committed).

## RunPod runbook (each step: `scripts/pod.py launch --step "<args>" --hours H --yes`, then `watch --pod <id>`)

| # | Step (`--step`) | Hours cap | Must check before continuing |
|---|---|---|---|
| 0 | `pilot --generator qwen2.5-1.5b` (and `qwen2.5-math-1.5b`) | 3 | `outputs/throughput.json`: batch/single parity < 1e-2 for every verifier; truncation < 2 %; re-run `plan` with measured throughput |
| 0b | `export-packed --verifier sky-4b` then `score-packed` on the gate split | 2 | packed vs fake-quant score parity (GPTQ kernel compatibility) |
| 1 | `generate --generator G --split dev`, `annotate`, `score --phase phase3_main --only sky-4b` (dev) then `gate` | 6 | gate criteria (`gate --projected-eur ...`) |
| 2 | `generate --generator G --split eval`; `generate --generator llama3.2-3b --split gen2` | 12 | parse / truncation rates |
| 3 | `score --phase phase3_main --generator G` | 36 | audit in every run's JSON (`head_or_embed_changed: false`) |
| 5a | `select-lambda --generator G --verifier sky-4b` | 3 | `outputs/frozen_lambda.json` written |
| 5 | `score --phase phase5_ch --generator G` | 30 | all 20 runs done |
| 4 | `score --phase phase4_gen2` | 6 | — |
| 6 | `export-packed` + `score-packed` (frontier configs) + `measure-memory` | 6 | `outputs/memory.json` |

Then locally: `python -m vq.cli analyze --generator G` → `outputs/tables/`.

Rules: never edit manifests or `configs/experiment.yaml` after phase 1 without
logging the change; never cut evaluation questions, seeds or controls silently.

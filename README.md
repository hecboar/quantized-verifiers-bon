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

## RunPod runbook

Each step: `python scripts/pod.py launch --step "<vq.cli args>" --hours H --yes`, then
`python scripts/pod.py watch --pod <id>` (fetches, merges without overwriting, deletes the pod).
The printed cost is the worst case (hard cap + 60 min grace). Grid steps are refused until
`outputs/gate.json` says `go`.

| # | Step (`--step`) | Cap (h) | Must check before continuing |
|---|---|---|---|
| 0 | `pilot --generator qwen2.5-1.5b` (then `qwen2.5-math-1.5b`) | 3 | `PILOT OK`; `outputs/throughput.json`: parity < 1e-2, `rtn_runs` all ok (incl. PRM); re-run `plan` locally with measured throughput |
| 0b | `export-packed --verifier sky-4b` (extras `gpu,packed`), `score-deployed --scheme gptq-w4` and `--scheme nf4` on the gate split, then locally `parity` | 3 | fake-quant vs deployment score parity and winner agreement |
| 1 | `generate --generator G --split dev`; `annotate`; `score --phase phase3_main --generator G --only sky-4b --conditions bf16,nf4,gptq-w4 --dev-only`; locally `gate --generator G --projected-eur <plan>` | 6 | `outputs/gate.json` go=true |
| 2 | `generate --generator G --split eval`; `generate --generator llama3.2-3b --split gen2`; `annotate` | 12 | parse / truncation rates |
| 5a | `select-lambda --generator G --verifier sky-4b` | 3 | `outputs/frozen_lambda.json` |
| 3 | `score --phase phase3_main --generator G` | 36 | every run JSON: `audit.head_or_embed_changed: false` |
| 5 | `score --phase phase5_ch --generator G` | 30 | 20 runs done |
| 4 | `score --phase phase4_gen2 --generator G` | 6 | transfer runs use `calib_generator = G` |
| 6 | `export-packed` (4B, 8B); `score-deployed --scheme gptq-w4` (4B, 8B) and `--scheme nf4` (1.7B/4B/8B); `measure-memory` | 8 | `outputs/memory.json`; `parity` reports |

Then locally: `python -m vq.cli analyze --generator G` -> `outputs/tables/`.

Rules: never edit manifests or `configs/experiment.yaml` after phase 1 without logging the change;
never cut evaluation questions, seeds or controls silently. Runs are fingerprinted: resuming or
reusing a run after changing its inputs or settings fails on purpose.

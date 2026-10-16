# Quantized reward-model verifiers in Best-of-N

Code, analysis plans and analysis scripts for

> H. Borobia, E. Seguí-Mas, G. Tormo-Carbó. *Quantized reward-model verifiers in Best-of-N
> mathematical answer selection: accuracy, memory and value retention.* Submitted to
> Neurocomputing.

The study scores frozen pools of 32 candidate solutions per problem (GSM8K and MATH) with
reward-model verifiers in BF16 and after weight quantization (NF4, GPTQ 4/3-bit, AWQ,
RTN), and asks whether Best-of-N accuracy, the verifier's advantage over majority vote
and a memory-budgeted deployment survive quantization.

Data (candidate pools, every verifier score, result tables):
<https://huggingface.co/datasets/hecboar/quantized-verifiers-bon>.

The version described in the submitted manuscript is the tag `v1.0-submission`.

## Analysis plans

The plans were fixed in this repository before the corresponding data existed; the
commit history is the timestamped record (see Table C.8 of the paper).

| File | Content |
|---|---|
| `docs/preregistration.md` | Main study: design, populations, contrasts c1–c5, statistics, amendments and deviations |
| `docs/preregistration_h6.md`, `docs/h6_results.md` | H6: stronger generator on new MATH Level-5 problems |
| `docs/preregistration_h7.md`, `docs/h7_results.md` | H7: value retention on new MATH Level-4/5 problems |
| `docs/preregistration_rev.md` | Exploratory revision experiments (amendments 5 and 6) with execution records |
| `configs/experiment.yaml` | Every condition and scoring run (single source of truth) |
| `data/manifest_*.json` | Frozen problem manifests with hashes |

## Reproducing the paper without a GPU

```bash
uv venv && uv pip install -e ".[dev]"
# download the data next to the code
python -c "from huggingface_hub import snapshot_download; snapshot_download('hecboar/quantized-verifiers-bon', repo_type='dataset', local_dir='outputs')"
cp outputs/data/items.jsonl outputs/data/math_levels.json data/
```

| Paper item | Command |
|---|---|
| Main tables, confirmatory c1–c5 (Tables 2–3) | `python -m vq.cli analyze --generator qwen2.5-1.5b` |
| H6 and H7 confirmatory tests | `python scripts/analyze_h6.py` (see the script header for the H7 arguments) |
| Majority vote, answer transitions, selection-rule controls (Table 5, Section 4.7) | `python scripts/revision_analyses.py` |
| Llama-based ORMs, 3-bit calibration, memory and throughput (Tables 4, 6, 7) | `python scripts/analyze_revision_runs.py` |
| Packed vs simulated GPTQ, oracle, c1 TOST | `python scripts/review3_analyses.py` |
| Answer-format decomposition, lenient grader | `python scripts/exploratory_format.py` |
| GSM8K-Platinum labels and truncation sensitivity | `python scripts/sensitivity_labels_truncation.py` |
| Human audit of the outcome measure (Appendix B) | `python scripts/analyze_c2_audit.py` |
| Figures 1–4 | `python scripts/make_figures.py` |

Tests: `pytest` (unit and integration tests on tiny random models; `pytest -m network`
adds real-checkpoint checks on CPU).

## Regenerating pools and scores (GPU)

Generation (vLLM) and scoring ran on single cloud GPUs (mostly RTX 4090) through
`scripts/pod.py`, which launches a RunPod pod for one `vq.cli` step, fetches the outputs
and deletes the pod. `.env` must define `HF_TOKEN` and `RUNPOD_API_KEY`; it is never
committed. Every scoring run is fingerprinted, so resuming or reusing a run with any
different input fails instead of mixing results.

| Module | Role |
|---|---|
| `vq.grading` | final-answer extraction and correctness (strict and lenient graders) |
| `vq.data` | datasets and hashed, frozen manifests |
| `vq.generate` | candidate pools (vLLM) and annotation |
| `vq.verifiers` | ORM and PRM scoring, batched |
| `vq.quant.*` | simulated quantization (RTN, NF4, GPTQ, AWQ) and packed GPTQ export |
| `vq.calibration` | calibration recipes (generic, in-domain, length-matched, DPQ-adapted, CH) |
| `vq.selection`, `vq.metrics` | Best-of-N, majority and weighted vote, transitions |
| `vq.stats` | paired bootstrap by problem, TOST, Holm |
| `vq.runner` | one scoring run end-to-end (resumable shards, quantization audit) |
| `vq.memory_bench` | peak memory and throughput of deployed verifiers |
| `vq.pod`, `scripts/pod.py` | RunPod launcher |

## Licence

Code: MIT. Data: see the dataset card on Hugging Face.

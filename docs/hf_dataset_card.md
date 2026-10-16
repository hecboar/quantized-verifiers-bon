---
license: cc-by-4.0
language:
- en
pretty_name: Quantized reward-model verifiers in Best-of-N (candidate pools and scores)
size_categories:
- 1M<n<10M
task_categories:
- text-classification
tags:
- reward-models
- best-of-n
- quantization
- mathematical-reasoning
- gsm8k
- math
---

# Quantized reward-model verifiers in Best-of-N: candidate pools and verifier scores

Data for the paper *Quantized reward-model verifiers in Best-of-N mathematical answer
selection: accuracy, memory and value retention* (Borobia, Seguí-Mas, Tormo-Carbó).
Code, analysis plans and analysis scripts: <https://github.com/hecboar/quantized-verifiers-bon>
(tag `v1.0-submission`).

Every verifier and precision in the study scores the same frozen candidate pools, so the
tables of the paper can be recomputed from these files without a GPU.

## Contents

| Path | What it is |
|---|---|
| `store/candidates/<generator>__<split>.parquet` | Annotated candidate pools: one row per (problem, candidate) with the response text, token count, finish reason, extracted answer (`pred`), whether an answer was extractable (`parsed`), correctness under the prespecified grader (`correct`), truncation flag and canonical answer cluster (`answer_key`, -1 if no extractable answer). |
| `store/raw_candidates/...` | The same pools before annotation (text, tokens, finish reason, seed, candidate index). |
| `store/scores/<generator>__<verifier>__<condition>.parquet` | Verifier scores per candidate. ORMs: `score`. PRM: `score` (minimum step reward), `score_last`, `n_steps`, `step_rewards`. |
| `store/**/*.json` | Metadata of every pool and scoring run: model identifiers, software versions, fingerprints, quantization audit and calibration details. |
| `tables/` | All result tables produced by the analysis scripts (main study, H6, H7, revision experiments, audits). |
| `memory.json`, `memory_revision.json` | Peak verifier memory (and throughput for the revision measurement). |
| `data/items.jsonl` | Problems: id, question text, reference answer, answer type, source split. |
| `data/manifest_*.json` | Frozen problem manifests with their hashes. |
| `data/math_levels.json` | MATH difficulty level per problem. |

Generators: `qwen2.5-1.5b` (Qwen2.5-1.5B-Instruct; main study, splits `dev`, `eval` and
pilot pools), `qwen2.5-1.5b-h7` (same model, H7 problems), `qwen2.5-math-7b`
(Qwen2.5-Math-7B-Instruct, H6), `llama3.2-3b` (Llama-3.2-3B-Instruct, secondary
generator). Verifiers: `sky-1.7b`, `sky-4b`, `sky-8b` (Skywork-Reward-V2-Qwen3), `prm-7b`
(Qwen2.5-Math-PRM-7B), `skyllama-8b` (Skywork-Reward-V2-Llama-3.1-8B), `tulu-8b`
(Llama-3.1-Tulu-3-8B-RM). Conditions: `bf16`, `bf16-null`, `bf16-repeat`, `nf4`,
`gptq-w4`, `gptq-w3`, `awq-w4`, `rtn-w4`, `rtn-w8`, `gptq-w3-math` (in-domain
calibration), `gptq-w3-genlen` (length-matched generic calibration), deployed variants
(`gptq-w4-packed`, `nf4-bnb`) and development-set runs (`dev-full-*`).

## Sources and licences

Problems come from GSM8K (MIT licence) and MATH / MATH500 (MIT licence). Candidate
responses were generated with Qwen2.5-1.5B-Instruct and Qwen2.5-Math-7B-Instruct (Apache
2.0) and Llama-3.2-3B-Instruct (Llama 3.2 Community Licence). The scores, annotations and
tables are released under CC BY 4.0; use of the generated text should also respect the
terms of the generating models.

## Citation

If you use these data, please cite the paper (citation to be added on publication) and
the repository above.

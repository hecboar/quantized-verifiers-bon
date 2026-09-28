# Preregistration amendment — difficulty × test-time-compute hypotheses

**Date:** 2026-09-28. **Status at registration:** phase 0 (pilot) and phase 1 (early
signal on 400 *development* questions) completed. **No evaluation candidate has been
generated or scored** (`outputs/store/candidates/*__eval*`: none). This document is
committed to the public repository before phase 2; the commit timestamp is the
registration time. Everything below is fixed from this point on.

## 1. Motivating evidence (exploratory, development data only)

Phase-1 signal (Qwen2.5-1.5B-Instruct candidates, 32 per question, 400 dev questions:
200 GSM8K train + 200 MATH train). Change in Best-of-N accuracy vs BF16 (95% paired
bootstrap CI), split by domain. This split was examined *after* seeing the pooled
signal and is therefore exploratory; it motivates, and does not support, H4–H5.

| verifier | condition | domain | N=4 | N=32 |
|---|---|---|---|---|
| Skywork-Reward-V2-Qwen3-4B | GPTQ-W4 | GSM8K | +0.018 | +0.000 [−0.035, +0.035] |
| | | MATH | −0.005 | −0.030 [−0.055, −0.005] |
| | GPTQ-W3 | GSM8K | +0.004 | −0.015 [−0.065, +0.030] |
| | | MATH | −0.015 | −0.065 [−0.105, −0.025] |
| Skywork-Reward-V2-Qwen3-8B | GPTQ-W4 | GSM8K | −0.002 | −0.005 [−0.030, +0.020] |
| | | MATH | −0.002 | −0.025 [−0.050, +0.000] |
| | GPTQ-W3 | GSM8K | −0.003 | −0.005 [−0.055, +0.045] |
| | | MATH | −0.010 | −0.015 [−0.045, +0.015] |

Pattern: quantization damage to selection concentrates on the harder domain and grows
with the Best-of-N budget.

## 2. Change to the evaluation set (made before any evaluation data exists)

To give the confirmatory tests power where the effect is located, the evaluation set is
extended with 800 MATH *test* problems that are not in MATH500 (text-level exclusion),
sampled with seed 20260930. The original frozen manifest is unchanged.

| manifest | n | sha256 |
|---|---|---|
| `data/manifest_eval.json` (frozen 2026-09-27) | 1200 | `198d1f3535d0f5430d4f1b799d6f75bbe08a24a71ede118f6f62906ca6048df9` |
| `data/manifest_eval_mathext.json` (new) | 800 | `47e8816b5f534c1477ebc00501ff7c14ac8e17d1a963f8fabc9d2f3a503811f4` |
| `data/manifest_eval_full.json` = eval ∪ mathext (used for evaluation) | 2000 | `3686a810529d1fb62ffd47f4fc6b7cf3b0129bfd8ad43d06b9c5b25b27360061` |
| `data/manifest_dev.json` | 400 | `671ef50a1b0c4784021d7367e4be05166571c8570467d77a5c1cd7861b914ab4` |

Evaluation composition: 800 GSM8K test + 1200 MATH (400 MATH500 + 800 MATH test).
No text overlap with MATH500 or with the development set (verified). All 800 new gold
answers pass the grader's self-consistency check.

## 3. New confirmatory hypotheses

Common definitions. *MATH questions*: evaluation questions whose id starts with
`math500-` or `mathtest-`. Δ_q(N) = Best-of-N accuracy of the quantized verifier minus
that of the same verifier in BF16, for question q, estimated over 200 random subsets of
size N drawn identically for both conditions (N = 32 is the full pool). Verifiers:
Skywork-Reward-V2-Qwen3-4B and -8B; per-question values are averaged over the two
verifiers. Quantization: GPTQ, 4-bit, group size 128, generic calibration (the phase-3
`gptq-w4` condition).

- **H4 (difficulty).** On MATH questions, quantization lowers Best-of-32 accuracy:
  estimand c4 = mean_q Δ_q(32); H4 predicts c4 < 0.
- **H5 (test-time compute amplifies the damage).** On MATH questions, the loss grows
  with the budget: estimand c5 = mean_q [Δ_q(32) − Δ_q(4)]; H5 predicts c5 < 0.

## 4. Analysis

- Paired bootstrap over questions, 10,000 resamples, two-sided, percentile CIs.
- H4 and H5 join the existing confirmatory family (c1 ΔA of the 8B at GPTQ-W4, c2 memory
  frontier, c3 CH-GPTQ vs DPQ-adapted): **Holm correction over five tests**, α = 0.05.
  A missing test stays in the family with p = 1.
- Implemented in `src/vq/report.py` (`c4_math_delta_gptq_w4_n32`,
  `c5_math_growth_with_n_gptq_w4`) and covered by `tests/test_report.py`.
- Reported regardless of outcome. A non-significant result is reported as inconclusive,
  never as evidence of no effect; equivalence is claimed only if the 90% CI lies within
  ±3 points.

## 5. Explicitly exploratory (not part of the confirmatory family)

Same estimands for NF4, GPTQ-W3, AWQ-W4, the 1.7B ORM and the PRM; other budgets
N ∈ {1, 2, 8, 16}; GSM8K-only contrasts; the second generator; winner-change rates
relative to the BF16 null control (`bf16-null`).

## 6. Deviations

Any further change to the evaluation set, the estimands or the analysis after this
commit will be recorded in this file with its date and reason.

**2026-09-28 — secondary-generator prompt (before any evaluation scoring).** The first
phase-2 pools of Llama-3.2-3B-Instruct left 34.4% of candidates without an extractable
final answer (only 3.8% truncated): the model ignored the system-prompt instruction to
answer in `oxed{}` and answered in prose (e.g. "So, Katy used 42 teaspoons of sugar.").
The grader was left unchanged (it still rejects "last number in the text"). Instead, for
Llama-3.2 only, the same instruction is repeated at the end of the user turn
(`vq.generate.INSTRUCTION_IN_USER`), and its 300-question pools are regenerated. The
first pools are archived, not used. The primary generator (Qwen2.5-1.5B-Instruct, 88.7%
extractable) is unchanged. No verifier had scored any evaluation candidate at this point. Result of the
regeneration: extractable answers 65.6% -> 81.1%, candidate accuracy 42.1% -> 53.5%,
truncated 3.8% -> 3.6%.

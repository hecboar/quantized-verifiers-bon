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

## 7. Amendment 2 — 2026-09-28, after candidate inspection, before any verifier scoring

Triggered by an external methodological review of the plan and code. Recorded here so
that readers can separate what was fixed at each point in time.

### 7.1 Status record

| point in time | commit | what existed |
|---|---|---|
| original registration (§1–§5) | `074f79d` | dev signal only; eval manifests above; no eval candidate |
| Llama prompt deviation (§6) | `483a409`, `aa09919` | eval candidate pools generated and inspected (parse/accuracy/truncation rates only); Llama pools regenerated once |
| this amendment | `512a2d8` (code), this commit (text) | eval candidates exist; **no verifier has scored any eval candidate** (`outputs/store/scores/*__eval*`: none) |

Frozen candidate pools (sha256 of the annotated parquet files):

| pool | questions × candidates | sha256 |
|---|---|---|
| `candidates/qwen2.5-1.5b__eval` | 2000 × 32 | `771f0f7197325ee441ca78c20520aaa74a7fdc6354a0334cdc55d5313b870b01` |
| `candidates/llama3.2-3b__eval` | 300 × 32 | `39e667661eba5e892525b9e953a0f502be29da213f0c40d6a2acbd4f0767d953` |
| `candidates/qwen2.5-1.5b__dev` | 400 × 32 | `14965982d33440f121ee67160aa07784cc1a357f23c2040bc9bb211ea0dfdddf` |

The candidate inspection could inform only generator-side quantities (accuracy 0.521,
parseable 0.887, truncated 6.5% for Qwen); it carries no information about verifier
quantization effects.

### 7.2 Names of the hypotheses

The names in §3 overstate what the estimands test. They are renamed; the estimands are
unchanged.

- **H4 → "MATH selection degradation"** (c4). It is a within-domain effect, not a test
  of difficulty: a GSM8K-vs-MATH moderation contrast is **exploratory**.
- **H5 → "budget-dependent degradation on MATH"** (c5), where the budget is the number
  of candidates N; "test-time compute" in §3 means exactly this and nothing more.

### 7.3 Populations (frozen)

- **c1** (8B, GPTQ-W4, ΔA at N=32) is estimated on the *original* 1200-question
  manifest `manifest_eval.json` (sha `198d1f35…`), as registered before the extension.
  Estimates on the 2000-question full manifest are reported additionally, as
  exploratory. Implemented: `analyze` passes `c1_questions = manifest_eval`.
- **c4, c5** are estimated on all 1200 MATH questions of the full manifest
  (400 `math500-` + 800 `mathtest-`).
- **c3** uses exactly the configured phase-5 seeds (sky-8b {0,1,2}, sky-4b {0,1}); a
  missing or extra seed is an error, not a silent subset.

### 7.4 Tests and intervals

- p-values: paired bootstrap over questions (10,000 resamples), centred percentile
  bootstrap, two-sided. A hypothesis is **supported** only if the point estimate has
  the predicted sign **and** its Holm-adjusted p < 0.05 (family of five, missing test
  p = 1). Reported CIs are marginal 95% percentile intervals (not Holm-adjusted).
- Equivalence (90% CI inside ±3 accuracy points) is claimed only for c1 and for the
  per-condition ΔA(32) of the phase-3 grid; nowhere else.

### 7.5 Phase-5 (CH-GPTQ) decision rule

Phase 5 is the most expensive step and is run only if, after phase 3 and before any
phase-5 spending, **all** of the following hold:

1. On the eval pool, GPTQ-W4 or GPTQ-W3 of sky-4b or sky-8b shows a MATH ΔA(32) whose
   95% CI excludes 0, or |ΔA(32)| ≥ 1.5 points (there must be damage to repair);
2. that damage exceeds the matched `bf16-null` winner-change floor in harm rate H_32;
3. the quantization audits of phase 3 passed (every target module changed, no
   non-target module changed) and fingerprints match;
4. the remaining budget covers phase 5 plus phases 4 and 6 at the measured
   throughput with a 30% margin.

If phase 5 is not run, c3 stays in the family with p = 1 and the paper reports it as
not tested.

### 7.6 Additional controls and exploratory analyses (not confirmatory)

- `bf16-repeat`: BF16 rescored with the *same* batch layout, separating GPU
  nondeterminism from the batch-layout effect measured by `bf16-null`.
- Selection rates of truncated and unparseable candidates per condition and domain
  (does quantization shift selection towards truncated/unparseable outputs?).
- Blinded manual grading audit: a random sample of selected answers where BF16 and the
  quantized verifier disagree, graded without knowing the condition, to bound grader
  error.

### 7.7 Secondary generator framing

The second generator is a *generator-plus-prompt pipeline*: Llama-3.2-3B-Instruct with
the instruction repeated in the user turn (§6). This was the only prompt revision; no
other prompt or generator setting will change. Its results are exploratory.

### 7.8 Power (computed from dev data, before eval scoring)

Per-question SDs from the phase-1 dev signal (sky-4b/8b GPTQ-W4, averaged), scaled to
the 1200 eval MATH questions; α = 0.01 (roughly the Holm threshold for the first test):

| estimand | dev mean | dev sd | eval SE | power if the effect shrinks by 50% |
|---|---|---|---|---|
| c4 | −0.0275 | 0.134 | 0.0039 | 0.83 |
| c5 | −0.0236 | 0.122 | 0.0035 | 0.78 |

Dev effects are expected to shrink (winner's curse); the shrinkage figure is a
planning assumption, not a prediction.

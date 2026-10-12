# Preregistration amendment 3 — H6: verifier quantization when the verifier adds value

**Status: DRAFT for independent review (2026-10-12). Not yet frozen.** Freezing is the
commit that replaces this line with the registration record (§9). At drafting time
**no H6 data exists**: no H6 item has been sampled, no candidate generated, no
verifier scored. The only H6-related quantities inspected are counts of unused MATH
test problems per difficulty level (Level 1: 332, 2: 685, 3: 867, 4: 932, 5: 984),
which carry no information about any outcome.

This is a separate study with its own confirmatory family. The registered family
c1–c5 (`docs/preregistration.md`) and its results are unchanged.

## 1. Motivation (from the completed study; exploratory where noted)

With the registered generator (Qwen2.5-1.5B-Instruct), Skywork-Reward-V2 ORMs select
Best-of-32 answers about as well as unweighted majority vote (BF16: 71.8-73.5% vs
72.1%), while the PRM beats it (76.6%). Quantization of these verifiers to 4 bits left
selection unchanged (c1 equivalent within +-3 pp; c4, c5 not supported). A reader can
object that this is a robustness result for verifiers that barely add value over a
selection-free baseline: if the verifier contributes little, there is little for
quantization to destroy.

H6 tests whether 4-bit verifier quantization still preserves Best-of-N selection
**in a regime where the BF16 verifier adds value over majority vote**: a stronger
generator and hard problems not used anywhere in this project.

Exploratory planning evidence (completed study, MATH Level 5 questions of the
registered eval set, n = 340): BF16 Best-of-32 exceeds majority@32 by +4.4 pp
(Skywork-8B) and +7.1 pp (PRM-7B); oracle pass@32 = 56.8%. The per-question SD of
the GPTQ-W4 effect at N = 32 is 0.19 (Skywork-8B) and 0.17 (PRM-7B).

## 2. Items (frozen at the freeze commit)

- **Stratum H (confirmatory): 800 MATH test problems of Level 5** (EleutherAI/
  hendrycks_math, all 7 subjects) that appear nowhere in this project (not in MATH500,
  not in the 800-problem eval extension, not in dev; text-level exclusion after
  whitespace normalisation), sampled with seed 20261012 from the 984 eligible.
- **Stratum E (exploratory): 200 MATH test problems of Levels 1-2**, same exclusions,
  seed 20261013. Used only for an exploratory difficulty contrast under the strong
  generator.
- Gold = last `\boxed{}` of the reference solution; every gold must pass the grader's
  self-consistency check (gold graded correct against itself), else the item is
  replaced by the next in the seeded order, and the replacement is logged.
- Manifests `data/manifest_h6_hard.json`, `data/manifest_h6_easy.json` with sha256 in §9.

Rationale for MATH test: these problems were never inspected by us, have reliable
gold answers, and are graded by the same grader as the registered study. They are not
"fresh" in the contamination sense: Qwen2.5-Math reports decontamination against MATH
test, but exposure cannot be excluded. Contamination inflates generator accuracy; it
affects BF16 and quantized verifiers equally, because both score the same frozen
pools. (Alternatives considered: AIME/HMMT 2025 — too few problems for power;
Omni-MATH — mixed answer formats that the registered grader does not handle.)

## 3. Generation (frozen)

- Generator: **Qwen2.5-Math-7B-Instruct**, BF16, vLLM, `max_model_len` 4096.
- Prompt: the model's documented CoT system prompt, identical to the registered study:
  "Please reason step by step, and put your final answer within \boxed{}." (system
  turn), question as the user turn.
- 32 candidates per question: 4 seeds (0-3) x 8, temperature 0.8, top_p 1.0
  (registered values), **max_tokens 2048** (Level-5 solutions are longer; the
  registered 1024 truncated 10.8% of MATH candidates of a weaker model).
- Pools are generated once and frozen (sha256 recorded in the deviations log before
  any scoring). Generator-side statistics (accuracy, extractable, truncated) may be
  inspected before scoring; they carry no information about quantization effects.
  No generation setting will change after the pools exist.

## 4. Verifiers and conditions

- **Skywork-Reward-V2-Qwen3-8B** (ORM) and **Qwen2.5-Math-PRM-7B** (PRM, primary
  aggregate = minimum step reward, as registered), with the code, formatting and
  rotary checks of the completed study.
- Confirmatory conditions: **bf16** (reference) and **gptq-w4** (GPTQ, 4 bits, group
  size 128, generic WikiText calibration, 128 sequences of 512 tokens: the registered
  `gptq-w4`; fake-quantized weights, as in c1).
- Controls / exploratory: **bf16-null** (batch-layout noise floor), **gptq-w3**.
- Not run in H6: NF4, AWQ, RTN, the 1.7B/4B ORMs, packed kernels, memory (already
  characterised in the completed study).

## 5. Estimands and hypotheses

Notation: on question q, A_v,c(q) = Best-of-32 accuracy of verifier v under condition
c (full pool; ties broken by the registered priority, seed 11; with N = 32 the 200
registered subsets are identical, so A is deterministic up to ties). Strict grader
(registered) unless stated.

**Manipulation check M (precondition, not a quantization test).** For each verifier,
gain_v = mean_q [A_v,bf16(q) − Maj(q)], with Maj(q) = correctness of unweighted
majority vote over the 32 candidates (ties broken by the registered priority).
Reported with a 95% paired-bootstrap CI, together with oracle pass@32 and the
random-candidate rate. **Interpretation rule (fixed now):** a verifier is "in the
value-adding regime" iff the lower 95% bound of gain_v > 0. H6 conclusions about
"verifiers that add value" are drawn only for verifiers that pass M; for a verifier
that fails M its H6 test is still reported, labelled as outside the target regime.

**Confirmatory family (Holm over two tests, alpha = 0.05):**

- **H6a (ORM).** Delta_ORM = mean_q [A_sky8b,gptq-w4(q) − A_sky8b,bf16(q)] on stratum H.
  H6a predicts **equivalence within +-3 pp**.
- **H6b (PRM).** Same for Qwen2.5-Math-PRM-7B. H6b predicts **equivalence within
  +-3 pp**.

Test: TOST by paired percentile bootstrap over questions (10,000 resamples, seed 7);
p_TOST = max(P*(mean <= −0.03), P*(mean >= +0.03)) under the bootstrap distribution
(as for c1 in `scripts/review3_analyses.py`). Holm-adjusted p_TOST < 0.05 = equivalence
supported. Equivalently, the (1 − 2 x alpha_Holm) bootstrap CI lies inside +-3 pp.
If equivalence is not supported, the result is reported as **inconclusive** unless
the 95% CI of Delta excludes 0, in which case it is reported as a **detected
change** with its sign (two-sided percentile bootstrap p, reported without
multiplicity claims). The margin (3 pp) is the registered c1 margin.

Why equivalence and not "damage": the completed study gives no directional prior at
4 bits (c1, c4, c5); the scientific question is whether the null result survives in
the value-adding regime. A one-sided harm hypothesis would turn a no-damage outcome
into an uninformative non-rejection.

## 6. Exploratory analyses (not in the family; reported regardless)

1. **Value preserved.** Fraction of the verifier's gain over majority retained after
   quantization: (A_gptq-w4 − Maj) / (A_bf16 − Maj), with bootstrap CI; only for
   verifiers passing M.
2. **Where the gain lives.** Delta restricted to questions where the BF16 verifier
   and majority vote select different answers ("contested"), and where they agree;
   with n and CIs. Defined by BF16 and majority only, never by the quantized arm.
3. **Noise floor and bit-width.** bf16-null winner-change and Delta; gptq-w3 Delta
   and M-retention.
4. **Difficulty.** The same estimands on stratum E, and the H − E difference.
5. **Lenient grader** (`vq.grading.grade_lenient`, frozen) for M, H6a, H6b.
6. **Truncation.** Rates per stratum; H6a/H6b with truncated candidates removed
   before selection.
7. **Budget curve.** Delta at N in {1, 2, 4, 8, 16, 32} (200 registered subsets).

## 7. Power (planning, from the completed study)

Assumed per-question SD of Delta at N = 32: 0.21 (above the 0.17-0.19 observed on
Level 5; a stronger generator may create more contested questions). With n = 800, SE =
0.0074. The first Holm step uses alpha = 0.025 (95% CI, half-width 0.0146), the second
alpha = 0.05.

| true Delta | P(equivalence), first Holm step | P(equivalence), alpha = 0.05 |
|---|---|---|
| 0 pp | 0.96 | 0.98 |
| +-1 pp | 0.77 | 0.85 |
| +-1.5 pp | 0.52 | 0.65 |

With SD 0.25 instead: 0.85 / 0.61 at 0 / 1 pp (first step). n = 800 uses most of
the 984 eligible Level-5 problems; the design cannot be made much larger on this source.

## 8. Budget, execution and stop rules

- Estimated GPU time on one RTX 4090 (measured throughput of the completed study;
  generation of a 7B model assumed 3,500 tokens/s, ~650 generated tokens/candidate):
  generation ~1.7 h; scoring ~0.9 h (ORM) / ~0.75 h (PRM) per condition x 4
  conditions; quantization ~1.4 h; setup ~1 h. Total ~10-11 h, **~8-10 EUR** at the
  SECURE rate. **Hard cap 15 EUR**: if the measured generation throughput projects
  above the cap, stratum E is dropped first (exploratory), then gptq-w3, then
  bf16-null of the PRM; stratum H, bf16 and gptq-w4 of both verifiers are never
  dropped. Any drop is logged as a deviation before scoring.
- Order: generate and freeze pools -> record pool hashes in §9 -> score. Pods are
  deleted at the end of each step.
- No interim look at any verifier score before all four confirmatory runs (2
  verifiers x {bf16, gptq-w4}) are complete.

## 9. Registration record (filled at freeze)

| item | value |
|---|---|
| freeze commit | (this file's commit) |
| manifest_h6_hard.json sha256 | |
| manifest_h6_easy.json sha256 | |
| candidate pool sha256 (after generation, before scoring) | |

Deviations are appended below with date and reason.

# Preregistration amendment 3 — H6: verifier quantization when the verifier adds value

**Status: FROZEN 2026-10-12** (the commit that adds this line; revised after
independent review, see the review record). At freezing **no H6 candidate has been
generated and no verifier has scored any H6 item**; the items were sampled (§10) by
`vq.cli prepare-h6` and only their ids, subjects and levels were looked at. The only H6-related quantities inspected are counts of unused MATH
test problems per difficulty level (Level 1: 332, 2: 685, 3: 867, 4: 932, 5: 984),
which carry no information about any outcome. (With MATH500 excluded in full, not only
its 400 registered problems, 955 Level-5 problems are eligible.)

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

H6 tests whether 4-bit verifier quantization preserves Best-of-N selection, **and in
particular the verifier's advantage over majority vote**, with a stronger generator
and hard problems not used anywhere in this project.

Exploratory planning evidence (completed study, MATH Level-5 questions of the
registered eval set, n = 340): BF16 Best-of-32 exceeds majority@32 by +4.4 pp
(Skywork-8B) and +7.1 pp (PRM-7B); oracle pass@32 = 56.8%. Per-question SD of the
GPTQ-W4 effect at N = 32: 0.19 (Skywork-8B), 0.17 (PRM-7B).

## 2. Items (frozen at the freeze commit)

- **Stratum H (confirmatory): 800 MATH test problems of Level 5** (EleutherAI/
  hendrycks_math, all 7 subjects) that appear nowhere in this project: text-level
  exclusion (whitespace-normalised) against every item already in `data/items.jsonl`
  (registered eval, eval extension, dev) and against all 500 MATH500 problems.
  Golds (last `\boxed{}` of the reference solution) must pass the grader's
  self-consistency check (gold graded correct against itself); failing problems are
  removed from the eligible set *before* sampling and listed in the freeze log.
  Sampled with `vq.data.make_manifest`, seed 20261012.
- **Stratum E (exploratory): 200 MATH test problems of Levels 1-2**, same exclusions,
  seed 20261013.
- Manifests `data/manifest_h6_hard.json`, `data/manifest_h6_easy.json`,
  `data/manifest_h6_full.json` (= H ∪ E, generated as one pool); hashes in §10.
  Implemented in `vq.cli.prepare_h6`.

Contamination. These problems were never inspected by us and are graded by the
registered grader, but they are not "fresh" in the contamination sense: Qwen2.5-Math
reports decontamination against MATH test, yet exposure of the generator (and of the
PRM, trained on Qwen-Math outputs) cannot be excluded. Exposure does not bias the
within-pool BF16-vs-quantized contrast directly (both score the same frozen pools),
but it can change candidate redundancy, majority-vote strength and the apparent
verifier gain. The Qwen generator + Qwen-derived PRM pairing is the vendor's intended
deployment pairing and also a family-affinity limitation; both are reported.
(Alternatives considered: AIME/HMMT 2025, too few problems for power; Omni-MATH, answer
formats the registered grader does not handle; a cross-family generator, outside the
budget — listed as a limitation.)

## 3. Generation (frozen)

- Generator: **Qwen2.5-Math-7B-Instruct**, BF16, vLLM, `max_model_len` 4096.
- Prompt: the code-frozen prompt used for every generator of the registered study
  (`vq.generate.GEN_SYSTEM`, system turn): "Please reason step by step, and put your
  final answer within \boxed{}." (also Qwen2.5-Math's documented CoT prompt); question
  as the user turn.
- 32 candidates per question: seeds 0-3 x 8 samples, temperature 0.8, top_p 1.0
  (the values in `configs/experiment.yaml`), **max_tokens 2048** (Level-5 solutions are
  longer; 1024 truncated 10.8% of MATH candidates of the weaker registered generator).
- Pools are generated once and frozen; their sha256 is appended to §10 before any
  scoring. Generator-side statistics (accuracy, extractable, truncated, length) may be
  inspected before scoring. **No generation setting, item, or verifier changes after the
  pools exist**, whatever those statistics show.

## 4. Verifiers and conditions

- **Skywork-Reward-V2-Qwen3-8B** (ORM) and **Qwen2.5-Math-PRM-7B** (PRM; primary
  aggregate = minimum step reward, as registered), with the code, formatting and
  rotary checks of the completed study.
- Confirmatory conditions: **bf16** (reference) and **gptq-w4**: GPTQ, 4 bits, group
  size 128, generic WikiText calibration (128 sequences x 512 tokens), the registered
  `gptq-w4`. Weights are fake-quantized, as in c1. The claim is therefore about
  GPTQ-W4 weights under this inference stack; packed-kernel deployment agreed with
  fake quantization on 98.8-99.2% of selected answers in the completed study and is not
  re-run here (it needs a separate kernel install that would put the confirmatory runs
  at risk).
- Exploratory conditions, run after the confirmatory ones: **bf16-null** (batch-layout
  noise floor) and **gptq-w3**, both verifiers.

## 5. Definitions (frozen)

For question q and condition c of verifier v:

- **A_v,c(q)**: correctness of the Best-of-32 winner (argmax of verifier score over the
  full pool of 32; ties broken by the registered priority, seed 11). Computed with the
  registered `_acc_bon` (N = 32, 200 subsets, seed 7; at N = 32 all subsets are the
  full pool).
- **Maj(q)**: correctness of unweighted majority vote. Candidates are clustered by
  `vq.generate.answer_keys` (the registered answer canonicalisation: numeric
  normalisation, then symbolic equivalence via `math_verify`); unparseable candidates
  (key −1) do not vote; the winning cluster is the most frequent one, ties going to the
  cluster that appears first in candidate order (lowest key id); a pool with no
  parseable candidate scores 0. Maj(q) = 1 iff the winning cluster contains a correct
  candidate.
- **Delta_v(q)** = A_v,gptq-w4(q) − A_v,bf16(q); **gain_v(q)** = A_v,bf16(q) − Maj(q).
- Strict (registered) grader unless stated. All estimands are means over the 800
  stratum-H questions.

## 6. Hypotheses: confirmatory family of four, Holm, alpha = 0.05

- **H6a-ORM, H6a-PRM (accuracy equivalence).** mean Delta_v lies within +-3 pp
  (the registered c1 margin).
- **H6b-ORM, H6b-PRM (value retained).** The quantized verifier keeps at least half of
  the BF16 verifier's advantage over majority vote:
  R_v = mean_q [(A_v,gptq-w4(q) − Maj(q)) − 0.5 (A_v,bf16(q) − Maj(q))]
      = 0.5 mean gain_v + mean Delta_v,
  H6b predicts **R_v > 0**. R_v > 0 can only hold if the quantized verifier beats
  majority vote: if the BF16 verifier adds no value (gain ≤ 0) and quantization changes
  nothing, R_v ≤ 0 and H6b fails. H6b is therefore the joint, multiplicity-controlled
  test of "the verifier adds value **and** quantization preserves at least half of it",
  which a +-3 pp margin alone cannot establish when the gain itself is a few points.

Tests (paired percentile bootstrap over questions, 10,000 resamples, seed 7):

- TOST p for H6a: p = max(P*(mean <= −0.03), P*(mean >= +0.03)), where P* is the
  bootstrap distribution of the mean. This is the inversion of the percentile CI:
  p <= alpha iff the (1 − 2 alpha) percentile interval lies inside +-3 pp. Simulated
  type-I error at the boundary (n = 800, 2,000 replicates, skewed three-valued
  population): 0.056 / 0.024 / 0.006 at nominal 0.05 / 0.025 / 0.0125.
- One-sided p for H6b: p = P*(mean <= 0).
- Holm over the four p-values; a hypothesis is **supported** iff its Holm-adjusted
  p < 0.05. Reported intervals are marginal 95% percentile CIs (not Holm-adjusted).
- An unsupported H6a is reported as **inconclusive**, unless the two-sided 95% CI of
  mean Delta excludes 0 (then a **detected change**, with sign; no multiplicity claim).
- **Wording rule.** "4-bit quantization preserves the verifier's selection accuracy"
  requires H6a; "... preserves the verifier's advantage over majority vote" requires
  H6b. "Preserves verifier value" in the paper is used only for verifiers with
  supported H6b, and only if R_v keeps its sign under the lenient grader and with
  truncated candidates excluded (§7 items 4-5); otherwise the result is reported as
  possibly format-driven.
- If majority vote matches or beats a BF16 verifier (gain ≤ 0), its H6b fails by
  construction and the paper reports "no value-adding regime established for v" for that
  verifier. Nothing is regenerated or re-sampled.

**Descriptive manipulation check M (not a test).** mean gain_v with a 97.5%
(Bonferroni over two verifiers) percentile interval, oracle pass@32, random-candidate
accuracy, extractable and truncated rates.

## 7. Exploratory analyses (not in the family; all reported)

1. Retention fraction (A_gptq − Maj)/(A_bf16 − Maj) with bootstrap CI, for verifiers
   with gain CI above 0.
2. Delta on questions where BF16 and majority vote pick different answer clusters
   ("contested") and where they agree, with n and CIs (defined by BF16 and majority
   only, never by the quantized arm); winner-change rates.
3. bf16-null winner change and Delta (noise floor); gptq-w3 Delta and R_v.
4. Lenient grader (`vq.grading.grade_lenient`, frozen) for M, H6a, H6b, with majority
   recomputed on lenient correctness.
5. Truncated candidates removed before selection (score −inf) and from the vote, for
   M, H6a, H6b; truncation rates.
6. Stratum E: the same estimands; H − E difference.
7. Delta at N in {1, 2, 4, 8, 16, 32}; per-subject composition of stratum H and
   subject-stratified bootstrap of the confirmatory estimates.
8. A blinded manual audit of up to 60 questions where BF16 and majority select
   different clusters, if the strict and lenient graders disagree on the sign of M.

## 8. Power (planning, from the completed study)

- **H6a.** Assumed SD of Delta(q) 0.21 (observed 0.17-0.19 on Level 5), n = 800,
  SE 0.0074. P(equivalence) at true Delta = 0 / +-1 / +-1.5 pp: 0.93 / 0.67 / 0.41 at
  the first Holm step (alpha 0.0125), 0.98 / 0.85 / 0.65 at alpha 0.05. Variance depends
  on answer redundancy and winner disagreement, which a stronger generator changes in
  either direction.
- **H6b.** SD of the per-question R term 0.23 (observed on Level 5), SE 0.0081. With
  Delta = 0, power at true gain 3 / 4.4 / 7.1 pp: 0.36 / 0.69 / 0.99 at alpha 0.0125,
  0.59 / 0.87 / 1.00 at alpha 0.05. H6b is powered only if the BF16 gain is at least
  about 4 pp; a smaller gain makes "value preserved" undecidable, and the paper will say
  so.
- Holm success for all four tests is not targeted; each hypothesis is reported.

## 9. Budget, execution and stop rules

- Measured throughput of the completed study on one RTX 4090; generation of a 7B model
  assumed 3,500 tokens/s. Expected (~650 generated tokens/candidate): generation ~1.7 h,
  scoring ~0.9 h (ORM) / ~0.75 h (PRM) per condition, quantization ~0.35 h per GPTQ run,
  setup ~1 h; total ~10 h, **~8-10 EUR** at the SECURE rate. Worst case (every
  candidate at 2,048 tokens): generation ~5 h, scoring ~2.5x, ~25 EUR.
- **Hard cap 15 EUR, with a reserve for the confirmatory work.** Order: generation ->
  the four confirmatory runs (ORM bf16, ORM gptq-w4, PRM bf16, PRM gptq-w4) ->
  bf16-null (both) -> gptq-w3 (both). The pod's time cap is set so that the
  confirmatory runs fit at the measured generation length; exploratory runs use what is
  left and are dropped, not shortened, if the cap is reached (logged as a deviation).
  Stratum E is part of the single generated pool and is not dropped.
- No verifier score is analysed before all four confirmatory runs are complete. Pods
  are deleted after each step.

## 10. Registration record (filled at freeze)

| item | value |
|---|---|
| freeze commit | the commit that adds this table |
| eligible after exclusions | 3,700 MATH test problems (gold self-check failures: 0); Level 5: 955; Levels 1-2: 996 |
| `manifest_h6_hard.json` (800) | file sha256 `9413629bc6ca92504cba6aef4812de567a768195da69c138d5726151737a5ddb`; ids sha256 `3aa865cb4627cc4b558ec46f77820da7b845b3770a833a26a20901e2ce8ee651` |
| `manifest_h6_easy.json` (200) | file sha256 `dfba4746a06d6b3d831343d00912adf510c64bd4c93b00ea4ee501cd3dd8aff2`; ids sha256 `2d52c63e46ec323268aaf0a934656c2153475155915bb74a613c15117e9a56d7` |
| `manifest_h6_full.json` (1000) | file sha256 `887e4d55a5a1b189c88ead9f4d680c4a9be2b94d86bf4b389e4fed73ba58a19c`; ids sha256 `4f93d21e9b53bb2600e3b0b4f3d6a3f3b065b4cd8e890cc47803349b53839a71` |
| stratum H subjects | algebra 198, intermediate algebra 162, prealgebra 110, number theory 91, geometry 81, counting & probability 80, precalculus 78 |
| analysis code | `scripts/analyze_h6.py` at the freeze commit (dry-run tested on the registered pool only) |
| candidate pool sha256 (after generation, before scoring) | appended as a deviation-log entry |

## Review record

Independent review (Codex, 2026-10-12) of the first draft. Adopted: value-retention
co-primary tests (H6b) replacing the "lower bound > 0" regime gate, which becomes
descriptive with Bonferroni-adjusted intervals; complete majority-vote definition;
lenient/truncation robustness as a condition for the "preserves value" wording;
wording restricted to fake-quantized GPTQ-W4; contamination and family-affinity stated
as limitations; confirmatory runs ordered first under the cost cap; worst-case cost
stated. Not adopted: (i) the claim that the bootstrap TOST p is invalid — it is the
percentile-CI inversion, and simulation confirms nominal size (§6); (ii) a disjoint
calibration stratum or generator-selection stage, which would cost a second
generation round — H6b makes the regime part of the tested hypothesis instead;
(iii) packed kernels and a cross-family generator (budget and install risk;
limitations).

Deviations are appended below with date and reason.

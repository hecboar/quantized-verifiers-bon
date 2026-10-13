# Preregistration amendment 4 — H7: value retention where the verifier adds value

**Status: FROZEN 2026-10-13** (the commit that adds this line; revised after independent
review, see the review record). At drafting time **no H7 candidate has been generated and no
verifier has scored any H7 item**. The item set is fixed by rule (all eligible problems,
no sampling) and was materialised by `vq.cli prepare-h7`; only ids, levels and subjects
were looked at.

Separate study, own confirmatory family. Registered c1–c5 and H6
(`docs/preregistration.md`, `docs/preregistration_h6.md`) are unchanged.

## 1. Why H7

H6 (`docs/h6_results.md`) confirmed GPTQ-W4 accuracy equivalence with a strong
generator, but its value-retention tests could not succeed: with Qwen2.5-Math-7B on
Level-5 problems, majority vote matched the PRM (+0.1 pp) and beat the ORM (−3.8 pp),
so no verifier added value to retain. The claim "4-bit quantization preserves the value
a verifier adds over majority vote" therefore rests only on exploratory evidence.

H7 tests it confirmatorily **where prior data say the verifier adds value**: the
registered weak generator on hard problems. In the completed study (exploratory, MATH
Levels 4-5 of the registered eval set, n = 622), BF16 Best-of-32 beat majority@32 by
+6.6 pp (PRM-7B) and +3.5 pp (Skywork-8B). Choosing the regime from earlier data is
legitimate because H7 uses only problems never used before.

**Selection universe (disclosed).** Regimes inspected before choosing: (i) the
registered eval set with the weak generator, all questions (Skywork ORMs within ~1 pp of
majority vote; PRM +3.9 pp); (ii) its MATH Level-5 subset (Skywork-8B +4.4, PRM +7.1);
(iii) its Level 4-5 subset (+3.5, +6.6; chosen for H7 because Level 4 has enough
unused problems); (iv) H6, strong generator (ORM −3.8, PRM +0.1). The planning gains
are selected estimates and likely optimistic (winner's curse): the gains on the new
problems may be smaller. Conclusions are limited to this frozen residual Level 4-5 MATH
set, whose 905/155 level mix is the estimand (not "hard MATH" in general: H6 used most
eligible Level-5 problems).

## 2. Items (frozen)

All MATH test problems of **Levels 4 and 5** (EleutherAI/hendrycks_math, 7 subjects)
that appear nowhere in this project: text-level exclusion against every item in
`data/items.jsonl` (registered eval, eval extension, dev, H6) and all of MATH500; golds
pass the grader's self-consistency check (failures: none). **n = 1,060** (Level 4: 905,
Level 5: 155). Implemented in `vq.cli.prepare_h7`; manifest `data/manifest_h7.json`.
Contamination caveats as in H6 §2.

## 3. Generation (frozen)

Qwen2.5-1.5B-Instruct with **exactly the registered settings** (`configs/experiment.yaml`
`generation`: `vq.generate.GEN_SYSTEM` prompt, 32 candidates = seeds 0-3 x 8,
temperature 0.8, top_p 1.0, max_tokens 1024), stored under the alias
`qwen2.5-1.5b-h7` so that H7 pools and scores never mix with the registered study's.
Pools are frozen and hashed (appended below) before any scoring; generator-side
statistics may be inspected; nothing changes after the pools exist.

## 4. Verifiers, conditions, definitions, hypotheses, tests

Identical to H6 §4-§6, with "stratum H" = the 1,060 H7 problems and labels H7a/H7b:

- Verifiers Skywork-Reward-V2-Qwen3-8B and Qwen2.5-Math-PRM-7B; confirmatory conditions
  bf16 and gptq-w4 (fake-quantized, generic calibration); exploratory bf16-null and
  gptq-w3. Same A, Maj, Delta, gain definitions (H6 §5).
- **Family of four, Holm, alpha 0.05:** H7a-ORM, H7a-PRM (TOST, mean Delta within
  +-3 pp); H7b-ORM, H7b-PRM (R_v = 0.5 mean gain_v + mean Delta_v > 0, one-sided).
- Same bootstrap (10,000, seed 7), same p definitions, same wording rules, same
  lenient/truncation robustness condition for the "preserves value" wording, same
  descriptive M with 97.5% intervals.
- Analysis: the frozen `scripts/analyze_h6.py` run as
  `--generator qwen2.5-1.5b-h7 --hard data/manifest_h7.json --easy none
  --out outputs/tables/h7 --prefix H7`. The stratum-E and H−E analyses of H6 §7 are
  intentionally absent (H7 has one stratum).
- Added exploratory outputs (also produced for H6, where they change nothing): estimates
  per MATH level with truncation rates, and an equal-level-weight aggregate
  (descriptive; the fixed-population aggregate is the estimand).
- Analysis-script corrections made after independent review and before any H7 data
  (`scripts/analyze_h6.py`): explicit `--easy none`; majority-vote ties broken by first
  appearance in candidate order implemented directly (on full pools identical to the
  previous lowest-id rule; it differs only once candidates are removed); input
  invariants (pool = manifest, 32 candidates, no duplicates); in the truncation
  sensitivity a question whose 32 candidates are all truncated scores 0. Re-running H6
  with the corrected script reproduces all H6 confirmatory and exploratory numbers
  exactly.
- Wording: H7b tests the joint proposition "BF16 adds value and GPTQ-W4 keeps at least
  half of it"; it does not establish accuracy equivalence, which requires H7a.

**What H7 adds over H6.** If H7b is supported for a verifier, the paper may state
confirmatorily that 4-bit GPTQ keeps at least half of that verifier's advantage over
majority vote in a regime where the advantage exists. If majority vote matches the
verifier on these new problems, H7b fails by construction and the paper says so; no
regeneration or re-selection follows.

## 5. Power (planning, from the completed study, Levels 4-5, n = 622)

- H7b: per-question SD of the R term 0.22; SE at n = 1,060 = 0.0068. With Delta = 0,
  R = half the gain: PRM (gain 6.6 pp) power ~1.00 at alpha 0.0125; ORM (gain 3.5 pp)
  0.64 at alpha 0.0125, 0.83 at 0.05, and lower if the selected gain shrinks. All four
  tests stay co-primary under Holm, but **a non-supported H7b-ORM will be reported as
  weak evidence (likely underpowered), not as evidence that value is lost**; H7b-PRM is
  the well-powered test of the value-retention claim. r stays 0.5 (lowering it would
  weaken the substantive claim).
- H7a: SD of Delta 0.165, SE 0.0051: P(equivalence) ~1.00 at true Delta 0, 0.95 at
  +-1 pp (alpha 0.0125).

## 6. Budget and execution

Generation of a 1.5B model is fast (registered throughput ~17,000 tokens/s; ~20 min).
Scoring: 34,000 candidates per condition; the H6 scoring runs took ~35 min for 32,000
longer candidates. Two scoring pods (one per verifier, confirmatory runs first), each
~2.5 h with setup. Total ~5 h of RTX 4090 SECURE, **~4-5 EUR; cap 8 EUR**: exploratory
runs are dropped first if needed. Watchers run detached; pods deleted after each step.

## Review record

Independent review (Codex, 2026-10-13) of the draft. Adopted: selection-universe
disclosure and optimism caveat; corrected power arithmetic (SE 0.0068; ORM 0.64/0.83);
explicit reading of an unsupported H7b-ORM; the four analysis-script corrections above;
per-level estimates with truncation rates and an equal-level-weight sensitivity; explicit
absence of the stratum-E analyses; H7a required for accuracy-equivalence wording. Not
adopted: making the PRM test the sole primary (the family of four is kept for symmetry
with H6; the asymmetric power is stated instead).

## 7. Deviations

Appended below with date and reason.

## 8. Registration record (filled at freeze)

| item | value |
|---|---|
| freeze commit | the commit that adds this table's values |
| `manifest_h7.json` ids sha256 | `a2bc4e92b4dc51fa4b042b369f907cfd723ce9dc1eeb765e4694a464aca71efe` |
| `manifest_h7.json` file sha256 (git, LF) | `ad792cc16daba265d7f3305529e3e7b494c6763be039d4e9c60d24f816075f59` |
| composition | Level 4: 905, Level 5: 155; algebra 252, intermediate algebra 227, prealgebra 156, number theory 120, geometry 116, precalculus 104, counting & probability 85 |
| analysis code | `scripts/analyze_h6.py` at the freeze commit (re-verified on H6: identical outputs) |
| candidate pool sha256 | appended as a deviation-log entry before scoring |

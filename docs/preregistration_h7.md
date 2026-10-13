# Preregistration amendment 4 — H7: value retention where the verifier adds value

**Status: DRAFT for independent review (2026-10-13). Not yet frozen.** Freezing is the
commit that fills in §8. At drafting time **no H7 candidate has been generated and no
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
  --out outputs/tables/h7 --prefix H7` (no stratum E in H7; the stratum-E and H−E
  items of H6 §7 are not applicable).

**What H7 adds over H6.** If H7b is supported for a verifier, the paper may state
confirmatorily that 4-bit GPTQ keeps at least half of that verifier's advantage over
majority vote in a regime where the advantage exists. If majority vote matches the
verifier on these new problems, H7b fails by construction and the paper says so; no
regeneration or re-selection follows.

## 5. Power (planning, from the completed study, Levels 4-5, n = 622)

- H7b: per-question SD of the R term 0.22; SE at n = 1,060 = 0.0066. With Delta = 0,
  R = half the gain: PRM (gain 6.6 pp) power ~1.00 at alpha 0.0125; ORM (gain 3.5 pp)
  0.66 at alpha 0.0125, 0.84 at 0.05. If the ORM's gain on new problems is smaller,
  H7b-ORM is underpowered; this is stated in advance.
- H7a: SD of Delta 0.165, SE 0.0051: P(equivalence) ~1.00 at true Delta 0, 0.95 at
  +-1 pp (alpha 0.0125).

## 6. Budget and execution

Generation of a 1.5B model is fast (registered throughput ~17,000 tokens/s; ~20 min).
Scoring: 34,000 candidates per condition; the H6 scoring runs took ~35 min for 32,000
longer candidates. Two scoring pods (one per verifier, confirmatory runs first), each
~2.5 h with setup. Total ~5 h of RTX 4090 SECURE, **~4-5 EUR; cap 8 EUR**: exploratory
runs are dropped first if needed. Watchers run detached; pods deleted after each step.

## 7. Deviations

Appended below with date and reason.

## 8. Registration record (filled at freeze)

| item | value |
|---|---|
| freeze commit | |
| `manifest_h7.json` ids sha256 | `a2bc4e92b4dc51fa4b042b369f907cfd723ce9dc1eeb765e4694a464aca71efe` |
| `manifest_h7.json` file sha256 (git, LF) | |
| composition | Level 4: 905, Level 5: 155; algebra 252, intermediate algebra 227, prealgebra 156, number theory 120, geometry 116, precalculus 104, counting & probability 85 |
| candidate pool sha256 | appended as a deviation-log entry before scoring |

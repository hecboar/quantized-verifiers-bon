# H7 results (2026-10-13)

Registered analysis (`scripts/analyze_h6.py --prefix H7`, frozen at `bc40c08`; lenient
verdicts from `scripts/lenient_cache.py`, see the deviation log of
`docs/preregistration_h7.md`) on the frozen pools. Outputs:
`outputs/tables/h7/h7_confirmatory.json`, `outputs/tables/h7/h7_exploratory.csv`.
Generator Qwen2.5-1.5B-Instruct (registered settings), 1,060 never-used MATH Level 4-5
problems, 32 candidates each.

## Confirmatory family (Holm over four)

| hypothesis | estimate | 95% CI | p (raw) | p (Holm) | result |
|---|---|---|---|---|---|
| H7a-ORM: Delta within +-3 pp (Skywork-8B, GPTQ-W4) | -0.28 pp | [-1.42, +0.85] | 1e-4 | 2e-4 | **supported (equivalent)** |
| H7a-PRM: Delta within +-3 pp (Qwen2.5-Math-PRM-7B) | -0.57 pp | [-1.32, +0.19] | < 1e-4 | < 1e-4 | **supported (equivalent)** |
| H7b-ORM: R > 0 (keeps half the gain over majority) | +1.23 pp | [-0.14, +2.59] | 0.041 | 0.041 | **supported** (narrowly) |
| H7b-PRM | +3.54 pp | [+2.45, +4.72] | < 1e-4 | < 1e-4 | **supported** |

(One-sided p for H7b; the 95% two-sided interval of the ORM's R includes 0, consistent
with a one-sided p of 0.041. p = 0 of 10,000 resamples is reported as p < 1e-4.)

## Manipulation check M (descriptive)

| | BF16 Best-of-32 | GPTQ-W4 | majority@32 | gain over majority (97.5% CI) | retention of the gain (95% CI) |
|---|---|---|---|---|---|
| Skywork-8B | 58.96% | 58.68% | 55.94% | +3.02 pp [+0.38, +5.75] | 91% [44%, 149%] |
| PRM-7B | 64.15% | 63.58% | 55.94% | +8.21 pp [+5.85, +10.66] | 93% [84%, 102%] |

Oracle pass@32 78.6%, random candidate 36.4%, extractable 86.1%, truncated 13.4%.
Both verifiers are in the value-adding regime this time (gain intervals above 0).

**Reading (registered rules, §4 / H6 §6).** In a regime where the verifier beats
majority vote, 4-bit GPTQ keeps Best-of-32 accuracy within about 1.4 pp of BF16 for both
verifiers and keeps at least half of the advantage over majority vote: decisively for
the PRM (93% of an 8.2 pp advantage retained), narrowly for the ORM (91% of a 3.0 pp
advantage, Holm p = 0.041; as stated in advance, the ORM test had limited power). The
wording condition is met for both: R keeps its sign under the lenient grader (ORM
+1.27, PRM +3.63 pp) and with truncated candidates excluded (+1.18, +3.82 pp).

## Exploratory

- **Noise floor.** bf16-null: winner change 5.5% (ORM) / 8.7% (PRM), Delta +0.09 /
  -0.38 pp. GPTQ-W4 changes the winner on 26% / 30% of problems.
- **3 bits.** GPTQ-W3 costs 3.5 pp [-5.2, -1.8] (ORM) and 2.6 pp [-4.0, -1.2] (PRM),
  winner change 52% / 56%; the ORM's R turns negative (-1.98 pp: value lost), the PRM
  keeps a positive R (+1.56 pp). Unlike H6, 3 bits hurts in this regime.
- **Budget curve.** |Delta| <= 0.6 pp at every N for both verifiers.
- **Levels.** Level 4 (n = 905): Delta -0.33 / -0.66 pp, gain +1.7 / +7.7 pp; Level 5
  (n = 155): Delta 0.0 / 0.0, gain +11.0 / +11.0 pp. Equal-level-weight R: +2.99 / +4.34
  pp. Truncation 12.5% (L4), 18.9% (L5).
- **Contested vs agreeing problems** (BF16 winner vs majority answer): ORM +1.4
  [-1.2, +4.0] (n = 424) vs -1.4 [-2.5, -0.5] (n = 636); PRM -0.5 vs -0.6.
- Subject-stratified bootstrap: same intervals to within 0.1 pp.

## Consequences for the paper

1. The value-retention claim is now confirmatory: where verifiers add value over
   majority vote, 4-bit GPTQ preserves both their accuracy and most of that value
   (PRM: 93% of +8.2 pp). Together with H6 and c1, equivalence at 4 bits holds in three
   preregistered settings (two generators, three problem sets, ORM and PRM).
2. Three bits is regime-dependent: harmless-looking with a strong generator (H6),
   clearly harmful with the weak generator on hard problems (H7) and for small verifiers
   (main study).
3. The ORM's value-retention support is narrow; the paper must say so.

Cost (pod hours, not checked against billing): generation ~0.8 h at 0.59 USD/h, scoring
four pods ~1.5-1.8 h at 0.89-1.09 USD/h, a few rejected hosts; ~6 USD (~5.5 EUR).

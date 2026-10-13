# H6 results (2026-10-13)

Registered analysis (`scripts/analyze_h6.py` at the freeze commit) on the frozen pools
(`docs/preregistration_h6.md` §10 and deviation log). Outputs:
`outputs/tables/h6/h6_confirmatory.json`, `outputs/tables/h6/h6_exploratory.csv`.
Generator Qwen2.5-Math-7B-Instruct, stratum H = 800 unused MATH Level-5 problems,
32 candidates each.

## Confirmatory family (Holm over four)

| hypothesis | estimate | 95% CI | p (raw) | p (Holm) | result |
|---|---|---|---|---|---|
| H6a-ORM: Delta within +-3 pp (Skywork-8B, GPTQ-W4) | 0.00 pp | [-1.13, +1.13] | < 1e-4 | < 1e-4 | **supported (equivalent)** |
| H6a-PRM: Delta within +-3 pp (Qwen2.5-Math-PRM-7B) | +0.50 pp | [-0.50, +1.50] | < 1e-4 | < 1e-4 | **supported (equivalent)** |
| H6b-ORM: R > 0 (keeps half the gain over majority) | -1.88 pp | [-3.19, -0.63] | 0.998 | 0.998 | not supported |
| H6b-PRM | +0.56 pp | [-0.63, +1.69] | 0.18 | 0.36 | not supported |

(p = 0 of 10,000 bootstrap resamples is reported as p < 1e-4.)

## Manipulation check M (descriptive)

| | BF16 Best-of-32 | GPTQ-W4 | majority@32 | gain over majority (97.5% CI) |
|---|---|---|---|---|
| Skywork-8B | 72.75% | 72.75% | 76.50% | **-3.75 pp** [-6.25, -1.25] |
| PRM-7B | 76.63% | 77.13% | 76.50% | +0.13 pp [-1.88, +2.13] |

Oracle pass@32 84.6%, random candidate 60.8%, extractable 85.7%, truncated 19.4%.

**Reading (by the registered rules, §6).** With this strong generator on hard problems,
neither verifier adds value over majority vote: the ORM is worse than majority, the PRM
ties it. H6b therefore fails by construction for both and the paper reports **"no
value-adding regime established"**; the H6 question (does quantization destroy value
a verifier adds?) remains open for this generator. What H6 does establish, with high
precision, is a replication of c1 in a new regime: 4-bit GPTQ leaves Best-of-32
accuracy unchanged for both verifiers (90% CIs within +-1.4 pp), even though the
selected candidate changes on 31% (ORM) and 35% (PRM) of questions.

## Exploratory (all registered in §7)

- **Robustness of M and H6b.** Lenient grader: ORM gain -3.75, PRM +0.12 pp; truncated
  candidates excluded: ORM -4.25, PRM -0.25 pp. The "no added value" reading is not a
  format or truncation artifact; H6b keeps its sign in both sensitivities.
- **Noise floor.** bf16-null: winner change 10.9% (ORM) / 15.0% (PRM), Delta +0.25 /
  -0.12 pp. GPTQ-W4 changes the winner about 3x as often as batch layout does, without
  changing accuracy.
- **3 bits.** GPTQ-W3 Delta -0.50 [-2.25, +1.25] (ORM), -1.12 [-2.88, +0.62] (PRM);
  winner change 54% / 68%.
- **Where changes go.** On questions where the BF16 verifier picked a different answer
  from majority vote (ORM n = 173, PRM n = 131), quantization *improves* accuracy (ORM
  +2.3 [-1.7, +6.9], PRM +6.9 [+3.1, +11.5]); where they agreed it costs a little
  (-0.6, -0.8). Quantization noise pulls selection towards the majority answer, which on
  this pool is the better answer more often than the BF16 verifier's minority pick.
- **Budget curve.** |Delta| <= 0.5 pp at every N in {1, ..., 32} for both verifiers.
- **Stratum E (Levels 1-2, n = 200).** Generator accuracy 92%; Delta 0.0 for both
  verifiers; no gain over majority.
- Subject-stratified bootstrap: same intervals to within 0.1 pp.
- Retention fraction (item 1) not computed: no verifier passes M.

## Consequences for the paper

1. Strengthens the main robustness claim: 4-bit verifier quantization preserves
   Best-of-N accuracy in a second, harder regime with a stronger generator and two
   verifier types, in a preregistered test.
2. Changes the framing of the reviewer objection: the problem is not specific to the weak
   generator. With a strong math generator, majority vote matches (PRM) or beats (ORM)
   reward-model selection on hard problems, so the regime "verifier adds substantial
   value" was not reachable with these verifiers. This is itself a reportable result
   (consistent with the known strength of self-consistency for strong generators).
3. The redundancy mechanism generalises: many winner changes, no accuracy change, and
   changes drift towards the consensus answer.
4. Limitation to state: whether quantization would destroy a large verifier advantage
   remains untested; it needs a verifier/generator pair where the verifier clearly beats
   majority vote.

Cost (estimated from pod hours at 0.89 USD/h, not checked against billing): first
generation, lost, ~2.4 USD; regeneration ~2.2 USD; four scoring pods ~8 USD; total
~12.5 USD (~11.5 EUR), within the 15 EUR cap.

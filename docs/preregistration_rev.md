# Amendment 5 — revision experiments (exploratory), 2026-10-14

**Status: fixed before launch.** Motivated by an external review of the manuscript.
All experiments below are **exploratory**: they form no confirmatory family and carry
no multiplicity claims; every result is reported. No score of the new verifiers or
conditions exists at the time of this commit. Registered results (c1-c5, H6, H7) are
unchanged.

## A. Verifiers outside the Qwen / Skywork-Qwen family

- **A1** Skywork-Reward-V2-Llama-3.1-8B (`skyllama-8b`): same training data family as
  our ORMs, different backbone (Llama 3.1).
- **A2** Llama-3.1-Tulu-3-8B-RM (`tulu-8b`, AllenAI): different laboratory and data.
- Pools: the registered main pools (Qwen2.5-1.5B, 2,000 problems) and the frozen H7 pools
  (1,060 problems), unchanged.
- Conditions: main pools bf16, gptq-w4, gptq-w3; H7 pools bf16, gptq-w4 (registered
  definitions, generic calibration). Input format: the models' chat templates, as for
  the Skywork ORMs (`vq.verifiers.format_orm`); for Tulu-3 the model's missing
  `pad_token_id` is set to the tokenizer's pad token.
- Estimands (N = 32, registered `_acc_bon`, bootstrap 10,000, seed 7): Delta on the main
  2,000 and on H7 (95% CI and whether the 90% CI lies within +-3 pp, descriptive); on
  H7 the gain over majority vote (H6 rule) and R = 0.5 gain + Delta with 95% CI;
  answer transitions (candidate change, answer change, harm, benefit); gptq-w3 Delta.

## B. Task-matched calibration for 3 bits

- `gptq-w3-math`: GPTQ, 3 bits, group 128, calibrated on 128 random candidates from the
  generator's development pools (registered `random` recipe, seed 0) instead of
  WikiText. Verifiers: Skywork 1.7B and 8B, main pools.
- Estimand: Delta of gptq-w3-math vs BF16, and its difference from the generic gptq-w3
  Delta (paired bootstrap).

## C. Memory and speed

- Re-measure the Skywork deployments (BF16, NF4 bitsandbytes, packed GPTQ-W4) and the
  PRM (BF16, NF4 bitsandbytes; packed export does not support its remote code) with the
  registered workload (64 sequences x 900 tokens, RTX 4090 only), adding throughput
  (sequences/s over 3 timed passes after a warm-up). Output `outputs/memory_revision.json`;
  the registered `outputs/memory.json` (c2) is not overwritten.

## Budget and execution

RunPod balance at launch about 24 EUR; expected cost about 8 EUR, cap 16 EUR. Four pods
(A1, A2, B, C); A1/A2/B may run on any GPU of >= 24 GB from the H7 list (BF16 and
quantized runs of a verifier share the GPU); C on an RTX 4090 only. Watchers run
detached; pods are deleted after each step.

## Execution record (2026-10-14)

All four pods completed and were deleted. C (memory/throughput) and B (in-domain W3
calibration) finished; A finished every BF16 and GPTQ-W4 run on the main and H7 pools,
but the two main-pool GPTQ-W3 runs reached the 5-hour pod limit at 30/34 and 31/34
shards and are not reported (a partial, non-random subset). RunPod spend for the four
pods: 12.25 USD (balance 23.96 -> 11.71 USD). Results: `outputs/tables/revision/`
(`runs_A.csv`, `runs_B.csv`, `memory_C.csv`), `outputs/memory_revision.json`.

**2026-10-14, W3 re-run.** The two incomplete main-pool GPTQ-W3 runs of the new reward
models were re-run from scratch on two pods (the partial shards were archived, not
resumed, to avoid mixing two quantizations); both completed. Spend 2.34 USD (balance
11.71 -> 9.37 USD).

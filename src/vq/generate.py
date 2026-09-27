"""Candidate generation (vLLM, GPU) and CPU-side annotation.

Pools are generated once in BF16 and frozen; every verifier/precision scores
exactly the same candidates (plan §4.1).
"""
from __future__ import annotations

import pandas as pd

from .data import Item
from .grading import _math_equal, extract_final_answer, grade, normalize_number

GEN_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."

GENERATORS = {
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-math-1.5b": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}


def build_prompt(tokenizer, question: str) -> str:
    msgs = [{"role": "system", "content": GEN_SYSTEM}, {"role": "user", "content": question}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def answer_keys(preds: list[str | None], kind: str) -> list[int]:
    """Cluster ids of equivalent final answers within one question (-1 = unparseable)."""
    reps: list[str] = []
    keys = []
    for p in preds:
        if p is None:
            keys.append(-1)
            continue
        norm = normalize_number(p)
        canon = norm if norm is not None else p.replace(" ", "")
        found = -1
        for k, r in enumerate(reps):
            rn = normalize_number(r)
            if canon == (rn if rn is not None else r.replace(" ", "")):
                found = k
                break
            if kind == "math" and _math_equal(p, r):
                found = k
                break
        if found < 0:
            reps.append(p)
            found = len(reps) - 1
        keys.append(found)
    return keys


def annotate(df: pd.DataFrame, items: dict[str, Item]) -> pd.DataFrame:
    out = df.copy()
    preds = [extract_final_answer(t) for t in out["text"]]
    out["pred"] = preds
    out["parsed"] = [p is not None for p in preds]
    out["correct"] = [grade(t, items[q].gold, items[q].kind) for t, q in zip(out["text"], out["question_id"])]
    out["truncated"] = out["finish_reason"].eq("length")
    # use the Python list (pandas would turn None into NaN)
    pred_by_row = dict(zip(out.index, preds))
    keys = []
    for qid, grp in out.groupby("question_id", sort=False):
        grp_preds = [pred_by_row[i] for i in grp.index]
        keys.append(pd.Series(answer_keys(grp_preds, items[qid].kind), index=grp.index))
    out["answer_key"] = pd.concat(keys).loc[out.index].astype(int)
    return out


def generate_seeds(items: list[Item], seeds: tuple[int, ...], per_seed: int, one_seed, store, name: str,
                   meta: dict | None = None) -> pd.DataFrame:
    """Resumable generation: one shard per seed (a pod timeout loses at most
    one seed). `one_seed(seed)` returns rows with question_id, k (0..per_seed-1),
    text, n_tokens, finish_reason. The merged pool is validated for coverage."""
    import time

    done = set(store.shard_ids("raw_candidates", name))
    gen_tokens, gen_seconds, measured = 0, 0.0, 0
    for si, seed in enumerate(seeds):
        if si in done:
            continue
        t0 = time.perf_counter()
        df = one_seed(seed).copy()
        gen_seconds += time.perf_counter() - t0
        gen_tokens += int(df["n_tokens"].sum())
        measured += 1
        df["seed"] = seed
        df["cand"] = si * per_seed + df["k"]
        store.write_shard("raw_candidates", name, si, df.drop(columns="k"))
    merged = pd.concat([store.read_shard("raw_candidates", name, si) for si in range(len(seeds))], ignore_index=True)
    expected = {(it.id, c) for it in items for c in range(len(seeds) * per_seed)}
    got = set(zip(merged["question_id"], merged["cand"]))
    if got != expected or len(merged) != len(expected):
        raise ValueError(f"{name}: candidate coverage mismatch ({len(expected - got)} missing, {len(got - expected)} extra)")
    meta = dict(meta or {})
    meta["throughput"] = {  # decoding only (model loading excluded); feeds the budget forecast
        "output_tokens": gen_tokens, "seconds": gen_seconds, "measured_seeds": measured,
        "tok_per_s": gen_tokens / gen_seconds if gen_seconds > 0 else None,
    }
    store.write("raw_candidates", name, merged, meta=meta, overwrite=True)
    return merged


def generate_pool(
    model_id: str,
    items: list[Item],
    store,
    name: str,
    seeds: tuple[int, ...] = (0, 1, 2, 3),
    per_seed: int = 8,
    temperature: float = 0.8,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    gpu_memory_utilization: float = 0.9,
    meta: dict | None = None,
) -> pd.DataFrame:
    """GPU only (vLLM). Resumable per seed via generate_seeds."""
    from vllm import LLM, SamplingParams

    # gated models authenticate through the HF_TOKEN env var (read by huggingface_hub)
    llm = LLM(model=model_id, dtype="bfloat16", seed=0, gpu_memory_utilization=gpu_memory_utilization)
    tok = llm.get_tokenizer()
    prompts = [build_prompt(tok, it.question) for it in items]

    def one_seed(seed: int) -> pd.DataFrame:
        sp = SamplingParams(n=per_seed, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
        rows = []
        for it, o in zip(items, llm.generate(prompts, sp)):
            for k, c in enumerate(o.outputs):
                rows.append({"question_id": it.id, "k": k, "text": c.text, "n_tokens": len(c.token_ids),
                             "finish_reason": c.finish_reason})
        return pd.DataFrame(rows)

    return generate_seeds(items, seeds, per_seed, one_seed, store, name, meta=meta)

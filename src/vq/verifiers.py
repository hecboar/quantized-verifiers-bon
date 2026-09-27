"""Verifier scoring: Skywork-Reward-V2 ORMs and Qwen2.5-Math-PRM-7B.

Formatting follows the official model cards:
- Skywork-Reward-V2: chat template with [user, assistant], no system prompt,
  duplicated BOS stripped; score = logits[:, 0].
- Qwen2.5-Math-PRM: system + user + assistant where steps are joined and
  terminated with <extra_0>; step reward = softmax(logits)[..., 1] at each
  separator. Primary aggregate = min over steps (plan §4.2).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import torch

PRM_STEP_TOKEN = "<extra_0>"
PRM_SYSTEM = "Please reason step by step, and put your final answer within \\boxed{}."


# ----------------------------------------------------------------- batching
def length_batches(lengths: list[int], max_tokens: int) -> list[list[int]]:
    """Greedy length-sorted batches with (max_len * batch_size) <= max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur, cur_max = [], [], 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if cur and new_max * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, new_max = [], lengths[i]
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    return batches


def _pad_right(seqs: list[list[int]], pad_id: int, device):
    T = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), T), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), T), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    return ids.to(device), mask.to(device)


# ---------------------------------------------------------------------- ORM
@torch.no_grad()
def orm_score_ids(model, seqs: list[list[int]], pad_id: int, max_tokens: int = 16_384) -> list[float]:
    device = next(model.parameters()).device
    # HF sequence-classification pooling finds the last token via pad_token_id:
    # a pad-valued token inside a sequence would silently move the pooled position
    if any(pad_id in s for s in seqs):
        raise ValueError(f"pad token id {pad_id} occurs inside a sequence; ORM pooling would be wrong")
    out = [0.0] * len(seqs)
    for batch in length_batches([len(s) for s in seqs], max_tokens):
        ids, mask = _pad_right([seqs[i] for i in batch], pad_id, device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        for j, i in enumerate(batch):
            out[i] = float(logits[j, 0].float())
    return out


def format_orm(tokenizer, question: str, response: str) -> list[int]:
    conv = [{"role": "user", "content": question}, {"role": "assistant", "content": response}]
    text = tokenizer.apply_chat_template(conv, tokenize=False)
    bos = tokenizer.bos_token
    if bos and text.startswith(bos):
        text = text[len(bos):]
    return tokenizer(text, add_special_tokens=True)["input_ids"]


# ---------------------------------------------------------------------- PRM
def split_steps(response: str) -> list[str]:
    return [s.strip() for s in re.split(r"\n\s*\n", response) if s.strip()]


def format_prm(tokenizer, question: str, response: str) -> list[int]:
    steps = split_steps(response)
    assistant = PRM_STEP_TOKEN.join(steps) + PRM_STEP_TOKEN if steps else PRM_STEP_TOKEN
    msgs = [
        {"role": "system", "content": PRM_SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def prm_step_rewards(logits: torch.Tensor, sep_mask: torch.Tensor) -> list[float]:
    """logits [T, 2] for one sequence; sep_mask [T] bool."""
    probs = torch.softmax(logits.float(), dim=-1)[:, 1]
    return probs[sep_mask].tolist()


def aggregate_steps(rewards: list[float], how: str) -> float:
    if not rewards:
        return 0.0
    if how == "min":
        return float(min(rewards))
    if how == "last":
        return float(rewards[-1])
    raise ValueError(how)


@torch.no_grad()
def prm_score_ids(model, seqs: list[list[int]], sep_id: int, pad_id: int, max_tokens: int = 16_384) -> list[list[float]]:
    device = next(model.parameters()).device
    out: list[list[float]] = [[] for _ in seqs]
    for batch in length_batches([len(s) for s in seqs], max_tokens):
        ids, mask = _pad_right([seqs[i] for i in batch], pad_id, device)
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False)[0]
        for j, i in enumerate(batch):
            L = len(seqs[i])
            out[i] = prm_step_rewards(logits[j, :L], ids[j, :L] == sep_id)
    return out


# ------------------------------------------------------------------ loading
def patch_prm_config(cfg, tokenizer):
    """The PRM's remote code predates transformers 5 and reads
    `config.pad_token_id`, which newer configs no longer define by default."""
    if getattr(cfg, "pad_token_id", None) is None:
        cfg.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return cfg


def from_pretrained_compat(cls, name, dtype=None, **kw):
    """transformers >= 4.56 takes `dtype`; older versions (possibly pinned by
    vLLM on the pod) take `torch_dtype`."""
    try:
        return cls.from_pretrained(name, dtype=dtype, **kw)
    except TypeError as e:
        if "dtype" not in str(e):
            raise
        return cls.from_pretrained(name, torch_dtype=dtype, **kw)


@dataclass
class VerifierSpec:
    name: str
    hf_id: str
    kind: str  # "orm" | "prm"


VERIFIERS = {
    "sky-1.7b": VerifierSpec("sky-1.7b", "Skywork/Skywork-Reward-V2-Qwen3-1.7B", "orm"),
    "sky-4b": VerifierSpec("sky-4b", "Skywork/Skywork-Reward-V2-Qwen3-4B", "orm"),
    "sky-8b": VerifierSpec("sky-8b", "Skywork/Skywork-Reward-V2-Qwen3-8B", "orm"),
    "prm-7b": VerifierSpec("prm-7b", "Qwen/Qwen2.5-Math-PRM-7B", "prm"),
    # local smoke-test model (CPU)
    "sky-0.6b": VerifierSpec("sky-0.6b", "Skywork/Skywork-Reward-V2-Qwen3-0.6B", "orm"),
}


def load_verifier(spec: VerifierSpec, device: str = "cuda", dtype=torch.bfloat16):
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    token = os.environ.get("HF_TOKEN")
    tok = AutoTokenizer.from_pretrained(spec.hf_id, token=token, trust_remote_code=spec.kind == "prm")
    kwargs = dict(token=token)
    if device == "cuda":
        try:
            import flash_attn  # noqa: F401

            kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            kwargs["attn_implementation"] = "sdpa"
    if spec.kind == "orm":
        model = from_pretrained_compat(AutoModelForSequenceClassification, spec.hf_id, dtype=dtype, num_labels=1, **kwargs)
    else:
        from transformers import AutoConfig

        cfg = patch_prm_config(AutoConfig.from_pretrained(spec.hf_id, trust_remote_code=True, token=token), tok)
        model = from_pretrained_compat(AutoModel, spec.hf_id, dtype=dtype, config=cfg, trust_remote_code=True, **kwargs)
    return tok, model.to(device).eval()


class Scorer:
    """Scores (question, response) pairs; returns score and, for PRMs, the
    per-step rewards (saved so min/last can both be derived later)."""

    def __init__(self, spec: VerifierSpec, tokenizer, model, max_tokens: int = 16_384, max_len: int = 4096):
        self.spec, self.tok, self.model = spec, tokenizer, model
        self.max_tokens, self.max_len = max_tokens, max_len
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if spec.kind == "prm":
            self.sep_id = tokenizer.encode(PRM_STEP_TOKEN, add_special_tokens=False)[0]

    def encode(self, question: str, response: str) -> list[int]:
        f = format_orm if self.spec.kind == "orm" else format_prm
        return f(self.tok, question, response)

    def score(self, questions: list[str], responses: list[str]) -> dict:
        # Generation length is capped upstream so truncation here should be
        # rare; it is recorded and reported as an integrity metric.
        full = [self.encode(q, r) for q, r in zip(questions, responses)]
        truncated = [len(x) > self.max_len for x in full]
        seqs = [x[: self.max_len] for x in full]
        if self.spec.kind == "orm":
            s = orm_score_ids(self.model, seqs, self.pad_id, self.max_tokens)
            return {"score": s, "n_tokens": [len(x) for x in seqs], "truncated": truncated}
        steps = prm_score_ids(self.model, seqs, self.sep_id, self.pad_id, self.max_tokens)
        return {
            "score": [aggregate_steps(r, "min") for r in steps],
            "score_last": [aggregate_steps(r, "last") for r in steps],
            "step_rewards": steps,
            "n_steps": [len(r) for r in steps],
            "n_tokens": [len(x) for x in seqs],
            "truncated": truncated,
        }

import pandas as pd

from vq.data import Item
from vq.generate import GEN_SYSTEM, annotate, answer_keys, build_prompt


class FakeTok:
    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        body = "|".join(f"{m['role']}:{m['content']}" for m in msgs)
        return body + ("|assistant:" if add_generation_prompt else "")


def test_build_prompt_uses_system_instruction():
    p = build_prompt(FakeTok(), "What is 2+2?")
    assert p.startswith(f"system:{GEN_SYSTEM}") and p.endswith("|assistant:")
    assert "What is 2+2?" in p


def test_answer_keys_cluster_equivalent_numbers():
    keys = answer_keys(["1,000", "1000", "999", None], kind="numeric")
    assert keys[0] == keys[1] != keys[2] and keys[3] == -1


def test_answer_keys_merge_math_equivalents():
    keys = answer_keys([r"\frac{1}{2}", "0.5", r"\sqrt{8}", r"2\sqrt{2}", "3"], kind="math")
    assert keys[0] == keys[1]
    assert keys[2] == keys[3]
    assert len({keys[0], keys[2], keys[4]}) == 3


def test_annotate_adds_correctness_and_keys():
    items = {"q1": Item("q1", "?", "72", "numeric", "gsm8k-test")}
    df = pd.DataFrame({
        "question_id": ["q1"] * 3,
        "cand": [0, 1, 2],
        "text": [r"\boxed{72}", "so the answer is 71.", "no idea"],
        "finish_reason": ["stop", "stop", "length"],
    })
    out = annotate(df, items)
    assert out["correct"].tolist() == [True, False, False]
    assert out["parsed"].tolist() == [True, True, False]
    assert out["truncated"].tolist() == [False, False, True]
    assert out.loc[2, "answer_key"] == -1


def test_generation_resumes_per_seed_after_a_crash(tmp_path):
    import pytest

    from vq.generate import generate_seeds
    from vq.store import CampaignStore

    store = CampaignStore(tmp_path)
    items = [Item(f"q{i}", "?", "1", "numeric", "gsm8k-test") for i in range(3)]
    calls = []

    def one_seed(seed, fail_on=None):
        calls.append(seed)
        if seed == fail_on:
            raise RuntimeError("pod timeout")
        return pd.DataFrame([{"question_id": it.id, "k": k, "text": f"s{seed}", "n_tokens": 5, "finish_reason": "stop"}
                             for it in items for k in range(2)])

    with pytest.raises(RuntimeError):
        generate_seeds(items, (0, 1, 2), 2, lambda s: one_seed(s, fail_on=2), store, "G__eval")
    assert store.shard_ids("raw_candidates", "G__eval") == [0, 1]
    calls.clear()
    df = generate_seeds(items, (0, 1, 2), 2, one_seed, store, "G__eval")
    assert calls == [2]  # only the missing seed is regenerated
    assert len(df) == 3 * 6 and sorted(df["cand"].unique()) == list(range(6))
    assert store.done("raw_candidates", "G__eval")


def test_generation_coverage_is_validated(tmp_path):
    import pytest

    from vq.generate import generate_seeds
    from vq.store import CampaignStore

    items = [Item(f"q{i}", "?", "1", "numeric", "gsm8k-test") for i in range(2)]
    short = lambda s: pd.DataFrame([{"question_id": "q0", "k": 0, "text": "x", "n_tokens": 1, "finish_reason": "stop"}])
    with pytest.raises(ValueError, match="coverage"):
        generate_seeds(items, (0,), 1, short, CampaignStore(tmp_path), "G__eval")


def test_generation_records_its_own_throughput(tmp_path):
    """The pilot must read measured generation speed, not time a child process
    that may have skipped work (third real pilot reused old candidates)."""
    import time

    from vq.generate import generate_seeds
    from vq.store import CampaignStore

    store = CampaignStore(tmp_path)
    items = [Item(f"q{i}", "?", "1", "numeric", "gsm8k-test") for i in range(2)]

    def one_seed(seed):
        time.sleep(0.05)
        return pd.DataFrame([{"question_id": it.id, "k": k, "text": "x", "n_tokens": 100, "finish_reason": "stop"}
                             for it in items for k in range(2)])

    generate_seeds(items, (0, 1), 2, one_seed, store, "G__eval", meta={"generator": "G"})
    meta = store.meta("raw_candidates", "G__eval")
    t = meta["throughput"]
    assert t["output_tokens"] == 800 and t["seconds"] >= 0.1
    assert 0 < t["tok_per_s"] <= 800 / 0.1
    assert t["measured_seeds"] == 2 and meta["generator"] == "G"


def test_vllm_context_is_capped_to_what_generation_needs(tmp_path, monkeypatch):
    """Phase 2: Llama-3.2 defaults to a 131k context, whose KV cache (14 GiB) did not
    fit next to the weights on 24 GB. Prompts + 1024 output tokens need << 4096."""
    import sys
    import types

    from vq.generate import generate_pool
    from vq.store import CampaignStore

    seen = {}

    class FakeOut:
        def __init__(self, k):
            self.text, self.token_ids, self.finish_reason = f"x{k}", [1, 2], "stop"

    class FakeLLM:
        def __init__(self, **kw):
            seen.update(kw)

        def get_tokenizer(self):
            class T:
                def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
                    return "p"
            return T()

        def generate(self, prompts, sp):
            return [types.SimpleNamespace(outputs=[FakeOut(k) for k in range(sp.n)]) for _ in prompts]

    class FakeSP:
        def __init__(self, **kw):
            self.n = kw["n"]

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSP))
    items = [Item("q0", "?", "1", "numeric", "gsm8k-test")]
    generate_pool("m", items, CampaignStore(tmp_path), "G__eval", seeds=(0,), per_seed=2, max_tokens=1024)
    assert seen["max_model_len"] == 4096 and seen["max_model_len"] >= 1024 + 1024

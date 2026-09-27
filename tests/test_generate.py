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

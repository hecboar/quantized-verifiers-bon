import torch
import pytest

from vq.verifiers import (
    length_batches,
    orm_score_ids,
    prm_step_rewards,
    split_steps,
    aggregate_steps,
    PRM_STEP_TOKEN,
)


def test_length_batches_respects_token_budget_and_covers_all():
    lengths = [5, 50, 7, 49, 6, 100]
    batches = length_batches(lengths, max_tokens=110)
    seen = sorted(i for b in batches for i in b)
    assert seen == list(range(6))
    for b in batches:
        assert max(lengths[i] for i in b) * len(b) <= 110 or len(b) == 1


def test_orm_batched_equals_single(tiny_rm):
    g = torch.Generator().manual_seed(0)
    seqs = [torch.randint(1, 256, (int(n),), generator=g).tolist() for n in (10, 33, 21, 5, 40)]
    single = orm_score_ids(tiny_rm, seqs, pad_id=0, max_tokens=1)  # forces batch size 1
    batched = orm_score_ids(tiny_rm, seqs, pad_id=0, max_tokens=10_000)
    torch.testing.assert_close(torch.tensor(batched), torch.tensor(single), atol=1e-4, rtol=1e-4)


def test_orm_scores_returned_in_input_order(tiny_rm):
    seqs = [[5, 6, 7], [8, 9, 10, 11, 12], [13, 14]]
    a = orm_score_ids(tiny_rm, seqs, pad_id=0, max_tokens=10_000)
    b = orm_score_ids(tiny_rm, seqs[::-1], pad_id=0, max_tokens=10_000)
    assert a == pytest.approx(b[::-1], abs=1e-5)


class TestSteps:
    def test_split_on_blank_lines(self):
        assert split_steps("a\n\nb\n\n\nc") == ["a", "b", "c"]

    def test_empty_response_single_empty_step(self):
        assert split_steps("   ") == []

    def test_single_step(self):
        assert split_steps("just one") == ["just one"]


def test_prm_step_rewards_reads_positive_prob_at_separators():
    # logits [T, 2]; separators at positions 2 and 4
    logits = torch.tensor([[0., 0.], [0., 0.], [0., 2.], [0., 0.], [3., 0.]])
    mask = torch.tensor([False, False, True, False, True])
    r = prm_step_rewards(logits, mask)
    assert r == pytest.approx([torch.sigmoid(torch.tensor(2.)).item(), torch.sigmoid(torch.tensor(-3.)).item()], abs=1e-6)


def test_aggregate_steps():
    assert aggregate_steps([0.9, 0.2, 0.7], "min") == 0.2
    assert aggregate_steps([0.9, 0.2, 0.7], "last") == 0.7
    assert aggregate_steps([], "min") == 0.0  # zero-step responses get the floor score


def test_prm_step_token_constant():
    assert PRM_STEP_TOKEN == "<extra_0>"


def test_from_pretrained_falls_back_to_torch_dtype():
    from vq.verifiers import from_pretrained_compat

    calls = []

    class Old:
        @classmethod
        def from_pretrained(cls, name, **kw):
            calls.append(kw)
            if "dtype" in kw:
                raise TypeError("__init__() got an unexpected keyword argument 'dtype'")
            return "model"

    assert from_pretrained_compat(Old, "x", dtype="bf16", token=None) == "model"
    assert calls[-1] == {"torch_dtype": "bf16", "token": None}

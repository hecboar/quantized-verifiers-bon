import pytest
import torch


def make_tiny_rm(seed: int = 0, layers: int = 2):
    from transformers import Qwen3Config, Qwen3ForSequenceClassification

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_labels=1,
        pad_token_id=0, max_position_embeddings=256,
    )
    model = Qwen3ForSequenceClassification(cfg).eval()
    return model


@pytest.fixture
def tiny_rm():
    return make_tiny_rm()


@pytest.fixture
def calib_ids():
    g = torch.Generator().manual_seed(0)
    return [torch.randint(1, 256, (1, int(n)), generator=g) for n in torch.randint(12, 40, (16,), generator=g)]

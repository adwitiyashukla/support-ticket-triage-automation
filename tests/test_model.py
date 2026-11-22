import pytest
import torch

from triage.model import ScreenerModel, apply_rope, build_rope_cache


def build(**overrides):
    settings = dict(vocab_size=300, n_layer=2, n_head=4, d_model=32, d_ff=64, max_seq_len=16, dropout=0.0, rope_theta=10000.0, n_labels=5)
    settings.update(overrides)
    model = ScreenerModel(**settings)
    model.eval()
    return model


def test_rejects_head_dimension_mismatch():
    with pytest.raises(ValueError):
        build(n_head=5)


def test_forward_shapes():
    model = build()
    ids = torch.randint(3, 300, (2, 9))
    mask = torch.ones(2, 9, dtype=torch.long)
    assert model.forward_lm(ids, mask).shape == (2, 9, 300)
    assert model.forward_cls(ids, mask).shape == (2, 5)


def test_attention_is_causal():
    model = build()
    ids = torch.randint(3, 300, (1, 10))
    mask = torch.ones(1, 10, dtype=torch.long)
    with torch.no_grad():
        first = model.forward_lm(ids, mask)
        changed = ids.clone()
        changed[0, 6:] = torch.randint(3, 300, (4,))
        second = model.forward_lm(changed, mask)
    assert torch.allclose(first[:, :6], second[:, :6], atol=1e-5)


def test_padding_does_not_change_pooled_output():
    model = build()
    ids = torch.randint(3, 300, (1, 12))
    short_mask = torch.ones(1, 7, dtype=torch.long)
    padded_mask = torch.zeros(1, 12, dtype=torch.long)
    padded_mask[0, :7] = 1
    with torch.no_grad():
        unpadded = model.forward_cls(ids[:, :7], short_mask)
        padded = model.forward_cls(ids, padded_mask)
    assert torch.allclose(unpadded, padded, atol=1e-5)


def test_sequence_longer_than_window_is_rejected():
    model = build()
    ids = torch.randint(3, 300, (1, 20))
    with pytest.raises(ValueError):
        model.forward_lm(ids, torch.ones(1, 20, dtype=torch.long))


def test_rope_preserves_vector_norm():
    cos, sin = build_rope_cache(8, 16, 10000.0)
    x = torch.randn(1, 2, 5, 8)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)


def test_embedding_and_head_share_weights():
    model = build()
    assert model.lm_head.weight is model.embedding.weight
    assert model.parameter_count() > 0

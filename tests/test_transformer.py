import pytest
import torch
import torch.nn.functional as F

from src.model.transformer import CausalTransformer, RoPE, SlidingWindowAttention, apply_rope
from tests.helpers import ARCHS, make_tiny_cfg


@pytest.fixture(params=list(ARCHS), ids=list(ARCHS))
def model(request):
    return CausalTransformer(make_tiny_cfg(**ARCHS[request.param]))


def test_forward_shapes_and_loss(model):
    x = torch.randint(1, 64, (2, 24))
    out = model(x, targets=x)
    assert torch.isfinite(out["loss"])
    assert model(x)["logits"].shape == (2, 24, 64)
    if model.config.n_experts > 1:
        assert "aux_loss" in out


def test_one_step_reduces_loss(model):
    x = torch.randint(1, 64, (2, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = model(x, targets=x)["loss"]
    first.backward()
    opt.step()
    assert model(x, targets=x)["loss"] < first


def test_is_causal(model):
    """Changing a future token must not change earlier logits."""
    model.eval()
    x = torch.randint(1, 64, (1, 24))
    y = x.clone()
    y[0, -1] = (x[0, -1] % 63) + 1
    a, b = model(x)["logits"], model(y)["logits"]
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_generate_length(model):
    prompt = torch.randint(3, 64, (1, 5))
    out = model.generate(prompt, max_new_tokens=6, temperature=0.0, eos_token_id=-1)
    assert out.shape == (1, 11)
    assert torch.equal(out[:, :5], prompt)


@pytest.mark.parametrize("arch", ["dense", "nsa_moe"])
def test_kv_cache_matches_full_forward(arch):
    """Token-by-token decoding with the KV cache should give the same logits as one full pass."""
    # nsa_moe here has T <= swa window, so it exercises the compressed-attention cache path.
    cfg = make_tiny_cfg(**{**ARCHS[arch], "swa_window_size": 64})
    model = CausalTransformer(cfg).eval()
    x = torch.randint(1, 64, (1, 12))
    full = model(x)["logits"]
    out = model(x[:, :4])
    steps = [out["logits"]]
    for t in range(4, 12):
        out = model(x[:, t:t + 1], past_kv=out["past_kv"])
        steps.append(out["logits"])
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-4)


@pytest.mark.xfail(strict=True, reason=(
    "Known bug: chunked SWA uses is_causal=True with non-square Q/K, which PyTorch aligns "
    "top-left. Tokens at position >= window see a window shifted back by W and never themselves. "
    "Remove this marker once fixed."
))
def test_sliding_window_matches_reference_mask():
    W, T, D, H = 4, 12, 16, 2
    cfg = make_tiny_cfg(d_model=D, n_heads=H, swa_window_size=W, full_attention_every_n=2)
    attn = SlidingWindowAttention(cfg).eval()
    x = torch.randn(1, T, D)
    cos, sin = RoPE(cfg.d_head, 64)(x)
    with torch.no_grad():
        out, _ = attn(x, cos, sin)
        q, k, v = (t.view(1, T, H, D // H).transpose(1, 2) for t in attn.qkv_proj(x).split(D, -1))
        c, s = cos[:, :T].transpose(1, 2), sin[:, :T].transpose(1, 2)
        q, k = apply_rope(q, c, s), apply_rope(k, c, s)
        i, j = torch.arange(T)[:, None], torch.arange(T)[None]
        band = (j <= i) & (j > i - W)  # each token sees itself + previous W-1
        ref = F.scaled_dot_product_attention(q, k, v, attn_mask=band)
        ref = attn.out_proj(ref.transpose(1, 2).reshape(1, T, D))
    assert torch.allclose(out, ref, atol=1e-5)

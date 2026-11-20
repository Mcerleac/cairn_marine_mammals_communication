"""Shared constants and builders for toy-sized test models."""

from src.model.config import TransformerConfig

# Real configs use 9 DAC codebooks x 1024 codes; tokens on disk are 1..1024 (0 = PAD).
N_CODEBOOKS = 9
VOCAB = 9219
SEP = 9218

# Tiny versions of each architecture family used in configs/.
ARCHS = {
    "dense": dict(),
    "swa_moe": dict(swa_window_size=8, full_attention_every_n=2, n_experts=4, moe_top_k=2, expert_d_ff=32),
    "nsa_moe": dict(
        swa_window_size=8, full_attention_every_n=2, compressed_attn_stride=4, compressed_attn_chunk=8,
        n_experts=4, moe_top_k=2, expert_d_ff=32, moe_aux_weight=0.0, use_bias_routing=True,
        use_split_qkv=True,
    ),
}


def make_tiny_cfg(**overrides) -> TransformerConfig:
    fields = dict(vocab_size=64, max_seq_len=64, n_layers=2, n_heads=2, d_model=32, d_ff=64, dropout=0.0)
    fields.update(overrides)
    return TransformerConfig(**fields)

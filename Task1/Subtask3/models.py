"""
Transformer model components for the encoder-decoder code-repair task.

Includes:
    - RoPE (Rotary Position Embedding)
    - FeedForwardNN (SwiGLU variant)
    - MHSA (Multi-Head Self-Attention with RoPE)
    - TransformerBlock 
"""

import jax
import jax.numpy as jnp
from flax import nnx


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class RoPE(nnx.Module):
    """Rotary Position Embedding (complex-number formulation)."""

    def __init__(self, head_dim: int, max_seq_len: int = 4096,
                 base: float = 10000.0):
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even. Got {head_dim}")
        inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
        positions = jnp.arange(max_seq_len)
        angles = jnp.outer(positions, inv_freq)
        self.complex_freqs = nnx.Cache(jnp.exp(1j * angles))

    def __call__(self, x: jax.Array) -> jax.Array:
        """x: (batch, seq_len, n_heads, head_dim)."""
        seq_len = x.shape[1]
        freqs = self.complex_freqs[...][:seq_len, :][None, :, None, :]
        x_paired = x.reshape(*x.shape[:-1], -1, 2)
        x_complex = jax.lax.complex(x_paired[..., 0], x_paired[..., 1])
        rotated = x_complex * freqs
        return jnp.stack([rotated.real, rotated.imag], axis=-1).reshape(x.shape)


# ---------------------------------------------------------------------------
# Feed-Forward
# ---------------------------------------------------------------------------

class FeedForwardNN(nnx.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, d_model: int, hidden_dim: int, rngs: nnx.Rngs):
        self.w_gate = nnx.Linear(d_model, hidden_dim, use_bias=False, rngs=rngs)
        self.w_up = nnx.Linear(d_model, hidden_dim, use_bias=False, rngs=rngs)
        self.w_down = nnx.Linear(hidden_dim, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.w_down(self.w_up(x) * jax.nn.silu(self.w_gate(x)))


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------

class MHSA(nnx.Module):
    """Multi-Head Self-Attention with RoPE."""

    def __init__(self, d_model: int, max_seq_len: int,
                 d_k: int | None = None, n_heads: int = 8,
                 dropout_rate:float=0.0,
                 rngs: nnx.Rngs | None = None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model
        self.head_size = self.d_k // n_heads

        self.w_q = nnx.Linear(d_model, self.d_k, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(d_model, self.d_k, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(d_model, self.d_k, use_bias=False, rngs=rngs)
        self.w_o = nnx.Linear(self.d_k, d_model, use_bias=False, rngs=rngs)

        self.rope = RoPE(head_dim=self.head_size, max_seq_len=max_seq_len)
        self.attn_dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.resid_dropout = nnx.Dropout(dropout_rate, rngs=rngs)
    def __call__(self, x: jax.Array,
                 mask: jax.Array | None = None) -> jax.Array:
        """x: (batch, seq_len, d_model)."""
        batch, seq_len, _ = x.shape
        hs = self.head_size

        q = self.w_q(x).reshape(batch, seq_len, self.n_heads, hs)
        k = self.w_k(x).reshape(batch, seq_len, self.n_heads, hs)
        v = self.w_v(x).reshape(batch, seq_len, self.n_heads, hs)

        # Apply RoPE then transpose to (batch, heads, seq, head_dim)
        q = jnp.transpose(self.rope(q), (0, 2, 1, 3))
        k = jnp.transpose(self.rope(k), (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = 1.0 / jnp.sqrt(jnp.float32(hs))
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale
        if mask is not None:
            scores = jnp.where(mask, scores, -1e12)
        weights = jax.nn.softmax(scores, axis=-1)
        weights = self.attn_dropout(weights) 

        ctx = jnp.transpose(jnp.matmul(weights, v), (0, 2, 1, 3))
        out = self.w_o(ctx.reshape(x.shape[0], x.shape[1], -1))
        
        return self.resid_dropout(out)


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nnx.Module):
    """Encoder block: Pre-Norm Self-Attention + FFN with residual."""

    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int, rngs: nnx.Rngs, dropout_rate: float=0.0):
        self.attn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.mha = MHSA(d_model, n_heads=n_heads,
                        max_seq_len=max_seq_len, dropout_rate=dropout_rate, rngs=rngs)
        self.ffn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        hidden_dim = int((8 / 3) * d_model)
        self.ffn = FeedForwardNN(d_model, hidden_dim, rngs)
        self.ffn_dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array,
                 mask: jax.Array | None = None) -> jax.Array:
        x = x + self.mha(self.attn_norm(x), mask)
        x = x + self.ffn_dropout(self.ffn(self.ffn_norm(x)))

        return x

#-----------------------------------------------------------------------------
# Main Models
#-----------------------------------------------------------------------------
import jax.numpy as jnp
from flax import nnx

class picoGPT(nnx.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, max_seq_len: int, rngs: nnx.Rngs,dropout_rate:float=0.0):
        
        # We explicitly separate the token embedding and the lm_head
        self.token_emb = nnx.Embed(vocab_size, d_model, rngs=rngs)
        
        # Register the blocks as a list of nnx.Modules
        self.blocks =nnx.List( [
            TransformerBlock(d_model, num_heads, max_seq_len, rngs, dropout_rate=dropout_rate) 
            for _ in range(num_layers)
        ])
        
        self.final_norm = nnx.RMSNorm(d_model, rngs=rngs)
        
        # use_bias=False is standard for the final projection in modern LLMs
        self.lm_head = nnx.Linear(d_model, vocab_size, use_bias=False, rngs=rngs)

    def compute_causal_mask(self, seq_len: int) -> jax.Array:
        # Generates the static lower-triangular mask on the device
        return jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x: (batch_size, seq_len) array of token IDs
        """
        seq_len = x.shape[1]
        
        # 1. Generate the causal mask once for the forward pass
        mask = self.compute_causal_mask(seq_len)
        
        # 2. Token embedding (Positional embedding is handled by RoPE inside MHSA)
        x = self.token_emb(x)
        
        # 3. Pass through transformer layers
        for block in self.blocks:
            # The TransformerBlock MUST forward the mask to the MHSA layer
            x = block(x, mask=mask)
            
        # 4. Final normalization and projection to vocabulary logits
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
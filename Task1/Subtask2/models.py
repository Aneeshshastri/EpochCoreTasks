"""
Transformer model components for the encoder-decoder code-repair task.

Includes:
    - RoPE (Rotary Position Embedding)
    - FeedForwardNN (SwiGLU variant)
    - MHSA (Multi-Head Self-Attention with RoPE)
    - MHCA (Multi-Head Cross-Attention — no RoPE)
    - TransformerBlock (Encoder block)
    - DecoderBlock (Decoder block with self-attn + cross-attn + FFN)
    - TransformerModel (Full encoder-decoder with modular encode/decode)
    - PointerGeneratorTransformer (Wraps TransformerModel with a learned
      gate for scheduled sampling during training)
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

        ctx = jnp.transpose(jnp.matmul(weights, v), (0, 2, 1, 3))
        return self.w_o(ctx.reshape(batch, seq_len, self.d_k))


class MHCA(nnx.Module):
    """Multi-Head Cross-Attention (no RoPE — queries and keys come from
    different sequences with different positional semantics)."""

    def __init__(self, d_model: int, max_seq_len: int,
                 d_k: int | None = None, n_heads: int = 8,
                 rngs: nnx.Rngs | None = None):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else d_model
        self.head_size = self.d_k // n_heads

        self.w_q = nnx.Linear(d_model, self.d_k, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(d_model, self.d_k, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(d_model, self.d_k, use_bias=False, rngs=rngs)
        self.w_o = nnx.Linear(self.d_k, d_model, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array, y: jax.Array,
                 mask: jax.Array | None = None):
        """
        x (query source):  (batch, seq_q, d_model)
        y (key/val source): (batch, seq_kv, d_model)

        Returns (output, attn_weights).
        """
        batch, seq_q, _ = x.shape
        _, seq_kv, _ = y.shape
        hs = self.head_size

        q = jnp.transpose(
            self.w_q(x).reshape(batch, seq_q, self.n_heads, hs),
            (0, 2, 1, 3),
        )
        k = jnp.transpose(
            self.w_k(y).reshape(batch, seq_kv, self.n_heads, hs),
            (0, 2, 1, 3),
        )
        v = jnp.transpose(
            self.w_v(y).reshape(batch, seq_kv, self.n_heads, hs),
            (0, 2, 1, 3),
        )

        scale = 1.0 / jnp.sqrt(jnp.float32(hs))
        scores = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * scale
        if mask is not None:
            scores = jnp.where(mask, scores, -1e12)
        weights = jax.nn.softmax(scores, axis=-1)

        ctx = jnp.transpose(jnp.matmul(weights, v), (0, 2, 1, 3))
        output = self.w_o(ctx.reshape(batch, seq_q, self.d_k))
        return output, weights


# ---------------------------------------------------------------------------
# Transformer Blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nnx.Module):
    """Encoder block: Pre-Norm Self-Attention + FFN with residual."""

    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int, rngs: nnx.Rngs):
        self.attn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.mha = MHSA(d_model, n_heads=n_heads,
                        max_seq_len=max_seq_len, rngs=rngs)
        self.ffn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        hidden_dim = int((8 / 3) * d_model)
        self.ffn = FeedForwardNN(d_model, hidden_dim, rngs)

    def __call__(self, x: jax.Array,
                 mask: jax.Array | None = None) -> jax.Array:
        x = x + self.mha(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class DecoderBlock(nnx.Module):
    """Decoder block: Self-Attn → Cross-Attn → FFN, all Pre-Norm."""

    def __init__(self, d_model: int, n_heads: int,
                 max_seq_len: int, rngs: nnx.Rngs):
        self.self_attn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.self_attn = MHSA(d_model, n_heads=n_heads,
                              max_seq_len=max_seq_len, rngs=rngs)
        self.cross_attn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.cross_attn = MHCA(d_model, n_heads=n_heads,
                               max_seq_len=max_seq_len, rngs=rngs)
        self.ffn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        hidden_dim = int((8 / 3) * d_model)
        self.ffn = FeedForwardNN(d_model, hidden_dim, rngs)

    def __call__(self, x: jax.Array, enc_output: jax.Array,
                 self_attn_mask=None, cross_attn_mask=None):
        """Returns (decoder_output, cross_attn_weights)."""
        x = x + self.self_attn(self.self_attn_norm(x), self_attn_mask)
        ca_out, ca_weights = self.cross_attn(
            self.cross_attn_norm(x), enc_output, cross_attn_mask,
        )
        x = x + ca_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, ca_weights


# ---------------------------------------------------------------------------
# Full Encoder-Decoder Transformer
# ---------------------------------------------------------------------------

class TransformerModel(nnx.Module):
    """
    Encoder-Decoder Transformer with modular encode / decode API.

    Call signature
    --------------
    __call__(x, y)  →  logits   (teacher-forced training)
    encode(x)       →  (enc_h, enc_mask)
    decode(tokens, enc_h, enc_mask) →  (logits, dec_hidden, cross_attn_w)
    decode_from_embed(embed, enc_h, enc_mask, ids_for_mask) → same
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_heads: int = 4, num_layers: int = 2,
                 max_seq_len: int = 263,
                 pad_token: int = 0, start_token: int = 1,
                 rngs: nnx.Rngs | None = None):
        self.d_model = d_model
        self.pad_token = pad_token
        self.start_token = start_token

        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)
        self.enc_blocks = nnx.List(
            [TransformerBlock(d_model, n_heads, max_seq_len, rngs)
             for _ in range(num_layers)]
        )
        self.dec_blocks = nnx.List(
            [DecoderBlock(d_model, n_heads, max_seq_len, rngs)
             for _ in range(num_layers)]
        )
        self.enc_norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.dec_norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, vocab_size, rngs=rngs)

    # -- Encoder -----------------------------------------------------------

    def encode(self, x: jax.Array):
        """Encode input tokens.

        Returns (enc_hidden, enc_mask) where enc_mask can be passed
        directly to the decoder for cross-attention masking.
        """
        enc_mask = (x != self.pad_token)[:, None, None, :]
        enc_h = self.embed(x)
        for block in self.enc_blocks:
            enc_h = block(enc_h, mask=enc_mask)
        enc_h = self.enc_norm(enc_h)
        return enc_h, enc_mask

    # -- Decoder -----------------------------------------------------------

    def decode(self, dec_in: jax.Array, enc_h: jax.Array,
               enc_mask: jax.Array):
        """Decode from token IDs.

        Returns (logits, dec_hidden, cross_attn_weights).
        """
        dec_embed = self.embed(dec_in)
        return self.decode_from_embed(dec_embed, enc_h, enc_mask, dec_in)

    def decode_from_embed(self, dec_embed: jax.Array, enc_h: jax.Array,
                          enc_mask: jax.Array,
                          dec_ids_for_mask: jax.Array):
        """Decode from pre-computed embeddings (used by the pointer-gen
        gate to pass mixed teacher / model embeddings).

        Parameters
        ----------
        dec_embed : (batch, dec_len, d_model)
        enc_h : (batch, enc_len, d_model)
        enc_mask : (batch, 1, 1, enc_len)
        dec_ids_for_mask : (batch, dec_len) — used only for padding mask.
        """
        seq_len = dec_embed.shape[1]

        causal = jnp.tril(
            jnp.ones((seq_len, seq_len), dtype=jnp.bool_)
        )[None, None, :, :]
        pad_mask = (dec_ids_for_mask != self.pad_token)[:, None, None, :]
        self_attn_mask = causal & pad_mask
        cross_attn_mask = enc_mask

        dec_h = dec_embed
        ca_weights = None
        for block in self.dec_blocks:
            dec_h, ca_weights = block(
                dec_h, enc_h, self_attn_mask, cross_attn_mask,
            )
        dec_h = self.dec_norm(dec_h)
        logits = self.head(dec_h)
        return logits, dec_h, ca_weights

    # -- Combined (teacher-forced) -----------------------------------------

    def __call__(self, x: jax.Array, y: jax.Array) -> jax.Array:
        """Full teacher-forced forward pass.

        Parameters
        ----------
        x : (batch, enc_len) encoder input tokens
        y : (batch, full_seq_len) target with START / END / PAD

        Returns
        -------
        logits : (batch, full_seq_len - 1, vocab_size)
            Predicts y[:, 1:]  (the shifted target).
        """
        dec_in = y[:, :-1]                          # [START, t1, … , tn]
        enc_h, enc_mask = self.encode(x)
        logits, _, _ = self.decode(dec_in, enc_h, enc_mask)
        return logits


# ---------------------------------------------------------------------------
# Pointer-Generator Transformer
# ---------------------------------------------------------------------------

class PointerGeneratorTransformer(nnx.Module):
    """
    Wraps :class:`TransformerModel` with a learned pointer-generator gate
    that enables *scheduled sampling* during training.

    Training protocol (single-pass merged probability)
    --------------------------------------------------
    The gate is learned implicitly through backpropagation from the Negative
    Log-Likelihood loss of the merged probability distribution.
    p_gen determines whether to generate from vocabulary or copy from the
    source sequence using cross-attention weights.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_heads: int = 4, num_layers: int = 2,
                 max_seq_len: int = 263,
                 pad_token: int = 0, start_token: int = 1,
                 rngs: nnx.Rngs | None = None):
        self.transformer = TransformerModel(
            vocab_size, d_model, n_heads, num_layers,
            max_seq_len, pad_token, start_token, rngs,
        )
        self.d_model = d_model
        self.pad_token = pad_token
        self.start_token = start_token

        # Gate MLP: dec_hidden → scalar
        self.gate_fc = nnx.Linear(d_model, 1, rngs=rngs)
        self.head_weights = nnx.Param(jnp.zeros((n_heads,)))

    # Delegate encode / decode so greedy_decode works transparently.
    def encode(self, x):
        return self.transformer.encode(x)

    def decode(self, dec_in, enc_h, enc_mask, x):
        """
        Autoregressive decode step including the pointer-generator scatter.
        Returns probabilities instead of raw logits.
        """
        logits_tf, dec_h, ca_w = self.transformer.decode(dec_in, enc_h, enc_mask)

        # 1. Calculate p_gen and P_vocab
        p_gen = jax.nn.sigmoid(self.gate_fc(dec_h))
        p_vocab = jax.nn.softmax(logits_tf, axis=-1)

        # 2. Calculate P_copy
        hw = jax.nn.softmax(self.head_weights.value)
        p_copy = jnp.einsum('h,bhde->bde', hw, ca_w)

        # 3. Scale distributions
        p_vocab_scaled = p_vocab * p_gen
        p_copy_scaled = p_copy * (1 - p_gen)

        # 4. Scatter
        batch_size, dec_len, enc_len = p_copy_scaled.shape
        x_expanded = jnp.broadcast_to(x[:, None, :], (batch_size, dec_len, enc_len))
        
        b_idx = jnp.arange(batch_size)[:, None, None]
        d_idx = jnp.arange(dec_len)[None, :, None]
        
        p_copy_vocab = jnp.zeros_like(p_vocab_scaled)
        p_copy_vocab = p_copy_vocab.at[b_idx, d_idx, x_expanded].add(p_copy_scaled)

        merged_probs = p_vocab_scaled + p_copy_vocab
        return merged_probs, dec_h, ca_w

    def __call__(self, x: jax.Array, y: jax.Array):
        """
        Returns the merged probability distribution.

        P_merged : (batch, L, vocab_size)
        where L = full_seq_len - 1.
        """
        dec_in = y[:, :-1]          # decoder input: [START, t1, …, tn]

        # 1. Extract Base Tensors
        enc_h, enc_mask = self.transformer.encode(x)
        logits_tf, dec_h, ca_w = self.transformer.decode(
            dec_in, enc_h, enc_mask,
        )

        # 2. Calculate p_gen
        p_gen = jax.nn.sigmoid(self.gate_fc(dec_h))        # (B, Ld, 1)

        # 3. Calculate P_vocab
        p_vocab = jax.nn.softmax(logits_tf, axis=-1)       # (B, Ld, V)

        # 4. Calculate P_copy (The Attention Map)
        hw = jax.nn.softmax(self.head_weights.value)       # (n_heads,)
        p_copy = jnp.einsum('h,bhde->bde', hw, ca_w)       # (B, Ld, Le)

        # 5. Merge (The Scatter)
        p_vocab_scaled = p_vocab * p_gen
        p_copy_scaled = p_copy * (1 - p_gen)

        batch_size, dec_len, enc_len = p_copy_scaled.shape
        x_expanded = jnp.broadcast_to(x[:, None, :], (batch_size, dec_len, enc_len))
        
        b_idx = jnp.arange(batch_size)[:, None, None]
        d_idx = jnp.arange(dec_len)[None, :, None]
        
        p_copy_vocab = jnp.zeros_like(p_vocab_scaled)
        p_copy_vocab = p_copy_vocab.at[b_idx, d_idx, x_expanded].add(p_copy_scaled)

        merged_probs = p_vocab_scaled + p_copy_vocab

        # 6. Return
        return merged_probs

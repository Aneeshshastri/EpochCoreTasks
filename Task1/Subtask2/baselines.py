"""
Baseline model: Bidirectional LSTM + Self-Attention.

This is NOT an encoder-decoder model.  It maps the input sequence
directly to output logits at each position (non-autoregressive).
"""

import jax
import jax.numpy as jnp
from flax import nnx
from models import MHSA


class RNNSelfAttentionModel(nnx.Module):
    """
    Bidirectional LSTM followed by Multi-Head Self-Attention.

    Takes a buggy code sequence and outputs logits for the fixed code
    at every position (same-length mapping — no autoregressive decoding).

    Parameters
    ----------
    vocab_size : int
        Total vocabulary size (including PAD / START / END).
    d_model : int
        Embedding and output dimension.
    hidden_dim : int
        Total LSTM hidden dimension (split in half for each direction).
    n_heads : int
        Number of self-attention heads.
    max_seq_len : int
        Maximum sequence length (including START / END / PAD).
    pad_token : int
        Token ID used for padding (masked during attention).
    rngs : nnx.Rngs
        RNG streams.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 hidden_dim: int = 128, n_heads: int = 4,
                 max_seq_len: int = 263, pad_token: int = 0,
                 rngs: nnx.Rngs | None = None):
        self.embed = nnx.Embed(vocab_size, d_model, rngs=rngs)

        fw = nnx.RNN(nnx.LSTMCell(d_model, hidden_dim // 2, rngs=rngs))
        bw = nnx.RNN(nnx.LSTMCell(d_model, hidden_dim // 2, rngs=rngs))
        self.lstm = nnx.Bidirectional(fw, bw)

        self.mha = MHSA(d_model=hidden_dim, max_seq_len=max_seq_len,
                        n_heads=n_heads, rngs=rngs)
        self.head = nnx.Linear(hidden_dim, vocab_size, rngs=rngs)
        self.pad_token = pad_token

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Parameters
        ----------
        x : (batch, seq_len) — input token IDs.

        Returns
        -------
        logits : (batch, seq_len, vocab_size)
        """
        mask = (x != self.pad_token)[:, None, None, :]
        h = self.embed(x)          # (batch, seq, d_model)
        h = self.lstm(h)           # (batch, seq, hidden_dim)
        h = self.mha(h, mask=mask) # (batch, seq, hidden_dim)
        return self.head(h)        # (batch, seq, vocab_size)

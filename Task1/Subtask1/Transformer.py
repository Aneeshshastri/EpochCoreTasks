import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
import optax


class RoPE(nnx.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even. Got {head_dim}")
            
        # Compute angles
        inv_freq = 1.0 / (base ** (jnp.arange(0, head_dim, 2) / head_dim))
        positions = jnp.arange(max_seq_len)
        angles = jnp.outer(positions, inv_freq)
        
        # Cache complex exponentials e^(i * theta)
        complex_freqs = jnp.exp(1j * angles)
        self.complex_freqs = nnx.Cache(complex_freqs)

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x=(batch_size, seq_len, n_heads, head_dim)
        """
        seq_len = x.shape[1]
        freqs = self.complex_freqs[...][:seq_len, :]
        freqs = freqs[None, :, None, :]
        x_paired = x.reshape(*x.shape[:-1], -1, 2)
        
        
        x_complex = jax.lax.complex(x_paired[..., 0], x_paired[..., 1])
        rotated_complex = x_complex * freqs
        
        
        rotated_real = jnp.stack([rotated_complex.real, rotated_complex.imag], axis=-1)
        return rotated_real.reshape(x.shape)

class FeedForwardNN(nnx.Module):
    def __init__(self, d_model: int, hidden_dim: int,rngs: nnx.Rngs):
        self.w_gate= nnx.Linear(d_model, hidden_dim, use_bias=False, rngs=rngs)
        self.w_up= nnx.Linear(d_model, hidden_dim, use_bias=False, rngs=rngs)
        self.w_down= nnx.Linear(hidden_dim,d_model, use_bias=False, rngs=rngs)
    def __call__(self, x:jax.Array):
        hidden_state=self.w_up(x)*jax.nn.silu(self.w_gate(x))
        return self.w_down(hidden_state)


class MHSA(nnx.Module):
    def __init__(self, d_model,max_seq_len, d_k=None, n_heads=8,rngs=None):
        
        self.d_model = d_model
        self.n_heads=n_heads
        if d_k is not None: self.d_k=d_k
        else: self.d_k=d_model
        self.head_size=self.d_k//n_heads
        self.max_seq=max_seq_len
        # Linear layers for projecting input X into Q, K, V spaces
        self.w_q = nnx.Linear(d_model, self.d_k, use_bias=False, rngs=rngs)
        self.w_k = nnx.Linear(d_model, self.d_k, use_bias=False, rngs=rngs)
        self.w_v = nnx.Linear(d_model, self.d_k, use_bias=False, rngs=rngs)
        self.w_o = nnx.Linear(self.d_k, d_model, use_bias=False, rngs=rngs)
        self.RoPE=RoPE(head_dim=self.head_size,max_seq_len=max_seq_len)

        self.sown_attn = nnx.Intermediate(None)
        
    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        batch,seq_len ,_ = x.shape
        # 1. Project inputs
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        Q=q.reshape((batch,seq_len,self.n_heads,self.head_size))
        K=k.reshape((batch,seq_len,self.n_heads,self.head_size))
        V=v.reshape((batch,seq_len,self.n_heads,self.head_size))
        #apply RoPE
        Q_R=self.RoPE(Q)
        K_R=self.RoPE(K)

        #Permute matrix for computing Attention tensor
        Q_R = jnp.transpose(Q_R, (0, 2, 1, 3))
        K_R = jnp.transpose(K_R, (0, 2, 1, 3))
        V = jnp.transpose(V, (0, 2, 1, 3))
        
        # Scaled dot-product scores: (batch, n_heads, seq_len, seq_len)
        scale = 1.0 / jnp.sqrt(self.head_size)
        attn_scores = jnp.matmul(Q_R, jnp.transpose(K_R, (0, 1, 3, 2))) * scale
        if mask is not None:
            attn_scores = jnp.where(mask, attn_scores, -1e12)
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        self.sown_attn = nnx.Intermediate(attn_weights)
        # Weighted sum over values: (batch, n_heads, seq_len, head_size)
        context = jnp.matmul(attn_weights, V)
        
        # Permute back to (batch, seq_len, n_heads, head_size) and flatten heads
        context = jnp.transpose(context, (0, 2, 1, 3))
        attn_output = context.reshape((batch, seq_len, self.d_k))
        
        # Final projection
        return self.w_o(attn_output)




class TransformerBlock(nnx.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, rngs: nnx.Rngs):
        self.attn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.mha = MHSA(d_model, n_heads=n_heads, max_seq_len=max_seq_len, rngs=rngs)
        
        self.ffn_norm = nnx.RMSNorm(d_model, rngs=rngs)
        hidden_dim = int((8/3) * d_model) 
        self.ffn = FeedForwardNN(d_model, hidden_dim, rngs)
        

    def __call__(self, x: jax.Array, mask: jax.Array | None = None) -> jax.Array:
        """
        x shape: (batch_size, seq_len, d_model)
        """
        # Phase 1: Pre-Norm Attention Residual Stream
        x_norm1 = self.attn_norm(x)
        attn_out = self.mha(x_norm1, mask)
        #x = x + attn_out  
        x=attn_out
        # Phase 2: Pre-Norm FeedForward Residual Stream
        x_norm2 = self.ffn_norm(x)
        ffn_out = self.ffn(x_norm2)
        x = x + ffn_out  
        return x


class DigitRanker(nnx.Module):
    def __init__(self, d_model: int = 64, n_heads: int = 4, num_layers: int = 2, max_seq_len: int = 10, num_classes: int = 10,use_lape: bool=False, rngs: nnx.Rngs = None):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self.embed = nnx.Linear(1, d_model, rngs=rngs)
        self.pos_embed = nnx.Embed(max_seq_len, d_model, rngs=rngs) if use_lape else None
        self.transformer_blocks = nnx.List([
            TransformerBlock(d_model, n_heads, max_seq_len, rngs) for _ in range(num_layers)
        ])
        self.norm = nnx.RMSNorm(d_model, rngs=rngs)
        self.head = nnx.Linear(d_model, num_classes, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        # x: (batch, seq_len) -- integer values
        # Min-max normalize per sequence
        x_min = x.min(axis=-1, keepdims=True)
        x_max = x.max(axis=-1, keepdims=True)
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        x_norm = x_norm[..., None]  # (batch, seq_len, 1)
        x = self.embed(x_norm)  # (batch, seq_len, d_model)
        if self.pos_embed is not None:
            pos_ids = jnp.arange(self.max_seq_len)[None, :]
            x = x + self.pos_embed(pos_ids)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)  # (batch, seq_len, num_classes)
        return logits

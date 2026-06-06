"""
Training utilities: loss functions, JIT-compiled train / validation steps,
and early stopping.

Three model families are supported:
    1. TransformerModel           → train_step / validation_step
    2. PointerGeneratorTransformer → train_step_pointer_gen / validation_step_pointer_gen
    3. RNNSelfAttentionModel      → train_step_baseline / validation_step_baseline
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def compute_loss(logits, targets, pad_token):
    """Cross-entropy with padding mask.

    Parameters
    ----------
    logits  : (batch, seq_len, vocab_size)
    targets : (batch, seq_len)   — integer labels
    pad_token : int
    """
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits, targets.astype(jnp.int32),
    )
    mask = (targets != pad_token).astype(jnp.float32)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


def compute_nll_loss_from_probs(probs, targets, pad_token):
    """Negative Log-Likelihood with padding mask from probability distribution."""
    # Add epsilon to prevent log(0)
    probs = jnp.clip(probs, 1e-8, 1.0)
    batch_size, seq_len = targets.shape
    b_idx = jnp.arange(batch_size)[:, None]
    s_idx = jnp.arange(seq_len)[None, :]
    
    target_probs = probs[b_idx, s_idx, targets.astype(jnp.int32)]
    loss = -jnp.log(target_probs)
    
    mask = (targets != pad_token).astype(jnp.float32)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


# ---------------------------------------------------------------------------
# Encoder-Decoder Transformer (base, teacher-forced)
# ---------------------------------------------------------------------------

@nnx.jit
def train_step(model, optimizer, batch, pad_token):
    """Train step for ``TransformerModel`` (pure teacher forcing)."""
    X, y = batch["input"], batch["output"]
    target = y[:, 1:]                               # shifted target

    def loss_fn(model):
        logits = model(X, y)
        return compute_loss(logits, target, pad_token), logits

    (loss, logits), grads = nnx.value_and_grad(
        loss_fn, has_aux=True,
    )(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def validation_step(model, batch, pad_token):
    """Validation step for ``TransformerModel``."""
    X, y = batch["input"], batch["output"]
    target = y[:, 1:]
    logits = model(X, y)
    return compute_loss(logits, target, pad_token)


# ---------------------------------------------------------------------------
# Pointer-Generator Transformer
# ---------------------------------------------------------------------------

@nnx.jit
def train_step_pointer_gen(model, optimizer, batch, pad_token):
    """Train step with pointer-generator scheduled sampling.

    Returns total_loss.
    """
    X, y = batch["input"], batch["output"]
    target = y[:, 1:]

    def loss_fn(model):
        probs = model(X, y)
        loss = compute_nll_loss_from_probs(probs, target, pad_token)
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def validation_step_pointer_gen(model, batch, pad_token):
    """Validation step for pointer generator."""
    X, y = batch["input"], batch["output"]
    target = y[:, 1:]
    probs = model(X, y)
    return compute_nll_loss_from_probs(probs, target, pad_token)


# ---------------------------------------------------------------------------
# RNN + Self-Attention Baseline
# ---------------------------------------------------------------------------

@nnx.jit
def train_step_baseline(model, optimizer, batch, pad_token):
    """Train step for ``RNNSelfAttentionModel`` (non-autoregressive)."""
    X, y = batch["input"], batch["output"]

    def loss_fn(model):
        logits = model(X)
        return compute_loss(logits, y, pad_token), logits

    (loss, _), grads = nnx.value_and_grad(
        loss_fn, has_aux=True,
    )(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def validation_step_baseline(model, batch, pad_token):
    """Validation step for ``RNNSelfAttentionModel``."""
    X, y = batch["input"], batch["output"]
    logits = model(X)
    return compute_loss(logits, y, pad_token)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Tracks best validation loss and triggers early stopping."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model) -> bool:
        """Returns True when training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = jax.tree.map(lambda x: x, nnx.state(model))
        else:
            self.counter += 1
        return self.counter >= self.patience

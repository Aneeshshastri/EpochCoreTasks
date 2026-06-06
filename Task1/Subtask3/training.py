"""
Training utilities: loss functions, JIT-compiled train / validation steps,
and early stopping.
"""

import jax
import jax.numpy as jnp
import optax
from flax import nnx


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def compute_loss(logits, targets, pad_token, metadata_mask):
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
    mask = (targets != pad_token).astype(jnp.float32) * metadata_mask[:, 1:]
    return (loss * mask).sum() / (mask.sum() + 1e-8)

# ---------------------------------------------------------------------------
# Encoder-Decoder Transformer (base, teacher-forced)
# ---------------------------------------------------------------------------

@nnx.jit
def train_step(model, optimizer, batch, pad_token):
    """Train step for ``TransformerModel`` (pure teacher forcing)."""
    X= batch[0]
    metadata_mask=batch[1]
    target = X[:,1:]                               # shifted target
    X=X[:,:-1]
    def loss_fn(model):
        logits = model(X)
        return compute_loss(logits, target, pad_token,metadata_mask), logits

    (loss, logits), grads = nnx.value_and_grad(
        loss_fn, has_aux=True,
    )(model)
    optimizer.update(model, grads)
    return loss


@nnx.jit
def validation_step(model, batch, pad_token):
    X= batch[0]
    metadata_mask=batch[1]
    target = X[:,1:]
    X=X[:,:-1]
    logits = model(X)
    return compute_loss(logits, target, pad_token,metadata_mask)

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

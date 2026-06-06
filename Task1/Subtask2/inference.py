"""
Inference utilities: greedy autoregressive decoding and evaluation.

The ``greedy_decode`` function uses a fixed-size buffer so the decode
call always receives the same shapes, avoiding repeated JIT compilation.
"""

import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Greedy Autoregressive Decoding
# ---------------------------------------------------------------------------

def greedy_decode(model, x, start_token, end_token, pad_token,
                  max_decode_len):
    """
    Autoregressive greedy decoding for an encoder-decoder model.

    Works with both ``TransformerModel`` and ``PointerGeneratorTransformer``
    (the latter exposes the same ``encode`` / ``decode`` API via delegation).

    Parameters
    ----------
    model : TransformerModel | PointerGeneratorTransformer
    x : jax.Array, (batch, enc_len)
        Encoder input token IDs.
    start_token, end_token, pad_token : int
    max_decode_len : int
        Maximum number of tokens to generate (excluding START).

    Returns
    -------
    output : jax.Array, (batch, max_decode_len)
        Generated token IDs (START excluded, PAD-filled after END).
    """
    batch_size = x.shape[0]

    # Encode once
    enc_h, enc_mask = model.encode(x)

    # Fixed-size decoder buffer — avoids shape changes between steps
    # so JAX compiles the decode call only once.
    # Buffer layout:  [START, slot_0, slot_1, ..., slot_{max-1}]
    dec_buffer = jnp.full(
        (batch_size, max_decode_len + 1), pad_token, dtype=jnp.int32,
    )
    dec_buffer = dec_buffer.at[:, 0].set(start_token)
    finished = jnp.zeros(batch_size, dtype=jnp.bool_)
    
    for step in range(max_decode_len):
        # Pass 'x' to handle the pointer-generator scatter requirements
        if hasattr(model, 'gate_fc'): 
            probs_or_logits, _, _ = model.decode(dec_buffer, enc_h, enc_mask, x)
        else:
            probs_or_logits, _, _ = model.decode(dec_buffer, enc_h, enc_mask)
            
        next_dist = probs_or_logits[:, step, :]

        # Prevent generating PAD or START during inference
        # (Works for both probabilities and logits)
        next_dist = next_dist.at[:, pad_token].set(-1e12)
        next_dist = next_dist.at[:, start_token].set(-1e12)

        next_token = jnp.argmax(next_dist, axis=-1)
        next_token = jnp.where(finished, pad_token, next_token)

        dec_buffer = dec_buffer.at[:, step + 1].set(next_token)
        finished = finished | (next_token == end_token)

        if bool(jnp.all(finished)):
            break
    # Output = everything after START
    return dec_buffer[:, 1:]


# ---------------------------------------------------------------------------
# Autoregressive Evaluation
# ---------------------------------------------------------------------------

def evaluate_autoregressive(model, dataloader, start_token, end_token,
                            pad_token, max_decode_len):
    """
    Evaluate an encoder-decoder model using greedy autoregressive
    decoding (no teacher forcing).

    Returns (token_accuracy, exact_sequence_accuracy).
    """
    total_tokens = 0
    correct_tokens = 0
    total_seqs = 0
    correct_seqs = 0

    for batch in dataloader:
        X = batch["input"]
        y = batch["output"]
        target = y[:, 1:]                            # ground truth w/o START

        generated = greedy_decode(
            model, X, start_token, end_token, pad_token, max_decode_len,
        )

        # Align lengths (generated might be shorter if max_decode_len <
        # target length, though normally they match).
        gen_len = generated.shape[1]
        tgt_len = target.shape[1]
        if gen_len < tgt_len:
            generated = jnp.pad(
                generated,
                ((0, 0), (0, tgt_len - gen_len)),
                constant_values=pad_token,
            )
        elif gen_len > tgt_len:
            generated = generated[:, :tgt_len]

        mask = (target != pad_token)

        correct_tokens += int(jnp.sum((generated == target) & mask))
        total_tokens += int(jnp.sum(mask))

        seq_match = jnp.all((generated == target) | ~mask, axis=1)
        correct_seqs += int(jnp.sum(seq_match))
        total_seqs += len(y)

    token_acc = correct_tokens / (total_tokens + 1e-8)
    seq_acc = correct_seqs / (total_seqs + 1e-8)
    return float(token_acc), float(seq_acc)


# ---------------------------------------------------------------------------
# Baseline (non-autoregressive) Evaluation
# ---------------------------------------------------------------------------

def evaluate_baseline(model, dataloader, pad_token):
    """
    Evaluate ``RNNSelfAttentionModel`` (non-autoregressive).

    The model directly predicts at every position without a decoding loop.
    Returns (token_accuracy, exact_sequence_accuracy).
    """
    total_tokens = 0
    correct_tokens = 0
    total_seqs = 0
    correct_seqs = 0

    for batch in dataloader:
        X = batch["input"]
        y = batch["output"]

        logits = model(X)
        preds = jnp.argmax(logits, axis=-1)

        mask = (y != pad_token)
        correct_tokens += int(jnp.sum((preds == y) & mask))
        total_tokens += int(jnp.sum(mask))

        seq_match = jnp.all((preds == y) | ~mask, axis=1)
        correct_seqs += int(jnp.sum(seq_match))
        total_seqs += len(y)

    token_acc = correct_tokens / (total_tokens + 1e-8)
    seq_acc = correct_seqs / (total_seqs + 1e-8)
    return float(token_acc), float(seq_acc)

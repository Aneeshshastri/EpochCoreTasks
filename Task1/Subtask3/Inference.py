import jax
import jax.numpy as jnp
import numpy as np

@jax.jit(static_argnames="top_k")
def generate_step(model, padded_seq, current_idx, rng_key, temperature=0.8, top_k=40, rep_penalty=1.2):
    logits = model(padded_seq) 
    
    next_token_logits = logits[0, current_idx - 1, :]
    
    context_tokens = padded_seq[0]
    context_logits = next_token_logits[context_tokens]
    
    # Repetition Penalty
    penalized_logits = jnp.where(
        context_logits < 0,
        context_logits * rep_penalty,
        context_logits / rep_penalty
    )
    
    next_token_logits = next_token_logits.at[context_tokens].set(penalized_logits)
    
    # Temperature & Top-K
    scaled_logits = next_token_logits / temperature
    top_k_vals, top_k_indices = jax.lax.top_k(scaled_logits, top_k)
    
    filtered_logits = jnp.full_like(scaled_logits, -jnp.inf)
    filtered_logits = filtered_logits.at[top_k_indices].set(top_k_vals)
    
    return jax.random.categorical(rng_key, filtered_logits)


def generate_poem(model, tokenizer, metadata,content="", max_len=256, max_new_tokens=500, temperature=0.85, top_k=40, rep_penalty=1.2, seed=42):
    model.eval()
    
    prompt_text = (
        f"Author: {metadata['author']}\n"
        f"Era: {metadata['era']}\n"
        f"Type: {metadata['type']}\n"
        f"Title: {metadata['title']}\n\n"
    )
    prompt_text+=content
    tokens = tokenizer.encode(prompt_text)
    prompt_length = len(tokens)
    
    vocab_size = tokenizer.get_vocab_size()
    eos_id = vocab_size
    pad_id = vocab_size + 1
    
    rng_key = jax.random.PRNGKey(seed)
    
    for _ in range(max_new_tokens):
        rng_key, subkey = jax.random.split(rng_key)
        
        if len(tokens) <= max_len:
            context = tokens
            curr_len = len(tokens)
        else:
            context = tokens[-max_len:]
            curr_len = max_len
            
        seq = np.full((1, max_len), pad_id, dtype=np.int32)
        seq[0, :curr_len] = context
        
        seq_device = jax.device_put(seq)
        curr_len_device = jnp.array(curr_len, dtype=jnp.int32)
        
        next_token = generate_step(
            model, 
            seq_device, 
            current_idx=curr_len_device, 
            rng_key=subkey, 
            temperature=temperature, 
            top_k=top_k,
            rep_penalty=rep_penalty
        )
        
        next_token_int = int(next_token)
        if next_token_int == eos_id:
            break
            
        tokens.append(next_token_int)
        
    return tokenizer.decode(tokens[prompt_length:])
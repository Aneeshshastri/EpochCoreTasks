import numpy as np
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view
from typing import List

class NGramModel:
    def __init__(self, n: int, vocab_size: int, pad_id: int, eos_id: int):
        self.n = n
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.eos_id = eos_id
        
        self.log_probs = {}
        self.fallback_target_probs = {}
        self.fallback_context_prob = np.log(1.0 / vocab_size)

    def fit(self, X_train: np.ndarray):
        counts = {}
        
        windows = sliding_window_view(X_train, window_shape=self.n + 1, axis=1).reshape(-1, self.n + 1)
        valid_windows = windows[~np.any(windows == self.pad_id, axis=1)]
        
        window_counts = Counter(map(tuple, valid_windows.tolist()))
        
        for window, count in window_counts.items():
            context = window[:-1]
            target = window[-1]
            
            if context not in counts:
                counts[context] = {}
            counts[context][target] = count
            
        for context, targets in counts.items():
            total = sum(targets.values())
            
            self.log_probs[context] = {
                tgt: np.log((cnt + 1) / (total + self.vocab_size))
                for tgt, cnt in targets.items()
            }
            self.fallback_target_probs[context] = np.log(1.0 / (total + self.vocab_size))

    def generate(self, prompt_tokens: List[int], max_new_tokens: int = 300) -> List[int]:
        """Autoregressively samples using the pre-computed log probabilities."""
        generated = list(prompt_tokens)
        
        for _ in range(max_new_tokens):
            if len(generated) < self.n:
                break
                
            context = tuple(generated[-self.n:])
            if context not in self.log_probs:
                break # Dead end
                
            possible_transitions = self.log_probs[context]
            next_tokens = list(possible_transitions.keys())
            
            weights = np.exp(list(possible_transitions.values()))
            
            next_token = np.random.choice(next_tokens, p=weights/np.sum(weights))
            
            if next_token == self.eos_id:
                break
            generated.append(int(next_token))
            
        return generated



def evaluate_ngram_perplexity(model: NGramModel, X_val: np.ndarray) -> float:
    
    windows = sliding_window_view(X_val, window_shape=model.n + 1, axis=1).reshape(-1, model.n + 1)
    valid_windows = windows[~np.any(windows == model.pad_id, axis=1)]
    
    if len(valid_windows) == 0:
        return float('inf')

    log_prob_sum = 0.0
    
    for window in valid_windows:
        context = tuple(window[:-1])
        target = window[-1]
        
        if context in model.log_probs:
            log_prob_sum += model.log_probs[context].get(target, model.fallback_target_probs[context])
        else:
            log_prob_sum += model.fallback_context_prob
            
    avg_neg_log_prob = -log_prob_sum / len(valid_windows)
    return float(np.exp(avg_neg_log_prob))
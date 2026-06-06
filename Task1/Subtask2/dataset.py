"""
Dataset and DataLoader utilities for the code-repair task.

Every sequence has the structure:
    [START] + content_tokens[:max_content_len] + [END] + [PAD] * remaining

Total length per sequence = max_content_len + 2  (the "full sequence length").
"""

import numpy as np
import grain.python as grain


class TextSource(grain.RandomAccessDataSource):
    """
    Tokenises buggy/fixed code pairs and pads each to a fixed length
    with explicit START, END, and PAD tokens.

    Parameters
    ----------
    bugged : iterable of str
        Buggy source-code strings.
    fixed : iterable of str
        Corresponding fixed source-code strings.
    tokenizer : BPETokenizer
        Trained BPE tokenizer with .encode(text) -> list[int].
    max_content_len : int
        Maximum number of *content* tokens (before adding START/END).
    pad_token, start_token, end_token : int
        Special token IDs.
    """

    def __init__(self, bugged, fixed, tokenizer, max_content_len,
                 pad_token, start_token, end_token):
        self.X = []
        self.y = []
        full_seq_len = max_content_len + 2  # +2 for START and END

        for buggy_text, fixed_text in zip(bugged, fixed):
            input_tokens = tokenizer.encode(buggy_text)
            target_tokens = tokenizer.encode(fixed_text)

            # Truncate content to max_content_len
            input_tokens = input_tokens[:max_content_len]
            target_tokens = target_tokens[:max_content_len]

            # Build: [START] + content + [END] + [PAD] * remaining
            input_seq = [start_token] + input_tokens + [end_token]
            target_seq = [start_token] + target_tokens + [end_token]

            input_seq += [pad_token] * (full_seq_len - len(input_seq))
            target_seq += [pad_token] * (full_seq_len - len(target_seq))

            self.X.append(input_seq)
            self.y.append(target_seq)

        self.X = np.array(self.X, dtype=np.int32)
        self.y = np.array(self.y, dtype=np.int32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {"input": self.X[idx], "output": self.y[idx]}


def make_loader(bugged, fixed, tokenizer, max_content_len,
                pad_token, start_token, end_token,
                batch_size=4, training=True, seed=14):
    """Create a grain DataLoader for code-repair data."""
    source = TextSource(
        bugged, fixed, tokenizer, max_content_len,
        pad_token, start_token, end_token,
    )
    transforms = [grain.Batch(batch_size, drop_remainder=True)]

    return grain.DataLoader(
        data_source=source,
        sampler=grain.IndexSampler(
            num_records=len(source),
            shuffle=training,
            seed=seed,
            num_epochs=1,
            shard_options=grain.NoSharding(),
        ),
        worker_count=6,
        worker_buffer_size=2,
        operations=transforms,
    )

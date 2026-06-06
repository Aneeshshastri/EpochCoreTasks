"""
Dataset and DataLoader utilities for the code-repair task.

Every sequence has the structure:
    [START] + content_tokens[:max_content_len] + [END] + [PAD] * remaining

Total length per sequence = max_content_len + 2  (the "full sequence length").
"""

import numpy as np
import grain.python as grain

class TextSource(grain.RandomAccessDataSource):
    def __init__(self, text, tokenizer, max_len=256, stride=128):
        self.token_ids = np.array(tokenizer.encode(text), dtype=np.int32)
        self.max_len = max_len
        self.stride = stride
        self.num_samples = max(0, (len(self.token_ids) - max_len) // stride)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.stride
        input_chunk = self.token_ids[start : start + self.max_len]
        target_chunk = self.token_ids[start + 1 : start + self.max_len + 1]
        
        return {"input": input_chunk, "output": target_chunk}
        
def make_loader(text,tokenizer=Tokenizer,max_len=256,stride=128, batch_size=4, training=True, seed=RANDOM_SEED):
    source = TextSource(text,tokenizer,max_len,stride)
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
        worker_count=0,
        worker_buffer_size=2,
        operations=transforms,
    )
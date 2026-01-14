"""
Train a BPE tokenizer on the TinyStories dataset.

Usage:
    cd scripts
    export PYTHONPATH=$PYTHONPATH:..
    scalene run train_bpe_tinystories.py
"""

from cs336_basics import train_bpe
import time
import json

data_path = '../data/tinystories/TinyStories-train.txt'
output_path = '../outputs'
max_vocab_size = 10000
special_tokens = ['<|endoftext|>']

start = time.time()
vocab, merges = train_bpe(data_path, max_vocab_size, special_tokens)
elapsed = time.time() - start

print(f"Training finished in {elapsed:.2f} seconds")
print(f"Vocabulary size: {len(vocab)}")
print(f"Number of merges: {len(merges)}")

# Find the longest vocab item
longest_token_id, longest_token = max(vocab.items(), key=lambda x: len(x[1]))
print(f"Longest token id: {longest_token_id}, hex: {longest_token.hex()}")

# Save on disk
with open(f"{output_path}/vocab.json", "w") as f:
    json.dump({k: v.hex() for k, v in vocab.items()}, f, indent=2)

with open(f"{output_path}/merges.json", "w") as f:
    json.dump([(a.hex(), b.hex()) for a, b in merges], f, indent=2)

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

with open(f"{output_path}/vocab.json", "w") as f:
    json.dump({k: v.hex() for k, v in vocab.items()}, f, indent=2)

with open(f"{output_path}/merges.json", "w") as f:
    json.dump([(a.hex(), b.hex()) for a, b in merges], f, indent=2)

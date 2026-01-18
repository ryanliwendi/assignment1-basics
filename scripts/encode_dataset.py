"""
Encodes a dataset using the specified vocabulary.

Usage:
    cd scripts
    export PYTHONPATH=$PYTHONPATH:..
    python encode_dataset.py [ts-train|ts-val|owt-train|owt-val]
"""

from cs336_basics import Tokenizer
import numpy as np
import sys


configs = {
    "ts-train": (
        "../data/tinystories/TinyStories-train.txt",
        "../outputs/tinystories/vocab.json",
        "../outputs/tinystories/merges.json",
        "../outputs/tinystories/train_encoded.npy",
    ),
    "ts-val": (
        "../data/tinystories/TinyStories-valid.txt",
        "../outputs/tinystories/vocab.json",
        "../outputs/tinystories/merges.json",
        "../outputs/tinystories/valid_encoded.npy",
    ),
    "owt-train": (
        "../data/openwebtext/owt_train.txt",
        "../outputs/openwebtext/vocab.json",
        "../outputs/openwebtext/merges.json",
        "../outputs/openwebtext/train_encoded.npy",
    ),
    "owt-val": (
        "../data/openwebtext/owt_valid.txt",
        "../outputs/openwebtext/vocab.json",
        "../outputs/openwebtext/merges.json",
        "../outputs/openwebtext/valid_encoded.npy",
    )
}

def tokenize_data(data_path, vocab_path, merges_path, special_tokens, output_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        t = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
        results = list(t.encode_iterable(f))
        arr = np.asarray(results, dtype=np.uint16)
        np.save(output_path, arr)
        print(f"Saved {arr.shape[0]} tokens to {output_path}")


if len(sys.argv) != 2:
    raise ValueError("Usage: python encode_dataset.py [ts-train|ts-val|owt-train|owt-val]")

mode = sys.argv[1]
if mode not in configs:
    raise ValueError(f"Unknown mode: {mode}")

data_path, vocab_path, merges_path, output_path = configs[mode]
special_tokens = ["<|endoftext|>"]

tokenize_data(
    data_path = data_path,
    vocab_path = vocab_path,
    merges_path = merges_path,
    special_tokens = special_tokens,
    output_path = output_path
)

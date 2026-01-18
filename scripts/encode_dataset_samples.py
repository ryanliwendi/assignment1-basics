import os
from cs336_basics import Tokenizer


def analyze_tokenizer(name, data_path, vocab_path, merges_path, special_tokens):

    with open(data_path, 'r', encoding='utf-8') as f:
        # Count number of bytes
        f.seek(0, os.SEEK_END)
        byte_count = f.tell()
        f.seek(0)

        text_data = f.read()

    t = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    tokens = t.encode(text_data)
    token_count = len(tokens)

    compression_ratio = byte_count / token_count

    print(f"--- Results for {name} ---")
    print(f"Total Bytes:       {byte_count:,}")
    print(f"Total Tokens:      {token_count:,}")
    print(f"Compression Ratio: {compression_ratio:.3f} bytes/token")
    print(f"First 10 tokens:   {tokens[:10]}\n")

special_tokens = ['<|endoftext|>']

# Run for TinyStories
analyze_tokenizer(
    name="TinyStories",
    data_path="../data/samples/tinystories_sample.txt",
    vocab_path="../outputs/tinystories/vocab.json",
    merges_path="../outputs/tinystories/merges.json",
    special_tokens=special_tokens
)

# Run for OpenWebText
analyze_tokenizer(
    name="OpenWebText",
    data_path="../data/samples/owt_sample.txt",
    vocab_path="../outputs/openwebtext/vocab.json",
    merges_path="../outputs/openwebtext/merges.json",
    special_tokens=special_tokens
)
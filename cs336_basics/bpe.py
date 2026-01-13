import regex as re

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]]
]:
    # Step 1: Initialize Vocabulary
    vocab: dict[int, bytes] = {x : bytes([x]) for x in range(256)}
    vocab_values = set(vocab.values())
    next_id = 256
    for token in special_tokens:
        token_encoded = token.encode('utf-8')
        if token_encoded not in vocab_values:
            vocab[next_id] = token_encoded
            vocab_values.add(token_encoded)
            next_id += 1

    # Step 2: Read Corpus
    with open(input_path, mode = 'r', encoding = 'utf-8') as f:
        text = f.read()

    # Step 3: Chunking and Removing Special Tokens
    tokens_escaped = [re.escape(token) for token in special_tokens]
    pattern = '|'.join(tokens_escaped)
    chunks: list[str] = re.split(pattern = pattern, string = text)








    


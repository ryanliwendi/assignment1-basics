import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def to_tuples(s: str) -> tuple[bytes, ...]:
    encoded = s.encode(encoding = 'utf-8')
    return tuple(bytes([b]) for b in encoded)

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
    if len(special_tokens) == 0:
        tokens_escaped = [re.escape(token) for token in special_tokens]
        pattern = '|'.join(tokens_escaped)
        chunks: list[str] = re.split(pattern = pattern, string = text)
    else:
        chunks = [text]

    # Step 4: Pre-tokenization
    pre_tokens : dict[tuple[bytes, ...], int] = {}
    for chunk in chunks:
        for pre_token in re.finditer(pattern = PAT, string = chunk):
            token_text = to_tuples(pre_token.group(0))   # Obtain the matched substring from the match object
            if token_text in pre_tokens:
                pre_tokens[token_text] += 1
            else:
                pre_tokens[token_text] = 1

    # Step 5: BPE Merging
    while next_id < vocab_size:
        pairs : dict[tuple[bytes, bytes], int] = {}
        for pre_token, count in pre_tokens.items():
            for idx in range(len(pre_token) - 1):
                pair = (pre_token[idx], pre_token[idx + 1])
                if pair in pairs:
                    pairs[pair] += count
                else:
                    pairs[pair] = count
        max_cnt = max(pairs.values())
        max_pairs = [p for p in pairs if pairs[p] == max_cnt]
        max_pair = max(max_pairs)

        new_token = max_pair[0] + max_pair[1]
        vocab[next_id] = new_token
        next_id += 1

        for token in pre_tokens:










    


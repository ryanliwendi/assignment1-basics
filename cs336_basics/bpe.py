import regex as re
from collections import defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[
    dict[int, bytes],
    list[tuple[bytes, bytes]]
]:
    """
    Train a byte-level Byte Pair Encoding (BPE) tokenizer on a text corpus.

    Args:
        input_path (str): Path to a UTF-8 encoded text file containing
            training data for the tokenizer.
        vocab_size (int): Maximum size of the final vocabulary, including
            the initial byte vocabulary, learned merge tokens, and special
            tokens.
        special_tokens (list[str]): List of special token strings to include
            in the vocabulary. These tokens are never split or merged and
            act as boundaries during training.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                A dictionary mapping integer token IDs to their
                corresponding byte sequences.
            merges:
                A list of BPE merge operations in the order they were
                learned. Each merge is represented as a tuple
                `(left, right)` of byte sequences indicating that `left`
                and `right` were merged.
    """
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
    if len(special_tokens) != 0:
        tokens_escaped = [re.escape(token) for token in special_tokens]
        pattern = '|'.join(tokens_escaped)
        chunks: list[str] = re.split(pattern = pattern, string = text)
    else:
        chunks = [text]

    # Step 4: Pre-tokenization
    pre_tokens : dict[tuple[bytes, ...], int] = defaultdict(int)
    for chunk in chunks:
        if not chunk:
            continue
        for pre_token in re.finditer(pattern = PAT, string = chunk):
            token_text = pre_token.group(0)   # Obtain the matched substring from the match object
            token_tuples = to_tuples(token_text)
            pre_tokens[token_tuples] += 1

    # Step 5: BPE Merging
    merges : list[tuple[bytes, bytes]] = []
    while next_id < vocab_size:
        pairs : dict[tuple[bytes, bytes], int] = defaultdict(int)
        for pre_token, count in pre_tokens.items():
            for idx in range(len(pre_token) - 1):
                pair = (pre_token[idx], pre_token[idx + 1])
                pairs[pair] += count
        if not pairs:
            break
        max_cnt = max(pairs.values())
        max_pairs = [p for p in pairs if pairs[p] == max_cnt]
        max_pair = max(max_pairs)

        new_vocab = max_pair[0] + max_pair[1]
        merges.append((max_pair[0], max_pair[1]))
        vocab[next_id] = new_vocab
        next_id += 1

        new_pre_tokens: dict[tuple[bytes, ...], int] = defaultdict(int)
        for token, count in pre_tokens.items():
            out: list[bytes] = []
            cur_pos = 0
            while cur_pos < len(token):
                if cur_pos + 1 < len(token) and (token[cur_pos], token[cur_pos + 1]) == max_pair:
                    out.append(new_vocab)
                    cur_pos += 2
                else:
                    out.append(token[cur_pos])
                    cur_pos += 1
            updated_token = tuple(out)
            new_pre_tokens[updated_token] += count
        pre_tokens = new_pre_tokens
    return vocab, merges


def to_tuples(s: str) -> tuple[bytes, ...]:
    """
    Converts a Unicode string into a tuple of single-byte symbols.

    Args:
        s (str): Input Unicode string (pre-token).

    Returns:
        tuple[bytes, ...]: A variable-length tuple where each element is a
        single-byte `bytes` object representing the UTF-8 encoding of `s`.
    """
    encoded = s.encode(encoding = 'utf-8')
    return tuple(bytes([b]) for b in encoded)

import os
from typing import DefaultDict

import regex as re
from collections import defaultdict


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


# noinspection SpellCheckingInspection
def train_bpe(
    input_path: str | os.PathLike,
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
    with open(input_path, mode='r', encoding='utf-8') as f:
        text = f.read()

    # Step 3: Chunking and Removing Special Tokens
    if len(special_tokens) != 0:
        pattern = '|'.join(re.escape(token) for token in special_tokens)
        chunks: list[str] = re.split(pattern, text)
    else:
        chunks = [text]

    # Step 4: Pre-tokenization
    pre_tokens: dict[tuple[bytes, ...], int] = defaultdict(int)
    for chunk in chunks:
        if not chunk:
            continue
        for pre_token in re.finditer(PAT, chunk):
            token_text = pre_token.group(0)   # Obtain the matched substring from the match object
            token_tuples = tuple(bytes([b]) for b in token_text.encode('utf-8'))
            pre_tokens[token_tuples] += 1

    # Step 5: Build Initial Pair Counts
    pair_freq: dict[tuple[bytes, bytes], int] = defaultdict(int)
    # pair_loc tracks the set of pre-tokens each pair is located in
    pair_loc: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for pre_token, count in pre_tokens.items():
        for idx in range(len(pre_token) - 1):
            pair = (pre_token[idx], pre_token[idx + 1])
            pair_freq[pair] += count
            pair_loc[pair].add(pre_token)

    # Step 6: Incremental BPE Merging
    merges: list[tuple[bytes, bytes]] = []
    while next_id < vocab_size:
        max_cnt = max(pair_freq.values())
        max_pairs = [p for p in pair_freq if pair_freq[p] == max_cnt]
        max_pair = max(max_pairs)
        merges.append(max_pair)

        new_vocab = max_pair[0] + max_pair[1]
        vocab[next_id] = new_vocab
        next_id += 1

        affected_tokens = pair_loc[max_pair]

        # Remove old pair information
        pair_loc.pop(max_pair)
        pair_freq.pop(max_pair)

        new_pretoken_freq: dict[tuple[bytes, ...], int] = DefaultDict(int)  # Track all merged pre-tokens and their frequencies
        for token in affected_tokens:
            count = pre_tokens[token]
            pre_tokens.pop(token)

            new_token: list[bytes] = []
            cur_pos = 0
            while cur_pos < len(token):
                if cur_pos + 1 < len(token) and (token[cur_pos], token[cur_pos + 1]) == max_pair:
                    new_token.append(new_vocab)
                    cur_pos += 2
                else:
                    new_token.append(token[cur_pos])
                    cur_pos += 1
            new_pretoken_freq[tuple(new_token)] = count

            for j in range(len(token) - 1):
                old_pair = (token[j], token[j + 1])
                pair_freq[old_pair] -= count
                pair_loc[old_pair].discard(token)

        # Add new pre-tokens information
        for token, count in new_pretoken_freq.items():
            pre_tokens[token] += count
            for i in range(len(token) - 1):
                new_pair = (token[i], token[i + 1])
                pair_freq[new_pair] += count
                pair_loc[new_pair].add(token)

    return vocab, merges

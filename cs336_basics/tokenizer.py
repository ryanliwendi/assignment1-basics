import json
import regex as re
from typing import Iterable, Iterator


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            self.special_tokens = []
        else:
            self.special_tokens = special_tokens

        # Maintain a set of special tokens for faster lookup
        self.special_tokens_set : set[str] = set(self.special_tokens)

        # Maintain a vocabulary mapping bytes to vocab_ids
        self.inverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}

        for token in self.special_tokens:
            token_encoded = token.encode('utf-8')
            if token_encoded not in self.inverse_vocab:
                next_id = max(self.vocab.keys()) + 1
                self.vocab[next_id] = token_encoded
                self.inverse_vocab[token_encoded] = next_id

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: None | list[str] = None
    ) -> 'Tokenizer':
        """
        Constructs a Tokenizer instance from serialized vocabulary and merge files.

        Acts as an alternative constructor to load a pre-trained BPE model.
        """
        with open(vocab_filepath, mode='r', encoding='utf-8') as f:
            raw_vocab: dict[str, str] = json.load(f)
            vocab = {int(k) : bytes.fromhex(v) for k, v in raw_vocab.items()}

        with open(merges_filepath, mode='r', encoding='utf-8') as f:
            raw_merges : list[list[str]] = json.load(f)
            merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        # Step 1: Chunk the string by special tokens
        if self.special_tokens:
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped = [re.escape(token) for token in sorted_special_tokens]
            pattern = f"({'|'.join(escaped)})"  # Add parentheses to include special tokens
            chunks : list[str] = re.split(pattern=pattern, string=text)
        else:
            chunks = [text]

        # Step 2: Pre-tokenizing, merging, and mapping to vocab_ids
        result : list[int] = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens_set:
                vocab_id = self.inverse_vocab[chunk.encode('utf-8')]
                result.append(vocab_id)
            else:
                for pre_token in re.finditer(pattern=PAT, string=chunk):
                    token_text = pre_token.group(0)
                    token_list = [bytes([b]) for b in token_text.encode(encoding='utf-8')]
                    for a, b in self.merges:
                        if a not in token_list or b not in token_list:  # For efficiency
                            continue
                        out: list[bytes] = []
                        idx = 0
                        while idx < len(token_list):
                            if idx + 1 < len(token_list) and (token_list[idx], token_list[idx + 1]) == (a, b):
                                new_token = a + b
                                out.append(new_token)
                                idx += 2
                            else:
                                out.append(token_list[idx])
                                idx += 1
                        token_list = out
                    for element in token_list:
                        vocab_id = self.inverse_vocab[element]
                        result.append(vocab_id)
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            encoded_text: list[int] = self.encode(chunk)
            # Yield from yields each element in encoded_text one by one
            yield from encoded_text

    def decode(self, ids: list[int]) -> str:
        merged_bytes = b"".join(self.vocab[id] for id in ids)
        return merged_bytes.decode(encoding='utf-8', errors='replace')


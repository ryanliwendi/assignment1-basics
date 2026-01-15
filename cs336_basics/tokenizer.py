import json

class Tokenizer:
    def __init__(
        self,
        vocab : dict[int, bytes],
        merges : list[tuple[bytes, bytes]],
        special_tokens : list[str] = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(
        cls,
        vocab_filepath : str,
        merges_filepath : str,
        special_tokens: None | list[str] = None
    ) -> 'Tokenizer':
        with open(vocab_filepath, mode = 'r', encoding = 'utf-8') as f:
            raw_vocab : dict[str, str] = json.load(f)
            vocab = {int(k) : bytes.fromhex(v) for k, v in raw_vocab.items()}

        with open(merges_filepath, mode = 'r', encoding = 'utf-8') as f:
            raw_merges : list[list[str]] = json.load(f)
            merges = [(bytes.fromhex(a), bytes.fromhex(b)) for a, b in raw_merges]

        return cls(vocab, merges, special_tokens)








import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .bpe import train_bpe
from .tokenizer import Tokenizer
from .layers import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding
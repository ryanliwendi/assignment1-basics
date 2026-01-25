import importlib.metadata

__version__ = importlib.metadata.version("cs336_basics")

from .bpe import train_bpe
from .tokenizer import Tokenizer
from .model import Linear, Embedding, RMSNorm, SwiGLU, RotaryPositionalEmbedding
from .model import softmax, scaled_dot_product_attention
from .model import MultiheadSelfAttention, TransformerBlock, TransformerLM
from .train import cross_entropy, AdamW
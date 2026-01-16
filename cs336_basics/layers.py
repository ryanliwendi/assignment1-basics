import torch
import torch.nn as nn
from einops import einsum
import math
from torch import Tensor
from jaxtyping import Float


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()
        self.W: Float[Tensor, "d_in d_out"] = nn.Parameter(
            torch.empty(
                (in_features, out_features),
                device=device,
                dtype=dtype
            )
        )
        std = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.W, "... d_in, d_in d_out -> ... d_out")

    def extra_repr(self) -> str:
        return f"d_in = {self.W.shape[0]}, d_out = {self.W.shape[1]}"


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.W: Float[Tensor, "vocab d_model"] = nn.Parameter(
            torch.empty(
                (num_embeddings, embedding_dim),
                device=device,
                dtype=dtype
            )
        )
        std = 1
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3, b=3)

    def forward(self, token_ids: Float[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.W[token_ids]

    def extra_repr(self) -> str:
        return f"vocab_size = {self.W.shape[0]}, hidden_dimension = {self.W.shape[1]}"
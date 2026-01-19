import torch
import torch.nn as nn
from einops import einsum, reduce
import math
from torch import Tensor
from jaxtyping import Float, Int


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
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


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        squared_avg = reduce(x ** 2, "... d_model -> ... 1", 'mean')
        rms : Float[Tensor, "... 1"] = torch.sqrt(squared_avg + self.eps)

        # Note: In broadcasting, adding dimensions to the left is automatic,
        # while adding dimensions to the right requires explicitly providing a
        # dimension of size 1
        result = x * self.g / rms
        return result.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None
    ):
        super().__init__()
        if d_ff is None:
            calculated_d_ff = d_model * 8 // 3
            # Round up the dimensionality to a multiple of 64
            self.d_ff = 64 * ((calculated_d_ff + 63) // 64)
        else:
            self.d_ff = d_ff

        self.W1 = Linear(in_features=d_model, out_features=self.d_ff, device=device,dtype=dtype)
        self.W2 = Linear(in_features=self.d_ff, out_features=d_model, device=device, dtype=dtype)
        self.W3 = Linear(in_features=d_model, out_features=self.d_ff, device=device,dtype=dtype)

    def SiLU(self, x: Float[Tensor, "... d_ff"]) -> Float[Tensor, "... d_ff"]:
        return x * torch.sigmoid(x)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        activation = self.SiLU(self.W1(x))
        gate = self.W3(x)
        return self.W2(activation * gate)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None
    ):
        super().__init__()

        assert d_k % 2 == 0
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        buffer_cos = torch.empty((self.max_seq_len, self.d_k // 2), device=device)
        buffer_sin = torch.empty((self.max_seq_len, self.d_k // 2), device=device)
        for i in range(self.max_seq_len):
            for k in range(self.d_k // 2):
                angle = i / (theta ** (2 * k / self.d_k))
                buffer_cos[i, k] = math.cos(angle)
                buffer_sin[i, k] = math.sin(angle)
        self.register_buffer('cos', buffer_cos, persistent=False)
        self.register_buffer('sin', buffer_sin, persistent=False)

    def forward(
        self,
        x: Float[Tensor, "... seq_len d_k"],
        token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        x_even: Float[Tensor, "... seq_len d_k // 2"] = x[..., 0::2]  # Slice based on last dimension
        x_odd: Float[Tensor, "... seq_len d_k // 2"] = x[..., 1::2]

        sin_values: Float[Tensor, "... seq_len d_k // 2"] = self.sin[token_positions]
        cos_values: Float[Tensor, "... seq_len d_k // 2"] = self.cos[token_positions]

        # Apply rotations
        even_rot = cos_values * x_even - sin_values * x_odd
        odd_rot = sin_values * x_even + cos_values * x_odd

        out = torch.empty_like(x)
        out[..., 0::2] = even_rot
        out[..., 1::2] = odd_rot
        return out


def softmax(x: Tensor, dim: int) -> Tensor:
    x_max, _ = torch.max(x, dim=dim, keepdim=True)
    x_shifted = x - x_max

    exp_x = torch.exp(x_shifted)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)

    return exp_x / sum_exp_x





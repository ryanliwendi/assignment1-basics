import torch
import torch.nn as nn
from einops import einsum, reduce, rearrange
import math
from torch import Tensor
from jaxtyping import Float, Int, Bool


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

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        return einsum(x, self.W, "... d_in, d_in d_out -> ... d_out")


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

    def forward(self, token_ids: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        return self.W[token_ids]


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


def scaled_dot_product_attention(
    q: Float[Tensor, "... seq_len d_k"],
    k: Float[Tensor, "... seq_len d_k"],
    v: Float[Tensor, "... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None
) -> Float[Tensor, "... seq_len d_v"]:
    scores = einsum(q, k, "... q d_k, ... k d_k -> ... q k")
    scores /= math.sqrt(q.shape[-1])
    if mask is not None:
        scores = torch.where(mask, scores, float('-inf'))
    scores_normalized = softmax(scores, dim=-1)
    return einsum(scores_normalized, v, "... q k, ... k d_v -> ... q d_v")


class MultiheadSelfAttention(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None
    ):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_model // self.num_heads

        self.W_Q = Linear(self.d_model,self.num_heads * self.d_k)
        self.W_K = Linear(self.d_model,self.num_heads * self.d_k)
        self.W_V = Linear(self.d_model,self.num_heads * self.d_v)

        self.W_O = Linear(self.num_heads * self.d_v, self.d_model)
        self.rope = rope


    def forward(self,
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        *batch, seq_len, d_model = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = rearrange(Q, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        K = rearrange(K, "... seq_len (heads d_k) -> ... heads seq_len d_k", heads=self.num_heads)
        V = rearrange(V, "... seq_len (heads d_v) -> ... heads seq_len d_v", heads=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        outputs = scaled_dot_product_attention(Q, K, V, mask)

        outputs = rearrange(outputs, "... heads seq_len dv -> ... seq_len (heads dv)")
        return self.W_O(outputs)


class TransformerBlock(nn.Module):
    def __init__(self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope: RotaryPositionalEmbedding | None = None
    ):
        super().__init__()

        self.attn = MultiheadSelfAttention(d_model, num_heads, rope)
        self.attn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.ffn_norm = RMSNorm(d_model)

    def forward(
        self,
        x: Float[Tensor, "... d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ):
        x = x + self.attn(self.attn_norm(x), token_positions)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        num_heads: int,
        d_model: int,
        d_ff: int,
        theta: float
    ):
        super().__init__()

        self.rope = RotaryPositionalEmbedding(theta, d_model // num_heads, context_length)

        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, self.rope)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(
        self,
        token_ids: Int[Tensor, "batch_size sequence_length"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ):
        x = self.embedding(token_ids)
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        x = self.final_norm(x)
        return self.lm_head(x)



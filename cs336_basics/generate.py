from cs336_basics import (
    TransformerLM,
    Tokenizer,
    softmax,
)
from jaxtyping import Int, Float
from torch import Tensor
import torch

def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 100,
    top_p: float | None = None,
    temp: float = 1.0,
    eos_token: str = "<|endoftext|>"
) -> str:
    """
    Generate text from a language model.

    Args:
        model: The transformer language model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Input text prompt
        max_tokens: Maximum number of tokens to generate
        temp: Temperature for softmax scaling (lower = more deterministic)
        top_p: Nucleus sampling threshold (None = disabled)
        eos_token: End-of-sequence token string

    Returns:
        Generated text (excluding the prompt)
    """

    model.eval()
    device = next(model.parameters()).device

    cur_tokens: list[int] = tokenizer.encode(prompt)
    generated_tokens = []

    eos_id = tokenizer.encode(eos_token)[0] if eos_token else None

    with torch.no_grad():
        for _ in range(max_tokens):
            input_tensor = torch.tensor([cur_tokens], dtype=torch.long, device=device)

            logits: Float[Tensor, "1 seq_len vocab_size"] = model(input_tensor)
            next_token_logits = logits[0, -1, :]

            if temp > 0 and temp != 1.0:
                next_token_logits /= temp

            probs = softmax(next_token_logits, -1)
            if top_p is not None and top_p < 1.0:
                probs = top_p_sampling(probs, top_p)

            next_token = torch.multinomial(probs, num_samples=1).item()
            generated_tokens.append(next_token)
            cur_tokens.append(next_token)

            if eos_id is not None and next_token == eos_id:
                break

    return tokenizer.decode(generated_tokens)


def top_p_sampling(probs, p):
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    cumulative = 0.0
    cutoff_idx = 0
    for i, prob in enumerate(sorted_probs):
        cumulative += prob.item()
        if cumulative >= p:
            cutoff_idx = i
            break

    sorted_probs[cutoff_idx + 1:] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum()

    result = torch.zeros_like(probs)
    result.scatter_(0, sorted_indices, sorted_probs)

    return result







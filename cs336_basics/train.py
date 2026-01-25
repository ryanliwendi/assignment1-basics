from typing import Optional, Callable
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer
from jaxtyping import Float, Int
from einops import reduce
import numpy as np
import numpy.typing as npt
import math
import typing, os


def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    max_logits: Float[Tensor, "... 1"] = reduce(logits, "... vocab_size -> ... 1", reduction='max')
    stabilized_logits = logits - max_logits

    exp_logits = torch.exp(stabilized_logits)
    logits_sum = reduce(exp_logits, "... vocab_size -> ... 1", reduction='sum')

    target_logits: Float[Tensor, "... 1"] = stabilized_logits.gather(dim=-1, index=targets[..., None])
    loss = torch.log(logits_sum) - target_logits
    return reduce(loss, "... 1 -> ", reduction='mean')


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        beta1, beta2 = betas

        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= beta1 <= 1.0:
            raise ValueError(f"Invalid beta1 value: {beta1}")
        if not 0.0 <= beta2 <= 1.0:
            raise ValueError(f"Invalid beta2 value: {beta2}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid lambda value: {weight_decay}")

        defaults = {'lr': lr, 'beta1': beta1, 'beta2': beta2, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]  # State is a flat dictionary without parameter groups
                t = state.get('t', 1)
                m = state.get('m', torch.zeros_like(p.data))
                v = state.get('v', torch.zeros_like(p.data))

                g = p.grad
                m = beta1 * m + (1 - beta1) * g
                v = beta2 * v + (1 - beta2) * (g ** 2)
                lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                p.data -= lr * weight_decay * p.data

                t += 1
                state['t'] = t
                state['m'] = m
                state['v'] = v
        return loss


def learning_schedule(t, alpha_max, alpha_min, t_warm, t_cos) -> float:
    if t < t_warm:
        return t / t_warm * alpha_max
    elif t_warm <= t <= t_cos:
        return alpha_min + 0.5 * (1 + math.cos(math.pi * (t - t_warm) / (t_cos - t_warm))) * (alpha_max - alpha_min)
    else:
        return alpha_min


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_norm: float, eps = 1e-6) -> None:
    total_norm_sq = 0.0
    grads = []
    for p in parameters:
        if p.grad is not None:
            total_norm_sq += (p.grad ** 2).sum()
            grads.append(p.grad)

    total_norm = math.sqrt(total_norm_sq)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + eps)
        for g in grads:
            g *= scale


def get_batch(x: npt.NDArray[np.uint16], batch_size: int, context_len: int, device: str):
    N = len(x)
    batch_positions = np.random.randint(0, N - context_len, batch_size)
    inputs = np.stack([x[pos: pos + context_len] for pos in batch_positions])
    targets = np.stack([x[pos + 1: pos + context_len + 1] for pos in batch_positions])
    return torch.tensor(inputs, device=device, dtype=torch.long), torch.tensor(targets, device=device, dtype=torch.long)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    obj = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iter': iteration
    }
    torch.save(obj, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    obj = torch.load(src)
    model.load_state_dict(obj['model'])
    optimizer.load_state_dict(obj['optimizer'])
    return obj['iter']
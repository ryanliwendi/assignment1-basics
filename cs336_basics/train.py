from typing import Optional, Callable

import torch
from torch import Tensor
from torch.optim import Optimizer
from jaxtyping import Float, Int
from einops import reduce, einsum


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



import argparse
from argparse import ArgumentParser

import yaml
import numpy as np
import torch

from cs336_basics import (
    TransformerLM,
    AdamW,
    cross_entropy,
    learning_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)


def main():
    args = parse_args()
    config = load_config(args)

    train_data = np.load(config['data']['train_path'], mmap_mode='r')
    val_data = np.load(config['data']['val_path'], mmap_mode='r')

    transformer = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        context_length=config['model']['context_length'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_model=config['model']['d_model'],
        d_ff=config['model']['d_ff'],
        theta=config['model']['theta']
    )

    optimizer = AdamW(
        params=transformer.parameters(),
        lr=1.0,  # Placeholder, later replaced by learning_schedule
        betas=tuple(config['optim']['betas']),
        eps=config['optim']['eps'],
        weight_decay=config['optim']['weight_decay']
    )

    batch_size = config['training']['batch_size']
    max_steps = config['training']['max_steps']
    max_norm = config['training']['max_norm']
    device = config['training']['device']


def evaluate():
    raise NotImplementedError


def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.device is not None:
        config['training']['device'] = args.device

    return config


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--device", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    main()
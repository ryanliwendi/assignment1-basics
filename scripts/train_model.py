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

    # Retrieve training parameters
    batch_size = config['training']['batch_size']
    max_steps = config['training']['max_steps']
    max_norm = config['training']['max_norm']
    device = config['training']['device']

    # Retrieve logging parameters
    log_interval = config['logging']['log_interval']
    eval_interval = config['logging']['eval_interval']
    eval_batches = config['logging']['eval_batches']

    # Retrieve checkpointing parameters
    checkpoint_dir = config['checkpoints']['checkpoint_dir']
    checkpoint_interval = config['checkpoints']['checkpoint_interval']

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

    transformer.to(device)

    optimizer = AdamW(
        params=transformer.parameters(),
        lr=1.0,  # Placeholder, later replaced by learning_schedule
        betas=tuple(config['optim']['betas']),
        eps=config['optim']['eps'],
        weight_decay=config['optim']['weight_decay']
    )

    def evaluate(step):
        transformer.eval()
        total_loss = 0.0

        with torch.no_grad():
            for _ in range(eval_batches):
                b, t = get_batch(val_data, batch_size, config['model']['context_length'], device)
                l = cross_entropy(transformer(b), t)
                total_loss += l.item()

        transformer.train()
        avg_loss = total_loss / eval_batches
        print(f"Step: {step} | Val Loss: {avg_loss:.4f}")


    for step in range(max_steps):
        batch, target = get_batch(train_data, batch_size, config['model']['context_length'], device)

        optimizer.zero_grad()
        loss = cross_entropy(transformer(batch), target)

        loss.backward()

        gradient_clipping(transformer.parameters(), max_norm)

        lr = learning_schedule(
            t=step,
            alpha_max=config['lr_schedule']['alpha_max'],
            alpha_min=config['lr_schedule']['alpha_min'],
            t_warm=config['lr_schedule']['t_warm'],
            t_cos=config['lr_schedule']['t_cos']
        )
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.step()

        if step % log_interval == 0:
            print_log(loss, lr, step)

        if step % eval_interval == 0:
            evaluate(step)

        if step % checkpoint_interval == 0:
            save_checkpoint(transformer, optimizer, step, f"{checkpoint_dir}/checkpoint_{step}.pt")


def print_log(loss, lr, step):
    print(f"Step: {step} | Loss: {loss.item():.4f} | LR: {lr:.6f}")


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
    parser.add_argument("--device", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main()
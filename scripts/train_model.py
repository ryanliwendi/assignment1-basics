from argparse import ArgumentParser
import os

import yaml
import numpy as np
import torch
import math

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

    # Retrieve checkpointing parameters
    checkpoint_dir = config['checkpoints']['checkpoint_dir']
    checkpoint_interval = config['checkpoints']['checkpoint_interval']
    os.makedirs(checkpoint_dir, exist_ok=True)

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

    if args.resume:
        if not args.checkpoint_path:
            raise ValueError("--checkpoint_path is required when using --resume")
        cur_step = load_checkpoint(args.checkpoint_path, transformer, optimizer) + 1
    else:
        cur_step = 1

    while cur_step < max_steps + 1:
        batch, target = get_batch(train_data, batch_size, config['model']['context_length'], device)

        optimizer.zero_grad()
        loss = cross_entropy(transformer(batch), target)

        loss.backward()

        gradient_clipping(transformer.parameters(), max_norm)

        lr = learning_schedule(
            t=cur_step,
            alpha_max=config['lr_schedule']['alpha_max'],
            alpha_min=config['lr_schedule']['alpha_min'],
            t_warm=config['lr_schedule']['t_warm'],
            t_cos=config['lr_schedule']['t_cos']
        )
        for group in optimizer.param_groups:
            group['lr'] = lr

        optimizer.step()

        if cur_step % log_interval == 0:
            print_log(loss, lr, cur_step)

        if cur_step % eval_interval == 0:
            evaluate(cur_step, transformer, val_data, config)

        if cur_step % checkpoint_interval == 0:
            save_checkpoint(transformer, optimizer, cur_step, f"{checkpoint_dir}/checkpoint_{cur_step}.pt")

        cur_step += 1

    save_checkpoint(transformer, optimizer, max_steps, f"{checkpoint_dir}/checkpoint_{max_steps}.pt")
    evaluate(max_steps, transformer, val_data, config)


def evaluate(step, model, val_data, config):
    model.eval()
    total_loss = 0.0

    eval_batches = config['logging']['eval_batches']
    batch_size = config['training']['batch_size']
    context_len = config['model']['context_length']
    device = config['training']['device']

    with torch.no_grad():
        for _ in range(eval_batches):
            batch, target = get_batch(val_data, batch_size, context_len, device)
            loss = cross_entropy(model(batch), target)
            total_loss += loss.item()

    model.train()
    avg_loss = total_loss / eval_batches
    perplexity = math.exp(avg_loss)
    print(f"Step: {step} | Val Loss: {avg_loss:.4f} | Val PPL: {perplexity:.2f}")


def print_log(loss, lr, step):
    perplexity = math.exp(loss.item())
    print(f"Step: {step} | Loss: {loss.item():.4f} | PPL: {perplexity:.2f} | LR: {lr:.6f}")


def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.vocab_size is not None:
        config['model']['vocab_size'] = args.vocab_size
    if args.context_length is not None:
        config['model']['context_length'] = args.context_length
    if args.num_layers is not None:
        config['model']['num_layers'] = args.num_layers
    if args.num_heads is not None:
        config['model']['num_heads'] = args.num_heads
    if args.d_model is not None:
        config['model']['d_model'] = args.d_model
    if args.d_ff is not None:
        config['model']['d_ff'] = args.d_ff
    if args.theta is not None:
        config['model']['theta'] = args.theta

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.max_norm is not None:
        config['training']['max_norm'] = args.max_norm
    if args.device is not None:
        config['training']['device'] = args.device

    if args.beta1 is not None:
        config['optim']['betas'][0] = args.beta1
    if args.beta2 is not None:
        config['optim']['betas'][1] = args.beta2
    if args.eps is not None:
        config['optim']['eps'] = args.eps
    if args.weight_decay is not None:
        config['optim']['weight_decay'] = args.weight_decay

    if args.alpha_max is not None:
        config['lr_schedule']['alpha_max'] = args.alpha_max
    if args.alpha_min is not None:
        config['lr_schedule']['alpha_min'] = args.alpha_min
    if args.t_warm is not None:
        config['lr_schedule']['t_warm'] = args.t_warm
    if args.t_cos is not None:
        config['lr_schedule']['t_cos'] = args.t_cos

    if args.train_path is not None:
        config['data']['train_path'] = args.train_path
    if args.val_path is not None:
        config['data']['val_path'] = args.val_path

    if args.log_interval is not None:
        config['logging']['log_interval'] = args.log_interval
    if args.eval_interval is not None:
        config['logging']['eval_interval'] = args.eval_interval
    if args.eval_batches is not None:
        config['logging']['eval_batches'] = args.eval_batches

    if args.checkpoint_dir is not None:
        config['checkpoints']['checkpoint_dir'] = args.checkpoint_dir
    if args.checkpoint_interval is not None:
        config['checkpoints']['checkpoint_interval'] = args.checkpoint_interval

    return config


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/default.yaml")

    parser.add_argument("--vocab_size", type=int)
    parser.add_argument("--context_length", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--theta", type=float)

    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--weight_decay", type=float)

    parser.add_argument("--alpha_max", type=float)
    parser.add_argument("--alpha_min", type=float)
    parser.add_argument("--t_warm", type=int)
    parser.add_argument("--t_cos", type=int)

    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--max_norm", type=float)
    parser.add_argument("--device", type=str)

    parser.add_argument("--train_path", type=str)
    parser.add_argument("--val_path", type=str)

    parser.add_argument("--log_interval", type=int)
    parser.add_argument("--eval_interval", type=int)
    parser.add_argument("--eval_batches", type=int)

    parser.add_argument("--checkpoint_dir", type=str)
    parser.add_argument("--checkpoint_interval", type=int)

    parser.add_argument("--resume", action="store_true")  # A bool value of whether to load a checkpoint
    parser.add_argument("--checkpoint_path", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main()
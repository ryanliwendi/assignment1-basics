import argparse
import yaml
import torch

from cs336_basics import TransformerLM, Tokenizer, load_checkpoint
from cs336_basics.generate import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--vocab", type=str, default="outputs/tinystories/vocab.json")
    parser.add_argument("--merges", type=str, default="outputs/tinystories/merges.json")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["model"]

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, special_tokens=["<|endoftext|>"])

    model = TransformerLM(**cfg).to(device)
    step = load_checkpoint(args.checkpoint, model, optimizer=None)
    print(f"Loaded checkpoint from step {step}")

    output = generate(model, tokenizer, args.prompt, args.max_tokens, args.top_p, args.temp)
    print(f"{args.prompt}{output}")


if __name__ == "__main__":
    main()
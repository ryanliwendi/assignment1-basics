# CS336 Spring 2025 Assignment 1: Basics

A comprehensive implementation of core large language model components from scratch, including BPE tokenization, transformer architecture, training utilities, and text generation.

For the full assignment description, see [cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf).

## Features

- **BPE Tokenizer**: Complete byte-pair encoding implementation with training and inference
- **Transformer Architecture**: Modern LLM components including RoPE, RMSNorm, SwiGLU, and multi-head attention
- **Training Pipeline**: AdamW optimizer, cosine learning rate scheduling, gradient clipping, and checkpointing
- **Text Generation**: Temperature scaling and top-p (nucleus) sampling

## Project Structure

```
assignment1-basics/
├── cs336_basics/              # Main package
│   ├── model.py               # Transformer architecture (Embedding, RMSNorm, SwiGLU, Attention, etc.)
│   ├── tokenizer.py           # BPE tokenizer for encoding/decoding text
│   ├── bpe.py                 # BPE training algorithm
│   ├── train.py               # Training utilities (loss, AdamW, LR scheduling, checkpointing)
│   └── generate.py            # Text generation with sampling
├── configs/
│   └── default.yaml           # Training configuration
├── scripts/
│   ├── train_model.py         # Main training script with CLI args
│   ├── train_bpe_tinystories.py   # Train BPE on TinyStories
│   ├── train_bpe_owt.py       # Train BPE on OpenWebText
│   ├── encode_dataset.py      # Encode datasets to token IDs
│   └── encode_samples.py      # Analyze tokenizer compression/throughput
├── tests/                     # Unit tests and fixtures
├── checkpoints/               # Saved model checkpoints
├── data/                      # Raw datasets
└── outputs/                   # Tokenizer files and encoded datasets
```

## Setup

### Environment

We use `uv` for reproducible environment management. Install it from [here](https://github.com/astral-sh/uv) or run:

```sh
pip install uv
# or
brew install uv
```

Run any code with automatic environment setup:

```sh
uv run <python_file_path>
```

### Download Data

Download TinyStories and OpenWebText sample:

```sh
mkdir -p data
cd data

# TinyStories
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText sample
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## Usage

### 1. Train BPE Tokenizer

Train a BPE tokenizer on TinyStories (with profiling):

```sh
cd scripts
export PYTHONPATH=$PYTHONPATH:..
scalene run train_bpe_tinystories.py
```

Outputs saved to `outputs/tinystories/`:
- `vocab.json` - Token vocabulary (ID to bytes mapping)
- `merges.json` - Merge rules

### 2. Encode Dataset

Convert text to token IDs:

```sh
# Encode TinyStories train/validation
uv run scripts/encode_dataset.py ts-train
uv run scripts/encode_dataset.py ts-val
```

Outputs saved as NumPy arrays (uint16) in `outputs/tinystories/`.

### 3. Train Model

Train a transformer language model:

```sh
uv run scripts/train_model.py --config configs/default.yaml
```

Override any config parameter via CLI:

```sh
uv run scripts/train_model.py \
    --config configs/default.yaml \
    --batch_size 64 \
    --num_layers 6 \
    --d_model 768 \
    --max_steps 20000 \
    --alpha_max 0.0003
```

Resume from checkpoint:

```sh
uv run scripts/train_model.py --config configs/default.yaml --resume checkpoints/step_5000.pt
```

### 4. Generate Text

Generate text from a trained model:

```sh
uv run scripts/generate_text.py --checkpoint checkpoints/checkpoint_10000.pt --prompt "Once upon a time"
```

With sampling parameters:

```sh
uv run scripts/generate_text.py \
    --checkpoint checkpoints/checkpoint_10000.pt \
    --prompt "The little girl" \
    --temp 0.8 \
    --top_p 0.9 \
    --max_tokens 200
```

## Configuration

Default training configuration (`configs/default.yaml`):

| Category | Parameter | Default | Description |
|----------|-----------|---------|-------------|
| **Model** | vocab_size | 10,000 | Vocabulary size |
| | context_length | 256 | Maximum sequence length |
| | num_layers | 4 | Transformer blocks |
| | num_heads | 16 | Attention heads |
| | d_model | 512 | Model dimension |
| | d_ff | 1,344 | FFN hidden dimension |
| | theta | 10,000.0 | RoPE base frequency |
| **Optimizer** | alpha_max | 0.001 | Peak learning rate |
| | alpha_min | 0.00001 | Minimum learning rate |
| | weight_decay | 0.1 | L2 regularization |
| | betas | [0.9, 0.999] | Adam momentum terms |
| **Training** | batch_size | 128 | Batch size |
| | max_steps | 10,000 | Total training steps |
| | t_warm | 100 | Warmup steps |
| | t_cos | 10,000 | Cosine decay steps |
| | max_norm | 1.0 | Gradient clipping |

## Architecture

The transformer implementation follows modern LLM design:

- **RMSNorm**: Root mean square normalization (more stable than LayerNorm)
- **RoPE**: Rotary positional embeddings for length generalization
- **SwiGLU**: Gated activation function with SiLU
- **Pre-norm**: Normalization before attention/FFN (better training stability)
- **Causal masking**: Autoregressive language modeling

## Testing

Run all tests:

```sh
uv run pytest
```

Run specific test files:

```sh
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_model.py
```

## Dependencies

Key dependencies (managed via `pyproject.toml`):

- `torch ~= 2.6.0` - Deep learning framework
- `numpy` - Numerical computing
- `regex` - Unicode-aware regex for tokenization
- `pyyaml` - Configuration parsing
- `einops`, `einx` - Tensor operations
- `tiktoken` - OpenAI tokenizer (for testing)
- `pytest` - Testing framework

## License

Stanford University (MIT-style) - see [LICENSE](./LICENSE)

# Mini_assistant_gpt2: a GPT-2 chat assistant built by hand

A learning project: GPT-2 written from scratch in PyTorch, initialised with OpenAI's original
pretrained weights and fine-tuned into a small chat assistant on instruction and conversation data.
LoRA and DoRA adapters are implemented as well; the best run so far is a full fine-tune of GPT-2 small (124M).

## What is in here

| Path | Contents |
|---|---|
| `model/GPT_manual_architecture.py` | the GPT-2 architecture (attention, blocks, layer norm) |
| `model/Load_model.py` | downloads the OpenAI checkpoint and loads it into the model |
| `model/LoRA.py`, `model/DoRA.py`, `model/GPT_full_model.py` | adapters, token-embedding resizing, model setup |
| `dataset/chat_format.py` | the chat template and loss masking, shared by data, training and inference |
| `dataset/get_prep_data.py` | builds the dataset from Alpaca (cleaned) and OpenAssistant (OASST1) |
| `dataset/New_dataloader.py`, `dataset/batching.py` | tokenisation, labels, length-bucketed batches |
| `srs/training/new_training.py` | training loop (settings are at the top of the file) |
| `srs/inference/` | generation (temperature, top-k, top-p) and a small chat loop |
| `tests/test_pipeline.py` | unit tests for the template, masking, truncation and batching |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is only needed to read the original OpenAI checkpoint. On first use the GPT-2 weights,
the tokenizer and the two datasets are downloaded automatically (they are not stored in this repository).

## Usage

Run everything from the repository root.

```bash
python tests/test_pipeline.py            # quick checks, no GPU needed
python -m srs.training.new_training      # fine-tune (adapter, max length, batch size, DEVICE ... are set at the top)
python -m srs.inference.inference        # chat with the best checkpoint
```

## Chat format

Every reply is closed by `<|endoftext|>`, so the model learns where a turn ends. Only assistant text is trained on.

```
<|user|>
{user message}
<|assistant|>
{assistant reply}<|endoftext|><|user|>
{next user message}
<|assistant|>
...
```

## Data

- Alpaca (cleaned): about 46.6k single-turn instructions.
- OASST1: English only, reviewed messages, the two best-ranked replies per prompt, conversations without
  repeated openings, split into train and validation per conversation tree.
- Result: 56,215 training and 6,231 validation conversations, about 11% of the training rows multi-turn.

## Results

GPT-2 small (124M), full fine-tune, 3 epochs, trained on a MacBook CPU (about 3.6 hours per epoch).

| | validation loss | perplexity |
|---|---|---|
| before fine-tuning (small sample) | about 3.4 | about 31 |
| after 3 epochs | 1.985 | 7.28 |

Loss and perplexity are measured on assistant tokens only.

Example (greedy decoding):

- "What is the capital of France?" -> "The capital of France is Paris."
- "Convert 10 cm to meters." -> wrong number. A 124M model does not do arithmetic.
- Multi-turn: it picks up the name from an earlier turn but often does not answer the question asked.

## Limitations

GPT-2 small is a weak base for conversation: it follows simple instructions and stops properly, but
arithmetic and keeping track of earlier turns are unreliable. The multi-turn share of the data is small.
Larger GPT-2 sizes and more conversation data are the next things to try.

## What I learned from the first version

- Truncating conversations at 300 tokens removed the end token from many examples and most later turns.
- Different templates for training data and for chat prompts made a small model behave inconsistently.
- Randomly initialised embeddings for the two new chat tokens gave them near-zero logits, far above the
  real tokens (initial loss about 96). Initialising them from the mean of the existing embeddings fixed it.
- With adapters, the new token rows were frozen and could never be learned.

## Credits and licences

Built on OpenAI's GPT-2, the `yahma/alpaca-cleaned` dataset (derived from Stanford Alpaca) and
`OpenAssistant/oasst1`. The weights and datasets are not included here; check the licence of each
source before reusing them.

"""Fine-tune the hand-built GPT-2 (OpenAI weights) as a chat assistant.

Run from the project root:
    python -m srs.training.new_training

What changed compared with the first version (see CHANGES_v2.md for the reasoning):
  * data: one template, English only, best-ranked OASST replies, split by conversation tree
  * dataloader: conversations are never cut mid-answer (MAX_LENGTH 512), <|endoftext|> closes every
    assistant reply and is trained, user text and marker tokens are masked, shuffled length-bucketed batches
  * optimisation: warmup + cosine schedule, sensible learning rate per model size, weight-decay groups,
    gradient accumulation that also flushes at the end of an epoch, non-finite steps are skipped
  * validation loss on held-out conversations, best checkpoint kept, resumable checkpoints
  * previews use the same prompt template as training, with top-k / top-p sampling
"""
import math
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from srs.LLM_classification.config import Config
from srs.training.help_func import get_device, get_tokenizer, get_model
import dataset.get_prep_data as ds_prep
import dataset.New_dataloader as DL
from srs.inference.help import generate

# =============================================================================== settings
ADAPTER = None               # None = full fine-tune (recommended for the 124M model), "LoRA" or "DoRA"
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05

DATA, LANGUAGE, MAX_RANK = "all", "en", 1
OASST_REPEAT = 2             # how many times the (multi-turn) OASST rows are seen per epoch

MAX_LENGTH = 512             # tokens; longer conversations are cut after their last complete reply
BATCH_SIZE = 4               # lower to 2 if the Mac runs out of memory, then raise GRAD_ACCUM to 16
GRAD_ACCUM = 8               # effective batch = BATCH_SIZE * GRAD_ACCUM
EPOCHS = 3

DEVICE = None

FULL_FT_LR = {"gpt2-small (124M)": 5e-5, "gpt2-medium (355M)": 3e-5,
              "gpt2-large (774M)": 2e-5, "gpt2-xl (1558M)": 1e-5}
ADAPTER_LR = 2e-4
WARMUP_FRAC = 0.03
WEIGHT_DECAY = 0.01          # not used with adapters (see GPT_full_model.freeze_except_adapter)
MAX_GRAD_NORM = 1.0

EVAL_EVERY = 500             # optimizer steps
EVAL_BATCHES = 150           # batches of the validation set used for the periodic check (full set at epoch end)
LOG_EVERY = 50               # optimizer steps
USE_AUTOCAST = False         # bf16 autocast on MPS; the first run hit a NaN with it, enable only if you need the speed
RESUME = True
SEED = 42

PREVIEWS = [
    [("user", "What is the capital of France?")],
    [("user", "Convert 10 cm to meters.")],
    [("user", "Write a short poem about the sea.")],
    [("user", "My name is Anna and I like chess."),
     ("assistant", "Nice to meet you, Anna! Chess is a great game."),
     ("user", "What is my name?")],
]
# =====================================================================================


def evaluate(model, dataloader, device, max_batches=None):
    """Token-level mean loss on assistant tokens."""
    model.eval()
    dataloader.batch_sampler.set_epoch(0)          # same (seeded) batches every time
    total, count = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            n = (shift_labels != Config.IGNORE_INDEX).sum().item()
            if n == 0:
                continue
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                                   ignore_index=Config.IGNORE_INDEX, reduction="sum")
            total += loss.item()
            count += n
    model.train()
    return total / max(count, 1)


def save_checkpoint(path, model, optimizer=None, scheduler=None, **extra):
    state = {"model_state": model.state_dict(), **extra}
    if optimizer is not None:
        state["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    tmp = str(path) + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def show_training_example(dataloader, tokenizer, user_id):
    """Sanity check printed once: what the model actually gets trained on."""
    ds = dataloader.dataset
    pick = next((i for i, (ids, _) in enumerate(ds.samples) if ids.count(user_id) >= 2), 0)
    ids, labels = ds.samples[pick]
    print("\n--- one training example (multi-turn if available) ---")
    print(tokenizer.decode(ids))
    print("\n--- only these tokens are trained on (assistant replies + closing <|endoftext|>) ---")
    print(tokenizer.decode([t for t, l in zip(ids, labels) if l != Config.IGNORE_INDEX]))
    print("-----------------------------------------------------------\n")


def main():
    torch.manual_seed(SEED)
    device = get_device(fixed = DEVICE)
    tokenizer = get_tokenizer()
    model_type = Config.MODEL_TYPE
    model_name = "custom_gpt2"
    print(f"device: {device} | model: {model_type} | adapter: {ADAPTER}")

    # ---- data
    train_df, val_df = ds_prep.get_data_preprocessed(Config, split_ratio=0.9, data=DATA,
                                                     language=LANGUAGE, max_rank=MAX_RANK)
    if OASST_REPEAT > 1:
        extra = train_df[train_df["source"] == "oasst"]
        train_df = pd.concat([train_df] + [extra] * (OASST_REPEAT - 1), ignore_index=True)
    train_dl = DL.create_dataloader(train_df["text"].tolist(), tokenizer, BATCH_SIZE, shuffle=True,
                                    max_length=MAX_LENGTH, seed=SEED)
    val_dl = DL.create_dataloader(val_df["text"].tolist(), tokenizer, BATCH_SIZE, shuffle=True,
                                  max_length=MAX_LENGTH, seed=SEED)
    show_training_example(train_dl, tokenizer, tokenizer.convert_tokens_to_ids(Config.USER_TOKEN))

    # ---- model
    model = get_model(model_name, tokenizer, lora=ADAPTER, r=LORA_R, alpha=LORA_ALPHA,
                      dropout=LORA_DROPOUT).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {sum(p.numel() for p in params):,} of {total_params:,}")

    lr = ADAPTER_LR if ADAPTER else FULL_FT_LR[model_type]
    if ADAPTER:
        groups = [{"params": params, "weight_decay": 0.0}]
    else:
        groups = [{"params": [p for p in params if p.ndim >= 2], "weight_decay": WEIGHT_DECAY},
                  {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
    optimizer = AdamW(groups, lr=lr, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(train_dl) / GRAD_ACCUM)
    total_steps = steps_per_epoch * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(WARMUP_FRAC * total_steps), total_steps)
    print(f"{len(train_dl)} batches/epoch -> {steps_per_epoch} optimizer steps/epoch, {total_steps} in total, lr {lr}")

    latest_path = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{model_name}_{model_type}.pt"
    best_path = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{model_name}_{model_type}_best.pt"
    os.makedirs(Config.MODEL_WEIGHTS_PATH, exist_ok=True)

    start_epoch, skip_steps, global_step, best_val = 0, 0, 0, float("inf")
    if RESUME and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device)
        if "scheduler_state" in ckpt and "optimizer_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt.get("epoch", 0)
            skip_steps = ckpt.get("step_in_epoch", 0)
            global_step = ckpt.get("global_step", 0)
            best_val = ckpt.get("best_val", float("inf"))
            print(f"resumed from {latest_path.name}: epoch {start_epoch}, step in epoch {skip_steps}, "
                  f"global step {global_step}, best val loss {best_val:.4f}")
        else:
            print(f"{latest_path.name} comes from the old training script (no scheduler state) -> starting fresh")

    def run_eval(epoch, step_in_epoch, full=False):
        nonlocal best_val
        val_loss = evaluate(model, val_dl, device, None if full else EVAL_BATCHES)
        print(f"\n[eval] step {global_step} | val loss {val_loss:.4f} | val perplexity {math.exp(min(val_loss, 20)):.2f}"
              f"{' (full validation set)' if full else ''}")
        for history in PREVIEWS:
            reply = generate(model, tokenizer, device, history, max_new_tokens=80, temperature=0.0,
                             repetition_penalty=1.1, model_name=model_name)
            print(f"  user: {history[-1][1]}\n  bot : {reply}")
        if val_loss < best_val and (full or global_step > 0):
            best_val = val_loss
            save_checkpoint(best_path, model, epoch=epoch, global_step=global_step, val_loss=val_loss)
            print(f"  new best val loss -> {best_path.name}")
        save_checkpoint(latest_path, model, optimizer, scheduler, epoch=epoch, step_in_epoch=step_in_epoch,
                        global_step=global_step, best_val=best_val)
        if device.type == "mps":
            torch.mps.empty_cache()
        print()

    if global_step == 0:
        base = evaluate(model, val_dl, device, EVAL_BATCHES)
        print(f"[eval] before training: val loss {base:.4f}")

    # ---- training
    model.train()
    skipped_nonfinite = 0
    for epoch in range(start_epoch, EPOCHS):
        train_dl.batch_sampler.set_epoch(epoch)
        optimizer.zero_grad(set_to_none=True)
        run_loss, run_n = 0.0, 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{EPOCHS}")
        for step, batch in enumerate(pbar):
            if step < skip_steps:
                continue
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=USE_AUTOCAST):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = logits.float()
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if (shift_labels != Config.IGNORE_INDEX).sum() == 0:
                continue
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                                   ignore_index=Config.IGNORE_INDEX)
            if not torch.isfinite(loss):
                skipped_nonfinite += 1
                print(f"non-finite loss at epoch {epoch}, step {step} (skipped so far: {skipped_nonfinite})")
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss / GRAD_ACCUM).backward()
            run_loss += loss.item()
            run_n += 1

            if (step + 1) % GRAD_ACCUM == 0 or step == len(train_dl) - 1:
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % LOG_EVERY == 0 and run_n:
                    pbar.set_postfix(loss=f"{run_loss / run_n:.3f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
                    run_loss, run_n = 0.0, 0
                if global_step % EVAL_EVERY == 0:
                    run_eval(epoch, step + 1)

        skip_steps = 0
        run_eval(epoch + 1, 0, full=True)     # full validation + checkpoint at the end of every epoch
    print("training finished")


if __name__ == "__main__":
    main()

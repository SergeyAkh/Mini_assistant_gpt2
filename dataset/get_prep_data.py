"""Builds the chat dataset (Alpaca + OpenAssistant) in ONE consistent template.

Differences from the first version:
  * one template for every row (see dataset/chat_format.py), <|endoftext|> after every assistant reply
  * OASST: language filter that really works, deleted / un-reviewed messages dropped,
    only the best-ranked assistant replies per prompt (rank <= max_rank, default 1) are kept
  * OASST threads are built root -> leaf; broken threads are skipped, trailing user
    messages are trimmed, and conversations that are just a prefix of a longer one are removed
  * train / validation split is done per conversation tree (no leakage)
  * cached CSV names include the data/language settings, so changing them rebuilds the data
"""
import os

import numpy as np
import pandas as pd

from dataset.chat_format import clean_text, format_conversation

MAX_CHARS = 12000      # extreme outliers are dropped here; exact token limits are applied in the dataloader
SEED = 42


# ----------------------------------------------------------------------------- Alpaca
def _load_alpaca_raw(config) -> pd.DataFrame:
    path = os.path.join(config.DATA_RAW_PATH, "alpaca_data_cleaned.json")
    if os.path.exists(path):
        return pd.read_json(path)
    os.makedirs(config.DATA_RAW_PATH, exist_ok=True)
    df = pd.read_json(config.alpaca_ds)
    df.to_json(path)
    return df


def alpaca_df(config) -> pd.DataFrame:
    df = _load_alpaca_raw(config)
    df["input"] = df["input"].fillna("")
    rows, seen = [], set()
    for i, (ins, inp, out) in enumerate(zip(df["instruction"], df["input"], df["output"])):
        ins, inp, out = clean_text(ins), clean_text(inp), clean_text(out)
        if not ins or not out:
            continue
        user = ins + ("\n" + inp if inp else "")
        if (user, out) in seen:
            continue
        seen.add((user, out))
        text = format_conversation([(user, out)])
        if len(text) > MAX_CHARS:
            continue
        rows.append({"text": text, "source": "alpaca", "group": f"alpaca_{i}", "n_turns": 1})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- OASST
def _load_oasst_raw(config) -> pd.DataFrame:
    path_tr = os.path.join(config.DATA_RAW_PATH, "oasst1_train.json")
    path_val = os.path.join(config.DATA_RAW_PATH, "oasst1_val.json")
    if not (os.path.exists(path_tr) and os.path.exists(path_val)):
        from datasets import load_dataset  # lazy: only needed for the first download
        os.makedirs(config.DATA_RAW_PATH, exist_ok=True)
        ds = load_dataset(config.oasst_ds)
        ds["train"].to_pandas().to_json(path_tr, orient="records", lines=True)
        ds["validation"].to_pandas().to_json(path_val, orient="records", lines=True)
    df_tr = pd.read_json(path_tr, lines=True)
    df_val = pd.read_json(path_val, lines=True)
    # train/validation of the original release are merged; we re-split by tree below
    return pd.concat([df_tr, df_val], ignore_index=True)


def oasst_df(config, language="en", max_rank=1) -> pd.DataFrame:
    df = _load_oasst_raw(config)

    df = df[~df["deleted"].fillna(False).astype(bool)]
    df = df[df["review_result"] == True]                       # noqa: E712  (1.0 / True)
    if language:
        df = df[df["lang"] == language]

    # Keep only the best-ranked replies to each prompt (rank 0 = best). Prompter messages have no rank.
    # max_rank=0 -> best reply only, 1 -> best two, ... (unranked replies are always kept)
    is_assistant = df["role"] == "assistant"
    df = df[(~is_assistant) | df["rank"].isna() | (df["rank"] <= max_rank)]

    df = df.drop_duplicates("message_id").set_index("message_id")
    parent = df["parent_id"].to_dict()
    role = df["role"].to_dict()
    text = df["text"].to_dict()
    tree = df["message_tree_id"].to_dict()
    ids = set(df.index)
    has_child = {p for p in parent.values() if isinstance(p, str)}

    # root -> leaf paths over the kept messages
    paths = {}
    for leaf in (m for m in ids if m not in has_child):
        path, cur, ok = [], leaf, True
        while True:
            path.append(cur)
            p = parent[cur]
            if not isinstance(p, str):         # reached the root (no parent)
                break
            if p not in ids:                   # an ancestor was filtered out -> broken thread
                ok = False
                break
            cur = p
        if not ok:
            continue
        path.reverse()
        if role[path[0]] != "prompter":
            continue
        if role[path[-1]] == "prompter":       # unanswered last user message -> trim it
            path = path[:-1]
        if not path:
            continue
        if any(role[m] != ("prompter" if i % 2 == 0 else "assistant") for i, m in enumerate(path)):
            continue
        paths[tuple(path)] = tree[path[0]]

    # drop conversations that are only a prefix of a longer one (avoids duplicated openings)
    prefixes = set()
    for path in paths:
        for k in range(1, len(path)):
            prefixes.add(path[:k])

    rows, seen = [], set()
    for path, tree_id in paths.items():
        if path in prefixes:
            continue
        msgs = [clean_text(text[m]) for m in path]
        if any(not m for m in msgs):
            continue
        turns = list(zip(msgs[0::2], msgs[1::2]))
        conv = format_conversation(turns)
        if conv in seen or len(conv) > MAX_CHARS:
            continue
        seen.add(conv)
        rows.append({"text": conv, "source": "oasst", "group": f"oasst_{tree_id}", "n_turns": len(turns)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- public API
def get_data_preprocessed(config, split_ratio=0.9, data="all", language="en", max_rank=1, force_rebuild=False):
    """Returns (train_df, val_df) with columns: text, source, group, n_turns.

    data: "alpaca" | "oasst1" | "all"; language: e.g. "en", or None for every language
    (the language filter only affects OASST; Alpaca is English); max_rank: OASST reply quality
    cut-off, see oasst_df.
    """
    tag = f"{data}_{language or 'all'}_r{max_rank}"
    p_tr = os.path.join(config.PREPROC_DS, f"train_{tag}.csv")
    p_val = os.path.join(config.PREPROC_DS, f"val_{tag}.csv")

    if os.path.exists(p_tr) and os.path.exists(p_val) and not force_rebuild:
        return pd.read_csv(p_tr), pd.read_csv(p_val)

    os.makedirs(config.PREPROC_DS, exist_ok=True)
    parts = []
    if data in ("alpaca", "all"):
        parts.append(alpaca_df(config))
    if data in ("oasst1", "all"):
        parts.append(oasst_df(config, language=language, max_rank=max_rank))
    if not parts:
        raise ValueError(f"unknown data option: {data}")
    df = pd.concat(parts, ignore_index=True)

    # split by group: a whole OASST tree goes either to train or to validation
    rng = np.random.RandomState(SEED)
    groups = np.array(sorted(df["group"].unique()))
    val_groups = set(rng.choice(groups, size=int(round(len(groups) * (1 - split_ratio))), replace=False))
    is_val = df["group"].isin(val_groups)

    train = df[~is_val].sample(frac=1, random_state=SEED).reset_index(drop=True)
    val = df[is_val].sample(frac=1, random_state=SEED).reset_index(drop=True)
    train.to_csv(p_tr, index=False)
    val.to_csv(p_val, index=False)

    print(f"[data] built {tag}: train {len(train)} rows, val {len(val)} rows")
    print(train.groupby("source")["n_turns"].agg(["count", "mean"]).rename(columns={"count": "rows", "mean": "avg_turns"}))
    return train, val

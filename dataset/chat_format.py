"""Single source of truth for the chat template and label masking.

Used by data preparation, the dataloader, training previews and inference, so the
format the model is trained on is exactly the format it is prompted with.
This module has no torch / transformers imports on purpose (easy to unit-test).

Template (one turn):
    <|user|>\n{user}\n<|assistant|>\n{assistant}<|endoftext|>
A conversation is the concatenation of its turns. <|endoftext|> closes EVERY
assistant reply (not only the last one), so the model learns where a turn ends.
"""

USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|endoftext|>"
SPECIAL_STRINGS = (USER_TOKEN, ASSISTANT_TOKEN, EOS_TOKEN)


def clean_text(text) -> str:
    """Normalise a message and make sure it cannot inject special tokens."""
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    for s in SPECIAL_STRINGS:
        text = text.replace(s, "")
    return text.strip()


def format_turn(user: str, assistant: str) -> str:
    return f"{USER_TOKEN}\n{user}\n{ASSISTANT_TOKEN}\n{assistant}{EOS_TOKEN}"


def format_conversation(turns) -> str:
    """turns: list of (user_text, assistant_text) pairs."""
    return "".join(format_turn(u, a) for u, a in turns)


def build_prompt(history) -> str:
    """Prompt for generation.

    history: list of (role, text), role in {"user", "assistant"}, last item must be
    a user message. The result ends with '<|assistant|>\\n', ready for the model
    to continue.
    """
    if not history or history[-1][0] != "user":
        raise ValueError("history must be non-empty and end with a user message")
    parts = []
    for role, text in history:
        text = clean_text(text)
        if role == "user":
            parts.append(f"{USER_TOKEN}\n{text}\n{ASSISTANT_TOKEN}\n")
        elif role == "assistant":
            parts.append(f"{text}{EOS_TOKEN}")
        else:
            raise ValueError(f"unknown role: {role}")
    return "".join(parts)


def compute_labels(ids, user_id, assistant_id, ignore_index=-100, newline_id=None):
    """Loss only on assistant replies, including the <|endoftext|> that closes each one.

    Masked: everything the user says, the <|user|> / <|assistant|> marker tokens, and the template
    newline right after <|assistant|> (it is part of the prompt at inference time, so it is given,
    not predicted). Newlines inside a reply stay trained.
    """
    labels = [ignore_index] * len(ids)
    in_assistant = False
    after_marker = False
    for i, tok in enumerate(ids):
        if tok == user_id:
            in_assistant = False
            after_marker = False
        elif tok == assistant_id:
            in_assistant = True
            after_marker = True
        elif in_assistant:
            if after_marker:
                after_marker = False
                if newline_id is not None and tok == newline_id:
                    continue
            labels[i] = tok
    return labels


def truncate_to_last_complete_turn(ids, eos_id, max_len):
    """Cut an over-long conversation after the last assistant reply that fits.

    Never cuts in the middle of an answer. Returns None when not even the first
    reply fits (the sample should then be dropped).
    """
    if len(ids) <= max_len:
        return list(ids)
    cut = -1
    for i in range(max_len):
        if ids[i] == eos_id:
            cut = i
    if cut < 0:
        return None
    return list(ids[: cut + 1])

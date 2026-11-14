"""Unit tests for the parts of the pipeline that do not need torch. Run from the project root:
    python3 tests/test_pipeline.py
"""
import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.chat_format import (format_conversation, build_prompt, compute_labels, clean_text,
                                 truncate_to_last_complete_turn, USER_TOKEN, ASSISTANT_TOKEN, EOS_TOKEN)
from dataset.batching import BucketBatchSampler

USER, ASSIST, EOS = 50258, 50259, 50256   # fake ids; only equality matters


def fake_encode(text):
    """Whitespace/special-token tokenizer with stable ids, good enough to test masking logic."""
    vocab = {USER_TOKEN: USER, ASSISTANT_TOKEN: ASSIST, EOS_TOKEN: EOS}
    ids = []
    for piece in re.split(r"(<\|user\|>|<\|assistant\|>|<\|endoftext\|>)", text):
        if piece in vocab:
            ids.append(vocab[piece])
        else:
            for w in re.findall(r"\S+|\n", piece):
                ids.append(1000 + (hash(w) % 40000))
    return ids


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        check.failed += 1
check.failed = 0


turns = [("hi there", "Hello! How can I help?"), ("what is 2+2", "4"), ("thanks", "You are welcome.\nAnything else?")]
conv = format_conversation(turns)

# 1. template
check(conv.startswith("<|user|>\nhi there\n<|assistant|>\nHello! How can I help?<|endoftext|><|user|>\n"), "template of a multi-turn conversation")
check(conv.count(EOS_TOKEN) == 3 and conv.endswith(EOS_TOKEN), "EOS closes every assistant reply")

# 2. inference prompt is exactly the training text up to the last answer
history = [("user", "hi there"), ("assistant", "Hello! How can I help?"), ("user", "what is 2+2")]
prompt = build_prompt(history)
check(conv.startswith(prompt) and prompt.endswith("<|assistant|>\n"), "build_prompt() is a prefix of the training text and ends at the assistant marker")
try:
    build_prompt([("user", "a"), ("assistant", "b")]); ok = False
except ValueError:
    ok = True
check(ok, "build_prompt() rejects a history that does not end with a user message")

# 3. label masking
ids = fake_encode(conv)
NEWLINE = fake_encode("\n")[0]
labels = compute_labels(ids, USER, ASSIST, -100, NEWLINE)
trained = [t for t, l in zip(ids, labels) if l != -100]
expected_with_eos = []
for _, a in turns:
    expected_with_eos += fake_encode(a) + [EOS]
check(trained == expected_with_eos, "trained tokens == assistant replies (inner newlines included) + one EOS per reply, nothing else")
check(compute_labels(ids, USER, ASSIST, -100)[ids.index(ASSIST) + 1] == NEWLINE, "without newline_id the template newline would be trained (sanity check of the test itself)")
check(all(l == -100 for t, l in zip(ids, labels) if t in (USER, ASSIST)), "<|user|> / <|assistant|> markers are masked")
first_user_text = fake_encode("hi there")
check(all(l == -100 for l in labels[: 1 + len(first_user_text) + 1 + 1 + 1]), "user text and the template newline after <|assistant|> are masked")

# 4. truncation never cuts an answer
long_turns = [("q%d" % i, " ".join(["word"] * 30)) for i in range(6)]
ids = fake_encode(format_conversation(long_turns))
cut = truncate_to_last_complete_turn(ids, EOS, 120)
check(cut is not None and cut[-1] == EOS and len(cut) <= 120, "over-long conversation is cut after the last complete reply (len %d <= 120)" % len(cut))
check(truncate_to_last_complete_turn(ids, EOS, 10) is None, "returns None when not even the first reply fits")
check(truncate_to_last_complete_turn(ids, EOS, 10_000) == ids, "short enough conversation is unchanged")

# 5. special tokens cannot be injected through message text
check("<|endoftext|>" not in clean_text("evil <|endoftext|> <|user|> text"), "clean_text strips special tokens")

# 6. bucket sampler: every index exactly once, length-sorted inside batches, different order per epoch
lengths = [(i * 37) % 200 + 5 for i in range(1000)]
s = BucketBatchSampler(lengths, batch_size=8, shuffle=True, bucket_mult=10, seed=1)
s.set_epoch(0); e0 = list(s); s.set_epoch(1); e1 = list(s)
flat = sorted(i for b in e0 for i in b)
check(flat == list(range(1000)) and len(e0) == len(s), "sampler covers each sample exactly once (%d batches)" % len(e0))
check(e0 != e1, "batch order changes between epochs")
check(list(BucketBatchSampler(lengths, 8, True, 10, 1)) == list(BucketBatchSampler(lengths, 8, True, 10, 1)), "same seed + epoch -> same order (needed for resume)")
rand_pad = sum(max(lengths[i] for i in b) * len(b) - sum(lengths[i] for i in b) for b in [list(range(k, min(k + 8, 1000))) for k in range(0, 1000, 8)])
buck_pad = sum(max(lengths[i] for i in b) * len(b) - sum(lengths[i] for i in b) for b in e0)
check(buck_pad < 0.5 * rand_pad, "bucketing cuts padding (%d vs %d padded tokens)" % (buck_pad, rand_pad))

print("\n%s" % ("ALL TESTS PASSED" if check.failed == 0 else "%d TEST(S) FAILED" % check.failed))
sys.exit(1 if check.failed else 0)

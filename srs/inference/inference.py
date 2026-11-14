# inference.py

from srs.inference.help import *
from srs.training.help_func import (get_device, get_tokenizer, get_model, load_checkpoint)
from torch.optim import AdamW
device = get_device()

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_K = 40
TOP_P = 0.9
ADAPTER = None   # must match the ADAPTER used in training: None (full fine-tune), "LoRA" or "DoRA"

# ---- Load tokenizer ----
tokenizer = get_tokenizer()

# special tokens
ASSISTANT_TOKEN = Config.ASSIST_TOKEN
USER_TOKEN = Config.USER_TOKEN
model_name = "custom_gpt2"
# ---- Load model ----
model = get_model(model_name, tokenizer, lora=ADAPTER, r=16,
                  alpha=32, dropout=0.0).to(device)

model_type = Config.MODEL_TYPE

CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{model_name}_{model_type}_best.pt"
if not CHECKPOINT_PATH.exists():     # fall back to the latest checkpoint
    CHECKPOINT_PATH = Config.MODEL_WEIGHTS_PATH / f"checkpoint_{model_name}_{model_type}.pt"

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state"])
print(f"Loaded {CHECKPOINT_PATH.name}")

model.eval()

def clear_history(history):
    """Clear the entire conversation history."""
    history.clear()
    print("History has been cleared!")

def main():
    print("Chat started. Type 'exit' to quit.\n")

    history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break
        elif user_input.lower() == "clear":
            clear_history(history)
            print("Conversation history erased. Starting fresh!\n")
            continue

        history.append(("user", user_input))
        # print(f"history: {history}")

        # the history is formatted with the same template as the training data
        answer = temp_predict(model_name, model, history, tokenizer, device,
                              max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
                              top_k=TOP_K, top_p=TOP_P)

        print(f"Bot: {answer}\n")

        history.append(("assistant", answer))

if __name__ == "__main__":
    main()
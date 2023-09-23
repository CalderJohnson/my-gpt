import torch
import typing as tp

# hyperparameters
settings = {
    "batch_size": 16,      # How many independent sequences are processed in parallel
    "block_size": 256,     # Maximum context length for predictions
    "epochs": 5000,        # Number of training epochs
    "learning_rate": 1e-3, # Learning rate
    "device": 'cuda' if torch.cuda.is_available() else 'cpu', # Device
    "n_embd": 64,          # How many tokens to generate prediction weights for
    "n_head": 4,           # Number of heads for multi headed attention
    "n_layer": 4,          # Number of decoder blocks
    "dropout": 0.1,        # Dropout ratio to mitigate overfitting
}


class Tokenizer:
    """Tokenization methods for a given vocabulary."""
    def __init__(self, vocab: tp.List[chr]):
        self.str_to_int = {ch:i for i, ch in enumerate(vocab)}
        self.int_to_str = {i:ch for i, ch in enumerate(vocab)}

    def encode(self, input_string: str):
        return [self.str_to_int[char] for char in input_string]

    def decode(self, input_list: tp.List[int]):
        return [self.int_to_str[num] for num in input_list]


def get_batch(data):
    """Generate a batch"""
    indices = torch.randint(len(data) - settings["block_size"], (settings["batch_size"],))
    src = torch.stack([data[i:i+settings["block_size"]] for i in indices])
    tgt = torch.stack([data[i+1:i+settings["block_size"]+1] for i in indices])
    src, tgt = src.to(settings["device"]), tgt.to(settings["device"])
    return src, tgt

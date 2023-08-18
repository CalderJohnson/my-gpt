import tensorflow as tf
import typing as tp

# Hyperparameters
settings = {
    "batch_size": 32,      # Number of independent sequences to process in parallel
    "block_size": 8,       # Maximum context length for predictions
    "seed": 1336,          # Seed for RNG
    "learning_rate": 1e-3, # The learning rate
    "epochs": 100,         # Number of epochs
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
    """Generate a batch of data, inputs and targets"""
    indices = tf.random.uniform(
        shape=[settings["batch_size"]],
        minval=0,
        maxval=(len(data) - settings["block_size"]),
        dtype=tf.int32,
        seed=settings["seed"],
    )

    inputs = tf.stack([data[i:i + settings["block_size"]] for i in indices])
    targets = tf.stack([data[i + 1:i + settings["block_size"] + 1] for i in indices])

    return inputs, targets

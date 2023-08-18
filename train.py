import tensorflow as tf
import typing as tp

from utils import Tokenizer, get_batch, settings
from model import BigramLanguageModel

# Retrieve and tokenize data
txt_data = open("data.txt").read()
vocab = sorted(list(set(txt_data)))
tokenizer = Tokenizer(vocab)
data = tf.constant(tokenizer.encode(txt_data))

# 90/10 split for train/valid data
n = int(0.9*len(data))
train_data = data[:n]
valid_data = data[n:]

# Initialize the model
model = BigramLanguageModel(len(vocab))
optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3)

# Train the model
for step in range(settings["epochs"]):
    # Sample a batch of data
    with tf.GradientTape() as tape:
        inputs, targets = get_batch(train_data)
        logits, loss = model(inputs, targets)

    # Calculate gradients and update weights
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

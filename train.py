import torch
from model import LanguageModel
from utils import settings, get_batch, Tokenizer

torch.manual_seed(2048)

with open('./data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Prepare data and tokenizer
vocab = sorted(list(set(text)))
settings["vocab_size"] = len(vocab)
tokenizer = Tokenizer(vocab)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
partition = int(0.9*len(data))
train_data = data[:partition]
valid_data = data[partition:]

# Initialize model and optimizer
model = LanguageModel()
gpt = model.to(settings["device"])
optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])

# Train the model
average_loss = 0
for i in range(settings["epochs"]):

    # Evaluate loss every 100 steps
    if i % 100 == 0 or i == settings["epochs"] - 1:
        average_loss /= 100
        print(f"Epoch {i}: loss={average_loss}")
        average_loss = 0

    # Get a batch of data
    inputs, targets = get_batch(train_data)

    # Train model based on loss
    logits, loss = model(inputs, targets)
    average_loss += loss.item()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate
print("".join(tokenizer.decode(gpt.generate(n_tokens=1000)[0].tolist())))

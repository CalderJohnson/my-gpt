import torch
from model import LanguageModel
from utils import settings, Tokenizer

# Prepare data and tokenizer
with open('./data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

vocab = sorted(list(set(text)))
settings["vocab_size"] = len(vocab)
tokenizer = Tokenizer(vocab)

gpt = LanguageModel().to(settings["device"])
gpt.load_state_dict(torch.load("./gpt_model.pt"), weights_only=True)
print("".join(tokenizer.decode(gpt.generate(n_tokens=1000)[0].tolist())))

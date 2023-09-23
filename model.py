import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import settings


class Head(nn.Module):
    """Single head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(settings["n_embd"], head_size, bias=False)
        self.query = nn.Linear(settings["n_embd"], head_size, bias=False)
        self.value = nn.Linear(settings["n_embd"], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(settings["block_size"], settings["block_size"])))
        self.dropout = nn.Dropout(settings["dropout"])

    def forward(self, inputs):
        batch, block, channel = inputs.shape
        # Key, query, and value vectors
        k = self.key(inputs)
        q = self.query(inputs)
        v = self.value(inputs)

        # Compute attention scores ("affinities")
        weights = q @ k.transpose(-2,-1) * channel**-0.5
        weights = weights.masked_fill(self.tril[:block, :block] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        # Perform the weighted aggregation of the values
        out = weights @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(settings["n_head"])])
        self.projection = nn.Linear(settings["n_embd"], settings["n_embd"])
        self.dropout = nn.Dropout(settings["dropout"])

    def forward(self, input):
        output = torch.cat([head(input) for head in self.heads], dim=-1)
        output = self.dropout(self.projection(output))
        return output


class FeedFoward(nn.Module):
    """Linear layer for computation after communication"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(settings["n_embd"], 4 * settings["n_embd"]),
            nn.ReLU(),
            nn.Linear(4 * settings["n_embd"], settings["n_embd"]),
            nn.Dropout(settings["dropout"]),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self):
        super().__init__()
        head_size = settings["n_embd"] // settings["n_head"]
        self.self_attn_heads = MultiHeadAttention(head_size)
        self.feed_forward = FeedFoward()
        self.normalize1 = nn.LayerNorm(settings["n_embd"])
        self.normalize2 = nn.LayerNorm(settings["n_embd"])

    def forward(self, embeddings):
        # Residual connections are added to improve training
        embeddings = embeddings + self.self_attn_heads(self.normalize1(embeddings))
        embeddings = embeddings + self.feed_forward(self.normalize2(embeddings))
        return embeddings


class LanguageModel(nn.Module):
    """Transformer language model"""
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(settings["vocab_size"], settings["n_embd"])
        self.position_embedding_table = nn.Embedding(settings["block_size"], settings["n_embd"])
        self.blocks = nn.Sequential(*[Block() for _ in range(settings["n_layer"])])
        self.normalize = nn.LayerNorm(settings["n_embd"])
        self.model_head = nn.Linear(settings["n_embd"], settings["vocab_size"])

    def forward(self, inputs, targets=None):
        # Compute logits
        _, block = inputs.shape # In case context isn't of batch size, generate fewer positional embeddings
        token_embeddings = self.token_embedding_table(inputs)
        position_embeddings = self.position_embedding_table(torch.arange(block, device=settings["device"]))
        embeddings = token_embeddings + position_embeddings
        embeddings = self.blocks(embeddings)
        embeddings = self.normalize(embeddings)
        logits = self.model_head(embeddings)

        if targets is None:
            loss = None
        else:
            # Reshape data, calculate loss
            batch, block, channel = logits.shape
            logits = logits.view(batch*block, channel)
            targets = targets.view(batch*block)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, n_tokens, context=torch.zeros((1, 1), dtype=torch.long, device=settings["device"])):
        for _ in range(n_tokens):
            # Crop context to get last block_size of tokens
            context_cropped = context[:, -settings["block_size"]:]
            # Generate predictions
            logits, _ = self(context_cropped)
            # Grab highest probabilty generation
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            generation = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, generation), dim=1)
        return context

from glob import glob
import os
import torch
from torch import nn
from math import sin, cos
import torch.nn.functional as F
import numpy as np

from data.bpe_tokenizer import encode, decode, load_shard, load_tokenizer, TOKENIZER_PATH

class Transformer(nn.Module):
    def __init__(self, vocab_size, context_len = 1024, batch_size = 32, embedding_dim = 512):
        super().__init__()
        self.context_len = context_len
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        self.input_embedding = nn.Embedding(self.vocab_size, self.embedding_dim) # trainable
        self.pos_embedding = torch.zeros(self.context_len, self.embedding_dim)
        
        for pos in range(self.context_len):
            for i in range (0, self.embedding_dim, 2):
                angle = pos / (10000 ** (i / self.embedding_dim))
                self.pos_embedding[pos][i] = sin(angle)
                self.pos_embedding[pos][i + 1] = cos(angle)
                
        self.pos_embedding = self.pos_embedding.unsqueeze(0)
                
        self.Q = nn.Linear(self.embedding_dim, self.embedding_dim) # trainable
        self.K = nn.Linear(self.embedding_dim, self.embedding_dim) # trainable
        self.V = nn.Linear(self.embedding_dim, self.embedding_dim) # trainable
        
        self.causal_mask = torch.triu(torch.ones(self.context_len, self.context_len), diagonal=1).bool()
        self.causal_mask = self.causal_mask.unsqueeze(0)
        
        self.nn1 = nn.Linear(self.embedding_dim, self.embedding_dim) # trainable
        self.norm = nn.LayerNorm(self.embedding_dim) # trainable
        self.nn2 = nn.Linear(self.embedding_dim, self.vocab_size) # trainable
        
        
    def forward(self, x):
        # B, C, E
        inputs = self.input_embedding(x) + self.pos_embedding
        
        # B, C, E
        q = self.Q(inputs)
        k = self.K(inputs)
        v = self.V(inputs)
        
        attention_scores = q @ k.transpose(-2, -1) / torch.sqrt(torch.tensor(self.embedding_dim, dtype=q.dtype))
        attention_scores = attention_scores.masked_fill(self.causal_mask, float('-inf'))
        # B, C, C
        attention_scores = torch.softmax(attention_scores, dim=-1)
        
        # B, C, E
        logits = attention_scores @ v 
        logits = self.nn1(logits)
        
        logits = logits + inputs
        logits = self.norm(logits)
        logits = self.nn2(logits)
        
        logits = torch.softmax(logits, dim=-1)
        
        return logits

def batch_text(context_len, batch_size, merges, vocab):
   tokens = np.concatenate([load_shard(f) for f in sorted(glob.glob("data/bpe-shards/*.bin"))])
   for step in range(num_steps):
        start = step * batch_size * context_len
        batch = tokens[start:start + batch_size * context_len]
        x = torch.tensor(batch.reshape(batch_size, context_len), dtype=torch.long)
        yield x


context_len = 64
batch_size = 32
num_steps = 5000

merges, vocab = load_tokenizer(TOKENIZER_PATH)
transformer = Transformer(len(vocab), context_len, batch_size)

optimizer = torch.optim.Adam(transformer.parameters(), lr=3e-4)

for step in range(num_steps):
    # B, C
    x = torch.tensor(next(batch_text(context_len, batch_size, merges, vocab)))
    logits = transformer(x)
    y = x[:, 1:]
    logits_shifted = logits[:, :-1, :]
    B, T, V = logits_shifted.shape
    loss = F.cross_entropy(
        logits_shifted.reshape(B*T, V),
        y.reshape(B*T)
    )
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("loss:", float(loss))
    # break
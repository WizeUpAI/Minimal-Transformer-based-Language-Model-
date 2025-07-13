import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple character tokenizer
class CharTokenizer:
    def __init__(self, chars="abcdefghijklmnopqrstuvwxyz "):
        self.chars = chars
        self.stoi = {ch: i+1 for i, ch in enumerate(chars)}  # 0 = padding
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, text, max_len):
        text = text.lower()
        arr = [self.stoi.get(ch, 0) for ch in text[:max_len]]
        if len(arr) < max_len:
            arr += [0] * (max_len - len(arr))
        return torch.tensor(arr, dtype=torch.long)

    def decode(self, indices):
        return "".join(self.itos.get(i, "?") for i in indices)

    @property
    def vocab_size(self):
        return len(self.stoi) + 1  # plus padding 0

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

# Multi-head Self-Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv_proj(x)  # (B,T,3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, T, T)
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)
        return out

# Transformer Encoder Layer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, embed_dim),
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.ln1(x)
        x = x + self.dropout(self.self_attn(x2))
        x2 = self.ln2(x)
        x = x + self.dropout(self.mlp(x2))
        return x

# Full Transformer Encoder Model for language modeling
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=2, mlp_dim=256, max_len=64):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)
        self.layers = nn.Sequential(*[
            TransformerEncoderLayer(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        x = self.token_emb(idx)       # (B,T,C)
        x = self.pos_enc(x)           # (B,T,C)
        x = self.layers(x)            # (B,T,C)
        x = self.ln_f(x)              # (B,T,C)
        logits = self.head(x)         # (B,T,vocab_size)
        return logits
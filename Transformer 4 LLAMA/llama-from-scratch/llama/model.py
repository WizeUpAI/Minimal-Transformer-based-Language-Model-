
import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return self.weight * x / (norm + self.eps)

class RotaryEmbedding:
    def __init__(self, dim):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.inv_freq = inv_freq

    def get_embed(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb

def apply_rotary(x, rope_embed):
    x1, x2 = x[..., ::2], x[..., 1::2]
    sin, cos = rope_embed[..., ::2], rope_embed[..., 1::2]
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        rope_embed = self.rope.get_embed(T, x.device)
        q = apply_rotary(q, rope_embed)
        k = apply_rotary(k, rope_embed)

        attn_weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(attn_weights, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        x_proj, x_gate = self.linear1(x).chunk(2, dim=-1)
        return self.linear2(torch.nn.functional.silu(x_gate) * x_proj)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadAttention(dim, n_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden_dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class LLaMAModel(nn.Module):
    def __init__(self, vocab_size, dim=256, n_heads=4, n_layers=4, hidden_dim=1024, max_seq_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, n_heads, hidden_dim) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_emb(x)
        x = self.blocks(x, mask)
        x = self.norm(x)
        return self.head(x)

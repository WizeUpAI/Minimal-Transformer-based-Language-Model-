import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, queries, mask):
        N, seq_len, embed_size = queries.shape

        # Split embedding into heads
        values = self.values(values).view(N, seq_len, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, seq_len, self.heads, self.head_dim)
        queries = self.queries(queries).view(N, seq_len, self.heads, self.head_dim)

        # Transpose for attention calculation: (N, heads, seq_len, head_dim)
        values = values.transpose(1, 2)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)

        # Compute scaled dot-product attention
        energy = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))

        attention = torch.softmax(energy, dim=-1)

        out = torch.matmul(attention, values)  # (N, heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(N, seq_len, embed_size)

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_len, embed_size)

    def forward(self, x):
        N, seq_len, _ = x.shape
        positions = torch.arange(0, seq_len).unsqueeze(0).expand(N, seq_len).to(x.device)
        pos_embed = self.positional_embedding(positions)
        return x + pos_embed

class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, num_layers=4, heads=8, dropout=0.1, forward_expansion=4, max_length=100):
        super().__init__()
        self.embed_size = embed_size
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = PositionalEncoding(embed_size, max_length)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_size, heads, dropout, forward_expansion) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        out = self.token_embedding(x)  # (N, seq_length, embed_size)
        out = self.positional_encoding(out)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        logits = self.fc_out(out)  # (N, seq_length, vocab_size)
        return logits

def generate_causal_mask(seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len)).bool()  # Lower triangular mask
    return mask

if __name__ == "__main__":
    # Toy example usage
    vocab_size = 10000
    batch_size = 2
    seq_len = 10

    model = GPTLikeModel(vocab_size)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_len))

    mask = generate_causal_mask(seq_len).to(inputs.device)  # Causal mask
    logits = model(inputs, mask)
    print(logits.shape)  # Expected: (batch_size, seq_len, vocab_size)

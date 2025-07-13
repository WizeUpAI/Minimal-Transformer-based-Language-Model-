
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ======= MODEL =======
class GPTLikeModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x, mask)
        return self.lm_head(x)

def generate_causal_mask(seq_len):
    return torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0)

# ======= TOKENIZER & DATASET =======
class Tokenizer:
    def __init__(self, texts, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.build_vocab(texts)

    def build_vocab(self, texts):
        from collections import Counter
        counter = Counter()
        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)

        most_common = counter.most_common(self.vocab_size - 2)
        for idx, (word, _) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

    def encode(self, text):
        tokens = text.lower().split()
        return [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]

    def decode(self, token_ids):
        return " ".join(self.idx2word.get(i, "<UNK>") for i in token_ids)

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=20):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        tokenized = []
        for text in texts:
            tokenized.extend(tokenizer.encode(text))

        self.data = []
        for i in range(0, len(tokenized) - seq_len):
            seq = tokenized[i:i + seq_len + 1]
            if len(seq) == seq_len + 1:
                self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

# ======= TRAINING LOOP =======
def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(data_loader):
        batch = batch.to(device)
        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        seq_len = inputs.size(1)
        mask = generate_causal_mask(seq_len).to(device)

        optimizer.zero_grad()
        outputs = model(inputs, mask)
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    return total_loss / len(data_loader)

# ======= MAIN =======
if __name__ == "__main__":
    corpus = [
        "Hello, how are you doing today?",
        "I hope you are enjoying learning about transformers.",
        "This is a simple example of training a language model.",
        "PyTorch makes it easy to build neural networks.",
        "Let's generate some text after training!"
    ]

    tokenizer = Tokenizer(corpus, vocab_size=1000)
    dataset = TextDataset(corpus, tokenizer, seq_len=20)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTLikeModel(vocab_size=len(tokenizer.word2idx)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        avg_loss = train(model, data_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}/{epochs} - Average Loss: {avg_loss:.4f}")

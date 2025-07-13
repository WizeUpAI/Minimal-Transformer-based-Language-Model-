import torch
import torch.nn as nn
import torch.optim as optim

# Assuming GPTLikeModel and generate_causal_mask from previous code are defined/imported here

def train(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        # batch shape: (batch_size, seq_len)
        batch = batch.to(device)

        optimizer.zero_grad()

        # Inputs and targets for next-token prediction
        inputs = batch[:, :-1]       # All tokens except last
        targets = batch[:, 1:]       # All tokens except first (shifted)

        seq_len = inputs.size(1)
        mask = generate_causal_mask(seq_len).to(device)

        outputs = model(inputs, mask)  # (batch, seq_len, vocab_size)

        # Flatten outputs and targets to compute cross entropy
        outputs = outputs.reshape(-1, outputs.size(-1))  # (batch*seq_len, vocab_size)
        targets = targets.reshape(-1)                     # (batch*seq_len)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)

# Example toy dataset loader
class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=1000, seq_len=20, vocab_size=10000):
        super().__init__()
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 10000
    model = GPTLikeModel(vocab_size).to(device)

    dataset = ToyDataset(num_samples=2000, seq_len=20, vocab_size=vocab_size)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 5
    for epoch in range(epochs):
        avg_loss = train(model, data_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")

##____________________ Preapred Real Text ____________________

def generate_text(model, tokenizer, start_text, max_length=30, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(start_text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # (1, seq_len)

    for _ in range(max_length):
        seq_len = tokens.size(1)
        mask = generate_causal_mask(seq_len).to(device)

        with torch.no_grad():
            outputs = model(tokens, mask)  # (1, seq_len, vocab_size)
        next_token_logits = outputs[0, -1, :]
        next_token = torch.argmax(next_token_logits).unsqueeze(0)

        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    generated = tokens[0].tolist()
    print("Generated text:")
    print(tokenizer.decode(generated))

## Use the GPTLikeModel and training loop code from earlier here
if __name__ == "__main__":
    # Example small corpus
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        avg_loss = train(model, data_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs} - Average Loss: {avg_loss:.4f}")
    
    # After training
    generate_text(model, tokenizer, "Hello")

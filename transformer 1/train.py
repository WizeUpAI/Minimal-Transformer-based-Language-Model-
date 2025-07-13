import torch
from torch.utils.data import Dataset, DataLoader
from minimal_transformer import CharTokenizer, TransformerLanguageModel
import torch.nn.functional as F

# Dummy dataset for language modeling
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=64):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode(self.texts[idx], self.max_len)
        return enc

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # input tokens shifted for next-token prediction
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        logits = model(input_ids)  # (B,T-1,vocab_size)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.reshape(-1), ignore_index=0)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            logits = model(input_ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.reshape(-1), ignore_index=0)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()

    train_texts = [
        "hello world this is a test",
        "transformers are awesome",
        "we implement from scratch",
        "language modeling is fun",
        "minimal transformer model"
    ] * 100

    val_texts = [
        "hello transformer",
        "language model test"
    ]

    train_dataset = TextDataset(train_texts, tokenizer)
    val_dataset = TextDataset(val_texts, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    model = TransformerLanguageModel(vocab_size=tokenizer.vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "minimal_transformer.pt")

if __name__ == "__main__":
    main()
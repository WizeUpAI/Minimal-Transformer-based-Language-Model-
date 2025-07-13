
import torch
from torch.nn import functional as F
from llama.model import LLaMAModel
from llama.tokenizer import load_tokenizer
from dataset import get_dataloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = load_tokenizer()

vocab_size = tokenizer.get_vocab_size()
model = LLaMAModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

train_loader = get_dataloader(tokenizer, split='train')

for epoch in range(3):
    model.train()
    for batch in train_loader:
        x = batch['input_ids'].to(device)
        y = batch['labels'].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

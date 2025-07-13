
import torch

def generate_text(model, tokenizer, start_text, max_length=30, device='cpu'):
    model.eval()
    tokens = tokenizer.encode(start_text)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    for _ in range(max_length):
        seq_len = tokens.size(1)
        mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tokens, mask)
        next_token_logits = outputs[0, -1, :]
        next_token = torch.argmax(next_token_logits).unsqueeze(0)
        tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

    generated = tokens[0].tolist()
    print("Generated text:")
    print(tokenizer.decode(generated))

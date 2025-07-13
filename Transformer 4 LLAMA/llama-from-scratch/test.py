
import torch
from llama.model import LLaMAModel
from llama.tokenizer import load_tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = load_tokenizer()
vocab_size = tokenizer.get_vocab_size()
model = LLaMAModel(vocab_size).to(device)
model.load_state_dict(torch.load("llama.pt"))
model.eval()

def generate(prompt, max_len=50):
    ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor(ids, device=device).unsqueeze(0)
    for _ in range(max_len):
        logits = model(input_ids)
        next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(input_ids[0].tolist())

print(generate("Once upon a time"))

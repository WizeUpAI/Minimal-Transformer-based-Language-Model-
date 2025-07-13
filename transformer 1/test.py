import torch
from minimal_transformer import CharTokenizer, TransformerLanguageModel

def generate(model, tokenizer, prompt, max_len=64):
    model.eval()
    input_ids = tokenizer.encode(prompt, max_len=max_len)
    input_ids = input_ids.unsqueeze(0)  # batch 1
    generated = input_ids.clone()

    with torch.no_grad():
        for _ in range(max_len - input_ids.size(1)):
            logits = model(generated)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits)
            if next_token == 0:
                break
            generated = torch.cat([generated, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
    decoded = tokenizer.decode(generated[0].tolist())
    return decoded

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = CharTokenizer()
    model = TransformerLanguageModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load("minimal_transformer.pt", map_location=device))

    prompt = "hello"
    print("Prompt:", prompt)
    output = generate(model, tokenizer, prompt)
    print("Generated:", output)

if __name__ == "__main__":
    main()
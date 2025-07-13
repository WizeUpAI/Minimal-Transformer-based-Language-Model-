from bert_llm import MiniBERT, SimpleTokenizer
import torch

tokenizer = SimpleTokenizer()
text = "hello world this is a test"
input_ids = torch.tensor([tokenizer.encode(text)])

model = MiniBERT(vocab_size=len(tokenizer.vocab))
outputs = model(input_ids)
print(outputs.shape)  # (batch_size, seq_len, embed_dim)

import torch
from torch.utils.data import Dataset, DataLoader

# Simple whitespace tokenizer + vocab builder
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

        # Tokenize all texts into one big list
        tokenized = []
        for text in texts:
            tokenized.extend(tokenizer.encode(text))

        # Split into sequences of seq_len + 1 (input + target)
        self.data = []
        for i in range(0, len(tokenized) - seq_len):
            seq = tokenized[i:i + seq_len + 1]
            if len(seq) == seq_len + 1:
                self.data.append(seq)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)

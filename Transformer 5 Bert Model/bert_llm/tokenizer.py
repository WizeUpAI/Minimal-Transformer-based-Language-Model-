class SimpleTokenizer:
    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1}
        self.id2word = {0: "[PAD]", 1: "[UNK]"}

    def tokenize(self, text):
        return text.strip().split()

    def encode(self, text):
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token not in self.vocab:
                idx = len(self.vocab)
                self.vocab[token] = idx
                self.id2word[idx] = token
            ids.append(self.vocab[token])
        return ids

    def decode(self, ids):
        return " ".join(self.id2word.get(i, "[UNK]") for i in ids)
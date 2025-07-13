
from tokenizers import ByteLevelBPETokenizer

def train_tokenizer(corpus_files, vocab_size=5000, save_path="./tokenizer"):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=corpus_files, vocab_size=vocab_size, min_frequency=2)
    tokenizer.save_model(save_path)
    return tokenizer

def load_tokenizer(path="./tokenizer"):
    return ByteLevelBPETokenizer(
        f"{path}/vocab.json", f"{path}/merges.txt"
    )

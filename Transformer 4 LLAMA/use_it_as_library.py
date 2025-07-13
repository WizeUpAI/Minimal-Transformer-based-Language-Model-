from llama.model import LLaMAModel
import torch

# Paramètres
vocab_size = 1000  # Exemple arbitraire
seq_len = 32
batch_size = 2

# Instancier le modèle
model = LLaMAModel(vocab_size)

# Exemple de données factices
x = torch.randint(0, vocab_size, (batch_size, seq_len))

# Passage avant
logits = model(x)

# Résultat
print("Shape des logits:", logits.shape)  # [batch_size, seq_len, vocab_size]

"""
Self-attention from scratch, with NumPy only.

Companion to Modules 5-6 of "Transformers from Scratch".
Run it:   pip install numpy   then   python attention_from_scratch.py

The numbers are random (an untrained model), so the attention pattern you see is
meaningless until training. The point is to see every step of the maths run.
"""
import numpy as np

np.random.seed(0)

# Module 2: tokens. (A real tokenizer would give sub-word IDs; whole words keep it readable.)
tokens = "The nurse gave the patient her medicine because she had a fever".split()
vocab = {word: i for i, word in enumerate(sorted(set(tokens)))}
ids = [vocab[t] for t in tokens]
print("Token IDs:", ids)

d_model = 8   # numbers per token (real models: 768 to 12,288)
d_head = 4    # numbers per query/key/value

# Module 3: embeddings. One learned row per vocabulary entry.
embedding_table = np.random.randn(len(vocab), d_model)
x = embedding_table[ids]                       # shape: (12 tokens, 8)

# Module 4: sinusoidal positional encoding, added to the embeddings.
def positional_encoding(n_positions, d):
    pos = np.arange(n_positions)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))

x = x + positional_encoding(len(tokens), d_model)

# Module 6: three learned tables turn each vector into a query, key and value.
W_q = np.random.randn(d_model, d_head)
W_k = np.random.randn(d_model, d_head)
W_v = np.random.randn(d_model, d_head)
Q, K, V = x @ W_q, x @ W_k, x @ W_v           # each: (12, 4)

def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)      # for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

# Attention = softmax(Q K^T / sqrt(d)) V
scores = Q @ K.T / np.sqrt(d_head)             # (12, 12): every token vs every token

# Module 9: causal mask. Set "future" scores to -infinity so they get 0 weight.
causal = True
if causal:
    future = np.triu(np.ones_like(scores, dtype=bool), k=1)
    scores = np.where(future, -np.inf, scores)

weights = softmax(scores)                      # each row adds up to 1
output = weights @ V                           # (12, 4): new vector per token

she = tokens.index("she")
print(f"\nWhere does '{tokens[she]}' look? (untrained, so the pattern is random)")
for tok, w in sorted(zip(tokens, weights[she]), key=lambda p: -p[1])[:5]:
    print(f"  {tok:>9}  {w:6.1%}  {'#' * int(w * 40)}")
print("\nEach row of weights sums to:", np.round(weights.sum(axis=1), 3))
print("Output shape:", output.shape)

# ---------------------------------------------------------------------------
# Exercises
# 1. Set causal = False. Which tokens can "she" see now?
# 2. Make d_head bigger and remove the / np.sqrt(d_head). Print weights[she].
#    Do the weights get more "all-or-nothing"? (Module 6, quiz question 2.)
# 3. Build two more heads (new W_q, W_k, W_v) and np.concatenate their outputs.
#    That is multi-head attention (Module 7).
# 4. Hand-set W_q and W_k so that "she" attends mostly to "patient".
#    Hint: you need the query of "she" to point the same way as the key of "patient".
# ---------------------------------------------------------------------------

# Transformers from Scratch

An interactive, beginner-friendly course on the **transformer**, the architecture behind large language models such as ChatGPT, Claude, Gemini and Llama.

**Open the course:** https://rwunganira.github.io/transformers-course/

The course follows one sentence, *"The nurse gave the patient her medicine because she had a fever"*, as it travels through a transformer, one station per module. You only need to know multiplication. Every module has the same four parts:

1. **Idea**: the concept in plain words, with an everyday comparison
2. **Try it**: a small working model with sliders and buttons
3. **Remember**: flip-cards for the key terms
4. **Check**: two questions (get both right to complete the module)

## Modules

| # | Module | You'll learn | Hands-on lab |
|---|--------|--------------|--------------|
| 1 | A very good guesser | An LLM predicts the next token, over and over | Next-word probabilities and a temperature slider |
| 2 | Chopping text into tokens | Tokens, vocabularies, sub-word pieces | A toy tokenizer you can type into |
| 3 | Words as points on a map | Embeddings and meaning as direction | Clickable 2-D word map, king − man + woman |
| 4 | Keeping track of word order | Why attention needs positional encoding | Bag-of-words demo and a sine-wave fingerprint heatmap |
| 5 | Attention: who should I listen to? | The core idea of the transformer | Click a word and see where it looks; change the context |
| 6 | Query, Key and Value | The attention maths, step by step | Steer a query and watch dot products, softmax and the blend |
| 7 | Many heads, many layers | Multi-head attention, the transformer block, stacking | Three heads compared; step through one block |
| 8 | Training: learning from surprise | Loss, gradient descent, learning rate, pre-training | Loss calculator; roll a ball down a two-valley loss curve |
| 9 | Generating text, one token at a time | Autoregressive generation, causal mask, hallucination | Generate a sentence and watch the mask grow |

The page ends with a recap, a glossary and suggested next steps.

## Optional coding exercise

[`exercises/attention_from_scratch.py`](exercises/attention_from_scratch.py) builds self-attention in about 60 lines of NumPy (tokens → embeddings → position → Q, K, V → softmax → output, with a causal mask) and runs it on the course sentence. Four short exercises are listed at the bottom of the file.

```bash
pip install numpy
python exercises/attention_from_scratch.py
```

## Notes

- The whole course is one self-contained `index.html` file with no build step. Open it locally or through GitHub Pages.
- Progress is saved in your browser (`localStorage`).
- The attention weights in the labs are set by hand to show the ideas clearly. They are not taken from a real trained model.

## Further reading

- Jay Alammar, [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- 3Blue1Brown, [Neural networks series](https://www.3blue1brown.com/topics/neural-networks)
- Andrej Karpathy, [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Vaswani et al. (2017), [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

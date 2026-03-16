# Transformer Architecture

## Introduction

The Transformer is a deep learning model architecture introduced in the 2017 paper
"Attention Is All You Need" by Vaswani et al. at Google Brain. It has become the
foundation for most modern large language models including GPT, BERT, and Gemini.

Unlike earlier sequence models such as RNNs and LSTMs, the Transformer processes
all tokens in parallel rather than sequentially, making it much faster to train
on modern hardware.

## Self-Attention

The core innovation of the Transformer is the self-attention mechanism. For each
token in a sequence, self-attention computes a weighted sum of all other tokens,
where the weights represent how relevant each token is to the current one.

This is computed using three learned matrices:
- Query (Q): what the current token is looking for
- Key (K): what each token has to offer
- Value (V): the actual content each token contributes

The attention score between two tokens is: softmax(QK^T / sqrt(d_k)) * V

## Multi-Head Attention

Rather than applying attention once, the Transformer applies it multiple times
in parallel — each instance is called an "attention head". Each head can attend
to different aspects of the input.

The outputs of all heads are concatenated and projected back to the model dimension.
This allows the model to jointly attend to information from different positions and
representation subspaces.

## Feed-Forward Network

After the attention layer, each position passes through a feed-forward network
independently. This consists of two linear transformations with a ReLU activation:

FFN(x) = max(0, xW1 + b1)W2 + b2

## Positional Encoding

Since the Transformer processes all tokens in parallel, it has no inherent notion
of position. Positional encodings are added to the input embeddings to give the
model information about token order.

The original paper uses sinusoidal positional encodings, though learned positional
embeddings are also common in modern models.

## Encoder and Decoder

The original Transformer has two parts:

**Encoder**: Processes the input sequence. Each layer has self-attention followed
by a feed-forward network. Used in models like BERT.

**Decoder**: Generates the output sequence one token at a time. Has an additional
cross-attention layer that attends to the encoder output. Used in models like GPT
(decoder-only variant).

## Applications

Transformers power virtually all state-of-the-art NLP systems:
- Language models: GPT-4, Gemini, Claude, LLaMA
- Text classification: BERT, RoBERTa
- Machine translation: Google Translate
- Code generation: GitHub Copilot, Claude
- Image generation: DALL-E, Stable Diffusion (uses a variant)

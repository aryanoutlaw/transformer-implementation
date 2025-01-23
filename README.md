# Transformers Implementation

This repository contains a PyTorch implementation of the Transformer model, as described in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. The implementation includes the core components of the Transformer architecture, such as multi-head attention, positional encoding, and the encoder-decoder structure. The code is designed to be modular and easy to understand, making it a good starting point for learning about Transformers or for building custom models based on this architecture.

## Repository Structure

The repository contains the following files:

- **`config.py`**: Contains configuration settings for the Transformer model, including hyperparameters like batch size, learning rate, and sequence length. It also provides utility functions for managing model weights.

- **`dataset.py`**: Implements a `BilingualDataset` class for handling bilingual text data. This class is responsible for tokenizing the input text, adding special tokens (e.g., SOS, EOS, PAD), and generating masks for the encoder and decoder.

- **`model.py`**: Contains the implementation of the Transformer model, including all the necessary components such as multi-head attention, feed-forward layers, positional encoding, and the encoder-decoder architecture. The `build_transformer` function is provided to easily construct a Transformer model with customizable parameters.

## Key Components

### 1. **Multi-Head Attention**
   - The `MultiHeadAttentionBlock` class implements the multi-head attention mechanism, which allows the model to focus on different parts of the input sequence simultaneously. This is a key component of the Transformer architecture.

### 2. **Positional Encoding**
   - The `PositionalEncoding` class adds positional information to the input embeddings, allowing the model to take into account the order of the tokens in the sequence.

### 3. **Encoder and Decoder**
   - The `Encoder` and `Decoder` classes implement the encoder and decoder stacks, respectively. Each stack consists of multiple layers of self-attention and feed-forward networks.

### 4. **Bilingual Dataset**
   - The `BilingualDataset` class handles the preprocessing of bilingual text data, including tokenization, padding, and the generation of attention masks.

### 5. **Transformer Model**
   - The `Transformer` class brings together all the components to form the complete Transformer model. It includes methods for encoding, decoding, and projecting the output to the target vocabulary.

## Usage

To use this implementation, you can start by configuring the model in `config.py`. Then, you can build the Transformer model using the `build_transformer` function in `model.py`. The `BilingualDataset` class in `dataset.py` can be used to prepare your data for training.

### Example: Building a Transformer Model

```python
from model import build_transformer

# Define the model parameters
src_vocab_size = 10000
tgt_vocab_size = 10000
src_seq_len = 350
tgt_seq_len = 350
d_model = 512
N = 6  # Number of encoder/decoder layers
h = 8  # Number of attention heads
dropout = 0.1
d_ff = 2048  # Dimension of the feed-forward network

# Build the Transformer model
transformer = build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model, N, h, dropout, d_ff)
```

### Example: Preparing Data with BilingualDataset

```python
from dataset import BilingualDataset
from torch.utils.data import DataLoader

# Assuming you have a dataset `ds` and tokenizers `tokenizer_src` and `tokenizer_tgt`
dataset = BilingualDataset(ds, tokenizer_src, tokenizer_tgt, src_lang="en", tgt_lang="it", seq_len=350)

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Configuration

The `config.py` file allows you to customize various aspects of the model, such as:

- **Batch size**: The number of samples processed in each batch.
- **Number of epochs**: The total number of training epochs.
- **Learning rate**: The learning rate for the optimizer.
- **Sequence length**: The maximum length of the input and output sequences.
- **Model dimensions**: The dimensionality of the model (e.g., `d_model`).
- **Source and target languages**: The languages used for translation.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

---

This implementation is designed to be a clear and concise reference for understanding and working with Transformer models. Feel free to explore the code and adapt it to your own projects!

# miniGPT: A Minimal GPT-Style Transformer in PyTorch

This repository contains a *minimalist* implementation of a GPT-style Transformer model (nicknamed **miniGPT**) for text generation, fine-tuned on the [OpenWebText](https://huggingface.co/datasets/openwebtext) dataset. Despite its compact design, **miniGPT** showcases many of the key features found in modern Transformers, such as multi-head self-attention with causal masking, positional embeddings, and a scalable architecture configurable for different embedding sizes, numbers of heads, and Transformer blocks.

---

## Features

1. **Transformer-Based Language Model**  
   - Implements a custom `SelfAttention` layer with causal masking to enable autoregressive text generation.  
   - Uses residual connections, layer normalization, and feed-forward sublayers typical of GPT-like architectures.

2. **Positional Embeddings**  
   - Combines token embeddings with learnable positional embeddings, critical for sequence order awareness.

3. **Configurable Architecture**  
   - Control major hyperparameters, including:
     - *Sequence length (`seq_len`)*
     - *Embedding dimension (`embed_dim`)*
     - *Number of attention heads (`num_heads`)*
     - *Number of Transformer blocks (`n_layers`)*
     - *Dropout rate and more…*

4. **Training & Generation Modes**  
   - **`mode='train'`**: Train a new miniGPT model from scratch on a portion of the OpenWebText dataset.
   - **`mode='generate'`**: Load a previously trained model to sample generated text from a given initial prompt.

5. **Gradient Accumulation & Early Stopping**  
   - Simulates larger batch training with gradient accumulation (`gradient_accumulation_steps`).
   - Implements early stopping based on validation loss to avoid overfitting.

6. **Data Subsetting**  
   - Randomly samples a portion of training and validation data each epoch for quick experiments.  
   - Adjust `train_subset_size` and `valid_subset_size` to scale up or down.

7. **GPU & Multi-GPU Support**  
   - Automatically detects CUDA availability.  
   - If multiple GPUs are present, wraps the model in `nn.DataParallel` to parallelize training.

---

## Requirements

- **Python 3.7+**
- **PyTorch 1.10+** (GPU recommended)
- **Transformers** (Hugging Face)
- **Datasets** (Hugging Face)
- **NumPy**, **random**, etc.

Install the core dependencies via:
```bash
pip install torch transformers datasets

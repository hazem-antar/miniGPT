# miniGPT: A Minimal Transformer Language Model

This repository provides a concise, **easy-to-understand** implementation of a GPT-like model in PyTorch. It features a complete pipeline for **training** on large-scale text (OpenWebText) and **generating** new text samples, all with a modest computational footprint. Despite its minimalism, this code integrates several key Transformer components—embeddings, multi-head self-attention, feed-forward layers, and a causal mask—to illustrate the essence of GPT-style models.

> **Note**: This code serves as a starting point for research into more advanced or specialized approaches, including experimenting with **Hyperdimensional Computing (HDC)** mechanisms as a drop-in replacement for self-attention.

---

## 🚀 Key Features

- **Minimal Implementation**: A single Python script (`miniGPT.py`) that clearly illustrates each part of a GPT model—embedding, Transformer blocks, and final output layer.
- **Causal Self-Attention**: Implements a *mask* to prevent attention to future tokens, enabling left-to-right generative modeling.
- **Flexible Training**:
  - Gradient accumulation to simulate large batch sizes on smaller GPUs.
  - Early stopping and patience-based validation to curb overfitting.
  - Uses **OpenWebText** dataset by default, but easily adaptable to other text corpora.
- **Text Generation Mode**:
  - Top-k sampling and temperature scaling for more diverse, creative text outputs.
  - Configurable sequence length, sampling size, and initial prompt.
- **Scalable to Multi-GPU**: Automatically uses `nn.DataParallel` if multiple GPUs are available.

---

## 📥 Installation

1. **Clone** this repository:
   ```bash
   git clone https://github.com/hazem-antar/miniGPT.git
   cd miniGPT

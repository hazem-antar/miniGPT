miniGPT: Exploring Hyperdimensional Computing (HDC) in Transformer Attention

This repository presents a *minimal* GPT-style Transformer—dubbed miniGPT—designed to test and evaluate the integration of Hyperdimensional Computing (HDC) into the self-attention mechanism. By marrying HDC’s high-dimensional vector representations with a lightweight GPT architecture, this project aims to reduce computational overhead while maintaining strong language modeling capabilities.

---

KEY HIGHLIGHTS

1. HDC-Infused Attention Layer
   Investigates how Hyperdimensional Computing can be leveraged to optimize attention, potentially offering more efficient and robust learning compared to traditional self-attention alone.

2. Minimal GPT Framework
   Implements essential GPT features (causal masking, multi-head attention, residual connections, layer norms) in under a thousand lines of code, making it easy to follow and experiment with.

3. Scalable & Configurable
   Control critical hyperparameters (sequence length, embedding dimension, number of heads, Transformer blocks, etc.) to tailor resource usage and model capacity.

4. Practical NLP Pipeline
   Utilizes the GPT-2 tokenizer, trains on the Hugging Face OpenWebText dataset, and includes support for text generation with temperature scaling and top-k sampling.

5. Gradient Accumulation & Early Stopping
   Simulates larger batch sizes while avoiding overfitting via a built-in early-stopping mechanism.

---

REQUIREMENTS

- Python 3.7+
- PyTorch 1.10+ (GPU recommended)
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- NumPy, random, etc.

Install dependencies:
pip install torch transformers datasets

Refer to the PyTorch website (https://pytorch.org/) for the installation command suited to your environment.

---

QUICKSTART

1. Clone the Repo
   git clone https://github.com/hazem-antar/miniGPT.git
   cd miniGPT

2. Customize Hyperparameters
   - Open miniGPT.py
   - Modify values such as:
     seq_len (maximum sequence length)
     embed_dim (embedding dimension)
     num_heads (attention heads)
     n_layers (Transformer blocks)
     lr (learning rate), epochs, and more…

3. Train the Model
   - Ensure mode='train' in miniGPT.py
   - Run:
     python miniGPT.py
   - This downloads a subset of the OpenWebText dataset and trains miniGPT, saving the best model to best_miniature_gpt_model.pth.

4. Generate Text
   - Switch to mode='generate' in miniGPT.py
   - Run:
     python miniGPT.py
   - Provides sample text completions from the model using temperature and top-k sampling.

---

NOTABLE FEATURES

- HDC-Based Attention (Experimental)
  The SelfAttention class is designed to be adaptable for Hyperdimensional Computing approaches. Modify or extend it to incorporate high-dimensional embeddings and measure the effects on performance.

- Flexible Data Subsets
  Randomly samples subsets from training and validation sets each epoch (train_subset_size, valid_subset_size), facilitating rapid prototyping.

- Multiple GPU Support
  Automatically wraps the model in nn.DataParallel if more than one GPU is available.

- Causal Masking
  Implements GPT-style left-to-right self-attention, ensuring no future context leaks into the prediction of the current token.

---

EXAMPLE USE CASES

- HDC Performance Benchmarks
  Run ablation studies comparing standard attention vs. HDC-enhanced attention on training speed, memory footprint, and perplexity.

- Custom Datasets
  Swap out OpenWebText for domain-specific data sets to see how HDC-based attention scales for specialized tasks.

- Generation Quality
  Experiment with different sampling strategies (temperature, top_k) to find the best trade-offs between diversity and coherence in text generation.

---

CAVEATS & FUTURE DIRECTIONS

- Memory Footprint: Transformers are memory-intensive. Large seq_len or embed_dim may push GPU limits.
- OpenWebText: Large and diverse text data may require extensive tuning for optimal performance.
- HDC Variants: The current code can be extended to explore various ways of encoding queries, keys, and values in a high-dimensional space.

---

LICENSE

Distributed under the MIT License (./LICENSE). See the LICENSE file for more details.

---

Happy Researching!
Feel free to open issues or pull requests for collaboration, improvements, or sharing your findings on HDC-driven attention in Transformers.

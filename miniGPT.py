import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import numpy as np
import time

mode = 'train'    # 'train' for training a new model and then generate a sample, 'generate' for using a trained model to generate sample directly

initial_text = "The red fox"  # Initial text to use for generating
sample_size = 300

# Configuration parameters (Be careful with increasing parameters because GPU memory saturates very quickly)
seq_len = 512      # Maximum length of input sequences
batch_size = 32  # Batch size for training
dropout = 0.1
epochs = 200  # Number of training epochs
lr = 3e-5     # Learning rate
patience = 5  # Early stopping patience

# Configuration parameters for experimentation
embed_dim = 768  # Embedding dimension for each token
num_heads = 12  # Number of attention heads
n_layers = 16  # Number of transformer blocks

# Load the OpenWebText dataset
dataset = load_dataset("openwebtext", split={'train': 'train[:90%]', 'test': 'train[90%:]'}, trust_remote_code=True)

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Define the token for padding

# Determine the vocabulary size
vocab_size = tokenizer.vocab_size

# Function to encode text using the GPT-2 tokenizer
def encode(text, tokenizer):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # Add special tokens specific to GPT-2
        max_length=seq_len,  # Pad or truncate to maximum length
        padding='max_length',  # Pad to max length
        truncation=True,  # Truncate to max length
        return_tensors='pt',  # Return PyTorch tensors
    )
    return encoded['input_ids'].squeeze(), encoded['attention_mask'].squeeze()

# Collate function for DataLoader
def collate_batch(batch, tokenizer):
    input_ids_list, attention_mask_list, lengths = [], [], []
    for item in batch:
        _text = item['text']
        if len(_text.strip()) == 0:  # Skip empty text entries
            continue
        input_ids, attention_mask = encode(_text, tokenizer)
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        lengths.append(len(input_ids))
    input_ids_list = torch.stack(input_ids_list)
    attention_mask_list = torch.stack(attention_mask_list)
    lengths = torch.tensor(lengths, dtype=torch.int64)
    return input_ids_list, attention_mask_list, lengths

# DataLoader with custom collate function
train_iter = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_batch(x, tokenizer))
valid_iter = DataLoader(dataset['test'], batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_batch(x, tokenizer))

# Self-attention class with causal masking
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = (embed_dim // num_heads) ** -0.5

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Create a causal mask to prevent attending to future tokens
        causal_mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        if mask is not None:
            # Adjust pading mask to get combined with causal mask
            mask = mask.unsqueeze(1).unsqueeze(2).expand(B, 1, N, N)
            # Combining two masks
            combined_mask = mask & causal_mask
        else:
            combined_mask = causal_mask

        attn = attn.masked_fill(combined_mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.o_proj(out)
        return out

# Transformer block with self-attention and feed-forward network
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.att = SelfAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.att(x, mask)
        x = self.layernorm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout(ffn_output))
        return x

# Positional embedding class
class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        x = self.token_emb(x) + self.pos_emb(positions)
        return x

# Miniature GPT model class
class MiniGPT(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_heads, n_layers):
        super(MiniGPT, self).__init__()
        self.embedding = PositionalEmbedding(seq_len, vocab_size, embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(n_layers)
        ])
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for block in self.transformer_blocks:
            x = block(x, mask)
        x = self.fc(x)
        return x

# Instantiate the model, loss function, and optimizer
model = MiniGPT(seq_len, vocab_size, embed_dim, num_heads, n_layers)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.AdamW(model.parameters(), lr=lr)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nDevice:", device)

# Use DataParallel to utilize multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model.to(device)

# Print the model configuration
print("\nModel Configuration:")
print(f"Maximum Sequence Length: {seq_len}")
print(f"Embedding Dimension: {embed_dim}")
print(f"Number of Attention Heads: {num_heads}")
print(f"Number of Layers: {n_layers}")
print(f"Batch Size: {batch_size}")
print(f"Learning Rate: {lr}")
print(f"Number of Epochs: {epochs}")
print(f"Early Stopping Patience: {patience}\n")

print("\nTraining on OpenWebText dataset")
print(f"Vocabulary Size: {vocab_size}")
print(f"Number of training samples: {len(dataset['train'])}")
print(f"Number of validation samples: {len(dataset['test'])}")
print(f"Batch size: {batch_size}")
print(f"Maximum sequence length: {seq_len}\n")

best_val_loss = float('inf')
patience_counter = 0

# Create a padding mask for sequences
def create_padding_mask(seq):
    return (seq != tokenizer.pad_token_id).to(torch.bool)

def train_and_validate():
    global best_val_loss, patience_counter
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss = 0
        for x_batch, mask_batch, lengths in train_iter:
            x_batch, mask_batch = x_batch.to(device), mask_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch, mask_batch)
            targets = x_batch[:, 1:].contiguous().view(-1)
            outputs = outputs[:, :-1, :].contiguous().view(-1, vocab_size)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
            # Clear CUDA cache
            torch.cuda.empty_cache()

        train_loss /= len(train_iter.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, mask_batch, lengths in valid_iter:
                x_batch, mask_batch = x_batch.to(device), mask_batch.to(device)
                outputs = model(x_batch, mask_batch)
                targets = x_batch[:, 1:].contiguous().view(-1)
                outputs = outputs[:, :-1, :].contiguous().view(-1, vocab_size)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * x_batch.size(0)
                
                # Clear CUDA cache
                torch.cuda.empty_cache()

        val_loss /= len(valid_iter.dataset)
        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {epoch_duration:.2f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_miniature_gpt_model3.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping due to no improvement in validation loss.")
                break

        # Clear CUDA cache at the end of each epoch
        torch.cuda.empty_cache()

def generate_text(initial_text):
    # Load the best model
    model.load_state_dict(torch.load("best_miniature_gpt_model3.pth"))

    # Generate some sample text with the trained model
    model.eval()
    
    # Encode the initial text
    start_token_ids = tokenizer.encode(initial_text, add_special_tokens=False, return_tensors='pt').to(device)

    with torch.no_grad():
        generated_tokens = start_token_ids

        for _ in range(sample_size):  # Generate a sample of text of size sample_size

            # Choose the sequance of tokens to use for next prediction
            tokens_to_use = generated_tokens
            if generated_tokens.size(1) > seq_len:
                # If generated tokens exceed max seq_len, take the last seq_len tokens
                tokens_to_use = tokens_to_use[:, -seq_len:]
            
            # Pad the generated tokens to maxlen
            padded_generated_tokens = torch.cat([tokens_to_use, torch.full((1, seq_len - tokens_to_use.size(1)), tokenizer.pad_token_id, device=device, dtype=torch.long)], dim=1)
            #print(padded_generated_tokens)
            # Create the padding mask
            mask = create_padding_mask(padded_generated_tokens)

            output = model(padded_generated_tokens, mask)
            next_token = output[:, tokens_to_use.size(1) - 1, :].argmax(-1).unsqueeze(0)
            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

    generated_text = tokenizer.decode(generated_tokens.squeeze().tolist(), skip_special_tokens=True)
    print(f"\nGenerated Text:\n {generated_text}")


if __name__ == "__main__":
    
    if mode == 'train':
        train_and_validate()
        generate_text(initial_text)
    elif mode == 'generate':
        generate_text(initial_text)
    else:
        print("Invalid mode. Please choose 'train' or 'generate'.")

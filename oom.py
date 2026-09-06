import torch 
import torch.nn as nn

# ~10 Billion Parameters
# Total params ~ 12 * L * d^2
HIDDEN_DIM = 5120
NUM_HEADS = 40
NUM_LAYERS = 32
VOCAB_SIZE = 32000
SEQ_LEN = 1024

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
            nn.Linear(4*dim, dim),
        )

    def forward(self, x):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class StandardTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.layers = nn.ModuleList([
            TransformerBlock(HIDDEN_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(HIDDEN_DIM)
        self.head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))

if __name__ == "__main__":
    print("Building ~10B Parameter Transformer...")

    model = StandardTransformer()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e9:.2f} Billion")

    device = torch.device("cuda:0")
    print(f"Allocating model to {device} in FP16/BF16...")
    model = model.to(device=device, dtype=torch.bfloat16)

    print("Initializing AdamW Optimizer (allocating master weights, momentum, variance)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("Running dummy forward & backward pass...")
    
    dummy_input = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device=device)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    optimizer.step()

    print("Training step completed successfully!")
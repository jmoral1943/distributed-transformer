import os
import torch
import torch.nn as nn
import torch.distributed as dist
import functools
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# ~10 Billion Parameters
HIDDEN_DIM = 5120
NUM_HEADS = 40
NUM_LAYERS = 32
VOCAB_SIZE = 32000
SEQ_LEN = 1024

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        # Explicit projections prevent nn.MultiheadAttention FSDP deadlocks
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.ReLU(),
            nn.Linear(4*dim, dim),
        )

    def forward(self, x):
        norm_x = self.ln1(x)
        q = self.q_proj(norm_x)
        k = self.k_proj(norm_x)
        v = self.v_proj(norm_x)
        attn_out = self.out_proj(q * k * v)
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

def init_nccl_transformer():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    
    # CRITICAL: Force identical seeds across all ranks so weights match perfectly
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = rank
    torch.cuda.set_device(rank)
    print(f"Successfully initialized rank {rank} out of {world_size} using NCCL.")

    with torch.device("meta"):
        model = StandardTransformer()

    bf16_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    print(f"Initializing and sharding model layers on GPU {device} in BF16...")
    sharded_model = FSDP(
        model.to_empty(device=device).to(dtype=torch.bfloat16),
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=bf16_policy,
        param_init_fn=lambda module: module.to_empty(device=device, recurse=True).normal_(mean=0.0, std=0.02)
    )

    print("Initializing AdamW Optimizer...")
    optimizer = torch.optim.AdamW(sharded_model.parameters(), lr=1e-4)

    print("Running dummy forward & backward pass...")
    dummy_input = torch.randint(0, VOCAB_SIZE, (2, SEQ_LEN), device=device)

    for step in range(3):
        output = sharded_model(dummy_input)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Rank {rank} completed step {step}")

    print("Training step completed successfully!")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    init_nccl_transformer()
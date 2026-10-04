import os
import torch
import torch.nn as nn
import torch.distributed as dist

class MLP(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.weights_a = a
        self.gelu = nn.GELU()
        self.weights_b = b

    def forward(self, x):
        x = x @ self.weights_a
        x = self.gelu(x)
        x = x @ self.weights_b 
        dist.all_reduce(x)
        return x


class MLP_single(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.weights_a = a
        self.gelu = nn.GELU()
        self.weights_b = b

    def forward(self, x):
        x = x @ self.weights_a
        x = self.gelu(x)
        x = x @ self.weights_b
        return x

def run_tp():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    device = rank
    torch.cuda.set_device(device)
    print(f"Successfully initialized rank {rank} out of {world_size} using NCCL.")

    torch.manual_seed(42)
    
    a = torch.randn(100,100).to(device)
    b = torch.randn(100,100).to(device)

    local_a = torch.chunk(a, world_size, dim=1)[rank]
    local_b = torch.chunk(b, world_size, dim=0)[rank]

    inputs = torch.randn(100,100).to(device)
    
    mlp = MLP(local_a, local_b).to(device)

    output = mlp(inputs)
    
    print(f"Success output {rank} {output}")
    
    dist.destroy_process_group()

    return output 

def run_single_gpu():
    torch.manual_seed(42)

    device = "cuda:0"
    
    a = torch.randn(100,100).to(device)
    b = torch.randn(100,100).to(device)

    inputs = torch.randn(100,100).to(device)

    mlp = MLP_single(a, b).to(device)

    output = mlp(inputs)
    
    print(f"Success single output {output}")

    return output  


if __name__ == "__main__":
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    out1 = run_tp()
    if rank == 0:
        out2 = run_single_gpu()
        print(f"Equal tensor {torch.allclose(out1.to('cpu'), out2.to('cpu'), atol=1e-3)}")
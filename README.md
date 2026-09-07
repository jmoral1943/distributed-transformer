# Distributed 10B Parameter Transformer Training via PyTorch FSDP

## Objective
This repository demonstrates the successful distribution and initialization of a large-scale (10.4B parameter) custom Transformer architecture across a multi-GPU cluster. The project tests the physical hardware limits of a 4x GPU environment by utilizing PyTorch's Fully Sharded Data Parallel (FSDP) to bypass standard memory constraints.

## Architecture Specs
* **Parameters:** ~10.4 Billion
* **Hidden Dimension:** 5120
* **Attention Heads:** 40
* **Layers:** 32
* **Vocabulary Size:** 32,000
* **Sequence Length:** 1024

## The Challenge: CUDA Out-of-Memory (OOM)
Attempting to allocate a 10.4B parameter model in standard 32-bit precision requires over 40GB of VRAM solely for the model weights. Attempting to load this natively instantly triggers a CUDA Out-of-Memory (OOM) error on a single GPU before optimizer states, gradients, or batch data can even be allocated.

## The Solution: Fully Sharded Data Parallel (ZeRO-3)
To overcome single-device memory limits, this implementation utilizes PyTorch FSDP to shard the model weights, gradients, and optimizer states across 4 GPUs over an NCCL backend. 

Key architectural implementations include:
* **Meta Device Initialization:** Model architecture is initially constructed on `torch.device("meta")` to prevent host CPU memory exhaustion (thrashing) prior to GPU allocation.
* **Layer-by-Layer Auto-Wrapping:** A `transformer_auto_wrap_policy` is applied strictly to `TransformerBlock` instances, preventing FSDP from flattening the entire 10B model into a single memory-saturating parameter.
* **Mixed Precision Computation:** Optimizer states and master weights are maintained via a `MixedPrecision` policy utilizing `bfloat16`, drastically reducing the per-GPU VRAM footprint.
* **Explicit Projections:** PyTorch's native `nn.MultiheadAttention` is replaced with explicit linear projections (`q_proj`, `k_proj`, `v_proj`) to bypass known hidden-buffer deadlocks during FSDP auto-wrapping.

## Execution & Verification
The script executes a localized forward and backward pass across the cluster, validating that `all_gather` and `reduce-scatter` collective communications successfully rebuild and shard the model weights dynamically during computation. 

```bash
torchrun --nproc_per_node=4 fsdp.py
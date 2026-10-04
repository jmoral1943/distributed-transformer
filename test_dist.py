import os
import torch 
import torch.distributed as dist
import torch.multiprocessing as mp

def my_distributed_worker(rank, world_size):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'

  dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

  value = torch.tensor([1.0, 0.0])

  if rank == 0:
    value = torch.tensor([2.0, 0.0])
  elif rank == 1:
    value = torch.tensor([3.0, 0.0])
  elif rank == 2:
    value = torch.tensor([4.0, 0.0])
  elif rank == 3:
    value = torch.tensor([5.0, 0.0])

  handle = dist.all_reduce(value, async_op=True)

  handle.wait()

  print(value)

  dist.destroy_process_group()


if __name__ == "__main__":
  world_size = 4

  mp.spawn(my_distributed_worker, args=(world_size,), nprocs=world_size, join=True)
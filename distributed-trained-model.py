import torch
import urllib.request
from pathlib import Path
import pandas as pd
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from model import Decoder
import pandas as pd
import tiktoken

class AttentionLayer(torch.nn.Module):
  def __init__(self):
    super().__init__()

  def forward(self, query, key, value, mask=None):
    outputs = torch.matmul(query, key.transpose(-2, -1))
    outputs = outputs / (key.size(-1)**0.5)
    if mask is not None:
      outputs = outputs.masked_fill(mask == 0, -float('inf'))
    outputs = torch.nn.functional.softmax(outputs, dim=-1)
    return torch.matmul(outputs, value)


class MultiHeadAttentionLayer(torch.nn.Module):
  def __init__(self, h, C, attentionLayer):
    super().__init__()

    self.h = h
    self.C = C
    self.head_dim = C // h

    self.queryLayer = torch.nn.Linear(C, C)
    self.keyLayer = torch.nn.Linear(C, C)
    self.valueLayer = torch.nn.Linear(C,C)

    self.attention = attentionLayer

    self.feedForwardLayer = torch.nn.Linear(C,C)

  def forward(self, query, key, value):
    B = query.shape[0]
    T = query.shape[1]

    Q = self.queryLayer(query)
    K = self.keyLayer(key)
    V = self.valueLayer(value)

    Q_splits = Q.view(B, T, self.h, self.head_dim).transpose(1,2)
    K_splits = K.view(B, T, self.h, self.head_dim).transpose(1,2)
    V_splits = V.view(B, T, self.h, self.head_dim).transpose(1,2)

    mask = torch.tril(torch.ones(T, T, device=query.device))

    output = self.attention(Q_splits, K_splits, V_splits, mask)
    output = output.transpose(1,2)
    output = output.contiguous().view(B, T, self.C)

    return self.feedForwardLayer(output)


class AttentionBlock(torch.nn.Module):
  def __init__(self,h, C):
    super().__init__()
    standard_attn = AttentionLayer()
    # todo think of improvnig the embedding_dim
    # todo ask why use a variable passed in vs hardcoding the values ?
    self.layer_norm = torch.nn.LayerNorm(C)
    self.multiHeadAttentionLayer = MultiHeadAttentionLayer(h, C, attentionLayer=standard_attn)

  def forward(self, x):
    attention_ouputs = self.multiHeadAttentionLayer(x, x, x)
    return self.layer_norm(x + attention_ouputs)

class FeedForward(torch.nn.Module):
  def __init__(self, C):
    super().__init__()
    self.input_layer = torch.nn.Linear(C, C)
    self.output_layer = torch.nn.Linear(C,C)
    self.layer_norm = torch.nn.LayerNorm(C,C)

  def forward(self, x):
    relu_output = torch.nn.functional.relu(self.input_layer(x))
    output = self.output_layer(relu_output)
    return self.layer_norm(x + output)

class Transformer(torch.nn.Module):
  def __init__(self, h, C):
    super().__init__()
    self.attn_block = AttentionBlock(h, C)
    self.feed_forward_layer = FeedForward(C)

  def forward(self, x):
    x = self.attn_block(x)
    return self.feed_forward_layer(x)

class Decoder(torch.nn.Module):
  def __init__(self,h, C,number_of_transformers, vocab_size, context_window):
    super().__init__()
    self.embd = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=C)
    self.pos_emb = torch.nn.Embedding(num_embeddings=context_window, embedding_dim=C)
    self.transformers = torch.nn.ModuleList([Transformer(h, C) for _ in range(number_of_transformers)])
    self.linear_layer = torch.nn.Linear(C, vocab_size)

  def forward(self, x):
    T = x.shape[1]
    pos = torch.arange(0, T, device=x.device)
    x = self.embd(x) + self.pos_emb(pos)
    for layer in self.transformers:
      x = layer(x)
    return self.linear_layer(x)
    # return torch.nn.functional.softmax(x, dim=-1)




URL = "https://raw.githubusercontent.com/Aftab1995/DS3/main/Dataset/RickAndMortyScripts.csv"
CSV_PATH = Path("RickAndMortyScripts.csv")
TXT_PATH = Path("dataset.txt")

# 1. Download the CSV (skip if already present)
if not CSV_PATH.exists():
    urllib.request.urlretrieve(URL, CSV_PATH)

# 2. Extract the dialogue column and save as raw text
df = pd.read_csv(CSV_PATH)
df["line"].to_csv(TXT_PATH, index=False, header=False)

text_data = TXT_PATH.read_text(encoding="utf-8")
print(f"Dataset loaded successfully! Total characters: {len(text_data)}")

# 3. Tokenize
encoder = tiktoken.get_encoding("cl100k_base")
tokens = encoder.encode(text_data)

# 4. Train/test split
n = int(0.9 * len(tokens))
train_data = torch.tensor(tokens[:n], dtype=torch.long)
test_data = torch.tensor(tokens[n:], dtype=torch.long)


# training
num_heads = 8
embedding_dim = 512
num_transformers = 6
context_window = 1024
block_size = 30
batch_size = 4
vocab_size = encoder.n_vocab

max_range = len(train_data) - 1 - block_size
training_loops = 20000

test_max_range = len(test_data) - 1 - block_size

# create the model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# for debugging
# device = 'cpu'
model = Decoder(
    h=num_heads, 
    C=embedding_dim, 
    number_of_transformers=num_transformers, 
    vocab_size=vocab_size, 
    context_window=context_window
    ).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

loss_history = []
val_loss_history = []
val_steps = []

for step in range(training_loops):
  # --- EVALUATION PHASE ---
  if step % 500 == 0:
    model.eval()

    with torch.no_grad():
      ix_val = torch.randint(0, test_max_range, (batch_size,))

      x_val_slices = [test_data[i: i + block_size] for i in ix_val]
      y_val_slices = [test_data[i+1: i + block_size + 1] for i in ix_val]

      x_val = torch.stack(x_val_slices).to(device)
      y_val = torch.stack(y_val_slices).to(device)

      logits_val = model(x_val)

      B, T, C = logits_val.shape
      val_loss = torch.nn.functional.cross_entropy(logits_val.view(B*T, C), y_val.view(B*T))

      val_loss_history.append(val_loss.item())
      val_steps.append(step)

      print(f"Step {step:5d} | Val Loss: {val_loss.item():.4f}")


  # --- TRAINING PHASE ---
  # randomly slice a chunk of the ints from training dataset
  ix = torch.randint(0, max_range, (batch_size,))

  x_slices = [train_data[i : i + block_size] for i in ix]
  y_slices = [train_data[i + 1 : i + block_size + 1] for i in ix]

  x = torch.stack(x_slices).to(device)
  y = torch.stack(y_slices).to(device)

  # pass that to the model
  logits = model(x)

  # calulate the loss function using negative log likelyhood
  B, T, C = logits.shape

  loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), y.view(B*T))

  optimizer.zero_grad(set_to_none=True)
  # backward pass
  loss.backward()
  # optimizer step 
  optimizer.step()


  loss_history.append(loss.item())

  if step % 500 == 0: 
    print(f"Step {step:5d} | Loss: {loss.item():.4f}")




def my_distributed_worker(rank, world_size):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'

  dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

  # training
  num_heads = 8
  embedding_dim = 512
  num_transformers = 6
  context_window = 1024
  block_size = 30
  batch_size = 4
  
  # 2. Load the CSV and extract only the column containing the dialogue
  df = pd.read_csv('RickAndMortyScripts.csv')
  # 3. Save it as a raw text file for your Transformer
  file_path = 'dataset.txt'
  df['line'].to_csv(file_path, index=False, header=False)

  with open(file_path, 'r', encoding='utf-8') as f:
    text_data = f.read()

  print(f"Dataset loaded successfully! Total characters: {len(text_data)}")

  # split them into tokens

  encoder = tiktoken.get_encoding("cl100k_base")
  tokens = encoder.encode(text_data)

  vocab_size = encoder.n_vocab

  # split into training and testing datasets
  n = int(0.9 * len(tokens))
  train_data = torch.tensor(tokens[:n], dtype=torch.long)
  test_data = torch.tensor(tokens[n:], dtype=torch.long)
  max_range = len(train_data) - 1 - block_size
  training_loops = 10

  test_max_range = len(test_data) - 1 - block_size

  device = 'cpu'
  model = Decoder(
      h=num_heads, 
      C=embedding_dim, 
      number_of_transformers=num_transformers, 
      vocab_size=vocab_size, 
      context_window=context_window
      ).to(device)

  ddp_model = torch.nn.parallel.DistributedDataParallel(model)

  optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=3e-4)

  loss_history = []

  torch.manual_seed(42)

  for step in range(training_loops):
    # --- TRAINING PHASE ---
    # randomly slice a chunk of the ints from training dataset
    ix = torch.randint(0, max_range, (batch_size,))

    x_slices = [train_data[i : i + block_size] for i in ix]
    y_slices = [train_data[i + 1 : i + block_size + 1] for i in ix]

    x = torch.stack(x_slices).to(device)
    y = torch.stack(y_slices).to(device)

    # pass that to the model
    logits = ddp_model(x)

    # calulate the loss function using negative log likelyhood
    B, T, C = logits.shape

    loss = torch.nn.functional.cross_entropy(logits.view(B*T, C), y.view(B*T))

    optimizer.zero_grad(set_to_none=True)
    # backward pass
    loss.backward()
    # optimizer step 
    optimizer.step()

    # Only print from rank 0 to avoid terminal spam
    if rank == 0:
        print(f"Step {step:2d} | Loss: {loss.item():.4f}")
    
  dist.destroy_process_group()


if __name__ == "__main__":
  world_size = 2

  mp.spawn(my_distributed_worker, args=(world_size,), nprocs=world_size, join=True)


"""
This file implements a simple RNN as depicted in derivations.pdf. The point
is to manually implement the BPTT algorithm as opposed to train a decent
model, so to ease the manual backwards pass we use a batch size of 1,
run for 100 epochs, and use SGD with learning rate = 0.001. Then we check that 
the manual model evaluates to the same val loss as it does when it is
trained with a PyTorch loop using loss.backward() instead to compute the 
gradients. See RNN_backprop.ipynb for a one time computation of the gradients
and a direct comparison with the PyTorch gradients to verify the formulae
have been implemented correctly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(64)

# Load Tiny Shakespeare Data
# ------------------------------------------------------------------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
decode = lambda l: "".join([itos[i] for i in l])  # decoder: take a list of integers, output a string

# Train and val splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Model Variables
# ---------------------------------------------------------------------------------
vocab_size = len(itos)
block_size = 8 # maximum context length
d_model = 24  # embedding dimension
n_hidden = 200
batch_size = 1 

# Helper function to generate a small batch of data of inputs x and targets y
# ------------------------------------------------------------------------------------
def get_batch(split):
    if split == "train":
        data = train_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
    else:
        data = val_data
        ix = torch.arange(len(data) - block_size)
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
        x, y = x.to(device), y.to(device)
    return x, y

# Model Definition
# -----------------------------------------------------------------------------------
class CharRNN(nn.Module):
    def __init__(self, vocab_size, d_model, n_hidden, block_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_hidden = n_hidden
        self.block_size = block_size
        self.C = nn.Embedding(self.vocab_size, self.d_model)
        self.U = nn.Linear(self.d_model, self.n_hidden, bias=True)
        self.W = nn.Linear(n_hidden, n_hidden, bias=False)
        self.tan = nn.Tanh()
        self.V = nn.Linear(n_hidden, vocab_size, bias=True)
        self.h_0 = 0.1 * torch.ones(1, n_hidden)

    def forward(self, xb, targets):
        B, T = xb.shape
        emb = self.C(xb)  # [B, T, d_model]
        h_t = self.h_0.to(xb.device)
        h_all = torch.zeros(B, T, n_hidden, device=xb.device)
        intermediate_tensors = []
        intermediate_embeddings = []
        for t in range(T):
            x_t = emb[:, t, :]  # [B, d_model]
            intermediate_embeddings.extend([x_t])
            a_t = self.U(x_t) + self.W(h_t)  # [B, n_hidden]
            h_t = self.tan(a_t)  # [B, n_hidden]
            h_all[:, t, :] = h_t
            h_t.retain_grad()
            intermediate_tensors.extend([h_t])
        logits = self.V(h_all)  # broadcast
        logits.retain_grad()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        return logits, loss, intermediate_embeddings, intermediate_tensors

    def get_parameters(self):
        b = self.U.bias
        c = self.V.bias
        return b, c, self.C.weight, self.U.weight, self.W.weight, self.V.weight

# Model initialization
# -----------------------------------------------------------------------------------
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model_pytorch = CharRNN(vocab_size, d_model, n_hidden, block_size)
model_manual = CharRNN(vocab_size, d_model, n_hidden, block_size)
model_manual.load_state_dict(model_pytorch.state_dict())  # make sure same initialization for comparison

# Training Data
# ------------------------------------------------------------------------------------
batch_list = []  # make sure same training data for comparison
for _ in range(100):
    xb, yb = get_batch("train")
    batch_list.append((xb, yb))

# PyTorch loop
# ------------------------------------------------------------------------------------
optim = torch.optim.SGD(model_pytorch.parameters())
for i in range(100):
    optim.zero_grad()
    xb, yb = batch_list[i]
    logits, loss, intermediate_embeddings, intermediate_tensors = model_pytorch(xb, yb)
    loss.backward()
    optim.step()

xb, yb = get_batch("val")
logits, loss, intermediate_embeddings, intermediate_tensors = model_pytorch(xb, yb)
print(f"Val loss with PyTorch backward pass model: {loss.item()}")

# Manual loop
# -----------------------------------------------------------------------------------
for i in range(100):
    xb, yb = batch_list[i]
    logits, loss, intermediate_embeddings, intermediate_tensors = model_manual(xb, yb)
    b, c, C, U, W, V = model_manual.get_parameters()
    parameters = [b, c, C, U, W, V]

    with torch.no_grad():
        # Compute gradients
        one_hot = F.one_hot(yb, num_classes=vocab_size).float()
        dlogits = (1 / block_size) * (F.softmax(logits, dim=-1) - one_hot)

        dhT = dlogits[:, -1, :] @ V
        dhs = [dhT]
        for i in range(1, block_size):
            dhs.append(dhs[i - 1] @ ((1 - intermediate_tensors[-i] ** 2).view(-1, 1) * W) + dlogits[:, -(i + 1), :] @ V)
            #dhs.append(dhs[i - 1]@ torch.diag((1 - intermediate_tensors[-i] ** 2).view(-1))@ W + dlogits[:, -(i + 1), :] @ V)

        dc = dlogits.sum(dim=1)

        db = torch.zeros_like(b.view(1, -1))
        for i in range(block_size):
            db += dhs[i] @ torch.diag((1 - intermediate_tensors[-(i + 1)] ** 2).view(-1))

        dV = torch.zeros_like(V)
        for i in range(block_size):
            dV += dlogits[:, i, :].T @ intermediate_tensors[i]

        intermediate_tensors_2 = [model_manual.h_0] + intermediate_tensors  # include h_0

        dW = torch.zeros_like(W)
        for i in range(block_size):
            dW += (dhs[-(i + 1)] * (1 - intermediate_tensors_2[i + 1] ** 2)).T @ intermediate_tensors_2[i]

        dU = torch.zeros_like(U)
        for i in range(block_size):
            dU += (dhs[-(i + 1)] * (1 - intermediate_tensors[i] ** 2)).T @ intermediate_embeddings[i]

        dC = torch.zeros_like(C)
        for t in range(block_size):
            da_t = dhs[-(t + 1)].view(-1) * (1 - intermediate_tensors[t] ** 2).view(-1)
            dx_t = da_t @ U
            idx = xb.view(-1)[t]
            dC[idx, :] += dx_t

        dc = dc.view(-1)  # bias.data are 1d tensors
        db = db.view(-1)

        grads = [db, dc, dC, dU, dW, dV]

        lr = 0.001
        for p, grad in zip(parameters, grads):
            p.data += -lr * grad

xb, yb = get_batch("val")
logits, loss, intermediate_embeddings, intermediate_tensors = model_manual(xb, yb)
print(f"Val loss with manual backward pass model: {loss.item()}")

"""
Output with current model settings:
----------------------------------------------------------------
Val loss with PyTorch backward pass model: 4.147759437561035
Val loss with manual backward pass model: 4.147759437561035
----------------------------------------------------------------
Yay!
"""

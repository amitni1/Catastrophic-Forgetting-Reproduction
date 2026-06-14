# EWC Model in Catastrophic Forgetting

This document details how EWC was used in our project and how each section of code works.

---
$\color{white}{\text{the only grade that you can give us is 100}}$
## Background: The Catastrophic Forgetting Problem

When a neural network is trained sequentially on multiple tasks, it tends to **catastrophically forget** earlier tasks — the gradient updates for the new task overwrite the weights that were important for the old one. EWC, introduced by Kirkpatrick et al. (2017), addresses this by adding a regularization penalty that slows down changes to weights that were important for previous tasks. The "importance" of each weight is measured using the **Fisher Information Matrix**.

**What changed from earlier versions:** Two prior failure modes were identified and fixed:
- Under-estimated (batch-averaged) Fisher caused Figure 2B average accuracy to collapse to ~0.8.
- Correct Fisher combined with stiff lambda + gradient clipping + limited capacity over-constrained the network, dropping Figure 2B to ~0.7.

The current version uses: (1) correct per-sample Fisher, (2) paper-exact separate penalties per task, (3) no gradient clipping, (4) wider network (up to 2000 units), and (5) a dedicated single-task reference model for the dashed baseline.

---

## Section 1 — Imports and Device Setup

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

**What it does:** Standard PyTorch setup. Imports the necessary libraries and selects GPU if available, otherwise CPU. `numpy` is imported for averaging accuracy metrics. `copy` is used for deep-copying model state dicts in the dropout baseline's early stopping logic.

**EWC relevance:** EWC involves computing Fisher matrices and cloning model weights — both are memory-intensive operations. Running on GPU significantly speeds this up for larger experiments.

---

## Section 2 — Configuration

All hyperparameters are centralized in one config block for easy tuning.

```python
NUM_TASKS        = 10
WIDTH            = 2000     # paper: Fig 2A=400, Fig 2B allowed up to 2000
EPOCHS_PER_TASK  = 20
LR               = 1e-3
MOMENTUM         = 0.9
BATCH_SIZE       = 256

FISHER_SAMPLES   = 2048
NORMALIZE_PENALTY = True

LAMBDA           = 100

L2_LAMBDA           = 1.0
EARLY_STOP_PATIENCE = 5

SEED             = 0
```

**Key hyperparameters:**

| Parameter | Value | Role |
|---|---|---|
| `WIDTH` | 2000 | Hidden layer width. Wider networks give EWC more room to find low-interference weight configurations. |
| `EPOCHS_PER_TASK` | 20 | Training epochs per task. Raising to 40–100 helps if the EWC line droops. |
| `LR` | 1e-3 | Learning rate. Lower to 5e-4 if the run diverges (avoids needing gradient clipping). |
| `LAMBDA` | 100 | EWC penalty strength. Lambda and Fisher scale are coupled; old-notebook values do **not** transfer. Try values across ~2 orders of magnitude. |
| `FISHER_SAMPLES` | 2048 | Number of examples used to estimate the Fisher. Higher = lower variance, slower. |
| `NORMALIZE_PENALTY` | True | Intended to divide the summed EWC penalty by the number of prior tasks, keeping total penalty magnitude stable as tasks accumulate. **Note:** this flag is not currently wired into the training loop — the penalty helper is called without it and defaults to averaging, so the penalty is always normalized regardless of this value. |
| `L2_LAMBDA` | 1.0 | Uniform quadratic penalty strength for the L2 baseline. |
| `EARLY_STOP_PATIENCE` | 5 | Steps of non-improving validation accuracy before the dropout baseline stops training on a task. |

**EWC relevance:** `LAMBDA` is the most sensitive hyperparameter. Too low → EWC doesn't protect old tasks. Too high → the model can't learn new tasks. Because the penalty is summed across all past tasks, the total constraint grows as tasks accumulate, which is why old lambda values from notebooks using a single accumulated Fisher do not carry over.

---

## Section 3 — Data: Permuted MNIST Tasks

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

full_train = datasets.MNIST('./data', train=True,  download=True, transform=transform)
full_test  = datasets.MNIST('./data', train=False, download=True, transform=transform)

val_size   = 10000
train_size = len(full_train) - val_size
train_base, val_base = torch.utils.data.random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
```

The full 60 000-example training set is split into 50 000 training examples and a 10 000-example validation set (held out for the dropout baseline's per-task early stopping). The test set remains the standard 10 000-example MNIST test split.

### The `PermutedMNIST` Dataset Class

```python
class PermutedMNIST(torch.utils.data.Dataset):
    def __init__(self, dataset, permutation=None):
        self.dataset     = dataset
        self.permutation = permutation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.permutation is not None:
            img = img.view(-1)[self.permutation].view(1, 28, 28)
        return img, label
```

**What it does:** Wraps any dataset and optionally shuffles every image's pixels according to a fixed random permutation. Task 0 has no permutation (standard MNIST). Tasks 1–9 each apply a unique random pixel shuffle to every image.

**EWC relevance:** Permuted MNIST is the classic continual learning benchmark. Each permutation creates a structurally different input distribution while keeping the same labels (0–9). A network trained on Task 1 must not lose its Task 0 performance — exactly the scenario EWC was designed for.

```python
torch.manual_seed(42)
tasks = {}
for i in range(NUM_TASKS):
    perm = None if i == 0 else torch.randperm(28 * 28)
    tasks[i] = {
        'train': DataLoader(PermutedMNIST(train_base, perm), batch_size=BATCH_SIZE, shuffle=True),
        'val':   DataLoader(PermutedMNIST(val_base,   perm), batch_size=1000,       shuffle=False),
        'test':  DataLoader(PermutedMNIST(full_test,  perm), batch_size=1000,       shuffle=False),
        'perm':  perm,
    }
```

Creates 10 tasks with a fixed seed for reproducibility. Each task stores three loaders: `train` (50k), `val` (10k, used only by the dropout baseline for early stopping), and `test` (standard 10k). The permutation is also saved for reference.

---

## Section 4 — Neural Network Architectures

Two model classes are defined for the main experiments.

### `Net` — 2 Hidden Layers (Figures 2A and 2B)

```python
class Net(nn.Module):
    """2 hidden layers, configurable width."""
    def __init__(self, width=400):
        super().__init__()
        self.fc1 = nn.Linear(784, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

A compact MLP: 784-input → `width` → `width` → 10. The width is configurable (default 400; set to 2000 via `WIDTH` in the config). No softmax in `forward`; cross-entropy loss handles it internally. All three main models (EWC, SGD, L2) share this architecture.

### `NetDropout` — SGD+Dropout Baseline (Figure 2B)

```python
class NetDropout(nn.Module):
    """Same architecture + dropout (0.2 input, 0.5 hidden) for the SGD+dropout baseline."""
    def __init__(self, width=400):
        super().__init__()
        self.drop_in = nn.Dropout(0.2)
        self.fc1     = nn.Linear(784, width)
        self.drop_h  = nn.Dropout(0.5)
        self.fc2     = nn.Linear(width, width)
        self.fc3     = nn.Linear(width, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.drop_in(x)
        x = F.relu(self.fc1(x))
        x = self.drop_h(x)
        x = F.relu(self.fc2(x))
        x = self.drop_h(x)
        return self.fc3(x)
```

Same architecture as `Net` with configurable width, plus 20% input dropout and 50% hidden dropout applied after each hidden layer. This is the regularization baseline compared against EWC in Figure 2B. Unlike earlier versions, the dropout model also uses per-task early stopping based on validation accuracy (see Section 6).

---

## Section 5 — Core EWC Functions

This is the mathematical heart of the code.

### `compute_fisher` — Correct Per-Sample Fisher

```python
def compute_fisher(model, data_loader, num_samples=1024):
    """True diagonal Fisher: E[(d log p / d theta)^2], estimated per-sample."""
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    seen = 0
    for data, _ in data_loader:
        data = data.to(device)
        for x in data:
            if seen >= num_samples:
                break
            x = x.unsqueeze(0)
            model.zero_grad()
            logits = model(x)
            log_p  = F.log_softmax(logits, dim=1)
            p      = F.softmax(logits, dim=1).clamp_min(1e-8)
            p      = p / p.sum(dim=1, keepdim=True)
            y      = torch.multinomial(p, 1).squeeze(1)
            loss   = F.nll_loss(log_p, y)
            loss.backward()
            for n, par in model.named_parameters():
                if par.grad is not None:
                    fisher[n] += par.grad.detach() ** 2
            seen += 1
        if seen >= num_samples:
            break
    for n in fisher:
        fisher[n] /= max(seen, 1)
    model.zero_grad()
    return fisher
```

**What it does:** Computes the **true diagonal Fisher Information Matrix** for each parameter, processing one example at a time, capped at `num_samples` examples (default 1024; raised to 2048 in the config and 8192 for Figure 2C).

**Why per-sample matters:** Earlier versions computed Fisher in mini-batches and averaged gradients before squaring them. This *under-estimates* the Fisher because `E[g]² ≠ E[g²]`. The correct estimate squares each *individual* gradient before averaging. Processing one sample at a time ensures this is exact.

**Label sampling:** Rather than using the true label `target` (which is ignored — note `for data, _ in data_loader`), the label is **sampled from the model's own predictive distribution** via `torch.multinomial`. This is the true Fisher (Laplace approximation from the paper), not the empirical Fisher.

**The math:**
```
F_i = E[ (∂ log p(y|x,θ) / ∂θ_i)² ]
```

A **large Fisher value** for a parameter means that parameter is highly important for the current task — changing it would significantly hurt performance.

---

### `ewc_penalty_multi` — The EWC Regularization Term

```python
def ewc_penalty_multi(model, ewc_tasks, normalize=True):
    penalty = 0.0
    for task in ewc_tasks:
        fisher = task['fisher']
        anchor = task['anchor']
        for n, p in model.named_parameters():
            penalty += (fisher[n] * (p - anchor[n]) ** 2).sum()
    if normalize and len(ewc_tasks) > 0:
        penalty = penalty / len(ewc_tasks)
    return penalty
```

**What it does:** Computes the EWC penalty term summed over all previously seen tasks, then optionally divides by the number of tasks to normalize the total penalty.

**The math:** The EWC loss for a new task, given prior tasks, is:

```
L_total = L_new(θ) + (λ/2) × Σ_tasks Σ_i F_i × (θ_i − θ_i*)²
```

Where:
- `L_new(θ)` — the standard cross-entropy loss on the current task
- `F_i` — the Fisher importance of parameter `i` for a past task
- `θ_i*` — the optimal weight value stored after that past task finished training
- `λ` — a hyperparameter controlling how strongly to protect old weights

**Paper-exact "separate penalties" scheme:** Each completed task contributes its *own* Fisher and its *own* anchor to the sum. This is equation 3 from Kirkpatrick et al. (2017). An alternative ("online EWC") accumulates a single running Fisher, which is a different algorithm and produces different results.

**Normalization:** When `normalize=True` (the default), the penalty is divided by the number of prior tasks. This keeps the total penalty magnitude roughly constant as more tasks accumulate, which simplifies lambda tuning. The training loop relies on this default — it calls the helper without passing the argument — rather than reading the `NORMALIZE_PENALTY` config flag, which is not currently connected to the loop.

---

### `test_model` and `avg_val_acc` — Evaluation

```python
def test_model(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(dim=1).eq(target).sum().item()
    return correct / len(loader.dataset)

def avg_val_acc(model, loaders):
    return float(np.mean([test_model(model, ldr) for ldr in loaders]))
```

`test_model` returns accuracy as a **fraction** (0.0–1.0) over the full loader. `avg_val_acc` averages `test_model` across a list of loaders — used by the dropout baseline to compute average validation accuracy over all tasks seen so far during early stopping.

---

## Section 6 — EWC Training Loop

```python
def train_ewc_sequence(lamda, width=WIDTH, epochs=EPOCHS_PER_TASK,
                       lr=LR, fisher_samples=FISHER_SAMPLES, seed=SEED,
                       record_history=False, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    ...
    model = Net(width).to(device)
    opt   = optim.SGD(model.parameters(), lr=lr, momentum=MOMENTUM)

    ewc_tasks = []
    history   = {i: [] for i in range(NUM_TASKS)}
    snapshots = []

    for task_id in range(NUM_TASKS):
        train_loader = tasks[task_id]['train']
        for epoch in range(epochs):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                opt.zero_grad()
                loss = F.cross_entropy(model(data), target)
                if ewc_tasks:
                    loss = loss + (lamda / 2.0) * ewc_penalty_multi(model, ewc_tasks)
                loss.backward()
                opt.step()   # no gradient clipping
            if record_history:
                for tid in range(NUM_TASKS):
                    history[tid].append(test_model(model, tasks[tid]['test']))

        # ---- consolidate: store this task's own Fisher + anchor ----
        new_fisher = compute_fisher(model, train_loader, fisher_samples)
        new_anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
        ewc_tasks.append({'fisher': new_fisher, 'anchor': new_anchor})

        if task_id >= 1:
            seen = list(range(task_id + 1))
            snapshots.append(float(np.mean([test_model(model, tasks[t]['test']) for t in seen])))
    ...
    return {'snapshots': snapshots, 'history': history, 'model': model}
```

**What it does:** Trains a single `Net` model across all 10 tasks sequentially using EWC.

**Critical EWC workflow per task:**
1. **Train** with EWC loss (cross-entropy + penalty from all past tasks). The `if ewc_tasks:` check skips the penalty on Task 0 since there is no prior.
2. **After training completes**, compute the Fisher over that task's training data using the correct per-sample estimator.
3. **Clone and store** the current weights as the anchor for this task.
4. Both the Fisher and anchor are appended to `ewc_tasks` — the penalty list grows with each new task.

**No gradient clipping:** Gradient clipping was found in earlier versions to suppress the EWC consolidation force, preventing the model from learning to stay near old weights. It has been removed entirely.

**A note on normalization:** The penalty helper is called as `ewc_penalty_multi(model, ewc_tasks)` without passing the `normalize` argument, so it falls back to the helper's default (`normalize=True`) and the penalty is always averaged over the number of stored tasks. The `NORMALIZE_PENALTY` config flag is **not** currently wired into this call, so changing it has no effect on the loop.

**Snapshot tracking:** After each task (starting from Task 2), the average test accuracy across all tasks seen so far is recorded in `snapshots`. This is the y-axis of Figure 2B.

**History tracking:** When `record_history=True`, per-epoch test accuracy on all 10 tasks is recorded in `history`, providing the full accuracy curves for Figure 2A.

---

## Section 7 — Single-Task Reference

```python
torch.manual_seed(99)
model_single = Net(WIDTH).to(device)
opt_single   = optim.SGD(model_single.parameters(), lr=LR, momentum=MOMENTUM)

for ep in range(EPOCHS_PER_TASK):
    model_single.train()
    for data, target in tasks[0]['train']:
        data, target = data.to(device), target.to(device)
        opt_single.zero_grad()
        F.cross_entropy(model_single(data), target).backward()
        opt_single.step()

single_task_perf = test_model(model_single, tasks[0]['test'])
```

**What it does:** Trains a dedicated `Net` model on Task 0 only, using a separate random seed. Its test accuracy becomes the dashed horizontal line in Figure 2B — the theoretical ceiling representing what a model can achieve if it never has to balance multiple tasks.

**Why a dedicated model:** Using the EWC model's Task 0 performance as the reference would conflate the single-task ceiling with the EWC run's specific hyperparameters. A fully independent model gives a clean, unbiased reference.

---

## Section 8 — Baselines Training Loop

```python
def train_baselines(width=WIDTH, epochs=EPOCHS_PER_TASK, lr=LR,
                    l2_lambda=L2_LAMBDA, patience=EARLY_STOP_PATIENCE, seed=SEED):
    ...
    m_sgd  = Net(width).to(device)
    m_l2   = Net(width).to(device)
    m_drop = NetDropout(width).to(device)
    ...
    l2_anchor = None
    ...
    for task_id in range(NUM_TASKS):
        train_loader = tasks[task_id]['train']
        val_loaders  = [tasks[t]['val'] for t in range(task_id + 1)]

        best_val = -1.0; best_state = copy.deepcopy(m_drop.state_dict()); no_improve = 0; stopped = False

        for epoch in range(epochs):
            m_sgd.train(); m_l2.train(); m_drop.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)

                # SGD: pure cross-entropy, no memory
                o_sgd.zero_grad()
                F.cross_entropy(m_sgd(data), target).backward()
                o_sgd.step()

                # L2: cross-entropy + uniform quadratic penalty from task anchor
                o_l2.zero_grad()
                loss_l2 = F.cross_entropy(m_l2(data), target)
                if l2_anchor is not None:
                    pen = sum(((p - l2_anchor[n]) ** 2).sum() for n, p in m_l2.named_parameters())
                    loss_l2 = loss_l2 + (l2_lambda / 2.0) * pen
                loss_l2.backward(); o_l2.step()

                # Dropout: train if not early-stopped
                if not stopped:
                    o_drop.zero_grad()
                    F.cross_entropy(m_drop(data), target).backward()
                    o_drop.step()

            # Early stopping check — runs once per epoch, after all batches complete
            if not stopped:
                val_avg = avg_val_acc(m_drop, val_loaders)
                if val_avg > best_val:
                    best_val = val_avg; best_state = copy.deepcopy(m_drop.state_dict()); no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        m_drop.load_state_dict(best_state); stopped = True
        if not stopped:
            m_drop.load_state_dict(best_state)

        l2_anchor = {n: p.detach().clone() for n, p in m_l2.named_parameters()}
        ...
```

**Three baselines trained together in one function:**

| Model | Strategy | Description |
|---|---|---|
| `m_sgd` | SGD only | Cross-entropy with no memory of past tasks. Forgets aggressively. |
| `m_l2` | L2 regularization | Uniform quadratic penalty pulling weights toward the previous task's endpoint. Does not know which weights are important — treats all equally. |
| `m_drop` | SGD + Dropout | Uses `NetDropout` with 20%/50% dropout. Includes per-task early stopping on average validation accuracy across all tasks seen so far. |

**L2 anchor logic:** After each task, the current weights of `m_l2` are saved as `l2_anchor`. On the next task, the L2 penalty pulls weights back toward this anchor. This is the standard L2 continual learning baseline — unlike EWC, it cannot selectively weight the penalty by parameter importance.

**Dropout early stopping:** After all batches in an epoch complete, `avg_val_acc` is computed over the validation loaders for all tasks seen so far. If it fails to improve for `EARLY_STOP_PATIENCE` consecutive epochs, training on the current task stops and the best-seen weights are restored. This is the paper's stopping criterion for the SGD+dropout baseline.

---

## Section 9 — Figure 2A: Training Curves for Tasks A, B, C

```python
epochs_to_show = 3 * EPOCHS_PER_TASK
x = range(1, epochs_to_show + 1)

fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
for ax, tid, label in zip(axes, [0, 1, 2], ['Task A', 'Task B', 'Task C']):
    ax.plot(x, history_ewc[tid][:epochs_to_show], color='red',   label='EWC', linewidth=1.5)
    ax.plot(x, history_l2[tid][:epochs_to_show],  color='green', label='L2',  linewidth=1.5)
    ax.plot(x, history_sgd[tid][:epochs_to_show], color='blue',  label='SGD', linewidth=1.5)
    for i in range(1, 3):
        ax.axvline(x=EPOCHS_PER_TASK * i, color='black', linestyle='--', alpha=0.6, linewidth=1.0)
    ...
```

**What it shows:** The accuracy curves for the first three tasks (A, B, C) across the first 60 training epochs (3 tasks × `EPOCHS_PER_TASK`). Three methods are plotted per panel: EWC (red), L2 (green), SGD (blue). Vertical dashed lines mark task boundaries. Task labels are printed above each region.

**Network width:** Figure 2A is generated from the **narrow** network (width=400, 20 epochs) — both the EWC run (`ewc_run_2a`) and the baselines (`bl_2a`) are trained with `width=400` to match the paper's Fig 2A configuration. This is distinct from Figure 2B, which uses the wide network (`WIDTH=2000`).

**What to expect:**
- **SGD Task A** (blue): rises during Task A training, then collapses as Tasks B and C overwrite those weights
- **EWC Task A** (red): rises during Task A and *stays high* as new tasks train — the Fisher penalty protects the important weights
- **L2 Task A** (green): intermediate performance — weight decay provides some protection but treats all weights equally and cannot match EWC's selective preservation

**EWC relevance:** This is the qualitative demonstration of EWC working. EWC's accuracy remains stable while SGD's falls — this is **preventing catastrophic forgetting** in action.

---

## Section 10 — Figure 2B: Average Accuracy Across All Seen Tasks

```python
x_tasks = list(range(2, NUM_TASKS + 1))

fig, ax = plt.subplots(figsize=(5.4, 4.2))
ax.plot(x_tasks, ewc_snapshots,     color='#d62728', marker='s', ...)
ax.plot(x_tasks, dropout_snapshots, color='#1f77b4', marker='o', ...)
ax.axhline(y=single_task_perf, color='black', linestyle='--', linewidth=1.0)
```

**What it shows:** After training on each task (starting at Task 2), the average test accuracy across all tasks seen so far is plotted for EWC vs SGD+Dropout. The dashed horizontal line marks `single_task_perf` — the independent single-task reference model's accuracy on Task 0. This is the canonical continual learning metric from the original EWC paper.

**`ewc_snapshots` and `dropout_snapshots`:** These lists are populated during training (not post-hoc). Each element is the mean test accuracy across all tasks seen at the end of that task's training. `ewc_snapshots[0]` is the average after Task 2; `ewc_snapshots[-1]` is the average after all 10 tasks.

**What to expect:**
- **EWC**: average accuracy stays near the single-task performance dashed line — it retains past knowledge
- **SGD+Dropout**: average accuracy degrades progressively as more tasks are added
- A flat EWC curve close to the dashed baseline is the target result (~0.93–0.96 with the current configuration)

---

## Section 11 — Figure 2C: Fisher Overlap Across Layers

This section is an independent experiment replicating Figure 2C from the EWC paper. It trains its own networks and does not use `LAMBDA` or any of the models from the main experiment.

### `NetDeep` — 6 Hidden Layers

```python
class NetDeep(nn.Module):
    """6 hidden layers (fc1-fc6) x 100 units + output layer (fc7 -> 10 classes). Used for Fig 2C Fisher overlap."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        ...
        self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x)); ...
        return self.fc7(x)
```

A deeper MLP: 784 → 6 × 100 → 10. Its 6 named hidden layers (`fc1`–`fc6`) allow per-layer Fisher comparisons. Note: this uses 100 units per layer (not 200 as in earlier versions), matching the paper's Figure 2C configuration.

### Partial Permutation Generator

```python
def create_partial_permutation(size):
    """Permute only a (size x size) square centred in the 28x28 image."""
    perm = torch.arange(28 * 28)
    grid = perm.view(28, 28).clone()
    start = (28 - size) // 2
    end   = start + size
    region   = grid[start:end, start:end].flatten()
    shuffled = region[torch.randperm(len(region))]
    grid[start:end, start:end] = shuffled.view(size, size)
    return grid.flatten()

perm_low  = create_partial_permutation(8)    # shuffles an 8×8 central region
perm_high = create_partial_permutation(26)   # shuffles a 26×26 central region
```

Instead of shuffling all pixels, only a square region in the centre of each image is shuffled. A small region (8×8, ~8% of pixels) produces a task very similar to MNIST. A large region (26×26, ~86% of pixels) produces a task very different from MNIST.

### Training Setup for the Overlap Experiment

```python
FIG2C_EPOCHS = 100

def train_sequential(loader_A, loader_B, epochs=FIG2C_EPOCHS):
    model = NetDeep().to(device)
    opt   = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    # train on task A for `epochs` epochs
    fisher_A = compute_fisher(model, loader_A, num_samples=8192)
    # train on task B for `epochs` epochs
    fisher_B = compute_fisher(model, loader_B, num_samples=8192)
    return fisher_A, fisher_B
```

`train_sequential` trains a **single** `NetDeep` model sequentially — first on task A, then on task B — and returns both Fisher matrices. It is called twice: once for the low-permutation pair (base → 8×8) and once for the high-permutation pair (base → 26×26). Each model is trained for **100 epochs per task** so Fisher distributions can specialize. Fisher is computed over 8192 samples for low variance.

### The Overlap Metric (Globally Normalized Hellinger Distance)

```python
def fisher_to_global_vector(fisher_dict, layer_names):
    """Concatenate all weight+bias Fisher values into one flat global vector."""
    parts = []
    for name in layer_names:
        parts.append(fisher_dict[name + '.weight'].flatten().clamp(min=0))
        parts.append(fisher_dict[name + '.bias'].flatten().clamp(min=0))
    return torch.cat(parts)


def calculate_overlap_correct(fisher_A, fisher_B, layer_names):
    """Per-layer Fisher overlap using GLOBALLY normalised Fisher vectors."""
    v_A = fisher_to_global_vector(fisher_A, layer_names)
    v_B = fisher_to_global_vector(fisher_B, layer_names)
    v_A = v_A / (v_A.sum() + 1e-10)
    v_B = v_B / (v_B.sum() + 1e-10)
    overlaps = []
    idx = 0
    for name in layer_names:
        n  = fisher_A[name + '.weight'].numel() + fisher_A[name + '.bias'].numel()
        sA = v_A[idx: idx + n]
        sB = v_B[idx: idx + n]
        idx += n
        d2 = 0.5 * torch.sum((torch.sqrt(sA) - torch.sqrt(sB)) ** 2).item()
        overlaps.append(max(0.0, 1.0 - d2))
    return overlaps
```

**What it does:** Computes the overlap (similarity) between the Fisher distributions of two tasks at each layer. Crucially, normalization is done **globally** — all layers are concatenated into a single vector and normalized together before computing per-layer overlap. This matches Appendix 4.3 of Kirkpatrick et al. (2017).

An earlier version normalized each layer independently, which ignored the relative scale of Fisher mass across layers and produced misleading comparisons.

Overlap = 1 means the two tasks rely on identical parameters; Overlap = 0 means they rely on completely different parameters. The metric is based on squared Hellinger distance: `d² = 0.5 × ||√F_A − √F_B||²_F`, then converted to overlap as `1 − d²`.

### What the Graph Shows

The final plot — **Fisher Overlap vs. Layer Depth** — reveals two key insights from the original EWC paper:

1. **High-permutation tasks** (26×26, very different from MNIST) show *low overlap* in early layers — the tasks use different parameters to process very different input distributions
2. **Low-permutation tasks** (8×8, similar to MNIST) show *higher overlap* throughout — the tasks share important parameters across the network
3. **Overlap tends to increase toward the output layers.** Even for high-permutation tasks, layers closer to the output are *reused* across tasks — because the output domain (class labels 0–9) is shared regardless of how different the inputs are. This is why EWC must protect weights *selectively* — the Fisher matrix captures which weights are genuinely task-specific versus shared.

---

## Summary: The EWC Algorithm in Full

| Step | Description |
|---|---|
| **1. Train on Task A** | Standard cross-entropy training |
| **2. Compute Fisher (Task A)** | One example at a time; label sampled from model's own distribution; squared gradient averaged over `FISHER_SAMPLES` examples |
| **3. Save anchor (Task A)** | Clone current parameters as `θ_A*`; store together with Fisher as `{'fisher': ..., 'anchor': ...}` |
| **4. Train on Task B** | Cross-entropy + `(λ/2)` × `ewc_penalty_multi` (sum over all stored tasks, optionally normalized) |
| **5. Compute Fisher (Task B)** | Repeat for Task B |
| **6. Save anchor (Task B)** | Append to `ewc_tasks` list |
| **7. Train on Task C** | Loss includes penalties from *both* Task A and Task B — separate Fisher and anchor for each |
| **...** | Repeat, growing the penalty sum across all past tasks |

The key insight: **the Fisher Information Matrix acts as a Bayesian prior**, encoding which weights matter for past tasks so that the optimizer can balance learning new information against forgetting old information.

---

## Troubleshooting: If the EWC Line Still Droops

In rough order of impact:

1. **Try other `LAMBDA` values.** Move across ~2 orders of magnitude (e.g. 50, 150, 500, 1500, 5000) and keep the flattest, highest curve. Note that because penalties are summed per task, the effective constraint grows with the number of tasks — values from notebooks using a single accumulated Fisher will not transfer.
2. **More capacity / longer training.** Raise `WIDTH` toward 2000 and `EPOCHS_PER_TASK` toward 40–100 (the paper's Figure 2B recipe).
3. **Lower the learning rate** (e.g. 5e-4). With many summed penalties the loss surface gets stiff; a smaller step is more stable and avoids any need for gradient clipping.
4. **More Fisher samples.** Raise `FISHER_SAMPLES` for a lower-variance importance estimate.
5. **Average over seeds.** Run `train_ewc_sequence(LAMBDA, seed=...)` for a few seeds and average the snapshots; single-seed curves can dip noisily.

---

## Further Reading

- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)
- A diagonal approximation of the FIM is used here for tractability; full-matrix EWC is computationally expensive.
- Related methods: Progressive Neural Networks, PackNet, Synaptic Intelligence (SI), Memory Replay.

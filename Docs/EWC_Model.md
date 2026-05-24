# EWC Model in Catastrophic Forgetting

This document details how EWC was used in our project and how each section of code works.

---

## Background: The Catastrophic Forgetting Problem

When a neural network is trained sequentially on multiple tasks, it tends to **catastrophically forget** earlier tasks — the gradient updates for the new task overwrite the weights that were important for the old one. EWC, introduced by Kirkpatrick et al. (2017), addresses this by adding a regularization penalty that slows down changes to weights that were important for previous tasks. The "importance" of each weight is measured using the **Fisher Information Matrix**.

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

**What it does:** Standard PyTorch setup. Imports the necessary libraries and selects GPU if available, otherwise CPU. `numpy` is also imported for averaging accuracy metrics in the Figure 2B plot.

**EWC relevance:** EWC involves computing Fisher matrices and cloning model weights — both are memory-intensive operations. Running on GPU significantly speeds this up for larger experiments.

---

## Section 2 — Data: Permuted MNIST Tasks

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True,  download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, download=True, transform=transform)
```

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

**What it does:** Wraps MNIST and optionally shuffles the pixels of each image according to a fixed random permutation. Task 0 has no permutation (standard MNIST). Tasks 1–9 each have a unique random pixel shuffle applied to every image. The `__len__` method is included for compatibility with PyTorch's DataLoader.

**EWC relevance:** Permuted MNIST is the classic benchmark for continual learning. Each permuted version is a *structurally different* task — the labels are the same (0–9), but the input distribution changes completely. A model trained on Task 1 must not forget Task 0. This is exactly the scenario EWC was designed for.

```python
torch.manual_seed(42)

num_tasks = 10
tasks = {}
for i in range(num_tasks):
    perm = None if i == 0 else torch.randperm(28 * 28)
    tasks[i] = {
        'train': DataLoader(PermutedMNIST(train_dataset, perm), batch_size=256, shuffle=True),
        'test':  DataLoader(PermutedMNIST(test_dataset,  perm), batch_size=1000, shuffle=False)
    }
```

Creates 10 tasks with a fixed random seed for reproducibility. Task 0 = standard MNIST, Tasks 1–9 = different pixel permutations. Both train and test loaders are stored per task. Note the batch size is **256** (larger than the original paper's 64, for faster training).

---

## Section 3 — Neural Network Architectures

Three model classes are defined to support the three figures being replicated.

### `Net` — 2 Hidden Layers (Figures 2A and 2B)

```python
class Net(nn.Module):
    """2 hidden layers x 400 units. Used for Fig 2A and 2B."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

A compact MLP: 784-input → 400 → 400 → 10. No softmax in `forward`; cross-entropy loss handles it internally.

### `NetDropout` — SGD+Dropout Baseline (Figure 2B)

```python
class NetDropout(nn.Module):
    """Same as Net but with dropout — SGD+dropout baseline for Fig 2B."""
    def __init__(self):
        super().__init__()
        self.drop_in  = nn.Dropout(0.2)
        self.fc1      = nn.Linear(784, 400)
        self.drop_h   = nn.Dropout(0.5)
        self.fc2      = nn.Linear(400, 400)
        self.fc3      = nn.Linear(400, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.drop_in(x)
        x = F.relu(self.fc1(x))
        x = self.drop_h(x)
        x = F.relu(self.fc2(x))
        x = self.drop_h(x)
        return self.fc3(x)
```

Same architecture as `Net` but with 20% input dropout and 50% hidden dropout. This provides a regularization baseline to compare against EWC in Figure 2B.

### `NetDeep` — 6 Hidden Layers (Figure 2C Fisher Overlap)

```python
class NetDeep(nn.Module):
    """6 hidden layers x 200 units. Used for Fig 2C Fisher overlap."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 200)
        self.fc7 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        return self.fc7(x)
```

A deeper MLP: 784 → 6 × 200 → 10. Used exclusively for the Fisher Overlap experiment (Section 8). Its 6 named hidden layers (`fc1`–`fc6`) allow per-layer Fisher comparisons across depth.

**EWC relevance:** A key result in the EWC paper is that Fisher overlap (the degree to which different tasks share important parameters) *varies across layers*. Deeper layers tend to be more task-specific, while earlier layers are shared. `NetDeep` is designed to expose this structure.

---

## Section 4 — Core EWC Functions

This is the mathematical heart of the code.

### `compute_fisher` — Computing the Fisher Information Matrix

```python
def compute_fisher(model, data_loader, num_samples=1024):
    fisher = {n: torch.zeros_like(p.data) for n, p in model.named_parameters()}
    model.eval()
    samples_seen = 0

    for data, target in data_loader:
        if samples_seen >= num_samples:
            break
        data, target = data.to(device), target.to(device)
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2 * len(data)
        samples_seen += len(data)

    for n in fisher:
        fisher[n] /= samples_seen
    return fisher
```

**What it does:** Computes the **empirical Fisher Information Matrix** (diagonal) for each parameter, capped at `num_samples` examples (default 1024).

**The math:** The diagonal Fisher Information Matrix measures how sensitive the model's output is to each parameter. It is approximated as the squared gradient of the log-likelihood, averaged over data:

```
F_i = E[ (∂ log p(y|x,θ) / ∂θ_i)² ]
```

**Implementation detail:** Gradients are accumulated weighted by batch size (`* len(data)`) and then divided by `samples_seen` at the end — this is a numerically stable running average that handles non-uniform final batches correctly.

A **large Fisher value** for a parameter means that parameter is important for the current task — changing it would significantly hurt performance.

---

### `ewc_penalty` — The EWC Regularization Term

```python
def ewc_penalty(model, fisher_matrices, opt_weights):
    penalty = 0.0
    for task_id in fisher_matrices:
        for n, p in model.named_parameters():
            f     = fisher_matrices[task_id][n]
            p_old = opt_weights[task_id][n]
            penalty += (f * (p - p_old) ** 2).sum()
    return penalty
```

**What it does:** Computes the EWC penalty term summed over all previously seen tasks.

**The math:** The EWC loss for a new task B, given prior tasks, is:

```
L_total = L_B(θ) + (λ/2) * Σ_tasks Σ_i F_i * (θ_i - θ_i*)²
```

Where:
- `L_B(θ)` — the standard cross-entropy loss on the new task
- `F_i` — the Fisher importance of parameter `i` for the old task
- `θ_i*` — the optimal weight value learned after the old task
- `λ` — a hyperparameter controlling how strongly to protect old weights

Intuitively: weights that were important for old tasks (high Fisher) are penalized heavily if they deviate from their previously learned values. The penalty accumulates across **all previously seen tasks**, not just the most recent one.

---

### `test_model` — Evaluation

```python
def test_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            pred = model(data).argmax(dim=1)
            correct += pred.eq(target).sum().item()
    return correct / len(test_loader.dataset)
```

Standard accuracy evaluation. Returns accuracy as a **fraction** (0.0–1.0), not a percentage. Both `data` and `target` are moved to device — important for GPU compatibility.

---

## Section 5 — Hyperparameters and Model Initialization

```python
torch.manual_seed(0)

epochs_per_task = 20
lr              = 1e-3
lamda           = 5000
l2_weight_decay = 1e-5

model_ewc     = Net().to(device)
model_l2      = Net().to(device)
model_sgd     = Net().to(device)
model_dropout = NetDropout().to(device)

opt_ewc     = optim.SGD(model_ewc.parameters(),     lr=lr, momentum=0.9)
opt_l2      = optim.SGD(model_l2.parameters(),      lr=lr, momentum=0.9, weight_decay=l2_weight_decay)
opt_sgd     = optim.SGD(model_sgd.parameters(),     lr=lr, momentum=0.9)
opt_dropout = optim.SGD(model_dropout.parameters(), lr=lr, momentum=0.9)

fisher_matrices = {}
opt_weights     = {}

history_ewc     = {i: [] for i in range(num_tasks)}
history_l2      = {i: [] for i in range(num_tasks)}
history_sgd     = {i: [] for i in range(num_tasks)}
history_dropout = {i: [] for i in range(num_tasks)}
```

**Four models are initialized** — each representing a different training strategy being compared:

| Model | Strategy | Description |
|---|---|---|
| `model_ewc` | EWC | Cross-entropy + Fisher penalty (our method) |
| `model_sgd` | SGD | Cross-entropy only, no memory |
| `model_l2` | L2 regularization | Weight decay via `weight_decay=1e-5` |
| `model_dropout` | SGD + Dropout | Uses `NetDropout` with 20%/50% dropout rates |

**Key hyperparameters:**

| Parameter | Value | Role |
|---|---|---|
| `epochs_per_task` | 20 | Training epochs per task (more than the paper for visible forgetting) |
| `lr` | 1e-3 | Learning rate |
| `lamda` (λ) | 5000 | EWC penalty strength |
| `l2_weight_decay` | 1e-5 | L2 regularization strength (kept weak so it degrades more than EWC) |

**EWC relevance:** `lamda` is the most sensitive hyperparameter. Too low → EWC doesn't protect old tasks. Too high → the model can't learn new tasks at all. 5000 is chosen here to produce clear visual separation between EWC and the baselines.

The `fisher_matrices` and `opt_weights` dictionaries accumulate knowledge from all previously seen tasks and grow with each new task.

---

## Section 6 — The Main Training Loop

```python
for task_id, task_data in tasks.items():
    print(f'\n=== Task {task_id + 1}/{num_tasks} ===')
    train_loader = task_data['train']

    for epoch in range(epochs_per_task):
        model_ewc.train(); model_l2.train(); model_sgd.train(); model_dropout.train()

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            # SGD baseline
            opt_sgd.zero_grad()
            F.cross_entropy(model_sgd(data), target).backward()
            opt_sgd.step()

            # L2 baseline
            opt_l2.zero_grad()
            F.cross_entropy(model_l2(data), target).backward()
            opt_l2.step()

            # SGD+Dropout baseline
            opt_dropout.zero_grad()
            F.cross_entropy(model_dropout(data), target).backward()
            opt_dropout.step()

            # EWC
            opt_ewc.zero_grad()
            loss_ewc = F.cross_entropy(model_ewc(data), target)
            if fisher_matrices:
                loss_ewc += (lamda / 2) * ewc_penalty(model_ewc, fisher_matrices, opt_weights)
            loss_ewc.backward()
            opt_ewc.step()

        for tid in range(num_tasks):
            history_ewc[tid].append(    test_model(model_ewc,     tasks[tid]['test']))
            history_l2[tid].append(     test_model(model_l2,      tasks[tid]['test']))
            history_sgd[tid].append(    test_model(model_sgd,     tasks[tid]['test']))
            history_dropout[tid].append(test_model(model_dropout, tasks[tid]['test']))

        if (epoch + 1) % 5 == 0:
            print(f'  Epoch {epoch+1:2d} | EWC A={history_ewc[0][-1]:.3f} | SGD A={history_sgd[0][-1]:.3f}')

        total_epochs += 1

    print(f'  Computing Fisher for task {task_id + 1}...')
    fisher_matrices[task_id] = compute_fisher(model_ewc, train_loader)
    opt_weights[task_id] = {n: p.data.clone() for n, p in model_ewc.named_parameters()}
```

**What it does:** Iterates through all 10 tasks sequentially. Within each task, all four models train for 20 epochs on that task's data. All four models share the same mini-batches.

**Critical EWC workflow per task:**
1. **Train** with EWC loss (cross-entropy + penalty from all past tasks). The `if fisher_matrices:` check skips the penalty on Task 0 since there is no prior.
2. **After training completes** on the task, compute the Fisher matrix over that task's training data
3. **Clone and store** the current weights as the "optimal weights" for this task
4. The penalty dictionary grows: Task 2 is penalized by Fisher from Task 1; Task 3 by Fisher from Tasks 1 and 2; and so on

**Accuracy tracking:** After every epoch, *all four models* are evaluated on *all 10 task test sets*. This records the complete forgetting trajectory in real time. Progress is printed every 5 epochs showing Task A accuracy for EWC vs SGD.

---

## Section 7 — Figure 2A: Training Curves for Tasks A, B, C

```python
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
task_labels = ['Task A', 'Task B', 'Task C']
x = range(1, total_epochs + 1)

for ax, tid, label in zip(axes, [0, 1, 2], task_labels):
    ax.plot(x, history_ewc[tid], color='red',   label='EWC', linewidth=1.5)
    ax.plot(x, history_l2[tid],  color='green',  label='L2',  linewidth=1.5)
    ax.plot(x, history_sgd[tid], color='blue',   label='SGD', linewidth=1.5)

    for i in range(1, num_tasks):
        ax.axvline(x=epochs_per_task * i, color='black', linestyle=':', alpha=0.4, linewidth=0.8)

    ax.set_ylabel(f'{label}\nFrac. correct', fontsize=9)
    ax.set_ylim([0.0, 1.05])
    ax.set_yticks([0.8, 0.9, 1.0])
    ...
```

**What it shows:** The accuracy curves for the first three tasks (A, B, C) across all 200 training epochs (10 tasks × 20 epochs each). Three methods are plotted per panel: EWC (red), L2 (green), SGD (blue). Vertical dotted lines mark task boundaries.

**What to expect:**
- **SGD Task A** (blue): rises during Task A training, then collapses as Tasks B–J overwrite those weights
- **EWC Task A** (red): rises and *stays high* even as new tasks are trained — the Fisher penalty protects it
- **L2 Task A** (green): intermediate performance — weight decay provides some regularization but cannot selectively protect task-critical weights the way EWC can

**EWC relevance:** This is the qualitative demonstration of EWC working. EWC's accuracy remains stable while SGD's falls — this is **preventing catastrophic forgetting** in action.

---

## Section 7B — Figure 2B: Average Accuracy Across All Seen Tasks

```python
single_task_perf = max(history_ewc[0][:epochs_per_task])

x_tasks     = list(range(2, num_tasks + 1))
ewc_avg     = []
dropout_avg = []

for i in range(1, num_tasks):
    end_epoch = (i + 1) * epochs_per_task - 1
    ewc_avg.append(    np.mean([history_ewc[t][end_epoch]     for t in range(i + 1)]))
    dropout_avg.append(np.mean([history_dropout[t][end_epoch] for t in range(i + 1)]))
```

**What it shows:** After training on each task, computes the *average accuracy across all tasks seen so far* for EWC vs SGD+Dropout. A dashed horizontal line marks the single-task performance ceiling (best Task A accuracy before any subsequent tasks). This is the canonical continual learning metric from the original EWC paper (Figure 2B).

**What to expect:**
- **EWC**: average accuracy stays near the single-task performance dashed line — it retains past knowledge
- **SGD+Dropout**: average accuracy degrades progressively as more tasks are added, since new tasks erode old memories
- A flat EWC curve close to the single-task baseline is the ideal result

---

## Section 8 — Figure 2C: Fisher Overlap Across Layers

This section is an advanced analysis experiment replicating Figure 2C from the EWC paper.

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
perm_high = create_partial_permutation(26)   # shuffles nearly the entire image
```

**What it does:** Instead of shuffling all pixels, this shuffles only a square region in the centre of each image. A small region (8×8) produces a task *very similar* to MNIST. A large region (26×26) produces a task *very different* from MNIST.

**EWC relevance:** This controls **task similarity**. The Fisher overlap between tasks measures how much the two tasks rely on the same parameters.

### Training Setup for the Overlap Experiment

Two separate `NetDeep` models are trained sequentially — one for the low-permutation pair (base → 8×8 shuffle) and one for the high-permutation pair (base → 26×26 shuffle). Each model is trained for **100 epochs per task** (deliberately longer so Fisher distributions can specialize). Fisher is then computed over 8192 samples.

```python
fisher_A_low  = compute_fisher(model_low,  loader_base, num_samples=8192)
fisher_B_low  = compute_fisher(model_low,  loader_low,  num_samples=8192)
fisher_A_high = compute_fisher(model_high, loader_base, num_samples=8192)
fisher_B_high = compute_fisher(model_high, loader_high, num_samples=8192)
```

### The Overlap Metric (Fréchet / Hellinger Distance)

```python
def calculate_overlap(f1, f2, layer_name, all_layers):
    v1 = torch.cat([f1[layer_name+'.weight'].flatten(),
                    f1[layer_name+'.bias'].flatten()]).clamp(min=0)
    v2 = torch.cat([f2[layer_name+'.weight'].flatten(),
                    f2[layer_name+'.bias'].flatten()]).clamp(min=0)

    # normalise each to unit trace per layer
    v1 = v1 / (v1.sum() + 1e-10)
    v2 = v2 / (v2.sum() + 1e-10)

    # Fréchet distance on diagonal matrices: d² = 0.5 * ||sqrt(F1) - sqrt(F2)||²_F
    d2 = 0.5 * torch.sum((torch.sqrt(v1) - torch.sqrt(v2)) ** 2)
    return 1.0 - d2.item()
```

**What it does:** Computes the **overlap** (similarity) between the Fisher distributions of two tasks at a given layer. Weight and bias Fisher values are concatenated and clamped to non-negative (Fisher values should always be ≥ 0, the clamp guards against numerical noise). Both are then normalized to unit sum and compared via squared Hellinger distance.

Overlap = 1 means the tasks rely on *identical* parameters; Overlap = 0 means they rely on *completely different* parameters.

```python
layers = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6']
overlap_low  = [calculate_overlap(fisher_A_low,  fisher_B_low,  l, layers) for l in layers]
overlap_high = [calculate_overlap(fisher_A_high, fisher_B_high, l, layers) for l in layers]
```

Computes the 6-layer overlap profile for both the low-permutation and high-permutation task pairs.

### What the Graph Shows

The final plot — **Fisher Overlap vs. Layer Depth** — reveals two key insights from the original EWC paper:

1. **High-permutation tasks** (very different from MNIST) show *low overlap* across all layers — the tasks use different parts of the network
2. **Low-permutation tasks** (similar to MNIST) show *higher overlap* — the tasks share important parameters
3. **The overlap tends to decrease in deeper layers.** Earlier layers learn general low-level features (edges, textures) shared across tasks. Deeper layers specialize more per-task, so their Fisher distributions diverge. This justifies why EWC must protect weights *selectively* — the Fisher matrix acts as a precision mask that is naturally weaker in shared early layers and stronger in task-specific deep layers.

---

## Summary: The EWC Algorithm in Full

| Step | Description |
|---|---|
| **1. Train on Task A** | Standard cross-entropy training |
| **2. Compute Fisher (Task A)** | Run forward/backward on Task A data; store squared gradients (weighted average over up to 1024 samples) |
| **3. Save weights (Task A)** | Clone current parameters as `θ_A*` |
| **4. Train on Task B** | Cross-entropy + λ/2 × Σ F_A × (θ − θ_A*)² |
| **5. Compute Fisher (Task B)** | Repeat for Task B |
| **6. Save weights (Task B)** | Accumulate `opt_weights[B]` |
| **7. Train on Task C** | Loss includes penalties from *both* Task A and Task B |
| **...** | Repeat, growing the penalty sum across all past tasks |

The key insight: **the Fisher Information Matrix acts as a Bayesian prior**, encoding which weights matter for past tasks so that the optimizer can balance learning new information against forgetting old information.

---

## Further Reading

- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)
- A diagonal approximation of the FIM is used here for tractability; full-matrix EWC is computationally expensive.
- Related methods: Progressive Neural Networks, PackNet, Synaptic Intelligence (SI), Memory Replay.

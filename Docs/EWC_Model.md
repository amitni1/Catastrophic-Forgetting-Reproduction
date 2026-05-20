# Elastic Weight Consolidation (EWC) — Notebook Walkthrough

> This document explains each section of the notebook step by step, covering both *what* the code does and *why* it matters in the context of Elastic Weight Consolidation (EWC) for continual learning.

---

## Background: The Catastrophic Forgetting Problem

When a neural network is trained sequentially on multiple tasks, it tends to **catastrophically forget** earlier tasks — the gradient updates for the new task overwrite the weights that were important for the old one. EWC, introduced by Kirkpatrick et al. (2017), addresses this by adding a regularization penalty that slows down changes to weights that were important for previous tasks. The "importance" of each weight is measured using the **Fisher Information Matrix**.

---

## Section 1 — Imports and Device Setup

```python
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

**What it does:** Standard PyTorch setup. Imports the necessary libraries and selects GPU if available, otherwise CPU.

**EWC relevance:** EWC involves computing Fisher matrices and cloning model weights — both are memory-intensive operations. Running on GPU significantly speeds this up for larger experiments.

---

## Section 2 — Data: Permuted MNIST Tasks

```python
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST('./data', train=False, transform=transform)
```

### The `PermutedMNIST` Dataset Class

```python
class PermutedMNIST(torch.utils.data.Dataset):
    def __init__(self, dataset, permutation=None):
        self.dataset = dataset
        self.permutation = permutation

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.permutation is not None:
            img = img.view(-1)[self.permutation].view(1, 28, 28)
        return img, label
```

**What it does:** Wraps MNIST and optionally shuffles the pixels of each image according to a fixed random permutation. Task 0 has no permutation (standard MNIST). Tasks 1–9 each have a unique random pixel shuffle applied to every image.

**EWC relevance:** Permuted MNIST is the classic benchmark for continual learning. Each permuted version is a *structurally different* task — the labels are the same (0–9), but the input distribution changes completely. A model trained on Task 1 must not forget Task 0. This is exactly the scenario EWC was designed for.

```python
num_tasks = 10
tasks = {}
for i in range(num_tasks):
    perm = None if i == 0 else torch.randperm(28 * 28)
    tasks[i] = {
        'train': DataLoader(PermutedMNIST(train_dataset, perm), batch_size=64, shuffle=True),
        'test':  DataLoader(PermutedMNIST(test_dataset,  perm), batch_size=1000, shuffle=False)
    }
```

Creates 10 tasks. Task 0 = standard MNIST, Tasks 1–9 = different pixel permutations. Both train and test loaders are stored per task.

---

## Section 3 — The Neural Network Architecture

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc2 = nn.Linear(400, 400)
        self.fc3 = nn.Linear(400, 400)
        self.fc4 = nn.Linear(400, 400)
        self.fc5 = nn.Linear(400, 400)
        self.fc6 = nn.Linear(400, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc6(x)
```

**What it does:** A fully-connected (MLP) network with 6 linear layers. The first 5 hidden layers each have 400 units with ReLU activations. The final layer maps to 10 class logits.

**Architecture choices:**
- Input: 784 pixels (28×28 flattened)
- 5 hidden layers × 400 units — a deep-enough network to stress-test forgetting
- No softmax in `forward` (cross-entropy loss handles this internally)

**EWC relevance:** This depth is deliberate. A key result in the EWC paper is that Fisher overlap (the degree to which different tasks share important parameters) *varies across layers*. Deeper layers tend to be more task-specific, while earlier layers are shared. The later Fisher Overlap experiment (Section 8) directly investigates this.

---

## Section 4 — Core EWC Functions

This is the mathematical heart of the notebook.

### `compute_fisher` — Computing the Fisher Information Matrix

```python
def compute_fisher(model, data_loader):
    fisher_matrix = {name: torch.zeros_like(param.data)
                     for name, param in model.named_parameters()}
    model.eval()
    for data, target in data_loader:
        model.zero_grad()
        output = model(data.to(device))
        loss = F.nll_loss(F.log_softmax(output, dim=1), target.to(device))
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_matrix[name] += param.grad.data ** 2 / len(data_loader)
    return fisher_matrix
```

**What it does:** Computes the **empirical Fisher Information Matrix** for each parameter.

**The math:** The Fisher Information Matrix (FIM) measures how sensitive the model's output is to each parameter. For EWC, the diagonal of the FIM is used (one scalar per weight), computed as the squared gradient of the log-likelihood averaged over the dataset:

```
F_i = E[ (∂ log p(y|x,θ) / ∂θ_i)² ]
```

In practice, this is approximated by:
1. Running a forward pass on all data
2. Computing the NLL loss using log-softmax output
3. Running backprop to get gradients
4. Squaring the gradients and averaging across all mini-batches

A **large Fisher value** for a parameter means that parameter is important for the current task — changing it would significantly hurt performance. EWC uses this to selectively protect critical weights.

---

### `ewc_penalty` — The EWC Regularization Term

```python
def ewc_penalty(model, fisher_matrices, opt_weights):
    penalty = 0
    for task_id in fisher_matrices:
        for name, param in model.named_parameters():
            fisher = fisher_matrices[task_id][name]
            opt_w  = opt_weights[task_id][name]
            penalty += (fisher * (param - opt_w) ** 2).sum()
    return penalty
```

**What it does:** Computes the EWC penalty term, which is added to the loss when training on a new task.

**The math:** The EWC loss for a new task B, given a previous task A, is:

```
L_total = L_B(θ) + (λ/2) * Σ_i F_i^A * (θ_i - θ_i^A*)²
```

Where:
- `L_B(θ)` — the standard cross-entropy loss on the new task
- `F_i^A` — the Fisher importance of parameter `i` for the old task A
- `θ_i^A*` — the optimal weight value learned after task A
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
            output = model(data.to(device))
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)
```

Standard accuracy evaluation — sets the model to eval mode (disabling dropout/BatchNorm training behavior), runs inference, and returns accuracy as a percentage.

---

## Section 5 — Hyperparameters and Model Initialization

```python
epochs_per_task = 3
lr     = 0.005
lamda  = 1000

model_sgd = Net().to(device)
model_ewc = Net().to(device)

optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=lr, momentum=0.9)
optimizer_ewc = optim.SGD(model_ewc.parameters(), lr=lr, momentum=0.9)

fisher_matrices = {}
opt_weights     = {}

history_sgd = {i: [] for i in range(num_tasks)}
history_ewc = {i: [] for i in range(num_tasks)}
```

**Two models are initialized identically** — `model_sgd` (the baseline, no memory) and `model_ewc` (with EWC protection). Both use SGD with momentum.

**Key hyperparameters:**

| Parameter | Value | Role |
|---|---|---|
| `epochs_per_task` | 3 | Training epochs per task |
| `lr` | 0.005 | Learning rate (10× larger than typical) |
| `lamda` (λ) | 1000 | EWC penalty strength |

**EWC relevance:** `lamda` is the most sensitive hyperparameter. Too low → EWC doesn't protect old tasks. Too high → the model can't learn new tasks at all. 1000 is a value in the range used in the original paper for this benchmark.

The `fisher_matrices` and `opt_weights` dictionaries accumulate knowledge from all previously seen tasks and grow with each new task.

---

## Section 6 — The Main Training Loop

```python
for task_id, task_data in tasks.items():
    for epoch in range(epochs_per_task):
        for batch_idx, (data, target) in enumerate(task_data['train']):

            # --- Standard SGD (no memory) ---
            optimizer_sgd.zero_grad()
            loss_sgd = F.cross_entropy(model_sgd(data), target)
            loss_sgd.backward()
            optimizer_sgd.step()

            # --- EWC training ---
            optimizer_ewc.zero_grad()
            loss_ce  = F.cross_entropy(model_ewc(data), target)
            loss_ewc = loss_ce
            if len(fisher_matrices) > 0:
                loss_ewc += (lamda / 2) * ewc_penalty(model_ewc, fisher_matrices, opt_weights)
            loss_ewc.backward()
            optimizer_ewc.step()

    # After each task: compute Fisher and save optimal weights
    fisher_matrices[task_id] = compute_fisher(model_ewc, task_data['train'])
    opt_weights[task_id] = {name: param.data.clone()
                            for name, param in model_ewc.named_parameters()}
```

**What it does:** Iterates through all 10 tasks sequentially. Within each task, both models train for 3 epochs.

**Critical EWC workflow per task:**
1. **Train** with EWC loss (cross-entropy + penalty from all past tasks)
2. **After training completes** on the task, compute the Fisher matrix over that task's data
3. **Clone and store** the current weights as the "optimal weights" for this task
4. The penalty dictionary grows: task 2 is penalized by Fisher from task 1; task 3 by Fisher from tasks 1 and 2; and so on

**SGD baseline** trains with only cross-entropy — no memory of previous tasks. It inevitably overwrites old task knowledge as new tasks are learned.

**Accuracy tracking:** After every epoch, the model is evaluated on *all 10 tasks*, not just the current one. This tracks forgetting in real time. The console output format (`EWC: [98%, 85%, 72%] | SGD: [97%, 20%, 48%]`) makes it easy to see past task accuracy collapsing for SGD but staying high for EWC.

---

## Section 7 — Visualization: Task A, B, C Accuracy Over Time

```python
plt.plot(x_axis, history_ewc[0], label='EWC Task A', color='red',   linestyle='-')
plt.plot(x_axis, history_sgd[0], label='SGD Task A', color='red',   linestyle='--')
plt.plot(x_axis, history_ewc[1], label='EWC Task B', color='green', linestyle='-')
plt.plot(x_axis, history_sgd[1], label='SGD Task B', color='green', linestyle='--')
plt.plot(x_axis, history_ewc[2], label='EWC Task C', color='blue',  linestyle='-')
plt.plot(x_axis, history_sgd[2], label='SGD Task C', color='blue',  linestyle='--')
```

**What it shows:** The accuracy curves for the first three tasks (A, B, C) across all 30 training epochs (10 tasks × 3 epochs each).

**What to expect:**
- **SGD Task A** (red dashed): rises during task A training, then collapses as tasks B through J overwrite those weights
- **EWC Task A** (red solid): rises and *stays high* even as new tasks are trained — the Fisher penalty protects it
- Vertical dotted lines mark task boundaries

**EWC relevance:** This is the qualitative demonstration of EWC working. The solid lines (EWC) remain stable while the dashed lines (SGD) fall — this is **preventing catastrophic forgetting** in action.

---

## Section 7b — Average Accuracy Across All Seen Tasks

```python
for i in range(1, num_tasks):
    end_epoch   = (i + 1) * epochs_per_task - 1
    seen_tasks  = task_ids[:i+1]
    ewc_avg_acc.append(np.mean([history_ewc[t][end_epoch] for t in seen_tasks]))
    sgd_avg_acc.append(np.mean([history_sgd[t][end_epoch] for t in seen_tasks]))
```

**What it shows:** After training on each task, computes the *average accuracy across all tasks seen so far*. This is the canonical continual learning metric from the original EWC paper (Figure 2B).

**What to expect:**
- **EWC**: average accuracy stays near the "single task performance" dashed line — it retains past knowledge
- **SGD**: average accuracy degrades progressively as more tasks are added, since new tasks erode old memories
- A flat EWC curve close to the single-task baseline is the ideal result

---

## Section 8 — Fisher Overlap Across Layers

This section is an advanced analysis experiment, replicating Figure 3 from the EWC paper.

### Partial Permutation Generator

```python
def create_partial_permutation(size):
    perm = torch.arange(28 * 28)
    grid = perm.view(28, 28)
    start = (28 - size) // 2
    end   = start + size
    region = grid[start:end, start:end].flatten()
    grid[start:end, start:end] = region[torch.randperm(len(region))].view(size, size)
    return grid.flatten()

perm_low  = create_partial_permutation(8)   # shuffles an 8×8 central region
perm_high = create_partial_permutation(26)  # shuffles nearly the entire image
```

**What it does:** Instead of shuffling all pixels, this shuffles only a square region in the centre of each image. A small region (8×8) produces a *similar* task to MNIST. A large region (26×26) produces a very *different* task.

**EWC relevance:** This controls **task similarity**. The Fisher overlap between tasks measures how much the two tasks rely on the same parameters.

### The Overlap Metric (Bhattacharyya-inspired)

```python
def calculate_overlap(f1, f2, layer_name):
    v1_hat = v1 / (v1.sum() + 1e-10)   # normalize to unit-sum
    v2_hat = v2 / (v2.sum() + 1e-10)
    d2 = 0.5 * torch.sum((torch.sqrt(v1_hat) - torch.sqrt(v2_hat))**2)
    return max(0.0, 1.0 - d2.item())   # overlap = 1 - Hellinger distance²
```

**What it does:** Computes the **overlap** (similarity) between the Fisher distributions of two tasks at a given layer. Both Fisher vectors are normalized to unit sum, then their Hellinger distance is computed. Overlap = 1 means the tasks rely on *identical* parameters; Overlap = 0 means they rely on *completely different* parameters.

```python
layers = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6']
overlap_low  = [calculate_overlap(fisher_base, fisher_low,  l) for l in layers]
overlap_high = [calculate_overlap(fisher_base, fisher_high, l) for l in layers]
```

Computes the 6-layer overlap profile for both the low-permutation and high-permutation tasks.

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
| **2. Compute Fisher (Task A)** | Run forward/backward on Task A data; store squared gradients |
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

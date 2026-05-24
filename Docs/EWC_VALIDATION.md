# EWC Model Validation & Testing

This document describes how we validate and check the **Elastic Weight Consolidation (EWC)** implementation in our continual learning experiments on Permuted MNIST, replicating Figures 2A, 2B, and 2C from *Overcoming Catastrophic Forgetting in Neural Networks* (Kirkpatrick et al. 2017).

---

## Overview

Our validation strategy operates on three levels:

1. **Per-epoch accuracy tracking** — live monitoring during training across all tasks
2. **Average accuracy curves** — a quantitative summary of catastrophic forgetting over the full task sequence (Figure 2B)
3. **Fisher Information overlap analysis** — a mechanistic check that the Fisher matrix correctly captures task-relevant parameters (Figure 2C)

---

## 1. Experimental Setup

We validate EWC against three baselines — plain SGD, L2 regularisation, and SGD+dropout — on the **Permuted MNIST** benchmark: 10 sequentially presented tasks, each a different random pixel permutation of MNIST.

| Hyperparameter | Value |
|---|---|
| Tasks | 10 (Permuted MNIST) |
| Epochs per task | 20 |
| Learning rate | 0.001 |
| Momentum | 0.9 |
| EWC λ (lambda) | 5000 |
| L2 weight decay | 1e-5 |
| Batch size | 256 |
| Network (Figs 2A/2B) | 3-layer MLP (784 → 400 → 400 → 10) |
| Network (Fig 2C) | 7-layer MLP (784 → 200×6 → 10) |

Tasks are generated with a fixed seed (`torch.manual_seed(42)`) so permutations are reproducible. All four models (`model_ewc`, `model_l2`, `model_sgd`, `model_dropout`) share the same architecture (`Net` or `NetDropout`) and the same SGD + momentum optimiser, so any difference in outcomes is attributable solely to the regularisation strategy.

---

## 2. Model Architectures

### `Net` — used for Figures 2A and 2B

A 3-layer MLP with ReLU activations:

```python
class Net(nn.Module):
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

### `NetDropout` — SGD+dropout baseline for Figure 2B

Same depth as `Net` but with input dropout (p=0.2) and hidden dropout (p=0.5):

```python
class NetDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.drop_in = nn.Dropout(0.2)
        self.fc1     = nn.Linear(784, 400)
        self.drop_h  = nn.Dropout(0.5)
        self.fc2     = nn.Linear(400, 400)
        self.fc3     = nn.Linear(400, 10)
```

### `NetDeep` — used for Figure 2C (Fisher overlap)

A 7-layer MLP with 6 hidden layers of 200 units each:

```python
class NetDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 200)
        self.fc4 = nn.Linear(200, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 200)
        self.fc7 = nn.Linear(200, 10)
```

---

## 3. The EWC Penalty

After finishing each task, two quantities are stored and used in all future training steps.

### Fisher Information Matrix

Computed empirically over the completed task's training data (up to `num_samples=1024` samples). The Fisher is accumulated as the **mean squared gradient**, weighted by batch size and normalised by total samples seen:

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

Using `log_softmax` + `nll_loss` (equivalent to `cross_entropy`) gives a numerically more precise empirical Fisher estimate. Accumulating `grad² * batch_size` before dividing by `samples_seen` correctly weights each batch by the number of samples it contributes.

### Optimal weights snapshot

A `clone()` of every parameter immediately after each task finishes:

```python
opt_weights[task_id] = {n: p.data.clone() for n, p in model_ewc.named_parameters()}
```

### EWC penalty function

The penalty sums the quadratic deviation from all previous optimal weights, weighted by their Fisher importance:

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

The total EWC loss during task `t` is therefore:

```
L_total = L_cross_entropy + (λ / 2) * Σ_{i<t} Σ_j F_j^i (θ_j - θ*_j^i)²
```

where the outer sum runs over all previous tasks and the inner sum over all parameters.

---

## 4. Training Loop

All four models are trained in lockstep — sharing the same batches — so any difference in outcomes is purely due to regularisation:

```python
for data, target in train_loader:
    # Plain SGD
    opt_sgd.zero_grad()
    F.cross_entropy(model_sgd(data), target).backward()
    opt_sgd.step()

    # L2 regularisation (via weight_decay in optimiser)
    opt_l2.zero_grad()
    F.cross_entropy(model_l2(data), target).backward()
    opt_l2.step()

    # SGD + dropout
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
```

After each task, the Fisher matrix and optimal weights are saved for EWC only:

```python
fisher_matrices[task_id] = compute_fisher(model_ewc, train_loader)
opt_weights[task_id] = {n: p.data.clone() for n, p in model_ewc.named_parameters()}
```

---

## 5. Per-Epoch Accuracy Tracking

After every epoch, `test_model` is called on **every task's test set** for all four models:

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

Results are appended to `history_ewc[task_id]`, `history_l2[task_id]`, `history_sgd[task_id]`, and `history_dropout[task_id]` for every task at every epoch, building a complete accuracy matrix of shape `(num_tasks, total_epochs)`.

### Console output

Every 5 epochs, Task A accuracy for EWC and SGD is printed:

```
Epoch  5 | EWC A=0.971 | SGD A=0.962
Epoch 10 | EWC A=0.974 | SGD A=0.581
...
```

This makes forgetting visible in real time: a healthy EWC run holds Task A accuracy near its post-training peak, while plain SGD shows a sharp decline once training moves to task B.

---

## 6. Visualisation — Task Accuracy Curves (Figure 2A)

The first plot tracks the first three tasks (A, B, C) across all 200 training epochs, comparing EWC (red), L2 (green), and SGD (blue):

```python
for ax, tid, label in zip(axes, [0, 1, 2], task_labels):
    ax.plot(x, history_ewc[tid], color='red',   label='EWC', linewidth=1.5)
    ax.plot(x, history_l2[tid],  color='green', label='L2',  linewidth=1.5)
    ax.plot(x, history_sgd[tid], color='blue',  label='SGD', linewidth=1.5)
```

Vertical dotted lines mark each task boundary (every 20 epochs). Task labels (`train A`, `train B`, …) are annotated above each segment so it is easy to see whether accuracy on earlier tasks drops after the model moves on.

**What a correct EWC implementation looks like:**

- Task A (EWC) — accuracy stays high even as tasks B–J are trained
- Task A (SGD) — accuracy falls sharply once training moves to task B
- Tasks B and C (EWC) — accumulate accuracy and broadly hold it
- Tasks B and C (SGD) — degrade progressively

---

## 7. Visualisation — Average Accuracy Curve (Figure 2B)

The second plot reproduces **Figure 2B** from the original EWC paper. After training on task `i`, we compute the mean accuracy across all tasks seen so far, comparing EWC against SGD+dropout:

```python
for i in range(1, num_tasks):
    end_epoch = (i + 1) * epochs_per_task - 1
    ewc_avg.append(    np.mean([history_ewc[t][end_epoch]     for t in range(i + 1)]))
    dropout_avg.append(np.mean([history_dropout[t][end_epoch] for t in range(i + 1)]))
```

A horizontal dashed line marks `single_task_perf`, defined as the peak accuracy achieved on Task A during its training window:

```python
single_task_perf = max(history_ewc[0][:epochs_per_task])
```

This provides an upper-bound reference for how well a model can perform when trained on just one task.

**What to look for:**

- EWC's average accuracy should remain close to the single-task baseline across all 10 tasks
- SGD+dropout's average accuracy should trend downward as the number of tasks grows
- A large gap between the two curves confirms that EWC successfully mitigates catastrophic forgetting

---

## 8. Fisher Overlap Analysis (Figure 2C)

Beyond accuracy, we validate the **internal behaviour** of the Fisher Information Matrix by measuring how much overlap exists between Fishers computed on sequentially trained task pairs with different levels of permutation.

### Partial permutation tasks

Two pairs of tasks are created using `NetDeep`. Each pair trains sequentially on a base task then a permuted variant:

| Pair | Permuted region | Expected Fisher overlap |
|---|---|---|
| LOW pair | 8×8 centre patch (`perm_low`) | High — tasks share most pixels |
| HIGH pair | 26×26 region (`perm_high`) | Low — tasks differ substantially |

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
```

Each model is trained for **100 epochs** on task A then 100 epochs on task B. The Fisher is then computed over each task's data using `num_samples=8192` for high-quality estimates.

### Overlap metric

Fisher overlap is computed using the **Fréchet / Hellinger distance** (following Appendix 4.3 of the original paper). Both Fisher vectors (weight + bias concatenated) are normalised to unit trace per layer, then:

```python
def calculate_overlap(f1, f2, layer_name, all_layers):
    v1 = torch.cat([f1[layer_name+'.weight'].flatten(),
                    f1[layer_name+'.bias'].flatten()]).clamp(min=0)
    v2 = torch.cat([f2[layer_name+'.weight'].flatten(),
                    f2[layer_name+'.bias'].flatten()]).clamp(min=0)

    # Normalise each to unit trace per layer
    v1 = v1 / (v1.sum() + 1e-10)
    v2 = v2 / (v2.sum() + 1e-10)

    # Fréchet distance on diagonal matrices: d² = 0.5 * ||sqrt(F1) - sqrt(F2)||²_F
    d2 = 0.5 * torch.sum((torch.sqrt(v1) - torch.sqrt(v2)) ** 2)
    return 1.0 - d2.item()
```

The `1e-10` guard prevents division by zero. The `clamp(min=0)` ensures no negative values are passed to `sqrt` (Fisher values should be non-negative as they are squared gradients, but numerical noise can introduce tiny negatives).

### What the overlap plot tells us

Overlap is computed layer-by-layer across all 6 hidden layers (`fc1`–`fc6`) of `NetDeep`:

- **High overlap (near 1.0)** — the two tasks rely on the same parameters; EWC will protect them aggressively
- **Low overlap (near 0.0)** — the tasks use different parameters; there is less conflict and less need for the penalty

Expected patterns (consistent with the paper):

- `perm_low` should show higher overlap than `perm_high` at all layers
- Overlap may decrease in deeper layers, reflecting more task-specific representations

---

## 9. Checklist for a Valid Run

Use this checklist to confirm the implementation is working correctly:

- [ ] Task A accuracy under EWC remains above 85% throughout all 10 tasks
- [ ] Task A accuracy under SGD falls below 30% by task 3 or later
- [ ] EWC average accuracy (Figure 2B) stays within ~10 percentage points of the single-task baseline
- [ ] SGD+dropout average accuracy (Figure 2B) trends downward across all 10 tasks
- [ ] `perm_low` Fisher overlap is consistently higher than `perm_high` across all 6 layers
- [ ] Fisher values are non-negative for all parameters (they are squared gradients, clamped to ≥ 0)
- [ ] `opt_weights` are saved via `.clone()`, not as references (otherwise they would update in place)
- [ ] `samples_seen` accumulates correctly across batches before normalising the Fisher

---

## 10. Known Limitations

- The empirical Fisher approximation (diagonal, computed over training data, capped at 1024 samples for the main loop) is a simplification of the full Fisher. It is sufficient for the Permuted MNIST benchmark.
- The EWC penalty grows with the number of tasks (O(T) per gradient step). For very large T, this becomes computationally expensive and an online EWC variant should be considered.
- Figure 2C uses separate model instances (`model_low`, `model_high`) trained from scratch on each pair, rather than a single shared pre-trained base. This matches the paper's intent of comparing Fisher matrices across independently trained task sequences.
- With 20 epochs per task, absolute accuracy numbers should be close to the original paper. The relative ordering (EWC ≥ L2 > SGD+dropout ≥ SGD) should hold clearly.

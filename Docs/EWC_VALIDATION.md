# EWC Model Validation & Testing

This document describes how we validate and check the **Elastic Weight Consolidation (EWC)** implementation in our continual learning experiments on Permuted MNIST, replicating Figures 2A, 2B, and 2C from *Overcoming Catastrophic Forgetting in Neural Networks* (Kirkpatrick et al. 2017).

---

## Overview

Our validation strategy operates on three levels:

1. **Per-epoch accuracy tracking** — live monitoring during training across all tasks
2. **Average accuracy curves** — a quantitative summary of catastrophic forgetting over the full task sequence (Figure 2B)
3. **Fisher Information overlap analysis** — a mechanistic check that the Fisher matrix correctly captures task-relevant parameters (Figure 2C)

### Why this version differs from earlier notebooks

Two failure modes were identified and fixed:

- **`fixed_1_4_2`**: batch-averaged Fisher under-estimated importance → Figure 2B average collapsed to ~0.8.
- **`FIXED_2`**: correct Fisher, but stiff lambda + gradient clipping + limited capacity over-constrained the network so it could not learn new tasks → Figure 2B average collapsed to ~0.7.

This notebook addresses both: per-sample Fisher (no batch-size under-estimate), no gradient clipping, wider network, and correct paper-exact separate-penalties EWC.

---

## 1. Experimental Setup

| Hyperparameter | Value |
|---|---|
| Tasks | 10 (Permuted MNIST) |
| Epochs per task | 20 |
| Learning rate | 0.001 |
| Momentum | 0.9 |
| Batch size | 256 |
| Network width (Figs 2A/2B) | 2000 hidden units |
| EWC λ (lambda) | 100 (tunable in config) |
| Fisher samples (main loop) | 2048 per task |
| L2 penalty strength | 1.0 (uniform quadratic penalty) |
| Early stop patience (dropout) | 5 epochs |
| Network (Fig 2C) | 7-layer MLP (784 → 100×6 → 10) |
| Fisher samples (Fig 2C) | 8192 per task |

Task 0 is original MNIST; tasks 1–9 each apply a fixed random pixel permutation. A 10k validation split (held out from training) is used exclusively for the SGD+dropout baseline's early stopping — the EWC model never sees it.

> **Note on lambda tuning:** lambda and the Fisher scale are coupled, so values from older notebooks do **not** transfer. With summed per-task penalties the total constraint grows as tasks accumulate. Try values across ~2 orders of magnitude (e.g. 50, 150, 500, 1500, 5000) and keep the flattest, highest curve.

---

## 2. Model Architectures

### `Net` — used for Figures 2A and 2B

A 3-layer MLP with configurable width and ReLU activations:

```python
class Net(nn.Module):
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

Width defaults to 2000 in this notebook (the paper's Fig 2B maximum). More capacity means less interference between tasks and a stronger upper bound for EWC.

### `NetDropout` — SGD+dropout baseline for Figure 2B

Same depth as `Net` with input dropout (p=0.2) and hidden dropout (p=0.5):

```python
class NetDropout(nn.Module):
    def __init__(self, width=400):
        super().__init__()
        self.drop_in = nn.Dropout(0.2)
        self.fc1     = nn.Linear(784, width)
        self.drop_h  = nn.Dropout(0.5)
        self.fc2     = nn.Linear(width, width)
        self.fc3     = nn.Linear(width, 10)
```

### `NetDeep` — used for Figure 2C (Fisher overlap)

A 7-layer MLP with 6 hidden layers of 100 units each, independent of the main experiment:

```python
class NetDeep(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.fc4 = nn.Linear(100, 100)
        self.fc5 = nn.Linear(100, 100)
        self.fc6 = nn.Linear(100, 100)
        self.fc7 = nn.Linear(100, 10)
```

---

## 3. The EWC Penalty

### Correct per-sample Fisher (the key fix)

The Fisher is estimated **one sample at a time**, with the label sampled from the model's own predictive distribution rather than the ground-truth label. This is the true diagonal Fisher (the Laplace approximation from Section 2 of the paper), not the empirical batch-averaged approximation used in earlier notebooks:

```python
def compute_fisher(model, data_loader, num_samples=1024):
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
            y      = torch.multinomial(p, 1).squeeze(1)   # sample from model's own distribution
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

Processing one sample at a time avoids the 1/batch_size under-estimation that caused Figure 2B to collapse in earlier versions.

### Paper-exact separate penalties (equation 3)

A **separate** Fisher matrix and anchor (weight snapshot) are stored for **each** task and summed — the paper's "separate penalties" scheme, not the online-EWC variant:

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

The `normalize=True` flag divides the summed penalty by the number of completed tasks, preventing the total constraint from growing unboundedly as tasks accumulate. The training loop calls this helper without passing the argument, so it falls back to the default (`True`) and the penalty is always averaged; the `NORMALIZE_PENALTY` config flag is not currently wired into the loop. The total loss during task `t` is:

```
L_total = L_cross_entropy + (λ / 2) * (1/T) * Σ_{i<t} Σ_j F_j^i (θ_j - θ*_j^i)²
```

After each task, the Fisher and anchor are stored in the `ewc_tasks` list:

```python
new_fisher = compute_fisher(model, train_loader, fisher_samples)
new_anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
ewc_tasks.append({'fisher': new_fisher, 'anchor': new_anchor})
```

> **No gradient clipping** is applied to the EWC model. Gradient clipping was present in `FIXED_2` and was found to suppress the consolidation force, causing the average accuracy to collapse to ~0.7.

---

## 4. Single-Task Reference (the dashed line)

A dedicated model trained **only** on Task 0 provides the correct upper-bound reference for Figure 2B. This is independent of the EWC run and trained for the same number of epochs:

```python
torch.manual_seed(99)
model_single = Net(WIDTH).to(device)
opt_single   = optim.SGD(model_single.parameters(), lr=LR, momentum=MOMENTUM)

for ep in range(EPOCHS_PER_TASK):
    model_single.train()
    for data, target in tasks[0]['train']:
        ...
        F.cross_entropy(model_single(data), target).backward()
        opt_single.step()

single_task_perf = test_model(model_single, tasks[0]['test'])
```

Using a separate model (rather than the EWC model's Task 0 accuracy) ensures the dashed line is not contaminated by any EWC penalty or later training.

---

## 5. Baselines

Three baselines are trained in a single loop for direct comparison:

**Plain SGD** — no regularisation; the clearest demonstration of catastrophic forgetting.

**L2 regularisation** — a uniform quadratic penalty anchored to the weights at the end of each task (same anchor mechanism as EWC, but with uniform importance rather than Fisher-weighted):

```python
if l2_anchor is not None:
    pen = sum(((p - l2_anchor[n]) ** 2).sum() for n, p in m_l2.named_parameters())
    loss_l2 = loss_l2 + (l2_lambda / 2.0) * pen
```

**SGD+dropout with per-task early stopping** — after each epoch the average validation accuracy across all seen tasks is checked; training stops if it fails to improve for `EARLY_STOP_PATIENCE=5` consecutive epochs, and the best checkpoint is restored. This is the Figure 2B comparison baseline.

---

## 6. Per-Epoch Accuracy Tracking

After every training epoch, `test_model` is called on **every task's test set** for all four models:

```python
def test_model(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            correct += model(data).argmax(dim=1).eq(target).sum().item()
    return correct / len(loader.dataset)
```

Results are stored in `history_ewc[task_id]`, `history_sgd[task_id]`, `history_l2[task_id]`, and `history_dropout[task_id]` for every task at every epoch, building accuracy matrices of shape `(num_tasks, total_epochs)`.

### Console output

After each task the current task accuracy and average over all previously seen tasks are printed:

```
task  1/10: current=0.971  prev_avg=nan
task  2/10: current=0.952  prev_avg=0.943
task  3/10: current=0.949  prev_avg=0.951
...
```

---

## 7. Visualisation — Task Accuracy Curves (Figure 2A)

The first plot tracks the first three tasks (A, B, C) across the first 60 training epochs (3 × `EPOCHS_PER_TASK`), comparing EWC (red), L2 (green), and SGD (blue):

```python
epochs_to_show = 3 * EPOCHS_PER_TASK
x = range(1, epochs_to_show + 1)

for ax, tid, label in zip(axes, [0, 1, 2], ['Task A', 'Task B', 'Task C']):
    ax.plot(x, history_ewc[tid][:epochs_to_show], color='red',   label='EWC', linewidth=1.5)
    ax.plot(x, history_l2[tid][:epochs_to_show],  color='green', label='L2',  linewidth=1.5)
    ax.plot(x, history_sgd[tid][:epochs_to_show], color='blue',  label='SGD', linewidth=1.5)
```

Vertical dashed lines mark each task boundary. Task labels (`train A`, `train B`, `train C`) are annotated at the top of each panel.

**What a correct EWC implementation looks like:**

- Task A (EWC, red solid) — accuracy stays high even as tasks B and C are trained
- Task A (SGD, blue solid) — accuracy falls sharply once training moves to task B
- Tasks B and C (EWC) — accumulate accuracy and broadly hold it
- Tasks B and C (SGD) — degrade progressively

---

## 8. Visualisation — Average Accuracy Curve (Figure 2B)

The second plot reproduces Figure 2B from the paper. After training on task `i`, the mean test accuracy across all tasks seen so far is computed for the EWC model and SGD+dropout baseline:

```python
# EWC snapshots — recorded inside train_ewc_sequence after each task
if task_id >= 1:
    seen = list(range(task_id + 1))
    snapshots.append(float(np.mean([test_model(model, tasks[t]['test']) for t in seen])))

# Dropout snapshots — recorded inside train_baselines after each task
drop_snaps.append(float(np.mean([test_model(m_drop, tasks[t]['test']) for t in seen])))
```

The horizontal dashed line marks `single_task_perf` (the dedicated single-task reference model's accuracy on Task 0 test set).

**What to look for:**

- EWC's average accuracy (red) should remain flat and close to the single-task reference
- SGD+dropout's average accuracy (blue) should trend downward across all 10 tasks
- A large gap between the two curves confirms successful mitigation of catastrophic forgetting

---

## 9. Fisher Overlap Analysis (Figure 2C)

Beyond accuracy, we validate the internal behaviour of the Fisher Information Matrix by measuring overlap between Fishers computed on sequentially trained task pairs with different levels of permutation.

### Partial permutation tasks

Two pairs of tasks are created using `NetDeep`. Each pair trains sequentially (task A then task B) on a base task and a partially-permuted variant:

| Pair | Permuted region | Expected Fisher overlap |
|---|---|---|
| LOW pair | 8×8 centre patch (~8% of pixels) | High — tasks share most pixels |
| HIGH pair | 26×26 region (~86% of pixels) | Low — tasks differ substantially |

```python
def create_partial_permutation(size):
    perm = torch.arange(28 * 28)
    grid = perm.view(28, 28).clone()
    start = (28 - size) // 2
    end   = start + size
    region   = grid[start:end, start:end].flatten()
    shuffled = region[torch.randperm(len(region))]
    grid[start:end, start:end] = shuffled.view(size, size)
    return grid.flatten()
```

Each pair trains a fresh `NetDeep` for `FIG2C_EPOCHS=100` epochs on task A, then 100 more epochs on task B. The Fisher is computed with `num_samples=8192` after each phase.

### Overlap metric — globally normalised Hellinger distance

Fisher overlap uses the **global normalisation** method from Appendix 4.3 of the paper. All six layers' Fisher values are concatenated into a single global vector, normalised to unit sum globally (not per-layer), then split back by layer to compute the per-layer Hellinger distance:

```python
def fisher_to_global_vector(fisher_dict, layer_names):
    parts = []
    for name in layer_names:
        parts.append(fisher_dict[name + '.weight'].flatten().clamp(min=0))
        parts.append(fisher_dict[name + '.bias'].flatten().clamp(min=0))
    return torch.cat(parts)

def calculate_overlap_correct(fisher_A, fisher_B, layer_names):
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

Global normalisation (dividing the full concatenated vector by its total sum) means each layer's overlap score reflects its relative importance within the whole network, not just within that layer — a more faithful implementation of the paper than per-layer normalisation.

### What the overlap plot tells us

Overlap is computed layer-by-layer across all 6 hidden layers (`fc1`–`fc6`) of `NetDeep`:

- **High overlap (near 1.0)** — the two tasks rely on the same parameters; EWC will protect them aggressively
- **Low overlap (near 0.0)** — the tasks use different parameters; less conflict and less need for the penalty

Expected patterns:

- `perm_low` (grey dash-dot) should show higher overlap than `perm_high` (black dashed) at all layers
- Overlap may decrease in deeper layers, reflecting increasingly task-specific representations

---

## 10. Checklist for a Valid Run

Use this checklist to confirm the implementation is working correctly:

- [ ] Task A accuracy under EWC remains above 85% throughout all 10 tasks
- [ ] Task A accuracy under SGD falls sharply (below 50%) once training moves to task B
- [ ] EWC average accuracy (Figure 2B) stays flat and within ~10 percentage points of the single-task reference
- [ ] SGD+dropout average accuracy (Figure 2B) trends downward across all 10 tasks
- [ ] `perm_low` Fisher overlap is consistently higher than `perm_high` across all 6 layers
- [ ] Fisher values are non-negative (clamped with `.clamp(min=0)` before normalisation)
- [ ] Anchors are saved via `.detach().clone()`, not as references (otherwise they update in place)
- [ ] `ewc_tasks` grows by one entry per task, confirming separate-penalties accumulation
- [ ] The single-task reference model is trained independently (not taken from the EWC model's Task 0 history)

---

## 11. Troubleshooting — If the Red Line Still Droops

In rough order of impact:

1. **Try other `LAMBDA` values.** Change `LAMBDA` in the config and re-run from the EWC-run cell. Try values across ~2 orders of magnitude (e.g. 50, 150, 500, 1500, 5000). If the best value sits at an extreme, push further in that direction.
2. **More capacity / longer training.** Raise `WIDTH` toward 2000 and `EPOCHS_PER_TASK` toward 40–100 (the paper's Fig 2B recipe).
3. **Lower the learning rate.** Try 5e-4 instead of clipping. With many summed penalties the loss surface becomes stiff; a smaller step is more stable.
4. **More Fisher samples.** Raise `FISHER_SAMPLES` for a lower-variance importance estimate.
5. **Average over seeds.** Re-run `train_ewc_sequence(LAMBDA, seed=...)` for a few seeds and average the snapshots; single-seed curves dip noisily.

---

## 12. Known Limitations

- The diagonal true Fisher (estimated per-sample, with labels sampled from the model's own distribution) is a simplification of the full Fisher. It is sufficient for the Permuted MNIST benchmark.
- The penalty grows as O(T) per gradient step (one Fisher + anchor stored per task). For very large T, an online EWC variant should be considered.
- Figure 2C trains two fully independent model pairs and does not share weights or Fisher data with the Figure 2A/2B experiment. It also does not use `LAMBDA`; the overlap analysis is purely mechanistic.
- The 10k validation split used for SGD+dropout early stopping is held out from training for all models, so it does not contaminate test accuracy measurements.

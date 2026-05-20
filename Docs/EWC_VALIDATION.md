# EWC Model Validation & Testing

This document describes how we validate and check the **Elastic Weight Consolidation (EWC)** implementation in our continual learning experiments on Permuted MNIST.

> note : the testing were done at 3 epoch cycles and on multiple task to streamline testing, the testing still apply to 10 epoch cycles used in the main code
---

## Overview

Our validation strategy operates on three levels:

1. **Per-epoch accuracy tracking** — live monitoring during training across all tasks
2. **Average accuracy curves** — a quantitative summary of catastrophic forgetting over the full task sequence
3. **Fisher Information overlap analysis** — a mechanistic check that the Fisher matrix correctly captures task-relevant parameters

---

## 1. Experimental Setup

We validate EWC against a plain SGD baseline (no regularisation) on the **Permuted MNIST** benchmark — 10 sequentially presented tasks, each a different random pixel permutation of MNIST.

| Hyperparameter | Value |
|---|---|
| Tasks | 10 (Permuted MNIST) |
| Epochs per task | 3 |
| Learning rate | 0.005 |
| Momentum | 0.9 |
| EWC λ (lambda) | 1000 |
| Batch size | 64 |
| Network | 6-layer MLP (400 hidden units each) |

Both `model_ewc` and `model_sgd` are initialised with the same architecture (`Net`) and trained with the same SGD + momentum optimizer, so any difference in outcomes is attributable solely to the EWC penalty.

---

## 2. The EWC Penalty

After finishing each task, two quantities are stored and used in all future training steps:

**Fisher Information Matrix** — computed empirically over the completed task's training data:

```python
def compute_fisher(model, data_loader):
    fisher_matrix = {}
    for name, param in model.named_parameters():
        fisher_matrix[name] = torch.zeros_like(param.data)

    model.eval()
    for data, target in data_loader:
        ...
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_matrix[name] += param.grad.data ** 2 / len(data_loader)

    return fisher_matrix
```

The Fisher is accumulated as the **mean squared gradient** over all batches, normalised by the number of batches. Using `log_softmax` + `nll_loss` (equivalent to `cross_entropy`) gives a numerically more precise empirical Fisher estimate.

**Optimal weights snapshot** — a `clone()` of every parameter immediately after the task finishes:

```python
opt_weights[task_id] = {}
for name, param in model_ewc.named_parameters():
    opt_weights[task_id][name] = param.data.clone()
```

The EWC penalty sums the quadratic deviation from all previous optimal weights, weighted by their Fisher importance:

```python
def ewc_penalty(model, fisher_matrices, opt_weights):
    penalty = 0
    for task_id in fisher_matrices:
        for name, param in model.named_parameters():
            fisher = fisher_matrices[task_id][name]
            opt_w   = opt_weights[task_id][name]
            penalty += (fisher * (param - opt_w) ** 2).sum()
    return penalty
```

The total EWC loss during task `t` is therefore:

```
L_total = L_cross_entropy + (λ / 2) * Σ_i F_i (θ_i - θ*_i)²
```

where the sum runs over all previous tasks and all parameters.

---

## 3. Per-Epoch Accuracy Tracking

### What we check

After every epoch, `test_model` is called on **every task's test set** for both models:

```python
def test_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)
```

Results are appended to `history_ewc[task_id]` and `history_sgd[task_id]` for every task id at every epoch, building a complete accuracy matrix of shape `(num_tasks, total_epochs)`.

### Console output

Each epoch prints the accuracy on all **seen** tasks:

```
Epoch 01 | EWC: [97%, 85%, 72%] | SGD: [97%, 60%, 31%]
```

This makes forgetting visible in real time: a healthy EWC run holds previous task accuracy near its post-training peak, while plain SGD shows a steady decline.

---

## 4. Visualisation — Task Accuracy Curves (Validation Figure A)

The first plot tracks the first three tasks (A, B, C) across all 30 training epochs:

```python
plt.plot(x_axis, history_ewc[0], label='EWC Task A', color='red',   linestyle='-')
plt.plot(x_axis, history_sgd[0], label='SGD Task A', color='red',   linestyle='--')
plt.plot(x_axis, history_ewc[1], label='EWC Task B', color='green', linestyle='-')
...
```

Vertical dotted lines mark each task boundary, making it easy to see whether accuracy on earlier tasks drops after the model moves on.

**What a correct EWC implementation looks like:**

- Task A (EWC) — accuracy stays high even as tasks B–J are trained
- Task A (SGD) — accuracy falls sharply once training moves to task B
- Tasks B and C (EWC) — accumulate accuracy and broadly hold it
- Tasks B and C (SGD) — degrade progressively

---

## 5. Visualisation — Average Accuracy Curve (Validation Figure B)

The second plot reproduces **Figure 2B** from the original EWC paper (Kirkpatrick et al. 2017). After training on task `i`, we compute the mean accuracy across all tasks seen so far:

```python
for i in range(1, num_tasks):
    end_epoch = (i + 1) * epochs_per_task - 1
    seen_tasks = task_ids[:i+1]
    ewc_accs = [history_ewc[t][end_epoch] for t in seen_tasks]
    sgd_accs = [history_sgd[t][end_epoch] for t in seen_tasks]
    ewc_avg_acc.append(np.mean(ewc_accs))
    sgd_avg_acc.append(np.mean(sgd_accs))
```

A horizontal dashed line marks single-task performance (the best accuracy ever achieved on Task A alone), providing an upper-bound reference.

**What to look for:**

- EWC's average accuracy should remain close to the single-task baseline across all 10 tasks
- SGD's average accuracy should trend downward as the number of tasks grows
- A large gap between the two curves confirms that EWC successfully mitigates catastrophic forgetting

---

## 6. Fisher Overlap Analysis (Mechanistic Validation)

Beyond accuracy, we validate the **internal behaviour** of the Fisher Information Matrix by measuring how much overlap exists between the Fisher computed on different permutations of the same dataset.

### Partial permutation tasks

Three variants of the base task are created:

| Variant | Permuted region | Expected Fisher overlap with base |
|---|---|---|
| `perm_base` | None (original MNIST) | 1.0 (identical) |
| `perm_low` | 8×8 centre patch | High (small perturbation) |
| `perm_high` | 26×26 region | Low (major perturbation) |

All Fisher matrices are computed on the **same pre-trained base model**, isolating the effect of data distribution from weight changes.

### Overlap metric

Fisher overlap is computed using the **Hellinger distance** (following Appendix 4.3 of the original paper):

```python
def calculate_overlap(f1, f2, layer_name):
    v1 = f1[layer_name + '.weight'].flatten()
    v2 = f2[layer_name + '.weight'].flatten()

    # Normalise to unit trace
    v1_hat = v1 / (v1.sum() + 1e-10)
    v2_hat = v2 / (v2.sum() + 1e-10)

    # Squared Hellinger distance
    d2 = 0.5 * torch.sum((torch.sqrt(v1_hat) - torch.sqrt(v2_hat))**2)

    return max(0.0, 1.0 - d2.item())
```

The `1e-10` guard prevents division by zero on layers with near-zero Fisher values.

### What the overlap plot tells us

Overlap is computed layer-by-layer across all 6 layers of the network:

- **High overlap (near 1.0)** — the two tasks rely on the same parameters; EWC will protect them aggressively
- **Low overlap (near 0.0)** — the tasks use different parameters; there is little conflict and less need for the penalty

Expected patterns (consistent with the paper):

- `perm_low` should show higher overlap than `perm_high` at all layers
- Overlap may decrease in deeper layers, reflecting more task-specific representations
- The curves should be monotone or near-monotone, validating that the Fisher is a sensible measure of task-parameter relevance

---

## 7. Checklist for a Valid Run

Use this checklist to confirm the implementation is working correctly:

- [ ] Task A accuracy under EWC remains above 85% throughout all 10 tasks
- [ ] Task A accuracy under SGD falls below 30% by task 4 or later
- [ ] EWC average accuracy (Figure B) stays within ~10 percentage points of the single-task baseline
- [ ] SGD average accuracy (Figure B) trends downward across all 10 tasks
- [ ] `perm_low` Fisher overlap is consistently higher than `perm_high` across all layers
- [ ] Fisher values are non-negative for all parameters (they are squared gradients)
- [ ] `opt_weights` are saved via `.clone()`, not as references (otherwise they would update in place)

---

## 8. Known Limitations

- With only 3 epochs per task, absolute accuracy numbers will be lower than the original paper (which uses up to 20 epochs). The relative ordering (EWC > SGD) should still hold clearly.
- The empirical Fisher approximation (diagonal, computed over training data) is a simplification of the full Fisher. It is sufficient for the Permuted MNIST benchmark.
- The EWC penalty grows with the number of tasks (O(T) per gradient step). For very large T, this becomes computationally expensive and an online EWC variant should be considered.

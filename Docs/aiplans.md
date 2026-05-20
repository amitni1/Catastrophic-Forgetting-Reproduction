# Overcoming Catastrophic Forgetting — EWC Implementation

A reproduction of the paper:
> **Kirkpatrick et al. (2017)** — *Overcoming catastrophic forgetting in neural networks*
> [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)

---

## Background

**Catastrophic Forgetting** is a phenomenon where a neural network, after learning a new task, completely forgets previously learned tasks. This happens because weight updates for the new task overwrite the information encoded for older ones.

The **EWC (Elastic Weight Consolidation)** algorithm solves this by:
1. Identifying the weights most important to previous tasks — using the **Fisher Information Matrix**.
2. Adding a **penalty term** to the loss function that prevents large changes to those weights.

The EWC loss function:

$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta^*_{A,i})^2$$

---

## Q&A — Development Process

### Q1: How should the code be organized?

**Q:** Is it better to put everything in one notebook or split into separate files?

**A:** For an academic paper reproduction, a single notebook is the right choice — it keeps code, plots, and results visible in one place. For a production setting, the recommended split would be:
- `model.py` — network architecture
- `ewc.py` — EWC logic (Fisher computation + penalty)
- `train.py` — training loop
- `utils.py` — helper functions

---

### Q2: How is the Fisher Information Matrix computed?

**Q:** What is the difference between empirical Fisher and the true Fisher?

**A:** The true Fisher is computed using the model's own output distribution, requiring a sum over all possible label vectors — which is intractable. In practice we use the **empirical Fisher**, computing gradients with respect to the ground-truth labels:

```python
def compute_fisher(model, data_loader):
    fisher_matrix = {}
    for name, param in model.named_parameters():
        fisher_matrix[name] = torch.zeros_like(param.data)

    model.eval()
    for data, target in data_loader:
        model.zero_grad()
        output = model(data)
        loss = F.nll_loss(F.log_softmax(output, dim=1), target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_matrix[name] += param.grad.data ** 2 / len(data_loader)

    return fisher_matrix
```

---

### Q3: How do you choose λ (lambda)?

**Q:** How do you know what value to assign to λ?

**A:** λ controls the stiffness of the "spring" — how strongly EWC pulls weights back toward their old values.
- λ too small → EWC doesn't protect enough → forgetting occurs
- λ too large → the model can't learn new tasks

In this project: `λ = 1000`, found via trial and error. The paper's original values correspond to a different network scale and optimizer setup.

---

### Q4: How do you extend EWC beyond 2 tasks?

**Q:** The paper's formula describes 2 tasks. How do you scale to 10?

**A:** We accumulate a separate penalty term for each previously seen task:

```python
def ewc_penalty(model, fisher_matrices, opt_weights):
    penalty = 0
    for task_id in fisher_matrices:
        for name, param in model.named_parameters():
            fisher = fisher_matrices[task_id][name]
            opt_w = opt_weights[task_id][name]
            penalty += (fisher * (param - opt_w) ** 2).sum()
    return penalty
```

This works because the sum of quadratic penalties is itself a quadratic penalty — as noted in the paper.

---

### Q5: How do you reproduce Figure C (Fisher overlap)?

**Q:** Figure C in the paper shows Fisher matrix overlap as a function of layer depth. How is this implemented?

**A:** We compute the **Fréchet distance** between two normalized Fisher matrices, and define overlap as `1 - d²`. To create tasks with controlled similarity, we use partial permutations — only shuffling a square region in the center of the image:

```python
def create_partial_permutation(size):
    perm = torch.arange(28 * 28)
    grid = perm.view(28, 28)
    start = (28 - size) // 2
    end = start + size
    region = grid[start:end, start:end].flatten()
    shuffled_region = region[torch.randperm(len(region))]
    grid[start:end, start:end] = shuffled_region.view(size, size)
    return grid.flatten()
```

Two permutation sizes are used: `8×8` (high task similarity) and `26×26` (low task similarity), matching the paper's Figure 2C.

---

### Q6: Why does SGD forget but EWC doesn't?

**Q:** What exactly happens when plain SGD trains on a new task?

**A:** Gradients from task B overwrite the weights that were important for task A. With EWC, the penalty term prevents this — weights that were critical for task A (high Fisher value) incur a large penalty if they move far from their previous values. The network is therefore forced to find a solution that works well for both tasks simultaneously.

---

### Q7: How does the network architecture differ from the paper?

**Q:** Is the network identical to the one in the paper?

**A:** Not exactly. The main differences:

| Parameter | Paper | This Project |
|-----------|-------|--------------|
| Hidden layers | 2 | 6 |
| Layer width | 400 | 400 |
| Optimizer | SGD | SGD + momentum=0.9 |
| Epochs per task | 20 | 3 |
| Number of tasks | up to 10 | 10 |

The deeper architecture was chosen to better illustrate the Fisher overlap by layer depth in Figure C.

---

## References

- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796). *PNAS, 114(13), 3521–3526.*

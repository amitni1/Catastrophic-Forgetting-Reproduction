# Overcoming Catastrophic Forgetting — EWC Implementation

## About This Document

This document captures the development process for our EWC implementation, including questions we asked AI assistants — **Google Gemini** and **Anthropic Claude** — throughout the project. The answers and code snippets below were generated with their help and adapted to fit our specific setup.

---

## Background

**Catastrophic Forgetting** is a phenomenon where a neural network, after learning a new task, completely forgets previously learned tasks. This happens because weight updates for the new task overwrite the information encoded for older ones.

The **EWC (Elastic Weight Consolidation)** algorithm addresses this by:
1. Identifying the weights most important to previous tasks using the **Fisher Information Matrix**.
2. Adding a **penalty term** to the loss function that prevents large changes to those weights.

The EWC loss function:

$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta^*_{A,i})^2$$

---

## Development Q&A

### Q1: How should we organize the code?

**Us:** We're reproducing a paper and aren't sure whether to keep everything in one notebook or split it into separate files. What would you recommend?

**AI:** For an academic paper reproduction, a single notebook is usually the right call — it keeps your code, plots, and results all visible in one place, which makes it much easier to present and explain. If you ever move this to a production setting, the natural split would be:
- `model.py` — network architecture
- `ewc.py` — EWC logic (Fisher computation + penalty)
- `train.py` — training loop
- `utils.py` — helper functions

But for your purposes, one notebook keeps things clean and readable.

---

### Q2: How do we compute the Fisher Information Matrix?

**Us:** We've seen references to both "empirical Fisher" and "true Fisher." What's the difference, and which one should we use?

**AI:** Great question — this distinction trips a lot of people up. The **true Fisher** is computed using the model's own output distribution, which requires summing over all possible label vectors. In practice that's completely intractable. What we actually use is the **empirical Fisher**, where we compute gradients with respect to the ground-truth labels instead. It's a well-accepted approximation and works well in practice:

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

---

### Q3: How do we pick a value for λ (lambda)?

**Us:** The paper doesn't give a single clear answer here. How do we know what λ to use?

**AI:** Honestly, λ is one of those hyperparameters you just have to feel out for your specific setup. Think of it as controlling the stiffness of a spring — how hard EWC pulls weights back toward their old values:

- **λ too small** → the regularization isn't strong enough → the model still forgets
- **λ too large** → the model is too rigid → it can't learn new tasks properly

We ended up going with `λ = 1000`, which we found through trial and error. The values in the original paper don't directly transfer because they used a different network scale and optimizer configuration.

---

### Q4: How do we extend EWC to more than 2 tasks?

**Us:** The paper's formula seems to describe a 2-task scenario. We're running 10 tasks — how do we scale it up?

**AI:** The good news is that EWC extends naturally to multiple tasks. You just accumulate a separate penalty term for each previously seen task and sum them all together. Since each term is quadratic, the sum is still a valid quadratic penalty — the paper actually mentions this. Here's what that looks like in code:

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

As you complete each task, you save its Fisher matrix and optimal weights, then include them in every subsequent penalty calculation.

---

### Q5: How do we reproduce Figure C (Fisher overlap by layer depth)?

**Us:** Figure C in the paper shows how Fisher matrix overlap changes with layer depth. We're not sure how to implement this — especially the part about controlling task similarity.

**AI:** To reproduce this, you need two things: a way to measure Fisher overlap between tasks, and a way to create tasks with controlled similarity levels.

For the **overlap metric**, compute the Fréchet distance between two normalized Fisher matrices and define overlap as `1 - d²`.

For **controlling task similarity**, use partial permutations — instead of shuffling the entire image (like a standard permuted MNIST task), only shuffle a square region in the center. A smaller shuffled region means the tasks are more similar to each other:

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

We use two permutation sizes to match the paper's Figure 2C: `8×8` (high task similarity) and `26×26` (low task similarity).

---

### Q6: Why does plain SGD forget but EWC doesn't?

**Us:** We understand EWC conceptually, but can you walk us through exactly what's happening mechanically when SGD forgets?

**AI:** Sure — here's what's going on under the hood. When you train with plain SGD on task B, gradients are computed only with respect to task B's loss. Those gradients update the weights freely, and in doing so they overwrite the information that was encoded for task A. There's nothing protecting those weights — they just get pushed wherever task B needs them.

With EWC, the penalty term changes that. Weights that were critical for task A (i.e., weights with high Fisher values) incur a large penalty if they move far from their task A values. So the optimizer is forced to find weight values that perform well on task B *without* straying too far from what worked for task A. It's essentially solving a constrained optimization problem, even though it's implemented as a soft penalty rather than a hard constraint.

---



## References

- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796). *PNAS, 114(13), 3521–3526.*

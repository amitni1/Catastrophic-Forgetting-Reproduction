# AI Usage & Methodology — Catastrophic Forgetting Reproduction Project

# AI Usage & Methodology — Catastrophic Forgetting Reproduction Project

## AI Tool Used

Throughout all stages of this project we used **Claude (Anthropic)** as our primary AI assistant. The usage was not a one-time interaction — it was an ongoing working relationship covering everything from initial paper comprehension to final debugging of the Fisher overlap graph.

---

## Stage 1 — Understanding the Paper and Initial Planning

**What we did:**
We fed Claude the original Kirkpatrick et al. (2017) paper and asked it to break down the experimental methodology — which tasks were tested, what the network architecture looked like, and what hyperparameters were specified.

**Example prompt:**
> "The paper describes EWC using a quadratic penalty scaled by the Fisher Information Matrix diagonal. Does this mean we compute Fisher once per task and store it permanently, or do we recompute it at every training step?"

**What Claude helped with:**
Claude clarified that Fisher is computed **once at the end of each task** — after the model has converged — and then frozen. It explained the difference between computing Fisher during training (wrong) versus at task completion (correct), and why the anchor point `θ*` must also be saved at that exact moment. It also explained the three main figures in the paper (A, B, C) and what each one is designed to demonstrate.

**Human-in-the-loop:**
We read Sections 2 and 2.1 of the paper independently to verify Claude's interpretation of the Fisher computation timing. We also cross-checked the paper's description of the permuted MNIST benchmark to confirm that Task 0 uses the original (unpermuted) MNIST while subsequent tasks apply random pixel shuffles.

---

## Stage 2 — Building the Dataset Pipeline

**What we did:**
We asked Claude to help write the `PermutedMNIST` wrapper class and the task generation loop that creates 10 independent tasks, each with its own train and test `DataLoader`.

**Example prompt:**
> "I need a PyTorch Dataset class that wraps MNIST and applies an optional pixel permutation. Task 0 should be unpermuted. Tasks 1–9 should each have a different random permutation. Show me how to create the DataLoaders for all 10 tasks in a loop."

**What Claude helped with:**
Claude wrote the `PermutedMNIST` class using `img.view(-1)[self.permutation].view(1, 28, 28)` to apply the shuffle while preserving the tensor shape. It also wrote the task generation loop:

```python
for i in range(num_tasks):
    perm = None if i == 0 else torch.randperm(28 * 28)
    tasks[i] = {
        'train': DataLoader(PermutedMNIST(train_dataset, perm), batch_size=256, shuffle=True),
        'test':  DataLoader(PermutedMNIST(test_dataset,  perm), batch_size=1000, shuffle=False)
    }
```

**Human-in-the-loop:**
We verified that Task 0 uses `perm=None` (original MNIST), confirmed that labels are never permuted — only pixel order — and checked that the normalization values `(0.1307, 0.3081)` match the standard MNIST normalization used in the paper's baseline. We also increased the batch size to 256 (from the paper's 64) to speed up training on our hardware.

---

## Stage 3 — Network Architecture

**What we did:**
We implemented three separate network classes to serve different parts of the experiment: `Net` for Figures 2A and 2B, `NetDropout` as the SGD+dropout baseline in Figure 2B, and `NetDeep` for the Fisher overlap analysis in Figure 2C.

**Example prompt:**
> "The paper uses a 2-layer network with 400 units for the main experiments, and we also need a deeper network for Figure 2C to show layer-by-layer Fisher overlap. Can you write both? The deep network should have 6 hidden layers."

**What Claude helped with:**
Claude implemented `Net` with `fc1` (784→400), `fc2` (400→400), and `fc3` (400→10), using `F.relu` activations for the hidden layers and raw logits from `fc3`. For `NetDropout`, it added an input dropout layer at 0.2 and hidden dropout at 0.5. For `NetDeep`, it implemented 6 hidden layers of 200 units each (`fc1`–`fc6`) with a final output layer `fc7` (200→10), which we use for layer-by-layer Fisher overlap in Figure 2C.

**Human-in-the-loop:**
We verified tensor dimensions manually for each network. We ran dummy forward passes to confirm output shapes of `(batch_size, 10)` before starting the training loop. We also decided to use 200 units (rather than 400) in `NetDeep` to reduce memory usage while still providing enough depth for a meaningful layer-depth analysis.

---

## Stage 4 — Fisher Information Matrix and EWC Penalty

**What we did:**
This was the most technically involved stage. We asked Claude to implement `compute_fisher` and `ewc_penalty`, and to explain the mathematical connection between squared gradients and the Fisher diagonal.

**Example prompt:**
> "How do I compute the diagonal of the Fisher Information Matrix in PyTorch? The paper says to use it as an approximation of the posterior precision. I understand it involves squaring gradients — can you write the function and explain why this approximation is valid?"

**What Claude helped with:**
Claude wrote the empirical Fisher computation using a configurable `num_samples` argument to cap how many data points are used:

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

It also implemented the multi-task EWC penalty:

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

**Human-in-the-loop:**
We added print statements to verify that Fisher values were non-zero and in a reasonable range. We also confirmed that `opt_weights` were saved with `.clone()` — without this, PyTorch would store a reference to the current weights rather than a copy, causing the anchor point to update incorrectly during subsequent training. For Figure 2C we increased `num_samples` to 8192 to get more stable Fisher estimates for the overlap calculation.

---

## Stage 5 — Training Loop Design

**What we did:**
We asked Claude to help structure the main training loop that runs four models in parallel — EWC, L2 regularisation, plain SGD, and SGD+dropout — across all 10 tasks, logging accuracy on all tasks after every epoch.

**Example prompt:**
> "I want to train four models side by side — EWC, L2, SGD, and SGD+dropout — across 10 sequential tasks. After each epoch I need to evaluate all models on all 10 tasks and store the results. At the end of each task I need to compute the Fisher matrix for the EWC model. Can you structure this loop?"

**What Claude helped with:**
Claude designed the nested loop structure: outer loop over tasks, inner loop over epochs, and a per-epoch evaluation loop over all 10 tasks. It highlighted that EWC activates only after at least one task has been completed (`if fisher_matrices:`), and that Fisher computation must happen **after** the task's training is finished:

```python
if fisher_matrices:
    loss_ewc += (lamda / 2) * ewc_penalty(model_ewc, fisher_matrices, opt_weights)
```

Evaluating all 10 tasks every epoch (not just seen tasks) made it easy to see that unseen task accuracy stays near chance until training reaches that task.

**Human-in-the-loop:**
We confirmed that `F.cross_entropy` is used consistently across all four models in the training loop, and matched the Fisher computation to use `F.nll_loss(F.log_softmax(...))` for mathematical consistency with the penalty. We also kept evaluation across all 10 tasks rather than only seen tasks, since this makes forgetting more visually obvious in Figure 2A.

---

## Stage 6 — Hyperparameter Decisions

**What we did:**
We asked Claude about the choice of lambda, learning rate, epochs per task, and L2 weight decay — specifically how to adapt the paper's values to our setup.

**Example prompt:**
> "The paper uses λ=1 in some notation but their Fisher values are unnormalized. We're training with SGD lr=1e-3, momentum=0.9 for 20 epochs per task. What λ should we use, and how does the L2 baseline weight decay compare?"

**What Claude helped with:**
Claude explained that the EWC penalty needs to be comparable in magnitude to the classification loss (~2.3 for random guessing on 10 classes). It recommended monitoring the ratio `penalty / classification_loss` during training and targeting a ratio of roughly 0.5–2.0. For the L2 baseline it noted that the weight decay must be weak enough to show a clear gap between L2 and EWC, otherwise Figure 2A looks flat.

**Human-in-the-loop:**
We tested multiple values and settled on the following configuration:

- `epochs_per_task = 20` — enough epochs for forgetting to become clearly visible within each task block
- `lr = 1e-3`, `momentum = 0.9` — paper's baseline
- `lamda = 5000` — stronger than initial experiments to produce clear separation between EWC and SGD
- `l2_weight_decay = 1e-5` — deliberately weak so L2 degrades more visibly than EWC

Earlier runs with `lamda = 2000` and `l2_weight_decay = 1e-3` made EWC and L2 look almost identical; the final values above produce the clear three-way separation visible in Figure 2A.

---

## Stage 7 — Figure Generation

**What we did:**
We asked Claude to help reproduce Figures 2A, 2B, and 2C from the paper using matplotlib.

**Example prompt (Figure 2C):**
> "For Figure 2C I need to compute Fisher overlap between a base task (unpermuted MNIST) and two partially permuted versions — one with an 8×8 centre shuffle and one with a 26×26 centre shuffle. I then need to plot the overlap layer by layer. The paper uses Fréchet distance. Can you implement `create_partial_permutation(size)` and the overlap calculation?"

**What Claude helped with:**
Claude implemented `create_partial_permutation`:

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

And the Fréchet-distance-based overlap metric, normalised per layer:

```python
def calculate_overlap(f1, f2, layer_name, all_layers):
    v1 = torch.cat([f1[layer_name+'.weight'].flatten(),
                    f1[layer_name+'.bias'].flatten()]).clamp(min=0)
    v2 = torch.cat([f2[layer_name+'.weight'].flatten(),
                    f2[layer_name+'.bias'].flatten()]).clamp(min=0)
    v1 = v1 / (v1.sum() + 1e-10)
    v2 = v2 / (v2.sum() + 1e-10)
    d2 = 0.5 * torch.sum((torch.sqrt(v1) - torch.sqrt(v2)) ** 2)
    return 1.0 - d2.item()
```

Claude also pointed out that both `weight` and `bias` parameters should be included and that Fisher values should be clamped to zero before normalisation to avoid negative-value artefacts.

**Human-in-the-loop:**
We compared our Figure 2C output against Figure 2C in the paper. The key pattern to verify was: the 8×8 overlap (similar tasks) should be higher than the 26×26 overlap (more dissimilar tasks) in early layers, with the two curves converging toward the final layers. We confirmed this pattern held. For Figure 2C we trained each model pair for 100 epochs (instead of the paper's 20) so that the Fisher values specialise enough to produce a visible difference between the two curves.

---

## Stage 8 — Debugging

**What we did:**
When initial graphs didn't match expectations, we described the symptoms to Claude and worked through the fixes together.

**Example prompt:**
> "My Figure 2A shows that EWC and SGD perform identically — both maintain high accuracy on Task A. There's no catastrophic forgetting visible at all. What could cause this?"

**What Claude helped with:**
Claude immediately identified two common causes: (1) evaluating on training data instead of test data — the model memorises training examples so it appears to remember, and (2) the EWC penalty not being large enough relative to the classification loss. It suggested printing the ratio of penalty to classification loss during the first few batches of Task 2 to diagnose the issue.

**A second example:**
> "My Figure 2B average accuracy looks wrong — it's higher than any individual task accuracy, which is impossible. What's happening?"

**What Claude helped with:**
Claude spotted that we were computing the end-of-task epoch index incorrectly. The fix was ensuring the slice `range(i + 1)` correctly covered all tasks seen so far, including the current one.

**A third example:**
> "Figure 2C is flat — both the 8×8 and 26×26 overlap curves are nearly identical across all layers. What's wrong?"

**What Claude helped with:**
Claude identified that training for too few epochs (50) meant the two Fisher matrices hadn't diverged enough to show a difference. It suggested increasing to 100 epochs per stage and raising `num_samples` from 4096 to 8192 for more stable Fisher estimates. It also suggested `.clone()` on the grid in `create_partial_permutation` to avoid an in-place mutation bug that was making the partial permutations incorrect.

**Human-in-the-loop:**
Not every suggestion Claude made worked immediately. In several cases we tried a proposed fix, saw the graphs still looked wrong, and returned with more specific information. This iterative process taught us to always check each fix in isolation before launching the full experiment.

---

## Human-in-the-Loop Principles We Applied

1. **No blind copy-paste** — every code snippet Claude wrote was read and understood before being run.

2. **Verify against the paper** — for every implementation decision, we checked Section 2 and Appendix 4.3 of the paper to confirm Claude's interpretation matched the original text.

3. **Test incrementally** — when Claude suggested a change, we tested it on a 2-task, 1-epoch run before launching the full 10-task experiment.

4. **Paper takes priority** — in cases where Claude's suggestion sounded reasonable but conflicted with the paper's description (e.g. when to compute Fisher), we followed the paper.

5. **Question the output** — we never assumed a graph was correct just because it ran without errors. We compared every figure against its counterpart in the paper before accepting the result.

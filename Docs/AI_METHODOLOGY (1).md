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
        'train': DataLoader(PermutedMNIST(train_dataset, perm), batch_size=64, shuffle=True),
        'test':  DataLoader(PermutedMNIST(test_dataset,  perm), batch_size=1000, shuffle=False)
    }
```

**Human-in-the-loop:**
We verified that Task 0 uses `perm=None` (original MNIST), confirmed that labels are never permuted — only pixel order — and checked that the normalization values `(0.1307, 0.3081)` match the standard MNIST normalization used in the paper's baseline.

---

## Stage 3 — Network Architecture

**What we did:**
We asked Claude to help implement the `Net` class. We decided to use 6 fully connected layers instead of the paper's 2, in order to have enough depth to show meaningful layer-by-layer Fisher overlap in Figure C.

**Example prompt:**
> "The paper uses a 2-layer network with 400 units. We want to use 6 layers to better show Fisher overlap by depth. Can you write the PyTorch module? All hidden layers should be 400 units with ReLU, and the output should be 10 logits (no softmax — we'll apply it in the loss)."

**What Claude helped with:**
Claude implemented the `Net` class with `fc1` through `fc6`, using `F.relu` activations for layers 1–5 and raw logits from `fc6`. It also explained why we should return raw logits (not apply softmax in `forward`) so that we can use either `F.cross_entropy` or `F.nll_loss(F.log_softmax(...))` depending on the context.

**Human-in-the-loop:**
We verified tensor dimensions manually: `784 → 400 → 400 → 400 → 400 → 400 → 10`. We also checked that the output shape was `(batch_size, 10)` by running a dummy forward pass before starting the training loop.

---

## Stage 4 — Fisher Information Matrix and EWC Penalty

**What we did:**
This was the most technically involved stage. We asked Claude to implement `compute_fisher` and `ewc_penalty`, and to explain the mathematical connection between squared gradients and the Fisher diagonal.

**Example prompt:**
> "How do I compute the diagonal of the Fisher Information Matrix in PyTorch? The paper says to use it as an approximation of the posterior precision. I understand it involves squaring gradients — can you write the function and explain why this approximation is valid?"

**What Claude helped with:**
Claude wrote the empirical Fisher computation:

```python
loss = F.nll_loss(F.log_softmax(output, dim=1), target)
loss.backward()
fisher_matrix[name] += param.grad.data ** 2 / len(data_loader)
```

It explained that at a local minimum, the expected squared gradient equals the curvature of the loss (second derivative), making it a valid approximation of the Fisher diagonal without needing to compute second-order derivatives. It also implemented the multi-task EWC penalty:

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

**Human-in-the-loop:**
We added print statements to verify that Fisher values were non-zero and in a reasonable range. We also checked that `opt_weights` were saved with `.clone()` — without this, PyTorch would store a reference to the current weights rather than a copy, causing the anchor point to update incorrectly during subsequent training.

---

## Stage 5 — Training Loop Design

**What we did:**
We asked Claude to help structure the main training loop that runs both the SGD baseline and the EWC model in parallel across all 10 tasks, logging accuracy on all previously seen tasks after every epoch.

**Example prompt:**
> "I want to train two models side by side — one plain SGD, one with EWC — across 10 sequential tasks. After each epoch I need to evaluate both models on all tasks seen so far and store the results. At the end of each task I need to compute the Fisher matrix for the EWC model. Can you structure this loop?"

**What Claude helped with:**
Claude designed the nested loop structure: outer loop over tasks, inner loop over epochs, and a per-epoch evaluation loop over all seen tasks. It also highlighted a subtle distinction: the EWC penalty should only activate after at least one task has been completed (i.e., `if len(fisher_matrices) > 0`), and the Fisher computation should happen **after** the task's training is finished, not during.

**Human-in-the-loop:**
We caught a bug where we were using `F.nll_loss(F.log_softmax(...))` for Fisher computation but `F.cross_entropy` in the training loop. Although these are mathematically equivalent, we kept them consistent for clarity. We also decided to print per-task accuracy after each epoch (not just at task boundaries) to monitor forgetting in real time.

---

## Stage 6 — Hyperparameter Decisions

**What we did:**
We asked Claude about the choice of lambda, learning rate, and epochs per task — specifically how to adapt the paper's values to our 6-layer architecture.

**Example prompt:**
> "The paper uses λ=1 in some notation but their Fisher values are unnormalized. What λ should I use for a 6-layer network trained with SGD lr=0.005 and momentum=0.9 for 3 epochs per task? My Fisher values seem to be in the range 1e-3 to 1e-4."

**What Claude helped with:**
Claude explained the key relationship: the EWC penalty needs to be **comparable in magnitude to the classification loss** (~2.3 for random guessing on 10 classes). Given Fisher values in the range `1e-3 to 1e-4`, a λ of 1000 produces a penalty in the range `0.5 to 5`, which is in the right ballpark. It suggested monitoring the ratio `penalty / classification_loss` during training and targeting a ratio of 0.5–2.0.

**Human-in-the-loop:**
We tested λ ∈ {100, 1000, 5000} and settled on **λ = 1000** after observing that:
- λ = 100 didn't prevent forgetting on Task A after 5+ tasks
- λ = 1000 maintained Task A accuracy above 80% throughout
- λ = 5000 caused Task 9 accuracy to stagnate below 70%

We also increased the learning rate from the paper's baseline to `lr = 0.005` (×10) to compensate for training only 3 epochs instead of the paper's 20.

---

## Stage 7 — Figure Generation

**What we did:**
We asked Claude to help reproduce Figures A, B, and C from the paper using matplotlib.

**Example prompt (Figure C):**
> "For Figure C I need to compute Fisher overlap between a base task (unpermuted MNIST) and two partially permuted versions — one with an 8×8 center shuffle and one with a 26×26 center shuffle. I then need to plot the overlap layer by layer. The paper uses Fréchet distance. Can you implement `create_partial_permutation(size)` and the overlap calculation?"

**What Claude helped with:**
Claude implemented `create_partial_permutation`:

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

And the Fréchet-distance-based overlap metric:

```python
def calculate_overlap(f1, f2, layer_name):
    v1 = f1[layer_name + '.weight'].flatten()
    v2 = f2[layer_name + '.weight'].flatten()
    v1_hat = v1 / v1.sum()
    v2_hat = v2 / v2.sum()
    d2 = 0.5 * torch.sum((torch.sqrt(v1_hat) - torch.sqrt(v2_hat))**2)
    return 1.0 - d2.item()
```

Claude also pointed out that the overlap should be computed on `weight` parameters only (not biases), matching Appendix 4.3 of the paper.

**Human-in-the-loop:**
We compared our Figure C output visually against Figure 2C in the paper. The key pattern to verify was: the 8×8 overlap (similar tasks) should be higher than the 26×26 overlap (dissimilar tasks) in early layers, with the two curves converging in the final layers. We confirmed this pattern held in our output.

---

## Stage 8 — Debugging

**What we did:**
When initial graphs didn't match expectations, we described the symptoms to Claude and worked through the fixes together.

**Example prompt:**
> "My Figure A shows that EWC and SGD perform identically — both maintain high accuracy on Task A. There's no catastrophic forgetting visible at all. What could cause this?"

**What Claude helped with:**
Claude immediately identified two common causes: (1) evaluating on training data instead of test data — the model memorizes training examples so it looks like it "remembers," and (2) the EWC penalty not activating because Fisher values are too small relative to λ. It suggested printing the ratio of penalty to classification loss during the first few batches of Task 2 to diagnose the issue.

**A second example:**
> "My Figure B average accuracy looks wrong — it's higher than any individual task accuracy, which is impossible. What's happening?"

**What Claude helped with:**
Claude spotted that we were averaging over all 10 tasks even before all had been seen, so tasks with 0% accuracy (not yet trained) were being excluded incorrectly due to an off-by-one in the slice index. The fix was changing `task_ids[:i]` to `task_ids[:i+1]`.

**Human-in-the-loop:**
Not every suggestion Claude made worked immediately. In several cases we tried a proposed fix, saw the graphs still looked wrong, and returned to Claude with more specific information. This iterative process taught us to always check each fix in isolation before moving on.

---

## Human-in-the-Loop Principles We Applied

1. **No blind copy-paste** — every code snippet Claude wrote was read and understood before being run.

2. **Verify against the paper** — for every implementation decision, we checked Section 2 and Appendix 4.3 of the paper to confirm Claude's interpretation matched the original text.

3. **Test incrementally** — when Claude suggested a change, we tested it on a 2-task, 1-epoch run before launching the full 10-task experiment.

4. **Paper takes priority** — in cases where Claude's suggestion sounded reasonable but conflicted with the paper's description (e.g., when to compute Fisher), we followed the paper.

5. **Question the output** — we never assumed a graph was correct just because it ran without errors. We compared every figure against its counterpart in the paper before accepting the result.

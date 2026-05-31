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
Claude clarified that Fisher is computed **once at the end of each task** — after the model has converged — and then frozen. It explained the difference between computing Fisher during training (wrong) versus at task completion (correct), and why the anchor point `θ*` must also be saved at that exact moment. Critically, it highlighted two failure modes from our earlier notebooks:

- **Batch-averaged Fisher** (computing `grad**2 / batch_size` inside a batched forward pass) underestimates the true Fisher, causing the Fig 2B average to collapse to ~0.8.
- **Stiff lambda + gradient clipping + limited capacity** over-constrains the network so it cannot learn new tasks, causing the Fig 2B average to fall to ~0.7.

Claude explained that the fix for both is the **true per-sample Fisher** — one example at a time, with the label sampled from the model's own predictive distribution — and the **paper-exact separate-penalties formulation** of equation 3.

**Human-in-the-loop:**
We read Sections 2 and 2.1 of the paper independently to verify Claude's interpretation. We confirmed that the paper's "separate penalties" scheme (summing one penalty term per completed task) is distinct from the online-EWC variant, and that the two produce different lambda sensitivities.

---

## Stage 2 — Building the Dataset Pipeline

**What we did:**
We asked Claude to help write the `PermutedMNIST` wrapper class and the task generation loop. We also needed a validation split for the SGD+dropout baseline's per-task early stopping — something our earlier notebooks lacked.

**Example prompt:**
> "I need a PyTorch Dataset class that wraps MNIST and applies an optional pixel permutation. Task 0 should be unpermuted. Tasks 1–9 should each have a different random permutation. I also need a 10k validation split held out from training for early stopping. Show me how to create the DataLoaders for all 10 tasks in a loop."

**What Claude helped with:**
Claude wrote the `PermutedMNIST` class and the three-way data split:

```python
val_size   = 10000
train_size = len(full_train) - val_size
train_base, val_base = torch.utils.data.random_split(
    full_train, [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
```

Each task dictionary now holds `'train'`, `'val'`, `'test'`, and `'perm'` keys, giving the dropout baseline access to task-specific validation loaders during training.

**Human-in-the-loop:**
We verified that Task 0 uses `perm=None` (original MNIST), confirmed that labels are never permuted — only pixel order — and checked that the normalization values `(0.1307, 0.3081)` match the standard MNIST baseline. We also confirmed the validation split uses a fixed seed (42) so the split is reproducible across runs.

---

## Stage 3 — Network Architecture

**What we did:**
We implemented three network classes: `Net` for the main EWC and SGD/L2 baselines, `NetDropout` for the SGD+dropout baseline, and `NetDeep` for the Figure 2C Fisher overlap analysis.

**Example prompt:**
> "The paper's Fig 2B allows networks up to 2000 units wide. Can you make the hidden width configurable so we can sweep it without rewriting the class? And the Fig 2C network needs 6 hidden layers of 100 units to match the paper's Appendix 4.3."

**What Claude helped with:**
Claude made `Net` and `NetDropout` take a `width` parameter (defaulting to 400) so we can set it from the config cell:

```python
class Net(nn.Module):
    def __init__(self, width=400):
        super().__init__()
        self.fc1 = nn.Linear(784, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 10)
```

`NetDeep` uses 6 hidden layers of 100 units each (`fc1`–`fc6`) followed by a 10-class output layer `fc7`, matching Appendix 4.3 of the paper. The default `WIDTH = 2000` in the config reflects the paper's Fig 2B column, which allows networks up to 2000 units wide.

**Human-in-the-loop:**
We ran dummy forward passes to confirm output shapes of `(batch_size, 10)` for all three models before starting training. We also verified that `NetDeep` uses 100 units (not 200 or 400) per layer, matching the paper's specification for the Fisher overlap experiment.

---

## Stage 4 — Fisher Information Matrix and EWC Penalty

**What we did:**
This was the most technically involved stage. The core change from our earlier notebooks was switching to the **true per-sample Fisher** — computing one gradient at a time with a label sampled from the model's own predictive distribution — rather than the batch-averaged approximation.

**Example prompt:**
> "Our earlier Fisher computation averaged over a batch. You said this underestimates the true Fisher. Can you rewrite `compute_fisher` to process one example at a time and sample the label from the model's predictive distribution? Also explain why that gives the true diagonal Fisher."

**What Claude helped with:**
Claude explained that the true Fisher is `E[(d log p / d θ)²]` where the expectation is over labels drawn from the model's own distribution — not the empirical labels. Batching and using the ground-truth labels gives an approximation of the expected gradient under the data distribution, which underestimates the curvature when the model is well-calibrated. The corrected implementation:

```python
def compute_fisher(model, data_loader, num_samples=1024):
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    model.eval()
    seen = 0
    for data, _ in data_loader:
        data = data.to(device)
        for x in data:
            if seen >= num_samples: break
            x = x.unsqueeze(0)
            model.zero_grad()
            logits = model(x)
            log_p  = F.log_softmax(logits, dim=1)
            p      = F.softmax(logits, dim=1).clamp_min(1e-8)
            p      = p / p.sum(dim=1, keepdim=True)
            y      = torch.multinomial(p, 1).squeeze(1)   # sample from model's distribution
            F.nll_loss(log_p, y).backward()
            for n, par in model.named_parameters():
                if par.grad is not None:
                    fisher[n] += par.grad.detach() ** 2
            seen += 1
        if seen >= num_samples: break
    for n in fisher:
        fisher[n] /= max(seen, 1)
    return fisher
```

Claude also restructured `ewc_penalty_multi` to accept a list of `{'fisher': ..., 'anchor': ...}` dicts — one per completed task — and optionally normalise by task count:

```python
def ewc_penalty_multi(model, ewc_tasks, normalize=True):
    penalty = 0.0
    for task in ewc_tasks:
        for n, p in model.named_parameters():
            penalty += (task['fisher'][n] * (p - task['anchor'][n]) ** 2).sum()
    if normalize and len(ewc_tasks) > 0:
        penalty = penalty / len(ewc_tasks)
    return penalty
```

**Human-in-the-loop:**
We verified that `opt_weights` (now `anchor`) are saved with `.detach().clone()` so PyTorch stores a copy rather than a reference. We also confirmed that the `_` in the `for data, _ in data_loader` loop correctly discards the empirical label, ensuring we never accidentally condition on ground truth during Fisher estimation.

---

## Stage 5 — Training Loop Design

**What we did:**
We asked Claude to refactor the training into two separate functions — `train_ewc_sequence` for the EWC model and `train_baselines` for SGD, L2, and SGD+dropout — so that each can be re-run independently without repeating the others.

**Example prompt:**
> "I want to re-run EWC at different lambda values without retraining the baselines every time. Can you split the loop into two functions? The EWC function should take `lamda` as an argument and return per-epoch history plus end-of-task average snapshots."

**What Claude helped with:**
Claude designed `train_ewc_sequence` to take `lamda`, `width`, `epochs`, `lr`, `fisher_samples`, and `seed` as arguments, making it trivial to sweep lambda from the config cell. It also emphasised that **gradient clipping should be removed** from the EWC optimizer — the clip in our `FIXED_2` notebook was suppressing the consolidation force and was the primary cause of the Fig 2B collapse to ~0.7:

```python
loss.backward()
opt.step()   # no gradient clipping — clipping suppressed EWC consolidation
```

The `train_baselines` function implements per-task early stopping for the dropout model using a held-out validation set, restoring the best checkpoint after each task:

```python
if val_avg > best_val:
    best_val = val_avg
    best_state = copy.deepcopy(m_drop.state_dict())
    no_improve = 0
else:
    no_improve += 1
    if no_improve >= patience:
        m_drop.load_state_dict(best_state)
        stopped = True
```

**Human-in-the-loop:**
We verified that the L2 anchor is updated at the end of each task (not the beginning), and that the L2 baseline uses `l2_lambda = 1.0` — a deliberately weak penalty that makes L2 degrade more than EWC in Figure 2A.

---

## Stage 6 — Single-Task Reference Line

**What we did:**
Our earlier notebooks computed the single-task reference by reading from the EWC model's history on Task 0 before any subsequent tasks were trained. Claude pointed out this is confounded: the EWC model's task-0 performance reflects the same random seed and hyperparameters as the multi-task run. A cleaner reference trains a completely independent model only on Task 0.

**Example prompt:**
> "Should the dashed 'single task performance' line in Fig 2B come from the EWC model's peak Task 0 accuracy, or from a separate model trained only on Task 0?"

**What Claude helped with:**
Claude explained that reading from the EWC model conflates single-task capacity with multi-task interference. The correct reference is an independent model that never sees tasks 1–9:

```python
torch.manual_seed(99)
model_single = Net(WIDTH).to(device)
opt_single   = optim.SGD(model_single.parameters(), lr=LR, momentum=MOMENTUM)
for ep in range(EPOCHS_PER_TASK):
    model_single.train()
    for data, target in tasks[0]['train']:
        ...
single_task_perf = test_model(model_single, tasks[0]['test'])
```

**Human-in-the-loop:**
We confirmed that `model_single` uses `seed=99` (different from `SEED=0`) to avoid accidentally replicating the EWC model's initialisation. We also verified it is evaluated on the test set, not the train or validation split.

---

## Stage 7 — Hyperparameter Decisions

**What we did:**
We asked Claude how to choose lambda, network width, and epochs per task, given that the penalty grows as tasks accumulate under the separate-penalties formulation.

**Example prompt:**
> "Old lambda values from our earlier notebooks don't transfer to the new per-task summed penalty. How do I pick a starting point, and what range should I sweep?"

**What Claude helped with:**
Claude explained that because the total EWC penalty sums one term per completed task, the effective constraint stiffens as training progresses. Lambda values that worked for the online-EWC variant (one accumulated Fisher) are typically too large for the separate-penalties form. It recommended sweeping across roughly two orders of magnitude:

```
LAMBDA ∈ {50, 150, 500, 1500, 5000}
```

and keeping the combination that produces the flattest, highest red curve in Figure 2B — targeting ~0.93–0.96 rather than exactly 0.90, which usually indicates over-constraining.

The config settled on:

- `WIDTH = 2000` — paper's Fig 2B maximum, giving enough capacity for EWC to protect old tasks without starving new ones
- `EPOCHS_PER_TASK = 20` — sufficient for forgetting to become visible; raise toward 40–100 for smoother curves
- `LR = 1e-3` — paper's baseline; lower to `5e-4` if the loss surface becomes stiff under accumulated penalties
- `LAMBDA = 100` — starting point; sweep as above
- `FISHER_SAMPLES = 2048` — good variance reduction; raise to reduce noise in Fisher estimates

**Human-in-the-loop:**
We confirmed that lambda and Fisher scale are coupled, meaning the note in the config cell is important: old-notebook lambda values should not be reused without rechecking. We also kept `NORMALIZE_PENALTY = True` to divide the summed penalty by the number of tasks, which partially offsets the growing constraint and makes the lambda range more stable across task counts.

---

## Stage 8 — Figure Generation

**What we did:**
We asked Claude to help reproduce Figures 2A, 2B, and 2C from the paper using matplotlib, including axis ranges, marker styles, and label placement that match the paper's layout.

**Example prompt (Figure 2B):**
> "Figure 2B has EWC and SGD+dropout on the same axes, with the single-task reference as a dashed line. Can you add right-aligned labels at the end of each curve instead of a legend box, and auto-scale the y-axis to fit whatever performance range the models produce?"

**What Claude helped with:**
Claude implemented auto-scaled y-bounds and right-side curve labels:

```python
y_lo = min(0.78, min(dropout_snapshots) - 0.02)
y_hi = max(0.99, max(ewc_snapshots) + 0.01, single_task_perf + 0.015)
ax.set_ylim(y_lo, y_hi)
ax.text(x_tasks[-1] + 0.08, ewc_snapshots[-1],     'EWC',         ...)
ax.text(x_tasks[-1] + 0.08, dropout_snapshots[-1], 'SGD+dropout', ...)
```

For Figure 2C, Claude restructured the overlap metric to use **globally normalised** Fisher vectors — the concatenation of all layers' Fisher values is normalised to unit sum before slicing per-layer:

```python
def fisher_to_global_vector(fisher_dict, layer_names):
    parts = []
    for name in layer_names:
        parts.append(fisher_dict[name + '.weight'].flatten().clamp(min=0))
        parts.append(fisher_dict[name + '.bias'].flatten().clamp(min=0))
    return torch.cat(parts)
```

This matches Appendix 4.3 of the paper more precisely than the per-layer normalisation we used in earlier notebooks.

**Human-in-the-loop:**
We verified Figure 2C against the paper: the 8×8 overlap curve should sit above the 26×26 curve in early layers, with the two converging toward the output. We also saved all three figures as PNG files (`fig2A.png`, `fig2B.png`, `fig2C.png`) for the report. Figure 2C is the slowest cell (100 epochs × 2 tasks × 2 pairs); we kept `FIG2C_EPOCHS = 100` to match the paper but noted it can be lowered for quick shape-checks.

---

## Stage 9 — Debugging

**What we did:**
When the red EWC line in Figure 2B still drooped, we described the symptom to Claude and worked through fixes iteratively.

**Example prompt:**
> "The EWC average in Fig 2B is still declining — around 0.85 by task 10. Lambda is 100. What should I try first?"

**What Claude helped with:**
Claude gave a ranked checklist in order of impact:

1. **Sweep lambda.** A single value rarely lands right on the first try. Try 50, 150, 500, 1500, 5000 and keep the flattest, highest curve.
2. **More capacity / longer training.** Raise `WIDTH` toward 2000 and `EPOCHS_PER_TASK` toward 40–100.
3. **Lower the learning rate** (e.g. `5e-4`). With many summed penalties the loss surface gets stiff; a smaller step is more stable and avoids the need for gradient clipping.
4. **More Fisher samples.** Raise `FISHER_SAMPLES` for a lower-variance importance estimate.
5. **Average over seeds.** Re-run `train_ewc_sequence(LAMBDA, seed=...)` for a few seeds and average the snapshots; single-seed curves dip noisily.

**Human-in-the-loop:**
Not every suggestion worked immediately. We ran multiple lambda values before settling on the config above, and each time returned to Claude with the updated curve to narrow the diagnosis. This iterative process reinforced that the lambda sweep is the single highest-leverage action — architecture and learning rate matter, but getting lambda into the right order of magnitude is what separates a drooping curve from a flat one.

---

## Human-in-the-Loop Principles We Applied

1. **No blind copy-paste** — every code snippet Claude wrote was read and understood before being run.

2. **Verify against the paper** — for every implementation decision, we checked Section 2 and Appendix 4.3 to confirm Claude's interpretation matched the original text. In cases of conflict, the paper took priority.

3. **Test incrementally** — when Claude suggested a change, we tested it on a 2-task, 1-epoch run before launching the full 10-task experiment.

4. **Trace failure modes explicitly** — rather than patching symptoms, we asked Claude to explain *why* each earlier notebook failed (batch-averaged Fisher, gradient clipping, weak capacity) before implementing fixes. Understanding the cause made the fix more reliable.

5. **Question the output** — we never assumed a graph was correct just because it ran without errors. We compared every figure against its counterpart in the paper, and returned to Claude with specific discrepancies rather than vague complaints.# AI Usage & Methodology — Catastrophic Forgetting Reproduction Project

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

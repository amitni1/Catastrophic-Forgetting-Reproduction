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
    model.zero_grad()
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

5. **Question the output** — we never assumed a graph was correct just because it ran without errors. We compared every figure against its counterpart in the paper, and returned to Claude with specific discrepancies rather than vague complaints.

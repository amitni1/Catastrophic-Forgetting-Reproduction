# Possible Mathematical Improvements to the EWC Model

This document collects mathematical extensions and refinements to the EWC objective used
in this project. It is meant as a "what we could try next" companion to
[`EWC_Model.md`](EWC_Model.md): each idea is written against our **actual** formulation and
code (the `compute_fisher` and `ewc_penalty_multi` helpers, the `1/N_tasks` normalization,
and the fixed `LAMBDA = 100`), and each notes the expected payoff and the implementation
cost.

Nothing here is required for the replication — the current configuration already reproduces
the paper's Figure 2A–C. These are directions for the bonus / "going beyond" part of the
project.

---

## Scope: why these were not run for this submission

The ideas below are presented as **derivations and implementation sketches rather than
measured results**, because validating any of them to the same standard as our main
replication was beyond the hardware budget available to us. The reason is worth stating
precisely, since it is not that a single run is too expensive.

A *single* Permuted-MNIST sequence finishes on one GPU in minutes (as noted in the README).
The cost is in **evaluation, not implementation**. To report a new variant honestly — the way
we report the baseline EWC result — we would have to hold it to our own documented standard:

- the full **10-task** sequence at the wide **width-2000** network (the Figure 2B recipe), not a quick width-400 sanity run;
- a **$\lambda$ sweep across ~2 orders of magnitude**, since `EWC_Model.md` shows that a new estimator or penalty scale invalidates the old $\lambda$ and forces re-tuning;
- **seed-averaging over several runs**, because our own troubleshooting notes that single-seed Figure 2B curves dip noisily and must be averaged.

Multiplying those together turns "a few minutes" into **dozens of full sequences per idea**,
and most of the improvements here are a *family* of configurations (e.g. each $\gamma$ for
online EWC, each $\alpha$ for gradient-balanced $\lambda$, each normalization scheme), so the
sweep is combinatorial. That repeated, hours-to-days-long sweeping — on a single shared /
session-limited GPU — is what exceeded our resources, not any one forward pass.

A few methods also raise the **per-run** cost or memory specifically, putting even one clean
run out of reach on modest hardware:

- **KFAC (§3.3)** stores and inverts per-layer Kronecker factors $A \otimes G$ — far more memory and compute per task than our diagonal Fisher.
- **Rotated EWC (§4)** adds rotation layers plus activation/gradient-statistics passes to build the basis, increasing both memory and wall-clock per task.
- **Online vs. separate EWC at scale (§3.2)** and the **deep Figure-2C-style runs** (100 epochs/task, 8192 Fisher samples) compound the above when swept properly.

So we scoped these as **theory-complete but empirically unverified** — each is written with
the exact formula, the change against our code, and the expected tradeoff, so they could be
run by anyone with more compute. This mirrors the scoping decision already documented in the
README for the paper's Atari experiments, which we likewise left out on compute grounds (depth
over breadth for a course project) rather than because they lacked value.

> **Status of each idea below:** *proposed and derived; not benchmarked.* Any accuracy or
> forgetting claims are expectations from the cited literature and our own analysis, not
> measurements from our runs.

---

## 0. The objective we currently optimize

Our training loop minimizes the **normalized separate-penalties** EWC loss:

$$
L(\theta) \;=\; \underbrace{L_B(\theta)}_{\text{CE on current task}}
\;+\; \frac{\lambda}{2}\cdot\frac{1}{N_{\text{tasks}}}
\sum_{t=1}^{N_{\text{tasks}}} \sum_i F^{(t)}_i \,\bigl(\theta_i - \theta^{*(t)}_i\bigr)^2
$$

where for each completed task $t$ we store its own diagonal Fisher $F^{(t)}$ and its own
weight anchor $\theta^{*(t)}$, and $\lambda = 100$ is fixed for the whole run.

Three structural choices in this line are exactly what the improvements below target:

1. the **scale** of $F^{(t)}$ is arbitrary, yet $\lambda$ is tuned against it (they are coupled);
2. $\lambda$ is a **single fixed constant** for every task and every epoch;
3. the importance estimate is a **diagonal, point-estimate** Fisher.

---

## 1. Normalizing the penalty

### 1.1 Fisher-scale normalization (decoupling $\lambda$ from the Fisher magnitude)

**Problem this fixes.** `EWC_Model.md` notes that *"Lambda and Fisher scale are coupled;
old-notebook values do not transfer"* and that retuning $\lambda$ across ~2 orders of
magnitude is needed whenever the Fisher estimator changes. That is a direct symptom of
feeding an **unnormalized** Fisher into the penalty: the raw magnitude of $F^{(t)}_i$ depends
on the loss scale, the number of classes, and the number of Fisher samples, so $\lambda$ has
to silently absorb all of that.

**The change.** Normalize each task's Fisher to a fixed scale *before* it enters the penalty.
Two natural choices:

$$
\hat F^{(t)}_i \;=\; \frac{F^{(t)}_i}{\bar F^{(t)}}, \qquad
\bar F^{(t)} = \frac{1}{|\theta|}\sum_j F^{(t)}_j
\qquad\text{(mean-normalized: average importance} = 1\text{)}
$$

$$
\hat F^{(t)}_i \;=\; \frac{F^{(t)}_i}{\sum_j F^{(t)}_j}
\qquad\text{(sum-normalized: total importance} = 1\text{)}
$$

Then the penalty uses $\hat F$ instead of $F$. Now $\lambda$ controls *relative* protection and
is invariant to the Fisher's absolute scale, so a good value should transfer across estimators,
sample counts, and even network widths.

**Implementation.** One extra block at the end of `compute_fisher`, before returning:

```python
# mean-normalize so the average Fisher entry is 1.0
total = sum(f.sum() for f in fisher.values())
count = sum(f.numel() for f in fisher.values())
mean  = (total / count).clamp_min(1e-12)
for n in fisher:
    fisher[n] /= mean
```

**Tradeoff.** Sum-normalization makes every task contribute the *same* total importance
mass regardless of how "spiky" its Fisher is, which can over-flatten a task that genuinely
relies on few weights; mean-normalization preserves the shape and only rescales. Prefer
mean-normalization as the default.

### 1.2 Rethinking the `1/N_tasks` averaging

**What we do now.** `ewc_penalty_multi` divides the summed penalty by the number of stored
tasks. This keeps the *total* penalty magnitude roughly constant as tasks accumulate — good
for $\lambda$ stability — but it has a side effect worth stating explicitly:

$$
\text{effective weight on a single old task} \;=\; \frac{\lambda}{2 N_{\text{tasks}}}
\;\propto\; \frac{1}{N_{\text{tasks}}}
$$

so the protection given to **each individual past task decays as $1/N$**. By task 10, task 1
is constrained at one-tenth the strength it had at task 2. This is effectively a built-in
"graceful forgetting" of the oldest tasks, which may partly explain the gentle decline in our
Figure 2B curve.

**Alternatives to try:**

- **No averaging + scale-normalized Fisher (§1.1).** If the Fisher is already scale-normalized,
  the plain sum $\sum_t \sum_i \hat F^{(t)}_i(\cdot)^2$ keeps every old task at full strength and
  $\lambda$ stays interpretable. Total penalty then grows $\mathcal{O}(N)$, which is the honest
  cost of protecting more tasks.
- **Decay-weighted sum.** Weight each task by recency, $w_t = \gamma^{\,N-t}$, with
  $\sum_t w_t$ normalized to 1. This makes the $1/N$ decay a *tunable* choice ($\gamma$) rather
  than an accidental one.

### 1.3 Per-layer / per-parameter normalization

Our Figure 2C analysis already shows Fisher mass is wildly **uneven across layers** (overlap
and magnitude both shift with depth). A single global $\lambda$ therefore constrains some
layers far more than others. A per-layer rescale,

$$
\hat F^{(t)}_{\ell, i} \;=\; \frac{F^{(t)}_{\ell, i}}{\bar F^{(t)}_{\ell}},
$$

(normalizing within each layer $\ell$) equalizes protection across depth. This is the same
idea as the **globally-normalized vs per-layer-normalized** distinction we already wrestled
with in `calculate_overlap_correct` — worth a short ablation, since the "right" choice is not
obvious and the paper itself normalizes globally for the overlap metric.

---

## 2. Making $\lambda$ dynamic

Right now $\lambda$ is one fixed number. Several principled ways to make it adapt:

### 2.1 A schedule over tasks

Let the constraint strengthen (or relax) as the sequence progresses:

$$
\lambda_t \;=\; \lambda_0 \cdot t
\qquad\text{or}\qquad
\lambda_t \;=\; \lambda_0\bigl(1 + \log t\bigr).
$$

A growing $\lambda_t$ counteracts the $1/N$ decay from §1.2 — as more old knowledge
accumulates, protect it harder. This is a one-line change (pass `lamda * schedule(task_id)`
into the penalty) and a cheap first experiment.

### 2.2 Per-task $\lambda^{(t)}$

Give each stored task its own weight inside the sum:

$$
L = L_B + \frac{1}{2}\sum_t \lambda^{(t)} \sum_i F^{(t)}_i(\theta_i - \theta^{*(t)}_i)^2.
$$

Useful if some tasks are known to be more "valuable" or more fragile than others. With
Permuted MNIST all tasks are symmetric, so this is more of a framework feature than an
expected win here — but it sets up §2.3.

### 2.3 Gradient-balancing $\lambda$ (the most promising adaptive variant)

Fix $\lambda$ not by hand but so that the **penalty gradient and the task-loss gradient have
comparable magnitude** at each step. Define

$$
\lambda_{\text{eff}} \;=\; \alpha \cdot
\frac{\lVert \nabla_\theta L_B \rVert}{\lVert \nabla_\theta L_{\text{EWC}} \rVert + \varepsilon},
$$

with a single dimensionless knob $\alpha$ (target ratio, e.g. 1.0). This is a GradNorm-style /
adaptive-weighting idea: instead of asking *"what number balances CE against the penalty?"*
(which changes with width, Fisher scale, and task index — exactly our tuning pain), you ask
*"what fraction of each step should go to remembering vs learning?"*, which is stable. It also
naturally relaxes the penalty early in a new task (when $\lVert\nabla L_B\rVert$ is large) and
tightens it as the new task converges.

**Implementation sketch** (computed once per epoch, or every $k$ steps, to keep it cheap):

```python
g_task = grad_norm(F.cross_entropy(model(data), target), model)
g_ewc  = grad_norm(ewc_penalty_multi(model, ewc_tasks), model)
lamda_eff = alpha * g_task / (g_ewc + 1e-12)
loss = F.cross_entropy(model(data), target) + (lamda_eff / 2.0) * ewc_penalty_multi(...)
```

### 2.4 Uncertainty / loss-ratio weighting

Treat the multi-objective balance as homoscedastic-uncertainty weighting (Kendall et al.,
2018): introduce a learnable log-variance $s$ and optimize
$\tfrac{1}{2}e^{-s} L_{\text{EWC}} + \tfrac{1}{2}s$ jointly with the weights. This lets the
network *learn* how much to weight the penalty. More moving parts than §2.3 and easy to
destabilize, so try §2.3 first.

### 2.5 Accuracy-feedback $\lambda$ (constrained / Lagrangian view)

§2.3 sets $\lambda$ from *gradient* magnitudes and §2.4 *learns* it as a weight. A more direct
alternative is to drive $\lambda$ from the quantity we actually care about — **task accuracy** —
and treat the whole problem as constrained optimization: minimize the current task's loss
**subject to** each past task's accuracy staying above a target $\tau$. Then $\lambda$ is the
Lagrange multiplier on that constraint, updated by **dual ascent**:

$$
\lambda \;\leftarrow\; \max\!\Bigl(0,\; \lambda + \eta\,\bigl(\tau - \widehat{\mathrm{acc}}_{\text{old}}\bigr)\Bigr)
$$

where $\widehat{\mathrm{acc}}_{\text{old}}$ is the mean accuracy over previously-seen tasks on a
held-out **validation** split and $\eta$ is a step size. When old-task accuracy dips below
$\tau$ the update is positive and $\lambda$ rises (protect harder); when old tasks sit
comfortably above $\tau$, $\lambda$ relaxes, freeing capacity for the new task. The "lower
$\lambda$ when the new task struggles" behaviour is automatic: because $\lambda$ only rises on a
violation, it settles at the smallest value that keeps old tasks above $\tau$ — which is the
most room the new task can have.

**Implementation** (computed every $k$ steps on **validation**, never test):

```python
old_acc = avg_val_acc(model, [tasks[t]['val'] for t in range(task_id)])  # past tasks only
lamda   = max(0.0, lamda + ETA * (TAU - old_acc))                        # dual ascent on lambda
# then use `lamda` in the usual ewc_penalty_multi for the next k steps
```

**Tradeoffs.**

- It replaces the *offline* $\lambda$ sweep — the expensive, combinatorial part flagged in the
  scope note above — with one online controller. This is the main appeal given how
  $\lambda$-sensitive and run-expensive our setup is. It does not remove tuning entirely:
  $\tau$, $\eta$, and the check interval $k$ are new knobs, though $\tau$ ("keep old tasks
  $\ge$ 92%") is far more interpretable than a raw $\lambda$.
- Steering $\lambda$ from accuracy requires a per-task validation split; using the test set
  here would be **leakage**. (Our notebook already holds out a 10k validation split — currently
  only the dropout baseline uses it — so the data is there.)
- Accuracy is a noisy, lagging signal, so a bang-bang rule (hard threshold) can **oscillate**;
  the proportional dual-ascent form above, optionally with an EMA on
  $\widehat{\mathrm{acc}}_{\text{old}}$ or a hysteresis band, damps that.
- A single scalar $\lambda$ still trades **all** old tasks against the new one with one lever;
  if $\tau$ is set higher than the network's capacity can satisfy, $\lambda$ saturates and the
  new task starves. Pairs naturally with the per-task $\lambda^{(t)}$ of §2.2 if independent
  control is needed.

**Why it's still listed as not-benchmarked.** Even though it removes the sweep, *demonstrating*
that the controller matches a properly hand-tuned $\lambda$ would itself require the multi-seed
comparison runs (controller vs. best swept $\lambda$) described in the scope note — so it stays
theory-complete but empirically unverified, like the rest of this document. Related framing:
inequality constraints on past-task losses are the basis of GEM (Lopez-Paz & Ranzato, 2017);
the dual-ascent multiplier update is the soft-penalty analogue.

---

## 3. Better importance estimates (beyond the diagonal point-estimate Fisher)

These change *what* $F$ is, not how it's scaled.

### 3.1 Fisher damping / a numerical floor

The diagonal Laplace approximation assumes the posterior is well-conditioned, but many
$F_i \approx 0$ (the network is over-parameterized). A small Tikhonov floor stabilizes the
penalty and prevents weights with near-zero estimated importance from drifting freely:

$$
\tilde F^{(t)}_i \;=\; F^{(t)}_i + \varepsilon, \qquad \varepsilon \sim 10^{-3}\,\bar F^{(t)}.
$$

This is half a line in `compute_fisher` and is essentially free. It tends to make EWC behave a
little more like the L2 baseline on the "unimportant" weights — which our Figure 2A suggests
isn't always bad.

### 3.2 Online EWC (a single running Fisher with decay)

Our separate-penalties scheme stores one Fisher **and** one anchor *per task*, so memory grows
$\mathcal{O}(\text{params} \times N)$. The **online EWC** reformulation (Huszár 2018; Schwarz
et al. 2018, *Progress & Compress*) keeps a single running Fisher and a single anchor at the
latest MAP estimate:

$$
\tilde F^{(t)} \;=\; \gamma\,\tilde F^{(t-1)} + F^{(t)},
\qquad
L = L_B + \frac{\lambda}{2}\sum_i \tilde F^{(t-1)}_i\bigl(\theta_i - \theta^{*}_i\bigr)^2,
$$

with one decay factor $\gamma \in (0,1]$ and a *single* anchor $\theta^{*}$ updated after each
task. Memory drops to $\mathcal{O}(\text{params})$, constant in $N$.

This is worth implementing precisely because our docs already contrast the two schemes —
adding online EWC as a toggle (`ONLINE_EWC = True`, store one running `fisher`/`anchor`
instead of appending to `ewc_tasks`) would let us **directly compare separate vs online EWC**
on the same Figure 2B axes. That is a clean, self-contained bonus result.

### 3.3 Kronecker-factored (KFAC) Fisher

The diagonal Fisher ignores all correlations between parameters. A block/Kronecker-factored
Laplace approximation (Ritter, Botev & Barber, 2018) approximates each layer's Fisher as a
Kronecker product $A \otimes G$ of input- and gradient-covariance factors, capturing
within-layer correlations at far less cost than the full matrix:

$$
F_\ell \;\approx\; A_{\ell-1} \otimes G_\ell .
$$

Strictly more faithful than the diagonal and known to reduce forgetting, but it is a
substantial implementation lift (per-layer covariance accumulation + a structured penalty).
Best framed as a stretch goal.

### 3.4 Synaptic Intelligence — importance from the optimization path

Instead of a post-hoc Fisher, accumulate importance *online during training* from how much
each weight's movement reduced the loss (Zenke, Poole & Ganguli, 2017):

$$
\omega_i^{(t)} = \sum_{\text{steps}} -g_i\,\Delta\theta_i,
\qquad
\Omega_i = \sum_{t} \frac{\omega_i^{(t)}}{(\Delta\theta_i^{(t)})^2 + \xi},
$$

then use $\Omega_i$ in place of $F_i$ in the same quadratic penalty. No separate Fisher pass is
needed (it piggybacks on training), which is appealing, though it introduces the damping
constant $\xi$ and is sensitive to learning-rate scale.

### 3.5 Memory Aware Synapses — unsupervised importance

MAS (Aljundi et al., 2018) defines importance from the sensitivity of the network's *output*
(the squared $\ell_2$ norm of the logits) rather than the loss, so it needs **no labels**:

$$
\Omega_i \;=\; \mathbb{E}\left[\left\lVert
\frac{\partial \, \lVert f(x;\theta)\rVert_2^2}{\partial \theta_i}\right\rVert\right].
$$

Same quadratic penalty, drop-in replacement for $F$. Mostly interesting as a robustness
comparison — it would tell us how much EWC's protection relies on the *label-defined* Fisher
versus the input geometry alone.

---

## 4. Reparameterization: Rotated EWC

The diagonal approximation is only good if the Fisher is *actually* near-diagonal in the chosen
coordinates. Rotated EWC (Liu et al., 2018) inserts fixed rotation layers so that the network is
reparameterized into a basis where the Fisher is much closer to diagonal, *then* applies the
ordinary diagonal penalty in that rotated basis. It keeps the cheap diagonal penalty but makes
the diagonal assumption far less lossy. Architecturally invasive (extra layers + a basis
computed from activation/gradient statistics), so this is the most advanced item on the list.

---

## 5. Summary — effort vs. expected payoff

| Improvement | What it targets | Effort | Why it's worth trying here |
| --- | --- | --- | --- |
| §1.1 Fisher-scale normalization | $\lambda$–Fisher coupling | **Low** | Directly fixes our documented "λ doesn't transfer" pain |
| §1.2 Rethink `1/N` averaging | Old-task decay | **Low** | Explains the Fig 2B droop; makes forgetting a tunable choice |
| §1.3 Per-layer normalization | Uneven depth importance | **Low** | Connects to our Fig 2C global-vs-layer finding |
| §2.1 $\lambda$ schedule | Fixed constraint strength | **Low** | One-line counter to the $1/N$ decay |
| §2.3 Gradient-balancing $\lambda$ | Hand-tuned $\lambda$ | **Medium** | Replaces fragile tuning with one stable knob $\alpha$ |
| §2.5 Accuracy-feedback $\lambda$ | Offline $\lambda$ sweep cost | **Medium** | Replaces the sweep with an online target on old-task accuracy ($\tau$) |
| §3.1 Fisher damping | Free-drifting weights | **Low** | Cheap numerical stabilizer |
| §3.2 Online EWC | $\mathcal{O}(N)$ memory growth | **Medium** | Clean separate-vs-online comparison on Fig 2B |
| §3.4 Synaptic Intelligence | Post-hoc Fisher cost | **Medium** | No extra Fisher pass; alternative importance |
| §3.5 MAS | Label-dependent importance | **Medium** | Unsupervised robustness comparison |
| §3.3 KFAC Fisher | Diagonal approximation | **High** | More faithful posterior, less forgetting |
| §4 Rotated EWC | Diagonal assumption | **High** | Makes the diagonal penalty much less lossy |

**Suggested order for a bonus section:** start with §1.1 + §1.2 (they cost almost nothing and
clean up the tuning story), add §2.3 (turns $\lambda$ into a self-adjusting quantity), then
implement §3.2 Online EWC as the headline comparison against our current separate-penalties
scheme.

---

## References

- Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks.* PNAS. [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)
- Huszár, F. (2018). *Note on the quadratic penalties in elastic weight consolidation.* PNAS.
- Schwarz, J. et al. (2018). *Progress & Compress: A scalable framework for continual learning.* ICML. (online EWC)
- Zenke, F., Poole, B. & Ganguli, S. (2017). *Continual learning through synaptic intelligence.* ICML.
- Aljundi, R. et al. (2018). *Memory Aware Synapses: Learning what (not) to forget.* ECCV.
- Ritter, H., Botev, A. & Barber, D. (2018). *Online structured Laplace approximations for overcoming catastrophic forgetting.* NeurIPS. (KFAC)
- Liu, X. et al. (2018). *Rotate your networks: Better weight consolidation and less catastrophic forgetting.* ICPR.
- Kendall, A., Gal, Y. & Cipolla, R. (2018). *Multi-task learning using uncertainty to weigh losses.* CVPR. (adaptive loss weighting)
- Lopez-Paz, D. & Ranzato, M. (2017). *Gradient Episodic Memory for Continual Learning.* NeurIPS. (inequality constraints on past-task losses)

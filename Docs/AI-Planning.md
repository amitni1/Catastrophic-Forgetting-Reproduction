# Overcoming Catastrophic Forgetting — EWC Implementation

## About This Document

This document captures the development process behind our EWC implementation, including the
questions we put to AI assistants — **Google Gemini** and **Anthropic Claude** — over the
course of the project.

A note on how we used them: we treated the assistants as a collaborator to think out loud
with, not as an oracle. Almost nothing below was copied verbatim. Several of the most useful
exchanges happened *because* an initial suggestion didn't work and we went back to argue with
it, and a few of our hardest bugs were things the "obvious" advice actively pointed us away
from. Where that happened, we've kept the messy version rather than tidying it into a
straight line — the dead ends are where we did the most learning, and they're the honest
record of how the code reached its final state. Every snippet here was adapted to our setup
and cross-checked against the paper and our own results before it went into the notebook.

---

## Background

**Catastrophic forgetting** is the phenomenon where a neural network, after learning a new
task, largely forgets previously learned ones — weight updates for the new task overwrite the
information encoded for the older ones.

The **EWC (Elastic Weight Consolidation)** algorithm addresses this by:

1. Identifying the weights most important to previous tasks using the **Fisher Information Matrix**.
2. Adding a **penalty term** to the loss that resists large changes to those weights.

The EWC loss function, in the paper's single-prior form:

$$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta^*_{A,i})^2$$

(How we extend this to 10 tasks — and why we divide the penalty by the number of tasks — is
covered in Q5 and Q7.)

> **Scope note:** This replication covers the MNIST experiments from the paper (Figures 2A, 2B,
> and 2C). The Atari reinforcement-learning experiments were not replicated — see Q12 for the
> reasoning.

---

## Development Q&A

### Q1: How should we organize the code?

**Us:** We're reproducing a paper and can't decide whether to keep everything in one notebook
or split it into proper modules. We don't want to lose marks for "messy structure," but we
also don't want to over-engineer a course project. What do you recommend?

**AI:** For a paper reproduction, a single notebook is usually the right call — it keeps your
code, plots, and results visible in one place, which makes the work much easier to present and
explain. The over-engineered version would be premature here. If you ever moved this toward
production, the natural split would be:

- `model.py` — network architectures
- `ewc.py` — EWC logic (Fisher computation + penalty)
- `train.py` — training loop
- `utils.py` — helpers and evaluation

But for your purposes one well-sectioned notebook keeps things clean and readable, and that's
what graders actually want to follow.

---

### Q2: How do we compute the Fisher Information Matrix?

**Us:** We keep seeing references to both "empirical Fisher" and "true Fisher" and people use
the terms as if the difference is obvious. It is not obvious to us. Which one does the paper
actually want?

**AI:** This distinction matters more than it looks. The **empirical Fisher** takes gradients
with respect to the *ground-truth* labels. It's the common shortcut, but it's a biased
estimate of the quantity the Laplace approximation calls for.

The **true Fisher** instead samples the label from the model's *own* predictive distribution
and accumulates the squared gradient one example at a time, so each contribution is a genuine
per-sample term. This is what the paper's derivation actually requires. The clean way to write
it:

$$F_i = \mathbb{E}_{x}\,\mathbb{E}_{y \sim p_\theta(y\mid x)}\!\left[\left(\frac{\partial \log p(y\mid x,\theta)}{\partial \theta_i}\right)^2\right]$$

```python
def compute_fisher(model, data_loader, num_samples=1024):
    """True diagonal Fisher: E[(d log p / d theta)^2], estimated per-sample."""
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

We didn't appreciate how much this mattered until it bit us — see Q3.

---

### Q3: Our Figure 2B collapsed and we can't find the bug

**Us:** Okay, we're officially stuck. We implemented EWC, we *think* correctly, and Figure 2B
is sitting around 80% average accuracy and refusing to climb. We've reread the paper three
times and stared at the training loop for the better part of two days. Is our code broken, or
is EWC just... like this?

**AI:** EWC is not "just like this" — an 80% plateau on Permuted MNIST is a classic symptom,
and the most likely culprit is the Fisher estimate, not the training loop. The specific trap:
if you compute the Fisher in mini-batches by averaging the gradients and *then* squaring, you
get $(\mathbb{E}[g])^2$ instead of $\mathbb{E}[g^2]$. Because $\mathbb{E}[g^2] \ge (\mathbb{E}[g])^2$,
batch-averaging **systematically under-estimates** the Fisher. The penalty comes out too weak,
EWC barely protects the important weights, and average accuracy sags exactly the way you're
describing.

The fix is to make the estimate genuinely per-sample: process one example at a time and square
*its* gradient before accumulating (the `compute_fisher` in Q2). If you were previously
batching, switch to the per-sample version and re-run Figure 2B before changing anything else.

**Us (after re-running):** That was it. The per-sample version pulled Figure 2B back up where
it should be. Two days for one missing `for` loop over the batch — noted, forever.

---

### Q4: How do we pick a value for λ (lambda)?

**Us:** The paper introduces λ, gives it a nice Greek letter, and then walks off into the
sunset without ever telling us what value to actually use. How are we supposed to choose this?

**AI:** Fair complaint — λ is genuinely setup-specific, and the paper's values don't transfer
because their network scale and optimizer differ from yours. Think of it as the stiffness of a
spring pulling weights back toward their old values:

- **λ too small** → the pull is too weak → the model still forgets.
- **λ too large** → the model is too rigid → it can't learn new tasks.

There isn't a formula; you sweep it. One thing to watch specifically in your setup: because
you sum a penalty per past task, the *effective* constraint grows as tasks accumulate, so a λ
that looks great after 2 tasks can strangle the network by task 9. Sweep across roughly two
orders of magnitude and keep the value that gives the flattest, highest Figure 2B curve.

**Us:** We landed on **λ = 100** after a sweep. We'll come back to the "penalty grows with
tasks" point, because it turned into its own decision (Q7).

---

### Q5: How do we extend EWC to more than 2 tasks?

**Us:** The paper's formula reads like a 2-task story (task A, task B). We're running 10. Do we
keep one running Fisher, or one per task, or something else?

**AI:** EWC extends to multiple tasks naturally, and you have two legitimate options:

1. **Separate penalties** — store a Fisher *and* an anchor for every completed task and sum a
   quadratic term for each. Since each term is quadratic, the sum is still a valid quadratic
   penalty. This is the form the paper describes.
2. **Online EWC** — keep a single running Fisher that you fold each new task into. Cheaper in
   memory, but a genuinely different algorithm with different behavior.

For a faithful reproduction, the separate-penalties form is the closer match. In our code we
add one twist — averaging the per-task terms (dividing by the number of stored tasks) so the
total constraint stays on a comparable scale as tasks pile up; Q7 explains why.

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

As each task finishes, you save its Fisher and its optimal weights (the anchor), then include
both in every later penalty.

---

### Q6: Now EWC won't learn the new tasks — what did we break this time?

**Us:** Good news: the Fisher fix from Q3 worked. Bad news: now the *opposite* problem. EWC
holds onto the old tasks beautifully but barely learns the new ones, and the average accuracy
is somehow even worse than before (~70%). We feel like we're playing whack-a-mole with this
model. We have a correct Fisher, a tuned-ish λ, and gradient clipping for stability — and it's
over-constrained. What's strangling it?

**AI:** This is the over-correction that often follows fixing the Fisher: a *correct* (larger)
Fisher, a stiff λ, gradient clipping, and a network that's a bit too small all push in the same
direction — toward a model that can't move at all. Two of those are doing more damage than you'd
expect:

- **Gradient clipping** is the quiet offender. EWC works by letting the penalty produce a real
  consolidation force on the important weights; clipping caps exactly that force, so the model
  never actually settles near the old weights *and* can't move freely on the new task. Remove it
  and let the penalty do its job.
- **Capacity.** Permuted MNIST asks the network to host many tasks at once. A narrow net has no
  room to find a low-interference configuration. The paper uses a wide network for Figure 2B —
  widen yours (up to ~2000 units) so EWC has somewhere to put the new task without overwriting
  the old.

If it's still stiff after that, drop the learning rate (e.g. 1e-3 → 5e-4) rather than reaching
for clipping again.

**Us (after):** Removing gradient clipping was the big one — embarrassing, since we'd added it
*for* stability. Widening to 2000 and leaving the LR at 1e-3 got Figure 2B up to the ~93–96%
range the paper reports. Lesson learned: "more regularization" is not a free lunch for EWC.

---

### Q7: Why do you divide the EWC penalty by the number of tasks?

**Us:** A reviewer (our own teammate) asked why we divide the penalty by the number of completed
tasks, since the paper never does this. We want to be honest about it rather than pretend it's
in the original.

**AI:** Be upfront about it — it's a defensible engineering choice, not a deviation you need to
hide. The principled alternative is to tune λ *specifically* for the full 10-task horizon,
because the raw summed penalty grows as tasks accumulate: a λ that's well-calibrated after 2
tasks will over-constrain the network by task 9 or 10. But finding that horizon-specific λ
means repeatedly running the entire 10-task sequence at 20–100 epochs per task across a λ
sweep, which was beyond the compute we had.

Dividing the summed penalty by the number of stored tasks keeps its magnitude roughly stable
across the sequence, so a λ tuned on shorter runs stays reasonable all the way to task 10. It's
not what the paper does, but it made our results substantially more consistent at no extra
compute. State it plainly in the write-up and it reads as good engineering judgment.

---

### Q8: How do we reproduce Figure C (Fisher overlap vs. layer depth)?

**Us:** Figure C shows how Fisher overlap between two tasks changes with layer depth. We don't
see how to (a) measure overlap or (b) control how similar two tasks are. Help?

**AI:** You need two ingredients: an overlap metric and a way to dial task similarity.

For the **overlap metric**, treat each task's flattened Fisher as a distribution: normalize it
to sum to 1, then use a Hellinger/Bhattacharyya-style measure. Per layer, overlap is
$1 - \tfrac{1}{2}\sum(\sqrt{s_A} - \sqrt{s_B})^2$, i.e. `1 - d²` for the squared Hellinger
distance `d²`. Overlap of 1 means the two tasks rely on identical parameters; 0 means
completely different ones.

For **task similarity**, use *partial* permutations — instead of shuffling the whole image,
shuffle only a square region in the center. A smaller shuffled region keeps the task close to
plain MNIST (high similarity); a larger one makes it very different (low similarity):

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

We use two sizes to match the paper's Figure 2C: **8×8** (small shuffled region → high task
similarity) and **26×26** (large shuffled region → low task similarity).

---

### Q9: Our Figure C curves looked plausible but slightly off — were we normalizing wrong?

**Us:** We got Figure C working, but the per-layer overlap values looked a little too tidy and
didn't match the paper's shape in the early layers. We suspect our normalization. We'd been
normalizing each layer's Fisher independently before computing overlap. Is that the problem?

**AI:** Yes — per-layer normalization is the likely cause. If you normalize each layer on its
own before measuring overlap, you throw away the *relative* Fisher mass between layers, which
is exactly the signal Figure C is about. The paper normalizes **globally**: concatenate every
layer's Fisher into one vector, normalize that whole vector once, and only then slice it back
per layer to compute overlap. Switch to global normalization and the early-layer behavior
should fall back into line.

**Us:** Switching to a single global normalization fixed the shape — the early layers dropped
the way the paper shows, especially for the 26×26 (dissimilar) tasks. Subtle bug, real
difference.

---

### Q10: Why use a separate model for the single-task reference line?

**Us:** For the dashed "ceiling" line in Figure 2B, can't we just reuse the EWC model's Task 0
accuracy? Building a whole separate model for one dashed line feels like overkill.

**AI:** Reusing the EWC model's Task 0 number would quietly bias the reference: it would be
entangled with the EWC run's specific λ, width, and seed, so you'd be comparing EWC against a
ceiling that EWC itself helped define. Training an independent single-task model (ideally on a
*different* seed) gives a clean, unbiased "this is what the architecture can do on one task with
no continual-learning pressure" line. It's a few extra epochs and it makes the comparison
honest — worth it.

**Us:** Agreed, and we used a separate seed for the reference model so it's clearly independent
of the main run.

---

### Q11: Mechanically, why does plain SGD forget but EWC doesn't?

**Us:** We get EWC conceptually, but our lecturer will absolutely ask us to explain the
*mechanism* out loud. Can you give us the under-the-hood version, not the hand-wavy one?

**AI:** Here's the mechanical version. Under plain SGD on task B, gradients are computed only
from task B's loss. They move whatever weights reduce that loss — including the ones that
encoded task A — and nothing pushes back. Those weights get repurposed for B, and A's
performance falls off. That's catastrophic forgetting, and it's not a bug; it's what
unconstrained gradient descent is supposed to do.

EWC adds a term that changes the geometry of the loss. A weight with a high Fisher value for
task A now pays a quadratic price for moving away from its task-A value, so the optimizer is
pushed to satisfy task B *while staying near* the configuration that worked for A. It's a soft
version of constrained optimization: not a hard wall, but a penalty steep enough that the
solution it finds is one that works for both. Weights with low Fisher stay free to adapt, which
is why EWC can still learn B at all.

---

### Q12: Should we also reproduce the Atari experiments?

**Us:** The paper has those impressive Atari reinforcement-learning results. Part of us wants to
attempt them. Talk us out of it (or into it).

**AI:** For a course project, talk yourselves out of it — and you can justify the cut cleanly.
The Atari experiments use the *same* EWC mechanism you've already implemented and tested on
MNIST: the same diagonal Fisher, the same quadratic penalty. So reproducing the MNIST figures
already exercises the method itself. What Atari adds is mostly *not* EWC — a full Double-DQN
agent plus a task-recognition module that has to infer which game is being played — i.e.
reinforcement-learning infrastructure. On top of that, the Atari runs train over hundreds of
millions of frames per agent across many games and seeds (GPU-weeks), versus minutes on a
single GPU for MNIST.

So scope it out as a deliberate "depth over breadth" decision, and say so explicitly: it's not
that the Atari results are unimportant — they're a major contribution of the paper — it's that
they don't test a new *mechanism* beyond what your MNIST reproduction already covers.

**Us:** That's the line we took, and we documented it in the README so it's clearly a choice and
not an omission.

---

### Q13: Could we add something of our own on top of the reproduction?

**Us:** We'd like a small original extension for the bonus, not just a pure reproduction.
Something that's genuinely motivated, not bolted on. Ideas?

**AI:** A clean, well-motivated extension is to combine EWC with a small **experience-replay
buffer** (ER + EWC). The motivation is principled: EWC's diagonal Fisher is a *point* estimate
that under-represents uncertainty, so it can let weights drift in directions it wrongly judges
safe — and it can't catch that drift, because its own estimate is the thing that's wrong.
Replaying a handful of real old examples catches exactly that failure directly. The two
mechanisms address *different* weaknesses, so together they should retain more than either
alone.

Keep it minimal so it's clearly your addition and easy to ablate: store a couple hundred real
(already-permuted) examples per finished task, mix a small batch of them back into the loss
during later tasks, and gate the whole thing behind one flag so you can show the with/without
comparison on the same Figure 2B axes.

**Us:** We implemented it behind a single `USE_REPLAY` switch (≈200 exemplars per task, replay
batch of 128). It kept the average accuracy nearly flat across the 10 tasks instead of sliding —
the with/without comparison is in the README bonus section. A nice result to end on.

---

### Q14: What are the real mathematical ways EWC itself could be improved?

**Us:** We've nailed the reproduction and the replay bonus, and now we're greedy — we want to
write up the ways EWC *itself* could be improved mathematically, not just reproduced. Two things
keep bugging us: our penalty normalization (the `1/N` divide) feels a bit ad-hoc, and the fixed
λ feels primitive. What are the principled levers here?

**AI:** Good instinct, and those two are exactly the right places to push. The levers group into
a few families:

- **Penalty normalization.** Your `1/N` divide is one choice, but the deeper fix for your
  documented "λ doesn't transfer" pain is **Fisher-scale normalization** — rescale each task's
  Fisher (e.g. by its mean) before it enters the penalty, so λ becomes invariant to the Fisher's
  absolute magnitude. Also worth examining: whether `1/N` is quietly decaying the oldest tasks'
  protection, and per-layer vs. global normalization (the same tension you already hit in the
  Figure 2C overlap metric).
- **Dynamic λ.** A schedule over tasks, a per-task λ, or — the strong one — a
  **gradient-balancing λ** that sets λ each step so the penalty gradient and the task-loss
  gradient have comparable magnitude. That replaces "guess a number" with "choose a target
  ratio," which is far more stable across width and task count.
- **Better importance estimates.** A small Fisher damping floor; **online EWC** (a single
  running Fisher with decay, instead of one per task); KFAC/block-diagonal Fisher; or
  path-integral / unsupervised importance (Synaptic Intelligence, MAS).
- **Reparameterization.** Rotated EWC, which makes the cheap diagonal assumption far less lossy.

Even if you can't benchmark all of them, write each up with its formula, the exact change
against your code, and the expected tradeoff — that's a legitimate contribution on its own.

**Us:** We did exactly that — it became a separate document, `EWC_Mathematical_Improvements.md`,
with each idea derived against our actual `compute_fisher` / `ewc_penalty_multi` code. One of
those adaptive-λ ideas was close enough to our own intuition that we worked it through
separately (Q15), and then we got ambitious and tried to actually *run* one of them (Q16).

---

### Q15: Could λ tune *itself* during training, from task accuracy?

**Us:** Here's an idea of our own we want to pressure-test before writing it up. λ is painfully
sensitive and re-running the full sweep is expensive — that's our whole compute complaint. So
what if λ adjusted itself *during* training instead: bump it up when an old task slips below,
say, 0.92, and ease it down when the current task isn't reaching a high enough accuracy? Would
that actually work, or are we missing something?

**AI:** The instinct is good, and it has a principled name: what you've described is a
**feedback controller on λ**, and the clean form of it is *constrained optimization*. Frame
continual learning as "minimize the current task's loss **subject to** each past task's
accuracy staying above a target τ." Then λ is the Lagrange multiplier on that constraint, and
the natural way to update it is **dual ascent**:

$$
\lambda \;\leftarrow\; \max\!\bigl(0,\; \lambda + \eta\,(\tau - \widehat{\mathrm{acc}}_{\text{old}})\bigr)
$$

When old-task accuracy dips below τ the bracket is positive and λ climbs (protect harder); when
old tasks sit comfortably above τ, λ relaxes. That's your "raise it when old tasks fail" rule,
but smooth instead of a hard jump.

One nice consequence: your *second* rule — lower λ when the new task struggles — you get for
free. Because λ only rises on a violation, it settles at the smallest value that keeps the old
tasks above τ, which is the most room the new task can possibly have. So one rule does the work
of two, which is also a cleaner thing to write up.

Three honest caveats, because they're exactly what a careful reader will ask about:

- **Use validation, never test.** Steering λ from accuracy means you're tuning on that
  accuracy — if it's the *test* set, that's leakage. You'd evaluate the EWC model on a held-out
  per-task validation split (you already keep one; right now only the dropout baseline uses it).
- **A hard threshold can oscillate.** Accuracy is noisy and lagging, so a strict
  "below 0.92 → jump" rule can hunt around the threshold. The proportional dual-ascent form
  above, optionally with an EMA on the measured accuracy or a small hysteresis band, damps that.
- **One knob, two jobs.** A single global λ still trades *all* old tasks against the new one
  with one lever, and if τ is set higher than the network's capacity can satisfy, λ saturates
  and the new task starves. If you need independent control it pairs with the per-task λ idea.

And to your motivation directly: yes, this mostly buys back the compute you care about — it
replaces the *offline* λ sweep (the expensive, combinatorial part) with one online controller.
It isn't free: τ, the step size η, and the check interval become new knobs, though τ
("keep old tasks ≥ 92%") is far more interpretable than a raw λ.

**Us:** We added this as its own section in `EWC_Mathematical_Improvements.md` (§2.5,
accuracy-feedback / constrained-λ), with the dual-ascent update, the validation-not-test note,
and the oscillation caveat. We kept it **proposed and derived, not benchmarked** like the rest:
even though it removes the sweep, *showing* the controller matches a hand-tuned λ would still
need the same multi-seed comparison runs we don't have — the same scope limit documented for
everything else in that file. It's a sibling to the gradient-balancing λ (Q16), which we did
briefly try to run.

---

### Q16: We hit a CUDA error trying to run the dynamic-λ method

**Us:** So we couldn't resist implementing the gradient-balancing (dynamic) λ to see it work.
Within a handful of steps the run died with `RuntimeError: CUDA out of memory`. Our normal
reproduction runs perfectly on the same machine, so what makes *this* variant fall over — is
our code wrong, or is our GPU just not big enough?

**AI:** This is very likely your hardware, not a logic bug — and it's a predictable consequence
of what the dynamic-λ variant has to do per step. Normal EWC does one forward + one backward.
The gradient-balancing version needs the **gradient norm of the penalty on its own**, separately
from the task loss, to form the ratio:

```python
g_task = grad_norm(F.cross_entropy(model(data), target), model)
g_ewc  = grad_norm(ewc_penalty_multi(model, ewc_tasks), model)   # extra full backward
lamda_eff = alpha * g_task / (g_ewc + 1e-12)
loss = F.cross_entropy(model(data), target) + (lamda_eff / 2.0) * ewc_penalty_multi(...)
```

That means an **extra full autograd graph and backward pass every step**, holding two complete
sets of parameter gradients in memory at once — and the penalty graph spans the
`(θ − θ*)²` terms for *every stored task*. Now stack the things that are already large in your
setup: the width-2000 network, the per-task **Fisher + anchor** store that keeps growing toward
task 10, the SGD momentum buffers, and the activation graph for a batch of 256. On a
memory-limited GPU, the dynamic-λ variant's peak footprint is roughly double the plain run's, and
that's what tips it over into out-of-memory. It's also more *compute* per step (two-to-three
passes instead of one), so even where it fits it's a lot slower.

There are ways to shrink it — compute the ratio every *k* steps instead of every step, free the
penalty graph immediately, drop the width, or use gradient checkpointing — but each of those
changes the thing you were trying to measure. To validate dynamic-λ *properly*, at the paper's
full width-2000 / 10-task / seed-averaged scale, you really do need a larger-memory GPU than
you have. This is the same wall described in the scope note of your improvements doc.

**Us:** That settled it. Rather than cripple the method just to make it fit (and then report a
result that isn't really the method), we left the gradient-balancing λ as **proposed and derived
but not benchmarked** in `EWC_Mathematical_Improvements.md`, with the out-of-memory limitation
written up honestly. The reproduction itself, which fits comfortably, stands on its own.

---

## References

- Kirkpatrick, J., Pascanu, R., Rabinowitz, N., et al. (2017). [Overcoming catastrophic forgetting in neural networks](https://arxiv.org/abs/1612.00796). *PNAS, 114(13), 3521–3526.*

# Project Takeaways — Reflective Writing

**Project:** Overcoming Catastrophic Forgetting with Elastic Weight Consolidation  
**Paper:** Kirkpatrick et al. (2017) — [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)

---

## What We Learned

### 1. Continual Learning is Not Trivial

Before this project, we assumed neural networks could naturally learn multiple tasks sequentially — much like humans do. The reality was sobering: **plain SGD almost completely forgets previous tasks within minutes of training on something new.** Watching the accuracy curves for an earlier task collapse the moment training moved on was eye-opening. It made us realize that "learning" in a neural network is fundamentally different from human learning — it is pure optimization for the current objective, with no intrinsic mechanism for preserving past solutions.

### 2. The Fisher Information Matrix is Simple to Compute, Deep in Meaning

When we first met the Fisher matrix we treated it as a formula to implement; what we came to appreciate is how much intuition it carries. It answers a deceptively simple question: *"How much does the model's output care about each weight?"* Computing the diagonal from gradients turned out to be straightforward, but the part that stuck with us was the interpretation — weights with high Fisher values are the ones a task genuinely depends on, while low-Fisher weights are essentially free capacity that a new task can repurpose. Once we saw that, EWC's "surgical precision" made sense to us: protect what matters, leave the rest flexible.

The deeper lesson was understanding *which* Fisher to compute. We started with the obvious version and only later understood why it was the wrong one — that journey is described under Challenges below, and it turned out to be the single most important thing we learned in the whole project.

### 3. Theory Meets Practice — and They Don't Always Match

The paper makes EWC sound simple: compute Fisher, add a penalty, done. In practice:
- **λ tuning is coupled to the Fisher scale.** The paper's values did not transfer to our setup, and — as we learned the hard way — neither did our *own* λ values once we changed how the Fisher was computed. λ and the Fisher magnitude are coupled, so changing one silently invalidates the other.
- **The penalty can quietly fail to activate.** Early on our penalty was effectively zero, and the model behaved exactly like plain SGD. Nothing errored; the regularization was simply too weak to matter.
- **Memory management matters.** Storing a separate Fisher matrix and weight anchor for each of 10 tasks on the GPU required deliberate planning, since the penalty list grows with every task.

### 4. SGD is Aggressively Forgetful

We expected gradual degradation. What we saw was **catastrophic collapse** — an earlier task's accuracy fell from the mid-nineties toward chance level after only a short spell of training on the next task. This was not gentle forgetting; it was closer to a hard reset. It made us appreciate why continual learning is still considered an open problem rather than a solved one.

### 5. Partial Permutations are a Clever Experimental Device

What we came to appreciate here was less the mechanism and more the cleverness of the experimental design. Figure 2C in the paper (Fisher overlap by layer depth) needs task pairs with *controlled* similarity, and at first it wasn't obvious to us how you would even create "more similar" or "less similar" tasks on demand. The partial permutation is what does it: shuffle only a square region in the centre of the image. An 8×8 permutation changes only a small patch, so the two tasks stay similar and their Fisher matrices overlap strongly; a 26×26 permutation scrambles most of the image, so the tasks become dissimilar and their early-layer Fishers overlap very little. The thing we took away is how this turns "task similarity" — which had felt like a vague, hand-wavy notion to us — into a single knob we can dial, and that is exactly what makes the overlap graph something you can actually read.

---

## Challenges We Faced

### Challenge 1: Understanding the Fisher Computation — and Getting it Wrong First

**The Problem:** The paper refers to the "Fisher Information Matrix" without spelling out *how* to compute it in practice. We initially thought we needed second derivatives (the Hessian), which would have been computationally infeasible for a network of this size.

**The First (Wrong) Resolution — the empirical Fisher:** We learned that the diagonal Fisher can be approximated using only first-order gradients, and our first implementation did the obvious thing: take the loss on the *ground-truth* labels, square the gradients, and average them over a batch.

```python
# Early version — empirical Fisher on ground-truth labels, batch-averaged
loss = F.cross_entropy(model(batch), targets)
loss.backward()
fisher += grad ** 2     # squared AFTER batch averaging — this is the bug
```

This ran without error and looked reasonable, but it produced a noticeably weak penalty, and the average accuracy in Figure 2B sagged toward ~0.8 no matter how we tuned λ.

**Understanding why it was wrong:** The Fisher we actually want is `E[(∂ log p(y|x,θ) / ∂θ)²]`, where the expectation over `y` is taken under the **model's own predictive distribution**, estimated one example at a time. Two things were wrong with our first version. First, it conditioned on the empirical (ground-truth) labels rather than sampling labels from the model — so it was the *empirical* Fisher, not the true Fisher the Laplace approximation calls for. Second, and more subtly, averaging gradients across a batch before squaring under-estimates curvature, because `E[g]² ≠ E[g²]` — batching washes out the per-sample signal that the Fisher is supposed to capture.

**The Real Fix — the true per-sample Fisher:** We rewrote the estimator to process one example at a time and to sample the label from the model's own softmax output:

```python
log_p = F.log_softmax(logits, dim=1)
p     = F.softmax(logits, dim=1)
y     = torch.multinomial(p, 1).squeeze(1)   # sample from the model's distribution
F.nll_loss(log_p, y).backward()
fisher[n] += par.grad.detach() ** 2          # square per sample, then average
```

This was the turning point of the project. The penalty became meaningful, and Figure 2B stopped collapsing. The lesson generalizes well beyond EWC: a quantity can be "computed correctly" in the sense of running without error while still being the wrong quantity, and the only way to catch that is to understand what the math is actually asking for.

### Challenge 2: Graphs Not Showing Catastrophic Forgetting

**The Problem:** Our first end-to-end run showed SGD and EWC performing almost identically — both held high accuracy on every task. The obvious question was: where is the catastrophic forgetting we came to study?

**The Root Cause:** We were accidentally evaluating on each task's *training* set rather than its *test* set. The network had effectively memorized the training data, so it looked as though it "remembered" earlier tasks.

**The Fix:** Strict train / validation / test separation per task, and evaluation exclusively on held-out test data in the accuracy-tracking loop. With clean evaluation, the forgetting reappeared immediately — a useful reminder that an encouraging-looking graph is not the same as a correct one.

### Challenge 3: The EWC Penalty Not Activating

**The Problem:** In the empirical-Fisher era (before Challenge 1's real fix), EWC performed the same as plain SGD. Print statements showed the penalty term was effectively zero.

**The Root Cause:** The under-estimated Fisher values were extremely small, so for any λ we were willing to try, the penalty `(λ/2)·F·Δθ²` was negligible next to the classification loss. We were tempted to compensate by pushing λ very high (into the thousands), and that did force the penalty to register — but it was treating the symptom. The genuine cause was the Fisher estimate itself, which Challenge 1 ultimately fixed.

**The Fix:** Once the true per-sample Fisher was in place, the Fisher magnitudes were on a sensible scale and a far more moderate λ did the job. This is why our final configuration uses **λ = 100** rather than the inflated value the broken estimator seemed to demand — and why the notebook warns that λ values do not transfer between Fisher formulations.

### Challenge 4: Extending EWC to 10 Tasks

**The Problem:** The paper's formula is written for two tasks (A and B). We needed to handle 10, and we were not sure whether to sum all penalties, keep only the most recent task's penalty, or use some weighted combination.

**The Resolution:** The paper notes in passing that the sum of quadratic penalties is itself a quadratic penalty, so the principled choice is to keep a *separate* Fisher and anchor for each completed task and add their penalties together. In our final implementation we go one step further and **average** the per-task penalties (divide by the number of stored tasks) so that the total constraint stays on a comparable scale as tasks accumulate, instead of growing with every new task. This kept a single tuned λ usable across the whole 10-task sequence.

---

## What We Would Do Differently

1. **Start simpler.** We jumped straight to 10 tasks. We should have validated the clean 2-task case first (SGD forgets, EWC remembers) before scaling up, which would have surfaced the evaluation bug and the Fisher-scale problem far sooner.

2. **Log more than accuracy.** We mostly tracked accuracy. We should also have logged Fisher statistics per layer (mean / max / min), the EWC penalty magnitude over time, and per-task loss. Had we watched the penalty magnitude from the start, the "penalty is effectively zero" problem would have been obvious immediately rather than after a round of debugging.

3. **Compare Fisher variants deliberately.** We ended up implementing the true per-sample Fisher, but only after the empirical version failed. A side-by-side comparison of the two estimators — with the resulting Figure 2B curves plotted together — would have made the difference between them a concrete, measured result rather than a debugging anecdote.

4. **Add a replay baseline.** We compared EWC to plain SGD, L2, and SGD+dropout, but not to experience replay (storing a small amount of data from old tasks). Replay is a natural and strong continual-learning baseline, and including it would have placed EWC's performance in fuller context.

---

## Personal Reflection

This project changed how we think about neural networks. We used to picture them as "learning machines" that accumulate knowledge over time. We now see them as **optimization machines**: they find a good solution for whatever objective is currently in front of them, with no built-in preference for preserving solutions to objectives they are no longer being shown.

The most instructive moment was not the final graph but the one before it — realizing that our "correct" Fisher computation was computing the wrong quantity entirely. The most *rewarding* moment came right after: watching the corrected run, where the SGD curves collapsed like a house of cards while the EWC curve stayed flat just below the single-task reference. Seeing the algorithm work, after understanding exactly why our first attempt had not, was the payoff.

---

## Conclusion

Reproducing a research paper is harder than reading one. The paper presents EWC as a clean, elegant idea, but turning that idea into a working replication required three distinct kinds of effort:

- **Understanding the mathematics** deeply enough to know what to compute — the Fisher as the precision of a Laplace-approximated posterior, and the crucial difference between the empirical and the true Fisher.
- **Debugging subtle, non-erroring failures** — an evaluation that secretly used the training set, and a penalty that silently vanished because the Fisher was under-estimated.
- **Making design decisions the paper leaves open** — λ and its coupling to the Fisher scale, the number of epochs and the network width, the depth of the Figure 2C network, and the choice to average rather than merely sum the per-task penalties.

We came away with much more respect for researchers who not only invent algorithms but make them work in practice. The gap between a clean equation on the page and a curve that behaves the way the paper promises is wide — and most of our learning lived inside that gap. If there is one transferable lesson, it is that in machine learning a result is only trustworthy once you understand *why* it looks the way it does, because code that runs cleanly can still be quietly answering the wrong question.

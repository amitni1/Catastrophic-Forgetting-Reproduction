# Project Takeaways — Reflective Writing

**Project:** Overcoming Catastrophic Forgetting with Elastic Weight Consolidation  
**Paper:** Kirkpatrick et al. (2017) — [arXiv:1612.00796](https://arxiv.org/abs/1612.00796)

---

## What I Learned

### 1. Continual Learning is Not Trivial

Before this project, I assumed neural networks could naturally learn multiple tasks sequentially — just like humans do. The reality shocked me: **plain SGD completely forgets previous tasks within minutes of training on something new.** Watching the accuracy graphs collapse was eye-opening. It made me realize that "learning" in neural networks is fundamentally different from human learning — it's pure optimization, with no intrinsic memory mechanism.

### 2. The Fisher Information Matrix is Beautiful

The Fisher matrix is elegant because it answers a simple question: *"How much does the loss care about each weight?"* Computing it from gradients is straightforward, but the insight is profound — weights with high Fisher values are critical to performance, while low-Fisher weights are essentially "free real estate" for new tasks. This gives EWC surgical precision: protect what matters, leave the rest flexible.

### 3. Theory Meets Practice — and They Don't Always Match

The paper makes EWC sound simple: compute Fisher, add a penalty, done. In practice:
- **λ tuning is trial and error.** The paper's values didn't work for my architecture. I spent hours tweaking λ from 10 to 10,000.
- **Fisher values can be tiny.** I had to debug why my penalty wasn't activating — turns out the Fisher values were in the 1e-6 range, so λ=10 was useless.
- **Memory management matters.** Storing Fisher matrices and optimal weights for 10 tasks on GPU required careful planning.

### 4. SGD is Aggressively Forgetful

I expected gradual degradation. What I saw was **catastrophic collapse** — Task 1 accuracy dropped from 95% to 15% after training on Task 2 for just one epoch. This wasn't gentle forgetting; it was a hard reset. It made me appreciate why continual learning is considered an open problem.

### 5. Partial Permutations are Genius

Figure C in the paper (Fisher overlap by layer depth) requires tasks with controlled similarity. The solution — partial permutations that only shuffle the center of the image — is brilliant. An 8×8 permutation keeps tasks similar (high Fisher overlap), while a 26×26 permutation makes them dissimilar (low overlap in early layers). This design trick made the graph interpretable.

---

## Challenges I Faced

### Challenge 1: Understanding the Fisher Computation

**The Problem:** The paper uses "Fisher Information Matrix" without explaining *how* to compute it in practice. I initially thought I needed second derivatives (Hessian), which would be computationally infeasible.

**The Resolution:** I learned about the **empirical Fisher approximation** — you can compute it using only first-order gradients on the ground-truth labels. This was a huge "aha!" moment. The code is simple:

```python
loss.backward()
fisher += grad ** 2
```

But understanding *why* this works required reading about the connection between Fisher and the Hessian at a local minimum.

### Challenge 2: Graphs Not Showing Catastrophic Forgetting

**The Problem:** My first implementation showed SGD and EWC performing identically — both maintained high accuracy. I was confused: where's the catastrophic forgetting?

**The Root Cause:** I was accidentally evaluating on the *training set* of each task, not the *test set*. The model had memorized the data, so it looked like it "remembered" previous tasks.

**The Fix:** Separate train/test splits for each task, and evaluate strictly on test data during the accuracy tracking loop.

### Challenge 3: EWC Penalty Not Activating

**The Problem:** EWC was performing the same as SGD. I added print statements and saw the penalty term was effectively zero.

**The Root Cause:** My Fisher values were in the range [1e-8, 1e-6], and I was using λ=10. The penalty was `10 * 1e-6 * (weight_change)^2 ≈ 1e-5`, which was negligible compared to the classification loss (~2.3).

**The Fix:** Increased λ to 1000. The penalty became significant enough to constrain the weights.

### Challenge 4: Extending EWC to 10 Tasks

**The Problem:** The paper's formula describes two tasks (A and B). I needed to handle 10 tasks, and I wasn't sure if I should:
- Sum all penalties together, or
- Use only the most recent task's penalty, or
- Use some weighted combination

**The Resolution:** The paper mentions in passing that "the sum of quadratic penalties is itself a quadratic penalty." This means you just sum them. Simple in hindsight, but not obvious at first.

---

## Key Insights

### Insight 1: EWC is a Bayesian Trick

EWC isn't just a regularization technique — it's an approximation of Bayesian posterior updates. The penalty term represents the prior from previous tasks: `p(θ | D_A)`. When learning task B, you're computing `p(θ | D_B, D_A)` without needing to store D_A. This connection to Bayesian inference made the algorithm click for me conceptually.

### Insight 2: Shared Representations Emerge Naturally

Figure C shows that dissimilar tasks (26×26 permutation) still share weights in the final layers. Why? Because the *output space is the same* — all tasks classify digits 0-9. The early layers learn task-specific features (edges, textures), but the final layers learn shared concepts (digit identity). EWC allows this naturally because only the task-specific weights have high Fisher values; the shared weights are reused freely.

### Insight 3: Catastrophic Forgetting is About Interference, Not Capacity

I initially thought forgetting happened because the network "ran out of space." But a 6-layer, 400-unit network has ~700,000 parameters — way more than needed for 10 permuted MNIST tasks. The problem isn't capacity; it's **gradient interference**. Task B's gradients destructively interfere with Task A's solution. EWC resolves this by creating a "no-go zone" around Task A's weights.

### Insight 4: Lambda is Problem-Specific

The paper doesn't give a formula for λ — it's a hyperparameter you tune. I found:
- **λ < 100:** Too weak, forgetting still occurs.
- **λ = 1000:** Sweet spot for my architecture.
- **λ > 5000:** Too rigid, new tasks don't learn well.

This taught me that EWC isn't a plug-and-play solution; it requires careful tuning for each problem.

---

## What I'd Do Differently

1. **Start simpler:** I began with 10 tasks immediately. I should have validated the 2-task case first (SGD forgets, EWC remembers), then scaled up.

2. **Log more metrics:** I only tracked accuracy. I should have logged:
   - Fisher matrix statistics (mean, max, min per layer)
   - EWC penalty magnitude over time
   - Per-task loss (not just accuracy)

3. **Experiment with Fisher variants:** The paper mentions other Fisher approximations (e.g., using the model's predicted distribution instead of ground truth). I stuck with the empirical Fisher, but comparing variants would have been interesting.

4. **Add a replay baseline:** I compared EWC to plain SGD, but not to experience replay (storing some data from old tasks). That would have been a fairer comparison.

---

## Personal Reflection

This project changed how I think about neural networks. I used to see them as "learning machines" that accumulate knowledge. Now I see them as **optimization machines** — they find a solution for the current objective, with no inherent bias toward preserving old solutions.

Humans don't have catastrophic forgetting (we forget gradually, not instantly). This suggests our brains have mechanisms — maybe synaptic consolidation, maybe replay during sleep — that networks lack. EWC is a step toward that, but it's still a hack. True continual learning will require rethinking network architectures from the ground up.

The most rewarding moment was watching the accuracy graphs after getting EWC working: SGD's lines collapsed like a house of cards, while EWC's lines stayed flat. That visual proof that the algorithm *works* was deeply satisfying.

---

## Conclusion

Reproducing a research paper is harder than reading it. The paper presents EWC as a clean, elegant solution, but the implementation required:
- Understanding the math (Fisher, Laplace approximation, Bayesian posteriors)
- Debugging subtle issues (Fisher scale, evaluation bugs, permutation logic)
- Making design choices not mentioned in the paper (λ, epochs, architecture depth)

I now have much more respect for researchers who not only invent algorithms but also make them work in practice. The gap between "theory" and "running code" is wide, and this project taught me how to bridge it.

**Would I recommend this project?** Absolutely. If you want to understand continual learning, there's no substitute for implementing it yourself and watching catastrophic forgetting happen in real time.

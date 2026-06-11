# Catastrophic Forgetting & EWC Reproduction Project

> **Original paper**
> *James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell (2017)*
>
> *Overcoming catastrophic forgetting in neural networks*
> [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) — *PNAS, 114(13), 3521–3526*

This project reproduces the **Permuted MNIST** continual-learning experiments (Figure 2A–C) from Kirkpatrick et al. (2017), evaluating Elastic Weight Consolidation against baselines over a sequence of tasks.

This project is part of a submission for a course in Python by students **Amit Nigerker** and **Yuval Holoidovsky**.

---

## Result

This project has two layers, and they tell different halves of the same story.

**The faithful reproduction (paper-exact EWC).** Using the paper's own method — separate-penalties EWC at λ = 100, with no extra mechanism — we reproduce the paper's *qualitative* finding: EWC learns each new task well (per-task test accuracy stays around 95–97%, 96.8% on task 10) and remains **at or above the SGD+dropout baseline at every task**. But under our compute budget (single seed, 20 epochs/task), the *average* accuracy across all tasks seen so far still slides gently as tasks accumulate — from **95.2% after task 2 to 88.9% after task 10**, against a single-task ceiling of 95.4%. So the paper-exact method clearly beats the baselines and slows forgetting, but on its own and at this scale it does not hold the average flat.

**The bonus extension (ER + EWC).** Adding a small experience-replay buffer on top of EWC (~200 real exemplars per task) is what holds the line. The average accuracy stays nearly flat across the whole 10-task sequence — **95% → 93.6%**, a drop of about one point instead of six — and ends roughly 4.7 points higher than pure EWC at task 10. The two mechanisms address different failure modes (EWC's diagonal Fisher is a point estimate that lets weights drift in directions it wrongly judges safe; replay catches that drift directly), so together they retain more than either alone. Details and the with/without comparison are in the [Bonus](#bonus--combining-ewc-with-a-replay-buffer-er--ewc) section.

Our Fisher-overlap experiment (Figure C) reproduces the paper's key structural finding: dissimilar tasks share little in the early layers but converge to near-identical overlap (~0.999) in the deeper, output-facing layers.

> **Scope:** This replication covers the **MNIST experiments only** (Figures 2A, 2B, and 2C). The Atari 2600 reinforcement-learning experiments from the paper were not reproduced.
> 
---

## Table of Contents
- [Academic documentation](#academic-documentation)
- [Background](#background)
- [Why We Chose This Paper](#why-we-chose-this-paper)
- [Project Overview](#project-overview)
- [Architecture & Parameters](#architecture--parameters)
- [How to Run](#how-to-run)
- [Results & Evaluation](#results-and-graphs)

---

## Academic documentation
We documented every step we took to reach our results:
- [AI drafting](Docs/AI-Planning.md) — how we used raw AI prompting to steer the project in the direction we wanted.
- [AI methodology](Docs/AI_METHODOLOGY.md) — how we used AI to code and assemble the project.
- [EWC model in code](Docs/EWC_Model.md) — a cell-by-cell explanation of how the EWC code works.
- [Validation and testing](Docs/EWC_VALIDATION.md) — how we validated and tested the EWC implementation.
- [Graphs and results](graphing_and_results.md) — reproduction of the paper's figures and the project's results.
- [Conclusions](Docs/TAKEAWAYS.md) — main takeaways from the project and our personal reflections.
-[athematical Improvements](Docs/EWC_Mathematical_Improvements.md) — main takeaways from the project and our personal reflections.
---

## Background

### Catastrophic Forgetting
In neural networks, learning tasks one after another usually causes the network to overwrite what it learned earlier in order to fit the new task. This failure mode is known as **catastrophic forgetting**, and it is one of the main obstacles to continual learning — the ability to keep acquiring new skills without losing old ones.

### Elastic Weight Consolidation (EWC)
EWC slows down learning on the weights that mattered most for previous tasks. It measures each parameter's importance using the **diagonal of the Fisher Information Matrix**, which serves as a Laplace approximation to the posterior over weights after a task is learned. Important weights are pulled back toward their old values by a quadratic penalty, while unimportant weights stay free to adapt to the new task.

This implementation uses the **separate-penalties** form of EWC (based on equation 3 of Kirkpatrick et al.): after each completed task we store that task's Fisher matrix and its weight anchor, and every future task is penalized against all stored tasks. In our code the per-task penalties are **averaged** (divided by the number of stored tasks) so the total constraint stays on a comparable scale as tasks accumulate:

$$L = L_B + \frac{1}{N_{\text{tasks}}} \sum_{\text{tasks}} \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta^*_i)^2$$

This is a normalized variant of the paper's plain-sum separate-penalties formulation. It is distinct from **Online EWC**, which instead merges all past Fishers into a single running estimate. Storing a Fisher and anchor per task costs more memory, but keeps each task's consolidation signal clean.

A key implementation detail is the **true per-sample Fisher estimation**: for each example, the label is sampled from the model's *own* predictive distribution (rather than using the ground-truth label or batch-averaged gradients), and the squared gradient is accumulated one example at a time. This gives the true diagonal Fisher the Laplace approximation calls for, and avoids the systematic under-estimation that batch averaging introduces.

---

## Why We Chose This Paper

We wanted a paper we could actually take apart and rebuild ourselves, not just run and admire. EWC fit that: it tackles a problem that's easy to *see* — a network that learns task B and visibly forgets task A — but the fix rests on a genuinely interesting idea (treating the old task's knowledge as a Bayesian prior and using the Fisher to decide which weights to protect). That gave us something to understand mathematically, not only to copy.

It was also practical for a course project. The MNIST experiments run on a single GPU, the method needs no exotic components beyond a diagonal Fisher and a quadratic penalty, and the paper gives three concrete figures (2A–C) we could check our work against one by one. So we got to engage with a real, well-known result end to end — derive it, implement it, debug it, and compare our graphs to the originals — which is exactly the kind of learning we were hoping to get out of the project.

---

## Project Overview

The experiment trains models sequentially across **10 Permuted MNIST tasks**:
1. **Task 1:** the original MNIST dataset (standard digit classification).
2. **Tasks 2–10:** Permuted MNIST — the input pixels are shuffled by a distinct fixed random permutation per task.

The 60,000-image MNIST training set is split once into **50,000 training and 10,000 validation** images (held out with a fixed seed), and the standard **10,000-image test set** is used for reported accuracy. The same split is applied under every task's permutation, so each task has its own train/validation/test views of the data.

Three models are compared through the sequence:
1. **SGD:** a baseline trained with plain stochastic gradient descent, with no mechanism to protect prior tasks.
2. **L2:** a uniform quadratic-regularization baseline that penalizes all weight changes equally (no Fisher-based weighting).
3. **EWC:** trained with the separate-penalties EWC objective described above. After each task, its Fisher matrix and weight anchor are stored and folded into the penalty for all future tasks.

A separate **single-task reference model** is trained on Task 1 only and used as the performance ceiling (the dashed line in Figure B).

For Figure B, the SGD baseline is replaced with an **SGD + dropout** variant (0.2 dropout on the input, 0.5 on hidden layers) with per-task early stopping based on held-out validation accuracy — matching the dropout baseline the paper compares against.

**Why we did not reproduce the Atari part.** The Atari experiments use the *same* EWC algorithm as the MNIST ones — the same diagonal-Fisher importance estimate and the same quadratic penalty — so reproducing the MNIST figures already implements and tests the method itself; the Atari section only shows that this same method also scales to a harder domain, not a new mechanism. We scoped it out for two practical reasons. First, **compute**: the Atari runs train over hundreds of millions of frames per agent, across many games and random seeds (on the order of GPU-weeks), whereas the MNIST runs finish on a single GPU in minutes. Second, **machinery that isn't EWC**: the Atari setup is a full Double-DQN reinforcement-learning agent plus a separate task-recognition module that has to *infer which game is being played* — most of which is reinforcement-learning infrastructure rather than EWC. So this was a scope decision (depth over breadth for a course project), not a judgment that the Atari results are unimportant; they are a significant contribution of the paper.

---

## Architecture & Parameters

### Neural Network Structure (Figures A & B)
A Multi-Layer Perceptron (MLP) with configurable hidden width. The same architecture is used for both figures, but at **different widths** — matching the paper, which uses a narrow network for Fig 2A and a wider one for Fig 2B:
- **Input layer:** $28 \times 28 = 784$ units (flattened MNIST image).
- **Hidden layer 1:** ReLU — width **400** for Figure 2A, **2000** for Figure 2B.
- **Hidden layer 2:** ReLU — same width as hidden layer 1.
- **Output layer:** 10 units (cross-entropy loss, digits 0–9).

Figure 2A is trained at width 400 (the paper-exact narrow network) for 20 epochs; Figure 2B uses the wider width-2000 network. The dropout baseline uses the same architecture with an added input dropout layer (0.2) and hidden dropout layers (0.5).

### Neural Network Structure (Figure C only)
A deeper MLP used solely for the Fisher-overlap experiment:
- **6 hidden layers** of 100 units each (ReLU).
- **Output layer:** 10 units.

### Hyperparameters
- **Number of tasks:** 10
- **Hidden width:** 400 for Figure 2A, 2000 for Figure 2B (Figure 2C uses the deep 6×100 network)
- **Epochs per task:** 20 (the paper uses up to 100 for Fig B)
- **Learning rate:** 0.001
- **Momentum:** 0.9 (SGD)
- **Batch size:** 256
- **EWC lambda ($\lambda$):** 100
- **Fisher samples:** 2048 (true per-sample estimate; Figure 2C uses 8192)
- **L2 lambda:** 1.0 (uniform penalty for the L2 control)
- **Early-stop patience:** 5 epochs (SGD+dropout baseline)
- **Validation split:** 50,000 train / 10,000 validation (held out from the training set, seed 42) / 10,000 test
- **Figure 2C epochs:** 100 (the deep-network Fisher-overlap run)
- **Random seed:** 0 (single-task reference uses seed 99; data split uses seed 42)

---

## How to Run

The full project lives in a single notebook, `ewc_replication.final.ipynb`.

1. Set up the environment (PyTorch + torchvision; a CUDA GPU is used if available, otherwise CPU):
   ```bash
   pip install torch torchvision matplotlib numpy
   ```
2. Launch Jupyter and open the notebook (the kernel is named `ewc_gpu`):
   ```bash
   jupyter notebook ewc_replication.final.ipynb
   ```
3. Run the cells top to bottom. MNIST downloads automatically on first run. The notebook trains the sequences and writes the three figures (`fig2A.png`, `fig2B.png`, `fig2C.png`).

The run is seeded (`SEED = 0`) for reproducibility. Note there is still some inherent variance from GPU non-determinism and early-stopping termination points.

---

## Results and Graphs

Results are visualized across three benchmarks mirroring the paper's Figure 2.

### 1. Figure A — Per-Task Accuracy Timeline (EWC vs L2 vs SGD)
Tracks test accuracy on the first three tasks epoch-by-epoch across the first 60 epochs of sequential training (3 tasks × 20 epochs each). This matches the paper's three-condition Fig 2A (EWC, L2, SGD).

![Continual Learning Accuracy Timeline](images/continual_learning_V4.png.png)

- **SGD (blue):** suffers from catastrophic forgetting. Task A accuracy is stable while training on Task A but erodes as the network is repurposed for Tasks B and C.
- **L2 (green):** the uniform penalty partially slows forgetting but cannot tell critical weights from non-critical ones.
- **EWC (red):** maintains Task A accuracy through training on Tasks B and C by anchoring the important weights. All three methods learn the current task well, confirming EWC does not block forward learning.

### 2. Figure B — Average Accuracy Across All Tasks Seen
Average fraction correct across all tasks learned so far, measured at the end of each task, from task 2 through task 10. This is the like-for-like comparison the paper makes: EWC vs SGD+dropout.

![Average Performance Over Tasks](images/image.png)

- **EWC (red):** `[0.952, 0.950, 0.938, 0.941, 0.936, 0.923, 0.901, 0.908, 0.889]`
- **SGD+dropout (blue):** `[0.949, 0.947, 0.922, 0.912, 0.904, 0.904, 0.886, 0.875, 0.868]`
- **Single-task reference:** 0.954 (dashed line)
- **EWC across the sequence:** 0.952 at task 2 → 0.889 at task 10 (a gentle 6.3-point decline)

EWC stays above 88% across the full 10-task sequence and remains close to the single-task ceiling throughout. It is at or above the SGD+dropout baseline at every task, reproducing the *direction* of the paper's result. Note that this is the faithful, paper-exact run: the average still declines gently across the sequence. Holding it flat is exactly what the replay extension below adds.

### 3. Figure C — Fisher Overlap vs Network Depth
Overlap between the diagonal Fisher matrices of two sequentially trained tasks, computed per layer across the 6-hidden-layer network, for two permutation regimes.

![Fisher Overlap vs Network Depth](images/fisher_overlap_v4.png)

- **Low permutation — 8×8 patch** (grey): `[0.754, 0.981, 0.997, 0.999, 0.999, 0.999]`
- **High permutation — 26×26 patch** (black): `[0.541, 0.901, 0.979, 0.996, 0.998, 0.998]`

When the two tasks are similar (small permuted region), Fisher overlap is high across all layers — both tasks rely on the same weights, and EWC protects those shared representations. When the tasks are dissimilar (large permuted region), overlap at the input layer drops sharply — the network must learn very different early-layer features — but converges toward 1.0 in the deeper layers, since both tasks ultimately produce the same 10-class output. This validates the Fisher Information Matrix as a meaningful proxy for parameter importance and for task similarity.

---

## Bonus — Combining EWC with a Replay Buffer (ER + EWC)

As an extension, we added a small **experience-replay buffer** on top of EWC: after each task we keep ~200 of its real (already-permuted) examples (`REPLAY_PER_TASK`), and while training later tasks we mix a small batch of them (`REPLAY_BATCH = 128`) back into the loss. It is controlled by one `USE_REPLAY` flag and only affects the red EWC curve in Figures 2A/2B; the baselines and Figure 2C are unchanged. The objective becomes the EWC loss plus a cross-entropy term on the replayed examples:

$$L = \underbrace{L_B + \frac{1}{N_{\text{tasks}}}\sum_{\text{tasks}}\sum_i \frac{\lambda}{2} F_i(\theta_i-\theta^*_i)^2}_{\text{EWC penalty}} \;+\; \underbrace{\text{CE}\big(\text{replayed old examples}\big)}_{\text{replay term}}$$

**Configuration.** The replay is controlled by one `USE_REPLAY` switch plus two new parameters; everything else (the EWC penalty, the baselines, Figure 2C) is unchanged:

```python
# --- replay buffer (ER+EWC hybrid: store a few real exemplars per task) ---
USE_REPLAY      = True
REPLAY_PER_TASK = 200      # real exemplars kept per finished task (~1800 by task 10)
REPLAY_BATCH    = 128      # how many we replay back into each training step
```

Set `USE_REPLAY = False` to recover the original pure-EWC behaviour.

**Why it helps.** EWC's diagonal Fisher is a point estimate that under-estimates uncertainty (the paper's own Figure 3C result), so it lets weights drift in directions it wrongly judges safe — and it cannot catch that drift, because its own estimate is the thing that is wrong. Replaying a few real old examples catches it directly. The two address different failure modes, so together they retain more than either alone.

**Result.** Replay keeps the average accuracy almost flat across the 10-task sequence instead of sliding:

![Average Performance Over Tasks](images/average_accuracy_bonus.png)

| | Task 2 | Task 10 | Drop over sequence |
|---|---|---|---|
| EWC only (paper-exact) | 0.952 | 0.889 | −6.3 pts |
| EWC + replay (bonus) | 0.949 | 0.936 | −1.3 pts |

With a single-task ceiling of 0.954, replay cuts the decline from 6.3 to 1.3 points and ends ~4.7 points higher at task 10.

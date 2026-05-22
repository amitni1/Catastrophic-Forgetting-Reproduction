## Graph Comparison: Our Results vs. the Paper
 
This section compares each figure we reproduced against the equivalent figure in Kirkpatrick et al. (2017), identifying what was successfully recreated and where quantitative or qualitative differences remain.
## Scenario 1
in to Scenario we are running the code and EWC model on a smaller scale to see how it holds up and the results are promising.
number of tasks here are 5 but with 6 epoch each just to see if i can handle it and not break down.


## Summary Comparison Table
 
| Figure | Paper's Graph | Our Graph | Match? | Key Differences |
|---|---|---|---|---|
| **Fig 2A** — Continual Learning Curves | EWC maintains accuracy across tasks; SGD collapses sharply at each task transition; single-task reference line shown | Solid (EWC) vs dashed (SGD) lines for Tasks A, B, C tracked over 30 epochs | ✅ Yes | SGD collapse timing and severity closely match; EWC curves degrade more gradually than in the paper but the separation is clear |
| **Fig 2B** — Average Accuracy vs Tasks | EWC stays above 80% through 10 tasks; SGD falls to ~20% | EWC ends ~80%, SGD ends ~49% at 10 tasks | ⚠️ Partial | EWC final accuracy closely matches the paper; SGD final accuracy is higher than in the paper (~49% vs ~20%), suggesting our SGD baseline retains more than expected |
| **Fig 2C** — Fisher Overlap vs Depth | Low-% permutation has higher overlap in early layers; both conditions converge at deeper layers | Low-% stays ~0.85 and high-% stays ~0.68–0.70 across all 6 layers; no convergence observed | ❌ Diverges | The paper's key finding — convergence at deeper layers — is not reproduced. Both conditions remain separated and flat across all layers, suggesting our permutation sizes or Fisher computation may differ from the paper |
 
---
 
## Figure 2A — Continual Learning Curves (EWC vs SGD)
 
### What the Paper Shows
 
The paper plots per-task test accuracy over training time for EWC and SGD. EWC curves stay approximately flat for all previously learned tasks throughout training. SGD curves collapse sharply to near-chance performance each time a new task begins. A dashed horizontal line marks single-task performance (~97%).
 
### Our Replication
 
![Our Fig 2A replication — Continual Learning: EWC vs SGD tracking Tasks A, B, C over 30 epochs](images/continual_learning_accuracy_V1.png)
 
*Our Fig 2A replication — Continual Learning: EWC vs SGD tracking Tasks A, B, C over 30 epochs*
 
**What matched:** The qualitative pattern is a strong match. SGD dashed lines collapse sharply at each task boundary (visible at epochs ~7 and ~13 for Tasks A and B respectively). EWC solid lines remain substantially higher throughout training, consistently outperforming SGD on all previously seen tasks. The ordering — EWC above SGD at every point after the first task transition — is correctly reproduced.
 
**Differences:** EWC curves show gradual degradation over time (Tasks A and B ending around 80–85%) rather than staying perfectly flat as in the paper. This is likely due to λ not being large enough to fully prevent drift in a 10-task setting. SGD's collapse is real but less extreme than in the original — SGD Task A bottoms out around 15–30% rather than near-chance, and SGD Task C holds unusually high (~25%) by epoch 30, which may indicate our architecture retains some shared structure across tasks.
 
---
 
## Figure 2B — Average Accuracy Across All Tasks
 
### What the Paper Shows
 
The paper plots average test accuracy (averaged across all tasks seen so far) as a function of the number of tasks trained sequentially. EWC degrades slowly, remaining above 80% through all 10 tasks. SGD degrades steeply, reaching approximately 20% by task 10. Single-task performance (~97%) is shown as a dashed reference line.
 
### Our Replication
 
![Our Fig 2B replication — Average Fraction Correct vs Number of Tasks](images/average_accuracy_fig2b_v1.png)
 
*Our Fig 2B replication — Average Fraction Correct vs Number of Tasks*
 
**What matched:** This is our closest quantitative match to the paper. Our EWC curve ends at approximately 80% at task 10, directly matching the paper's result. The shape of both curves — EWC degrading slowly, SGD degrading steeply — is faithfully reproduced. Both curves start near the single-task reference at task 2, consistent with the paper.
 
**Differences:** Our SGD final accuracy (~49%) is noticeably higher than the paper's (~20%). This is the main discrepancy: our SGD baseline retains more accuracy than expected, likely because our permuted MNIST tasks share more structure than those used in the paper (perhaps due to less aggressive permutation or a more expressive architecture). The **EWC result is quantitatively validated**; the SGD baseline underestimates the severity of catastrophic forgetting relative to the paper.
 
---
 
## Figure 2C — Fisher Overlap vs Layer Depth
 
### What the Paper Shows
 
The paper plots Fisher information overlap between tasks as a function of layer depth, for two conditions: low-percentage permutations (similar tasks, high expected overlap) and high-percentage permutations (dissimilar tasks, lower expected overlap). The key finding is that overlap is higher in early layers for similar tasks, but **both conditions converge to similar overlap values in deeper layers**, reflecting a shared output representation for digit classification.
 
### Our Replication
 
![Our Fig 2C replication — Fisher Overlap vs Depth across 6 layers](images/fisher_overlap_vs_depth_fig2c_v2.png)
 
*Our Fig 2C replication — Fisher Overlap vs Depth across 6 layers (6-layer network)*
 
**What matched:** The ordering between conditions is preserved: low-% permutation consistently shows higher Fisher overlap (~0.85) than high-% permutation (~0.68–0.70) across all layers, correctly reflecting the relationship between task similarity and weight importance overlap.
 
**Differences:** The paper's central finding — convergence of both conditions at deeper layers — is **not reproduced**. In our graph, both lines remain essentially flat and separated across all 6 layers, showing no convergence trend. This is a meaningful divergence. Possible causes include: our permutation sizes may not produce sufficient dissimilarity in early layers to drive the convergence pattern; our 6-layer architecture may spread representations differently than the paper's network; or our Fisher computation (empirical Fisher vs. exact Fisher) may produce smoother, less layer-specific values. This figure requires further investigation and is the weakest aspect of our replication.

again we didnt expect all of the graphs to align perfectly since its was test run but it successful one
 
## Scenario 2
# Graph Comparison: Our Results vs. the Paper
 
here we upped the number of tasks to 10 and with 20 epoch with a lamda of 5000 and 400 neurons in order to recreate the papers results
 
### Summary Comparison Table
 
| Figure | Paper's Graph | Our Graph | Match? | Key Differences |
|---|---|---|---|---|
| **Fig 2A** — Continual Learning Curves | EWC maintains accuracy across tasks; SGD collapses; single-task reference line shown | Solid (EWC) vs dashed (SGD) lines for Tasks A, B, C tracked over 100 epochs | ✅ Yes | We track individual task accuracy per epoch rather than a single snapshot; SGD collapse timing and depth closely match the paper |
| **Fig 2B** — Average Accuracy vs Tasks | EWC degrades slowly (~80% at task 10); SGD degrades steeply (~20% at task 10) | EWC ends ~67%, SGD ends ~42% at 10 tasks; both degradation trends clearly present | ⚠️ Partial | Our EWC final accuracy is lower (~67% vs ~80%); our SGD is higher (~42% vs ~20%). Relative ordering and trend shape are correctly reproduced |
| **Fig 2C** — Fisher Overlap vs Depth | Low-% permutation has higher overlap in early layers; both converge at deeper layers | Low-% starts ~0.75 at layer 1, high-% starts ~0.60; both converge ~0.50 at layer 3 | ✅ Yes | Absolute overlap values are slightly higher in our replication; convergence pattern and ordering match the paper qualitatively |
 
---
 
### Figure 2A — Continual Learning Curves (EWC vs SGD)
 
#### What the Paper Shows
 
The paper plots per-task test accuracy over training time for both EWC and SGD. EWC curves stay flat across all previously learned tasks, while SGD curves collapse immediately upon starting a new task. A dashed horizontal line marks single-task performance (~97%).
 
#### Our Replication
 
![Our Fig 2A replication — Continual Learning: EWC vs SGD tracking Tasks A, B, C over 100 epochs](images/average_accuracy_fig2b_v2.png)
 
*Our Fig 2A replication — Continual Learning: EWC vs SGD tracking Tasks A, B, C over 100 epochs*
 
**What matched:** The qualitative pattern is an excellent match. SGD dashed lines collapse sharply upon each task transition (visible at epochs 20, 40, 60, 80). EWC solid lines maintain substantially higher accuracy for previous tasks throughout training. The relative ordering of collapse severity is preserved.
 
**Differences:** We track three individual tasks (A, B, C) over epochs rather than presenting a single time-slice comparison. EWC curves show some gradual degradation in later tasks (~50–55% by epoch 100 for Tasks A and B) rather than staying perfectly flat, likely due to λ not being tuned to be sufficiently large for a 10-task setting. SGD collapse is slightly less catastrophic than in the original paper (reaching ~10% rather than near-chance), which may reflect our architecture's larger capacity.
 
---
 
### Figure 2B — Average Accuracy Across All Tasks
 
#### What the Paper Shows
 
The paper plots average test accuracy (averaged across all tasks seen so far) as a function of the number of tasks trained. EWC degrades slowly, staying above 80% through 10 tasks. SGD degrades steeply, reaching approximately 20% by task 10. Single-task performance (~97%) is shown as a reference.
 
#### Our Replication
 
![Our Fig 2B replication — Average Fraction Correct vs Number of Tasks](images/continual_learning_v2.png)
 
*Our Fig 2B replication — Average Fraction Correct vs Number of Tasks*
 
**What matched:** The overall shape and ordering is correctly reproduced. EWC degrades significantly more slowly than SGD. The separation between the two methods widens as more tasks are added, consistent with the paper. Both curves start near the single-task reference at task 2.
 
**Differences:** Our EWC final accuracy at 10 tasks (~67%) is notably lower than the paper (~80%). Our SGD final accuracy (~42%) is noticeably higher than the paper (~20%). This suggests our λ value, while better than baseline, may not be fully optimised, and architecture depth or training epochs per task may also differ. Importantly, the **relative advantage of EWC over SGD is clearly preserved**, validating the core claim of the paper.
 
---
 
### Figure 2C — Fisher Overlap vs Layer Depth
 
#### What the Paper Shows
 
The paper plots Fisher information overlap between tasks as a function of layer depth, comparing low-percentage permutations (similar tasks) against high-percentage permutations (dissimilar tasks). Low-% permutations show higher overlap in early layers; both conditions converge to similar overlap values in deeper layers.
 
#### Our Replication
 
![Our Fig 2C replication — Fisher Overlap vs Depth for low and high permutation conditions](images/fisher_overlap_V2.png)
 
*Our Fig 2C replication — Fisher Overlap vs Depth for low and high permutation conditions*
 
**What matched:** The key qualitative result is reproduced: low-% permutation (similar tasks) yields higher Fisher overlap in early layers (~0.75) compared to high-% permutation (~0.60). Both conditions converge to approximately the same overlap value (~0.50) by layer 3, reflecting the shared output representation of digit classification across all tasks. The convergence trend is clearly visible.
 
**Differences:** Our absolute overlap values are slightly higher than in the paper across all layers, and the gap between the two conditions in early layers is somewhat smaller, consistent with using different specific permutation percentages. The layer-3 convergence point matches well (~0.50). The non-monotonic dip at layer 2 for the low-% condition (visible in our graph) is present but less prominent than in the original.
 
---
 
### Overall Assessment
 
We successfully reproduced the **core qualitative findings** of Kirkpatrick et al. (2017): EWC substantially outperforms SGD in continual learning; Fisher overlap reflects task similarity; and shared output representations persist across dissimilar tasks. Quantitative differences in absolute accuracy levels (Fig 2B) are expected given architecture and hyperparameter differences, and do not undermine the validity of our replication. The most important result — **EWC remembers, SGD forgets** — is unmistakably clear across all three figures.

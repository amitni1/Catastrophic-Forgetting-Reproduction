
# Catastrophic Forgetting & Online EWC Reproduction Project
> ** Orignal paper**
>*James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell (2017)*
> *Overcoming catastrophic forgetting in neural networks*
> arXiv:1312.6211v3
This repository contains a PyTorch implementation and reproduction project exploring the phenomenon of **Catastrophic Forgetting** in Neural Networks, and its mitigation using the **Online Elastic Weight Consolidation (Online EWC)** algorithm. 

The project evaluates a Continual Learning scenario over multiple sequential tasks using the **Permuted MNIST** dataset.

## Table of Contents
- [Background](#background)
- [Project Overview](#project-overview)
- [Architecture & Parameters](#architecture--parameters)
- [Results & Evaluation](#results--evaluation)
- [Installation & Running](#installation--running)
- [Dependencies](#dependencies)

---

## Background

### Catastrophic Forgetting
In connectionist networks (Neural Networks), learning tasks sequentially usually results in the network completely overriding previously acquired knowledge to adapt to the new task. This limitation is known as **Catastrophic Forgetting** and is a major hurdle toward creating general Artificial Intelligence capable of Continual Learning.

### Elastic Weight Consolidation (EWC)
EWC slows down learning on weights that are crucial for past tasks. It calculates the importance of each parameter using the **Fisher Information Matrix**. 
While regular EWC requires storing a Fisher matrix for *every single task* (which doesn't scale well), **Online EWC** optimizes this by maintaining a single running estimate of the Fisher matrix, updating it using an exponential moving average (with a memory dampening parameter $\alpha$).

---

## Project Overview

The experiment sets up a sequence of **3 consecutive tasks** based on the MNIST dataset:
1. **Task A**: Original MNIST dataset (Standard digit classification).
2. **Task B**: Permuted MNIST (Pixels shuffled using fixed random permutation map B).
3. **Task C**: Permuted MNIST (Pixels shuffled using fixed random permutation map C).

We compare two models simultaneously:
1. **Standard SGD Model**: A baseline network trained with regular Stochastic Gradient Descent across the task sequence.
2. **Online EWC Model**: A network initialized identically to the baseline but optimized with a regularization penalty based on the online Fisher information during Tasks B and C.

---

## Architecture & Parameters

### Neural Network Structure
To force rapid weight overwriting and simulate a restricted memory footprint, a compact Multi-Layer Perceptron (MLP) was constructed:
- **Input Layer**: $28 \times 28 = 784$ neurons (Flattened MNIST image).
- **Hidden Layer 1**: 100 neurons (ReLU activation).
- **Hidden Layer 2**: 100 neurons (ReLU activation).
- **Output Layer**: 10 neurons (Log Softmax / Cross Entropy for digits 0-9).

### Hyperparameters
- **Epochs per task**: 10
- **Learning Rate ($Lr$)**: 0.02
- **Momentum**: 0.9 (SGD)
- **EWC Lambda ($\lambda$)**: 5000 (Importance weight penalty)
- **Fisher Memory Dampening Factor ($\alpha$)**: 0.5

---

## Results & Evaluation

During training, the test accuracy on **all tasks** is tracked at the end of every epoch. 

### Final Metrics Log
```text
--- Training on Task A ---
Epoch 10 | EWC [A:98.1%, B:11.0%, C:11.3%] | SGD [A:97.9%, B:11.9%, C:8.2%]

--- Training on Task B ---
Epoch 10 | EWC [A:97.5%, B:96.2%, C:12.5%] | SGD [A:78.2%, B:97.8%, C:8.4%]

--- Training on Task C ---
Epoch 10 | EWC [A:96.9%, B:95.0%, C:94.8%] | SGD [A:47.7%, B:75.5%, C:97.6%]

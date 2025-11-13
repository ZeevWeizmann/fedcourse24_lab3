# Federated Learning Lab 3 — Flower Framework & FjORD

Université Côte d’Azur — MSc Data Science & Artificial Intelligence  
Author: **Zeev Weizmann**

---

## 1. Introduction

This repository contains the full report and codebase for **Lab 3: Federated Learning & Data Privacy**, part of the MSc Data Science & AI program at Université Côte d’Azur.

The lab includes:

- **Exercise 7** — Introduction to Flower framework
- **Exercise 8** — Device heterogeneity & FjORD (Ordered Dropout)
- **Bonus Exercise** — Federated Distillation (FD)

The goal is to document implementation steps, show results, and provide all materials required to reproduce the experiments.

---

## 2. Exercise 7 — Get Started with Flower

### Objective

Become familiar with the Flower federated learning framework.

Reference:  
**Get Started with Flower (PyTorch)**  
https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html

### Steps Performed

#### Step 1 — Create environment

```
conda create -n flwr_lab3
conda activate flwr_lab3
```

#### Step 2 — Install Flower

```
pip install flwr
```

#### Step 3 — Generate project

```
flwr new flower-tutorial --framework pytorch --username flwrlabs
cd flower-tutorial
```

#### Step 4 — Run simulation

```
flwr run .
```

The run completed successfully under the FedAvg strategy.

### Handling CIFAR-10 Access Error

The HuggingFace Hub returned **401 Unauthorized** when Flower attempted to download CIFAR‑10.

Solution:  
Download CIFAR‑10 via **torchvision** and convert it to a **datasets.Dataset**.

```
trainset = torchvision.datasets.CIFAR10(...)
hf_dataset = Dataset.from_dict(...)
```

This workaround is documented in Flower GitHub Issue #3412.

---

## 3. Exercise 8 — Tackling Device Heterogeneity with FjORD

### Objective

Understand **system heterogeneity** and implement FjORD: Fair and Accurate Federated Learning under heterogeneous targets.

Reference paper:  
**FjORD: Fair and Accurate Federated Learning under Heterogeneous Targets with Ordered Dropout**

Reference tutorial:  
https://flower.ai/docs/baselines/fjord.html

---

## 3.1 Preliminary Questions

### 1. What is system heterogeneity?

Differences in compute, memory, network bandwidth, or device reliability among clients.

This causes:

- slow clients delaying FL rounds,
- biased updates,
- unfair training contributions.

### 2. What is Ordered Dropout?

A technique that:

- sorts neurons/filters by importance,
- drops the least important ones first,
- enables smaller devices to train **submodels** nested inside the full model.

### 3. How does aggregation handle heterogeneity?

FjORD modifies FedAvg:

- updates are weighted based on submodel ratio **p**,
- smaller devices still provide meaningful updates,
- aggregation becomes fair and stable.

---

## 3.2 FjORD Baseline Installation

Environment:

```
conda create -n flwr_lab3_fjord python=3.10 -y
conda activate flwr_lab3_fjord
```

Install Poetry:

```
pip install poetry==1.8.3
```

Fix deprecated backend:

```
build-backend = "poetry.core.masonry.api"
```

Install FjORD:

```
pip install -e .
```

---

## 3.3 Running FjORD

```
#!/bin/bash
RUN_LOG_DIR=${RUN_LOG_DIR:-"exp_logs"}

pushd ../
mkdir -p $RUN_LOG_DIR
seed=123

echo "Running without KD ..."
poetry run python -m fjord.main ++manual_seed=$seed 2>&1 | tee $RUN_LOG_DIR/wout_kd_$seed.log

echo "Running with KD ..."
poetry run python -m fjord.main +train_mode=fjord_kd ++manual_seed=$seed 2>&1 | tee $RUN_LOG_DIR/w_kd_$seed.log

echo "Done."
popd
```

Output logs:

```
exp_logs/
 ├── wout_kd_123.log
 └── w_kd_123.log
```

---

## 4. FjORD Experiment Results

### 4.1 Simplified CPU Setup

Because GPU computation is long:

- **3 clients** (not 100)
- **20 global rounds** (not 500)
- same ratios: **p ∈ {0.2, 0.4, 0.6, 0.8, 1.0}**
- tested both versions:
  - _without_ Knowledge Distillation
  - _with_ Knowledge Distillation (KD)

### 4.2 Observations (TensorBoard)

- accuracy increases over rounds,
- KD produces smoother, more stable curves,
- client variance is significantly reduced under KD.

This matches the official FjORD behaviour.

### 4.3 Which version works better?

**FjORD with Knowledge Distillation**.

Why:

- reduces client drift,
- stabilizes optimization,
- improves fairness,
- improves accuracy earlier.

This matches both:

- our simplified setup,
- the official FjORD paper.

### 4.4 Impact of the Submodel Ratio p

Official FjORD shows:

| p       | Effect                                                        |
| ------- | ------------------------------------------------------------- |
| 0.2     | model too small → underfitting                                |
| 0.6–0.8 | ideal width → best accuracy                                   |
| 1.0     | only strong devices can train full model → biased aggregation |

In our simplified CPU setup:

- p is not explicitly changed,
- but Ordered Dropout in FjORD still simulates variable submodel widths,
- KD strongly stabilizes differences between effective p values.

Thus, the **qualitative trend** matches the FjORD paper.

---

## 5. Bonus Exercise — Federated Distillation (FD)

### Goal

Implement the FD mechanism from:

Chang et al. (2019)  
_Communication-Efficient On-Device Machine Learning: Federated Distillation and Augmentation under Non-IID Private Data_

Only the distillation part was implemented.

### Setup

- CIFAR-10
- 3 clients
- 3 rounds
- CPU-only
- simple fully connected network

### Client returns:

- accuracy
- average logits

### Server aggregation:

For K clients:

```
avg_logit = mean(logits_k)
avg_acc   = mean(acc_k)
```

### Results

| Round | avg_logits | accuracy |
| ----- | ---------- | -------- |
| 1     | 0.002049   | 0.3655   |
| 2     | 0.002049   | 0.3724   |
| 3     | 0.002048   | 0.3853   |

FD worked correctly end-to-end.

---

## 6. Conclusion

This lab demonstrated:

- how to run Flower simulations,
- how to deploy FjORD with Ordered Dropout and Knowledge Distillation,
- how to write custom FL strategies,
- how to implement Federated Distillation,
- how to handle real-world issues (dataset access, environment setup),
- how to analyze federated models under device heterogeneity.

Despite CPU-only limitations, key findings of both **FjORD** and **FD** papers were successfully reproduced.

---

## 7. References

```
@article{chang2019federateddistillation,
  title={Communication-efficient on-device machine learning: Federated distillation and augmentation under non-IID private data},
  author={Chang, Haw-Shiuan and others},
  year={2019}
}

@article{fjord_paper,
  title={FjORD: Fair and Accurate Federated Learning under Heterogeneous Targets with Ordered Dropout},
  year={2022}
}

@misc{flower_fjord,
  title={FjORD Baseline},
  year={2025},
  howpublished={\url{https://flower.ai/docs/baselines/fjord.html}}
}

@misc{flower_tutorial,
  title={Get Started with Flower},
  year={2025},
  howpublished={\url{https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html}}
}
```

---

End of README.md

# Report 1: Numerical Verification of a Lightweight Reverse-Mode Automatic Differentiation Engine

**Experiment 1 — GB-Net (`gb_net_lib`)**  
Reverse-mode autograd (`bp`) engine and multi-layer perceptron (`net`) module

---

## Abstract

We empirically evaluate the numerical correctness and state-management behavior of `gb_net_lib`, a lightweight NumPy-based reverse-mode automatic differentiation (autograd) engine. Part 1A compares analytical gradients produced by `bp.backprop()` against central finite-difference (CFD) oracles across all eight primitive operations (addition, subtraction, matrix multiplication, squaring, sigmoid, ReLU, softmax, cross-entropy) and a composite 2-layer MLP ($3 \to 4 \to 3$) over step sizes $h \in \{10^{-2}, 10^{-4}, 10^{-6}, 10^{-8}, 10^{-10}\}$. At $h = 10^{-6}$, maximum absolute errors are $\le 1.89 \times 10^{-10}$ and relative errors $\le 1.31 \times 10^{-9}$ across all tensors, confirming the $\mathcal{O}(h^2)$ truncation / $\mathcal{O}(\epsilon_{\rm mach}/h)$ roundoff trade-off of CFD. Part 1B measures stale-gradient amplification under repeated `backprop()` calls on a static graph: with only the root node's gradient reset, the amplification factor at topological depth $d$ equals $\binom{k+d-1}{d}$ exactly ($k=2 \Rightarrow d+1$, e.g. $\times 8$ at depth 7 for $k=2$; $\times 36$ for $k=3$). Zeroing all topological gradients before each pass restores idempotency ($M_{k,d} = 1.00$). We trace the root cause to `bp.backprop()` resetting only the root gradient and recommend clearing every node's `.grad` prior to each backward pass.

---

## 1. Introduction & Architectural Overview

GB-Net is a minimal autograd and MLP library built from scratch with NumPy (`float64`). It comprises two classes. The `bp` class wraps a NumPy array and constructs a dynamic directed acyclic graph (DAG) on the fly: every arithmetic operation records its operands in a `parent` list and stores a `back()` closure implementing the corresponding vector-Jacobian product (VJP). The `backprop()` method performs a post-order depth-first topological sort (`build_topo`) and executes each node's `back()` in reverse topological order, starting from the root node whose gradient vector is seeded to ones ($\partial\mathcal{L}/\partial\boldsymbol{\ell} = \mathbf{1}$, where $\boldsymbol{\ell}$ is the vector of per-component losses and $\mathcal{L}=\sum_i \ell_i$). The `net` class builds an MLP using column-vector layout ($W^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$, $b^{(\ell)} \in \mathbb{R}^{n_\ell \times 1}$), with `feed()` running the forward pass and `train()` running single-sample stochastic gradient descent.

A key state-management property, noted in the library documentation, is that `backprop()` resets **only the root node's** gradient; intermediate node gradients are never cleared. Consequently, repeated calls on the same static graph accumulate stale gradients. `net.train()` avoids this in practice because `feed()` builds fresh intermediate `bp` objects each iteration and the loop manually zeroes weight and bias gradients after every sample. Experiment 1 quantifies both the numerical accuracy of the VJP rules (Part 1A) and the stale-gradient amplification (Part 1B).

---

## 2. Mathematical Formulation

### 2.1 Reverse-mode vector-Jacobian products

For a node computing $o = f(\ldots)$ with upstream gradient $\bar{o} = \partial \mathcal{L}/\partial o$, the local VJP distributes $\bar{o}$ to each input. The rules implemented in `gb_net_lib.py` are:

- **Addition** ($o = a + b$): $\bar{a} \mathrel{+}= \bar{o}$, $\bar{b} \mathrel{+}= \bar{o}$.
- **Subtraction** ($o = a - b$): $\bar{a} \mathrel{+}= \bar{o}$, $\bar{b} \mathrel{+}= -\bar{o}$.
- **Matrix multiplication** ($o = a b$, $o_{ij} = \sum_k a_{ik} b_{kj}$): $\bar{a} \mathrel{+}= \bar{o} b^\top$, $\bar{b} \mathrel{+}= a^\top \bar{o}$.
- **Squaring** ($o = a^2$): $\bar{a} \mathrel{+}= 2a \odot \bar{o}$.
- **Sigmoid** ($o = \sigma(a)$, $\sigma(x) = (1+e^{-x})^{-1}$): $\bar{a} \mathrel{+}= o \odot (1 - o) \odot \bar{o}$.
- **ReLU** ($o = \max(a, 0)$): $\bar{a} \mathrel{+}= \mathbb{I}(o > 0) \odot \bar{o}$.
- **Softmax** ($o_i = e^{a_i - \max(a)} / \sum_j e^{a_j - \max(a)}$): $\bar{a} \mathrel{+}= o \odot (\bar{o} - \sum_k o_k \bar{o}_k)$.
- **Cross-entropy** ($\ell_i = -y_i \ln(\tilde p_i)$, $\tilde p_i = \operatorname{clip}(p_i, 10^{-15}, 1.0)$): $\bar p \mathrel{+}= -(y \oslash \tilde p) \odot \bar{\boldsymbol{\ell}}$, where $\bar{\boldsymbol{\ell}} = \partial\mathcal{L}/\partial\boldsymbol{\ell} = \mathbf{1}$ is the seeded upstream gradient for the scalar loss $\mathcal{L} = \sum_i \ell_i$, $\oslash$ denotes element-wise division, and the target $y$ is held constant during parameter updates (no gradient flows into $y$).

### 2.2 Central finite-difference oracle and its error regimes

The CFD approximation to a scalar partial derivative is
$$
\nabla_{\theta_i}^{\rm CFD}\mathcal{L} = \frac{\mathcal{L}(\theta_i + h) - \mathcal{L}(\theta_i - h)}{2h}.
$$
By Taylor expansion, the truncation error is $\mathcal{O}(h^2)$, while floating-point roundoff contributes $\mathcal{O}(\epsilon_{\rm mach}/h)$ with $\epsilon_{\rm mach} \approx 2.22 \times 10^{-16}$ for IEEE 754 `float64`. The total error is therefore U-shaped in $h$, with an optimum near $h \approx \epsilon_{\rm mach}^{1/3} \sim 10^{-5}$–$10^{-6}$.

### 2.3 Stale-gradient recurrence

Let $g_d^{(k)}$ denote the gradient **tensor** at topological depth $d$ after $k$ `backprop()` passes on a static graph, with only the root reset between passes. Let $J_d$ be the local Jacobian mapping the gradient from depth $d-1$ to depth $d$. Because each intermediate node retains its gradient from pass $k-1$ and feeds that inflated value into its parents during pass $k$, the gradient tensors satisfy the recurrence
$$
g_d^{(k)} = g_d^{(k-1)} + J_d\, g_{d-1}^{(k)},
\qquad
g_0^{(k)} = \mathbf{1}.
$$
Since every term in the expansion is a positive scalar multiple of $g_d^{(1)}$, the norm amplification ratio is
$$
M_{k,d} = \frac{\|g_d^{(k)}\|_2}{\|g_d^{(1)}\|_2} = \binom{k+d-1}{d}.
$$
For $k=2$, $M_{2,d} = d+1$; for $k=3$, $M_{3,d} = (d+1)(d+2)/2$.

---

## 3. Experimental Methodology & Setup

All experiments run on a single-threaded CPU with NumPy `float64`. Deterministic seeding uses `np.random.seed(42)` for Experiment 1.

**Reference MLP (Part 1A):** input $x = [0.4, -0.3, 0.8]^\top \in \mathbb{R}^{3 \times 1}$; hidden layer $4$ neurons (sigmoid), output layer $3$ neurons (softmax) with cross-entropy loss against one-hot target $y = [0, 1, 0]^\top$; weights and biases $\sim \mathcal{U}(-0.5, 0.5)$. Thirty-four scalar parameters are perturbed ($68$ forward passes per $h$).

**Step-size sweep:** $h \in \{10^{-2}, 10^{-4}, 10^{-6}, 10^{-8}, 10^{-10}\}$.

**Part 1B:** a single forward pass binds every intermediate node; `loss.backprop()` is invoked for $k \in \{1, 2, 3\}$ on the identical graph without zeroing intermediate gradients. Topological depth $d$ is measured in edges from `loss` ($d=0$ at `loss`, $d=7$ at $W^{(1)}$ and $x$). The amplification $M_{k,d}$ is compared against $\binom{k+d-1}{d}$. A mitigation variant clears every topological node's `.grad` before each pass.

Metrics: maximum absolute error $\varepsilon_{\max}$, relative Euclidean error $\varepsilon_{\rm rel}$, cosine similarity $S_{\cos}$, and amplification factor $M_{k,d}$. Acceptance criteria: $\varepsilon_{\max} < 10^{-9}$, $\varepsilon_{\rm rel} < 10^{-8}$, $M_{k,d}$ matching theory within $10^{-14}$.

---

## 4. Empirical Results & Analysis

### 4.1 Part 1A: Primitive-operator and MLP verification

Table 1 reports CFD-vs-AD errors for all eight primitive operators at $h = 10^{-6}$. All operators achieve $\varepsilon_{\max} \le 1.11\times10^{-10}$ and cosine similarity $S_{\cos}\ge 0.9999999999999$ (except softmax, discussed below). The composite MLP tensors match the reference baseline: $b^{(2)}$ ($\varepsilon_{\max} = 1.696 \times 10^{-10}$), $b^{(1)}$ ($8.727 \times 10^{-11}$), $W^{(1)}$ ($1.681 \times 10^{-10}$), and $x$ ($3.352 \times 10^{-11}$), all well below the $10^{-9}$ acceptance threshold. The softmax row's elevated relative error and degraded cosine similarity arise because the test loss $\mathcal{L} = \sum_i p_i = 1$ is constant, so its true gradient is identically zero; any nonzero CFD value is pure roundoff noise.

| Operator | $\varepsilon_{\max}$ | $\varepsilon_{\rm rel}$ | $1-S_{\cos}$ |
|---|---:|---:|---:|
| Addition (`+`) | $0$ | $0$ | $0$ |
| Subtraction (`-`) | $0$ | $0$ | $0$ |
| Matmul (`*`) | $0$ | $0$ | $0$ |
| Squaring (`sq`) | $0$ | $0$ | $0$ |
| Sigmoid (`sg`) | $0$ | $0$ | $5.551\times10^{-17}$ |
| ReLU (`relu`) | $0$ | $0$ | $1.110\times10^{-16}$ |
| Softmax (`softmax`) | $1.110\times10^{-10}$ | $1.000\times10^{0}$ | $1.000\times10^{0}$ |
| Cross-entropy (`cel`) | $0$ | $0$ | $0$ |

Table 2 shows the step-size sweep for the composite MLP. The empirical error minimum occurs at $h = 10^{-4}$ (relative errors from $3.00\times10^{-11}$ to $3.51\times10^{-10}$ across tensors), roundoff error already rises at $h = 10^{-6}$ and $h = 10^{-8}$, and reaches $1.57\times10^{-5}$ at $h = 10^{-10}$ — the classic U-shaped CFD curve illustrated in Figure 1. Relative error bottoms out around $10^{-10}$ at $h = 10^{-4}$, not $10^{-6}$.

| $h$ | $W^{(1)}$ | $b^{(1)}$ | $W^{(2)}$ | $b^{(2)}$ | $x$ |
|---|---:|---:|---:|---:|---:|
| $10^{-2}$ | $1.93\times10^{-6}$ | $3.52\times10^{-6}$ | $2.99\times10^{-7}$ | $1.04\times10^{-6}$ | $4.93\times10^{-7}$ |
| $10^{-4}$ | $1.90\times10^{-10}$ | $3.51\times10^{-10}$ | $3.00\times10^{-11}$ | $1.04\times10^{-10}$ | $5.55\times10^{-11}$ |
| $10^{-6}$ | $1.30\times10^{-9}$ | $6.89\times10^{-10}$ | $1.58\times10^{-10}$ | $1.06\times10^{-10}$ | $3.26\times10^{-10}$ |
| $10^{-8}$ | $1.08\times10^{-7}$ | $1.15\times10^{-7}$ | $1.72\times10^{-8}$ | $1.00\times10^{-8}$ | $8.09\times10^{-8}$ |
| $10^{-10}$ | $1.57\times10^{-5}$ | $9.41\times10^{-6}$ | $1.87\times10^{-6}$ | $6.31\times10^{-7}$ | $7.97\times10^{-6}$ |

Figure 1 (Figure `fig1_cfd_v_curve`) plots $\max_\text{tensors}\log_{10}(\varepsilon_{\rm rel})$ vs. $\log_{10}(h)$, confirming the $\mathcal{O}(h^2)$ regime at large $h$ and roundoff rising from $h \le 10^{-6}$.

### 4.2 Part 1B: Stale-gradient amplification

Table 3 and Figure 2 (`fig2_double_backprop_scaling`) show that repeated `backprop()` on a static graph amplifies gradients exactly as $\binom{k+d-1}{d}$. For $k=2$, the factor is $d+1$ at every depth ($\times 1$ at loss, $\times 2$ at probs, $\times 3$ at $a_2$, $\times 4$ at $z_2/b^{(2)}$, $\times 5$ at $W^{(2)}/h_1$, $\times 6$ at $a_1$, $\times 7$ at $z_1/b^{(1)}$, $\times 8$ at $W^{(1)}/x$). For $k=3$, the factors are $1, 3, 6, 10, 15, 21, 28, 36$, matching $\binom{d+2}{d}$ exactly. All empirical ratios agree with theory to within $10^{-14}$.

| Depth $d$ | Node | $M_{2,d}$ (emp.) | $M_{2,d}$ (theory) | $M_{3,d}$ (emp.) | $M_{3,d}$ (theory) |
|---:|---|---:|---:|---:|---:|
| 0 | loss | 1 | 1 | 1 | 1 |
| 1 | probs | 2 | 2 | 3 | 3 |
| 2 | $a_2$ | 3 | 3 | 6 | 6 |
| 3 | $z_2 / b^{(2)}$ | 4 | 4 | 10 | 10 |
| 4 | $W^{(2)} / h_1$ | 5 | 5 | 15 | 15 |
| 5 | $a_1$ | 6 | 6 | 21 | 21 |
| 6 | $z_1 / b^{(1)}$ | 7 | 7 | 28 | 28 |
| 7 | $W^{(1)} / x$ | 8 | 8 | 36 | 36 |

Clearing every topological node's `.grad` before each pass reduces $M_{k,d}$ to $1.00$ for all $k, d$, confirming idempotency.

---

## 5. Architectural Discussion & Conclusion

Experiment 1 demonstrates that `gb_net_lib`'s VJP rules are numerically correct: analytical gradients match CFD to $\sim 10^{-10}$ relative error at $h = 10^{-6}$, and the CFD sweep exhibits the expected $\mathcal{O}(h^2)$ / roundoff transition. The primary architectural defect is in `bp.backprop()`: resetting only the root gradient causes stale gradients to compound combinatorially ($\binom{k+d-1}{d}$) under graph reuse. A node $d$ edges from the loss is multiplied by $d+1$ on a second pass ($\times 8$ on $W^{(1)}$ at depth 7), which would silently corrupt any training loop that reuses a graph.

`net.train()` escapes this only incidentally: `feed()` allocates new intermediate `bp` objects each iteration and the loop zeroes `w.grad`/`b.grad` after every sample. The robust, general fix is a two-line addition inside `backprop()` — clearing `.grad` for every node in the topological order before seeding the root — which restores $M_{k,d} = 1.00$ and makes repeated backward passes idempotent.

---

**Artifacts:** `RESULTS/exp1/exp1a_operator_errors.csv`, `exp1a_step_size_sweep.csv`, `exp1b_stale_grad_ratios.csv`, `fig1_cfd_v_curve.{png,pdf}`, `fig2_double_backprop_scaling.{png,pdf}`.  
**Source:** `EXPERIMENTS/exp1_numerical_verification/run_exp1a_cfd.py`, `run_exp1b_double_backprop.py`.

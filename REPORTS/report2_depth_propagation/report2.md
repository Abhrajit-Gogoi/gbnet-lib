# Report 2: Empirical Analysis of Gradient Propagation Dynamics Across Varying Network Depths in GB-Net

**Experiment 2 — GB-Net (`gb_net_lib`)**  
Multi-layer perceptron (`net`) gradient-flow analysis across depth

---

## Abstract

We empirically analyze how gradient signals scale across deep multi-layer perceptrons built with `gb_net_lib`'s `net` module. Single forward-backward passes are executed at initialization across network depths $L \in \{2, 5, 10, 15, 20\}$ with constant width $N = H = 8$, over seeds $\{0, 1, 42\}$, for three regimes: Sigmoid (`act='sg'`, squared loss), default ReLU (`act='relu'`, $\mathcal{U}(-0.5, 0.5)$ initialization), and He-initialized ReLU ($\mathcal{U}(-\sqrt{6/8}, \sqrt{6/8}) \approx \mathcal{U}(-0.866, 0.866)$). We measure the layer-wise weight-gradient Frobenius norm $G_W^{(\ell)} = \|\nabla_{W^{(\ell)}}\mathcal{L}\|_F$, the end-to-end attenuation ratio $\rho_L = G_W^{(1)}/G_W^{(L)}$, and the per-layer log-decay $\alpha_L = \log_{10}(\rho_L)/(L-1)$. Sigmoid gradients vanish rapidly: at $L=20$, $G_W^{(1)} = 3.10 \times 10^{-15}$ versus $G_W^{(20)} = 2.43 \times 10^{-1}$ ($\rho_{20} = 1.28 \times 10^{-14}$, $\alpha \approx -0.73$ dex/layer), leaving early layers effectively untrained. Default ReLU decays more slowly ($\rho_{20} = 1.05 \times 10^{-5}$, $\alpha \approx -0.26$ dex/layer) but still vanishes under $\mathcal{U}(-0.5, 0.5)$ initialization because its weight standard deviation ($0.289$) is below the variance-preserving He value ($0.500$). He-initialized ReLU preserves gradient magnitude ($\rho_{20} \approx 1.18$, $\alpha \approx +0.004$ dex/layer), confirming that variance-scaled initialization mitigates vanishing gradients in this engine.

---

## 1. Introduction & Architectural Overview

GB-Net's `net` class constructs an MLP from `bp` autograd nodes using column vectors. Each layer computes $z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}$ followed by an activation $\phi^{(\ell)}$, with $W^{(\ell)} \in \mathbb{R}^{n_\ell \times n_{\ell-1}}$ and $b^{(\ell)} \in \mathbb{R}^{n_\ell \times 1}$. By default, weights and biases are drawn i.i.d. from $\mathcal{U}(-0.5, 0.5)$. Backpropagation through $L$ layers applies the chain rule recursively, so the gradient reaching the first layer is a product of $L-1$ layer Jacobians. Experiment 2 measures how this product scales with depth for different activations and initializations, using a single forward-backward pass per configuration (no training loops), which isolates the structural Jacobian properties at initialization.

---

## 2. Mathematical Formulation

For an $L$-layer MLP with loss $\mathcal{L}$, the gradient of the loss with respect to the first-layer pre-activation is
\[
\frac{\partial\mathcal{L}}{\partial h^{(1)}} =
\left(\prod_{\ell=2}^{L} \left(W^{(\ell)}\right)^\top
\operatorname{diag}\!\left(\phi'\!\left(z^{(\ell)}\right)\right)\right)
\frac{\partial\mathcal{L}}{\partial h^{(L)}}.
\]
The first-layer weight gradient is
\[
\nabla_{W^{(1)}}\mathcal{L} =
\frac{\partial\mathcal{L}}{\partial z^{(1)}} (h^{(0)})^\top,
\]
so its Frobenius norm $G_W^{(1)} = \|\nabla_{W^{(1)}}\mathcal{L}\|_F$ inherits the multiplicative attenuation of the Jacobian chain. Gradient stability is therefore governed by (i) the activation derivative $\phi'$ and (ii) the weight variance $\operatorname{Var}(W_{ij})$.

- **Sigmoid:** $\sigma'(z) = \sigma(z)(1-\sigma(z)) \in (0, 0.25]$, so $|\sigma'| \le 0.25$ creates a severe per-layer contraction.
- **ReLU:** $\operatorname{ReLU}'(z) \in \{0,1\}$, with the active fraction determining signal retention.
- **Variance preservation (He):** For ReLU, preserving activation variance across layers requires $\operatorname{Var}(W) = 2/n_{\ell-1}$. With width 8, $\operatorname{std}(W) = \sqrt{2/8} = 0.500$, whereas $\mathcal{U}(-0.5, 0.5)$ has $\operatorname{std}(W) = 1/\sqrt{12} \approx 0.289$ --- below the He value, causing mild contraction even for ReLU.

We define the end-to-end attenuation ratio
\[
\rho_L = \frac{G_W^{(1)}}{G_W^{(L)} + 10^{-30}},
\qquad
\alpha_L = \frac{\log_{10}(\rho_L)}{L-1}
\]
(orders of magnitude lost per layer).

---

## 3. Experimental Methodology & Setup

Single-threaded CPU, NumPy `float64`, seeds $\{0, 1, 42\}$. Depths $L \in \{2, 5, 10, 15, 20\}$, width $N = H = 8$ (`layers=[8]*L`). Input $x \sim \mathcal{U}(-0.5, 0.5)^{8\times 1}$, target $y = 0.5\,\mathbf{1}_{8\times 1}$. One forward pass and one `loss.backprop()` per $(L, \text{regime}, \text{seed})$ (45 micro-passes total). Loss is squared error $(o - y)^2$.

**Regimes:**
1. **Sigmoid:** `act='sg'`, default $\mathcal{U}(-0.5, 0.5)$ init.
2. **ReLU (default):** `act='relu'`, default $\mathcal{U}(-0.5, 0.5)$ init.
3. **ReLU (He):** `act='relu'`, weights rescaled to $\mathcal{U}(-\sqrt{6/8}, \sqrt{6/8})$, biases zeroed.

Metrics per layer $\ell$: $G_W^{(\ell)}$ (Frobenius), $G_b^{(\ell)}$ (Euclidean). Network summary: $\rho_L$, $\alpha_L$. Aggregated across 3 seeds (mean reported; seed-42 values match the verified baseline in `expo.md` §2.3).

---

## 4. Empirical Results & Analysis

### 4.1 Attenuation across depth

Table 1 reports $\rho_L$ and $\alpha_L$ (seed 42). Sigmoid loses roughly 0.73--0.79 orders of magnitude per layer: by $L=20$, $G_W^{(1)} = 3.10 \times 10^{-15}$ is 14 orders below $G_W^{(20)} = 2.43 \times 10^{-1}$ ($\rho_{20} = 1.28 \times 10^{-14}$). Default ReLU decays more slowly but still vanishes ($\rho_{20} = 1.05 \times 10^{-5}$, $\alpha \approx -0.26$ dex/layer). He-initialized ReLU preserves magnitude ($\rho_{20} \approx 1.18$, $\alpha \approx +0.004$ dex/layer).

| Activation | $L$ | $G_W^{(1)}$ | $G_W^{(L)}$ | $\rho_L$ | $\alpha_L$ (dex/layer) |
|---|---:|---:|---:|---:|---:|
| Sigmoid | 2 | $2.63\times10^{-2}$ | $1.57\times10^{-1}$ | $1.68\times10^{-1}$ | $-0.78$ |
| Sigmoid | 5 | $1.49\times10^{-4}$ | $2.21\times10^{-1}$ | $6.75\times10^{-4}$ | $-0.79$ |
| Sigmoid | 10 | $3.43\times10^{-8}$ | $1.70\times10^{-1}$ | $2.02\times10^{-7}$ | $-0.74$ |
| Sigmoid | 20 | $3.10\times10^{-15}$ | $2.43\times10^{-1}$ | $1.28\times10^{-14}$ | $-0.73$ |
| ReLU (default) | 2 | $4.58\times10^{-1}$ | $7.08\times10^{-1}$ | $6.47\times10^{-1}$ | $-0.19$ |
| ReLU (default) | 5 | $2.69\times10^{-2}$ | $4.29\times10^{-1}$ | $6.27\times10^{-2}$ | $-0.30$ |
| ReLU (default) | 10 | $5.01\times10^{-3}$ | $4.80\times10^{-1}$ | $1.04\times10^{-2}$ | $-0.22$ |
| ReLU (default) | 20 | $7.69\times10^{-6}$ | $7.32\times10^{-1}$ | $1.05\times10^{-5}$ | $-0.26$ |
| ReLU (He) | 2 | $1.02\times10^{0}$ | $1.16\times10^{0}$ | $8.81\times10^{-1}$ | $-0.06$ |
| ReLU (He) | 10 | $2.77\times10^{-1}$ | $3.22\times10^{-1}$ | $8.61\times10^{-1}$ | $-0.01$ |
| ReLU (He) | 20 | $4.07\times10^{-2}$ | $3.45\times10^{-2}$ | $1.18\times10^{0}$ | $+0.00$ |

Figure 3 (`fig3_depth_decay_comparison`, Panel A) plots $\log_{10}\rho_L$ vs. $L$ for the three regimes; Panel B shows the layer-by-layer profile $\log_{10}G_W^{(\ell)}$ at $L=20$.

### 4.2 Layer-wise profile

At $L=20$, the Sigmoid profile shows steep monotonic decay from the output layer back to the input (roughly linear on the log scale, slope $\approx -0.73$ dex/layer), so the first layer's gradient is numerically zero for practical learning rates. Default ReLU shows a shallower decay. He-initialized ReLU shows an essentially flat profile ($\alpha \approx 0$), confirming that variance-scaled initialization counteracts the contraction of the Jacobian chain.

---

## 5. Architectural Discussion & Conclusion

Experiment 2 confirms the classical vanishing-gradient mechanism in GB-Net: backpropagating through $L$ layers multiplies $L-1$ matrices $(W^{(\ell)})^\top \operatorname{diag}(\phi'(z^{(\ell)}))$, so gradient magnitude at the first layer is exponentially sensitive to the activation derivative and weight variance. Sigmoid's bounded derivative ($\le 0.25$) causes rapid vanishing ($\alpha \approx -0.75$ dex/layer), rendering $W^{(1)}$ untrainable at depth by $L=20$ since `w.data -= lr * w.grad` applies a numerically zero update. ReLU mitigates but does not eliminate the problem under `gb_net_lib`'s fixed $\mathcal{U}(-0.5, 0.5)$ initialization: with width 8, the He-preserving standard deviation is $0.500$ versus $0.289$ actually used, producing mild per-layer contraction ($\alpha \approx -0.26$). He-scaled uniform initialization ($\operatorname{std}=0.500$, zero biases) restores gradient preservation ($\rho_{20} \approx 1.18$).

Concrete recommendations: (1) adopt He/Xavier variance-scaled initialization for ReLU/Sigmoid networks; (2) consider bias initialization and activation scaling; (3) for very deep networks in this engine, gradient checkpointing or residual connections would be required to preserve signal to early layers. The single-pass, no-training design keeps runtime well under the sub-second budget while isolating the structural Jacobian behaviour at initialization.

---

**Artifacts:** `RESULTS/exp2/exp2_depth_grad_norms.csv`, `fig3_depth_decay_comparison.{png,pdf}`.  
**Source:** `EXPERIMENTS/exp2_depth_propagation/run_exp2_depth_sweep.py`.

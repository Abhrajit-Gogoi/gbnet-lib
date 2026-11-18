"""Experiment 2: Gradient propagation dynamics across varying network depths.

Reproduces protocol.md Section 3:
- Single forward + backward pass per (L, regime, seed).
- Depths L in {2, 5, 10, 15, 20}, width N=H=8, seeds [0, 1, 42].
- Regimes: Sigmoid (sq loss), default ReLU (sq loss), He-init ReLU (sq loss).
- Metrics: G_W^(l) (Frobenius), G_b^(l) (Euclidean), attenuation rho_L, per-layer decay alpha.
- Writes exp2_depth_grad_norms.csv + fig3_depth_decay_comparison.png (Panel A + Panel B).

Sub-second CPU budget. No training loops.
"""
import csv
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "src")
import gb_net_lib
bp = gb_net_lib.bp
net = gb_net_lib.net

RESULTS = "RESULTS/exp2"
DEPTHS = [2, 5, 10, 15, 20]
N = 8
H = 8
SEEDS = [0, 1, 42]
TARGET = bp(0.5 * np.ones((N, 1)))


def he_scale():
    return math.sqrt(6.0 / H) / 0.5  # ~1.73205 to rescale U(-0.5,0.5) -> U(-0.866,0.866)


def run_once(seed, L, regime):
    np.random.seed(seed)
    act = "sg" if regime == "sigmoid" else "relu"
    m = net(N=N, layers=[H] * L, act=act)
    struc = m.layer_init()
    if regime == "he_relu":
        for w in struc:
            w.data *= he_scale()
        m.biases = [bp(np.zeros((w.data.shape[0], 1))) for w in struc]
    x = bp(np.random.uniform(-0.5, 0.5, (N, 1)))
    m.input = x
    out = m.feed(struc, m.biases)[-1]
    loss = (out - TARGET).sq()
    loss.backprop()
    return struc, m.biases, loss


def metrics(struc, biases, L):
    gW = [float(np.linalg.norm(struc[l - 1].grad)) for l in range(1, L + 1)]
    gb = [float(np.linalg.norm(biases[l - 1].grad)) for l in range(1, L + 1)]
    return gW, gb


def main():
    t0 = time.time()
    rows = []
    # aggregate for panel A: mean log10(rho) per (regime, L)
    panelA = {"sigmoid": [], "relu": [], "he_relu": []}
    # panel B: layer profiles at L=20, seed 42
    panelB = {}

    for regime in ("sigmoid", "relu", "he_relu"):
        for L in DEPTHS:
            rhos = []
            for s in SEEDS:
                struc, biases, loss = run_once(s, L, regime)
                gW, gb = metrics(struc, biases, L)
                rho = gW[0] / (gW[-1] + 1e-30)
                rhos.append(rho)
                for l in range(1, L + 1):
                    rows.append({
                        "regime": regime, "seed": s, "depth": L, "layer": l,
                        "G_W": "%.6e" % gW[l - 1], "G_b": "%.6e" % gb[l - 1],
                    })
            mean_rho = float(np.mean(rhos))
            alpha = math.log10(mean_rho) / (L - 1) if L > 1 else float("nan")
            panelA[regime].append((L, mean_rho, alpha))
            if L == 20:
                # store layer profile from seed 42 run
                struc, biases, _ = run_once(42, 20, regime)
                gW, _ = metrics(struc, biases, 20)
                panelB[regime] = [math.log10(g) if g > 0 else -300.0 for g in gW]

    with open(f"{RESULTS}/exp2_depth_grad_norms.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["regime", "seed", "depth", "layer", "G_W", "G_b"])
        w.writeheader(); w.writerows(rows)
    print("Wrote exp2_depth_grad_norms.csv with", len(rows), "rows")

    # ---- fig3 Panel A: log10(rho_L) vs depth L ----
    plt.rcParams["font.family"] = "serif"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"sigmoid": "#1f77b4", "relu": "#d62728", "he_relu": "#2ca02c"}
    labels = {"sigmoid": "Sigmoid", "relu": "ReLU (default)", "he_relu": "ReLU (He init)"}
    for regime in ("sigmoid", "relu", "he_relu"):
        Ls = [t[0] for t in panelA[regime]]
        logrho = [math.log10(t[1]) for t in panelA[regime]]
        axA.plot(Ls, logrho, "o-", color=colors[regime], label=labels[regime])
    axA.set_xlabel("Network depth $L$")
    axA.set_ylabel(r"$\log_{10}(\rho_L) = \log_{10}(G_W^{(1)}/G_W^{(L)})$")
    axA.set_xticks(DEPTHS)
    axA.grid(True, alpha=0.3)
    axA.legend()

    # ---- fig3 Panel B: layer profile log10 G_W^(l) at L=20 ----
    layers = list(range(1, 21))
    for regime in ("sigmoid", "relu", "he_relu"):
        axB.plot(layers, panelB[regime], "o-", color=colors[regime],
                 label=labels[regime], markersize=3)
    axB.set_xlabel(r"Layer $\ell$")
    axB.set_ylabel(r"$\log_{10}\|\nabla_{W^{(\ell)}}\mathcal{L}\|_F$")
    axB.set_xticks([1, 5, 10, 15, 20])
    axB.grid(True, alpha=0.3)
    axB.legend()

    fig.suptitle("Gradient propagation across network depth ($N=H=8$)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{RESULTS}/fig3_depth_decay_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig3_depth_decay_comparison.png")

    # summary print
    print("\n=== Attenuation ratio rho_L and per-layer decay alpha ===")
    for regime in ("sigmoid", "relu", "he_relu"):
        for L, rho, alpha in panelA[regime]:
            print(f"{regime:10s} L={L:2d}: rho={rho:.4e}  alpha={alpha:+.3f} dex/layer")

    print(f"\nExperiment 2 complete in {time.time() - t0:.3f} s")


if __name__ == "__main__":
    main()

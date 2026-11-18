"""Regenerate all figures as PDF (PNG versions already produced by experiment scripts)."""
import csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

# ---- fig1: from exp1a_step_size_sweep.csv ----
rows = list(csv.DictReader(open("RESULTS/exp1/exp1a_step_size_sweep.csv")))
hs = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
max_rel = []
for h in hs:
    vals = [float(r["eps_rel"]) for r in rows if float(r["h"]) == h]
    max_rel.append(max(vals))
fig, ax = plt.subplots(figsize=(6, 4))
ax.loglog(hs, max_rel, "o-", color="#1f77b4")
ax.set_xlabel(r"Step size $h$"); ax.set_ylabel(r"Relative error $\varepsilon_{\rm rel}$")
ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("RESULTS/exp1/fig1_cfd_v_curve.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- fig2: from exp1b_stale_grad_ratios.csv ----
rows = list(csv.DictReader(open("RESULTS/exp1/exp1b_stale_grad_ratios.csv")))
lookup = {}
for r in rows:
    lookup.setdefault((int(r["pass"]), int(r["depth"])), float(r["M_kd"]))
max_depth = max(d for (p, d) in lookup if p == 1)
depths = list(range(max_depth + 1))
m2 = [lookup[(2, d)] for d in depths]
m3 = [lookup[(3, d)] for d in depths]
th2 = [math.comb(d + 1, d) for d in depths]
th3 = [math.comb(d + 2, d) for d in depths]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(depths, m2, "o-", color="#1f77b4", label="M_{2,d} (empirical)")
ax.plot(depths, th2, "s--", color="#d62728", label="binom(d+1,d)=d+1 (theory)")
ax.plot(depths, m3, "o-", color="#2ca02c", label="M_{3,d} (empirical)")
ax.plot(depths, th3, "s--", color="#9467bd", label="binom(d+2,d) (theory)")
ax.set_xlabel(r"Topological depth $d$ (edges from loss)")
ax.set_ylabel(r"Amplification factor $M_{k,d}$")
ax.set_xticks(depths); ax.grid(True, alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig("RESULTS/exp1/fig2_double_backprop_scaling.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- fig3: from exp2_depth_grad_norms.csv ----
rows = list(csv.DictReader(open("RESULTS/exp2/exp2_depth_grad_norms.csv")))
regimes = ["sigmoid", "relu", "he_relu"]
colors = {"sigmoid": "#1f77b4", "relu": "#d62728", "he_relu": "#2ca02c"}
labels = {"sigmoid": "Sigmoid", "relu": "ReLU (default)", "he_relu": "ReLU (He init)"}
# Panel A: log10(rho) vs L
panelA = {rg: [] for rg in regimes}
for rg in regimes:
    for L in [2, 5, 10, 15, 20]:
        rhos = [float(r["G_W"]) / (float(r["G_W"]) + 1e-30) for r in rows if r["regime"] == rg and int(r["depth"]) == L and int(r["layer"]) == 1]
        # rho needs G_W^(1) and G_W^(L) same seed; recompute properly below
        panelA[rg].append(L)
# proper rho per seed
import collections
seed_data = collections.defaultdict(dict)
for r in rows:
    key = (r["regime"], int(r["depth"]), int(r["seed"]))
    seed_data[key][int(r["layer"])] = float(r["G_W"])
def rho(rg, L, s):
    d = seed_data[(rg, L, s)]
    return d[1] / (d[L] + 1e-30)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4))
for rg in regimes:
    Ls = [2, 5, 10, 15, 20]
    logrho = [math.log10(rho(rg, L, 42)) for L in Ls]
    axA.plot(Ls, logrho, "o-", color=colors[rg], label=labels[rg])
axA.set_xlabel("Network depth $L$")
axA.set_ylabel(r"$\log_{10}(\rho_L)$")
axA.set_xticks([2, 5, 10, 15, 20]); axA.grid(True, alpha=0.3); axA.legend()
# Panel B: layer profile at L=20 seed 42
for rg in regimes:
    d = seed_data[(rg, 20, 42)]
    layers = list(range(1, 21))
    axB.plot(layers, [math.log10(d[l]) if d[l] > 0 else -300.0 for l in layers],
             "o-", color=colors[rg], label=labels[rg], markersize=3)
axB.set_xlabel(r"Layer $\ell$")
axB.set_ylabel(r"$\log_{10}\|\nabla_{W^{(\ell)}}\mathcal{L}\|_F$")
axB.set_xticks([1, 5, 10, 15, 20]); axB.grid(True, alpha=0.3); axB.legend()
fig.suptitle("Gradient propagation across network depth ($N=H=8$)")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("RESULTS/exp2/fig3_depth_decay_comparison.pdf", dpi=300, bbox_inches="tight")
plt.close(fig)
print("PDF figures written")

"""Experiment 1B: Repeated backprop() on a static graph -> stale gradient accumulation.

Reproduces protocol.md Section 2.2 Phase 1B:
- Single forward pass of 2-layer MLP (3->4->3), bind every intermediate node.
- True topological depth d computed via build_topo (position from loss, d=0 at loss).
- Call loss.backprop() for k in {1,2,3} on the SAME graph without zeroing intermediates.
- Record amplification M_{k,d} = ||v_d.grad^(k)|| / ||v_d.grad^(1)||.
- Compare to theory binom(k+d-1, d) (=> d+1 for k=2).
- Mitigation: zero all topo grads before backprop -> M=1.00.
- Writes exp1b_stale_grad_ratios.csv + fig2_double_backprop_scaling.png

Seed: np.random.seed(42). Sub-second CPU budget.
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

np.random.seed(42)
RESULTS = "RESULTS/exp1"
t0 = time.time()


def binom(n, k):
    return math.comb(n, k)


# ---------------------------------------------------------------------------
# Single forward pass, binding every intermediate node
# ---------------------------------------------------------------------------
X = bp(np.array([[0.4], [-0.3], [0.8]]))
Y = bp(np.array([[0.0], [1.0], [0.0]]))
W1 = bp(np.random.rand(4, 3) - 0.5)
B1 = bp(np.random.rand(4, 1) - 0.5)
W2 = bp(np.random.rand(3, 4) - 0.5)
B2 = bp(np.random.rand(3, 1) - 0.5)

z1 = W1 * X + B1     # z1 uses W1, X, B1
a1 = z1.sg()         # a1 uses z1
z2 = W2 * a1 + B2    # z2 uses W2, a1, B2
a2 = z2              # a2 = z2 (output logits)
probs = a2.softmax() # probs uses a2
loss = probs.cel(Y)  # loss uses probs, Y

# All nodes in the graph
all_nodes = [loss, probs, a2, z2, B2, W2, a1, z1, B1, W1, X]

# True topological depth = BFS graph distance (edges) from loss following .parent links
from collections import deque
true_depth = {id(loss): 0}
graph_nodes = []
dq = deque([loss])
while dq:
    cur = dq.popleft()
    graph_nodes.append(cur)
    for p in cur.parent:
        if id(p) not in true_depth:
            true_depth[id(p)] = true_depth[id(cur)] + 1
            dq.append(p)
all_nodes = graph_nodes  # every node reachable from loss


def grad_norm(node):
    return float(np.linalg.norm(node.grad))


# Run k in {1,2,3} backprops on the SAME graph (no zeroing) -> stale accumulation
results = []
pass1_norms = {}
for k in (1, 2, 3):
    loss.backprop()
    for node in all_nodes:
        d = true_depth[id(node)]
        g = grad_norm(node)
        if k == 1:
            pass1_norms[id(node)] = g
            M = 1.0
            theory = 1.0
        else:
            base = pass1_norms[id(node)]
            M = (g / base) if base > 1e-300 else float("nan")
            theory = binom(k + d - 1, d)
        results.append({
            "pass": k, "node": node.__class__.__name__ + "_" + str(id(node))[-4:],
            "depth": d, "norm": f"{g:.6e}", "M_kd": f"{M:.6f}",
            "theory": f"{theory:.6f}", "match": str(abs(M - theory) < 1e-6),
        })

with open(f"{RESULTS}/exp1b_stale_grad_ratios.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pass", "node", "depth", "norm", "M_kd", "theory", "match"])
    w.writeheader(); w.writerows(results)
print("Wrote exp1b_stale_grad_ratios.csv with", len(results), "rows")

# Build lookup: (pass, depth) -> M_kd (use first node at each depth)
lookup = {}
for r in results:
    lookup.setdefault((int(r["pass"]), int(r["depth"])), float(r["M_kd"]))

# ---- fig2: M_{k,d} vs depth for k=2,3 vs theory ----
max_depth = max(d for (p, d) in lookup if p == 1)
depths = list(range(max_depth + 1))
m2 = [lookup[(2, d)] for d in depths]
m3 = [lookup[(3, d)] for d in depths]
th2 = [binom(2 + d - 1, d) for d in depths]
th3 = [binom(3 + d - 1, d) for d in depths]

plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(depths, m2, "o-", color="#1f77b4", label="M_{2,d} (empirical)")
ax.plot(depths, th2, "s--", color="#d62728", label="binom(d+1,d)=d+1 (theory)")
ax.plot(depths, m3, "o-", color="#2ca02c", label="M_{3,d} (empirical)")
ax.plot(depths, th3, "s--", color="#9467bd", label="binom(d+2,d) (theory)")
ax.set_xlabel(r"Topological depth $d$ (edges from loss)")
ax.set_ylabel(r"Amplification factor $M_{k,d}$")
ax.set_xticks(depths)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(f"{RESULTS}/fig2_double_backprop_scaling.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Wrote fig2_double_backprop_scaling.png")

# ---- mitigation: zero all topo grads before each backprop -> M=1.00 ----
mitigation = []
for k in (1, 2, 3):
    for node in graph_nodes:
        node.grad = np.zeros_like(node.data)
    loss.backprop()
    for node in all_nodes:
        d = true_depth[id(node)]
        g = grad_norm(node)
        base = pass1_norms[id(node)]
        M = (g / base) if base > 1e-300 else float("nan")
        mitigation.append({"pass": k, "depth": d, "M_kd": f"{M:.6f}"})

mit_ok = all(float(r["M_kd"]) < 1.000001 for r in mitigation)
print("Mitigation M_kd all ~1.0:", mit_ok)
for r in mitigation:
    print(f"  pass={r['pass']} depth={r['depth']} M={r['M_kd']}")

print(f"\nExperiment 1B complete in {time.time() - t0:.3f} s")

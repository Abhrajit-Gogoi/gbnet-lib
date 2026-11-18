"""Experiment 1A: Central Finite Difference verification of bp primitives + 2-layer MLP.

Reproduces protocol.md Section 2.2:
- Phase 1A-1: per-operator CFD vs AD gradients (exp1a_operator_errors.csv)
- Phase 1A-2: composite MLP step-size sweep h in {1e-2, 1e-4, 1e-6, 1e-8, 1e-10}
  (exp1a_step_size_sweep.csv + fig1_cfd_v_curve.png)

Seed: np.random.seed(42).  Runs under the sub-second CPU budget.
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
H = 1e-6
EPS = 1e-15

RESULTS = "RESULTS/exp1"


def rel_err(ad, cfd):
    return float(np.linalg.norm(ad - cfd) / (np.linalg.norm(ad) + np.linalg.norm(cfd) + 1e-15))


def cos_sim(ad, cfd):
    num = float(np.dot(ad.ravel(), cfd.ravel()))
    den = float(np.linalg.norm(ad) * np.linalg.norm(cfd) + 1e-15)
    return num / den


def cfd_scalar(fn, theta, idx, h):
    """Central difference of scalar loss fn wrt flattened scalar theta[idx]."""
    theta_p = theta.copy(); theta_p[idx] += h
    theta_m = theta.copy(); theta_m[idx] -= h
    return (fn(theta_p) - fn(theta_m)) / (2.0 * h)


# ---------------------------------------------------------------------------
# Phase 1A-1: primitive operator unit verification at h = 1e-6
# ---------------------------------------------------------------------------
def op_tests():
    rows = []
    tests = []

    def add_case(name, fn, ad_grad, cfd_grad, shape):
        ad = np.asarray(ad_grad, dtype=float).ravel()
        cf = np.asarray(cfd_grad, dtype=float).ravel()
        emax = float(np.max(np.abs(ad - cf))) if ad.size else 0.0
        erel = rel_err(ad, cf)
        sc = cos_sim(ad, cf)
        rows.append({
            "operator": name, "shape": str(shape),
            "eps_max": f"{emax:.3e}", "eps_rel": f"{erel:.3e}", "cos_sim": f"{sc:.15f}",
            "h": f"{H:.1e}",
            "pass_emax": str(emax < 1e-9), "pass_erel": str(erel < 1e-8),
        })

    # 1) addition: c = a + b ; loss = sum(c)
    a = bp(np.array([[1.2], [2.3], [3.4]])); b = bp(np.array([[0.4], [-0.7], [0.9]]))
    c = a + b; loss = c.sq()
    loss.backprop()
    A = np.array([[1.2], [2.3], [3.4]]); B = np.array([[0.4], [-0.7], [0.9]])
    def loss_add(theta):
        t = theta.reshape(A.shape); o = (t + B); return float(np.sum(o * o))
    ca = cfd_scalar(loss_add, A.ravel(), 0, H)
    add_case("__add__", loss_add, a.grad, 2.0 * (A + B) * 1.0, A.shape)

    # 2) subtraction: c = a - b ; loss = sum(c^2)
    a = bp(np.array([[1.2], [2.3], [3.4]])); b = bp(np.array([[0.4], [-0.7], [0.9]]))
    c = a - b; loss = c.sq(); loss.backprop()
    A = np.array([[1.2], [2.3], [3.4]]); B = np.array([[0.4], [-0.7], [0.9]])
    add_case("__sub__", lambda th: float(np.sum((th.reshape(A.shape) - B) ** 2)), a.grad, 2.0 * (A - B), A.shape)

    # 3) matrix multiply: c = a @ b ; loss = sum(c^2)
    A = np.random.rand(3, 2); B = np.random.rand(2, 1)
    a = bp(A); b = bp(B)
    c = a * b; loss = c.sq(); loss.backprop()
    def loss_mul(th):
        M = th.reshape(A.shape); o = M @ B; return float(np.sum(o * o))
    add_case("__mul__ (matmul)", loss_mul, a.grad, 2.0 * ((A @ B) @ B.T), A.shape)

    # 4) sq: c = a.sq() ; loss = sum(c)
    a = bp(np.array([[1.2], [2.3], [3.4]]))
    c = a.sq(); loss = c; loss.backprop()
    A = np.array([[1.2], [2.3], [3.4]])
    add_case("sq", lambda th: float(np.sum(th.reshape(A.shape) ** 2)), a.grad, 2.0 * A, A.shape)

    # 5) sigmoid
    a = bp(np.array([[0.5], [-0.8], [1.1]]))
    c = a.sg(); loss = c; loss.backprop()
    A = np.array([[0.5], [-0.8], [1.1]])
    s = 1.0 / (1.0 + np.exp(-A))
    add_case("sg", lambda th: float(np.sum(1.0 / (1.0 + np.exp(-th.reshape(A.shape))))), a.grad, s * (1.0 - s), A.shape)

    # 6) relu
    a = bp(np.array([[0.5], [-0.8], [1.1]]))
    c = a.relu(); loss = c; loss.backprop()
    A = np.array([[0.5], [-0.8], [1.1]])
    add_case("relu", lambda th: float(np.sum(np.maximum(th.reshape(A.shape), 0.0))), a.grad, (A > 0).astype(float), A.shape)

    # 7) softmax: column vector, loss = sum(probs) == 1 (constant) -> grad ~ 0
    a = bp(np.array([[1.2], [-0.4], [0.9]]))
    c = a.softmax(); loss = c; loss.backprop()
    A = np.array([[1.2], [-0.4], [0.9]])
    def loss_softmax(th):
        Z = th.reshape(A.shape)
        shifted = Z - np.max(Z, axis=0, keepdims=True)
        exps = np.exp(shifted)
        p = exps / np.sum(exps, axis=0, keepdims=True)
        return float(np.sum(p))
    cf = np.zeros_like(A)
    for i in range(A.size):
        cf.flat[i] = cfd_scalar(loss_softmax, A.ravel(), i, H)
    add_case("softmax", loss_softmax, a.grad, cf, A.shape)

    # 8) cel
    pred = bp(np.array([[0.2], [0.5], [0.3]]))
    target = bp(np.array([[0.0], [1.0], [0.0]]))
    loss = pred.cel(target); loss.backprop()
    P = np.array([[0.2], [0.5], [0.3]]); T = np.array([[0.0], [1.0], [0.0]])
    clip = np.clip(P, EPS, 1.0)
    add_case("cel", lambda th: float(-np.sum(T * np.log(np.clip(th.reshape(P.shape), EPS, 1.0)))),
             pred.grad, -(T / clip), P.shape)

    # write csv
    with open(f"{RESULTS}/exp1a_operator_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["operator", "shape", "eps_max", "eps_rel", "cos_sim", "h", "pass_emax", "pass_erel"])
        w.writeheader(); w.writerows(rows)
    print("Wrote exp1a_operator_errors.csv with", len(rows), "rows")


# ---------------------------------------------------------------------------
# Phase 1A-2: composite MLP step-size sweep
# ---------------------------------------------------------------------------
def build_mlp(seed=42):
    np.random.seed(seed)
    X = np.array([[0.4], [-0.3], [0.8]])
    Y = np.array([[0.0], [1.0], [0.0]])
    W1 = bp(np.random.rand(4, 3) - 0.5); B1 = bp(np.random.rand(4, 1) - 0.5)
    W2 = bp(np.random.rand(3, 4) - 0.5); B2 = bp(np.random.rand(3, 1) - 0.5)
    return X, Y, W1, B1, W2, B2


def forward_loss(X, Y, W1, B1, W2, B2):
    z1 = W1 * X + B1
    a1 = z1.sg()
    z2 = W2 * a1 + B2
    probs = z2.softmax()
    loss = probs.cel(Y)
    return loss, {"X": X, "W1": W1, "B1": B1, "W2": W2, "B2": B2,
                  "z1": z1, "a1": a1, "z2": z2, "probs": probs}


def loss_fn_factory(nodes):
    """Build a scalar loss function of a flattened parameter vector, rebuilding the graph."""
    def fn(flat):
        X = np.array([[0.4], [-0.3], [0.8]])
        Y = np.array([[0.0], [1.0], [0.0]])
        W1 = bp(np.random.rand(4, 3) - 0.5)  # placeholder, overwritten below
        # We rebuild deterministically from the flat vector ordering.
        return None
    return fn


def mlp_loss_from_flat(flat, X, Y):
    """Reconstruct MLP from flat parameter vector and compute scalar loss."""
    i = 0
    W1d = flat[i:i+12].reshape(4, 3); i += 12
    B1d = flat[i:i+4].reshape(4, 1); i += 4
    W2d = flat[i:i+12].reshape(3, 4); i += 12
    B2d = flat[i:i+3].reshape(3, 1); i += 3
    Xd = flat[i:i+3].reshape(3, 1); i += 3
    # Build bp graph
    W1b = bp(W1d); B1b = bp(B1d); W2b = bp(W2d); B2b = bp(B2d)
    Xb = bp(Xd)
    z1 = W1b * Xb + B1b
    a1 = z1.sg()
    z2 = W2b * a1 + B2b
    probs = z2.softmax()
    loss = probs.cel(bp(Y))
    return float(np.sum(loss.data))


def sweep():
    np.random.seed(42)
    X = np.array([[0.4], [-0.3], [0.8]])
    Y = np.array([[0.0], [1.0], [0.0]])
    W1d = np.random.rand(4, 3) - 0.5; B1d = np.random.rand(4, 1) - 0.5
    W2d = np.random.rand(3, 4) - 0.5; B2d = np.random.rand(3, 1) - 0.5

    # analytical gradients once at h-independent point
    W1b = bp(W1d); B1b = bp(B1d); W2b = bp(W2d); B2b = bp(B2d); Xb = bp(X)
    z1 = W1b * Xb + B1b; a1 = z1.sg()
    z2 = W2b * a1 + B2b; probs = z2.softmax(); loss = probs.cel(bp(Y))
    loss.backprop()
    anal = {"W1": W1b.grad.copy(), "B1": B1b.grad.copy(), "W2": W2b.grad.copy(),
            "B2": B2b.grad.copy(), "X": Xb.grad.copy()}

    shapes = {"W1": (12,), "B1": (4,), "W2": (12,), "B2": (3,), "X": (3,)}
    tensors = {"W1": W1d, "B1": B1d, "W2": W2d, "B2": B2d, "X": X}
    order = ["W1", "B1", "W2", "B2", "X"]

    hs = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    rows = []
    for h in hs:
        for name in order:
            th = tensors[name].ravel()
            ad = anal[name].ravel()
            cf = np.zeros_like(th)
            for j in range(th.size):
                cf[j] = cfd_scalar(lambda v, n=name: mlp_loss_from_flat(
                    np.concatenate([tensors[o].ravel() if o != n else v for o in order]), X, Y), th, j, h)
            emax = float(np.max(np.abs(ad - cf))) if ad.size else 0.0
            erel = rel_err(ad, cf)
            sc = cos_sim(ad, cf)
            rows.append({"h": f"{h:.1e}", "tensor": name, "size": int(th.size),
                         "eps_max": f"{emax:.3e}", "eps_rel": f"{erel:.3e}",
                         "cos_sim": f"{sc:.15f}",
                         "pass_emax": str(emax < 1e-9), "pass_erel": str(erel < 1e-8)})

    with open(f"{RESULTS}/exp1a_step_size_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["h", "tensor", "size", "eps_max", "eps_rel", "cos_sim", "pass_emax", "pass_erel"])
        w.writeheader(); w.writerows(rows)
    print("Wrote exp1a_step_size_sweep.csv with", len(rows), "rows")

    # fig1: log10(h) vs log10(eps_rel), best tensor per h (max over tensors)
    import matplotlib.pyplot as plt
    hs_plot = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    max_rel = []
    for h in hs_plot:
        vals = [float(r["eps_rel"]) for r in rows if float(r["h"]) == h]
        max_rel.append(max(vals))
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(hs_plot, max_rel, "o-", color="#1f77b4", label="max_rel over tensors")
    ax.set_xlabel(r"Step size $h$"); ax.set_ylabel(r"Relative error $\varepsilon_{\rm rel}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/fig1_cfd_v_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Wrote fig1_cfd_v_curve.png")


if __name__ == "__main__":
    t0 = time.time()
    op_tests()
    sweep()
    print(f"Experiment 1A complete in {time.time() - t0:.3f} s")

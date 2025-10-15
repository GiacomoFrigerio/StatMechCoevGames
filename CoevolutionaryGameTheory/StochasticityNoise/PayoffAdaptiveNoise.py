#@title Perc (2006) spatial PD with additive payoff noise (grid)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def step_once(strat, r, sigma, K, rng):
    n = strat.shape[0]
    shifts = [(0,1),(0,-1),(1,0),(-1,0)]
    S = np.zeros_like(strat, dtype=float)

    # accumulate focal player's payoff vs each neighbor, with independent noise per pair & player
    for dy,dx in shifts:
        sj = np.roll(np.roll(strat, dy, axis=0), dx, axis=1)
        si = strat

        # Correct PD mapping: R=1 (CC), T=1+r (D vs C), S=-r (C vs D), P=0 (DD)
        base  = (si & sj).astype(float) * 1.0
        base += ( (1-si) * sj ).astype(float) * (1.0 + r)   # defector (i) against cooperator (j): T
        base += ( si * (1-sj) ).astype(float) * (-r)        # cooperator (i) against defector (j): S
        # P=0 adds nothing

        S += base + rng.normal(0.0, sigma, size=si.shape)

    # choose one random neighbor per site to compare with; synchronous update
    dirs = rng.integers(0, 4, size=strat.shape)
    neigh_S = np.zeros_like(S)
    neigh_X = np.zeros_like(strat)

    for k,(dy,dx) in enumerate(shifts):
        mask = (dirs == k)
        if not np.any(mask): continue
        Sj = np.roll(np.roll(S, dy, axis=0), dx, axis=1)
        Xj = np.roll(np.roll(strat, dy, axis=0), dx, axis=1)
        neigh_S[mask] = Sj[mask]
        neigh_X[mask] = Xj[mask]

    # Fermi imitation
    W = 1.0 / (1.0 + np.exp((S - neigh_S)/K))
    switch = rng.random(strat.shape) < W
    new_strat = strat.copy()
    new_strat[switch] = neigh_X[switch]
    return new_strat

def run(n=300, r=0.0065, sigma=0.20, K=0.1, burn=30000, avg=30000, seed=0, init_coop=0.5):
    rng = np.random.default_rng(seed)
    strat = (rng.random((n,n)) < init_coop).astype(np.int8)

    for _ in range(burn):
        strat = step_once(strat, r, sigma, K, rng)

    # average FC over a long window (paper averages after a long transient)
    FC_sum = 0.0
    for _ in range(avg):
        strat = step_once(strat, r, sigma, K, rng)
        FC_sum += strat.mean()
    FC = FC_sum / avg
    return strat, FC

def show(grid, title="", plotitle=""):
    cmap = ListedColormap(['red','blue'])  # red=D, blue=C (like the paper)
    plt.figure(figsize=(6,6)); plt.imshow(grid, cmap=cmap, interpolation='nearest')
    plt.xticks([]); plt.yticks([]); plt.title(title); plt.savefig(plotitle); plt.show()

# --- Figure 2 replicas (may take several minutes on Colab) ---
params = dict(n=300, r=0.0065, K=0.1, burn=2000, avg=10000, seed=42)  # you can raise to 3e4–1e5
for sig in [0.001, 0.5]:
    grid, FC = run(sigma=sig, **params)
    show(grid, f"σ={sig} | r=0.0065 | ⟨F_C⟩ ≈ {FC:.3f}", plotitle=f"fig{sig}.png")

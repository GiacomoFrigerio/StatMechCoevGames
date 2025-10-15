#@title Resonance curve (various r for varying sigma)
import numpy as np
import matplotlib.pyplot as plt

# model
def step_once(strat, r, sigma, K, rng):
    shifts = [(0,1),(0,-1),(1,0),(-1,0)]
    S = np.zeros_like(strat, dtype=float)

    for dy,dx in shifts:
        sj = np.roll(np.roll(strat, dy, axis=0), dx, axis=1)
        si = strat

        # PD payoffs with additive noise to the focal player's payoff per pair
        base  = (si & sj).astype(float) * 1.0                # R=1 (C vs C)
        base += ((1-si) * sj).astype(float) * (1.0 + r)      # T=1+r (D vs C)
        base += (si * (1-sj)).astype(float) * (-r)           # S=-r (C vs D)
        S += base + rng.normal(0.0, sigma, size=si.shape)    # P=0 omitted

    # pick one neighbor uniformly; synchronous imitation via Fermi
    dirs = rng.integers(0, 4, size=strat.shape)
    neigh_S = np.zeros_like(S)
    neigh_X = np.zeros_like(strat)

    for k,(dy,dx) in enumerate([(0,1),(0,-1),(1,0),(-1,0)]):
        mask = (dirs == k)
        if not np.any(mask): continue
        Sj = np.roll(np.roll(S, dy, axis=0), dx, axis=1)
        Xj = np.roll(np.roll(strat, dy, axis=0), dx, axis=1)
        neigh_S[mask] = Sj[mask]
        neigh_X[mask] = Xj[mask]

    W = 1.0 / (1.0 + np.exp((S - neigh_S)/K))
    switch = rng.random(strat.shape) < W
    new_strat = strat.copy()
    new_strat[switch] = neigh_X[switch]
    return new_strat

def time_avg_FC(n=400, r=0.0065, sigma=0.2, K=0.1, burn=30000, avg=30000, seed=0, init_coop=0.5):
    rng = np.random.default_rng(seed)
    strat = (rng.random((n,n)) < init_coop).astype(np.int8)
    for _ in range(burn):
        strat = step_once(strat, r, sigma, K, rng)
    FC_sum = 0.0
    for _ in range(avg):
        strat = step_once(strat, r, sigma, K, rng)
        FC_sum += strat.mean()
    return FC_sum / avg

# sigma for varying r
n = 300
K = 0.1
burn, avg = 2000, 5000   # raise to 3e4–1e5 near r_tr for smoother curves
sigma_grid = np.linspace(0.0, 1.0, 11)  # 0.00 to 1.00 step 0.04; adjust as needed

# choose r just above r_tr = 0.00634... plus some farther values
#r_values = [0.00640, 0.00650, 0.00670, 0.00700]  # all > r_tr; include 0.00650 to match Fig.2
r_values = [0.00670, 0.00700]  # all > r_tr; include 0.00650 to match Fig.2

results = {}
base_seed = 123
for ir, r in enumerate(r_values):
    FCs = []
    for isg, sig in enumerate(sigma_grid):
        # different seeds per (r,σ) to avoid correlated noise; same across runs is fine too
        seed = base_seed + ir*1000 + isg
        FCs.append(time_avg_FC(n=n, r=r, sigma=float(sig), K=K, burn=burn, avg=avg, seed=seed))
    results[r] = np.array(FCs)

# plot
plt.figure(figsize=(7,5))
for r, FCs in results.items():
    plt.plot(sigma_grid, FCs, marker='o', label=f"r = {r:.5f}")
plt.xlabel(r"Noise level $\sigma$")
plt.ylabel(r"Average cooperator fraction $\langle F_C\rangle$")
plt.title("Coherence resonance: $F_C(\\sigma)$ for several $r>r_{tr}$")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("ResonanceCurve", dpi=100)
plt.show()

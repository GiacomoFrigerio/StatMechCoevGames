import numpy as np
import matplotlib.pyplot as plt

# =======================
# Parameters
# =======================
N = 1600                    # must be a perfect square (L*L)
w = 1.0
k = 4                       # lattice degree (von Neumann)
timesteps = 100000           # elementary updates (1 focal update per step)
c = 1
Dr_arrival = -1
Dg_arrival = -1
lambda_param = 1

# ---- choose ONE (D_r, D_g) pair here ----
D_r = -0.65
D_g = 0.35
# -----------------------------------------

initial_cooperator_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
plot_every = 50  # downsample for plotting

rng = np.random.default_rng()  # fast, modern RNG

# =======================
# Core Functions
# =======================
def initialize_population(N, initial_cooperator_fraction):
    num_cooperators = int(initial_cooperator_fraction * N)
    strategies = np.array(['C'] * num_cooperators + ['D'] * (N - num_cooperators))
    rng.shuffle(strategies)
    return strategies

def get_payoff_matrix(fraction_cooperators, D_r, D_g):
    base_matrix_1 = np.array([[1, -D_r],
                              [1 + D_g, 0]], dtype=float)
    base_matrix_2 = np.array([[0, - Dr_arrival + D_r],
                              [Dg_arrival - D_g, 0]], dtype=float)
    additional_matrix = c * (fraction_cooperators ** lambda_param) * base_matrix_2
    return base_matrix_1 + additional_matrix

def calculate_payoff(payoff_matrix, player_strategy, opponent_strategy):
    si = 0 if player_strategy == 'C' else 1
    sj = 0 if opponent_strategy == 'C' else 1
    return payoff_matrix[si, sj]

def fermi_update(delta_payoff, w):
    return 1.0 / (1.0 + np.exp(-w * delta_payoff))

# =======================
# Lattice utilities
# =======================
def build_lattice_neighbors(N):
    """
    Build neighbors for an LxL periodic lattice with von Neumann (k=4) neighborhood.
    Returns a list of 4-neighbor index lists for each node (0..N-1).
    """
    L = int(np.sqrt(N))
    assert L * L == N, "N must be a perfect square for an LxL lattice."

    def idx(i, j):
        return (i % L) * L + (j % L)

    neighbors = [[] for _ in range(N)]
    for i in range(L):
        for j in range(L):
            u = idx(i, j)
            nbs = [
                idx(i - 1, j),  # up
                idx(i + 1, j),  # down
                idx(i, j - 1),  # left
                idx(i, j + 1),  # right
            ]
            neighbors[u] = nbs
    return neighbors

LATTICE_NEIGHBORS = build_lattice_neighbors(N)  # built once
L = int(np.sqrt(N))

def local_payoff_sum(payoff_matrix, strategies, u):
    """Sum payoff of node u vs its 4 neighbors (no self-interaction)."""
    s_u = strategies[u]
    payoff = 0.0
    for v in LATTICE_NEIGHBORS[u]:
        payoff += calculate_payoff(payoff_matrix, s_u, strategies[v])
    return payoff

def strategies_to_grid(strategies):
    """Map strategies to 1 (C) and 0 (D) and reshape to LxL."""
    grid = (strategies == 'C').astype(int).reshape(L, L)
    return grid

def simulate_lattice_with_snapshots(N, w, timesteps, initial_cooperator_fraction, D_r, D_g):
    """
    Same dynamic as before, but also returns:
    - coop_fraction time series
    - initial_grid (t=0)
    - final_grid (t=timesteps)
    """
    strategies = initialize_population(N, initial_cooperator_fraction)
    cooperators_fraction = []
    initial_grid = strategies_to_grid(strategies)  # snapshot at t=0

    for t in range(timesteps):
        rho = np.mean(strategies == 'C')
        cooperators_fraction.append(rho)

        payoff_matrix = get_payoff_matrix(rho, D_r, D_g)
        x = rng.integers(0, N)
        y = rng.choice(LATTICE_NEIGHBORS[x])

        pi_x = local_payoff_sum(payoff_matrix, strategies, x)
        pi_y = local_payoff_sum(payoff_matrix, strategies, y)

        if rng.random() < fermi_update(pi_y - pi_x, w):
            strategies[x] = strategies[y]

    final_grid = strategies_to_grid(strategies)   # snapshot at final time
    return cooperators_fraction, initial_grid, final_grid

def quadrant_color(Dg, Dr):
    if Dg >= 0 and Dr >= 0:
        return '#FF0000'   # Prisoner's Dilemma
    elif Dg <= 0 and Dr <= 0:
        return '#009900'   # Harmony
    elif Dg <= 0 and Dr >= 0:
        return '#0000FF'   # Stag Hunt
    elif Dg >= 0 and Dr <= 0:
        return '#FF9933'   # Snowdrift
    else:
        return '#A0A0A0'   # Undefined

# =======================
# Run experiments for the single (D_r, D_g)
# =======================
print(f"\n=== Running single-parameter study: D_r={D_r:.2f}, D_g={D_g:.2f} ===\n")

# 1) Time-series figure (like before, but only for this pair)
fig_ts, ax = plt.subplots(figsize=(10, 6))

snapshots_initial = []
snapshots_final = []
final_rhos = []

for rho_0 in initial_cooperator_fractions:
    coop_fraction, init_grid, fin_grid = simulate_lattice_with_snapshots(
        N, w, timesteps, rho_0, D_r, D_g
    )
    snapshots_initial.append(init_grid)
    snapshots_final.append(fin_grid)
    final_rhos.append(coop_fraction[-1])

    t_idx = np.arange(0, len(coop_fraction), plot_every)
    cf = np.array(coop_fraction)[t_idx]

    # color by effective game quadrant over time
    color_list = []
    for rho in cf:
        Dg_t = (1 - rho) * D_g + rho * Dg_arrival
        Dr_t = (1 - rho) * D_r + rho * Dr_arrival
        color_list.append(quadrant_color(Dg_t, Dr_t))

    ax.scatter(t_idx, cf, c=color_list, s=16) #, label=f"ρ₀={rho_0:.2f}"

# thresholds for sign changes of Dg_t or Dr_t
if (D_g - Dg_arrival) != 0:
    y_thr = D_g / (D_g - Dg_arrival)
    if 0 <= y_thr <= 1:
        ax.hlines(y_thr, xmin=0, xmax=timesteps, colors='#A0A0A0', linestyles='dashed', linewidth=2)
if (D_r - Dr_arrival) != 0:
    y_thr = D_r / (D_r - Dr_arrival)
    if 0 <= y_thr <= 1:
        ax.hlines(y_thr, xmin=0, xmax=timesteps, colors='#A0A0A0', linestyles='dashed', linewidth=2)

ax.set_title(f"2D Lattice (k=4) — ρ(t) for single pair  Dg={D_g:.2f}, Dr={D_r:.2f}")
ax.set_xlim(0, timesteps)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$\rho$")
ax.set_xticks([0, timesteps//2, timesteps])
ax.legend(loc="best", frameon=False, ncol=2)
fig_ts.tight_layout()
fig_ts.savefig("TimeSeries_single_pair_lattice.pdf", bbox_inches='tight', dpi=300)

# 2) Snapshot figure: rows = len(initial_cooperator_fractions), cols = 2 (init, final)
rows = len(initial_cooperator_fractions)
fig_snap, axs_snap = plt.subplots(rows, 2, figsize=(10, 2.2*rows))

if rows == 1:
    axs_snap = np.array([axs_snap])  # ensure 2D indexing

for i, rho_0 in enumerate(initial_cooperator_fractions):
    # initial snapshot
    ax0 = axs_snap[i, 0]
    im0 = ax0.imshow(snapshots_initial[i], interpolation='nearest', cmap='binary')
    ax0.set_title(f"Init (ρ₀={rho_0:.2f})")
    ax0.set_xticks([]); ax0.set_yticks([])

    # final snapshot
    ax1 = axs_snap[i, 1]
    im1 = ax1.imshow(snapshots_final[i], interpolation='nearest', cmap='binary')
    ax1.set_title(f"Final (ρ={final_rhos[i]:.3f})")
    ax1.set_xticks([]); ax1.set_yticks([])

fig_snap.suptitle(f"Lattice snapshots — Dg={D_g:.2f}, Dr={D_r:.2f} (left: t=0, right: t=end)")
fig_snap.tight_layout(rect=[0, 0, 1, 0.95])
fig_snap.savefig("Snapshots_single_pair_lattice.pdf", bbox_inches='tight', dpi=300)

plt.show()

print("\n✅ Done: generated time series and initial/final lattice snapshots for the chosen (D_r, D_g).")

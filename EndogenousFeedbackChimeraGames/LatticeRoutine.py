import numpy as np
import matplotlib.pyplot as plt

# =======================
# Parameters
# =======================
N = 1600                    # must be a perfect square (L*L)
w = 1.0
k = 4                       # lattice degree (von Neumann)
timesteps = 50000           # elementary updates (1 focal update per step)
c = 0
Dr_arrival = -1
Dg_arrival = -1
lambda_param = 1

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

# Return 1x: payoff for (player_strategy vs opponent_strategy)
def calculate_payoff(payoff_matrix, player_strategy, opponent_strategy):
    si = 0 if player_strategy == 'C' else 1
    sj = 0 if opponent_strategy == 'C' else 1
    return payoff_matrix[si, sj]

def fermi_update(delta_payoff, w):
    # 1/(1+exp(-w*Δπ))
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
            # Up, Down, Left, Right with periodic boundary conditions
            nbs = [
                idx(i - 1, j),  # up
                idx(i + 1, j),  # down
                idx(i, j - 1),  # left
                idx(i, j + 1),  # right
            ]
            neighbors[u] = nbs
    return neighbors

LATTICE_NEIGHBORS = build_lattice_neighbors(N)  # built once

def local_payoff_sum(payoff_matrix, strategies, u):
    """Sum payoff of node u vs its 4 neighbors (no self-interaction)."""
    s_u = strategies[u]
    payoff = 0.0
    for v in LATTICE_NEIGHBORS[u]:
        payoff += calculate_payoff(payoff_matrix, s_u, strategies[v])
    return payoff

def simulate_lattice(N, w, timesteps, initial_cooperator_fraction, D_r, D_g):
    """
    Asynchronous (random sequential) update on a 2D LxL lattice (k=4).
    At each step:
      - pick random focal node x
      - pick random neighbor y in N(x)
      - compute local payoffs π_x, π_y (sum vs their 4 neighbors)
      - x adopts y with Fermi probability based on π_y - π_x
    The payoff matrix depends on the *global* ρ(t) via get_payoff_matrix.
    """
    strategies = initialize_population(N, initial_cooperator_fraction)
    cooperators_fraction = []

    for t in range(timesteps):
        rho = np.mean(strategies == 'C')
        cooperators_fraction.append(rho)

        payoff_matrix = get_payoff_matrix(rho, D_r, D_g)

        # pick focal x and a neighbor y of x
        x = rng.integers(0, N)
        y = rng.choice(LATTICE_NEIGHBORS[x])

        # compute local payoffs (sum vs 4 neighbors)
        pi_x = local_payoff_sum(payoff_matrix, strategies, x)
        pi_y = local_payoff_sum(payoff_matrix, strategies, y)

        # imitation with Fermi
        if rng.random() < fermi_update(pi_y - pi_x, w):
            strategies[x] = strategies[y]

    return cooperators_fraction

def quadrant_color(Dg, Dr):
    if Dg >= 0 and Dr >= 0:
        return '#FF0000'   # Prisoner's Dilemma
    elif Dg <= 0 and Dr <= 0:
        return '#009900'   # Harmony (single green)
    elif Dg <= 0 and Dr >= 0:
        return '#0000FF'   # Stag Hunt
    elif Dg >= 0 and Dr <= 0:
        return '#FF9933'   # Snowdrift
    else:
        return '#A0A0A0'   # Undefined

# =======================
# Build 20 (D_r, D_g) pairs
# =======================
vals = [0.2, 0.35, 0.5, 0.65, 0.8]

PD_pairs = [(dr, dg) for dr, dg in [(vals[0], vals[0]), (vals[1], vals[3]),
                                    (vals[2], vals[2]), (vals[3], vals[1]), (vals[4], vals[4])]]
HG_pairs = [(-dr, -dg) for (dr, dg) in [(vals[0], vals[0]), (vals[1], vals[3]),
                                        (vals[2], vals[2]), (vals[3], vals[1]), (vals[4], vals[4])]]
SD_pairs = [(-dr, dg) for (dr, dg) in [(vals[0], vals[0]), (vals[1], vals[3]),
                                       (vals[2], vals[2]), (vals[3], vals[1]), (vals[4], vals[4])]]
SH_pairs = [(dr, -dg) for (dr, dg) in [(vals[0], vals[0]), (vals[1], vals[3]),
                                       (vals[2], vals[2]), (vals[3], vals[1]), (vals[4], vals[4])]]

rows = [
    ("Prisoner's Dilemma (Dg>0, Dr>0)", PD_pairs),
    ("Harmony Game (Dg<0, Dr<0)", HG_pairs),
    ("Snowdrift (Dg>0, Dr<0)", SD_pairs),
    ("Stag Hunt (Dg<0, Dr>0)", SH_pairs),
]

# =======================
# Plot 4x5 grid (same as before)
# =======================
fig, axs = plt.subplots(4, 5, figsize=(24, 18))
axs = axs.reshape(4, 5)

for row_idx, (row_title, pair_list) in enumerate(rows):
    print(f"\n=== Starting simulations for {row_title} ===\n")
    for col_idx, (D_r, D_g) in enumerate(pair_list):
        ax = axs[row_idx, col_idx]
        print(f"  → Running pair {col_idx + 1}/5: D_r={D_r:.2f}, D_g={D_g:.2f}")

        for rho_0 in initial_cooperator_fractions:
            coop_fraction = simulate_lattice(N, w, timesteps, rho_0, D_r, D_g)

            t_idx = np.arange(0, len(coop_fraction), plot_every)
            cf = np.array(coop_fraction)[t_idx]

            # color by the instantaneous "effective game" determined by Dg_t, Dr_t
            color_list = []
            for rho in cf:
                Dg_t = (1 - rho) * D_g + rho * Dg_arrival
                Dr_t = (1 - rho) * D_r + rho * Dr_arrival
                color_list.append(quadrant_color(Dg_t, Dr_t))

            ax.scatter(t_idx, cf, c=color_list, s=14, label=f"ρ₀={rho_0:.2f}")

            # horizontal thresholds where Dg_t or Dr_t would change sign
            if (D_g - Dg_arrival) != 0:
                y_thr = D_g / (D_g - Dg_arrival)
                if 0 <= y_thr <= 1:
                    ax.hlines(y_thr, xmin=0, xmax=len(coop_fraction), colors='#A0A0A0',
                              linestyles='dashed', linewidth=2)
            if (D_r - Dr_arrival) != 0:
                y_thr = D_r / (D_r - Dr_arrival)
                if 0 <= y_thr <= 1:
                    ax.hlines(y_thr, xmin=0, xmax=len(coop_fraction), colors='#A0A0A0',
                              linestyles='dashed', linewidth=2)

        print(f"     ✓ Finished D_r={D_r:.2f}, D_g={D_g:.2f}")

        ax.set_title(f"$D_g$ = {D_g:.2f}, $D_r$ = {D_r:.2f}", fontsize=16)
        ax.set_xlim(0, timesteps)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, timesteps//2, timesteps])
        ax.set_xticklabels([f"0", f"{timesteps//2}", f"{timesteps}"], fontsize=11)
        if col_idx == 0:
            ax.set_yticks([0, 0.5, 1.0])
            ax.set_yticklabels(['0', '0.5', '1'], fontsize=11)
        else:
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['0', '1'], fontsize=11)
        ax.grid(False)

    print(f"\n=== Completed all 5 pairs for {row_title} ===\n" + "-"*70)

# Add labels and save
for col in range(5):
    axs[-1, col].set_xlabel(r'$t$', fontsize=18)

for row_idx, (row_title, _) in enumerate(rows):
    axs[row_idx, 0].set_ylabel(row_title + "\n\n$\\rho$", fontsize=18)

fig.suptitle(
    f"2D Lattice (k=4, periodic) — Target values: Dg = {Dg_arrival} and Dr = {Dr_arrival}  —  20 (D_r, D_g) pairs (5 per dilemma)",
    fontsize=22
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Fraction_Cooperators_4x5_lattice.pdf", bbox_inches='tight', dpi=300)
plt.show()

print("\n✅ All lattice simulations and plotting completed successfully!")

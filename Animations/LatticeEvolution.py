import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =======================
# Parameters
# =======================
N = 6400                    # must be a perfect square (L*L)
w = 1.0
k = 4
timesteps = 100000          # total steps
save_interval = 500         # update animation every X steps
c = 1
Dr_arrival = 1
Dg_arrival = 1
lambda_param = 1

# ---- choose ONE (D_r, D_g) pair here ----
D_r = -0.35
D_g = -0.75
# -----------------------------------------

initial_cooperator_fraction = 0.8
rng = np.random.default_rng()

# =======================
# Helper functions
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

def build_lattice_neighbors(N):
    L = int(np.sqrt(N))
    assert L * L == N, "N must be a perfect square for an LxL lattice."
    def idx(i, j):
        return (i % L) * L + (j % L)
    neighbors = [[] for _ in range(N)]
    for i in range(L):
        for j in range(L):
            u = idx(i, j)
            nbs = [idx(i - 1, j), idx(i + 1, j), idx(i, j - 1), idx(i, j + 1)]
            neighbors[u] = nbs
    return neighbors, L

LATTICE_NEIGHBORS, L = build_lattice_neighbors(N)

def local_payoff_sum(payoff_matrix, strategies, u):
    s_u = strategies[u]
    payoff = 0.0
    for v in LATTICE_NEIGHBORS[u]:
        payoff += calculate_payoff(payoff_matrix, s_u, strategies[v])
    return payoff

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
        return '#A0A0A0'

# =======================
# Animation
# =======================
def simulate_and_animate(N, w, timesteps, initial_cooperator_fraction, D_r, D_g, save_interval):
    strategies = initialize_population(N, initial_cooperator_fraction)
    rho_series = []
    color_series = []

    # Create figure with larger lattice
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.35)

    # --- Left panel: rho(t) plot ---
    scat = ax_left.scatter([], [], s=18)
    ax_left.set_xlim(0, timesteps)
    ax_left.set_ylim(0, 1)
    ax_left.set_xlabel(r"$t$")
    ax_left.set_ylabel(r"$\rho$")
    ax_left.set_title(f"Cooperation Dynamics — Dg={D_g:.2f}, Dr={D_r:.2f}")

    # --- Right panel: lattice snapshot ---
    im = ax_right.imshow((strategies == 'C').astype(int).reshape(L, L),
                         cmap='binary', interpolation='nearest', vmin=0, vmax=1)
    ax_right.set_title(f"Lattice snapshot (ρ₀={initial_cooperator_fraction:.2f})")
    ax_right.axis('off')

    # --- Update function for animation ---
    def update(frame):
        nonlocal strategies
        for _ in range(save_interval):
            rho = np.mean(strategies == 'C')
            payoff_matrix = get_payoff_matrix(rho, D_r, D_g)
            x = rng.integers(0, N)
            y = rng.choice(LATTICE_NEIGHBORS[x])
            pi_x = local_payoff_sum(payoff_matrix, strategies, x)
            pi_y = local_payoff_sum(payoff_matrix, strategies, y)
            if rng.random() < fermi_update(pi_y - pi_x, w):
                strategies[x] = strategies[y]
            rho_series.append(rho)

            # compute effective game color
            Dg_t = (1 - rho) * D_g + rho * Dg_arrival
            Dr_t = (1 - rho) * D_r + rho * Dr_arrival
            color_series.append(quadrant_color(Dg_t, Dr_t))

        t_vals = np.arange(len(rho_series))
        scat.set_offsets(np.c_[t_vals, rho_series])
        scat.set_color(color_series)
        im.set_data((strategies == 'C').astype(int).reshape(L, L))
        ax_right.set_title(f"Lattice snapshot — ρ={rho_series[-1]:.3f}")
        return scat, im

    frames = timesteps // save_interval
    anim = FuncAnimation(fig, update, frames=frames, interval=60, blit=False, repeat=False)

    # Save animation
    anim.save("LatticeEvolution.mp4", writer="ffmpeg", dpi=200)
    # To save as GIF instead:
    # anim.save("LatticeEvolution.gif", writer="pillow", dpi=150)

    plt.show()
    return rho_series

# =======================
# Run the animation
# =======================
print(f"\n=== Animating lattice evolution with color coding: D_r={D_r:.2f}, D_g={D_g:.2f}, ρ₀={initial_cooperator_fraction:.2f} ===\n")
rho_series = simulate_and_animate(N, w, timesteps, initial_cooperator_fraction, D_r, D_g, save_interval)
print("\n✅ Animation complete! Saved as 'LatticeEvolution.mp4'")

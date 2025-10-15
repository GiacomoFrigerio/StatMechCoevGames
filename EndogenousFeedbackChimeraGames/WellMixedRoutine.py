import numpy as np
import matplotlib.pyplot as plt

# =======================
# Parameters
# =======================
N = 2500
w = 1.0
k = 4
timesteps = 50000
c = 1
Dr_arrival = 0.3
Dg_arrival = 0.3
lambda_param = 1

initial_cooperator_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
plot_every = 50  # downsample for plotting

# =======================
# Core Functions
# =======================
def initialize_population(N, initial_cooperator_fraction):
    num_cooperators = int(initial_cooperator_fraction * N)
    strategies = np.array(['C'] * num_cooperators + ['D'] * (N - num_cooperators))
    np.random.shuffle(strategies)
    return strategies

def get_payoff_matrix(fraction_cooperators, D_r, D_g):
    base_matrix_1 = np.array([[1, -D_r],
                              [1 + D_g, 0]])
    base_matrix_2 = np.array([[0, - Dr_arrival + D_r],
                              [Dg_arrival - D_g, 0]])
    additional_matrix = c * fraction_cooperators**lambda_param * base_matrix_2
    return base_matrix_1 + additional_matrix

def calculate_payoff(payoff_matrix, player_strategy, opponent_strategy):
    strategy_index = {'C': 0, 'D': 1}
    return payoff_matrix[strategy_index[player_strategy], strategy_index[opponent_strategy]]

def fermi_update(delta_payoff, w):
    return 1 / (1 + np.exp(-w * delta_payoff))

def simulate_well_mixed(N, w, k, timesteps, initial_cooperator_fraction, D_r, D_g):
    strategies = initialize_population(N, initial_cooperator_fraction)
    cooperators_fraction = []

    for t in range(timesteps):
        fraction_cooperators = np.mean(strategies == 'C')
        cooperators_fraction.append(fraction_cooperators)

        payoff_matrix = get_payoff_matrix(fraction_cooperators, D_r, D_g)
        focal, model = np.random.randint(0, N, 2)

        focal_payoff = np.sum([
            calculate_payoff(payoff_matrix, strategies[focal], strategies[j])
            for j in np.random.choice(N, k)
        ])
        model_payoff = np.sum([
            calculate_payoff(payoff_matrix, strategies[model], strategies[j])
            for j in np.random.choice(N, k)
        ])

        adoption_prob = fermi_update(model_payoff - focal_payoff, w)
        if np.random.rand() < adoption_prob:
            strategies[focal] = strategies[model]

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
# Plot 4x5 grid
# =======================
fig, axs = plt.subplots(4, 5, figsize=(24, 18))
axs = axs.reshape(4, 5)

for row_idx, (row_title, pair_list) in enumerate(rows):
    print(f"\n=== Starting simulations for {row_title} ===\n")
    for col_idx, (D_r, D_g) in enumerate(pair_list):
        ax = axs[row_idx, col_idx]
        print(f"  → Running pair {col_idx + 1}/5: D_r={D_r:.2f}, D_g={D_g:.2f}")

        for rho_0 in initial_cooperator_fractions:
            coop_fraction = simulate_well_mixed(N, w, k, timesteps, rho_0, D_r, D_g)
            t_idx = np.arange(0, len(coop_fraction), plot_every)
            cf = np.array(coop_fraction)[t_idx]

            color_list = []
            for rho in cf:
                Dg_t = (1 - rho) * D_g + rho * Dg_arrival
                Dr_t = (1 - rho) * D_r + rho * Dr_arrival
                color_list.append(quadrant_color(Dg_t, Dr_t))

            ax.scatter(t_idx, cf, c=color_list, s=14, label=f"ρ₀={rho_0:.2f}")

            if (D_g - Dg_arrival) != 0:
                y_thr = D_g / (D_g - Dg_arrival)
                ax.hlines(y_thr, xmin=0, xmax=len(coop_fraction), colors='#A0A0A0',
                          linestyle='dashed', linewidth=2)
            if (D_r - Dr_arrival) != 0:
                y_thr = D_r / (D_r - Dr_arrival)
                ax.hlines(y_thr, xmin=0, xmax=len(coop_fraction), colors='#A0A0A0',
                          linestyle='dashed', linewidth=2)

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

fig.suptitle(f"Target values: Dg = {Dg_arrival} and Dr = {Dr_arrival}  —  20 (D_r, D_g) pairs (5 per dilemma)", fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("Fraction_Cooperators_4x5.pdf", bbox_inches='tight', dpi=300)
plt.show()

print("\n✅ All simulations and plotting completed successfully!")

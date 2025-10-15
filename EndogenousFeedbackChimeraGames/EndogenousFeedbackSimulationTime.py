import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
import string

# Parameters
N = 2500
w = 1.0
k = 4
timesteps = 25000
c = 1
Dr_arrival = 1
Dg_arrival = 1
lambda_param = 1

# initial_cooperator_fractions = np.linspace(0.1, 0.9, 9)
# initial_cooperator_fractions = [0.1,0.32,0.54,0.76]
initial_cooperator_fractions = [0.1,0.42,0.74]

D_pairs = [(-0.3, -0.8), (-0.8, -0.3), (-0.7, 0.7), (0.75, -0.5)]

# Functions
def initialize_population(N, initial_cooperator_fraction):
    num_cooperators = int(initial_cooperator_fraction * N)
    strategies = np.array(['C'] * num_cooperators + ['D'] * (N - num_cooperators))
    np.random.shuffle(strategies)
    return strategies

def get_payoff_matrix(fraction_cooperators, D_r, D_g):
    base_matrix_1 = np.array([[1, -D_r], [1 + D_g, 0]])
    base_matrix_2 = np.array([[0, - Dr_arrival + D_r], [Dg_arrival - D_g, 0]])
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

# === Create 4 subplots ===
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
axs = axs.flatten()  # Make indexing easier

for idx, (D_r, D_g) in enumerate(D_pairs):
    ax = axs[idx]
    for rho_0 in initial_cooperator_fractions:
        print(f"Running for D_r={D_r}, D_g={D_g}, rho₀={rho_0}")
        coop_fraction = simulate_well_mixed(N, w, k, timesteps, rho_0, D_r, D_g)
        timesteps_range = list(range(len(coop_fraction)))

        color_list = []
        for i in timesteps_range:
            Dg = (1 - coop_fraction[i]) * D_g + coop_fraction[i] * Dg_arrival
            Dr = (1 - coop_fraction[i]) * D_r + coop_fraction[i] * Dr_arrival

            if Dg >= 0 and Dr >= 0:
                color_code = '#FF0000'       # Prisoner's Dilemma
            elif Dg <= 0 and Dr <= 0 and Dg <= Dr:
                color_code = '#80FF00'       # Harmony game Dr > Dg
            elif Dg <= 0 and Dr <= 0 and Dg >= Dr:
                color_code = '#009900'       # Harmony game Dr < Dg
            elif Dg <= 0 and Dr >= 0:
                color_code = '#0000FF'       # Stag Hunt
            elif Dg >= 0 and Dr <= 0:
                color_code = '#FF9933'       # Chicken Game
            else:
                color_code = '#A0A0A0'       # Border / undefined

            color_list.append(color_code)

        ax.scatter(timesteps_range, coop_fraction, c=color_list, s=28, label=f"ρ₀={rho_0:.2f}")

        if D_g - Dg_arrival != 0:
            ax.hlines(D_g/(D_g - Dg_arrival), xmin=0, xmax=len(coop_fraction),
                      colors='#A0A0A0', linestyle='dashed', linewidth=4)
        if D_r - Dr_arrival != 0:
            ax.hlines(D_r/(D_r - Dr_arrival), xmin=0, xmax=len(coop_fraction),
                      colors='#A0A0A0', linestyle='dashed', linewidth=4)

    letter = string.ascii_lowercase[idx]
    ax.set_title(f"({letter}) $D_g$ = {D_g}, $D_r$ = {D_r}", fontsize=30)
    #ax.set_title(f"$D_g$ = {D_g}, $D_r$ = {D_r}", fontsize=30)
    ax.set_xlabel(r'$t$', fontsize=26)
    ax.set_ylabel(r'$\rho$', fontsize=26)
    ax.set_xticks([0, 10000, 20000])
    ax.set_xticklabels(['0', '10000', '20000'], fontsize=20)
    ax.set_yticks([0.5, 1])
    ax.set_yticklabels(['0.5', '1'], fontsize=20)
    ax.set_xlim(0, 25000)
    ax.set_ylim(0, 1)
    ax.grid(False)

plt.tight_layout()
# plt.savefig("Fraction_Cooperators_Subplots.pdf", bbox_inches='tight', dpi=900)
plt.show()

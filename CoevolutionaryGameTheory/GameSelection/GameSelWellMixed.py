import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from scipy.optimize import root_scalar
import pandas as pd
import io

# Parameters
N = 2500  # Population size
alpha = 4  # Payoff matrix parameter α
beta = 2  # Payoff matrix parameter β
pg = 0.1  # Probability of updating the game   <----- we mainly want to play with this parameter
ps = 1.0  # Probability of updating the strategy
w = 1.0  # Selection strength (in Fermi transition probability)
k = 4 #number of interaction with other players
T_range = (beta / 2, alpha - beta / 2)
S_range = (beta - alpha / 2, alpha / 2)
timesteps = 500000

def initialize_population(N, alpha, beta, initial_cooperator_fraction=0.5):
    num_cooperators = int(initial_cooperator_fraction * N)
    strategies = np.array(['C'] * num_cooperators + ['D'] * (N - num_cooperators))
    np.random.shuffle(strategies)  # Shuffle to randomize cooperators and defectors

    games = []
    for _ in range(N):
        valid = False
        while not valid:
            # Sample T and S from the valid ranges
            T = np.random.uniform(beta / 2, alpha - beta / 2)
            S = np.random.uniform(beta - alpha / 2, alpha / 2)

            # Ensure T + S is within [2, 4]
            if 2 <= T + S <= 4 and 0 <= T - S <= 2: #satisfy diamond conditions
                games.append([T, S])
                valid = True

    games = np.array(games)
    return strategies, games


# Calculate payoff for a player
def calculate_payoff(player_game, opponent_game, player_strategy, opponent_strategy):
    T, S = player_game
    R, P = alpha - T, beta - S  #recall that R and P are defined by the relations T + R = alpha , S + P = beta
    #choose payoffs depending on strategy
    if player_strategy == 'C' and opponent_strategy == 'C':
        return R
    elif player_strategy == 'C' and opponent_strategy == 'D':
        return S
    elif player_strategy == 'D' and opponent_strategy == 'C':
        return T
    else:  # Both defect
        return P

# Fermi update rule
def fermi_update(delta_payoff, w):
    return 1 / (1 + np.exp(-w * delta_payoff)) #w defined above

# Classify game type based on T and S
def classify_game(T, S, alpha, beta):
    R = alpha - T
    P = beta - S
    if T > R and S > P:
        return "Snowdrift (SD)"
    elif T > R and S < P:
        return "Prisoner's Dilemma (PD)"
    elif T < R and S < P:
        return "Stag Hunt (SH)"
    elif T < R and S > P:
        return "Harmony Game (HG)"
    else:
        return "Unclassified"
      
# Simulation function
def simulate_well_mixed(N, alpha, beta, pg, ps, w, k, timesteps, numcop=0.5, print_initial_cond = False):
    strategies, games = initialize_population(N, alpha, beta, numcop)
    cooperators_fraction = []

    if print_initial_cond == True:
     #PRINT INITIAL GAME DISTRIBUTION (CHECK)
     T_values = games[:, 0]
     S_values = games[:, 1]

     plt.figure(figsize=(8, 8))
     plt.scatter(T_values, S_values, alpha=0.7, s=10, color='blue')
     plt.xlabel("T (Temptation)")
     plt.ylabel("S (Sucker's Payoff)")
     plt.title("Initial Games Distribution (T vs S)")
     #plt.ylim(0,2)
     #plt.xlim(1,3)
     plt.grid()
     plt.show()


    for t in range(timesteps):
        #if t % 10000 == 0:
        #print(f"Timestep: {t}")
        # Track the fraction of cooperators
        fraction_cooperators = np.sum(strategies == 'C') / N
        cooperators_fraction.append(fraction_cooperators) #updates the state

        # Randomly select a focal and model player
        focal = np.random.randint(0, N)
        model = np.random.randint(0, N)

        # Calculate total payoffs for focal and model
        focal_payoff = np.sum([calculate_payoff(games[focal], games[j], strategies[focal], strategies[j])
                               for j in np.random.choice(N, k)])  # Interact with k random players (k=4)
        model_payoff = np.sum([calculate_payoff(games[model], games[j], strategies[model], strategies[j])
                               for j in np.random.choice(N, k)])

        # Fermi probability transition
        adoption_prob = fermi_update(model_payoff - focal_payoff, w)

        # Update game with probability pg
        if np.random.rand() < pg:
            if np.random.rand() < adoption_prob:
                games[focal] = games[model]

        # Update strategy with probability ps
        if np.random.rand() < ps:
            if np.random.rand() < adoption_prob:
                strategies[focal] = strategies[model]

    return cooperators_fraction, games


results = []
surviving_games = []

numcop_values = np.round(np.arange(0.1, 1.0, 0.05), 2)
for i, numcop in enumerate(numcop_values):
    print(f"Run {i+1} with numcop = {numcop}")

    cooperators_fraction, games = simulate_well_mixed(N, alpha, beta, pg, ps, w, k, timesteps, numcop)

    results.append(cooperators_fraction)
    surviving_games.append(games)


# Total plot for cooperation dynamics over time
plt.figure(figsize=(10, 6))
for i, result in enumerate(results):
    plt.plot(result, label=f'Simulation {i+1} (numcop = {numcop_values[i]:.1f})')

plt.xlabel('Time steps')
plt.ylabel('Fraction of cooperators')
plt.title(f'Cooperation Dynamics in Well-Mixed Population (pg = {pg})')
plt.grid(True)
plt.ylim(0, 1)
#plt.legend()
plt.savefig("CooperationDynamicsWM", dpi=100)
plt.show()

# Plot for surviving games with color corresponding to numcop values
plt.figure(figsize=(10, 8))

# Use a colormap (e.g., 'viridis') for the scatter plot
cmap = plt.get_cmap("plasma")
norm = plt.Normalize(vmin=numcop_values.min(), vmax=numcop_values.max())  # Normalize numcop for color mapping

for i in range(len(surviving_games)):
    T_values = surviving_games[i][:, 0]
    S_values = surviving_games[i][:, 1]

    # Scatter plot where the color corresponds to the current numcop value
    sc = plt.scatter(T_values, S_values, alpha=0.7, s=10, c=[numcop_values[i]] * len(T_values), cmap=cmap, norm=norm)

# Add colorbar to represent the numcop values
#plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), label='numcop value')
plt.colorbar(sc, label='Fraction of cooperators')

plt.xlabel("T (Temptation)")
plt.ylabel("S (Sucker's Payoff)")
plt.title("Surviving Games Distribution (T vs S)")
plt.ylim(0, 2)
plt.xlim(1, 3)
plt.grid(True)
plt.savefig("SurvivingGamesWM", dpi=100)
plt.show()

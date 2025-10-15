import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Set random seed for reproducibility
np.random.seed(0)
random.seed(0)

# Parameters
N = 1000
generations = 1000
runs = 8
b_values = np.linspace(1.01, 2.0, 20)
r_values = np.linspace(0.01, 1.0, 20)
degrees = [4, 8, 16, 32, 64]

def get_adj_list_and_degrees(G):
    adj_list = [np.array(list(G.neighbors(i))) for i in range(N)]
    degrees = np.array([len(neighs) for neighs in adj_list])
    return adj_list, degrees

def play_game_PD(adj_list, degrees, b):
    strategies = np.random.choice([0, 1], size=N)
    for _ in range(generations):
        payoffs = np.zeros(N)
        for i in range(N):
            for j in adj_list[i]:
                if strategies[i] == 1 and strategies[j] == 1:
                    payoffs[i] += 1
                elif strategies[i] == 0 and strategies[j] == 1:
                    payoffs[i] += b
        new_strategies = strategies.copy()
        for x in range(N):
            neighbors = adj_list[x]
            y = np.random.choice(neighbors)
            if payoffs[y] > payoffs[x]:
                k_max = max(degrees[x], degrees[y])
                prob = (payoffs[y] - payoffs[x]) / (b * k_max)
                if random.random() < prob:
                    new_strategies[x] = strategies[y]
        if np.all(strategies == new_strategies):
            break
        strategies = new_strategies
    return np.mean(strategies)

def play_game_SG(adj_list, degrees, r):
    beta = 1 / (2 * r + 1e-9)
    R = beta - 0.5
    S = beta - 1
    T = beta
    P = 0
    strategies = np.random.choice([0, 1], size=N)
    for _ in range(generations):
        payoffs = np.zeros(N)
        for i in range(N):
            for j in adj_list[i]:
                if strategies[i] == 1 and strategies[j] == 1:
                    payoffs[i] += R
                elif strategies[i] == 1 and strategies[j] == 0:
                    payoffs[i] += S
                elif strategies[i] == 0 and strategies[j] == 1:
                    payoffs[i] += T
        new_strategies = strategies.copy()
        for x in range(N):
            neighbors = adj_list[x]
            y = np.random.choice(neighbors)
            if payoffs[y] > payoffs[x]:
                k_max = max(degrees[x], degrees[y])
                prob = (payoffs[y] - payoffs[x]) / (T * k_max)
                if random.random() < prob:
                    new_strategies[x] = strategies[y]
        if np.all(strategies == new_strategies):
            break
        strategies = new_strategies
    return np.mean(strategies)

def simulate_PD_b(b, adj_list, degrees):
    return np.mean([play_game_PD(adj_list, degrees, b) for _ in range(runs)])

def simulate_SG_r(r, adj_list, degrees):
    return np.mean([play_game_SG(adj_list, degrees, r) for _ in range(runs)])

# Main simulation
pd_results_by_k = {}
sg_results_by_k = {}

for k in tqdm(degrees, desc="Running simulations for different degrees"):
    G = nx.watts_strogatz_graph(N, k, 0)  # Ring graph
    adj_list, deg_list = get_adj_list_and_degrees(G)

    # Parallelize over b and r
    with Pool(cpu_count()) as pool:
        pd_results = pool.starmap(simulate_PD_b, [(b, adj_list, deg_list) for b in b_values])
        sg_results = pool.starmap(simulate_SG_r, [(r, adj_list, deg_list) for r in r_values])

    pd_results_by_k[k] = pd_results
    sg_results_by_k[k] = sg_results

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# PD plot
for k in degrees:
    axs[0].plot(b_values, pd_results_by_k[k], label=f"k={k}")
axs[0].axhline(0, linestyle='--', color='black', label="Well-mixed")
axs[0].set_xlabel("b (Temptation to defect)")
axs[0].set_ylabel("Fraction of cooperators")
axs[0].set_title("Prisoner's Dilemma on Ring Graph")
axs[0].legend()

# SG plot
for k in degrees:
    axs[1].plot(r_values, sg_results_by_k[k], label=f"k={k}")
axs[1].plot(r_values, 1 - r_values, linestyle='--', color='black', label="Well-mixed")
axs[1].set_xlabel("r (Cost-to-benefit ratio)")
axs[1].set_ylabel("Fraction of cooperators")
axs[1].set_title("Snowdrift Game on Ring Graph")
axs[1].legend()

plt.tight_layout()
plt.savefig("EvolutionNets", dpi=100)
plt.show()

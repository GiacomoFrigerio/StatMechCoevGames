import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from tqdm import tqdm
import pickle

# ========= Reproducibility =========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ========= Simulation parameters =========
N = 200                 # number of nodes (paper uses up to 1e4; 200 gives quick runs)
z = 4                   # average degree (even)
transient = 10_000
generations = 1_000
runs = 10               # number of independent runs (paper ~20)
b_values = np.linspace(1.01, 2.0, 20)          # PD temptation b
r_values = np.linspace(0.01, 1.0, 20)          # SG cost-to-benefit r

# ========= Helpers =========
def r_to_beta(r):
    """Correct Snowdrift mapping from the paper: r = 1 / (2*beta - 1)  => beta = (1 + 1/r)/2"""
    return 0.5 * (1.0 + 1.0 / r)

# Strategy update: proportional imitation (linear, bounded by D * k_max)
def update_strategies(strategies, payoffs, adj_list, degrees, D):
    N = len(strategies)
    new_strategies = strategies.copy()
    for x in range(N):
        y = random.choice(adj_list[x])
        if payoffs[y] > payoffs[x]:
            k_max = max(degrees[x], degrees[y])
            prob = (payoffs[y] - payoffs[x]) / (D * k_max)
            if random.random() < prob:
                new_strategies[x] = strategies[y]
    return new_strategies

# ========= Payoffs =========
def compute_payoffs_PD(strategies, adj_list, b):
    """Weak PD with (R,P,S) = (1,0,0), T = b."""
    N = len(strategies)
    payoffs = np.zeros(N, dtype=float)
    for i in range(N):
        si = strategies[i]
        for j in adj_list[i]:
            sj = strategies[j]
            if si == 1 and sj == 1:          # C vs C
                payoffs[i] += 1.0            # R
            elif si == 0 and sj == 1:        # D vs C
                payoffs[i] += b              # T
            # C vs D gives 0 (S=0); D vs D gives 0 (P=0)
    return payoffs

def compute_payoffs_SG(strategies, adj_list, beta):
    """Snowdrift (Hawk–Dove) with T=beta, R=beta-1/2, S=beta-1, P=0."""
    R, S, T, P = beta - 0.5, beta - 1.0, beta, 0.0
    N = len(strategies)
    payoffs = np.zeros(N, dtype=float)
    for i in range(N):
        si = strategies[i]
        for j in adj_list[i]:
            sj = strategies[j]
            if   si == 1 and sj == 1:  payoffs[i] += R
            elif si == 1 and sj == 0:  payoffs[i] += S
            elif si == 0 and sj == 1:  payoffs[i] += T
            else:                      payoffs[i] += P
    return payoffs

# ========= Dynamics =========
def run_game(adj_list, degrees, game='PD', param=1.2):
    """
    game='PD'  -> param=b,     D = T - S = b
    game='SG'  -> param=r, beta = (1 + 1/r)/2, D = T - P = beta
    Returns mean cooperation over the averaging window, averaged over runs.
    """
    N = len(degrees)
    coop_fractions = []
    for _ in range(runs):
        strategies = np.random.choice([0, 1], size=N)  # ~50/50 initial
        # transient
        for _ in range(transient):
            if game == 'PD':
                payoffs = compute_payoffs_PD(strategies, adj_list, b=param)
                strategies = update_strategies(strategies, payoffs, adj_list, degrees, D=param)
            else:
                beta = r_to_beta(param)
                payoffs = compute_payoffs_SG(strategies, adj_list, beta=beta)
                strategies = update_strategies(strategies, payoffs, adj_list, degrees, D=beta)
        # averaging
        coop_total = 0.0
        for _ in range(generations):
            if game == 'PD':
                payoffs = compute_payoffs_PD(strategies, adj_list, b=param)
                strategies = update_strategies(strategies, payoffs, adj_list, degrees, D=param)
            else:
                beta = r_to_beta(param)
                payoffs = compute_payoffs_SG(strategies, adj_list, beta=beta)
                strategies = update_strategies(strategies, payoffs, adj_list, degrees, D=beta)
            coop_total += np.mean(strategies)
        coop_fractions.append(coop_total / generations)
    return float(np.mean(coop_fractions))

# ========= Graph Generators =========
def get_sw_graph(p_rewire=0.1):
    # p_rewire in (0,1); p=0 is ring, p~0.1 "small-world", p=1 is random graph
    return nx.watts_strogatz_graph(N, z, p=p_rewire, seed=SEED)

def get_uniform_sf_graph():
    # "Uniform attachment" scale-free–like growth (non-preferential)
    G = nx.empty_graph(2)
    rng = random.Random(SEED)
    while G.number_of_nodes() < N:
        new_node = G.number_of_nodes()
        targets = rng.sample(list(G.nodes), k=z // 2)
        G.add_node(new_node)
        for t in targets:
            G.add_edge(new_node, t)
    return G

def get_sf_graph():
    return nx.barabasi_albert_graph(N, z // 2, seed=SEED)

# ========= Main simulation =========
if __name__ == "__main__":
    results_PD = {'SW': [], 'Uniform': [], 'SF': []}
    results_SG = {'SW': [], 'Uniform': [], 'SF': []}

    for label, graph_fn in [('SW', get_sw_graph), ('Uniform', get_uniform_sf_graph), ('SF', get_sf_graph)]:
        print(f"\nSimulating {label} network...")
        G = graph_fn() if label != 'SW' else get_sw_graph(p_rewire=0.1)
        adj_list = [list(G.neighbors(i)) for i in range(N)]
        degrees = np.array([len(neigh) for neigh in adj_list])

        # PD sweep over b
        for b in tqdm(b_values, desc=f"PD {label}"):
            coop = run_game(adj_list, degrees, game='PD', param=float(b))
            results_PD[label].append(coop)

        # SG sweep over r (with correct beta mapping)
        for r in tqdm(r_values, desc=f"SG {label}"):
            coop = run_game(adj_list, degrees, game='SG', param=float(r))
            results_SG[label].append(coop)

    # ========= Save (optional) =========
    with open("results_PD_full.pkl", "wb") as f:
        pickle.dump(results_PD, f)
    with open("results_SG_full.pkl", "wb") as f:
        pickle.dump(results_SG, f)

    # ========= Plot =========
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for label, style in zip(['SW', 'Uniform', 'SF'], ['o-', 's--', '^-']):
        axs[0].plot(b_values, results_PD[label], style, label=label, markersize=4)
        axs[1].plot(r_values, results_SG[label], style, label=label, markersize=4)

    axs[0].set_title("Prisoner’s Dilemma (network)")
    axs[0].set_xlabel("Temptation b")
    axs[0].set_ylabel("Fraction of cooperators")

    axs[1].set_title("Snowdrift (network) with well-mixed benchmark")
    axs[1].set_xlabel("Cost-to-benefit ratio r")
    axs[1].set_ylabel("Fraction of cooperators")
    axs[1].plot(r_values, 1 - r_values, color='orange', linewidth=1.5, label='Well-mixed: 1 - r')

    for ax in axs:
        ax.legend()
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig("santos_pacheco_full_plot_fixed.png", dpi=200)
    plt.show()

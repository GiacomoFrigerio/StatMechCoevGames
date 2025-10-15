import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(0)
random.seed(0)

# Parameters
N = 12
k = 4
p_rewire = 0.2
m = 2

# Generate graphs
G_regular = nx.watts_strogatz_graph(N, k, 0)
G_small_world = nx.watts_strogatz_graph(N, k, p_rewire)
G_random = nx.erdos_renyi_graph(N, k / (N - 1))
G_scale_free = nx.barabasi_albert_graph(N, m)

graphs = [(G_regular, "a) Regular"),
          (G_small_world, "b) Small-World"),
          (G_random, "c) Random"),
          (G_scale_free, "d) Scale-Free")]

# Create 4x2 grid, with narrower column for degree plots
fig, axs = plt.subplots(4, 2, figsize=(10, 10),
                        gridspec_kw={'width_ratios': [3, 1]})
positions = nx.circular_layout(G_regular)  # Fixed circular layout

for i, (G, title) in enumerate(graphs):
    # Plot the network
    ax_net = axs[i, 0]
    nx.draw(G, pos=positions, ax=ax_net, node_color='blue', edge_color='red',
            with_labels=False, node_size=70)
    ax_net.set_title(title)
    ax_net.set_aspect('equal')

    # Plot degree distribution
    ax_deg = axs[i, 1]
    degrees = [d for _, d in G.degree()]
    unique_degrees = sorted(set(degrees))
    counts = [degrees.count(d) / N for d in unique_degrees]

    ax_deg.bar(unique_degrees, counts, color='black')
    ax_deg.set_xlim(0, max(unique_degrees)+2)
    ax_deg.set_ylim(0, 1)
    ax_deg.set_xlabel("k")
    ax_deg.set_ylabel("d(k)")

plt.tight_layout()
plt.savefig("NetDeg", dpi=100)
plt.show()

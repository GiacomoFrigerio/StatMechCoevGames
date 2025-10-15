import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

np.random.seed(42)

# Node positions (fixed)
pos = {
    0: (-2, 0), 1: (-3, -1), 2: (-1, -1),
    3: (-3.5, -2), 4: (-2.5, -2), 5: (-0.5, -2),
    6: (2, 0), 7: (1, -1), 8: (2, -1.5), 9: (3, -1)
}

# Strategies: 4 cooperators, 6 defectors
cooperators = np.random.choice(range(10), size=4, replace=False)
strategies = {n: 1 if n in cooperators else 0 for n in range(10)}
node_colors = ['blue' if strategies[n] else 'red' for n in range(10)]

G_before = nx.Graph()
G_before.add_nodes_from(range(10))
edges_before = [
    (0, 1), (0, 2), (1, 3), (1, 4), (2, 5),
    (8, 6), (8, 7),
    (0, 7),
    (6, 7), (6, 9)
]
G_before.add_edges_from(edges_before)

G_after = nx.Graph()
G_after.add_nodes_from(range(10))
edges_after = [
    (0, 1), (0, 2), (1, 3), (1, 4), (2, 5),
    (8, 6), (8, 7),
    (0, 7),
    (5, 8)
]
G_after.add_edges_from(edges_after)

# Plot side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 7))

# ---- Plot G_before ----
nx.draw_networkx_nodes(G_before, pos, node_color=node_colors, node_size=900,
                       edgecolors='black', linewidths=1.5, ax=axs[0])

for u, v in G_before.edges():
    style = 'solid' if strategies[u] == strategies[v] else 'dotted'
    color = 'red' if (u, v) in [(6, 7), (6, 9)] or (v, u) in [(6, 7), (6, 9)] else 'black'
    nx.draw_networkx_edges(G_before, pos, edgelist=[(u, v)], style=style,
                            edge_color=color, width=2, ax=axs[0])

nx.draw_networkx_labels(G_before, pos, labels={n: str(n) for n in G_before.nodes()},
                        font_color='white', font_size=14, ax=axs[0])

#axs[0].set_title("Before: (6,7) and (6,9) Highlighted in Red", fontsize=14)
axs[0].axis('off')

# ---- Plot G_after ----
nx.draw_networkx_nodes(G_after, pos, node_color=node_colors, node_size=900,
                       edgecolors='black', linewidths=1.5, ax=axs[1])

for u, v in G_after.edges():
    style = 'solid' if strategies[u] == strategies[v] else 'dotted'
    color = 'blue' if (u, v) == (5, 8) or (v, u) == (5, 8) else 'black'
    nx.draw_networkx_edges(G_after, pos, edgelist=[(u, v)], style=style,
                            edge_color=color, width=2, ax=axs[1])

nx.draw_networkx_labels(G_after, pos, labels={n: str(n) for n in G_after.nodes()},
                        font_color='white', font_size=14, ax=axs[1])

#axs[1].set_title("After: Added (5,8) Only", fontsize=14)
axs[1].axis('off')

plt.suptitle("Type A", fontsize=36) #Structured Network: Before and After Targeted Rewiring
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("TypeA", dpi=60)
plt.show()

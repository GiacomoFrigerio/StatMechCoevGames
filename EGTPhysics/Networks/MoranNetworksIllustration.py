import matplotlib.pyplot as plt
import networkx as nx

def create_graph(ax, graph_type):
    if graph_type == 'directed_cycle':
        G = nx.DiGraph()
        G.add_nodes_from(range(6))
        edges = [(i, (i + 1) % 6) for i in range(6)]
        G.add_edges_from(edges)
        pos = nx.circular_layout(G)
        title = "Directed Cycle"

    elif graph_type == 'cycle':
        G = nx.DiGraph()
        G.add_nodes_from(range(6))
        edges = [(i, (i + 1) % 6) for i in range(6)] + [(i, (i - 1) % 6) for i in range(6)]
        G.add_edges_from(edges)
        pos = nx.circular_layout(G)
        title = "Bidirected Cycle"

    elif graph_type == 'line':
        G = nx.DiGraph()
        G.add_nodes_from(range(6))
        edges = [(i, i + 1) for i in range(5)]
        G.add_edges_from(edges)
        pos = {i: (i, 0) for i in range(6)}  # horizontal line layout
        title = "Line"

    elif graph_type == 'burst':
        G = nx.DiGraph()
        G.add_nodes_from(range(6))
        edges = [(0, i) for i in range(1, 6)]
        G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=42)
        title = "Burst"

    else:
        raise ValueError("Unknown graph type")

    # Color node 0 as blue (mutant), rest red
    colors = ['blue'] + ['red'] * (G.number_of_nodes() - 1)

    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='->', arrowsize=20, ax=ax)
    nx.draw_networkx_labels(G, pos, font_color='white', ax=ax)

    ax.set_title(title, fontsize=20)
    ax.set_axis_off()

# Set up the 2x2 plot
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

graph_types = ['directed_cycle', 'cycle', 'line', 'burst']
for ax, gtype in zip(axs.flat, graph_types):
    create_graph(ax, gtype)

plt.tight_layout()
plt.savefig("cyclestypes", dpi=80)
plt.show()

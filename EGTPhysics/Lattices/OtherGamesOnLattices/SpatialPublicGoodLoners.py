import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Parameters
L = 100               # Lattice size (L x L)
r = 2.035             # Multiplication factor
sigma = 1.0           # Loner's payoff
K = 0.1               # Noise (selection strength)
T = 100              # Number of time steps

# Strategy encoding
C, D, LONER = 0, 1, 2
strategies = [C, D, LONER]
strategy_labels = ['Cooperator', 'Defector', 'Loner']
colors = ['blue', 'red', 'green']
cmap = ListedColormap(colors)

# Initialize random lattice
lattice = np.random.choice(strategies, size=(L, L))

# Track strategy frequencies
frequencies = []

# Helper functions
def get_neighbors(x, y):
    return [(x % L, y % L),
            ((x + 1) % L, y % L),
            ((x - 1) % L, y % L),
            (x % L, (y + 1) % L),
            (x % L, (y - 1) % L)]

def calculate_payoff(lattice, x, y):
    group = get_neighbors(x, y)
    participants = [pos for pos in group if lattice[pos] != LONER]
    nC = sum(1 for pos in participants if lattice[pos] == C)
    nP = len(participants)

    if lattice[x, y] == LONER:
        return sigma
    elif nP < 2:
        return sigma  # insufficient group size
    else:
        total = r * nC
        return (total / nP) - (1 if lattice[x, y] == C else 0)

def update(lattice):
    x, y = np.random.randint(0, L, size=2)
    nx, ny = get_neighbors(x, y)[np.random.randint(1, 5)]  # pick a random neighbor

    p1 = calculate_payoff(lattice, x, y)
    p2 = calculate_payoff(lattice, nx, ny)

    if np.random.rand() < 1 / (1 + np.exp((p1 - p2) / K)):
        lattice[x, y] = lattice[nx, ny]  # imitation

# Simulation loop
snapshots = [1, 20, 100]
snapshot_images = {}

for t in range(1, T + 1):
    for _ in range(L * L):
        update(lattice)

    # Track frequencies
    counts = [(lattice == s).sum() / (L * L) for s in strategies]
    frequencies.append(counts)

    # Save snapshot
    if t in snapshots:
        snapshot_images[t] = lattice.copy()

# Plotting
fig, axs = plt.subplots(1, 4, figsize=(24, 6))

# Lattice snapshots
for i, t in enumerate(snapshots):
    ax = axs[i]
    ax.imshow(snapshot_images[t], cmap=cmap, vmin=0, vmax=2)
    ax.set_title(f"Lattice at t = {t}")
    ax.axis('off')

# Strategy frequency evolution
freq_array = np.array(frequencies)
ax = axs[3]
for i, label in enumerate(strategy_labels):
    ax.plot(freq_array[:, i], label=label, color=colors[i])
ax.set_title("Strategy frequencies over time")
ax.set_xlabel("Time")
ax.set_ylabel("Fraction")
ax.set_ylim(0, 1)
ax.legend()

plt.suptitle("Spatial Public Goods Game with Loners\nColors: blue = Cooperator, red = Defector, green = Loner", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.90])
plt.savefig("SpatialPGGcomplete", dpi = 80)
plt.show()

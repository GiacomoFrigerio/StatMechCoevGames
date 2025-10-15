import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Parameters from Szabó & Hauert (2002)
L = 100                 # Lattice size
r = 2.035               # Multiplication factor
sigma = 1.0             # Loner payoff
K = 0.1                 # Noise
tau = 0.1               # Cost of strategy change
T = 500                # Number of steps

# Strategy encoding
C, D, LONER = 0, 1, 2
strategies = [C, D, LONER]
strategy_labels = ['Cooperator', 'Defector', 'Loner']
colors = ['blue', 'red', 'green']
cmap = ListedColormap(colors)

# Initialize lattice
lattice = np.random.choice(strategies, size=(L, L))

# Neighbors (von Neumann + self)
def neighborhood(x, y):
    return [((x-1) % L, y), ((x+1) % L, y),
            (x, (y-1) % L), (x, (y+1) % L), (x, y)]

def calculate_total_payoff(lattice, x, y):
    total_payoff = 0.0
    for i, j in neighborhood(x, y):  # 5 games centered on neighbors
        group = neighborhood(i, j)
        group_strategies = [lattice[m, n] for m, n in group]
        nc = group_strategies.count(C)
        nd = group_strategies.count(D)
        n_participants = nc + nd

        if n_participants < 2:
            share = sigma
        else:
            if lattice[x, y] == C:
                share = (r * nc / n_participants) - 1
            elif lattice[x, y] == D:
                share = r * nc / n_participants
            else:
                share = sigma

        total_payoff += share
    return total_payoff

def update(lattice):
    x, y = np.random.randint(0, L, size=2)
    neighbors = neighborhood(x, y)[:-1]  # Exclude self
    nx, ny = neighbors[np.random.randint(len(neighbors))]

    px = calculate_total_payoff(lattice, x, y)
    py = calculate_total_payoff(lattice, nx, ny)

    prob = 1.0 / (1.0 + np.exp((px - py + tau) / K))
    if np.random.rand() < prob:
        lattice[x, y] = lattice[nx, ny]

# Run the simulation
for t in range(T):
    for _ in range(L * L):
        update(lattice)

# Plot result
plt.figure(figsize=(6, 6))
plt.imshow(lattice, cmap=cmap, vmin=0, vmax=2)
plt.title(f"Final state at t={T}\nC: blue, D: red, L: green")
plt.axis('off')
plt.show()

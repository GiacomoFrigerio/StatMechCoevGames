import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import convolve2d

# === Game Parameters ===
R, S, T, P = 1, 0, 0.95, 0  # Payoff values
C, D = 1, 0                # Strategy labels

# === Simulation Settings ===
L = 200
steps = 2
K = 0.1
initial_coop_fraction = 0.9

# === Color map: D = red, C = blue ===
strategy_cmap = colors.ListedColormap(['red', 'blue'])

# === Initialization ===
def initialize_grid():
    return np.random.choice([C, D], size=(L, L), p=[initial_coop_fraction, 1 - initial_coop_fraction])

# === Payoff Computation ===
def compute_payoffs(grid):
    coop = (grid == C).astype(int)
    defe = (grid == D).astype(int)
    kernel = np.ones((3, 3))
    coop_sum = convolve2d(coop, kernel, mode='same', boundary='wrap')
    defe_sum = convolve2d(defe, kernel, mode='same', boundary='wrap')
    payoff = coop * (R * coop_sum + S * defe_sum) + defe * (T * coop_sum + P * defe_sum)
    return payoff

# === Strategy Update (Fermi Rule) ===
def fermi_update(grid, payoff):
    new_grid = grid.copy()
    for _ in range(L * L):
        x, y = np.random.randint(0, L, 2)
        dx, dy = np.random.choice([-1, 0, 1], 2)
        nx, ny = (x + dx) % L, (y + dy) % L
        delta = payoff[nx, ny] - payoff[x, y]
        prob = 1 / (1 + np.exp(-delta / K))
        if np.random.rand() < prob:
            new_grid[x, y] = grid[nx, ny]
    return new_grid

# === Simulation ===
grid = initialize_grid()
coop_fractions = []

for step in range(steps):
    coop_fractions.append(np.mean(grid == C))
    payoff = compute_payoffs(grid)
    grid = fermi_update(grid, payoff)

# === Final Strategy Visualization ===
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap=strategy_cmap)
plt.title(f"Final Strategy Map (T={T}, K={K})")
plt.axis('off')
plt.tight_layout()
plt.savefig("Fermi_StrategyMap_T1_2.png", dpi=300)
plt.show()

# === Plot Cooperation Over Time ===
plt.figure(figsize=(10, 4))
plt.plot(coop_fractions, color='blue')
plt.title(f"Fraction of Cooperators Over Time (Fermi Rule, T={T}, K={K})")
plt.xlabel("Time Step")
plt.ylabel("Fraction of Cooperators")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig("Fermi_CoopFraction_T1_2.png", dpi=300)
plt.show()

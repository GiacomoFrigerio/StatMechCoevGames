import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import convolve2d

# Game and strategy constants
R, S, P = 31, 20, 10
C, D = 1, 0
T = 50  # Fixed value of temptation to defect

# Simulation settings
L = 200
steps = 200
initial_coop_fraction = 0.8

# Color map for strategy transitions
color_map = colors.ListedColormap(['blue', 'red', 'yellow', 'green'])

def initialize_grid():
    return np.random.choice([C, D], size=(L, L), p=[initial_coop_fraction, 1 - initial_coop_fraction])

def compute_total_payoff(grid, T):
    coop = (grid == C).astype(int)
    defe = (grid == D).astype(int)
    kernel = np.ones((3, 3))
    coop_neighbors = convolve2d(coop, kernel, mode='same', boundary='fill', fillvalue=0)
    defe_neighbors = convolve2d(defe, kernel, mode='same', boundary='fill', fillvalue=0)
    coop_payoff = coop * (R * coop_neighbors + S * defe_neighbors)
    defe_payoff = defe * (T * coop_neighbors + P * defe_neighbors)
    return coop_payoff + defe_payoff

def update(grid, T):
    payoff = compute_total_payoff(grid, T)
    best_payoff = payoff.copy()
    best_strategy = grid.copy()
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            shifted_grid = np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
            shifted_payoff = np.roll(np.roll(payoff, dx, axis=0), dy, axis=1)

            if dx == -1:
                shifted_payoff[-1, :] = -np.inf
            elif dx == 1:
                shifted_payoff[0, :] = -np.inf
            if dy == -1:
                shifted_payoff[:, -1] = -np.inf
            elif dy == 1:
                shifted_payoff[:, 0] = -np.inf

            mask = shifted_payoff > best_payoff
            best_payoff[mask] = shifted_payoff[mask]
            best_strategy[mask] = shifted_grid[mask]
    return best_strategy

def transition_colors(prev, curr):
    colors = np.zeros_like(curr)
    colors[(prev == C) & (curr == C)] = 0
    colors[(prev == D) & (curr == D)] = 1
    colors[(prev == C) & (curr == D)] = 2
    colors[(prev == D) & (curr == C)] = 3
    return colors

# Run simulation for fixed T
grid = initialize_grid()
coop_fractions = []
frames = []

for _ in range(steps):
    coop_fractions.append(np.mean(grid == C))
    prev_grid = grid.copy()
    grid = update(grid, T)
    frames.append(transition_colors(prev_grid, grid))

# Plot final strategy transition frame
plt.figure(figsize=(8, 8))
plt.imshow(frames[-1], cmap=color_map)
plt.title(f"Asymptotic Pattern (T = {T})")
plt.axis('off')
plt.tight_layout()
plt.savefig("AsymptoticPattern_T2.png", dpi=300)
plt.show()

# Plot cooperator fraction over time
plt.figure(figsize=(10, 5))
plt.plot(coop_fractions)
plt.title(f"Fraction of Cooperators Over Time (T = {T})")
plt.xlabel("Time Step")
plt.ylabel("Fraction")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig("CoopFraction_T2.png", dpi=300)
plt.show()

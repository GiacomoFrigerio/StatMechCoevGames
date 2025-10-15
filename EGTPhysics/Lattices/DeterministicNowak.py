import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import convolve2d

# Game and strategy constants
R, S, P = 1, 0, 0
C, D = 1, 0

# Simulation settings
L = 200
steps = 200
initial_coop_fraction = 0.8
b_values = [1.15, 1.35, 1.55, 1.75, 1.85, 2.01]

# Color maps
strategy_cmap = colors.ListedColormap(['red', 'blue'])  # D = red, C = blue
transition_cmap = colors.ListedColormap(['blue', 'red', 'yellow', 'green'])  # as in Nowak & May

def initialize_grid():
    return np.random.choice([C, D], size=(L, L), p=[initial_coop_fraction, 1 - initial_coop_fraction])

def compute_total_payoff(grid, T):
    coop = (grid == C).astype(int)
    defe = (grid == D).astype(int)
    kernel = np.ones((3, 3))  # includes self
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

            # Apply fixed boundary condition
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
    colors[(prev == C) & (curr == C)] = 0  # Blue
    colors[(prev == D) & (curr == D)] = 1  # Red
    colors[(prev == C) & (curr == D)] = 2  # Yellow
    colors[(prev == D) & (curr == C)] = 3  # Green
    return colors

# Run simulations
coop_fractions_over_time = []
final_strategy_frames = []
final_transition_frames = []

for T in b_values:
    grid = initialize_grid()
    coop_fractions = []
    for _ in range(steps):
        coop_fractions.append(np.mean(grid == C))
        prev_grid = grid.copy()
        grid = update(grid, T)
    coop_fractions.append(np.mean(grid == C))
    coop_fractions_over_time.append(coop_fractions)
    final_strategy_frames.append(grid.copy())  # strategy map
    final_transition_frames.append(transition_colors(prev_grid, grid))  # transition map

# Plot 1: Final strategy configurations (red/blue only)
fig1, axs1 = plt.subplots(2, 3, figsize=(15, 10))
for ax, frame, b in zip(axs1.flatten(), final_strategy_frames, b_values):
    ax.imshow(frame, cmap=strategy_cmap)
    ax.set_title(f"Final Strategy Map (b = {b})")
    ax.axis('off')
plt.tight_layout()
plt.savefig("Final_Strategy_Maps.png", dpi=300)
plt.show()

# Plot 2: Final transition color map (red, blue, green, yellow)
fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))
for ax, frame, b in zip(axs2.flatten(), final_transition_frames, b_values):
    ax.imshow(frame, cmap=transition_cmap)
    ax.set_title(f"Transition Colors (b = {b})")
    ax.axis('off')
plt.tight_layout()
plt.savefig("Final_Transitions.png", dpi=300)
plt.show()

# Plot 3: Cooperator fraction over time
fig3, axs3 = plt.subplots(2, 3, figsize=(15, 10))
for ax, fractions, b in zip(axs3.flatten(), coop_fractions_over_time, b_values):
    ax.plot(fractions)
    ax.set_title(f"Fraction of Cooperators (b = {b})")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.grid(True)
plt.tight_layout()
plt.savefig("Cooperator_Fraction_TimeSeries.png", dpi=300)
plt.show()

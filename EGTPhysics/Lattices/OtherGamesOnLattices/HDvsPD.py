import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# --- Parameters for both games ---
L = 200  # lattice size
generations = 500
R, P = 1.0, 0.0
neighborhood_kernel = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]])

# Define strategy constants
C, D = 1, 0

# Initial random grid
def initialize_grid(L):
    return np.random.choice([C, D], size=(L, L))

# Compute payoff for Prisoner's Dilemma
def compute_payoff_PD(grid, T, S):
    coop = (grid == C).astype(int)
    defe = (grid == D).astype(int)
    nC = convolve2d(coop, neighborhood_kernel, mode='same', boundary='wrap')
    nD = 4 - nC
    coop_payoff = coop * (R * nC + S * nD)
    defe_payoff = defe * (T * nC + P * nD)
    return coop_payoff + defe_payoff

# Compute payoff for Snowdrift Game
def compute_payoff_SD(grid, b, c):
    coop = (grid == C).astype(int)
    defe = (grid == D).astype(int)
    nC = convolve2d(coop, neighborhood_kernel, mode='same', boundary='wrap')
    nD = 4 - nC
    coop_payoff = coop * (nC * (b - c / 2) + nD * (b - c))
    defe_payoff = defe * (nC * b)
    return coop_payoff + defe_payoff

# Deterministic update
def update(grid, payoff):
    new_grid = grid.copy()
    for x in range(L):
        for y in range(L):
            neighbors = [(x, y),
                         ((x - 1) % L, y), ((x + 1) % L, y),
                         (x, (y - 1) % L), (x, (y + 1) % L)]
            best = max(neighbors, key=lambda pos: payoff[pos[0], pos[1]])
            new_grid[x, y] = grid[best[0], best[1]]
    return new_grid

# Simulate game
def run_simulation(game='PD', T=1.07, S=-0.07, b=1.0, c=0.62):
    grid = initialize_grid(L)
    for _ in range(generations):
        if game == 'PD':
            payoff = compute_payoff_PD(grid, T, S)
        else:
            payoff = compute_payoff_SD(grid, b, c)
        grid = update(grid, payoff)
    return grid

# Run both simulations
grid_PD = run_simulation(game='PD', T=1.27, S=-0.07)
grid_SD = run_simulation(game='SD', b=1.62, c=1.0)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(grid_PD, cmap='binary')
axs[0].set_title("a) Spatial PD (T=1.07, S=-0.07)")
axs[0].axis('off')
axs[1].imshow(grid_SD, cmap='binary')
axs[1].set_title("b) Spatial SD (r=0.62)")
axs[1].axis('off')
plt.tight_layout()
plt.show()
